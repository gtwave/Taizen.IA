"""Os "cérebros" dos agentes: regras determinísticas (RulesBrain) ou LLM (LLMBrain), com a mesma interface.

O grafo (graph.py) é idêntico nos dois casos. O LLM só ajusta o que o código já calculou, dentro de limites:
  - analista de dados: ajuste qualitativo do score entre -1 e +1;
  - analista de risco: vetos adicionais e alertas em texto;
  - gestor: pesos próximos da linha de base determinística (desvio máximo por ativo) — e tudo passa pelo validador.
"""

import json
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from src.committee.allocation import allocate
from src.committee.config import DISCLAIMER, Profile
from src.committee.metrics import zscore

MAX_DEVIATION = 0.15  # desvio máximo (em pontos de peso) entre a proposta do LLM e a linha de base


class BrainError(Exception):
    """O LLM não produziu uma saída válida dentro do número de tentativas."""


# ----------------------------------------------------------------------------- esquemas de saída
class AssetView(BaseModel):
    ticker: str
    stance: Literal["favoravel", "neutra", "cautelosa"]
    score_adjust: float = Field(ge=-1.0, le=1.0)
    rationale: str


class DataAnalystOutput(BaseModel):
    views: list[AssetView]


class RiskOutput(BaseModel):
    vetoes: list[str] = []
    flags: list[str] = []
    comments: str = ""


class ManagerOutput(BaseModel):
    weights: dict[str, float]
    rationale: str


def extract_json(text: str) -> dict:
    """Extrai o primeiro objeto JSON de um texto (tolera cercas ```json e comentários antes/depois)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("nenhum objeto JSON encontrado")
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError("JSON incompleto")


def stance_from_score(score: float) -> str:
    return "favoravel" if score > 0.5 else "cautelosa" if score < -0.5 else "neutra"


# ----------------------------------------------------------------------------- regras determinísticas
class RulesBrain:
    """Heurísticas quantitativas simples e reproduzíveis. Serve de linha de base e de plano B do LLM."""

    name = "rules"

    def data_analyst(self, dossiers: dict[str, dict], profile: Profile) -> dict[str, dict]:
        tickers = list(dossiers)
        mom = np.array([dossiers[t]["metricas"]["momento_6m"] for t in tickers])
        vol = np.array([dossiers[t]["metricas"]["volatilidade_anual"] for t in tickers])
        pl = np.array([dossiers[t].get("preco_lucro") or np.nan for t in tickers], dtype=float)
        earn_yield = np.where(np.isfinite(pl) & (pl > 0), 1.0 / pl, 0.0)
        z = 0.4 * zscore(mom) + 0.3 * zscore(earn_yield) - 0.3 * zscore(vol)
        out = {}
        for t, s, m, v, p in zip(tickers, z, mom, vol, pl):
            s = float(np.clip(s, -2, 2))
            pl_txt = f"P/L {p:.1f}" if np.isfinite(p) else "P/L indisponível"
            out[t] = {
                "score": round(s, 3), "stance": stance_from_score(s),
                "rationale": f"score quantitativo {s:+.2f} (momento 6m {m:+.0%}, {pl_txt}, volatilidade {v:.0%}).",
            }
        return out

    def risk_analyst(self, summaries: dict[str, dict], corr: dict, profile: Profile) -> dict:
        vetoes, flags = [], []
        for t, m in summaries.items():
            if m["drawdown_maximo"] < profile.veto_drawdown or m["volatilidade_anual"] > profile.veto_vol:
                vetoes.append(t)
                flags.append(f"{t}: risco acima do tolerado pelo perfil {profile.name} "
                             f"(drawdown {m['drawdown_maximo']:.0%}, volatilidade {m['volatilidade_anual']:.0%}).")
        names = list(summaries)
        high = [(a, b, corr[a][b]) for i, a in enumerate(names) for b in names[i + 1:] if corr[a][b] > 0.75]
        for a, b, c in high:
            flags.append(f"{a} e {b} são muito correlacionadas ({c:.2f}): diversificação limitada entre elas.")
        return {"vetoes": vetoes, "flags": flags, "comments": "análise determinística de risco"}

    def manager(self, baseline: dict[str, float], views: dict, risk: dict, profile: Profile, feedback=None) -> dict:
        return {"weights": dict(baseline), "rationale": "alocação determinística: volatilidade inversa inclinada pelo score, "
                f"com teto de {profile.max_weight:.0%} por ativo e piso de {profile.min_weight:.0%}."}


# ----------------------------------------------------------------------------- LLM
def _model_id(llm) -> str:
    return str(getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__)


def _ask(llm, system: str, user: str, schema, tries: int = 2, cache=None, accept=None):
    """Chama o LLM, valida a saída com Pydantic e reenvia o erro ao modelo se vier inválida.

    Com `cache` (ExactCache), a mesma pergunta ao mesmo modelo não é refeita — e só saídas JÁ VALIDADAS são guardadas:
    além do esquema, `accept(saída)` (opcional) confere as regras de negócio antes de guardar."""
    key = cache.key(_model_id(llm), system, user) if cache is not None else None
    if key is not None:
        cached = cache.get(key)
        if cached is not None:
            try:
                return schema.model_validate_json(cached)
            except ValidationError:
                pass  # entrada de cache de uma versão anterior do esquema: ignora e refaz
    messages = [("system", system), ("human", user)]
    last = None
    for _ in range(tries):
        resp = llm.invoke(messages)
        text = resp.content if isinstance(resp.content, str) else "".join(
            b.get("text", "") for b in resp.content if isinstance(b, dict))
        try:
            out = schema.model_validate(extract_json(text))
            if key is not None and (accept is None or accept(out)):
                cache.put(key, out.model_dump_json())
            return out
        except (ValueError, ValidationError) as exc:
            last = exc
            messages += [("ai", text), ("human", f"A saída é inválida: {exc}. Responda de novo apenas com o JSON corrigido.")]
    raise BrainError(f"saída inválida após {tries} tentativas: {last}")


class LLMBrain:
    """Mesma interface do RulesBrain, mas com chamadas a LLMs (modelo menor nos analistas, maior no gestor)."""

    name = "llm"

    def __init__(self, analyst_llm, manager_llm, rules: RulesBrain | None = None, cache=None):
        self.analyst_llm = analyst_llm
        self.manager_llm = manager_llm
        self.rules = rules or RulesBrain()
        self.cache = cache  # ExactCache opcional (respostas validadas)

    def data_analyst(self, dossiers, profile):
        base = self.rules.data_analyst(dossiers, profile)
        system = ("Você é o ANALISTA DE DADOS de um comitê de investimentos educacional. Para cada ativo, dê uma visão "
                  "qualitativa curta com base APENAS no dossiê (métricas e manchetes). Não invente fatos. "
                  "Responda somente com JSON: {\"views\": [{\"ticker\": str, \"stance\": \"favoravel|neutra|cautelosa\", "
                  "\"score_adjust\": número entre -1 e 1, \"rationale\": str}]}. " + DISCLAIMER)
        out = _ask(self.analyst_llm, system, json.dumps({"dossies": dossiers, "score_quantitativo": {t: v["score"] for t, v in base.items()}},
                                                        ensure_ascii=False), DataAnalystOutput, cache=self.cache)
        merged = {}
        for t, b in base.items():
            v = next((x for x in out.views if x.ticker == t), None)
            adj = v.score_adjust if v else 0.0
            score = float(np.clip(b["score"] + adj, -2, 2))
            merged[t] = {"score": round(score, 3), "stance": v.stance if v else b["stance"],
                         "rationale": (v.rationale if v else b["rationale"]), "ajuste_llm": round(adj, 3)}
        return merged

    def risk_analyst(self, summaries, corr, profile):
        base = self.rules.risk_analyst(summaries, corr, profile)
        system = ("Você é o ANALISTA DE RISCO de um comitê educacional. Com base nas métricas e na matriz de correlação, "
                  "aponte alertas e, se necessário, vetos (apenas tickers do universo). Responda somente com JSON: "
                  "{\"vetoes\": [str], \"flags\": [str], \"comments\": str}. " + DISCLAIMER)
        out = _ask(self.analyst_llm, system, json.dumps({"perfil": profile.name, "metricas": summaries, "correlacao": corr,
                                                         "vetos_das_regras": base["vetoes"]}, ensure_ascii=False), RiskOutput, cache=self.cache)
        vetoes = sorted(set(base["vetoes"]) | {t for t in out.vetoes if t in summaries})
        return {"vetoes": vetoes, "flags": base["flags"] + out.flags, "comments": out.comments or base["comments"]}

    def manager(self, baseline, views, risk, profile, feedback=None):
        system = ("Você é o GESTOR DE CARTEIRA de um comitê educacional. Parta da LINHA DE BASE de pesos e ajuste-a no máximo "
                  f"{MAX_DEVIATION:.0%} por ativo, justificando. Regras: pesos somam 1; cada peso entre {profile.min_weight} e "
                  f"{profile.max_weight}; ativos vetados recebem 0. Responda somente com JSON: "
                  "{\"weights\": {ticker: peso}, \"rationale\": str}. " + DISCLAIMER)
        user = {"perfil": profile.name, "linha_de_base": baseline, "visoes": views, "risco": risk}
        if feedback:
            user["erros_da_tentativa_anterior"] = feedback
        def accept(out: ManagerOutput) -> bool:  # só vale guardar propostas que o validador aprovaria
            from src.committee.graph import validate_portfolio
            return validate_portfolio(out.weights, list(baseline), profile, risk["vetoes"], baseline)["ok"]

        out = _ask(self.manager_llm, system, json.dumps(user, ensure_ascii=False), ManagerOutput, cache=self.cache, accept=accept)
        return {"weights": out.weights, "rationale": out.rationale}


def baseline_allocation(tickers: list[str], vols: np.ndarray, scores: np.ndarray, profile: Profile, vetoes: list[str]) -> dict[str, float]:
    w = allocate(vols, scores, profile, np.array([t in vetoes for t in tickers]))
    return {t: round(float(x), 6) for t, x in zip(tickers, w)}
