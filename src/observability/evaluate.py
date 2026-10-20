"""Avaliação de qualidade: verificações determinísticas, ancoragem de respostas (grounding), LLM-as-judge e portão de regressão.

Três camadas, da mais barata e confiável para a mais cara e subjetiva:
  1. invariantes do comitê (código puro, grátis, 100% reprodutível);
  2. ancoragem de respostas de texto: os números e as citações da resposta existem nas saídas das ferramentas?;
  3. LLM-as-judge: um segundo modelo pontua a resposta numa rubrica (custa dinheiro e tem vieses — use por amostragem).

Uso (portão de regressão, offline):  python -m src.observability.evaluate
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.committee.config import DISCLAIMER, PROFILES
from src.committee.graph import Deps, build_graph, validate_portfolio
from src.observability import golden
from src.observability.tracing import get_tracer


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "error"  # error: reprova o cenário | warn: só informa


# ----------------------------------------------------------------------------- 1) invariantes do comitê
def check_committee(final: dict, profile_name: str) -> list[Check]:
    profile = PROFILES[profile_name]
    rep, tickers = final["report"], final["data"]["tickers"]
    weights = {r["ticker"]: r["peso"] for r in rep["carteira"]}
    vetoes = final["risk"]["vetoes"]
    v = validate_portfolio(weights, tickers, profile, vetoes, final["baseline"])
    # o relatório arredonda para 4 casas: tolera até 0,0001 por ativo
    errors = [e for e in v["errors"] if "somam" not in e or abs(sum(weights.values()) - 1) > 0.002]
    notes = rep["notas"]
    return [
        Check("pesos_somam_1", abs(sum(weights.values()) - 1) < 0.002, f"soma={sum(weights.values()):.4f}"),
        Check("restricoes_do_perfil", not errors, "; ".join(errors)),
        Check("vetos_respeitados", all(weights.get(t, 0) <= 0.0005 for t in vetoes), f"vetos={vetoes}"),
        Check("aviso_legal_presente", rep.get("aviso") == DISCLAIMER),
        Check("llm_sem_falhas", not any("LLM falhou" in n for n in notes), "; ".join(n for n in notes if "LLM falhou" in n), "warn"),
        Check("sem_fallback", not any("fallback" in n for n in notes), f"tentativas={final.get('attempts')}", "warn"),
    ]


# ----------------------------------------------------------------------------- 2) ancoragem de respostas
_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])")
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b")
_CITE = re.compile(r"\[(\d+)\]")
ADVICE = re.compile(r"\b(compre|comprar|venda|vender|recomendo|recomendamos|vale a pena comprar|oportunidade de compra)\b", re.I)


def _candidates(token: str) -> list[tuple[float, int]]:
    """Interpretações de um número: '1.234,56' -> [(1234.56, 2)]. '1.234' é ambíguo (mil duzentos e trinta e quatro em pt-BR,
    um vírgula duzentos e trinta e quatro em JSON/inglês), então as duas leituras valem. 2º valor = casas decimais exibidas."""
    if "," in token:
        whole, dec = token.replace(".", "").split(",")
        return [(float(f"{whole}.{dec}"), len(dec))]
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", token):
        return [(float(token.replace(".", "")), 0), (float(token), len(token.split(".")[1]))]
    return [(float(token), len(token.split(".")[1]) if "." in token else 0)]


def numbers_in(text: str) -> list[list[tuple[float, int]]]:
    text = _DATE.sub(" ", _CITE.sub(" ", text))
    return [_candidates(m.group(1)) for m in _NUM.finditer(text)]


def check_grounding(answer: str, tool_outputs: list[str], n_sources: int = 0) -> list[Check]:
    """Cada número da resposta (exceto datas, marcadores [n] e inteiros pequenos) deve aparecer nas saídas das ferramentas,
    dentro da tolerância de arredondamento. Percentuais podem vir como fração (0,123 -> 12,3%)."""
    source = [v for t in tool_outputs for cands in numbers_in(t) for v, _ in cands]
    source += [v * 100 for v in source if abs(v) < 5] + [v / 100 for v in source if abs(v) >= 5]
    ungrounded = []
    for cands in numbers_in(answer):
        if any(d == 0 and abs(v) <= 31 for v, d in cands):
            continue  # inteiros pequenos (contagens, dias, "8 linhas") geram falsos alarmes
        if not any(abs(v - s) <= 0.5 * 10 ** -d + 1e-9 for v, d in cands for s in source):
            ungrounded.append(cands[0][0])
    cites = [int(c) for c in _CITE.findall(answer)]
    bad_cites = sorted({c for c in cites if c < 1 or c > n_sources})
    advice = ADVICE.findall(answer)
    return [
        Check("numeros_ancorados", not ungrounded, f"sem origem nas ferramentas: {ungrounded}"),
        Check("citacoes_validas", not bad_cites, f"citações fora de 1..{n_sources}: {bad_cites}"),
        Check("sem_recomendacao", not advice, f"termos: {sorted({a.lower() for a in advice})}"),
    ]


# ----------------------------------------------------------------------------- 3) LLM-as-judge
class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="1 = inventa fatos; 5 = tudo apoiado no contexto")
    relevance: int = Field(ge=1, le=5, description="1 = não responde; 5 = responde direto à pergunta")
    advice_free: bool = Field(description="true se NÃO recomenda compra/venda")
    comment: str = ""


JUDGE_SYSTEM = ("Você é um AVALIADOR rigoroso. Receberá PERGUNTA, CONTEXTO (saídas das ferramentas) e RESPOSTA. "
                "Pontue de 1 a 5: faithfulness (a resposta só afirma o que o contexto sustenta?), relevance (responde à pergunta?) "
                "e informe advice_free (true se a resposta NÃO recomenda comprar/vender). Não avalie estilo nem tamanho. "
                "Responda somente com JSON: {\"faithfulness\": int, \"relevance\": int, \"advice_free\": bool, \"comment\": str}.")


def judge_answer(judge_llm, question: str, answer: str, context: str, tries: int = 2) -> JudgeScore:
    """Pede a nota a um modelo juiz. `judge_llm` é qualquer objeto com .invoke(mensagens) (use modelo simulado nos testes)."""
    import json

    from src.committee.brains import _ask

    user = json.dumps({"PERGUNTA": question, "CONTEXTO": context, "RESPOSTA": answer}, ensure_ascii=False)
    return _ask(judge_llm, JUDGE_SYSTEM, user, JudgeScore, tries=tries)


# ----------------------------------------------------------------------------- portão de regressão
def run_scenario(sc: golden.Scenario) -> tuple[list[Check], dict]:
    series = golden.make_market(sc.market)
    deps = Deps(load_series=lambda tk: ({t: series[t] for t in tk}, {}),
                quote=lambda t: {"price_earnings": 8.0 + len(t), "fifty_two_week_range": "1 - 2"},
                news=lambda t: [], brain=sc.make_brain())
    final = build_graph(deps).invoke({"profile": sc.profile, "universe": golden.TICKERS, "attempts": 0})
    checks = check_committee(final, sc.profile)
    weights = {r["ticker"]: r["peso"] for r in final["report"]["carteira"]}
    # expectativas específicas do cenário
    if sc.brain == "llm_bad":
        checks.append(Check("bad_cai_no_fallback", final["attempts"] == 3 and any("fallback" in n for n in final["report"]["notas"]),
                            f"tentativas={final['attempts']}"))
    if sc.brain == "llm_flaky":
        checks.append(Check("flaky_recupera_na_2a", final["attempts"] == 2 and not any("fallback" in n for n in final["report"]["notas"]),
                            f"tentativas={final['attempts']}"))
    if sc.market == "crash" and sc.profile in ("conservador", "moderado"):
        checks.append(Check("ativo_em_colapso_vetado", weights["DDD4"] <= 0.0005, f"peso DDD4={weights['DDD4']:.2%}"))
    return checks, final


def run_regression(scenarios=None, log=print) -> dict:
    scenarios = scenarios or golden.all_scenarios()
    results, failed = [], []
    for sc in scenarios:
        t0 = time.perf_counter()
        with get_tracer().span(f"eval:{sc.id}", "eval", market=sc.market, profile=sc.profile, brain=sc.brain) as span:
            checks, _ = run_scenario(sc)
            bad = [c for c in checks if not c.ok and c.severity == "error"]
            span.set(passed=not bad, failed_checks=[c.name for c in bad])
        results.append({"id": sc.id, "ok": not bad, "ms": (time.perf_counter() - t0) * 1000,
                        "failed": [(c.name, c.detail) for c in bad], "warnings": [c.name for c in checks if not c.ok and c.severity == "warn"]})
        if bad:
            failed.append(sc.id)
            log(f"FALHOU {sc.id}: " + "; ".join(f"{c.name} ({c.detail})" for c in bad))
    return {"total": len(results), "passed": len(results) - len(failed), "failed": failed, "results": results}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = run_regression()
    warn = sum(1 for r in summary["results"] if r["warnings"])
    print(f"\n{summary['passed']}/{summary['total']} cenários aprovados | {warn} com avisos (fallback/LLM falhou, esperados nos modos flaky/bad)")
    sys.exit(0 if not summary["failed"] else 1)
