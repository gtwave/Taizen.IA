"""Cenários dourados (golden scenarios) do comitê: mercados sintéticos + LLMs roteirizados, tudo offline.

A ideia é testar o COMPORTAMENTO do sistema (restrições, vetos, nova tentativa, fallback) em situações controladas,
sem rede e sem gastar um centavo de API. Cada cenário combina um mercado, um perfil e um "gestor" simulado.
"""

import json
import re
from dataclasses import dataclass

import numpy as np
from langchain_core.messages import AIMessage

from src.committee.brains import LLMBrain, RulesBrain
from src.committee.market import PriceSeries

TICKERS = ["AAA3", "BBB4", "CCC3", "DDD4"]


def make_market(kind: str, n: int = 500, seed: int = 0) -> dict[str, PriceSeries]:
    """normal: 4 ativos com risco crescente | crash: o último ativo perde ~85% | correlated: todos seguem um fator comum."""
    rng = np.random.default_rng(seed)
    dates = np.arange(n) * 86400 + 1_600_000_000
    common = rng.normal(0.0004, 0.010, n - 1)
    out = {}
    for i, t in enumerate(TICKERS):
        if kind == "correlated":
            daily = common + rng.normal(0, 0.002, n - 1)
        else:
            daily = rng.normal(0.0004 * (i + 1), 0.010 * (i + 1), n - 1)
        if kind == "crash" and i == len(TICKERS) - 1:
            daily = daily.copy()
            daily[n // 2: n // 2 + 40] = -0.045  # ~40 pregões de queda forte
        out[t] = PriceSeries(t, dates, 20 * np.concatenate([[1.0], np.cumprod(1 + daily)]))
    return out


class ScriptedChat:
    """Modelo de chat simulado e reativo: lê o prompt e responde como um LLM de cada papel responderia.

    Modos do gestor: "good" (ajuste pequeno e válido), "flaky" (inválido na 1ª chamada, válido depois — exercita a nova
    tentativa) e "bad" (sempre inválido — exercita o fallback). Conta as chamadas para os testes.
    """

    def __init__(self, manager_mode: str = "good", model: str = "claude-sonnet-5"):
        self.manager_mode, self.model, self.calls, self._manager_calls = manager_mode, model, 0, 0

    def invoke(self, messages, config=None, **kwargs):
        self.calls += 1
        system = messages[0][1] if isinstance(messages[0], tuple) else messages[0].content
        user = next(m[1] if isinstance(m, tuple) else m.content for m in messages[1:] if (m[0] if isinstance(m, tuple) else m.type) in ("human", "user"))
        data = json.loads(re.search(r"\{.*\}", user, re.S).group(0))
        if "ANALISTA DE DADOS" in system:
            body = {"views": [{"ticker": t, "stance": "neutra", "score_adjust": 0.0, "rationale": "sem fatos novos além das métricas."} for t in data["dossies"]]}
        elif "ANALISTA DE RISCO" in system:
            body = {"vetoes": [], "flags": [], "comments": "sem observações adicionais."}
        else:
            self._manager_calls += 1
            base = dict(data["linha_de_base"])
            tickers = list(base)
            invalid = {t: 0.0 for t in tickers}
            invalid[tickers[0]] = 1.2  # soma 120% e estoura qualquer teto
            if self.manager_mode == "bad" or (self.manager_mode == "flaky" and self._manager_calls == 1):
                body = {"weights": invalid, "rationale": "convicção total no primeiro ativo."}
            else:
                active = [t for t in tickers if base[t] > 0.0005]
                hi, lo = max(active, key=base.get), min(active, key=base.get)
                if base[hi] - base[lo] >= 0.06:  # só transfere 3 p.p. do maior para o menor se isso não furar teto nem piso
                    base[hi] -= 0.03
                    base[lo] += 0.03
                body = {"weights": base, "rationale": "pequeno ajuste sobre a linha de base."}
        return AIMessage(content=json.dumps(body, ensure_ascii=False),
                         usage_metadata={"input_tokens": max(len(user) // 4, 1), "output_tokens": 120, "total_tokens": max(len(user) // 4, 1) + 120})


@dataclass(frozen=True)
class Scenario:
    market: str
    profile: str
    brain: str  # rules | llm_good | llm_flaky | llm_bad

    @property
    def id(self) -> str:
        return f"{self.market}/{self.profile}/{self.brain}"

    def make_brain(self):
        if self.brain == "rules":
            return RulesBrain()
        chat = ScriptedChat(self.brain.removeprefix("llm_"))
        return LLMBrain(chat, chat)


def all_scenarios() -> list[Scenario]:
    return [Scenario(m, p, b) for m in ("normal", "crash", "correlated") for p in ("conservador", "moderado", "arrojado")
            for b in ("rules", "llm_good", "llm_flaky", "llm_bad")]


# ----------------------------------------------------------------------------- versão compatível com LangChain
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402


class SimChat(BaseChatModel):
    """O ScriptedChat como modelo de chat do LangChain: aceita callbacks (tracing), tem nome de modelo e informa tokens.

    Os tokens são ESTIMADOS (caracteres/4) — servem para exercitar custo e tracing sem gastar nada."""

    model: str = "claude-sonnet-5"
    manager_mode: str = "good"
    _script: "ScriptedChat" = PrivateAttr()

    def model_post_init(self, __context) -> None:
        self._script = ScriptedChat(self.manager_mode, self.model)

    @property
    def _llm_type(self) -> str:
        return "sim-chat"

    @property
    def calls(self) -> int:
        return self._script.calls

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._script.invoke(messages))])
