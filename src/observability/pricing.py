"""Custo de tokens (FinOps): tabela de preços, conversão para reais e controle do orçamento mensal.

Os preços são de REFERÊNCIA (US$ por milhão de tokens) e mudam; confira a página oficial do provedor antes de decidir
algo com base neles. Para sobrescrever sem editar o código, defina `PRICING_JSON` com um objeto
`{"prefixo-do-modelo": [entrada, saída]}` e `USD_BRL` com a cotação do dólar.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

# (entrada, saída) em US$ por milhão de tokens. Casamento por prefixo mais longo do id do modelo.
REFERENCE_PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
}
CACHE_READ_FACTOR = 0.10   # leitura de prompt em cache do provedor: 10% do preço de entrada
CACHE_WRITE_FACTOR = 1.25  # escrita no cache de prompt (TTL de 5 min): 125% do preço de entrada
DEFAULT_USD_BRL = 5.5
MONTHLY_BUDGET_BRL = float(os.environ.get("MONTHLY_BUDGET_BRL", "50"))


def usd_brl() -> float:
    return float(os.environ.get("USD_BRL", DEFAULT_USD_BRL))


def price_table() -> dict[str, tuple[float, float]]:
    table = dict(REFERENCE_PRICES)
    extra = os.environ.get("PRICING_JSON")
    if extra:
        table.update({k: tuple(v) for k, v in json.loads(extra).items()})
    return table


def price_for(model: str | None) -> tuple[float, float] | None:
    if not model:
        return None
    best = None
    for prefix, p in price_table().items():
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, p)
    return best[1] if best else None


def cost_usd(model: str | None, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float | None:
    """Custo em US$ de uma chamada. `input_tokens` é o total de entrada; as parcelas de cache são subconjuntos dele.
    Devolve None se o modelo não está na tabela (melhor "não sei" do que um custo inventado)."""
    p = price_for(model)
    if p is None:
        return None
    pin, pout = p
    fresh = max(input_tokens - cache_read - cache_write, 0)
    return (fresh * pin + cache_read * pin * CACHE_READ_FACTOR + cache_write * pin * CACHE_WRITE_FACTOR + output_tokens * pout) / 1e6


def cost_brl(model: str | None, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float | None:
    usd = cost_usd(model, input_tokens, output_tokens, cache_read, cache_write)
    return None if usd is None else usd * usd_brl()


class BudgetExceeded(RuntimeError):
    """O gasto do mês já alcançou o orçamento: a chamada paga não deve ser feita."""


@dataclass
class BudgetStatus:
    spent_brl: float
    budget_brl: float
    projected_brl: float
    level: str  # ok | warn | exceeded

    @property
    def remaining_brl(self) -> float:
        return self.budget_brl - self.spent_brl


def month_start(now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def budget_status(spans: list[dict], budget_brl: float = MONTHLY_BUDGET_BRL, now: datetime | None = None, warn_at: float = 0.7) -> BudgetStatus:
    """Gasto do mês corrente a partir dos spans de LLM, projeção linear até o fim do mês e nível de alerta."""
    now = now or datetime.now(timezone.utc)
    start = month_start(now)
    spent = sum(s["attrs"].get("cost_brl") or 0.0 for s in spans if s["kind"] == "llm" and s["start"] >= start)
    elapsed_days = max((now.timestamp() - start) / 86400, 1.0)
    days_in_month = 31 if now.month == 12 else (now.replace(month=now.month + 1, day=1) - now.replace(day=1)).days
    projected = spent / elapsed_days * days_in_month
    level = "exceeded" if spent >= budget_brl else "warn" if spent >= warn_at * budget_brl or projected >= budget_brl else "ok"
    return BudgetStatus(spent, budget_brl, projected, level)


def assert_within_budget(spans: list[dict], budget_brl: float = MONTHLY_BUDGET_BRL, now: datetime | None = None) -> BudgetStatus:
    st = budget_status(spans, budget_brl, now)
    if st.level == "exceeded":
        raise BudgetExceeded(f"orçamento do mês esgotado: R$ {st.spent_brl:.2f} de R$ {st.budget_brl:.2f}")
    return st
