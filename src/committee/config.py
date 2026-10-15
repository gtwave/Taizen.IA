"""Configuração do comitê de agentes: universo de ações, perfis de investidor e caminhos de dados."""

import os
from dataclasses import dataclass

from src.config import DATA_DIR

PRICES_DIR = DATA_DIR / "raw" / "prices"
PORTFOLIOS_DIR = DATA_DIR / "processed" / "portfolios"

# A brapi.dev permite consultar estes quatro tickers sem cadastro; os demais exigem um token gratuito (BRAPI_TOKEN).
FREE_TICKERS = ["PETR4", "VALE3", "ITUB4", "MGLU3"]
EXTENDED_TICKERS = FREE_TICKERS + ["BBDC4", "BBAS3", "ABEV3", "WEGE3", "SUZB3", "RENT3"]

ANALYST_MODEL = os.environ.get("ANALYST_MODEL", "claude-haiku-4-5-20251001")
MANAGER_MODEL = os.environ.get("MANAGER_MODEL", os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"))

MAX_ATTEMPTS = 3  # tentativas do gestor antes de recorrer à alocação determinística

TRADING_DAYS = 252


def default_universe() -> list[str]:
    return EXTENDED_TICKERS if os.environ.get("BRAPI_TOKEN") else FREE_TICKERS


@dataclass(frozen=True)
class Profile:
    name: str
    max_weight: float   # teto por ativo
    min_weight: float   # piso por ativo mantido na carteira
    min_positions: int  # número mínimo de ativos
    tilt: float         # intensidade da inclinação pelo score dos analistas (0 = só risco)
    veto_drawdown: float  # ativos com drawdown máximo pior que isto são vetados (ex.: -0.6)
    veto_vol: float       # ... e volatilidade anualizada acima disto


PROFILES = {
    "conservador": Profile("conservador", max_weight=0.35, min_weight=0.10, min_positions=3, tilt=0.25, veto_drawdown=-0.55, veto_vol=0.45),
    "moderado": Profile("moderado", max_weight=0.45, min_weight=0.05, min_positions=3, tilt=0.60, veto_drawdown=-0.70, veto_vol=0.60),
    "arrojado": Profile("arrojado", max_weight=0.60, min_weight=0.00, min_positions=2, tilt=1.00, veto_drawdown=-0.90, veto_vol=0.90),
}

DISCLAIMER = (
    "Material educacional gerado por um sistema experimental. Não constitui recomendação de investimento, "
    "oferta ou consultoria de valores mobiliários; resultados passados não garantem resultados futuros."
)
