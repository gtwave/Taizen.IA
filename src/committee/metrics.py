"""Métricas de risco e retorno em numpy puro (funções determinísticas, sem LLM)."""

import numpy as np

from src.committee.config import TRADING_DAYS


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Retornos simples diários de uma série (T,) ou de uma matriz (T, N)."""
    prices = np.asarray(prices, dtype=float)
    return prices[1:] / prices[:-1] - 1.0


def annualized_return(prices: np.ndarray, days: int = TRADING_DAYS) -> float:
    """Retorno anualizado composto (CAGR) a partir do primeiro e do último preço."""
    n = len(prices) - 1
    return float((prices[-1] / prices[0]) ** (days / n) - 1.0) if n > 0 else 0.0


def annualized_vol(returns: np.ndarray, days: int = TRADING_DAYS) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(days))


def max_drawdown(prices: np.ndarray) -> float:
    """Pior queda do pico até o fundo (valor negativo, ex.: -0.35)."""
    prices = np.asarray(prices, dtype=float)
    peaks = np.maximum.accumulate(prices)
    return float(np.min(prices / peaks - 1.0))


def momentum(prices: np.ndarray, window: int) -> float:
    """Retorno acumulado nos últimos `window` pregões (0 se não houver histórico)."""
    if len(prices) <= window:
        return 0.0
    return float(prices[-1] / prices[-1 - window] - 1.0)


def range_position(prices: np.ndarray, window: int = TRADING_DAYS) -> float:
    """Posição do último preço entre a mínima (0) e a máxima (1) da janela."""
    w = np.asarray(prices[-window:], dtype=float)
    lo, hi = w.min(), w.max()
    return float((w[-1] - lo) / (hi - lo)) if hi > lo else 0.5


def correlation_matrix(returns: np.ndarray) -> np.ndarray:
    return np.corrcoef(returns, rowvar=False)


def covariance_matrix(returns: np.ndarray, days: int = TRADING_DAYS) -> np.ndarray:
    return np.cov(returns, rowvar=False, ddof=1) * days


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ cov @ w))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fração do risco (volatilidade) da carteira atribuível a cada ativo; soma 1."""
    w = np.asarray(weights, dtype=float)
    vol = portfolio_vol(w, cov)
    if vol == 0:
        return np.zeros_like(w)
    return w * (cov @ w) / vol**2


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Média ponderada das volatilidades individuais ÷ volatilidade da carteira (>= 1; maior = mais diversificada)."""
    w = np.asarray(weights, dtype=float)
    vol = portfolio_vol(w, cov)
    return float((w @ np.sqrt(np.diag(cov))) / vol) if vol else 1.0


def zscore(x: np.ndarray, clip: float = 2.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    return np.clip(z, -clip, clip)


def asset_summary(prices: np.ndarray) -> dict:
    """Resumo de risco/retorno de um ativo (usado no dossiê dos analistas)."""
    r = simple_returns(prices)
    return {
        "retorno_anual": round(annualized_return(prices), 4),
        "volatilidade_anual": round(annualized_vol(r), 4),
        "drawdown_maximo": round(max_drawdown(prices), 4),
        "momento_1m": round(momentum(prices, 21), 4),
        "momento_3m": round(momentum(prices, 63), 4),
        "momento_6m": round(momentum(prices, 126), 4),
        "posicao_52s": round(range_position(prices, TRADING_DAYS), 4),
    }
