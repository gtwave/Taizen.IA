"""Backtest walk-forward (sem olhar o futuro) do comitê de regras contra linhas de base simples.

Limitações assumidas (veja o Capítulo 7): universo pequeno e composto por empresas que SOBREVIVERAM até hoje,
fundamentos históricos indisponíveis (só momento e volatilidade), custo de transação fixo e ausência de benchmark
de mercado (índices exigem token na brapi). O LLM NÃO pode ser backtestado: ele já "conhece" o futuro.

Uso:  python -m src.committee.backtest [--tickers PETR4,VALE3,...] [--cost 0.001]
"""

import argparse
import sys

import numpy as np

from src.committee import metrics
from src.committee.allocation import allocate
from src.committee.brains import RulesBrain, baseline_allocation
from src.committee.config import PROFILES, TRADING_DAYS, Profile, default_universe
from src.committee.market import align, load_universe

LOOKBACK = TRADING_DAYS
STEP = 21  # rebalanceamento mensal (~21 pregões)


def weights_equal(prices_window: np.ndarray, profile: Profile | None = None) -> np.ndarray:
    n = prices_window.shape[1]
    return np.full(n, 1.0 / n)


def weights_inverse_vol(prices_window: np.ndarray, profile: Profile | None = None) -> np.ndarray:
    vols = np.array([metrics.annualized_vol(metrics.simple_returns(prices_window[:, j])) for j in range(prices_window.shape[1])])
    w = 1.0 / vols
    return w / w.sum()


def weights_committee(prices_window: np.ndarray, profile: Profile) -> np.ndarray:
    """Pesos do comitê de regras usando SOMENTE a janela recebida (sem P/L: não há fundamento histórico)."""
    n = prices_window.shape[1]
    tickers = [str(i) for i in range(n)]
    summaries = {t: metrics.asset_summary(prices_window[:, j]) for j, t in enumerate(tickers)}
    dossiers = {t: {"metricas": summaries[t], "preco_lucro": None} for t in tickers}
    brain = RulesBrain()
    views = brain.data_analyst(dossiers, profile)
    corr = {a: {b: 0.0 for b in tickers} for a in tickers}  # correlação não afeta o veto por perfil
    vetoes = brain.risk_analyst(summaries, corr, profile)["vetoes"]
    vols = np.array([summaries[t]["volatilidade_anual"] for t in tickers])
    scores = np.array([views[t]["score"] for t in tickers])
    if len(vetoes) >= n:
        return np.zeros(n)  # tudo vetado: fica em caixa (retorno 0) até a janela mostrar ativos aceitáveis
    base = baseline_allocation(tickers, vols, scores, profile, vetoes)
    return np.array([base[t] for t in tickers])


def walk_forward(prices: np.ndarray, weight_fn, profile: Profile | None = None, lookback: int = LOOKBACK,
                 step: int = STEP, cost: float = 0.001) -> dict:
    """Em cada rebalanceamento, os pesos usam prices[:t+1]; o retorno vem de prices[t:t+step+1]."""
    T, N = prices.shape
    equity = np.ones(T)
    value, w_prev, turnover = 1.0, np.zeros(N), 0.0
    t = lookback
    equity[:t] = np.nan
    history = []
    while t < T - 1:
        w = weight_fn(prices[t - lookback:t + 1], profile)  # nunca inclui preços depois de t
        turnover_t = float(np.abs(w - w_prev).sum()) if history else float(np.abs(w).sum())
        value *= 1.0 - cost * turnover_t
        turnover += turnover_t
        end = min(t + step, T - 1)
        growth = prices[t:end + 1] / prices[t]
        cash = 1.0 - w.sum()  # fração em caixa (0 quando totalmente investido)
        path = value * (growth @ w + cash)
        equity[t:end + 1] = path
        value = path[-1]
        w_prev = w * growth[-1] / (growth[-1] @ w + cash)  # pesos após a deriva até o próximo rebalanceamento
        history.append((t, w))
        t = end
    return {"equity": equity, "turnover": turnover, "weights": history}


def summarize(equity: np.ndarray) -> dict:
    eq = equity[~np.isnan(equity)]
    r = eq[1:] / eq[:-1] - 1
    cagr = metrics.annualized_return(eq)
    vol = metrics.annualized_vol(r)
    return {"retorno_total": float(eq[-1] / eq[0] - 1), "cagr": cagr, "volatilidade": vol,
            "drawdown_maximo": metrics.max_drawdown(eq), "retorno_por_risco": cagr / vol if vol else 0.0}


def run(tickers: list[str] | None = None, cost: float = 0.001, log=print) -> dict:
    tickers = tickers or default_universe()
    series, errors = load_universe(tickers, "5y", log=log)
    dates, prices, names = align(series)
    log(f"universo: {', '.join(names)} | {len(dates)} pregões | custo {cost:.2%} por unidade de giro | rebalanceamento a cada {STEP} pregões")
    strategies = {"igual peso": (weights_equal, None), "vol. inversa": (weights_inverse_vol, None)}
    for p in PROFILES.values():
        strategies[f"comitê {p.name}"] = (weights_committee, p)
    out = {}
    log(f"\n{'estratégia':20s} {'ret. total':>10s} {'CAGR':>7s} {'vol.':>6s} {'DD máx.':>8s} {'CAGR/vol.':>9s} {'giro':>6s}")
    for name, (fn, prof) in strategies.items():
        res = walk_forward(prices, fn, prof, cost=cost)
        s = summarize(res["equity"])
        s["giro_total"] = res["turnover"]
        out[name] = {**s, "equity": res["equity"].tolist()}
        log(f"{name:20s} {s['retorno_total']:10.1%} {s['cagr']:7.1%} {s['volatilidade']:6.1%} {s['drawdown_maximo']:8.1%} {s['retorno_por_risco']:9.2f} {res['turnover']:6.1f}")
    out["_meta"] = {"tickers": names, "dates": dates.tolist(), "lookback": LOOKBACK, "step": STEP, "cost": cost}
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers")
    ap.add_argument("--cost", type=float, default=0.001)
    a = ap.parse_args()
    run(a.tickers.split(",") if a.tickers else None, a.cost)
