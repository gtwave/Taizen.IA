"""Dados de mercado para o comitê: histórico de preços (brapi.dev) com cache em disco e validação."""

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import requests

from src.committee.config import PRICES_DIR

BRAPI = "https://brapi.dev/api"
TIMEOUT_S = 40
REQUEST_DELAY_S = 1.0
MAX_ABS_DAILY_RETURN = 0.30  # variação diária acima disto é suspeita (erro de dado ou evento não ajustado)


class MarketDataError(Exception):
    """Falha ao obter ou validar dados de mercado de um ticker."""


@dataclass
class PriceSeries:
    ticker: str
    dates: np.ndarray   # epoch em segundos (UTC)
    prices: np.ndarray  # preço ajustado (dividendos e desdobramentos), quando disponível
    source: str = "brapi"


def _headers() -> dict:
    token = os.environ.get("BRAPI_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def parse_history(ticker: str, payload: dict) -> PriceSeries:
    """Extrai a série do JSON da brapi, preferindo `adjustedClose`; falha de forma explícita se não houver dados."""
    results = payload.get("results") or []
    if not results:
        raise MarketDataError(f"{ticker}: resposta sem resultados ({payload.get('message', 'sem mensagem')})")
    rows = results[0].get("historicalDataPrice") or []
    pts = []
    for r in rows:
        px = r.get("adjustedClose")
        if px is None:
            px = r.get("close")
        if px is not None and r.get("date") is not None:
            pts.append((int(r["date"]), float(px)))
    if len(pts) < 30:
        raise MarketDataError(f"{ticker}: histórico insuficiente ({len(pts)} pontos)")
    pts.sort()
    return PriceSeries(ticker, np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))


def validate_series(s: PriceSeries) -> list[str]:
    """Avisos de qualidade: preços não positivos, datas repetidas e saltos diários suspeitos."""
    warnings = []
    if np.any(s.prices <= 0):
        warnings.append("há preços não positivos")
    if len(np.unique(s.dates)) != len(s.dates):
        warnings.append("há datas repetidas")
    rets = np.diff(s.prices) / s.prices[:-1]
    n_big = int(np.sum(np.abs(rets) > MAX_ABS_DAILY_RETURN))
    if n_big:
        warnings.append(f"{n_big} variação(ões) diária(s) acima de {MAX_ABS_DAILY_RETURN:.0%} (conferir ajustes)")
    return warnings


def fetch_history(ticker: str, range_: str = "5y", cache_dir=PRICES_DIR, max_age_h: float = 20.0,
                  session: requests.Session | None = None, now=time.time) -> PriceSeries:
    """Histórico diário com cache em disco (um arquivo por ticker e período)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker}_{range_}.json"
    if cache.exists() and (now() - cache.stat().st_mtime) / 3600 < max_age_h:
        return parse_history(ticker, json.loads(cache.read_text(encoding="utf-8")))
    http = session or requests
    resp = http.get(f"{BRAPI}/quote/{ticker}", params={"range": range_, "interval": "1d"}, headers=_headers(), timeout=TIMEOUT_S)
    if resp.status_code == 401:
        raise MarketDataError(f"{ticker}: a brapi.dev exige token para este ticker — defina BRAPI_TOKEN (cadastro gratuito)")
    if resp.status_code != 200:
        raise MarketDataError(f"{ticker}: HTTP {resp.status_code}")
    payload = resp.json()
    series = parse_history(ticker, payload)
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return series


def load_universe(tickers: list[str], range_: str = "5y", log=print, **kw) -> tuple[dict[str, PriceSeries], dict[str, str]]:
    """Baixa o histórico de cada ticker. Devolve (séries, erros por ticker) sem abortar pelo primeiro erro."""
    series, errors = {}, {}
    for i, t in enumerate(tickers):
        try:
            series[t] = fetch_history(t, range_, **kw)
        except (MarketDataError, requests.RequestException) as exc:
            errors[t] = str(exc)
            log(f"  [{t}] indisponível: {exc}")
        if i < len(tickers) - 1:
            time.sleep(REQUEST_DELAY_S if not kw.get("session") else 0)
    return series, errors


def align(series: dict[str, PriceSeries]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Alinha as séries nas datas em comum. Devolve (datas, matriz de preços T x N, tickers na ordem das colunas)."""
    tickers = list(series)
    common = set(series[tickers[0]].dates.tolist())
    for t in tickers[1:]:
        common &= set(series[t].dates.tolist())
    dates = np.array(sorted(common))
    cols = []
    for t in tickers:
        idx = {d: p for d, p in zip(series[t].dates.tolist(), series[t].prices.tolist())}
        cols.append([idx[d] for d in dates.tolist()])
    return dates, np.array(cols).T, tickers
