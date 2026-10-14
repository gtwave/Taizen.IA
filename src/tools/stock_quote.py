import os

import requests

BRAPI_BASE_URL = "https://brapi.dev/api"


def get_stock_quote(ticker: str) -> dict:
    """Busca a cotação atual de uma ação da B3 na brapi.dev.

    Sem token, a brapi só atende alguns tickers de teste (PETR4, VALE3, ITUB4, MGLU3); os demais exigem um token
    gratuito em BRAPI_TOKEN. Falhas voltam como {"error": ...} para que o modelo possa explicá-las ao usuário.
    """
    token = os.environ.get("BRAPI_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(f"{BRAPI_BASE_URL}/quote/{ticker.upper()}", headers=headers, timeout=10)
    except requests.RequestException as exc:
        return {"error": f"Falha de rede ao consultar '{ticker}': {exc.__class__.__name__}."}
    if resp.status_code == 401:
        return {"error": f"A brapi.dev exige token para '{ticker}'. Defina BRAPI_TOKEN (cadastro gratuito) ou use PETR4, VALE3, ITUB4 ou MGLU3."}
    if resp.status_code != 200:
        return {"error": f"Erro HTTP {resp.status_code} ao consultar '{ticker}'."}
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return {"error": f"Ticker '{ticker}' não encontrado."}
    quote = results[0]
    return {
        "symbol": quote.get("symbol"),
        "name": quote.get("longName"),
        "price": quote.get("regularMarketPrice"),
        "currency": quote.get("currency"),
        "change_percent": quote.get("regularMarketChangePercent"),
        "day_high": quote.get("regularMarketDayHigh"),
        "day_low": quote.get("regularMarketDayLow"),
        "fifty_two_week_range": quote.get("fiftyTwoWeekRange"),
        "price_earnings": quote.get("priceEarnings"),
        "market_time": quote.get("regularMarketTime"),
    }
