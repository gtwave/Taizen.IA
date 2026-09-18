import requests

BRAPI_BASE_URL = "https://brapi.dev/api"


def get_stock_quote(ticker: str) -> dict:
    """Busca a cotação atual de uma ação da B3 na brapi.dev."""
    resp = requests.get(f"{BRAPI_BASE_URL}/quote/{ticker.upper()}", timeout=10)
    resp.raise_for_status()
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
