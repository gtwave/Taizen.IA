"""Ferramenta search_news: recupera trechos de notícias do índice local (RAG) para o agente."""

import time

from src.rag.search import Retriever

_retriever: Retriever | None = None
EXCERPT_CHARS = 700


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def search_news(query: str, ticker: str | None = None, days: int | None = None, k: int = 4,
                retriever: Retriever | None = None) -> dict:
    """Busca trechos de notícias relevantes. Devolve poucos campos, para gastar poucos tokens.

    ticker: filtra por empresa citada (ex.: "VALE3"); days: só notícias dos últimos N dias.
    """
    r = retriever or get_retriever()
    since_ts = int(time.time() - days * 86400) if days else None
    hits = r.search(query, mode="hybrid", k=k * 2, ticker=ticker, since_ts=since_ts)
    results, seen = [], set()
    for h in hits:
        m = h.doc.metadata
        if m["article_id"] in seen:  # um trecho por artigo: diversidade de fontes
            continue
        seen.add(m["article_id"])
        body = h.doc.page_content.split("\n\n", 1)[-1].replace("\n", " ")
        results.append({
            "n": len(results) + 1,
            "fonte": m["source"],
            "titulo": m["title"],
            "data": m["published"][:10],
            "url": m["url"],
            "trecho": body[:EXCERPT_CHARS],
        })
        if len(results) == k:
            break
    if not results:
        return {"resultados": [], "aviso": "Nenhuma notícia relevante encontrada no índice para esta consulta."}
    return {"resultados": results}
