"""Busca interativa no índice (sem LLM, sem custo).

Uso:  python -m src.rag.search "dividendos da Vale" [--mode vector|mmr|bm25|hybrid] [--ticker VALE3] [-k 5]
"""

import argparse
import sys
import time

from src.rag.embeddings import E5Embeddings
from src.rag.retrieval import BM25Index, Hit, hybrid_search, make_where_fn, mmr_search, ticker_filter, vector_search
from src.rag.store import get_store


class Retriever:
    """Agrupa embeddings, índice vetorial e BM25 para uso repetido (carrega o modelo uma só vez)."""

    def __init__(self):
        self.store = get_store(E5Embeddings())
        self._bm25: BM25Index | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.from_store(self.store)
        return self._bm25

    def search(self, query: str, mode: str = "hybrid", k: int = 5, ticker: str | None = None,
               since_ts: int | None = None) -> list[Hit]:
        where = ticker_filter(ticker, since_ts)
        if mode == "vector":
            return vector_search(self.store, query, k, where)
        if mode == "mmr":
            return mmr_search(self.store, query, k, where=where)
        if mode == "bm25":
            return self.bm25.search(query, k, make_where_fn(ticker, since_ts))
        return hybrid_search(self.store, self.bm25, query, k, where, make_where_fn(ticker, since_ts))


def show(hits: list[Hit], width: int = 150) -> None:
    for h in hits:
        m = h.doc.metadata
        body = h.doc.page_content.split("\n\n", 1)[-1].replace("\n", " ")
        print(f"{h.rank}. [{h.score:.3f}] {m['source']} · {m['published'][:10]} · {m['title'][:90]}")
        print(f"     {body[:width]}...")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query")
    ap.add_argument("--mode", default="hybrid", choices=["vector", "mmr", "bm25", "hybrid"])
    ap.add_argument("--ticker")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    r = Retriever()
    t0 = time.perf_counter()
    hits = r.search(a.query, a.mode, a.k, a.ticker)
    dt = (time.perf_counter() - t0) * 1000
    show(hits)
    print(f"\n{len(hits)} resultado(s) em {dt:.0f} ms (modo {a.mode})")
