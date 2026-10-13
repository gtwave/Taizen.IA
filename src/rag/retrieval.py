"""Recuperação: busca vetorial, MMR, filtros por metadado, BM25 e busca híbrida (RRF)."""

import re
import unicodedata
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

STOPWORDS = set(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sobre entre e ou mas que se "
    "ao aos à às pelo pela pelos pelas seu sua seus suas este esta estes estas esse essa isso isto como mais "
    "menos já também ser foi são é está estão há tem têm ter não".split()
)


@dataclass
class Hit:
    doc: Document
    score: float
    method: str
    rank: int = 0
    extras: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.doc.id


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acentos, só palavras; descarta stopwords e tokens de 1 caractere."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and len(t) > 1]


def ticker_filter(ticker: str | None, since_ts: int | None = None) -> dict | None:
    """Filtro de metadados do Chroma: ticker citado e/ou publicação a partir de um instante."""
    conds = []
    if ticker:
        conds.append({f"has_{ticker.upper()}": True})
    if since_ts:
        conds.append({"published_ts": {"$gte": int(since_ts)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def vector_search(store: Chroma, query: str, k: int = 5, where: dict | None = None) -> list[Hit]:
    """Busca por similaridade. Com distância cosseno, similaridade = 1 - distância."""
    res = store.similarity_search_with_score(query, k=k, filter=where)
    return [Hit(d, 1.0 - dist, "vector", i + 1) for i, (d, dist) in enumerate(res)]


def mmr_search(store: Chroma, query: str, k: int = 5, fetch_k: int = 25, lambda_mult: float = 0.6,
               where: dict | None = None) -> list[Hit]:
    """Relevância máxima marginal: equilibra proximidade da pergunta e diversidade entre os resultados."""
    docs = store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=where)
    return [Hit(d, 0.0, "mmr", i + 1) for i, d in enumerate(docs)]


class BM25Index:
    """Índice lexical em memória (BM25Okapi) sobre os mesmos chunks do índice vetorial."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])

    @classmethod
    def from_store(cls, store: Chroma) -> "BM25Index":
        raw = store.get(include=["documents", "metadatas"])
        docs = [Document(page_content=t, metadata=m, id=i)
                for i, t, m in zip(raw["ids"], raw["documents"], raw["metadatas"])]
        return cls(docs)

    def search(self, query: str, k: int = 5, where_fn=None) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            if where_fn and not where_fn(self.docs[i].metadata):
                continue
            hits.append(Hit(self.docs[i], float(scores[i]), "bm25", len(hits) + 1))
            if len(hits) == k:
                break
        return hits


def rrf_fuse(rankings: list[list[Hit]], k: int = 5, rrf_k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = soma de 1 / (rrf_k + posição) em cada ranking."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for pos, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + pos)
            docs[hit.id] = hit.doc
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [Hit(docs[i], scores[i], "hybrid", r + 1) for r, i in enumerate(best)]


def hybrid_search(store: Chroma, bm25: BM25Index, query: str, k: int = 5, where: dict | None = None,
                  where_fn=None, fetch_k: int = 20) -> list[Hit]:
    """Combina busca vetorial e BM25 por RRF."""
    v = vector_search(store, query, fetch_k, where)
    b = bm25.search(query, fetch_k, where_fn)
    return rrf_fuse([v, b], k=k)


def make_where_fn(ticker: str | None, since_ts: int | None = None):
    """Equivalente em Python do filtro de metadados, para aplicar ao BM25."""
    if not ticker and not since_ts:
        return None

    def fn(meta: dict) -> bool:
        if ticker and not meta.get(f"has_{ticker.upper()}"):
            return False
        if since_ts and meta.get("published_ts", 0) < since_ts:
            return False
        return True

    return fn
