"""Utilitários de teste: embeddings falsos e determinísticos (sem modelo, sem rede, sem custo)."""

import hashlib
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import Embeddings

DIM = 64


class HashEmbeddings(Embeddings):
    """Bag-of-words por hashing: textos com palavras em comum ficam próximos. Só para testes."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def sample_articles():
    long_text = ("A Petrobras anunciou dividendos extraordinários. " * 30).strip()
    return [
        {"id": "a1", "source": "fonte_x", "title": "Petrobras anuncia dividendos", "url": "https://ex.com/1",
         "published": "2026-10-01T10:00:00+00:00", "published_ts": 1790848800, "text": long_text,
         "tickers": ["PETR4"]},
        {"id": "a2", "source": "fonte_y", "title": "Vale reduz produção de minério", "url": "https://ex.com/2",
         "published": "2026-10-02T10:00:00+00:00", "published_ts": 1790935200,
         "text": "A Vale informou redução na produção de minério de ferro no trimestre. " * 12,
         "tickers": ["VALE3"]},
        {"id": "a3", "source": "fonte_x", "title": "Banco Central mantém Selic", "url": "https://ex.com/3",
         "published": "2026-10-03T10:00:00+00:00", "published_ts": 1791021600,
         "text": "O Comitê de Política Monetária manteve a taxa Selic inalterada. " * 12, "tickers": []},
    ]
