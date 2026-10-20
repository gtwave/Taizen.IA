"""Caches para reduzir custo e latência: exato (respostas validadas do LLM) e semântico (perguntas parecidas).

Duas regras de segurança valem para os dois:
  1. só se guarda o que já foi VALIDADO (um JSON inválido em cache seria repetido para sempre);
  2. toda entrada tem TTL — dado de mercado envelhece, e uma resposta de ontem sobre "a cotação de hoje" é um erro.

O cache semântico tem um perigo extra: "cotação da VALE3" e "cotação da PETR4" são frases muito parecidas para um
modelo de embeddings, mas têm respostas diferentes. Por isso toda consulta carrega uma `guard` (chave de segurança, por
exemplo o conjunto de tickers citados): só há acerto se a guarda for idêntica, além da similaridade alta.
"""

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.observability.tracing import current_span


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_TICKER_RE = re.compile(r"\b[A-Za-z]{4}\d{1,2}\b")


def question_guard(question: str) -> str:
    """Chave de segurança de uma pergunta: tickers citados (explícitos ou pelo nome da empresa) e janela em dias."""
    from src.rag.companies import count_mentions

    tickers = {m.upper() for m in _TICKER_RE.findall(question)} | set(count_mentions(question))
    days = re.findall(r"(\d+)\s*(?:dias?|d)\b", question.lower())
    return "|".join(sorted(tickers)) + (f"#{'/'.join(days)}d" if days else "")


@dataclass
class Hit:
    answer: str
    kind: str  # exact | semantic
    similarity: float
    age_s: float
    matched_query: str


class _Store:
    """Persistência simples em JSON, com despejo do mais antigo quando passa de `max_entries`."""

    def __init__(self, path: Path | None, max_entries: int):
        self.path, self.max_entries = (Path(path) if path else None), max_entries
        self.entries: list[dict] = []
        if self.path and self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.entries = []

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, ensure_ascii=False), encoding="utf-8")

    def evict(self, now: float) -> int:
        before = len(self.entries)
        self.entries = [e for e in self.entries if now - e["created"] <= e["ttl"]]
        if len(self.entries) > self.max_entries:
            self.entries = sorted(self.entries, key=lambda e: e["created"])[-self.max_entries:]
        return before - len(self.entries)


class SemanticCache:
    def __init__(self, embeddings, path: Path | None = None, threshold: float = 0.92, ttl_s: float = 900,
                 max_entries: int = 500, clock=time.time):
        self.embeddings, self.threshold, self.ttl_s, self.clock = embeddings, threshold, ttl_s, clock
        self.store = _Store(path, max_entries)
        self.stats = {"exact": 0, "semantic": 0, "miss": 0, "guard_blocked": 0, "evicted": 0}

    def _record(self, hit: Hit | None) -> None:
        sp = current_span()
        if sp is not None:
            sp.set(cache="miss" if hit is None else hit.kind, cache_similarity=None if hit is None else round(hit.similarity, 4))

    def get(self, query: str, namespace: str = "", guard: str = "") -> Hit | None:
        now = self.clock()
        self.stats["evicted"] += self.store.evict(now)
        norm = normalize(query)
        pool = [e for e in self.store.entries if e["ns"] == namespace]
        for e in pool:  # 1) acerto exato: sem custo de embedding
            if e["norm"] == norm and e["guard"] == guard:
                self.stats["exact"] += 1
                hit = Hit(e["answer"], "exact", 1.0, now - e["created"], e["query"])
                self._record(hit)
                return hit
        if pool:  # 2) acerto semântico
            q = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            best, best_sim, blocked = None, -1.0, False
            for e in pool:
                v = np.asarray(e["vec"], dtype=np.float32)
                sim = float(q @ (v / (np.linalg.norm(v) or 1.0)))
                if sim < self.threshold:
                    continue
                if e["guard"] != guard:
                    blocked = True
                    continue
                if sim > best_sim:
                    best, best_sim = e, sim
            if best is not None:
                self.stats["semantic"] += 1
                hit = Hit(best["answer"], "semantic", best_sim, now - best["created"], best["query"])
                self._record(hit)
                return hit
            if blocked:
                self.stats["guard_blocked"] += 1
        self.stats["miss"] += 1
        self._record(None)
        return None

    def put(self, query: str, answer: str, namespace: str = "", guard: str = "", ttl_s: float | None = None) -> None:
        vec = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        self.store.entries.append({"ns": namespace, "query": query, "norm": normalize(query), "guard": guard, "answer": answer,
                                   "vec": [round(float(x), 6) for x in vec], "created": self.clock(),
                                   "ttl": self.ttl_s if ttl_s is None else ttl_s})
        self.stats["evicted"] += self.store.evict(self.clock())
        self.store.save()

    def hit_rate(self) -> float:
        total = sum(self.stats[k] for k in ("exact", "semantic", "miss"))
        return (self.stats["exact"] + self.stats["semantic"]) / total if total else 0.0


class ExactCache:
    """Cache exato de saídas JÁ VALIDADAS do LLM. Chave = hash(modelo, system, user)."""

    def __init__(self, path: Path | None = None, ttl_s: float = 6 * 3600, max_entries: int = 1000, clock=time.time):
        self.ttl_s, self.clock = ttl_s, clock
        self.store = _Store(path, max_entries)
        self.stats = {"hit": 0, "miss": 0}

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        return hashlib.sha256(json.dumps([model, system, user], ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        now = self.clock()
        self.store.evict(now)
        for e in self.store.entries:
            if e["key"] == key:
                self.stats["hit"] += 1
                sp = current_span()
                if sp is not None:
                    sp.set(cache="exact")
                return e["value"]
        self.stats["miss"] += 1
        return None

    def put(self, key: str, value: str, ttl_s: float | None = None) -> None:
        self.store.entries = [e for e in self.store.entries if e["key"] != key]
        self.store.entries.append({"key": key, "value": value, "created": self.clock(), "ttl": self.ttl_s if ttl_s is None else ttl_s})
        self.store.evict(self.clock())
        self.store.save()
