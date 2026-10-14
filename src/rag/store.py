"""Índice vetorial persistente (Chroma) e indexação incremental dos chunks."""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import CHROMA_DIR, COLLECTION, NEWS_FILE


def get_store(embeddings: Embeddings, persist_dir: Path | str = CHROMA_DIR, collection: str = COLLECTION) -> Chroma:
    """Abre (ou cria) a coleção. Distância cosseno, adequada a embeddings normalizados."""
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def load_articles(path: Path = NEWS_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_chunks(store: Chroma, chunks: list[Document], batch_size: int = 64, log=print) -> int:
    """Indexa só os chunks ainda ausentes (ids determinísticos). Retorna quantos foram adicionados."""
    existing = set(store.get(include=[])["ids"])
    todo = [c for c in chunks if c.id not in existing]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        store.add_documents(batch, ids=[c.id for c in batch])
        log(f"  indexados {min(start + batch_size, len(todo))}/{len(todo)}")
    return len(todo)
