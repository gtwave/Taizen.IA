"""Constrói o índice vetorial a partir do acervo bruto.

Uso:  python -m src.rag.index [--chunk-size 800] [--overlap 120] [--no-title]
"""

import argparse
import sys
import time

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from src.rag.chunking import articles_to_chunks
from src.rag.embeddings import E5Embeddings
from src.rag.store import get_store, index_chunks, load_articles


def run(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, with_title: bool = True, log=print) -> dict:
    articles = load_articles()
    chunks = articles_to_chunks(articles, chunk_size, overlap, with_title)
    log(f"{len(articles)} artigos -> {len(chunks)} chunks (tamanho {chunk_size}, sobreposição {overlap})")
    emb = E5Embeddings()
    t0 = time.perf_counter()
    _ = emb.model  # força o carregamento do modelo para medi-lo separadamente
    t_load = time.perf_counter() - t0
    store = get_store(emb)
    t1 = time.perf_counter()
    added = index_chunks(store, chunks, log=log)
    t_index = time.perf_counter() - t1
    rate = added / t_index if t_index and added else 0.0
    log(f"modelo carregado em {t_load:.1f}s | {added} chunks indexados em {t_index:.1f}s ({rate:.1f} chunks/s)")
    log(f"índice em {CHROMA_DIR}")
    return {"articles": len(articles), "chunks": len(chunks), "added": added, "load_s": t_load, "index_s": t_index}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    ap.add_argument("--no-title", action="store_true")
    a = ap.parse_args()
    run(a.chunk_size, a.overlap, not a.no_title)
