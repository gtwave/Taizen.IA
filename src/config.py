"""Configuração central da Fase 1: caminhos de dados e parâmetros do pipeline RAG."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Dados pesados ficam em DATA_DIR (HD externo, se configurado); sem a variável, usa ./data (ignorada pelo Git).
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

NEWS_FILE = RAW_DIR / "news.jsonl"
COLLECTION = "noticias_b3"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
