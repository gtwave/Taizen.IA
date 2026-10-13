"""Divisão de artigos em chunks com metadados (insumo do índice vetorial)."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def make_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS, length_function=len,
    )


def article_to_chunks(article: dict, splitter: RecursiveCharacterTextSplitter | None = None,
                      with_title: bool = True) -> list[Document]:
    """Converte um artigo em Documents. Cada chunk herda os metadados do artigo.

    with_title=True prefixa o título ao texto do chunk (contexto que ajuda a busca, ao custo de tokens).
    Chroma só aceita metadados escalares; por isso os tickers viram flags booleanas `has_<TICKER>`.
    """
    splitter = splitter or make_splitter()
    parts = splitter.split_text(article["text"])
    docs = []
    for i, part in enumerate(parts):
        meta = {
            "article_id": article["id"],
            "source": article["source"],
            "title": article["title"],
            "url": article["url"],
            "published": article.get("published") or "",
            "published_ts": article.get("published_ts") or 0,
            "chunk_index": i,
            "n_chunks": len(parts),
        }
        for t in article.get("tickers", []):
            meta[f"has_{t}"] = True
        content = f"{article['title']}\n\n{part}" if with_title else part
        docs.append(Document(page_content=content, metadata=meta, id=f"{article['id']}:{i}"))
    return docs


def articles_to_chunks(articles: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
                       with_title: bool = True) -> list[Document]:
    splitter = make_splitter(chunk_size, chunk_overlap)
    out: list[Document] = []
    for a in articles:
        out.extend(article_to_chunks(a, splitter, with_title))
    return out
