"""Embeddings locais (CPU) com multilingual-e5-small, no formato esperado pelo LangChain.

Os modelos E5 foram treinados com prefixos: "passage: " nos documentos e "query: " nas perguntas.
"""

from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL, use_prefix: bool = True, batch_size: int = 32):
        self.model_name = model_name
        self.use_prefix = use_prefix
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # import tardio: carregar custa segundos

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False)
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "passage: " if self.use_prefix else ""
        return self._encode([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        prefix = "query: " if self.use_prefix else ""
        return self._encode([prefix + text])[0]
