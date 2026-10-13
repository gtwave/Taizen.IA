import tempfile
import time
import unittest
from unittest.mock import MagicMock

from tests.helpers import HashEmbeddings, sample_articles
from src.rag.chunking import article_to_chunks, articles_to_chunks, make_splitter
from src.rag.embeddings import E5Embeddings
from src.rag.retrieval import (BM25Index, make_where_fn, rrf_fuse, tokenize, ticker_filter, vector_search,
                               hybrid_search, Hit)
from src.rag.store import get_store, index_chunks
from src.tools.news_search import search_news
from src.rag.search import Retriever


class TestChunking(unittest.TestCase):
    def test_size_overlap_and_metadata(self):
        art = sample_articles()[0]
        chunks = article_to_chunks(art, make_splitter(300, 50), with_title=False)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c.page_content) <= 300 for c in chunks))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        self.assertTrue(all(c.metadata["has_PETR4"] is True for c in chunks))
        self.assertTrue(all(c.metadata["n_chunks"] == len(chunks) for c in chunks))

    def test_title_header_is_optional(self):
        art = sample_articles()[1]
        with_t = article_to_chunks(art, with_title=True)[0].page_content
        without = article_to_chunks(art, with_title=False)[0].page_content
        self.assertTrue(with_t.startswith(art["title"]))
        self.assertFalse(without.startswith(art["title"]))

    def test_no_ticker_flags_when_none_cited(self):
        chunk = article_to_chunks(sample_articles()[2])[0]
        self.assertFalse(any(k.startswith("has_") for k in chunk.metadata))


class TestEmbeddingsPrefix(unittest.TestCase):
    def test_e5_prefixes(self):
        emb = E5Embeddings()
        fake = MagicMock()
        fake.encode.return_value = MagicMock(tolist=lambda: [[0.0]])
        emb._model = fake
        emb.embed_documents(["texto"])
        emb.embed_query("pergunta")
        self.assertEqual(fake.encode.call_args_list[0].args[0], ["passage: texto"])
        self.assertEqual(fake.encode.call_args_list[1].args[0], ["query: pergunta"])


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.store = get_store(HashEmbeddings(), self.tmp.name, "teste")
        self.chunks = articles_to_chunks(sample_articles(), chunk_size=300, chunk_overlap=50)
        self.added = index_chunks(self.store, self.chunks, log=lambda *_: None)

    def test_indexing_is_incremental(self):
        self.assertEqual(self.added, len(self.chunks))
        self.assertEqual(index_chunks(self.store, self.chunks, log=lambda *_: None), 0)

    def test_vector_search_ranks_relevant_article_first(self):
        hits = vector_search(self.store, "dividendos da Petrobras", k=3)
        self.assertEqual(hits[0].doc.metadata["article_id"], "a1")
        self.assertGreaterEqual(hits[0].score, hits[-1].score)

    def test_metadata_filters(self):
        hits = vector_search(self.store, "produção", k=5, where=ticker_filter("VALE3"))
        self.assertTrue(hits and all(h.doc.metadata.get("has_VALE3") for h in hits))
        recent = vector_search(self.store, "Selic", k=5, where=ticker_filter(None, since_ts=1791021000))
        self.assertTrue(recent and all(h.doc.metadata["published_ts"] >= 1791021000 for h in recent))

    def test_bm25_ignores_accents_and_stopwords(self):
        self.assertEqual(tokenize("Política Monetária da Selic"), ["politica", "monetaria", "selic"])
        bm = BM25Index.from_store(self.store)
        hits = bm.search("politica monetaria", k=3)
        self.assertEqual(hits[0].doc.metadata["article_id"], "a3")
        only_vale = bm.search("minério produção", k=5, where_fn=make_where_fn("VALE3"))
        self.assertTrue(all(h.doc.metadata.get("has_VALE3") for h in only_vale))

    def test_rrf_math_and_consensus(self):
        mk = lambda i: Hit(self.chunks[0].__class__(page_content=i, id=i), 0.0, "x")
        a, b, c = mk("A"), mk("B"), mk("C")
        fused = rrf_fuse([[a, b], [b, c]], k=3, rrf_k=60)
        self.assertEqual(fused[0].id, "B")  # presente nos dois rankings
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)

    def test_hybrid_returns_k_unique_chunks(self):
        bm = BM25Index.from_store(self.store)
        hits = hybrid_search(self.store, bm, "Petrobras dividendos", k=4)
        self.assertEqual(len({h.id for h in hits}), len(hits))
        self.assertLessEqual(len(hits), 4)


class TestNewsTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        store = get_store(HashEmbeddings(), self.tmp.name, "teste_tool")
        index_chunks(store, articles_to_chunks(sample_articles(), 300, 50), log=lambda *_: None)
        self.retriever = Retriever.__new__(Retriever)
        self.retriever.store, self.retriever._bm25 = store, None

    def test_one_excerpt_per_article_and_short_fields(self):
        out = search_news("Petrobras dividendos", k=3, retriever=self.retriever)
        ids = [r["url"] for r in out["resultados"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(out["resultados"][0]), {"n", "fonte", "titulo", "data", "url", "trecho"})
        self.assertLessEqual(len(out["resultados"][0]["trecho"]), 700)

    def test_empty_result_returns_warning(self):
        out = search_news("petróleo", ticker="WEGE3", retriever=self.retriever)
        self.assertEqual(out["resultados"], [])
        self.assertIn("aviso", out)

    def test_days_filter_excludes_old_articles(self):
        out = search_news("Selic", days=1, retriever=self.retriever)  # artigos de exemplo são de 2026-10-0x
        self.assertEqual(out["resultados"], [])


if __name__ == "__main__":
    unittest.main()
