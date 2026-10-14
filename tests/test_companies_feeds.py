import unittest
from unittest.mock import MagicMock

from tests import helpers  # noqa: F401  (ajusta o sys.path)
from src.rag import feeds
from src.rag.companies import find_tickers
from src.rag.ingest import merge_new

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>Santander (SANB11) paga JCP</title><link>https://ex.com/a</link>
<pubDate>Tue, 13 Oct 2026 11:57:28 +0000</pubDate>
<content:encoded><![CDATA[<p>O banco <a href="/x">Santander</a> ( <strong>SANB11</strong> ) aprovou <b>JCP</b>.</p><p>Segundo parágrafo.</p>]]></content:encoded>
</item>
<item><title>Sem link</title></item>
</channel></rss>"""


class TestCompanies(unittest.TestCase):
    def test_company_names_and_tickers(self):
        self.assertEqual(find_tickers("A Petrobras (PETR4) subiu"), ["PETR4"])
        self.assertEqual(find_tickers("", title="Resultado do Itaú"), ["ITUB4"])

    def test_single_passing_mention_in_body_is_ignored(self):
        self.assertEqual(find_tickers("O texto cita a Petrobras apenas uma vez."), [])

    def test_ambiguous_names_need_company_context(self):
        self.assertEqual(find_tickers("Vale a pena investir em renda fixa? Vale tudo por R$ 150 milhões?"), [])
        self.assertEqual(find_tickers("A Vale reportou lucro. Segundo a Vale, a produção subiu."), ["VALE3"])
        self.assertEqual(find_tickers("o minério vale mais hoje; vale3 caiu"), [])

    def test_word_boundary_avoids_substring_matches(self):
        self.assertEqual(find_tickers("Os imóveis localizam-se na região; localizamos o problema; localizado."), [])


class TestFeeds(unittest.TestCase):
    def test_parse_cleans_html_without_breaking_inline_tags(self):
        entries = feeds.parse_entries(RSS, "teste")
        self.assertEqual(len(entries), 1)  # item sem link é descartado
        e = entries[0]
        self.assertIn("Santander (SANB11) aprovou JCP.", e["text"])
        self.assertEqual(e["text"].count("\n"), 1)  # só entre parágrafos
        self.assertEqual(e["published"], "2026-10-13T11:57:28+00:00")
        self.assertEqual(e["id"], feeds.article_id("https://ex.com/a"))

    def test_merge_new_drops_duplicates_and_short_texts(self):
        base = {"source": "s", "title": "t", "url": "u", "published": None, "published_ts": None}
        fetched = [{**base, "id": "1", "text": "x" * 400}, {**base, "id": "1", "text": "x" * 400},
                   {**base, "id": "2", "text": "curto"}, {**base, "id": "3", "text": "Petrobras " * 60}]
        new, stats = merge_new({"9": {}}, fetched)
        self.assertEqual([a["id"] for a in new], ["1", "3"])
        self.assertEqual(stats, {"fetched": 4, "duplicates": 1, "too_short": 1, "new": 2})
        self.assertEqual(new[1]["tickers"], ["PETR4"])

    def test_robots_disallow_is_respected(self):
        feeds._robots_cache.clear()
        session = MagicMock()
        session.get.return_value = MagicMock(status_code=200, text="User-agent: *\nDisallow: /privado/")
        self.assertFalse(feeds.robots_allows("https://site.test/privado/feed", session))
        self.assertTrue(feeds.robots_allows("https://site.test/feed/", session))

    def test_fetch_feed_stops_on_http_error(self):
        feeds._robots_cache.clear()
        session = MagicMock()
        session.get.side_effect = [MagicMock(status_code=404, text=""),  # robots.txt ausente
                                   MagicMock(status_code=503, content=b"")]
        out = feeds.fetch_feed(feeds.Feed("t", "https://site.test/feed/", 3), session, log=lambda *_: None)
        self.assertEqual(out, [])
        self.assertEqual(session.get.call_count, 2)  # parou na primeira página com erro


if __name__ == "__main__":
    unittest.main()
