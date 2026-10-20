import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tests import helpers  # noqa: F401  (ajusta o sys.path)
from src import agent_rag
from src.committee.brains import BrainError, LLMBrain, ManagerOutput, _ask
from src.committee.config import DISCLAIMER
from src.observability import cache as cache_mod
from src.observability import evaluate, golden, pricing, report
from src.observability.cache import ExactCache, SemanticCache, question_guard
from src.observability.langchain_hook import TracingCallbackHandler
from src.observability.tracing import JsonlSink, MemorySink, Tracer, configure, get_tracer, read_spans
from tests.test_agent_rag import FakeToolModel, ask_tools_then_answer
from tests.helpers import HashEmbeddings


class PricedFake(BaseChatModel):
    """Modelo simulado com nome e uso de tokens, para o callback medir custo."""

    model: str = "claude-haiku-4-5-20251001"
    reply: str = "ok"
    tokens: tuple = (1000, 500)

    @property
    def _llm_type(self):
        return "priced-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessage(content=self.reply, usage_metadata={"input_tokens": self.tokens[0], "output_tokens": self.tokens[1],
                                                            "total_tokens": sum(self.tokens)})
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class TestTracing(unittest.TestCase):
    def test_nested_spans_share_trace_and_link_parents(self):
        sink = MemorySink()
        tr = Tracer([sink])
        with tr.span("raiz", "chain") as root:
            with tr.span("filho", "node"):
                pass
        child, parent = sink.spans  # filhos são gravados antes dos pais
        self.assertEqual(child["trace_id"], parent["trace_id"])
        self.assertEqual(child["parent_id"], root.span_id)
        self.assertIsNone(parent["parent_id"])
        self.assertGreaterEqual(parent["duration_ms"], child["duration_ms"])

    def test_error_is_recorded_and_propagated(self):
        sink = MemorySink()
        tr = Tracer([sink])
        with self.assertRaises(ValueError):
            with tr.span("falha"):
                raise ValueError("boom")
        self.assertEqual(sink.spans[0]["status"], "error")
        self.assertIn("boom", sink.spans[0]["error"])

    def test_disabled_tracer_writes_nothing_and_still_works(self):
        tr = Tracer()
        self.assertFalse(tr.enabled)
        with tr.span("x") as s:
            s.set(a=1)
        self.assertEqual(s.attrs["a"], 1)

    def test_jsonl_roundtrip_and_truncated_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            tr = Tracer([JsonlSink(Path(d))])
            with tr.span("a", "chain", x=1):
                pass
            file = next(Path(d).glob("traces-*.jsonl"))
            file.write_text(file.read_text(encoding="utf-8") + '{"name": "truncado', encoding="utf-8")
            spans = read_spans(d)
            self.assertEqual(len(spans), 1)
            self.assertEqual(spans[0]["attrs"]["x"], 1)
            self.assertEqual(read_spans(d, since=spans[0]["start"] + 10), [])

    def test_context_propagates_into_langgraph_threads(self):
        from tests.test_committee import run_graph
        from src.committee.brains import RulesBrain

        sink = MemorySink()
        configure([sink])
        try:
            with get_tracer().span("raiz") as root:
                run_graph(RulesBrain())
        finally:
            configure([])
        nodes = [s for s in sink.spans if s["kind"] == "node"]
        self.assertEqual({s["name"] for s in nodes}, {"coletar", "analista_dados", "analista_risco", "gestor", "validador", "relatorio"})
        self.assertTrue(all(s["trace_id"] == root.trace_id and s["parent_id"] == root.span_id for s in nodes))


class TestPricing(unittest.TestCase):
    def test_known_cost_and_prefix_match(self):
        usd = pricing.cost_usd("claude-haiku-4-5-20251001", 1000, 500)
        self.assertAlmostEqual(usd, (1000 * 1.0 + 500 * 5.0) / 1e6)
        self.assertAlmostEqual(pricing.cost_brl("claude-haiku-4-5", 1000, 500), usd * pricing.usd_brl())
        self.assertEqual(pricing.price_for("claude-sonnet-5-5"), (3.0, 15.0))  # prefixo mais longo vence

    def test_unknown_model_has_no_cost_instead_of_invented_one(self):
        self.assertIsNone(pricing.cost_usd("modelo-x", 10, 10))
        self.assertIsNone(pricing.cost_usd(None, 10, 10))

    def test_provider_cache_discounts_input(self):
        full = pricing.cost_usd("claude-sonnet-5", 10_000, 0)
        cached = pricing.cost_usd("claude-sonnet-5", 10_000, 0, cache_read=9_000)
        self.assertAlmostEqual(cached, (1000 * 3 + 9000 * 3 * 0.1) / 1e6)
        self.assertLess(cached, full * 0.25)

    def test_env_override(self):
        with patch.dict(os.environ, {"PRICING_JSON": json.dumps({"meu-modelo": [2, 4]}), "USD_BRL": "6"}):
            self.assertEqual(pricing.price_for("meu-modelo-v2"), (2, 4))
            self.assertAlmostEqual(pricing.cost_brl("meu-modelo", 1_000_000, 0), 12.0)

    def _spans(self, costs, when):
        return [{"kind": "llm", "start": when, "attrs": {"cost_brl": c}} for c in costs]

    def test_budget_levels_and_projection(self):
        now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
        t = datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp()
        self.assertEqual(pricing.budget_status(self._spans([1.0], t), 50, now).level, "ok")
        self.assertEqual(pricing.budget_status(self._spans([36.0], t), 50, now).level, "warn")
        st = pricing.budget_status(self._spans([60.0], t), 50, now)
        self.assertEqual(st.level, "exceeded")
        self.assertLess(st.remaining_brl, 0)
        with self.assertRaises(pricing.BudgetExceeded):
            pricing.assert_within_budget(self._spans([60.0], t), 50, now)

    def test_budget_ignores_previous_months(self):
        now = datetime(2026, 10, 15, tzinfo=timezone.utc)
        old = datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp()
        self.assertEqual(pricing.budget_status(self._spans([500.0], old), 50, now).spent_brl, 0.0)


class TestCallbackHandler(unittest.TestCase):
    def test_llm_call_becomes_span_with_tokens_and_cost(self):
        sink = MemorySink()
        handler = TracingCallbackHandler(Tracer([sink]), role="analista")
        PricedFake(callbacks=[handler]).invoke("Qual a cotação?")
        (s,) = sink.spans
        self.assertEqual(s["kind"], "llm")
        self.assertEqual(s["attrs"]["input_tokens"], 1000)
        self.assertEqual(s["attrs"]["output_tokens"], 500)
        self.assertAlmostEqual(s["attrs"]["cost_brl"], pricing.cost_brl("claude-haiku-4-5", 1000, 500), places=5)
        self.assertEqual(s["attrs"]["role"], "analista")
        self.assertNotIn("prompt", s["attrs"])  # conteúdo não é gravado por padrão
        self.assertEqual(len(s["attrs"]["prompt_hash"]), 10)

    def test_content_capture_is_opt_in(self):
        sink = MemorySink()
        PricedFake(callbacks=[TracingCallbackHandler(Tracer([sink]), capture_content=True)]).invoke("olá")
        self.assertIn("olá", sink.spans[0]["attrs"]["prompt"])

    def test_unknown_model_is_flagged_unpriced(self):
        sink = MemorySink()
        PricedFake(model="modelo-x", callbacks=[TracingCallbackHandler(Tracer([sink]))]).invoke("x")
        self.assertFalse(sink.spans[0]["attrs"]["priced"])
        self.assertIsNone(sink.spans[0]["attrs"]["cost_brl"])

    @patch("src.agent_rag._search_news", return_value={"resultados": [{"n": 1, "titulo": "t"}]})
    @patch("src.agent_rag._get_stock_quote", return_value={"symbol": "VALE3", "price": 67.9})
    def test_agent_run_produces_root_llm_and_tool_spans(self, quote, news):
        sink = MemorySink()
        configure([sink])
        try:
            agent = agent_rag.build_agent(FakeToolModel(script=ask_tools_then_answer()))
            agent_rag.run("Notícias e cotação da Vale?", agent=agent, log=lambda *_: None)
        finally:
            configure([])
        kinds = [s["kind"] for s in sink.spans]
        self.assertEqual(kinds.count("llm"), 2)
        self.assertEqual(sorted(s["name"] for s in sink.spans if s["kind"] == "tool"), ["tool:get_stock_quote", "tool:search_news"])
        root = next(s for s in sink.spans if s["name"] == "agent_rag.run")
        self.assertEqual(len({s["trace_id"] for s in sink.spans}), 1)
        self.assertTrue(all(s["parent_id"] == root["span_id"] for s in sink.spans if s["kind"] in ("llm", "tool")))


class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.cache = SemanticCache(HashEmbeddings(), threshold=0.9, ttl_s=100, clock=self.clock)

    def test_exact_hit_ignores_case_and_punctuation(self):
        self.cache.put("Qual a cotação da VALE3?", "R$ 67,90", guard="VALE3")
        hit = self.cache.get("qual a cotação da vale3", guard="VALE3")
        self.assertEqual((hit.kind, hit.answer), ("exact", "R$ 67,90"))

    def test_semantic_hit_for_reordered_words(self):
        self.cache.put("notícias recentes sobre a Vale e cotação", "resposta A", guard="VALE3")
        hit = self.cache.get("cotação e notícias recentes sobre a Vale", guard="VALE3")
        self.assertEqual(hit.kind, "semantic")
        self.assertGreater(hit.similarity, 0.9)

    def test_guard_blocks_similar_question_about_another_ticker(self):
        loose = SemanticCache(HashEmbeddings(), threshold=0.5, clock=self.clock)
        loose.put("qual a cotação atual da VALE3 hoje", "R$ 67,90", guard="VALE3")
        self.assertIsNone(loose.get("qual a cotação atual da PETR4 hoje", guard="PETR4"))
        self.assertEqual(loose.stats["guard_blocked"], 1)
        self.assertIsNotNone(loose.get("qual a cotação atual da VALE3 agora", guard="VALE3"))

    def test_ttl_expiry(self):
        self.cache.put("pergunta", "resposta", guard="")
        self.clock.t += 101
        self.assertIsNone(self.cache.get("pergunta"))
        self.assertEqual(self.cache.stats["evicted"], 1)

    def test_namespace_isolation_and_hit_rate(self):
        self.cache.put("pergunta", "r1", namespace="modelo-a")
        self.assertIsNone(self.cache.get("pergunta", namespace="modelo-b"))
        self.assertIsNotNone(self.cache.get("pergunta", namespace="modelo-a"))
        self.assertAlmostEqual(self.cache.hit_rate(), 0.5)

    def test_max_entries_evicts_oldest(self):
        small = SemanticCache(HashEmbeddings(), max_entries=2, ttl_s=1e9, clock=self.clock)
        for i, q in enumerate(["alfa", "beta", "gama"]):
            self.clock.t += 1
            small.put(q, str(i))
        self.assertIsNone(small.get("alfa"))
        self.assertIsNotNone(small.get("gama"))

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            SemanticCache(HashEmbeddings(), path, clock=self.clock).put("pergunta persistida", "ok")
            again = SemanticCache(HashEmbeddings(), path, clock=self.clock)
            self.assertEqual(again.get("pergunta persistida").answer, "ok")

    def test_question_guard_collects_tickers_company_names_and_window(self):
        self.assertEqual(question_guard("Como está a cotação da vale3?"), "VALE3")
        self.assertEqual(question_guard("notícias da Petrobras e da VALE3 nos últimos 7 dias"), "PETR4|VALE3#7d")
        self.assertNotEqual(question_guard("cotação da VALE3"), question_guard("cotação da PETR4"))

    def test_agent_uses_cache_and_skips_the_model_on_hit(self):
        c = SemanticCache(HashEmbeddings(), threshold=0.9)
        with patch("src.agent_rag._search_news", return_value={"resultados": []}), \
                patch("src.agent_rag._get_stock_quote", return_value={"price": 1}):
            model = FakeToolModel(script=ask_tools_then_answer())
            agent = agent_rag.build_agent(model)
            first, u1 = agent_rag.run("Notícias e cotação da Vale?", agent=agent, log=lambda *_: None, cache=c)
            calls_after_first = model.i
            second, u2 = agent_rag.run("Notícias e cotação da Vale?", agent=agent, log=lambda *_: None, cache=c)
        self.assertEqual(first, second)
        self.assertEqual(model.i, calls_after_first)  # nenhuma chamada nova ao modelo
        self.assertEqual(u2["cache"], "exact")
        self.assertEqual(u2["model_calls"], 0)

    def test_interrupted_answers_are_not_cached(self):
        from langchain_core.messages import AIMessage as M
        looping = [M(content="", tool_calls=[{"name": "get_stock_quote", "args": {"ticker": "VALE3"}, "id": f"c{n}", "type": "tool_call"}]) for n in range(40)]
        c = SemanticCache(HashEmbeddings())
        with patch("src.agent_rag._get_stock_quote", return_value={"price": 1}):
            agent_rag.run("laço infinito", agent=agent_rag.build_agent(FakeToolModel(script=looping)), log=lambda *_: None, cache=c)
        self.assertEqual(c.store.entries, [])


class JsonLLM:
    """LLM simulado mínimo para o _ask: devolve, em ordem, os textos do roteiro."""

    model = "fake-json"

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), 0

    def invoke(self, messages, **kw):
        self.calls += 1
        return AIMessage(content=self.texts[min(self.calls - 1, len(self.texts) - 1)])


class TestExactCacheWithAsk(unittest.TestCase):
    GOOD = json.dumps({"weights": {"A": 0.5, "B": 0.5}, "rationale": "ok"})

    def test_second_identical_call_does_not_hit_the_llm(self):
        cache, llm = ExactCache(), JsonLLM(self.GOOD)
        a = _ask(llm, "sys", "user", ManagerOutput, cache=cache)
        b = _ask(llm, "sys", "user", ManagerOutput, cache=cache)
        self.assertEqual(a, b)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(cache.stats, {"hit": 1, "miss": 1})

    def test_invalid_output_is_never_cached(self):
        cache, llm = ExactCache(), JsonLLM("isto não é JSON")
        with self.assertRaises(BrainError):
            _ask(llm, "sys", "user", ManagerOutput, cache=cache)
        self.assertEqual(cache.store.entries, [])

    def test_business_rule_gate_blocks_caching(self):
        cache, llm = ExactCache(), JsonLLM(self.GOOD)
        _ask(llm, "sys", "user", ManagerOutput, cache=cache, accept=lambda out: False)
        self.assertEqual(cache.store.entries, [])

    def test_different_model_or_prompt_is_a_different_key(self):
        k = ExactCache.key
        self.assertNotEqual(k("m1", "s", "u"), k("m2", "s", "u"))
        self.assertNotEqual(k("m1", "s", "u"), k("m1", "s", "u2"))

    def test_ttl(self):
        clock = FakeClock()
        cache = ExactCache(ttl_s=10, clock=clock)
        cache.put("k", "v")
        self.assertEqual(cache.get("k"), "v")
        clock.t += 11
        self.assertIsNone(cache.get("k"))

    def test_llmbrain_with_cache_skips_repeated_analyst_calls(self):
        chat = golden.ScriptedChat("good")
        brain = LLMBrain(chat, chat, cache=ExactCache())
        series = golden.make_market("normal")
        from src.committee import metrics
        from src.committee.config import PROFILES
        from src.committee.market import align

        _, prices, names = align(series)
        dossiers = {t: {"metricas": metrics.asset_summary(prices[:, j]), "preco_lucro": 8.0} for j, t in enumerate(names)}
        brain.data_analyst(dossiers, PROFILES["moderado"])
        before = chat.calls
        brain.data_analyst(dossiers, PROFILES["moderado"])
        self.assertEqual(chat.calls, before)


class TestEvaluation(unittest.TestCase):
    def test_grounded_answer_passes(self):
        tools = [json.dumps({"price": 67.9, "change_pct": 1.234}), json.dumps({"resultados": [{"n": 1}, {"n": 2}]})]
        checks = {c.name: c for c in evaluate.check_grounding("A Vale fechou a R$ 67,90 (+1,23%) [1][2], em 12/10/2026.", tools, n_sources=2)}
        self.assertTrue(all(c.ok for c in checks.values()), [(c.name, c.detail) for c in checks.values() if not c.ok])

    def test_invented_number_is_flagged(self):
        checks = {c.name: c for c in evaluate.check_grounding("A Vale fechou a R$ 71,25.", [json.dumps({"price": 67.9})], 0)}
        self.assertFalse(checks["numeros_ancorados"].ok)
        self.assertIn("71.25", checks["numeros_ancorados"].detail)

    def test_thousands_separator_and_fraction_as_percent(self):
        tools = [json.dumps({"market_cap": 1234567.5, "dy": 0.1234})]
        checks = {c.name: c for c in evaluate.check_grounding("Valor de mercado de 1.234.567,5 e dividend yield de 12,34%.", tools, 0)}
        self.assertTrue(checks["numeros_ancorados"].ok, checks["numeros_ancorados"].detail)

    def test_invalid_citation_and_advice_are_flagged(self):
        checks = {c.name: c for c in evaluate.check_grounding("Recomendo comprar [3].", [], n_sources=2)}
        self.assertFalse(checks["citacoes_validas"].ok)
        self.assertFalse(checks["sem_recomendacao"].ok)

    def test_judge_parses_scores_and_retries_invalid_json(self):
        good = json.dumps({"faithfulness": 4, "relevance": 5, "advice_free": True, "comment": "ok"})
        judge = JsonLLM("nota: ótimo", good)
        score = evaluate.judge_answer(judge, "pergunta", "resposta", "contexto")
        self.assertEqual((score.faithfulness, score.relevance, score.advice_free), (4, 5, True))
        self.assertEqual(judge.calls, 2)
        with self.assertRaises(BrainError):
            evaluate.judge_answer(JsonLLM(json.dumps({"faithfulness": 9, "relevance": 1, "advice_free": True})), "q", "a", "c")

    def test_committee_checks_catch_a_tampered_report(self):
        final = evaluate.run_scenario(golden.Scenario("crash", "conservador", "rules"))[1]
        self.assertTrue(all(c.ok for c in evaluate.check_committee(final, "conservador") if c.severity == "error"))
        vetoed = final["risk"]["vetoes"][0]
        for row in final["report"]["carteira"]:
            if row["ticker"] == vetoed:
                row["peso"] = 0.2
        bad = {c.name for c in evaluate.check_committee(final, "conservador") if not c.ok and c.severity == "error"}
        self.assertIn("vetos_respeitados", bad)

    def test_veto_is_never_undone_to_reach_min_positions(self):
        """Regressão da Fase 3: com vetos demais, o veto prevalece e o relatório avisa que a diversificação caiu."""
        final = evaluate.run_scenario(golden.Scenario("normal", "conservador", "rules"))[1]
        weights = {r["ticker"]: r["peso"] for r in final["report"]["carteira"]}
        self.assertGreaterEqual(len(final["risk"]["vetoes"]), 2)
        self.assertTrue(all(weights[t] == 0 for t in final["risk"]["vetoes"]))
        self.assertTrue(any("diversificação reduzida" in n for n in final["report"]["notas"]))
        self.assertEqual(final["report"]["aviso"], DISCLAIMER)

    def test_scripted_manager_modes(self):
        flaky = evaluate.run_scenario(golden.Scenario("normal", "arrojado", "llm_flaky"))[1]
        bad = evaluate.run_scenario(golden.Scenario("normal", "arrojado", "llm_bad"))[1]
        self.assertEqual(flaky["attempts"], 2)
        self.assertEqual(bad["attempts"], 3)
        self.assertTrue(any("fallback" in n for n in bad["report"]["notas"]))

    def test_full_regression_gate_passes(self):
        summary = evaluate.run_regression(log=lambda *_: None)
        self.assertEqual(summary["total"], 36)
        self.assertEqual(summary["failed"], [])


class TestReport(unittest.TestCase):
    def _spans(self):
        sink = MemorySink()
        tr = Tracer([sink])
        with tr.span("raiz", "chain"):
            PricedFake(callbacks=[TracingCallbackHandler(tr)]).invoke("a")
            PricedFake(model="modelo-x", callbacks=[TracingCallbackHandler(tr)]).invoke("b")
            with tr.span("cache", "cache") as s:
                s.set(cache="exact")
            with tr.span("cache", "cache") as s:
                s.set(cache="miss")
        return sink.spans

    def test_summary_numbers(self):
        s = report.summarize(self._spans())
        self.assertEqual(s["llm_calls"], 2)
        self.assertEqual(s["traces"], 1)
        self.assertEqual(s["cache"], {"hit": 1, "miss": 1})
        self.assertAlmostEqual(s["cost_brl"], pricing.cost_brl("claude-haiku-4-5", 1000, 500), places=5)  # o modelo sem preço não soma
        self.assertEqual(sum(r["unpriced"] for r in s["rows"]), 1)

    def test_format_mentions_unpriced_and_budget(self):
        text = report.format_report(self._spans())
        self.assertIn("sem preço", text)
        self.assertIn("orçamento do mês", text)
        self.assertIn("taxa de acerto 50%", text)

    def test_empty(self):
        self.assertIn("Nenhum span", report.format_report([]))

    def test_percentile(self):
        self.assertEqual(report.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(report.percentile([], 95), 0.0)


if __name__ == "__main__":
    unittest.main()
