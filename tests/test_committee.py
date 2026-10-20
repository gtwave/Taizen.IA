import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from tests import helpers  # noqa: F401  (ajusta o sys.path)
from src.committee import allocation, metrics
from src.committee.backtest import weights_committee, walk_forward, weights_equal
from src.committee.brains import BrainError, LLMBrain, RulesBrain, _ask, DataAnalystOutput, extract_json
from src.committee.config import PROFILES
from src.committee.graph import Deps, build_graph, validate_portfolio
from src.committee.market import MarketDataError, PriceSeries, fetch_history, parse_history, validate_series
from src.tools.stock_quote import get_stock_quote

TICKERS = ["AAA3", "BBB4", "CCC3", "DDD4"]


def synthetic_series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.arange(n) * 86400 + 1_600_000_000
    out = {}
    for i, t in enumerate(TICKERS):
        daily = rng.normal(0.0004 * (i + 1), 0.010 * (i + 1), n - 1)
        prices = 20 * np.concatenate([[1.0], np.cumprod(1 + daily)])
        out[t] = PriceSeries(t, dates, prices)
    return out


def make_deps(brain):
    return Deps(load_series=lambda tk: ({t: synthetic_series()[t] for t in tk}, {}),
                quote=lambda t: {"price_earnings": 8.0 + len(t), "fifty_two_week_range": "1 - 2"},
                news=lambda t: [], brain=brain)


def run_graph(brain, profile="moderado"):
    return build_graph(make_deps(brain)).invoke({"profile": profile, "universe": TICKERS, "attempts": 0})


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        p = np.array([100.0, 110.0, 99.0, 121.0])
        self.assertAlmostEqual(metrics.max_drawdown(p), 99 / 110 - 1)
        self.assertAlmostEqual(metrics.momentum(p, 2), 121 / 110 - 1)
        self.assertAlmostEqual(metrics.annualized_return(np.array([100.0, 121.0]), days=1), 0.21)
        self.assertEqual(metrics.range_position(np.array([5.0, 10.0, 7.5])), 0.5)

    def test_risk_contributions_sum_to_one_and_diversification(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, (300, 3))
        cov = metrics.covariance_matrix(r)
        w = np.array([0.5, 0.3, 0.2])
        self.assertAlmostEqual(metrics.risk_contributions(w, cov).sum(), 1.0)
        self.assertGreaterEqual(metrics.diversification_ratio(w, cov), 1.0)

    def test_correlation_of_identical_series_is_one(self):
        r = np.random.default_rng(2).normal(0, 0.01, 200)
        c = metrics.correlation_matrix(np.column_stack([r, r]))
        self.assertAlmostEqual(c[0, 1], 1.0)


class TestAllocation(unittest.TestCase):
    def setUp(self):
        self.vols = np.array([0.20, 0.30, 0.25, 0.40])

    def test_sum_caps_and_floors(self):
        for prof in PROFILES.values():
            w = allocation.allocate(self.vols, np.array([1.5, -1.0, 0.0, 0.5]), prof)
            self.assertAlmostEqual(w.sum(), 1.0)
            self.assertTrue((w <= max(prof.max_weight, 1 / 4) + 1e-9).all())
            self.assertTrue(((w == 0) | (w >= prof.min_weight - 1e-9)).all())

    def test_higher_score_gets_more_weight(self):
        vols = np.full(4, 0.3)
        w = allocation.allocate(vols, np.array([1.0, 0.0, 0.0, -1.0]), PROFILES["arrojado"])
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[3])

    def test_veto_is_a_hard_rule_even_when_it_breaks_min_positions(self):
        w = allocation.allocate(self.vols, np.zeros(4), PROFILES["moderado"], np.array([False, False, False, True]))
        self.assertEqual(w[3], 0.0)
        # 3 vetados, mínimo de 3 posições: a carteira concentra no único elegível em vez de desfazer o veto
        w2 = allocation.allocate(self.vols, np.zeros(4), PROFILES["moderado"], np.array([True, True, True, False]))
        self.assertEqual(list(w2 > 0), [False, False, False, True])
        self.assertAlmostEqual(w2.sum(), 1.0)

    def test_all_vetoed_raises(self):
        with self.assertRaises(ValueError):
            allocation.allocate(self.vols, np.zeros(4), PROFILES["moderado"], np.ones(4, dtype=bool))

    def test_infeasible_cap_raises(self):
        with self.assertRaises(ValueError):
            allocation.cap_and_normalize(np.ones(3), 0.2)


class TestMarket(unittest.TestCase):
    PAYLOAD = {"results": [{"historicalDataPrice": [
        {"date": 1_600_000_000 + i * 86400, "close": 10.0 + i, "adjustedClose": 5.0 + i * 0.5} for i in range(40)]}]}

    def test_parse_prefers_adjusted_close(self):
        s = parse_history("XXXX3", self.PAYLOAD)
        self.assertEqual(s.prices[0], 5.0)
        self.assertEqual(len(s.prices), 40)

    def test_parse_fails_loudly_without_data(self):
        with self.assertRaises(MarketDataError):
            parse_history("XXXX3", {"results": []})

    def test_validate_flags_suspicious_jump(self):
        s = PriceSeries("X", np.arange(5), np.array([10.0, 10.1, 20.0, 20.1, 20.2]))
        self.assertTrue(any("acima de 30%" in w for w in validate_series(s)))

    def test_fetch_uses_disk_cache(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            session = MagicMock()
            session.get.return_value = MagicMock(status_code=200, json=lambda: self.PAYLOAD)
            fetch_history("XXXX3", cache_dir=Path(d), session=session)
            fetch_history("XXXX3", cache_dir=Path(d), session=session)
            self.assertEqual(session.get.call_count, 1)  # a segunda leitura veio do cache

    def test_401_mentions_the_token(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            session = MagicMock()
            session.get.return_value = MagicMock(status_code=401)
            with self.assertRaises(MarketDataError) as ctx:
                fetch_history("WEGE3", cache_dir=Path(d), session=session)
            self.assertIn("BRAPI_TOKEN", str(ctx.exception))


class TestStockQuoteErrors(unittest.TestCase):
    @patch("src.tools.stock_quote.requests.get")
    def test_401_becomes_data_not_exception(self, mget):
        mget.return_value = MagicMock(status_code=401)
        out = get_stock_quote("WEGE3")
        self.assertIn("BRAPI_TOKEN", out["error"])

    @patch.dict("os.environ", {"BRAPI_TOKEN": "t0k3n"})
    @patch("src.tools.stock_quote.requests.get")
    def test_token_is_sent_as_header_not_in_url(self, mget):
        mget.return_value = MagicMock(status_code=200, json=lambda: {"results": [{"symbol": "WEGE3"}]})
        get_stock_quote("wege3")
        self.assertEqual(mget.call_args.kwargs["headers"], {"Authorization": "Bearer t0k3n"})
        self.assertNotIn("t0k3n", mget.call_args.args[0])


class TestBrains(unittest.TestCase):
    def test_extract_json_tolerates_fences_and_prose(self):
        self.assertEqual(extract_json('Claro!\n```json\n{"a": {"b": 1}}\n```\nfim'), {"a": {"b": 1}})
        with self.assertRaises(ValueError):
            extract_json("sem json")

    def test_ask_retries_with_the_error_and_then_succeeds(self):
        good = '{"views": [{"ticker": "AAA3", "stance": "neutra", "score_adjust": 0.2, "rationale": "ok"}]}'
        llm = GenericFakeChatModel(messages=iter([AIMessage(content="não sei"), AIMessage(content=good)]))
        out = _ask(llm, "sys", "user", DataAnalystOutput)
        self.assertEqual(out.views[0].ticker, "AAA3")

    def test_ask_rejects_out_of_range_adjustment(self):
        bad = '{"views": [{"ticker": "AAA3", "stance": "neutra", "score_adjust": 1.5, "rationale": "x"}]}'
        llm = GenericFakeChatModel(messages=iter([AIMessage(content=bad), AIMessage(content=bad)]))
        with self.assertRaises(BrainError):
            _ask(llm, "sys", "user", DataAnalystOutput)

    def test_llm_adjustment_is_bounded_and_added_to_the_quantitative_score(self):
        dossiers = {t: {"metricas": {"momento_6m": 0.1 * i, "volatilidade_anual": 0.3}, "preco_lucro": 8.0}
                    for i, t in enumerate(TICKERS)}
        views = [{"ticker": t, "stance": "favoravel", "score_adjust": 1.0, "rationale": "x"} for t in TICKERS]
        llm = GenericFakeChatModel(messages=iter([AIMessage(content=json.dumps({"views": views}))]))
        base = RulesBrain().data_analyst(dossiers, PROFILES["moderado"])
        out = LLMBrain(llm, llm).data_analyst(dossiers, PROFILES["moderado"])
        for t in TICKERS:
            self.assertAlmostEqual(out[t]["score"], min(2.0, base[t]["score"] + 1.0), places=3)


class TestGraph(unittest.TestCase):
    def test_rules_committee_runs_end_to_end(self):
        final = run_graph(RulesBrain())
        rep = final["report"]
        self.assertAlmostEqual(sum(r["peso"] for r in rep["carteira"]), 1.0, places=3)
        nos = [t["no"] for t in final["trace"]]
        for n in ["coletar", "analista_dados", "analista_risco", "gestor", "validador", "relatorio"]:
            self.assertIn(n, nos)
        self.assertEqual(final["attempts"], 1)
        self.assertIn("recomendação", rep["aviso"])

    def test_validator_catches_every_constraint(self):
        prof = PROFILES["moderado"]
        base = {"AAA3": 0.3, "BBB4": 0.3, "CCC3": 0.2, "DDD4": 0.2}
        ok = validate_portfolio(base, TICKERS, prof, [], base)
        self.assertTrue(ok["ok"])
        bad = validate_portfolio({"AAA3": 0.8, "BBB4": 0.4}, TICKERS, prof, ["BBB4"], base)
        text = " ".join(bad["errors"])
        for piece in ["somam", "teto", "vetado", "posições", "linha de base"]:
            self.assertIn(piece, text)

    def test_manager_gets_a_second_chance_after_invalid_proposal(self):
        base_run = run_graph(RulesBrain())
        b = base_run["baseline"]
        invalid = {"weights": {t: 0.5 for t in TICKERS}, "rationale": "ruim"}
        valid = {"weights": b, "rationale": "corrigido"}
        llm = GenericFakeChatModel(messages=iter([AIMessage(content=json.dumps(invalid)), AIMessage(content=json.dumps(valid))]))
        brain = LLMBrain(llm, llm)
        brain.data_analyst = RulesBrain().data_analyst   # isola o teste no gestor
        brain.risk_analyst = RulesBrain().risk_analyst
        final = run_graph(brain)
        self.assertEqual(final["attempts"], 2)
        self.assertTrue(final["validation"]["ok"])
        self.assertEqual(final["report"]["racional_gestor"], "corrigido")

    def test_fallback_to_baseline_when_manager_never_complies(self):
        invalid = AIMessage(content=json.dumps({"weights": {t: 0.9 for t in TICKERS}, "rationale": "x"}))
        llm = GenericFakeChatModel(messages=iter([invalid] * 12))
        brain = LLMBrain(llm, llm)
        brain.data_analyst = RulesBrain().data_analyst
        brain.risk_analyst = RulesBrain().risk_analyst
        final = run_graph(brain)
        self.assertEqual(final["attempts"], 3)
        self.assertTrue(any("fallback" in n for n in final["notes"]))
        self.assertAlmostEqual(sum(r["peso"] for r in final["report"]["carteira"]), 1.0, places=3)


class TestBacktestHasNoLookahead(unittest.TestCase):
    def test_weights_at_t_do_not_depend_on_future_prices(self):
        prices = np.column_stack([synthetic_series()[t].prices for t in TICKERS])
        altered = prices.copy()
        altered[330:] *= 50.0   # muda só o futuro (depois do rebalanceamento em t = 300)
        a = walk_forward(prices, weights_committee, PROFILES["moderado"])
        b = walk_forward(altered, weights_committee, PROFILES["moderado"])
        first_t, wa = a["weights"][0]
        _, wb = b["weights"][0]
        self.assertEqual(first_t, 252)
        np.testing.assert_allclose(wa, wb)
        # todos os rebalanceamentos até t = 329 usam só dados que não foram alterados
        for (ta, w1), (tb, w2) in zip(a["weights"], b["weights"]):
            if ta <= 329:
                np.testing.assert_allclose(w1, w2)

    def test_equal_weight_equity_starts_at_one_and_is_positive(self):
        prices = np.column_stack([synthetic_series()[t].prices for t in TICKERS])
        res = walk_forward(prices, weights_equal, None, cost=0.0)
        eq = res["equity"][~np.isnan(res["equity"])]
        self.assertAlmostEqual(eq[0], 1.0)
        self.assertTrue((eq > 0).all())


if __name__ == "__main__":
    unittest.main()
