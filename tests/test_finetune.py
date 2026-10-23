import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import helpers  # noqa: F401  (ajusta o sys.path)
from src.finetune import baselines, data, labeling, metrics
from src.finetune.labeling import LABELS


def item(title, text="", label="neutro"):
    return {"id": title[:20], "title": title, "text": text or title + " detalhes da notícia financeira.", "label": label, "source": "x", "published": "2026-10-10"}


class TestLabeling(unittest.TestCase):
    def test_sets_are_disjoint_and_indices_in_range(self):
        self.assertFalse(labeling.POSITIVE & labeling.NEGATIVE)
        self.assertLess(max(labeling.POSITIVE | labeling.NEGATIVE), 536)
        self.assertEqual((labeling.label_of(0), labeling.label_of(3), labeling.label_of(4)), ("positivo", "negativo", "neutro"))

    def test_write_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            news = Path(d) / "news.jsonl"
            arts = [{"id": f"a{i}", "title": f"t{i}", "text": f"texto {i}", "source": "s", "published": "2026-01-01"} for i in range(536)]
            news.write_text("\n".join(json.dumps(a) for a in arts), encoding="utf-8")
            labels = Path(d) / "labels.jsonl"
            self.assertEqual(labeling.write_labels(labels, news), 536)
            items = labeling.load_labeled(labels, news)
            self.assertEqual(len(items), 536)
            self.assertEqual(items[0]["label"], "positivo")
            self.assertEqual(sum(x["label"] == "negativo" for x in items), len(labeling.NEGATIVE))


class TestData(unittest.TestCase):
    def test_make_text_drops_title_repeated_in_body_and_caps_length(self):
        it = item("Ibovespa bate recorde", text="Ibovespa bate recorde. O índice fechou em alta de 1,38%." + " x" * 500)
        t = data.make_text(it, 100)
        self.assertLessEqual(len(t), 100)
        self.assertEqual(t.count("Ibovespa bate recorde"), 1)

    def test_near_duplicates_are_grouped_and_different_stories_are_not(self):
        items = [item("Ibovespa bate recorde e fecha em alta de 1,38% aos 209 mil pontos"),
                 item("Ibovespa fecha em alta de 1,38% e bate novo recorde aos 209 mil pontos"),
                 item("Petrobras anuncia dividendos extraordinários para os acionistas em dezembro"),
                 item("Dólar cai abaixo de R$ 5 com otimismo eleitoral e fluxo estrangeiro")]
        g = data.group_near_duplicates(items)
        self.assertEqual(g[0], g[1])
        self.assertEqual(len({g[0], g[2], g[3]}), 3)

    def test_split_has_no_leakage_is_deterministic_and_respects_sizes(self):
        rng = np.random.default_rng(0)
        items = [item(f"t{i}", label=str(rng.choice(LABELS, p=[0.15, 0.7, 0.15]))) for i in range(200)]
        groups = [i // 2 for i in range(200)]  # pares de quase-duplicatas
        sp = data.split_by_group(items, groups, seed=1)
        self.assertEqual(sp, data.split_by_group(items, groups, seed=1))
        self.assertEqual(sorted(sum(sp.values(), [])), list(range(200)))
        self.assertEqual(data.overlap_report(items, groups, sp)["leaking_groups"], 0)
        self.assertAlmostEqual(len(sp["train"]) / 200, 0.6, delta=0.08)
        self.assertGreaterEqual(min(len(sp["val"]), len(sp["test"])), 30)

    def test_class_weights_favor_rare_classes_and_average_one(self):
        w = data.class_weights(["neutro"] * 70 + ["positivo"] * 20 + ["negativo"] * 10)
        self.assertAlmostEqual(float(w.mean()), 1.0)
        self.assertGreater(w[LABELS.index("negativo")], w[LABELS.index("neutro")])


class TestMetrics(unittest.TestCase):
    def test_known_confusion_and_scores(self):
        y = ["positivo", "positivo", "negativo", "neutro", "neutro", "neutro"]
        p = ["positivo", "neutro", "negativo", "neutro", "neutro", "positivo"]
        s = metrics.summary(y, p)
        self.assertAlmostEqual(s["accuracy"], 4 / 6)
        self.assertEqual(s["confusion"][LABELS.index("negativo")][LABELS.index("negativo")], 1)
        self.assertAlmostEqual(s["per_class"]["negativo"]["f1"], 1.0)
        self.assertAlmostEqual(s["per_class"]["positivo"]["precision"], 0.5)
        self.assertAlmostEqual(s["per_class"]["positivo"]["recall"], 0.5)

    def test_bootstrap_interval_contains_estimate_and_shrinks_with_more_data(self):
        rng = np.random.default_rng(0)
        y = list(rng.choice(LABELS, 60))
        p = [t if rng.random() < 0.7 else str(rng.choice(LABELS)) for t in y]
        lo, hi = metrics.bootstrap_ci(y, p, n=300)
        self.assertLessEqual(lo, metrics.macro_f1(y, p))
        self.assertGreaterEqual(hi, metrics.macro_f1(y, p))
        y2, p2 = y * 5, p * 5
        lo2, hi2 = metrics.bootstrap_ci(y2, p2, n=300)
        self.assertLess(hi2 - lo2, hi - lo)

    def test_paired_bootstrap_detects_a_clearly_better_system(self):
        rng = np.random.default_rng(1)
        y = list(rng.choice(LABELS, 120))
        good = [t if rng.random() < 0.9 else str(rng.choice(LABELS)) for t in y]
        bad = [str(rng.choice(LABELS)) for _ in y]
        r = metrics.paired_bootstrap_diff(y, good, bad, n=300)
        self.assertGreater(r["mean"], 0.3)
        self.assertGreater(r["p_a_better"], 0.99)
        self.assertGreater(r["ci"][0], 0)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        pos = ["Ibovespa dispara e bate recorde com lucro acima do esperado", "Ações sobem com dividendos e recompra anunciados", "Bolsa avança com otimismo e ganhos fortes"]
        neg = ["Ações despencam após prejuízo e rebaixamento", "Bolsa tomba com queda forte e risco de recessão", "Empresa pede recuperação judicial e perde valor"]
        neu = ["Agenda da semana traz reuniões e indicadores", "Candidato participa de debate na televisão", "Loteria sorteia concurso acumulado nesta noite"]
        self.train = [item(t + f" caso {i}", label="positivo") for i, t in enumerate(pos * 6)] + [item(t + f" caso {i}", label="negativo") for i, t in enumerate(neg * 6)] + \
                     [item(t + f" caso {i}", label="neutro") for i, t in enumerate(neu * 10)]

    def test_majority(self):
        self.assertEqual(baselines.MajorityBaseline().fit(self.train).predict([item("qualquer")]), ["neutro"])

    def test_lexicon_on_obvious_cases(self):
        lex = baselines.LexiconBaseline().fit(self.train)
        self.assertEqual(lex.predict([item("Bolsa dispara com lucro recorde")]), ["positivo"])
        self.assertEqual(lex.predict([item("Ações despencam após prejuízo")]), ["negativo"])
        self.assertEqual(lex.predict([item("Debate na televisão mobiliza candidatos")]), ["neutro"])

    def test_tfidf_learns_the_training_signal(self):
        m = baselines.TfidfLR().fit(self.train, self.train[:30])
        pred = m.predict([item("Ibovespa dispara com lucro acima do esperado e dividendos"), item("Ações despencam com prejuízo e recessão")])
        self.assertEqual(pred, ["positivo", "negativo"])
        p = m.predict_proba([item("Ibovespa dispara")])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


def _tiny_model_and_tokenizer():
    """Modelo Qwen2 minúsculo (pesos aleatórios) + o tokenizer real do Qwen, só se já estiver no cache local; senão, pula."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from src.finetune import llm

        tok = AutoTokenizer.from_pretrained(llm.BASE_MODEL, local_files_only=True)
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"tokenizer do Qwen não está no cache local: {exc}")
    import torch

    cfg = AutoConfig.from_pretrained(llm.BASE_MODEL, local_files_only=True)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 32, 64, 2
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.max_position_embeddings = 2, 1, 1024
    cfg.layer_types = cfg.layer_types[:2] if getattr(cfg, "layer_types", None) else None
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    tok.padding_side = "right"
    return model.eval(), tok


class TestSlmPipelineWithTinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = _tiny_model_and_tokenizer()

    def test_label_tokens_are_distinct(self):
        from src.finetune import llm

        self.assertEqual(len(set(llm.label_token_ids(self.tok))), 3)

    def test_restricted_logits_equal_full_logits_at_the_last_position(self):
        import torch

        from src.finetune import llm

        ids = llm.label_token_ids(self.tok)
        enc, last = llm.encode_batch(self.tok, ["Ibovespa bate recorde", "Dólar cai com cautela eleitoral e petróleo em alta"])
        with torch.no_grad():
            fast = llm.label_logits(self.model, enc, last, ids)
            full = self.model(**enc).logits[torch.arange(len(last)), last][:, ids]
        torch.testing.assert_close(fast, full, rtol=1e-4, atol=1e-5)

    def test_scores_are_probabilities_and_padding_does_not_change_them(self):
        from src.finetune import llm

        p_batch = llm.score_texts(self.model, self.tok, ["curto", "um texto bem mais longo sobre o mercado financeiro brasileiro hoje"], batch_size=2)
        p_single = llm.score_texts(self.model, self.tok, ["curto"], batch_size=1)
        np.testing.assert_allclose(p_batch.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(p_batch[0], p_single[0], atol=1e-4)

    def test_long_text_is_truncated_to_the_token_budget(self):
        from src.finetune import llm

        enc, _ = llm.encode_batch(self.tok, ["palavra " * 2000], max_len=120)
        self.assertLessEqual(enc["input_ids"].shape[1], 140)

    def test_lora_training_reduces_the_loss_and_the_adapter_reloads(self):
        import copy

        import torch

        from src.finetune import llm, train_lora

        items = [item("Ibovespa dispara com lucro recorde", label="positivo"), item("Ações despencam após prejuízo", label="negativo"),
                 item("Agenda da semana traz indicadores", label="neutro")] * 4
        cfg = train_lora.TrainConfig(epochs=6, lr=5e-3, micro_batch=4, accum=1, rank=4, alpha=8, dropout=0.0, warmup=0.0, targets=["q_proj", "v_proj"])
        model = train_lora.apply_lora(copy.deepcopy(self.model), cfg)
        tr, tot = train_lora.trainable_params(model)
        self.assertLess(tr / tot, 0.2)
        with tempfile.TemporaryDirectory() as d:
            hist = train_lora.train(model, self.tok, items, items[:3], cfg, Path(d), log=lambda *_: None)
            losses = [s["loss"] for s in hist["steps"]]
            self.assertLess(np.mean(losses[-3:]), np.mean(losses[:3]))
            self.assertTrue((Path(d) / "adapter_config.json").exists())
            self.assertTrue((Path(d) / "train_log.json").exists())
            from peft import PeftModel

            reloaded = PeftModel.from_pretrained(copy.deepcopy(self.model), d).eval()
            p1 = llm.score_texts(model.eval(), self.tok, [items[0]["title"]])
            p2 = llm.score_texts(reloaded, self.tok, [items[0]["title"]])
            self.assertEqual(int(p1.argmax()), int(p2.argmax()))

    def test_int8_quantization_keeps_the_pipeline_working(self):
        import copy

        from src.finetune import export, llm

        q = export.quantize_int8(copy.deepcopy(self.model))
        p = llm.score_texts(q, self.tok, ["Ibovespa bate recorde"])
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
        self.assertLess(export.model_size_mb(q), export.model_size_mb(self.model))


class TestFewShotAndSelectiveQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = _tiny_model_and_tokenizer()

    def test_shots_come_from_the_given_set_are_balanced_and_deterministic(self):
        from src.finetune import llm

        train = [item(f"{lab} caso {i} sobre mercado", label=lab) for lab in LABELS for i in range(10)]
        shots = llm.pick_shots(train, per_class=2, seed=3)
        self.assertEqual(shots, llm.pick_shots(train, per_class=2, seed=3))
        self.assertEqual(sorted(lab for _, lab in shots), sorted(LABELS * 2))
        texts = {data.make_text(x, 230) for x in train}
        self.assertTrue(all(t in texts for t, _ in shots))

    def test_fewshot_prompt_has_one_turn_pair_per_shot_and_ends_at_the_assistant_turn(self):
        from src.finetune import llm

        shots = [("Bolsa dispara", "positivo"), ("Ações despencam", "negativo")]
        p = llm.build_prompt_fewshot(self.tok, "Agenda da semana", shots)
        self.assertEqual(p.count("<|im_start|>user"), 3)
        self.assertEqual(p.count("<|im_start|>assistant"), 3)  # 2 respostas dos exemplos + o turno aberto
        self.assertTrue(p.rstrip().endswith("<|im_start|>assistant"))
        probs = llm.score_texts(self.model, self.tok, ["Agenda da semana"], 1, lambda tk, t: llm.build_prompt_fewshot(tk, t, shots), 600)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_selective_quantization_only_touches_the_requested_layers(self):
        import copy

        import torch

        from src.finetune import quant_study

        m = quant_study.quantize_selected(copy.deepcopy(self.model), ["down_proj"])
        core = m.model
        self.assertNotIsInstance(core.layers[0].mlp.down_proj, torch.nn.Linear)
        self.assertIsInstance(core.layers[0].mlp.up_proj, torch.nn.Linear)
        self.assertIsInstance(core.layers[0].self_attn.q_proj, torch.nn.Linear)
        self.assertIs(quant_study.quantize_selected(copy.deepcopy(self.model), []).model.layers[0].mlp.down_proj.__class__, torch.nn.Linear)


class TestModelfile(unittest.TestCase):
    def test_modelfile_points_to_the_directory_and_keeps_the_system_prompt(self):
        from src.finetune import export, llm

        text = export.modelfile_text(Path("models") / "merged")
        self.assertIn("FROM models/merged", text)
        self.assertIn(llm.SYSTEM[:40], text)
        self.assertIn("temperature 0", text)


class TestSentimentTool(unittest.TestCase):
    def test_lexicon_backend_without_models(self):
        from src.tools.sentiment import SentimentClassifier

        out = SentimentClassifier("lexicon").predict(["Bolsa dispara e bate recorde com lucro acima do esperado", "Ações despencam após prejuízo e rebaixamento"])
        self.assertEqual([o["label"] for o in out], ["positivo", "negativo"])

    def test_auto_falls_back_when_there_is_no_adapter(self):
        from src.tools.sentiment import SentimentClassifier

        with tempfile.TemporaryDirectory() as d:
            self.assertIn(SentimentClassifier("auto", adapter_dir=Path(d)).backend, ("tfidf", "lexicon"))

    def test_calls_are_traced_as_local_zero_cost_model_spans(self):
        from src.observability.tracing import MemorySink, configure
        from src.tools.sentiment import SentimentClassifier

        sink = MemorySink()
        configure([sink])
        try:
            SentimentClassifier("lexicon").predict(["Bolsa sobe"])
        finally:
            configure([])
        (s,) = sink.spans
        self.assertEqual((s["kind"], s["attrs"]["local"], s["attrs"]["cost_brl"]), ("model", True, 0.0))

    def test_committee_news_gets_a_tone_per_headline(self):
        from unittest.mock import patch

        from src.committee.run import make_news_fn

        fake = {"resultados": [{"titulo": "Bolsa dispara", "fonte": "x", "data": "2026-10-10"}, {"titulo": "Ações despencam", "fonte": "y", "data": "2026-10-10"}]}
        with patch("src.tools.news_search.search_news", return_value=fake):
            fn = make_news_fn(True, sentiment_fn=lambda texts: ["positivo" if "dispara" in t else "negativo" for t in texts])
            items = fn("VALE3")
        self.assertEqual([i["tom"] for i in items], ["positivo", "negativo"])


class TestRobustnessHelpers(unittest.TestCase):
    def test_flip_rate_counts_only_eligible_items(self):
        from src.finetune.robustness import flip_rate

        y = ["neutro", "negativo", "positivo", "neutro"]
        base = ["neutro", "negativo", "positivo", "positivo"]      # o 4º já era previsto "positivo": não é elegível
        att = ["positivo", "negativo", "positivo", "positivo"]    # o 1º virou
        r = flip_rate(base, att, y, "positivo")
        self.assertEqual((r["eligible"], r["flipped"]), (2, 1))  # elegíveis: 1º e 2º (o 3º tem rótulo-alvo)
        self.assertAlmostEqual(r["rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
