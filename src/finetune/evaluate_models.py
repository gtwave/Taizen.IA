"""Compara TODOS os sistemas no mesmo teste: linhas de base, SLM zero-shot, SLM com LoRA (e a versão quantizada).

Regras de honestidade da avaliação:
  * o conjunto de TESTE só é usado aqui, uma vez por sistema; hiperparâmetros foram escolhidos na VALIDAÇÃO;
  * toda métrica vem com intervalo de confiança por bootstrap (o teste é pequeno) e as comparações são pareadas;
  * rótulos são "prata" (um anotador): os números medem concordância com ESSE anotador, não verdade absoluta.

Uso:  python -m src.finetune.evaluate_models [--no-lora] [--quant]      (salva data/finetune/results.json)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.finetune import baselines, data, labeling, llm, metrics
from src.finetune.labeling import FINETUNE_DIR, LABELS
from src.finetune.train_lora import SPLIT_SEED, make_split

RESULTS = FINETUNE_DIR / "results.json"
PROBS_DIR = FINETUNE_DIR / "probs"


def describe(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    s = metrics.summary(y_true, y_pred)
    s["name"] = name
    s["macro_f1_ci"] = metrics.bootstrap_ci(y_true, y_pred, metrics.macro_f1)
    s["accuracy_ci"] = metrics.bootstrap_ci(y_true, y_pred, metrics.accuracy)
    return s


def slm_predictions(model, tok, items: list[dict], tag: str, force: bool = False) -> tuple[list[str], np.ndarray]:
    """Previsões do SLM com cache em disco (o treino/inferência em CPU é lento: não repetir à toa)."""
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROBS_DIR / f"{tag}.npy"
    if path.exists() and not force and len(np.load(path)) == len(items):
        probs = np.load(path)
    else:
        _, probs = llm.predict_items(model, tok, items)
        np.save(path, probs)
    return [LABELS[i] for i in probs.argmax(axis=1)], probs


def run(use_lora: bool = True, quant: bool = False, log=print) -> dict:
    items = labeling.load_labeled()
    split = make_split(items, SPLIT_SEED)
    tr, va, te = ([items[i] for i in split[k]] for k in ("train", "val", "test"))
    y_val, y_te = [x["label"] for x in va], [x["label"] for x in te]
    out = {"split": {k: len(v) for k, v in split.items()}, "class_counts_test": {lab: y_te.count(lab) for lab in LABELS}, "systems": {}}
    preds: dict[str, list[str]] = {}

    for B in (baselines.MajorityBaseline(), baselines.LexiconBaseline(), baselines.TfidfLR()):
        B.fit(tr, va)
        preds[B.name] = B.predict(te)
        out["systems"][B.name] = {"test": describe(B.name, y_te, preds[B.name]), "val": describe(B.name, y_val, B.predict(va))}
        log(f"{B.name:10s} teste: acurácia {out['systems'][B.name]['test']['accuracy']:.3f} | F1 macro {out['systems'][B.name]['test']['macro_f1']:.3f}")

    model, tok = llm.load()
    t0 = time.time()
    p_val, _ = slm_predictions(model, tok, va, "zeroshot_val")
    preds["zero_shot"], pz = slm_predictions(model, tok, te, "zeroshot_test")
    out["systems"]["zero_shot"] = {"test": describe("zero_shot", y_te, preds["zero_shot"]), "val": describe("zero_shot", y_val, p_val), "latency_s": llm.time_per_item(model, tok, te)}
    log(f"zero_shot  teste: acurácia {out['systems']['zero_shot']['test']['accuracy']:.3f} | F1 macro {out['systems']['zero_shot']['test']['macro_f1']:.3f} ({time.time() - t0:.0f}s)")

    # degrau intermediário da Regra do Degrau: few-shot (6 exemplos resolvidos no prompt, sorteados no TREINO), sem treinar nada
    shots = llm.pick_shots(tr)
    fs = lambda tk, t: llm.build_prompt_fewshot(tk, t, shots)  # noqa: E731
    path = PROBS_DIR / "fewshot_test.npy"
    if path.exists() and len(np.load(path)) == len(te):
        pf = np.load(path)
    else:
        pf = llm.score_texts(model, tok, [data.make_text(x) for x in te], 4, fs, 1200)
        np.save(path, pf)
    preds["few_shot"] = [LABELS[i] for i in pf.argmax(axis=1)]
    t0 = time.perf_counter()
    llm.score_texts(model, tok, [data.make_text(x) for x in te[:6]], 1, fs, 1200)
    out["systems"]["few_shot"] = {"test": describe("few_shot", y_te, preds["few_shot"]), "latency_s": (time.perf_counter() - t0) / 6, "n_shots": len(shots)}
    log(f"few_shot   teste: acurácia {out['systems']['few_shot']['test']['accuracy']:.3f} | F1 macro {out['systems']['few_shot']['test']['macro_f1']:.3f}")

    if use_lora and llm.ADAPTER_DIR.exists():
        del model
        model, tok = llm.load(adapter=llm.ADAPTER_DIR)
        p_val, _ = slm_predictions(model, tok, va, "lora_val")
        preds["lora"], pl = slm_predictions(model, tok, te, "lora_test")
        out["systems"]["lora"] = {"test": describe("lora", y_te, preds["lora"]), "val": describe("lora", y_val, p_val), "latency_s": llm.time_per_item(model, tok, te)}
        log(f"lora       teste: acurácia {out['systems']['lora']['test']['accuracy']:.3f} | F1 macro {out['systems']['lora']['test']['macro_f1']:.3f}")
        if quant:
            from src.finetune import export

            merged = model.merge_and_unload()
            size_fp32 = export.model_size_mb(merged)  # antes: a quantização é in-place
            q = export.quantize_int8(merged)
            _, pq = llm.predict_items(q, tok, te)
            np.save(PROBS_DIR / "lora_int8_test.npy", pq)
            preds["lora_int8"] = [LABELS[i] for i in pq.argmax(axis=1)]
            out["systems"]["lora_int8"] = {"test": describe("lora_int8", y_te, preds["lora_int8"]), "latency_s": llm.time_per_item(q, tok, te),
                                           "size_mb": export.model_size_mb(q), "size_mb_fp32": size_fp32}
            log(f"lora_int8  teste: acurácia {out['systems']['lora_int8']['test']['accuracy']:.3f} | F1 macro {out['systems']['lora_int8']['test']['macro_f1']:.3f}")

    # comparações pareadas (mesmos exemplos): quem é melhor, com que confiança
    out["paired"] = {}
    for a, b in (("lora", "tfidf_lr"), ("lora", "lexicon"), ("lora", "zero_shot"), ("lora", "few_shot"), ("few_shot", "zero_shot"), ("tfidf_lr", "lexicon"), ("lora_int8", "lora")):
        if a in preds and b in preds:
            out["paired"][f"{a}_vs_{b}"] = metrics.paired_bootstrap_diff(y_te, preds[a], preds[b])
    out["preds_test"] = preds
    out["y_test"] = y_te
    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--quant", action="store_true", help="também avalia a versão int8 do modelo mesclado")
    a = ap.parse_args()
    r = run(not a.no_lora, a.quant)
    print("\ncomparações pareadas (F1 macro, A − B, IC 95%, P(A>B)):")
    for k, v in r["paired"].items():
        print(f"  {k:24s} {v['mean']:+.3f}  [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]  P={v['p_a_better']:.2f}")
