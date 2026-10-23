"""Estudo de quantização int8 dinâmica: QUAIS camadas aguentam ser quantizadas sem destruir o classificador?

A quantização ingênua de todas as camadas lineares derrubou o F1 macro do modelo mesclado (ver evaluate_models --quant). Aqui
quantizamos subconjuntos (atenção, MLP, partes do MLP) e medimos qualidade, tamanho e latência no mesmo teste.

Uso:  python -m src.finetune.quant_study            (precisa do adaptador; ~8 min em CPU)
"""

import copy
import json
import sys
import time

import numpy as np
import torch

from src.finetune import export, labeling, llm, metrics
from src.finetune.labeling import FINETUNE_DIR, LABELS
from src.finetune.train_lora import SPLIT_SEED, make_split

VARIANTS = {
    "fp32 (sem quantizar)": [],
    "só atenção (q,k,v,o)": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "só MLP gate+up": ["gate_proj", "up_proj"],
    "só MLP down": ["down_proj"],
    "MLP inteiro": ["gate_proj", "up_proj", "down_proj"],
    "tudo (ingênuo)": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def quantize_selected(model, kinds: list[str]):
    """Quantização int8 dinâmica só dos módulos cujo nome termina em um dos `kinds` (in-place)."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not kinds:
        return model
    names = {n for n, m in core.model.named_modules() if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in kinds}
    torch.ao.quantization.quantize_dynamic(core.model, names, dtype=torch.qint8, inplace=True)
    return model


def run(log=print) -> dict:
    items = labeling.load_labeled()
    te = [items[i] for i in make_split(items, SPLIT_SEED)["test"]]
    y = [x["label"] for x in te]
    model, tok = llm.load(adapter=llm.ADAPTER_DIR)
    merged = model.merge_and_unload()
    out = {}
    for name, kinds in VARIANTS.items():
        m = quantize_selected(copy.deepcopy(merged), kinds)
        t0 = time.perf_counter()
        pred, _ = llm.predict_items(m, tok, te)
        total = time.perf_counter() - t0
        s = metrics.summary(y, pred)
        out[name] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"], "size_mb": export.model_size_mb(m), "s_per_item_batch8": total / len(te),
                     "latency_s": llm.time_per_item(m, tok, te, n=8)}
        log(f"{name:24s} acc {s['accuracy']:.3f} | F1 macro {s['macro_f1']:.3f} | {out[name]['size_mb']:.0f} MB | {out[name]['latency_s']:.2f} s/notícia")
        del m
    (FINETUNE_DIR / "quant_study.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
