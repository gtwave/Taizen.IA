"""Fine-tuning com LoRA, em CPU, de um SLM como classificador de sentimento (laço de treino escrito à mão, ~80 linhas úteis).

LoRA (Low-Rank Adaptation) congela os pesos do modelo e treina, em cada matriz W (d×k) escolhida, só duas matrizes pequenas
A (r×k) e B (d×r): W' = W + (alpha/r)·B·A. Com r=16, treina-se ~1,7% dos parâmetros; o "adaptador" resultante tem dezenas de MB.

A perda é a entropia cruzada, com pesos por classe, sobre os logits dos 3 tokens de rótulo na última posição do prompt — a
mesma quantidade usada na inferência (`llm.label_logits`).

Uso:  python -m src.finetune.train_lora --epochs 3            # ~1 h em CPU de 8 núcleos
      python -m src.finetune.train_lora --limit 24 --epochs 1  # teste rápido
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.finetune import data, labeling, llm, metrics
from src.finetune.labeling import LABELS

SPLIT_SEED = 0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 2e-4
    micro_batch: int = 4
    accum: int = 2
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    warmup: float = 0.06
    weight_decay: float = 0.0
    clip: float = 1.0
    seed: int = 0
    targets: list = field(default_factory=lambda: list(TARGETS))


def apply_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=cfg.targets, task_type="CAUSAL_LM"))


def trainable_params(model) -> tuple[int, int]:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, sum(p.numel() for p in model.parameters())


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warm = max(int(total * cfg.warmup), 1)
    if step < warm:
        return cfg.lr * (step + 1) / warm
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def evaluate(model, tok, items: list[dict], batch_size: int = 8) -> dict:
    model.eval()
    pred, probs = llm.predict_items(model, tok, items, batch_size)
    s = metrics.summary([x["label"] for x in items], pred)
    s["pred"] = pred
    return s


def train(model, tok, train_items: list[dict], val_items: list[dict], cfg: TrainConfig, out_dir: Path, log=print) -> dict:
    """Treina o adaptador (o `model` já deve ter o LoRA aplicado). Guarda em `out_dir` o adaptador da melhor época (F1 macro na validação)."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ids = llm.label_token_ids(tok)
    weights = torch.tensor(data.class_weights([x["label"] for x in train_items]), dtype=torch.float32)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    eff_batch = cfg.micro_batch * cfg.accum
    steps_per_epoch = math.ceil(len(train_items) / eff_batch)
    total = steps_per_epoch * cfg.epochs
    hist = {"config": asdict(cfg), "n_train": len(train_items), "n_val": len(val_items), "steps": [], "epochs": [], "best_epoch": None}
    best_f1, step, t_start = -1.0, 0, time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = list(range(len(train_items)))
        random.Random(cfg.seed + epoch).shuffle(order)
        for b0 in range(0, len(order), eff_batch):
            batch = [train_items[i] for i in order[b0:b0 + eff_batch]]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            loss_sum = 0.0
            for m0 in range(0, len(batch), cfg.micro_batch):
                micro = batch[m0:m0 + cfg.micro_batch]
                enc, last = llm.encode_batch(tok, [data.make_text(x) for x in micro])
                logits = llm.label_logits(model, enc, last, ids)
                y = torch.tensor([LABELS.index(x["label"]) for x in micro])
                loss = torch.nn.functional.cross_entropy(logits, y, weight=weights, reduction="sum") / len(batch)
                loss.backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, cfg.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            hist["steps"].append({"step": step, "loss": loss_sum, "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.perf_counter() - t_start, 1)})
            if step % 5 == 0 or step == total:
                log(f"época {epoch}/{cfg.epochs} passo {step}/{total} perda {loss_sum:.4f} lr {opt.param_groups[0]['lr']:.2e} ({time.perf_counter() - t_start:.0f}s)")
        ev = evaluate(model, tok, val_items)
        hist["epochs"].append({"epoch": epoch, "val_accuracy": ev["accuracy"], "val_macro_f1": ev["macro_f1"], "elapsed_s": round(time.perf_counter() - t_start, 1)})
        log(f"== época {epoch}: validação acurácia {ev['accuracy']:.3f} | F1 macro {ev['macro_f1']:.3f}")
        if ev["macro_f1"] > best_f1:
            best_f1, hist["best_epoch"] = ev["macro_f1"], epoch
            model.save_pretrained(str(out_dir))
            log(f"   melhor até agora: adaptador salvo em {out_dir}")
    hist["train_seconds"] = round(time.perf_counter() - t_start, 1)
    hist["best_val_macro_f1"] = best_f1
    (out_dir / "train_log.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    return hist


def make_split(items: list[dict], seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    return data.split_by_group(items, data.group_near_duplicates(items), seed=seed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="usa só N exemplos de treino/validação (teste rápido)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", default=str(llm.ADAPTER_DIR))
    a = ap.parse_args()
    items = labeling.load_labeled()
    split = make_split(items)
    tr = [items[i] for i in split["train"]]
    va = [items[i] for i in split["val"]]
    if a.limit:
        tr, va = tr[:a.limit], va[:max(a.limit // 2, 4)]
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, rank=a.rank)
    print(f"treino {len(tr)} | validação {len(va)} | teste {len(split['test'])} (o teste só é usado em src.finetune.evaluate_models)")
    model, tok = llm.load(threads=a.threads)
    model = apply_lora(model, cfg)
    t, total = trainable_params(model)
    print(f"parâmetros treináveis: {t:,} de {total:,} ({t / total:.2%})")
    h = train(model, tok, tr, va, cfg, Path(a.out), log=lambda s: print(s, flush=True))
    print(f"fim: {h['train_seconds']:.0f}s | melhor época {h['best_epoch']} | F1 macro (validação) {h['best_val_macro_f1']:.3f}")
