"""Métricas de classificação e intervalos de confiança por bootstrap (só numpy).

Com conjuntos de teste pequenos (dezenas de exemplos), a incerteza é grande: toda comparação deve vir com intervalo.
"""

import numpy as np

from src.finetune.labeling import LABELS


def confusion(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(LABELS)}
    m = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m  # linhas = verdadeiro, colunas = previsto


def per_class(cm: np.ndarray) -> dict:
    out = {}
    for i, lab in enumerate(LABELS):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i, :].sum())}
    return out


def summary(y_true: list[str], y_pred: list[str]) -> dict:
    cm = confusion(y_true, y_pred)
    pc = per_class(cm)
    return {"accuracy": float(np.trace(cm) / max(cm.sum(), 1)), "macro_f1": float(np.mean([v["f1"] for v in pc.values()])), "per_class": pc, "confusion": cm.tolist()}


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return summary(y_true, y_pred)["macro_f1"]


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))


def bootstrap_ci(y_true: list[str], y_pred: list[str], metric=macro_f1, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confiança percentil (1-alpha) da métrica, reamostrando os exemplos com reposição."""
    rng = np.random.default_rng(seed)
    t, p = np.array(y_true), np.array(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        vals.append(metric(list(t[idx]), list(p[idx])))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(y_true: list[str], pred_a: list[str], pred_b: list[str], metric=macro_f1, n: int = 2000, seed: int = 0) -> dict:
    """Diferença métrica(A) − métrica(B) nos MESMOS exemplos reamostrados; devolve média, IC 95% e P(A > B)."""
    rng = np.random.default_rng(seed)
    t, a, b = np.array(y_true), np.array(pred_a), np.array(pred_b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        diffs.append(metric(list(t[idx]), list(a[idx])) - metric(list(t[idx]), list(b[idx])))
    d = np.array(diffs)
    return {"mean": float(d.mean()), "ci": (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))), "p_a_better": float((d > 0).mean())}
