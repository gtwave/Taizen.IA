"""Dados da Fase 5: texto de entrada, agrupamento de quase-duplicatas e divisão treino/validação/teste SEM vazamento.

Por que agrupar: o mesmo fato sai em vários veículos ("Ibovespa bate recorde" em quatro sites). Se uma versão cai no treino e
outra no teste, o modelo "decora" a notícia e a métrica de teste mente. A divisão é feita por GRUPOS de textos parecidos.
"""

import random
import re
from collections import Counter

import numpy as np

from src.finetune.labeling import LABELS

TEXT_CHARS = 420  # título + começo do texto: o suficiente para o tom, e curto para o treino em CPU


def make_text(item: dict, chars: int = TEXT_CHARS) -> str:
    """Entrada do modelo: título e abertura da notícia, em uma linha, com espaços normalizados."""
    body = re.sub(r"\s+", " ", item["text"]).strip()
    title = re.sub(r"\s+", " ", item["title"]).strip()
    if body.lower().startswith(title.lower()[:40]):
        body = body[len(title):].strip(" .:-–—")  # muitos feeds repetem o título no começo do texto
    return f"{title}. {body}"[:chars].strip()


def group_near_duplicates(items: list[dict], threshold: float = 0.4) -> list[int]:
    """Id de grupo por item: componentes conexos de pares com similaridade TF-IDF (cosseno) ≥ `threshold`."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [make_text(it, 300) for it in items]
    m = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(docs)
    sim = (m @ m.T).toarray()
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def split_by_group(items: list[dict], groups: list[int], fractions=(0.6, 0.2, 0.2), seed: int = 0, tries: int = 300) -> dict[str, list[int]]:
    """Divide índices em train/val/test inteiros por grupo, escolhendo (entre `tries` sorteios) a divisão cuja distribuição de
    classes e cujos tamanhos mais se aproximam das desejadas."""
    by_group: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    gids = list(by_group)
    total = Counter(it["label"] for it in items)
    n = len(items)
    rng = random.Random(seed)
    best, best_cost = None, float("inf")
    for _ in range(tries):
        rng.shuffle(gids)
        parts: list[list[int]] = [[], [], []]
        for g in gids:  # preenche as partes na ordem, respeitando a cota de tamanho
            for k in range(3):
                if len(parts[k]) + len(by_group[g]) <= fractions[k] * n * 1.04 or k == 2:
                    parts[k].extend(by_group[g])
                    break
        cost = 0.0
        for k, p in enumerate(parts):
            c = Counter(items[i]["label"] for i in p)
            cost += abs(len(p) / n - fractions[k]) * 4
            for lab in LABELS:
                cost += abs(c.get(lab, 0) / max(len(p), 1) - total[lab] / n)
        if cost < best_cost:
            best, best_cost = [list(p) for p in parts], cost
    return {"train": sorted(best[0]), "val": sorted(best[1]), "test": sorted(best[2])}


def class_weights(labels: list[str]) -> np.ndarray:
    """Pesos ∝ 1/sqrt(frequência), normalizados para média 1: corrige o desbalanceamento sem exagerar."""
    c = Counter(labels)
    w = np.array([1.0 / np.sqrt(max(c.get(lab, 1), 1)) for lab in LABELS])
    return w / w.mean()


def overlap_report(items: list[dict], groups: list[int], split: dict[str, list[int]]) -> dict:
    """Confere o vazamento: quantos grupos aparecem em mais de uma parte (deve ser 0)."""
    where: dict[int, set] = {}
    for name, idx in split.items():
        for i in idx:
            where.setdefault(groups[i], set()).add(name)
    return {"groups": len(where), "leaking_groups": sum(1 for s in where.values() if len(s) > 1)}
