"""Linhas de base: toda afirmação de "o fine-tuning melhorou" precisa de algo simples para comparar.

  majority  — sempre a classe mais frequente do treino (o piso: acurácia alta num conjunto desbalanceado, F1 macro baixo);
  lexicon   — contagem de palavras de alta/queda (regra escrita à mão, sem aprendizado);
  tfidf_lr  — TF-IDF + regressão logística (aprende com os rótulos; treina em segundos na CPU).
"""

import re
import unicodedata
from collections import Counter

import numpy as np

from src.finetune.data import make_text
from src.finetune.labeling import LABELS
from src.finetune.metrics import macro_f1

POS_WORDS = ["alta", "altas", "sobe", "sobem", "subiu", "dispara", "disparam", "salta", "recorde", "lucro", "lucra", "dividendos", "jcp", "recompra",
             "valorizacao", "valoriza", "eleva", "supera", "acima do esperado", "ganhos", "avanca", "avanco", "otimismo", "recupera", "oportunidade",
             "oportunidades", "melhora", "positiva", "compra", "preco-alvo", "bate recorde", "cresce", "crescimento"]
NEG_WORDS = ["queda", "cai", "caem", "despenca", "despencam", "tomba", "perde", "perdas", "prejuizo", "rebaixa", "rebaixamento", "calote", "recua", "recuo",
             "pressao", "risco", "riscos", "alerta", "inadimplencia", "corta", "cortes", "recessao", "desacelera", "desaceleracao", "preocupa", "crise",
             "pior", "fraco", "abaixo do esperado", "cautela", "incerteza", "recuperacao judicial", "bear market", "tombo"]


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


class MajorityBaseline:
    name = "majority"

    def fit(self, train: list[dict], val: list[dict] | None = None):
        self.label = Counter(x["label"] for x in train).most_common(1)[0][0]
        return self

    def predict(self, items: list[dict]) -> list[str]:
        return [self.label] * len(items)


class LexiconBaseline:
    name = "lexicon"

    def __init__(self, margin: int = 1):
        self.margin = margin

    def fit(self, train: list[dict], val: list[dict] | None = None):
        # única "aprendizagem": escolhe a margem (1 ou 2) que maximiza o F1 macro no treino
        best = (-1.0, 1)
        for m in (1, 2):
            self.margin = m
            f = macro_f1([x["label"] for x in train], self.predict(train))
            best = max(best, (f, m))
        self.margin = best[1]
        return self

    def score(self, item: dict) -> int:
        text = " " + _norm(make_text(item, 260)) + " "
        pos = sum(len(re.findall(rf"\b{re.escape(w)}\b", text)) for w in POS_WORDS)
        neg = sum(len(re.findall(rf"\b{re.escape(w)}\b", text)) for w in NEG_WORDS)
        return pos - neg

    def predict(self, items: list[dict]) -> list[str]:
        out = []
        for it in items:
            s = self.score(it)
            out.append("positivo" if s >= self.margin else "negativo" if s <= -self.margin else "neutro")
        return out


class TfidfLR:
    name = "tfidf_lr"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _vec(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    def fit(self, train: list[dict], val: list[dict] | None = None):
        from sklearn.linear_model import LogisticRegression

        self.vec = self._vec()
        x = self.vec.fit_transform([make_text(it) for it in train])
        y = [it["label"] for it in train]
        best = (-1.0, 3.0)
        for c in (0.5, 1.0, 3.0, 10.0, 30.0):  # C escolhido pela validação (nunca pelo teste)
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=self.seed).fit(x, y)
            if val:
                f = macro_f1([it["label"] for it in val], list(clf.predict(self.vec.transform([make_text(it) for it in val]))))
                if f > best[0]:
                    best = (f, c)
        self.C = best[1]
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed).fit(x, y)
        return self

    def predict_proba(self, items: list[dict]) -> np.ndarray:
        p = self.clf.predict_proba(self.vec.transform([make_text(it) for it in items]))
        order = [list(self.clf.classes_).index(lab) if lab in self.clf.classes_ else None for lab in LABELS]
        return np.column_stack([p[:, o] if o is not None else np.zeros(len(items)) for o in order])

    def predict(self, items: list[dict]) -> list[str]:
        return [LABELS[i] for i in self.predict_proba(items).argmax(axis=1)]
