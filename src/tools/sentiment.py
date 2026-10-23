"""Ferramenta de sentimento (Fase 5): classifica o tom de notícias financeiras com um modelo LOCAL — sem API paga.

Backends, do melhor ao mais simples (escolhidos automaticamente, ou forçados com `backend=`):
  lora     — SLM Qwen2.5-0.5B com adaptador LoRA treinado na Fase 5 (precisa do adaptador em models/lora-sentiment);
  tfidf    — TF-IDF + regressão logística treinada na hora a partir de data/finetune/labels.jsonl;
  lexicon  — contagem de palavras (sempre disponível).

Cada chamada vira um span de tracing (kind="model") com custo 0 (modelo local; energia não contabilizada).
"""

from pathlib import Path

from src.observability.tracing import get_tracer

_classifier = None


class SentimentClassifier:
    def __init__(self, backend: str = "auto", adapter_dir: Path | None = None):
        from src.finetune import llm

        self.adapter_dir = Path(adapter_dir) if adapter_dir else llm.ADAPTER_DIR
        self.backend = self._resolve(backend)
        self._model = self._tok = self._sk = None

    def _resolve(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if (self.adapter_dir / "adapter_config.json").exists():
            return "lora"
        from src.finetune.labeling import LABELS_FILE

        return "tfidf" if LABELS_FILE.exists() else "lexicon"

    def _load(self):
        if self.backend == "lora" and self._model is None:
            from src.finetune import llm

            self._model, self._tok = llm.load(adapter=self.adapter_dir)
        elif self.backend in ("tfidf", "lexicon") and self._sk is None:
            from src.finetune import baselines, labeling
            from src.finetune.train_lora import make_split

            if self.backend == "lexicon":
                self._sk = baselines.LexiconBaseline()
            else:
                items = labeling.load_labeled()
                sp = make_split(items)
                self._sk = baselines.TfidfLR().fit([items[i] for i in sp["train"]], [items[i] for i in sp["val"]])

    def predict(self, texts: list[str]) -> list[dict]:
        """Devolve, para cada texto, {"label": ..., "scores": {rótulo: prob}} (scores só nos backends que os têm)."""
        from src.finetune.labeling import LABELS

        with get_tracer().span(f"sentiment:{self.backend}", "model", backend=self.backend, n=len(texts), local=True, cost_brl=0.0, priced=True):
            self._load()
            items = [{"title": t, "text": ""} for t in texts]
            if self.backend == "lora":
                from src.finetune import llm

                _, probs = llm.predict_items(self._model, self._tok, [{"title": t, "text": ""} for t in texts])
            elif self.backend == "tfidf":
                probs = self._sk.predict_proba(items)
            else:
                return [{"label": lab} for lab in self._sk.predict(items)]
            return [{"label": LABELS[int(p.argmax())], "scores": {lab: round(float(x), 4) for lab, x in zip(LABELS, p)}} for p in probs]


def get_classifier(backend: str = "auto") -> SentimentClassifier:
    global _classifier
    if _classifier is None or (backend != "auto" and _classifier.backend != backend):
        _classifier = SentimentClassifier(backend)
    return _classifier


def classify_sentiment(text: str, backend: str = "auto") -> dict:
    """Tom (positivo/negativo/neutro) de um título/trecho de notícia financeira. Função-ferramenta; texto não confiável entra aqui:
    a saída é restrita a 3 rótulos, então uma instrução escondida no texto não consegue produzir texto livre."""
    return get_classifier(backend).predict([text])[0]
