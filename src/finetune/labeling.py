"""Rótulos de sentimento do acervo de notícias (Fase 5) e utilitários para gravá-los/lê-los.

PROCEDIMENTO DE ROTULAGEM (declarado, porque rótulos são parte do modelo):
  * Anotador único: o assistente de IA (Claude) desta sessão, lendo título + começo do texto de cada notícia, SEM chamar a
    API paga. Não há um segundo anotador; portanto NÃO existe concordância entre anotadores medida. São rótulos "prata"
    (silver labels): bons para aprender e comparar métodos, insuficientes para afirmar acurácia "real".
  * Pergunta feita a cada notícia: "qual o tom desta notícia para o MERCADO e o investidor?"
      positivo — favorece ativos/empresas/mercado (alta de índices, lucro acima do esperado, dividendos, recompra, melhora de
                 rating, projeções de valorização, juros/inflação surpreendendo para baixo);
      negativo — desfavorece (quedas, prejuízo, rebaixamento, inadimplência, riscos, projeções cortadas, recuperação judicial);
      neutro   — informativo, misto/sem direção, político-eleitoral, análise técnica, agenda e tudo que não é financeiro
                 (esportes, loteria, crimes, cultura, guias práticos).
  * Regra de desempate: notícias MISTAS (lucro sobe mas ações caem; alta com ressalvas) → neutro, salvo quando o título
    destaca claramente um sentido.
  * O conteúdo das notícias pertence aos veículos: os dados ficam só em `data/` (ignorado pelo Git); aqui só há os índices.

Os índices abaixo referem-se à posição no arquivo `data/raw/news.jsonl` NA DATA DA ROTULAGEM (23/10/2026, 536 artigos);
`write_labels` grava também o `id` do artigo, que é o identificador estável.
"""

import json
from pathlib import Path

from src.config import DATA_DIR, NEWS_FILE

FINETUNE_DIR = DATA_DIR / "finetune"
LABELS_FILE = FINETUNE_DIR / "labels.jsonl"
LABELS = ["negativo", "neutro", "positivo"]

POSITIVE = {
    0, 1, 6, 19, 22, 29, 33, 34, 74, 80, 84, 87, 100, 111, 117, 120, 121, 132, 144, 145, 150, 151, 152, 153, 154, 158, 167, 170,
    180, 189, 190, 194, 196, 197, 201, 211, 215, 216, 218, 222, 223, 237, 250, 252, 253, 254, 255, 259, 260, 261, 263, 264,
    272, 276, 280, 286, 293, 295, 297, 298, 308, 321, 324, 332, 337, 358, 359, 361, 362, 366, 367, 368, 372, 374, 376, 384,
    386, 398, 399, 401, 408, 411, 416, 419, 430, 441, 443, 447, 453, 455, 456, 457, 458, 467, 470, 479, 492, 495, 500, 508,
    525, 533,
}
NEGATIVE = {
    3, 7, 10, 11, 35, 50, 57, 58, 64, 82, 83, 88, 89, 92, 95, 115, 155, 164, 166, 212, 225, 238, 239, 251, 256, 267, 273, 278,
    282, 314, 315, 323, 355, 357, 369, 377, 378, 379, 394, 404, 409, 412, 413, 418, 423, 424, 425, 427, 428, 429, 445, 446,
    450, 452, 454, 472, 473, 476, 480, 493, 496, 497, 503, 507, 510, 513, 514, 516, 517, 521, 522, 523, 526, 528, 534,
}


def label_of(index: int) -> str:
    if index in POSITIVE:
        return "positivo"
    if index in NEGATIVE:
        return "negativo"
    return "neutro"


def read_articles(path: Path = NEWS_FILE) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_labels(path: Path = LABELS_FILE, news_path: Path = NEWS_FILE) -> int:
    """Grava `labels.jsonl` (id, índice, rótulo) para os 536 artigos do acervo da data da rotulagem."""
    arts = read_articles(news_path)[:536]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, a in enumerate(arts):
            f.write(json.dumps({"id": a["id"], "index": i, "label": label_of(i), "annotator": "claude-session", "method": "silver"}, ensure_ascii=False) + "\n")
    return len(arts)


def load_labeled(path: Path = LABELS_FILE, news_path: Path = NEWS_FILE) -> list[dict]:
    """Une rótulos e artigos pelo id. Cada item: id, label, title, text, source, published."""
    by_id = {a["id"]: a for a in read_articles(news_path)}
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        a = by_id.get(r["id"])
        if a is not None:
            out.append({"id": r["id"], "label": r["label"], "title": a["title"], "text": a["text"], "source": a["source"], "published": a["published"]})
    return out


if __name__ == "__main__":
    n = write_labels()
    from collections import Counter

    items = load_labeled()
    print(f"{n} rótulos gravados em {LABELS_FILE}")
    print(dict(Counter(x["label"] for x in items)))
