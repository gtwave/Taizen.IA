"""Avaliação da recuperação com um golden set de perguntas (tudo local, sem LLM, sem custo).

Cada pergunta traz `ev` (evidência): termos que o chunk precisa conter, todos, para ser considerado relevante
(comparação sem acentos e sem diferença de maiúsculas). É um oráculo simples e tem viés a favor da busca lexical,
porque a evidência é texto literal — veja o Capítulo 7. As perguntas foram escritas de forma parafraseada,
para não repetir as palavras exatas do trecho.

Métricas (média sobre as perguntas): Hit@k (algum chunk relevante entre os k primeiros), Hit@1 e MRR.

Atenção: as perguntas foram escritas para o acervo coletado em 13/10/2026; com outro acervo, adapte o GOLDEN.

Uso:  python -m src.rag.evaluate [--k 5] [--ablate]
"""

import argparse
import sys
import time
import unicodedata

from src.config import DATA_DIR
from src.rag.chunking import articles_to_chunks
from src.rag.embeddings import E5Embeddings
from src.rag.retrieval import BM25Index, hybrid_search, vector_search
from src.rag.store import get_store, index_chunks, load_articles

GOLDEN = [
    {"q": "Quanto cresceu o volume de esgoto medido pela Copasa no trimestre?", "ev": ["123,8 milhoes de metros cubicos"]},
    {"q": "Qual foi a alta do esgoto tratado pela companhia de saneamento de Minas Gerais?", "ev": ["123,8 milhoes de metros cubicos"]},
    {"q": "Qual a projeção de crescimento da economia americana em 2026 no resumo dos mercados?", "ev": ["2,3% em 2026"]},
    {"q": "O que aconteceria com as ações americanas se a promessa de produtividade da tecnologia falhar?", "ev": ["cair 20%"]},
    {"q": "Quanto caíram as teles europeias depois do anúncio da SpaceX sobre telefonia móvel?", "ev": ["deutsche telekom", "6,65%"]},
    {"q": "Que faixa de frequência a SpaceX pretende comprar da Grain Management?", "ev": ["800 mhz"]},
    {"q": "Qual a taxa básica de juros citada no debate sobre o split payment?", "ev": ["selic em 13,75%"]},
    {"q": "De quanto foi o salto da bolsa no pregão seguinte à votação, segundo o Fundo Verde?", "ev": ["avancou 7,7%"]},
    {"q": "Quanto o investidor recebe antecipadamente ao vender a opção, na reportagem sobre lucrar com ações em queda?", "ev": ["premio de r$ 0,20"]},
    {"q": "Qual percentual dos voos foi cancelado nos aeroportos da Arábia Saudita?", "ev": ["46,4%"]},
    {"q": "Que fatia do dinheiro especialistas mandam reservar para gastos extras na compra de imóvel em leilão?", "ev": ["entre 20% e 40%"]},
    {"q": "Qual o valor do prêmio de Economia de 2026 convertido para dólares?", "ev": ["us$ 1,2 milhao"]},
    {"q": "Quem foi a vencedora do Nobel de Literatura deste ano?", "ev": ["anne carson"]},
    {"q": "Quem paga o IPTU atrasado em imóveis da Caixa quando a dívida passa de 10% do valor de avaliação?", "ev": ["supera 10% do valor de avaliacao"]},
    {"q": "Qual banco anunciou distribuição de R$ 1,5 bilhão aos acionistas?", "ev": ["santander", "1,5 bilhao"]},
    {"q": "Qual a alta acumulada do Ibovespa na semana, citada na abertura de terça-feira?", "ev": ["8,82%"]},
]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def chunk_is_relevant(chunk_text: str, ev: list[str]) -> bool:
    t = norm(chunk_text)
    return all(term in t for term in ev)


def count_relevant(chunks, golden=GOLDEN) -> list[int]:
    return [sum(chunk_is_relevant(c.page_content, g["ev"]) for c in chunks) for g in golden]


def score(search_fn, golden=GOLDEN, k: int = 5) -> dict:
    """Médias de Hit@k, Hit@1 e MRR (posição do primeiro chunk relevante)."""
    hit = hit1 = mrr = 0.0
    for g in golden:
        hits = search_fn(g["q"])[:k]
        flags = [chunk_is_relevant(h.doc.page_content, g["ev"]) for h in hits]
        hit += any(flags)
        hit1 += bool(flags and flags[0])
        mrr += next((1.0 / (i + 1) for i, f in enumerate(flags) if f), 0.0)
    n = len(golden)
    return {"hit": hit / n, "hit1": hit1 / n, "mrr": mrr / n}


def build_variant(articles, name: str, embeddings, chunk_size=800, overlap=120, with_title=True, log=print):
    """Cria um índice experimental separado (coleção própria) e devolve (store, bm25, tempo de indexação)."""
    chunks = articles_to_chunks(articles, chunk_size, overlap, with_title)
    store = get_store(embeddings, DATA_DIR / "experiments" / name, name)
    t0 = time.perf_counter()
    index_chunks(store, chunks, log=lambda *_: None)
    dt = time.perf_counter() - t0
    return store, BM25Index.from_store(store), len(chunks), dt


def report(rows: list[tuple[str, dict]], k: int) -> None:
    print(f"{'configuração':36s} {'Hit@' + str(k):>6s} {'Hit@1':>6s} {'MRR':>6s}")
    for name, m in rows:
        print(f"{name:36s} {m['hit']:6.2f} {m['hit1']:6.2f} {m['mrr']:6.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ablate", action="store_true", help="compara prefixos, título e tamanho de chunk (reindexa)")
    a = ap.parse_args()
    arts = load_articles()
    emb = E5Embeddings()
    store, bm, n, dt = build_variant(arts, "base", emb)
    base_chunks = articles_to_chunks(arts)
    for g, c in zip(GOLDEN, count_relevant(base_chunks)):
        print(f"{c:3d} chunk(s) relevante(s) | {g['q'][:95]}")
    rows = []
    rows.append(("vetorial (base)", score(lambda q: vector_search(store, q, a.k), GOLDEN, a.k)))
    rows.append(("BM25 (base)", score(lambda q: bm.search(q, a.k), GOLDEN, a.k)))
    rows.append(("híbrida RRF (base)", score(lambda q: hybrid_search(store, bm, q, a.k), GOLDEN, a.k)))
    if a.ablate:
        emb_np = E5Embeddings(use_prefix=False)
        s2, b2, n2, _ = build_variant(arts, "sem_prefixo", emb_np)
        rows.append(("vetorial sem prefixos e5", score(lambda q: vector_search(s2, q, a.k), GOLDEN, a.k)))
        s3, b3, n3, _ = build_variant(arts, "sem_titulo", emb, with_title=False)
        rows.append(("vetorial sem título no chunk", score(lambda q: vector_search(s3, q, a.k), GOLDEN, a.k)))
        for size, ov in [(400, 60), (1500, 200)]:
            s4, b4, n4, _ = build_variant(arts, f"chunk_{size}", emb, size, ov)
            rows.append((f"vetorial chunk {size} ({n4} chunks)", score(lambda q: vector_search(s4, q, a.k), GOLDEN, a.k)))
            rows.append((f"híbrida chunk {size}", score(lambda q: hybrid_search(s4, b4, q, a.k), GOLDEN, a.k)))
    print()
    report(rows, a.k)
