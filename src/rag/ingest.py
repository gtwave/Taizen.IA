"""Ingestão: coleta os feeds, normaliza, remove duplicados e grava em JSONL.

Uso:  python -m src.rag.ingest [--pages N] [--sources infomoney,moneytimes]
"""

import argparse
import json
import sys

from src.config import NEWS_FILE, RAW_DIR
from src.rag.companies import find_tickers
from src.rag.feeds import FEEDS, Feed, fetch_feed, make_session

MIN_TEXT_CHARS = 300


def load_existing(path=NEWS_FILE) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {a["id"]: a for a in map(json.loads, f)}


def merge_new(existing: dict[str, dict], fetched: list[dict]) -> tuple[list[dict], dict]:
    """Devolve (artigos novos válidos, estatísticas). Descarta duplicados e textos curtos."""
    stats = {"fetched": len(fetched), "duplicates": 0, "too_short": 0, "new": 0}
    new = []
    seen = set(existing)
    for a in fetched:
        if a["id"] in seen:
            stats["duplicates"] += 1
        elif len(a["text"]) < MIN_TEXT_CHARS:
            stats["too_short"] += 1
        else:
            a["tickers"] = find_tickers(a["text"], a["title"])
            new.append(a)
            seen.add(a["id"])
    stats["new"] = len(new)
    return new, stats


def run(pages: int | None = None, sources: list[str] | None = None, log=print) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing()
    session = make_session()
    total = {"fetched": 0, "duplicates": 0, "too_short": 0, "new": 0}
    new_all: list[dict] = []
    for feed in FEEDS:
        if sources and feed.name not in sources:
            continue
        f = Feed(feed.name, feed.url, pages or feed.pages)
        log(f"Coletando {f.name} ({f.pages} página(s))...")
        fetched = fetch_feed(f, session, log)
        new, stats = merge_new({**existing, **{a["id"]: a for a in new_all}}, fetched)
        new_all.extend(new)
        log(f"  {f.name}: {stats['fetched']} lidos | {stats['new']} novos | "
            f"{stats['duplicates']} duplicados | {stats['too_short']} curtos")
        for k in total:
            total[k] += stats[k]
    if new_all:
        with NEWS_FILE.open("a", encoding="utf-8") as fh:
            for a in new_all:
                fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    log(f"Total: {total['new']} artigos novos gravados em {NEWS_FILE} (acervo: {len(existing) + len(new_all)})")
    return total


def retag(log=print) -> dict:
    """Recalcula os tickers de todos os artigos já coletados (útil após mudar as regras de detecção)."""
    arts = list(load_existing().values())
    before = sum(1 for a in arts if a.get("tickers"))
    for a in arts:
        a["tickers"] = find_tickers(a["text"], a["title"])
    with NEWS_FILE.open("w", encoding="utf-8") as fh:
        for a in arts:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    after = sum(1 for a in arts if a["tickers"])
    log(f"retag: artigos com ao menos um ticker: {before} -> {after} (de {len(arts)})")
    return {"before": before, "after": after, "total": len(arts)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, help="páginas por feed (sobrescreve o padrão)")
    ap.add_argument("--sources", help="lista de feeds separada por vírgula")
    ap.add_argument("--retag", action="store_true", help="só recalcula os tickers do acervo existente")
    args = ap.parse_args()
    if args.retag:
        retag()
        sys.exit(0)
    run(args.pages, args.sources.split(",") if args.sources else None)
