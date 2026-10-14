"""Executa o comitê de investimentos (educacional).

Uso:
  python -m src.committee.run --profile moderado                 # regras determinísticas: grátis
  python -m src.committee.run --profile arrojado --news          # acrescenta manchetes do índice RAG (local, grátis)
  python -m src.committee.run --profile moderado --brain llm     # usa a API paga (modelo menor + modelo maior)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from src.committee.brains import LLMBrain, RulesBrain
from src.committee.config import ANALYST_MODEL, MANAGER_MODEL, PORTFOLIOS_DIR, PROFILES, default_universe
from src.committee.graph import Deps, build_graph
from src.committee.market import load_universe
from src.tools.stock_quote import get_stock_quote


def make_news_fn(enabled: bool, days: int = 14, k: int = 2):
    if not enabled:
        return lambda ticker: []
    from src.tools.news_search import search_news  # carrega o modelo de embeddings sob demanda

    def news(ticker: str) -> list[dict]:
        res = search_news(f"notícias recentes sobre {ticker}", ticker=ticker, days=days, k=k)
        return [{"titulo": r["titulo"], "fonte": r["fonte"], "data": r["data"]} for r in res["resultados"]]

    return news


def make_brain(kind: str):
    if kind == "rules":
        return RulesBrain()
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic

    load_dotenv()
    return LLMBrain(ChatAnthropic(model=ANALYST_MODEL, max_tokens=1500), ChatAnthropic(model=MANAGER_MODEL, max_tokens=1500))


def print_report(rep: dict, log=print) -> None:
    log(f"\n=== Carteira sugerida — perfil {rep['perfil']} ===")
    log(f"{'ticker':7s} {'peso':>6s} {'stance':10s} {'score':>6s} {'vol':>5s} {'drawdown':>8s} {'risco%':>7s}")
    for r in sorted(rep["carteira"], key=lambda x: -x["peso"]):
        log(f"{r['ticker']:7s} {r['peso']:6.1%} {r['stance']:10s} {r['score']:+6.2f} {r['volatilidade']:5.0%} {r['drawdown']:8.0%} {r['contribuicao_risco']:7.1%}")
    log(f"\nvolatilidade estimada da carteira: {rep['volatilidade_carteira']:.1%} | razão de diversificação: {rep['razao_diversificacao']:.2f}")
    log(f"gestor: {rep['racional_gestor']}")
    for f in rep["alertas_risco"]:
        log(f"alerta: {f}")
    for n in rep["notas"]:
        log(f"nota: {n}")
    log(f"\n{rep['aviso']}")


def run(profile: str, brain: str = "rules", universe: list[str] | None = None, news: bool = False, save: bool = True, log=print) -> dict:
    universe = universe or default_universe()
    log(f"Perfil: {profile} | cérebro: {brain} | universo: {', '.join(universe)}")
    deps = Deps(
        load_series=lambda tickers: load_universe(tickers, "5y", log=log),
        quote=get_stock_quote,
        news=make_news_fn(news),
        brain=make_brain(brain),
    )
    graph = build_graph(deps)
    final = None
    for update in graph.stream({"profile": profile, "universe": universe, "attempts": 0}, stream_mode=["updates", "values"]):
        mode, payload = update
        if mode == "updates":
            for node, data in payload.items():
                tr = data.get("trace", [{}])[-1]
                log(f"[{node}] {tr.get('ms', 0)} ms — {tr.get('info', '')}")
        else:
            final = payload
    print_report(final["report"], log)
    if save:
        PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
        out = PORTFOLIOS_DIR / f"carteira_{profile}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.json"
        out.write_text(json.dumps({"report": final["report"], "trace": final["trace"]}, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"\nsalvo em {out}")
    return final


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="moderado", choices=list(PROFILES))
    ap.add_argument("--brain", default="rules", choices=["rules", "llm"])
    ap.add_argument("--tickers", help="lista separada por vírgula (padrão: universo gratuito, ou o ampliado com BRAPI_TOKEN)")
    ap.add_argument("--news", action="store_true", help="inclui manchetes recuperadas do índice RAG (local)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    run(a.profile, a.brain, a.tickers.split(",") if a.tickers else None, a.news, not a.no_save)
