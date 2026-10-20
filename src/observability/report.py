"""Painel de linha de comando: latência, tokens, custo, erros, cache e orçamento a partir dos traces gravados.

Uso:  python -m src.observability.report [--days 7] [--dir data/processed/traces]
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

from src.config import DATA_DIR
from src.observability import pricing
from src.observability.tracing import read_spans

TRACES_DIR = DATA_DIR / "processed" / "traces"


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def summarize(spans: list[dict]) -> dict:
    """Agrega os spans por (tipo, nome). Custos só entram quando o modelo tem preço conhecido."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in spans:
        groups[(s["kind"], s["name"])].append(s)
    rows = []
    for (kind, name), ss in sorted(groups.items()):
        ms = [s["duration_ms"] for s in ss]
        rows.append({
            "kind": kind, "name": name, "n": len(ss), "errors": sum(s["status"] == "error" for s in ss),
            "p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95), "max_ms": max(ms),
            "tokens_in": sum(s["attrs"].get("input_tokens", 0) for s in ss),
            "tokens_out": sum(s["attrs"].get("output_tokens", 0) for s in ss),
            "cost_brl": sum(s["attrs"].get("cost_brl") or 0.0 for s in ss),
            "unpriced": sum(1 for s in ss if kind == "llm" and not s["attrs"].get("priced", False)),
        })
    llm = [s for s in spans if s["kind"] == "llm"]
    cached = [s["attrs"]["cache"] for s in spans if "cache" in s["attrs"]]
    roots = [s for s in spans if s["parent_id"] is None]
    return {
        "rows": rows,
        "traces": len({s["trace_id"] for s in spans}),
        "roots": roots,
        "llm_calls": len(llm),
        "cost_brl": sum(s["attrs"].get("cost_brl") or 0.0 for s in llm),
        "error_rate": (sum(s["status"] == "error" for s in spans) / len(spans)) if spans else 0.0,
        "cache": {"hit": sum(c in ("exact", "semantic") for c in cached), "miss": sum(c == "miss" for c in cached)},
    }


def format_report(spans: list[dict], budget_brl: float = pricing.MONTHLY_BUDGET_BRL, now=None) -> str:
    if not spans:
        return "Nenhum span encontrado. Rode um comando com rastreamento ligado (ex.: python -m src.committee.run)."
    s = summarize(spans)
    out = [f"{len(spans)} spans em {s['traces']} trace(s) | taxa de erro {s['error_rate']:.1%} | chamadas ao LLM: {s['llm_calls']} "
           f"| custo R$ {s['cost_brl']:.4f}"]
    out.append(f"\n{'tipo':6s} {'nome':34s} {'n':>4s} {'err':>3s} {'p50 ms':>8s} {'p95 ms':>8s} {'máx ms':>8s} {'tok in':>7s} {'tok out':>7s} {'R$':>9s}")
    for r in s["rows"]:
        flag = " *" if r["unpriced"] else ""
        out.append(f"{r['kind']:6s} {r['name'][:34]:34s} {r['n']:4d} {r['errors']:3d} {r['p50_ms']:8.0f} {r['p95_ms']:8.0f} "
                   f"{r['max_ms']:8.0f} {r['tokens_in']:7d} {r['tokens_out']:7d} {r['cost_brl']:9.4f}{flag}")
    if any(r["unpriced"] for r in s["rows"]):
        out.append("* há chamadas de modelo sem preço na tabela: o custo delas não está somado.")
    c = s["cache"]
    if c["hit"] + c["miss"]:
        out.append(f"\ncache: {c['hit']} acerto(s), {c['miss']} falha(s) — taxa de acerto {c['hit'] / (c['hit'] + c['miss']):.0%}")
    st = pricing.budget_status(spans, budget_brl, now)
    out.append(f"\norçamento do mês: R$ {st.spent_brl:.4f} de R$ {st.budget_brl:.2f} (projeção R$ {st.projected_brl:.2f}) — {st.level}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--dir", default=str(TRACES_DIR))
    ap.add_argument("--budget", type=float, default=pricing.MONTHLY_BUDGET_BRL)
    a = ap.parse_args()
    print(format_report(read_spans(a.dir, since=time.time() - a.days * 86400), a.budget))
