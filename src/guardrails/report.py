"""Painel de auditoria dos guardrails: quantas intervenções, onde, por qual regra — sem nunca ver o conteúdo.

Uso:  python -m src.guardrails.report [--days 30] [--file data/processed/guard_audit.jsonl]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.guardrails.policy import AUDIT_FILE


def read_audit(path: Path = AUDIT_FILE, since: float | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # linha truncada por uma interrupção
        if since is None or rec.get("ts", 0) >= since:
            out.append(rec)
    return out


def summarize(records: list[dict]) -> dict:
    by_stage, by_action, by_rule = Counter(), Counter(), Counter()
    stage_action: Counter = Counter()
    repeated = Counter(r["content_hash"] for r in records if r.get("content_hash"))
    for r in records:
        by_stage[r["stage"]] += 1
        by_action[r["action"]] += 1
        stage_action[(r["stage"], r["action"])] += 1
        for rule in list(r.get("rules", [])) + list(r.get("pii", [])):
            by_rule[rule] += 1
    return {"total": len(records), "by_stage": dict(by_stage), "by_action": dict(by_action), "by_rule": dict(by_rule),
            "stage_action": {f"{s}/{a}": n for (s, a), n in stage_action.items()},
            "repeated_content": sum(1 for n in repeated.values() if n > 1), "monitor_only": by_action.get("monitor", 0)}


def format_report(records: list[dict]) -> str:
    if not records:
        return "Nenhuma decisão registrada. Rode o agente ou o comitê com guardrails (padrão: enforce)."
    s = summarize(records)
    lines = [f"{s['total']} decisões registradas (somente intervenções e sinais; textos limpos não geram linha)"]
    lines.append("\npor ponto de controle / ação:")
    for k, n in sorted(s["stage_action"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:24s} {n:5d}")
    lines.append("\nregras e tipos de dado mais acionados:")
    for k, n in sorted(s["by_rule"].items(), key=lambda kv: -kv[1])[:12]:
        lines.append(f"  {k:28s} {n:5d}")
    if s["monitor_only"]:
        lines.append(f"\n{s['monitor_only']} decisão(ões) em modo monitor: revise por amostragem antes de ligar o enforce.")
    if s["repeated_content"]:
        lines.append(f"{s['repeated_content']} conteúdo(s) idêntico(s) acionaram regras mais de uma vez (mesmo hash): possível tentativa repetida ou fonte ruidosa.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--file", default=str(AUDIT_FILE))
    a = ap.parse_args()
    print(format_report(read_audit(Path(a.file), since=time.time() - a.days * 86400)))
