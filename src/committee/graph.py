"""O comitê como um grafo LangGraph: coleta -> (analista de dados || analista de risco) -> gestor -> validador -> relatório.

Os nós "pensantes" usam um cérebro (RulesBrain ou LLMBrain); os nós de coleta, validação e relatório são código puro.
O validador fecha um laço com o gestor (nova tentativa com os erros) e, esgotadas as tentativas, a carteira de
linha de base determinística assume — o sistema nunca entrega uma carteira que viole as restrições.
"""

import operator
import time
from dataclasses import dataclass
from typing import Annotated, Callable, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from src.committee import metrics
from src.committee.brains import MAX_DEVIATION, BrainError, RulesBrain, baseline_allocation
from src.committee.config import DISCLAIMER, MAX_ATTEMPTS, PROFILES, Profile
from src.committee.market import align, validate_series
from src.observability.tracing import get_tracer


class CommitteeState(TypedDict, total=False):
    profile: str
    universe: list[str]
    data: dict
    dossiers: dict
    views: dict
    risk: dict
    baseline: dict
    proposal: dict
    validation: dict
    attempts: int
    report: dict
    notes: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]


@dataclass
class Deps:
    """Dependências injetáveis (permitem rodar o grafo inteiro nos testes, sem rede e sem LLM)."""
    load_series: Callable[[list[str]], tuple[dict, dict]]
    quote: Callable[[str], dict]
    news: Callable[[str], list[dict]]
    brain: object


def validate_portfolio(weights: dict[str, float], tickers: list[str], profile: Profile, vetoes: list[str],
                       baseline: dict[str, float]) -> dict:
    """Confere todas as restrições. Devolve {"ok": bool, "errors": [...]}."""
    errors = []
    unknown = sorted(set(weights) - set(tickers))
    if unknown:
        errors.append(f"tickers fora do universo: {unknown}")
    w = {t: float(weights.get(t, 0.0)) for t in tickers}
    if any(v < -1e-9 for v in w.values()):
        errors.append("há pesos negativos")
    total = sum(w.values())
    if abs(total - 1.0) > 0.005:
        errors.append(f"os pesos somam {total:.4f}, não 1,0")
    eligible = [t for t in tickers if t not in vetoes]
    # com poucos ativos elegíveis o teto efetivo sobe para 1/k (senão a soma 1 seria impossível)
    max_weight = max(profile.max_weight, 1.0 / len(eligible)) if eligible else profile.max_weight
    for t, v in w.items():
        if v > max_weight + 1e-6:
            errors.append(f"{t} tem {v:.1%}, acima do teto de {max_weight:.0%}")
        if 0.0005 < v < profile.min_weight - 1e-6:
            errors.append(f"{t} tem {v:.1%}, abaixo do piso de {profile.min_weight:.0%}")
        if t in vetoes and v > 0.0005:
            errors.append(f"{t} foi vetado pelo analista de risco e não pode ter peso")
        if abs(v - baseline.get(t, 0.0)) > MAX_DEVIATION + 1e-6:
            errors.append(f"{t} desvia {abs(v - baseline.get(t, 0.0)):.1%} da linha de base (máximo {MAX_DEVIATION:.0%})")
    min_positions = min(profile.min_positions, len(eligible))
    if sum(1 for v in w.values() if v > 0.0005) < min_positions:
        errors.append(f"menos de {min_positions} posições")
    return {"ok": not errors, "errors": errors}


def _timed(name: str):
    def deco(fn):
        def wrapper(state):
            t0 = time.perf_counter()
            with get_tracer().span(name, "node") as span:
                out = fn(state) or {}
                info = out.pop("_info", "")
                span.set(info=info)
            ms = round((time.perf_counter() - t0) * 1000)
            return {**out, "trace": [{"no": name, "ms": ms, "info": info}]}
        wrapper.__name__ = name
        return wrapper
    return deco


def build_graph(deps: Deps):
    brain, rules = deps.brain, RulesBrain()

    @_timed("coletar")
    def coletar(state: CommitteeState):
        profile = PROFILES[state["profile"]]
        series, errors = deps.load_series(state["universe"])
        if len(series) < profile.min_positions:
            raise RuntimeError(f"dados insuficientes: {len(series)} ativo(s) disponível(is), mínimo {profile.min_positions}. Erros: {errors}")
        dates, prices, tickers = align(series)
        rets = metrics.simple_returns(prices)
        corr = metrics.correlation_matrix(rets)
        cov = metrics.covariance_matrix(rets)
        summaries, dossiers, notes = {}, {}, []
        for j, t in enumerate(tickers):
            summaries[t] = metrics.asset_summary(prices[:, j])
            q = deps.quote(t)
            warnings = validate_series(series[t])
            notes += [f"{t}: {w}" for w in warnings]
            dossiers[t] = {"metricas": summaries[t], "preco_lucro": q.get("price_earnings"),
                           "faixa_52s": q.get("fifty_two_week_range"), "manchetes": deps.news(t)}
        notes += [f"{t}: excluído — {e}" for t, e in errors.items()]
        corr_d = {a: {b: round(float(corr[i, k]), 3) for k, b in enumerate(tickers)} for i, a in enumerate(tickers)}
        data = {"tickers": tickers, "summaries": summaries, "corr": corr_d, "cov": cov.tolist(), "n_dias": len(dates),
                "periodo": [int(dates[0]), int(dates[-1])], "errors": errors}
        return {"data": data, "dossiers": dossiers, "notes": notes, "_info": f"{len(tickers)} ativos, {len(dates)} pregões"}

    @_timed("analista_dados")
    def analista_dados(state: CommitteeState):
        profile = PROFILES[state["profile"]]
        try:
            views = brain.data_analyst(state["dossiers"], profile)
            note = []
        except BrainError as exc:
            views, note = rules.data_analyst(state["dossiers"], profile), [f"analista de dados: LLM falhou ({exc}); usadas as regras"]
        return {"views": views, "notes": note, "_info": f"{len(views)} visões ({brain.name})"}

    @_timed("analista_risco")
    def analista_risco(state: CommitteeState):
        profile = PROFILES[state["profile"]]
        d = state["data"]
        try:
            risk = brain.risk_analyst(d["summaries"], d["corr"], profile)
            note = []
        except BrainError as exc:
            risk, note = rules.risk_analyst(d["summaries"], d["corr"], profile), [f"analista de risco: LLM falhou ({exc}); usadas as regras"]
        return {"risk": risk, "notes": note, "_info": f"{len(risk['vetoes'])} veto(s), {len(risk['flags'])} alerta(s)"}

    @_timed("gestor")
    def gestor(state: CommitteeState):
        profile = PROFILES[state["profile"]]
        d, tickers = state["data"], state["data"]["tickers"]
        vols = np.array([d["summaries"][t]["volatilidade_anual"] for t in tickers])
        scores = np.array([state["views"][t]["score"] for t in tickers])
        baseline = baseline_allocation(tickers, vols, scores, profile, state["risk"]["vetoes"])
        if len(state["risk"]["vetoes"]) >= len(tickers):
            raise RuntimeError("todos os ativos foram vetados pelo analista de risco: não há carteira possível para este perfil")
        feedback = (state.get("validation") or {}).get("errors")
        attempts = state.get("attempts", 0) + 1
        try:
            proposal = brain.manager(baseline, state["views"], state["risk"], profile, feedback)
            note = []
        except BrainError as exc:
            proposal, note = rules.manager(baseline, state["views"], state["risk"], profile), [f"gestor: LLM falhou ({exc}); usada a linha de base"]
        return {"baseline": baseline, "proposal": proposal, "attempts": attempts, "notes": note,
                "_info": f"tentativa {attempts} ({brain.name})"}

    @_timed("validador")
    def validador(state: CommitteeState):
        profile = PROFILES[state["profile"]]
        v = validate_portfolio(state["proposal"]["weights"], state["data"]["tickers"], profile, state["risk"]["vetoes"], state["baseline"])
        return {"validation": v, "_info": "aprovada" if v["ok"] else f"{len(v['errors'])} erro(s)"}

    def route(state: CommitteeState) -> str:
        if state["validation"]["ok"]:
            return "relatorio"
        return "gestor" if state.get("attempts", 0) < MAX_ATTEMPTS else "fallback"

    @_timed("fallback")
    def fallback(state: CommitteeState):
        prop = {"weights": dict(state["baseline"]), "rationale": "carteira de linha de base (determinística): a proposta do gestor não passou na validação."}
        # o fallback também é validado: nunca se entrega uma carteira que viole as restrições
        check = validate_portfolio(prop["weights"], state["data"]["tickers"], PROFILES[state["profile"]], state["risk"]["vetoes"], state["baseline"])
        if not check["ok"]:
            raise RuntimeError(f"a linha de base também viola as restrições: {check['errors']}")
        return {"proposal": prop, "validation": {"ok": True, "errors": []},
                "notes": [f"fallback após {state.get('attempts', 0)} tentativa(s) do gestor: {state['validation']['errors']}"], "_info": "linha de base"}

    @_timed("relatorio")
    def relatorio(state: CommitteeState):
        d, tickers = state["data"], state["data"]["tickers"]
        w = np.array([state["proposal"]["weights"].get(t, 0.0) for t in tickers], dtype=float)
        w = w / w.sum()
        cov = np.array(d["cov"])
        rc = metrics.risk_contributions(w, cov)
        notes = list(state.get("notes", []))
        eligible = len(tickers) - len(state["risk"]["vetoes"])
        if eligible < PROFILES[state["profile"]].min_positions:
            notes.append(f"diversificação reduzida: só {eligible} ativo(s) não vetado(s), abaixo do mínimo de "
                         f"{PROFILES[state['profile']].min_positions} posições do perfil; o veto de risco prevalece.")
        rows = [{"ticker": t, "peso": round(float(w[i]), 4), "stance": state["views"][t]["stance"], "score": state["views"][t]["score"],
                 "volatilidade": d["summaries"][t]["volatilidade_anual"], "drawdown": d["summaries"][t]["drawdown_maximo"],
                 "contribuicao_risco": round(float(rc[i]), 4), "justificativa": state["views"][t]["rationale"]} for i, t in enumerate(tickers)]
        report = {"perfil": state["profile"], "carteira": rows, "volatilidade_carteira": round(metrics.portfolio_vol(w, cov), 4),
                  "razao_diversificacao": round(metrics.diversification_ratio(w, cov), 3), "racional_gestor": state["proposal"]["rationale"],
                  "alertas_risco": state["risk"]["flags"], "notas": notes, "aviso": DISCLAIMER}
        return {"report": report, "_info": f"volatilidade da carteira {report['volatilidade_carteira']:.1%}"}

    g = StateGraph(CommitteeState)
    for name, fn in [("coletar", coletar), ("analista_dados", analista_dados), ("analista_risco", analista_risco), ("gestor", gestor),
                     ("validador", validador), ("fallback", fallback), ("relatorio", relatorio)]:
        g.add_node(name, fn)
    g.add_edge(START, "coletar")
    g.add_edge("coletar", "analista_dados")
    g.add_edge("coletar", "analista_risco")
    g.add_edge(["analista_dados", "analista_risco"], "gestor")
    g.add_edge("gestor", "validador")
    g.add_conditional_edges("validador", route, {"relatorio": "relatorio", "gestor": "gestor", "fallback": "fallback"})
    g.add_edge("fallback", "relatorio")
    g.add_edge("relatorio", END)
    return g.compile()
