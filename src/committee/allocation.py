"""Alocação determinística: volatilidade inversa + inclinação por score, com tetos, pisos e vetos por perfil.

A ideia central do projeto: o LLM pode opinar e justificar, mas as RESTRIÇÕES (soma 100%, tetos, pisos) são
garantidas por código — por isso existe esta função e o validador do grafo.
"""

import numpy as np

from src.committee.config import Profile


def cap_and_normalize(w: np.ndarray, max_w: float, tol: float = 1e-12) -> np.ndarray:
    """Normaliza para soma 1 respeitando o teto por ativo (redistribui o excedente proporcionalmente)."""
    w = np.asarray(w, dtype=float).copy()
    n = len(w)
    if max_w * n < 1 - tol:
        raise ValueError(f"teto de {max_w:.0%} é inviável para {n} ativos (a soma não chega a 100%)")
    w = w / w.sum()
    for _ in range(n + 1):
        over = w > max_w + tol
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        free = ~over & (w < max_w - tol)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def allocate(vols: np.ndarray, scores: np.ndarray, profile: Profile, vetoed: np.ndarray | None = None) -> np.ndarray:
    """Pesos da carteira (soma 1). `vols` em base anual; `scores` padronizados (z, entre -2 e 2); `vetoed` booleano."""
    vols = np.asarray(vols, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(vols)
    vetoed = np.zeros(n, dtype=bool) if vetoed is None else np.asarray(vetoed, dtype=bool)
    active = ~vetoed
    if active.sum() < profile.min_positions:
        # veto removeria ativos demais: mantém os de menor volatilidade para respeitar o mínimo de posições
        order = np.argsort(vols)
        active = np.zeros(n, dtype=bool)
        active[order[: profile.min_positions]] = True
    raw = np.zeros(n)
    raw[active] = (1.0 / vols[active]) * np.exp(profile.tilt * scores[active])
    w = np.zeros(n)
    k = active.sum()
    cap = max(profile.max_weight, 1.0 / k)  # teto nunca inviabiliza a soma entre os ativos ativos
    w[active] = cap_and_normalize(raw[active], cap)
    # piso: eleva ativos abaixo do piso e retira o necessário dos maiores, respeitando o teto
    if profile.min_weight > 0:
        for _ in range(n):
            low = active & (w < profile.min_weight - 1e-12)
            if not low.any():
                break
            deficit = (profile.min_weight - w[low]).sum()
            w[low] = profile.min_weight
            donors = active & ~low & (w > profile.min_weight + 1e-12)
            if not donors.any():
                break
            room = w[donors] - profile.min_weight
            w[donors] -= deficit * room / room.sum()
    return w / w.sum()
