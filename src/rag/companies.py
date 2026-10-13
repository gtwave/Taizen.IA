"""Detecção heurística de empresas da B3 citadas em um texto (marcação de tickers).

É uma heurística simples (regex), não um NER: serve para criar metadados filtráveis e tem limites conhecidos.
Duas defesas contra falso positivo: fronteira de palavra (\\b) e exigência de menção no título ou em >= 2 pontos.
"""

import re

# ticker -> lista de (padrão regex, ignorar_maiusculas)
# Nomes ambíguos ("Vale", "WEG") só casam com a grafia exata e em contexto de empresa.
_SPEC = {
    "PETR4": [(r"\b(?:Petrobras|Petróleo Brasileiro|PETR[34])\b", True)],
    "VALE3": [(r"\bVALE3\b|(?:\b(?:a|da|na|pela|à|A|Da|Na|Pela)\s+Vale|\bVale\s+\(VALE3\)|\bVale S\.?A\.?)(?![\wÀ-ú])", False)],
    "ITUB4": [(r"\b(?:Itaú|Itau Unibanco|ITUB[34])\b", True)],
    "BBDC4": [(r"\b(?:Bradesco|BBDC[34])\b", True)],
    "BBAS3": [(r"\b(?:Banco do Brasil|BBAS3)\b", True)],
    "ABEV3": [(r"\b(?:Ambev|ABEV3)\b", True)],
    "WEGE3": [(r"\bWEG\b|\bWEGE3\b", False)],
    "MGLU3": [(r"\b(?:Magazine Luiza|Magalu|MGLU3)\b", True)],
    "SUZB3": [(r"\b(?:Suzano|SUZB3)\b", True)],
    "RENT3": [(r"\b(?:Localiza|RENT3)\b", True)],
    "SANB11": [(r"\b(?:Santander|SANB11)\b", True)],
    "JBSS3": [(r"\bJBS\b|\bJBSS3\b", False)],
    "EMBR3": [(r"\b(?:Embraer|EMBR3)\b", True)],
    "ELET3": [(r"\b(?:Eletrobras|Axia Energia|ELET[36])\b", True)],
}

TICKERS = list(_SPEC)

_COMPILED = {
    ticker: [re.compile(pat, re.IGNORECASE if ignore_case else 0) for pat, ignore_case in specs]
    for ticker, specs in _SPEC.items()
}


def count_mentions(text: str) -> dict[str, int]:
    """Número de menções a cada empresa (só as que aparecem)."""
    out = {}
    for ticker, pats in _COMPILED.items():
        n = sum(len(p.findall(text)) for p in pats)
        if n:
            out[ticker] = n
    return out


def find_tickers(text: str, title: str = "", min_mentions: int = 2) -> list[str]:
    """Empresas citadas no título ou com pelo menos `min_mentions` menções no corpo do texto."""
    in_title = set(count_mentions(title))
    body = count_mentions(text)
    return [t for t in TICKERS if t in in_title or body.get(t, 0) >= min_mentions]
