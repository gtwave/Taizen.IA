"""Detecção e mascaramento de dados pessoais e segredos (PII masking), com validação por dígitos verificadores.

Princípio de minimização (LGPD, art. 6º): o que não precisa chegar a um provedor de LLM não deve chegar. Os padrões de
CPF, CNPJ e cartão são confirmados por checksum (CPF/CNPJ por módulo 11, cartão por Luhn): isso derruba a maioria dos
falsos positivos (um preço como 12.345.678-90 não vira "CPF"). O CNPJ de empresas é informação pública, por isso o
mascaramento de CNPJ é opcional (`mask_cnpj`).

Os achados NÃO guardam o valor original: só tipo, posição e um fingerprint curto (hash com sal) para auditoria. O mapa
placeholder → valor só existe quando `reversible=True` e vive na memória do chamador.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field

_SALT = os.environ.get("GUARD_SALT", "taizen-guard")


def fingerprint(value: str) -> str:
    return hashlib.sha256((_SALT + value).encode("utf-8")).hexdigest()[:10]


# ----------------------------------------------------------------------------- validadores
def valid_cpf(digits: str) -> bool:
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for n in (9, 10):
        s = sum(int(d) * w for d, w in zip(digits[:n], range(n + 1, 1, -1)))
        if int(digits[n]) != (s * 10 % 11) % 10:
            return False
    return True


def valid_cnpj(digits: str) -> bool:
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    for n, weights in ((12, [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]), (13, [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])):
        r = sum(int(d) * w for d, w in zip(digits[:n], weights)) % 11
        if int(digits[n]) != (0 if r < 2 else 11 - r):
            return False
    return True


def luhn(digits: str) -> bool:
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


# ----------------------------------------------------------------------------- padrões
_ONLY_DIGITS = re.compile(r"\D")
PATTERNS: list[tuple[str, int, re.Pattern, object]] = [
    # (tipo, prioridade — menor vence, regex, validador do grupo 0 ou None)
    ("SEGREDO", 0, re.compile(r"\b(?:sk-ant-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9\-]{10,})"), None),
    ("SEGREDO", 0, re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}"), None),
    ("SEGREDO", 0, re.compile(r"(?i)(?<![a-z])(?:api[_\- ]?key|apikey|token|secret|senha|password)\s*[:=]\s*[\"']?[^\s\"']{8,}"), None),
    ("CARTAO", 1, re.compile(r"(?<![\d.+])(?:\d[ \-]?){12,18}\d(?!\d|\.\d)"), lambda s: luhn(_ONLY_DIGITS.sub("", s))),
    ("CPF", 2, re.compile(r"(?<![\d.\-/])\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?![\d\-/]|\.\d)"), lambda s: valid_cpf(_ONLY_DIGITS.sub("", s))),
    ("CNPJ", 3, re.compile(r"(?<![\d.\-/])\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?![\d\-/]|\.\d)"), lambda s: valid_cnpj(_ONLY_DIGITS.sub("", s))),
    # tamanhos limitados (RFC 5321: parte local ≤ 64, rótulo ≤ 63): sem isso, texto hostil como "1.1.1…" torna a busca quadrática (ReDoS)
    ("EMAIL", 4, re.compile(r"(?i)\b[a-z0-9._%+\-]{1,64}@[a-z0-9\-]{1,63}(?:\.[a-z0-9\-]{1,63}){0,5}\.[a-z]{2,24}\b"), None),
    ("TELEFONE", 5, re.compile(r"(?<![\d.])(?:\+55[ \-]?)?\(?[1-9]\d\)?[ \-]?9\d{4}[ \-]?\d{4}(?!\d|\.\d)"), None),
    ("TELEFONE", 5, re.compile(r"(?<![\d.])(?:\+55[ \-]?)?\([1-9]\d\)[ \-]?[2-5]\d{3}[ \-]?\d{4}(?!\d|\.\d)"), None),
]


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    placeholder: str
    fp: str  # fingerprint do valor (hash com sal), nunca o valor


@dataclass
class MaskResult:
    text: str
    findings: list[Finding] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original (só se reversible=True)

    @property
    def kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})


def detect(text: str, mask_cnpj: bool = False) -> list[tuple[str, int, int, str]]:
    """Todas as ocorrências validadas, sem sobreposição (vence a de menor prioridade numérica e, no empate, a mais longa)."""
    raw = []
    for kind, prio, rx, check in PATTERNS:
        if kind == "CNPJ" and not mask_cnpj:
            continue
        for m in rx.finditer(text):
            if check is None or check(m.group(0)):
                raw.append((prio, -(m.end() - m.start()), m.start(), m.end(), kind, m.group(0)))
    raw.sort()
    taken: list[tuple[int, int]] = []
    out = []
    for prio, _, s, e, kind, val in raw:
        if any(s < te and e > ts for ts, te in taken):
            continue
        taken.append((s, e))
        out.append((kind, s, e, val))
    return sorted(out, key=lambda x: x[1])


def mask(text: str, mask_cnpj: bool = False, reversible: bool = False) -> MaskResult:
    """Troca cada dado sensível por um marcador estável (`[CPF_1]`): o mesmo valor recebe sempre o mesmo marcador."""
    found = detect(text, mask_cnpj)
    if not found:
        return MaskResult(text)
    counters: dict[str, int] = {}
    seen: dict[tuple[str, str], str] = {}
    findings, mapping, parts, last = [], {}, [], 0
    for kind, s, e, val in found:
        key = (kind, _ONLY_DIGITS.sub("", val) if kind in ("CPF", "CNPJ", "CARTAO") else val.lower())
        if key not in seen:
            counters[kind] = counters.get(kind, 0) + 1
            seen[key] = f"[{kind}_{counters[kind]}]"
            if reversible:
                mapping[seen[key]] = val
        ph = seen[key]
        findings.append(Finding(kind, s, e, ph, fingerprint(val)))
        parts += [text[last:s], ph]
        last = e
    parts.append(text[last:])
    return MaskResult("".join(parts), findings, mapping)


def unmask(text: str, mapping: dict[str, str]) -> str:
    for ph, val in mapping.items():
        text = text.replace(ph, val)
    return text


def has_secrets(text: str) -> bool:
    return any(k == "SEGREDO" for k, *_ in detect(text))
