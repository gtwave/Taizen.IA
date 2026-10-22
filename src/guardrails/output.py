"""Guardrails de SAÍDA: o que o sistema devolve ao usuário também é verificado antes de sair.

Mesmo um modelo bem instruído pode, por erro ou por injeção, devolver: recomendação de compra/venda (proibida pela regra do
agente), links que ninguém forneceu (alucinados ou de exfiltração), segredos e dados pessoais, ou texto que viola o formato.
Cada verificação é código determinístico — não depende do modelo estar "bem comportado".
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from src.guardrails import pii

ADVICE = re.compile(r"\b(compre|comprar|venda|vender|recomendo|recomendamos|vale a pena comprar|oportunidade de compra)\b", re.I)
_URL = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
_IMG_PARAMS = re.compile(r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&=][^)\s]*\)")
LINK_REMOVED = "[link removido]"


@dataclass
class OutputVerdict:
    ok: bool
    violations: list[tuple[str, str]] = field(default_factory=list)  # (regra, detalhe sem dados sensíveis)
    text: str = ""  # versão segura: segredos/PII mascarados, links não autorizados removidos

    @property
    def rules(self) -> list[str]:
        return sorted({r for r, _ in self.violations})


def _host(url: str) -> str:
    return (urlparse(url.rstrip(".,;:")).hostname or "").lower()


def urls_in(text: str) -> set[str]:
    return {u.rstrip(".,;:)") for u in _URL.findall(text)}


def check_output(text: str, *, tool_texts: list[str] | tuple[str, ...] = (), allowed_hosts: set[str] | frozenset[str] = frozenset(),
                 forbid_advice: bool = True, required_text: str | None = None, max_chars: int = 4000,
                 mask_cnpj: bool = False) -> OutputVerdict:
    """Verifica uma resposta. Links só são aceitos se vieram das saídas das ferramentas (`tool_texts`) ou de hosts permitidos."""
    violations: list[tuple[str, str]] = []
    safe = text

    masked = pii.mask(safe, mask_cnpj=mask_cnpj)
    if masked.findings:
        kinds = masked.kinds
        rule = "segredo_na_saida" if "SEGREDO" in kinds else "dado_pessoal_na_saida"
        violations.append((rule, f"tipos: {', '.join(kinds)}"))
        safe = masked.text

    allowed_urls = set()
    for t in tool_texts:
        allowed_urls |= urls_in(t)
    allowed_hosts_all = {h.lower() for h in allowed_hosts} | {_host(u) for u in allowed_urls}
    bad = [u for u in urls_in(safe) if u not in allowed_urls and _host(u) not in allowed_hosts_all]
    if bad:
        violations.append(("url_nao_autorizada", f"{len(bad)} link(s) de host(s): {', '.join(sorted({_host(u) for u in bad}))}"))
        for u in sorted(bad, key=len, reverse=True):
            safe = safe.replace(u, LINK_REMOVED)
    if _IMG_PARAMS.search(safe):
        violations.append(("imagem_com_parametros", "possível exfiltração por imagem remota"))
        safe = _IMG_PARAMS.sub(LINK_REMOVED, safe)

    if forbid_advice and ADVICE.search(safe):
        violations.append(("recomendacao_de_investimento", f"termos: {', '.join(sorted({m.lower() for m in ADVICE.findall(safe)}))}"))
    if required_text and required_text not in safe:
        violations.append(("texto_obrigatorio_ausente", "aviso legal não presente"))
    if len(safe) > max_chars:
        violations.append(("saida_longa_demais", f"{len(safe)} > {max_chars} caracteres"))
        safe = safe[:max_chars] + "…"
    return OutputVerdict(not violations, violations, safe)
