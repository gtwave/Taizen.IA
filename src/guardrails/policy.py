"""A camada de guardrails: política, decisões, auditoria e integração com o tracing.

Três pontos de controle, cada um com um modo:
  entrada   — a pergunta do usuário: mascara dados pessoais e segredos; barra injeção direta clara;
  ferramenta— o texto que as ferramentas/RAG devolvem (o vetor de injeção INDIRETA): saneia, delimita e quarentena;
  saída     — a resposta final: links não autorizados, segredos, dados pessoais, recomendação de investimento.

Modos: `enforce` (age), `monitor` (só registra o que faria — ideal para medir falsos positivos antes de ligar) e `off`.
Toda decisão vira um span (`kind="guard"`) e uma linha de auditoria SEM o conteúdo original (só tipos, nível e fingerprint).
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DATA_DIR
from src.guardrails import injection, output, pii
from src.observability.tracing import get_tracer

AUDIT_FILE = DATA_DIR / "processed" / "guard_audit.jsonl"
REFUSAL_INPUT = "Não posso processar este pedido: ele contém instruções que tentam alterar as regras do sistema. Reformule a pergunta."


META_KEYS = frozenset({"url", "data", "fonte", "n", "ticker", "symbol"})


class _Quarantined(Exception):
    """Um texto de ferramenta foi posto em quarentena: o item que o contém deve ser descartado."""


@dataclass
class GuardConfig:
    mode: str = "enforce"            # enforce | monitor | off
    mask_pii: bool = True
    mask_cnpj: bool = False          # CNPJ de empresa é público; ative se tratar CNPJ de clientes
    block_input_level: str = "high"  # injeção direta: nível mínimo para recusar a pergunta
    quarantine_level: str = "high"   # conteúdo de ferramenta: nível mínimo para descartar o item
    wrap_tool_content: bool = True
    min_wrap_chars: int = 80
    stage_input: bool = True         # liga/desliga cada ponto de controle (útil para medir a contribuição de cada camada)
    stage_tool: bool = True
    stage_output: bool = True
    allowed_hosts: frozenset = frozenset()
    forbid_advice: bool = True
    required_text: str | None = None


@dataclass
class Decision:
    stage: str                       # input | tool | output
    action: str                      # allow | sanitize | block | monitor
    level: str = "low"
    rules: list[str] = field(default_factory=list)
    pii_kinds: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def intervened(self) -> bool:
        return self.action in ("sanitize", "block")


class Guard:
    def __init__(self, config: GuardConfig | None = None, audit_path: Path | None = AUDIT_FILE, tracer=None):
        self.config = config or GuardConfig()
        self.audit_path = Path(audit_path) if audit_path else None
        self.tracer = tracer or get_tracer()
        self.decisions: list[Decision] = []
        self._lock = threading.Lock()
        self.pii_mapping: dict[str, str] = {}

    # ------------------------------------------------------------------ infraestrutura
    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def _record(self, d: Decision, sample: str = "") -> Decision:
        if self.config.mode == "monitor" and d.action in ("sanitize", "block"):
            d = Decision(d.stage, "monitor", d.level, d.rules, d.pii_kinds, d.detail + " (monitor: nada foi alterado)")
        with self._lock:
            self.decisions.append(d)
        with self.tracer.span(f"guard:{d.stage}", "guard", action=d.action, level=d.level, rules=d.rules, pii=d.pii_kinds) as sp:
            sp.set(detail=d.detail)
        if self.audit_path:
            line = {"ts": round(time.time(), 3), "stage": d.stage, "action": d.action, "level": d.level, "rules": d.rules,
                    "pii": d.pii_kinds, "detail": d.detail, "content_hash": hashlib.sha256(sample.encode("utf-8")).hexdigest()[:10] if sample else ""}
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self._lock, open(self.audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            except OSError:
                pass  # auditoria não pode derrubar a aplicação
        return d

    def summary(self) -> dict:
        out = {"total": len(self.decisions), "intervened": 0, "by_action": {}, "by_rule": {}}
        for d in self.decisions:
            out["by_action"][d.action] = out["by_action"].get(d.action, 0) + 1
            out["intervened"] += d.intervened
            for r in d.rules + d.pii_kinds:
                out["by_rule"][r] = out["by_rule"].get(r, 0) + 1
        return out

    # ------------------------------------------------------------------ entrada
    def check_input(self, text: str) -> tuple[str, Decision]:
        """Devolve (texto a enviar ao modelo, decisão). Em `block`, o texto é uma recusa e NÃO deve ir ao modelo."""
        if not self.enabled or not self.config.stage_input:
            return text, Decision("input", "allow")
        cfg = self.config
        scan = injection.scan(text)
        rules = sorted({f.rule for f in scan.findings})
        if scan.at_least(cfg.block_input_level):
            d = self._record(Decision("input", "block", scan.level, rules, [], f"injeção direta (pontuação {scan.score})"), text)
            return (REFUSAL_INPUT if d.action == "block" else text), d
        out, kinds = text, []
        if cfg.mask_pii:
            m = pii.mask(text, cfg.mask_cnpj, reversible=True)
            if m.findings:
                kinds, out = m.kinds, m.text
                self.pii_mapping.update(m.mapping)
        if kinds or scan.level != "low":
            action = "sanitize" if kinds else "allow"
            d = self._record(Decision("input", action, scan.level, rules, kinds, "dados pessoais mascarados" if kinds else "sinal de injeção abaixo do limite"), text)
            return (out if d.action == "sanitize" else text), d
        return text, Decision("input", "allow")

    # ------------------------------------------------------------------ ferramenta (conteúdo não confiável)
    def check_untrusted(self, text: str, source: str = "externo") -> tuple[str, Decision]:
        """Saneia e delimita um texto de terceiros. `action == "block"` significa quarentena: o texto NÃO deve ser usado."""
        if not self.enabled or not self.config.stage_tool:
            return text, Decision("tool", "allow")
        cfg = self.config
        scan = injection.scan(text)
        rules = sorted({f.rule for f in scan.findings})
        if scan.at_least(cfg.quarantine_level):
            d = self._record(Decision("tool", "block", scan.level, rules, [], f"conteúdo de '{source}' em quarentena (pontuação {scan.score})"), text)
            return ("" if d.action == "block" else text), d
        clean = injection.sanitize(text) if scan.level != "low" or scan.invisible_chars else text
        kinds = []
        if cfg.mask_pii:
            m = pii.mask(clean, cfg.mask_cnpj)
            if m.findings:
                clean, kinds = m.text, m.kinds
        changed = clean != text
        d = Decision("tool", "sanitize" if changed else "allow", scan.level, rules, kinds, f"fonte '{source}'" if changed else "")
        if changed or scan.level != "low":
            d = self._record(d, text)
        use = text if d.action == "monitor" else clean
        wrap = cfg.wrap_tool_content and len(use) >= cfg.min_wrap_chars  # só trechos longos: títulos curtos não compensam o custo em tokens
        return (injection.wrap_untrusted(use, source) if wrap else use), d

    def guard_tool_result(self, result, source: str = "ferramenta"):
        """Percorre um resultado de ferramenta (dict/lista/str) e trata cada texto como não confiável.
        Itens de lista em quarentena são descartados e um aviso é acrescentado ao resultado."""
        if not self.enabled or not self.config.stage_tool:
            return result
        dropped = [0]

        def walk(x, key=""):
            if isinstance(x, str):
                if key in META_KEYS:  # metadados curtos (url, data, fonte): só remove caracteres invisíveis
                    return injection.sanitize(x)
                text, d = self.check_untrusted(x, f"{source}.{key}" if key else source)
                if d.action == "block":
                    raise _Quarantined()
                return text
            if isinstance(x, list):
                out = []
                for item in x:
                    try:
                        out.append(walk(item, key))
                    except _Quarantined:
                        dropped[0] += 1
                return out
            if isinstance(x, dict):
                return {k: walk(v, k) for k, v in x.items()}
            return x

        try:
            res = walk(result)
        except _Quarantined:  # o próprio resultado inteiro era um texto malicioso
            return {"aviso_seguranca": "resultado removido por conter possíveis instruções maliciosas."}
        if dropped[0] and isinstance(res, dict):
            res = {**res, "aviso_seguranca": f"{dropped[0]} item(ns) removido(s) por conterem possíveis instruções maliciosas."}
        return res

    # ------------------------------------------------------------------ saída
    def check_output(self, text: str, tool_texts: list[str] | tuple[str, ...] = ()) -> tuple[str, Decision]:
        if not self.enabled or not self.config.stage_output:
            return text, Decision("output", "allow")
        cfg = self.config
        v = output.check_output(text, tool_texts=tool_texts, allowed_hosts=cfg.allowed_hosts, forbid_advice=cfg.forbid_advice,
                                required_text=cfg.required_text, mask_cnpj=cfg.mask_cnpj)
        if v.ok:
            return text, Decision("output", "allow")
        severe = {"segredo_na_saida", "recomendacao_de_investimento"} & set(v.rules)
        action = "block" if "recomendacao_de_investimento" in v.rules else "sanitize"
        d = self._record(Decision("output", action, "high" if severe else "medium", v.rules, [], "; ".join(f"{r}: {x}" for r, x in v.violations)), text)
        if d.action == "block":
            return ("Não posso apresentar indicações de investimento. Posso resumir os fatos, o contexto e os riscos "
                    "com base nas fontes consultadas."), d
        return (v.text if d.action == "sanitize" else text), d
