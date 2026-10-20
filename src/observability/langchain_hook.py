"""Callback do LangChain que transforma chamadas de LLM e de ferramentas em spans.

Funciona com qualquer modelo de chat do LangChain (ChatAnthropic, modelos simulados) e com agentes/grafos: basta passar
`config={"callbacks": [TracingCallbackHandler()]}` ou `llm.with_config(callbacks=[...])`. Por padrão NÃO grava o texto
dos prompts e respostas (privacidade e tamanho): só tamanhos e um hash curto, suficiente para depurar repetições.
"""

import hashlib
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from src.observability import pricing
from src.observability.tracing import Span, Tracer, get_tracer


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _flatten(messages) -> str:
    parts = []
    for batch in messages:
        for m in batch:
            c = getattr(m, "content", m)
            parts.append(c if isinstance(c, str) else str(c))
    return "\n".join(parts)


def _model_name(serialized, kwargs) -> str:
    inv = kwargs.get("invocation_params") or {}
    meta = kwargs.get("metadata") or {}
    for v in (inv.get("model"), inv.get("model_name"), meta.get("ls_model_name"),
              (serialized or {}).get("kwargs", {}).get("model")):
        if v:
            return str(v)
    return "desconhecido"


class TracingCallbackHandler(BaseCallbackHandler):
    raise_error = False

    def __init__(self, tracer: Tracer | None = None, capture_content: bool = False, role: str | None = None):
        self.tracer = tracer or get_tracer()
        self.capture_content = capture_content
        self.role = role
        self._spans: dict[UUID, Span] = {}

    # ------------------------------------------------------------------ início
    def _begin(self, run_id, parent_run_id, name, kind, **attrs):
        parent = self._spans.get(parent_run_id) if parent_run_id else None
        span = self.tracer.start_span(name, kind, parent=parent, **attrs)
        self._spans[run_id] = span
        return span

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        model = _model_name(serialized, kwargs)
        text = _flatten(messages)
        attrs = {"model": model, "input_chars": len(text), "prompt_hash": _short_hash(text)}
        if self.role:
            attrs["role"] = self.role
        if self.capture_content:
            attrs["prompt"] = text
        self._begin(run_id, parent_run_id, f"llm:{model}", "llm", **attrs)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        model = _model_name(serialized, kwargs)
        text = "\n".join(prompts)
        self._begin(run_id, parent_run_id, f"llm:{model}", "llm", model=model, input_chars=len(text), prompt_hash=_short_hash(text))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        attrs = {"input_chars": len(str(input_str))}
        if self.capture_content:
            attrs["input"] = str(input_str)
        self._begin(run_id, parent_run_id, f"tool:{name}", "tool", **attrs)

    # ------------------------------------------------------------------ fim
    def on_llm_end(self, response, *, run_id, **kwargs):
        span = self._spans.pop(run_id, None)
        if span is None:
            return
        usage, text = {}, ""
        for gen_list in response.generations:
            for gen in gen_list:
                text += getattr(gen, "text", "") or ""
                msg = getattr(gen, "message", None)
                um = getattr(msg, "usage_metadata", None)
                if um:
                    usage = um
        if not usage:  # alguns provedores só informam em llm_output
            tu = (response.llm_output or {}).get("token_usage") or {}
            usage = {"input_tokens": tu.get("prompt_tokens", 0), "output_tokens": tu.get("completion_tokens", 0)}
        details = usage.get("input_token_details") or {}
        tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        cread, cwrite = int(details.get("cache_read", 0)), int(details.get("cache_creation", 0))
        model = span.attrs.get("model")
        cb = pricing.cost_brl(model, tin, tout, cread, cwrite)
        span.set(input_tokens=tin, output_tokens=tout, cache_read_tokens=cread, cache_write_tokens=cwrite,
                 output_chars=len(text), cost_brl=None if cb is None else round(cb, 6),
                 priced=cb is not None, tokens_estimated=not usage)
        if self.capture_content:
            span.set(output=text)
        self.tracer.end_span(span)

    def on_tool_end(self, output, *, run_id, **kwargs):
        span = self._spans.pop(run_id, None)
        if span is not None:
            text = str(getattr(output, "content", output))
            span.set(output_chars=len(text))
            if self.capture_content:
                span.set(output=text)
            self.tracer.end_span(span)

    def _fail(self, error, run_id):
        span = self._spans.pop(run_id, None)
        if span is not None:
            self.tracer.end_span(span, error=error)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._fail(error, run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._fail(error, run_id)
