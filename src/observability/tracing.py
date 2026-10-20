"""Rastreamento (tracing) leve e local: spans hierárquicos gravados em JSONL.

Um *trace* é a história de uma execução; um *span* é um trecho cronometrado dentro dela (um nó do grafo, uma chamada
ao LLM, uma ferramenta). Spans têm pai, o que permite reconstruir a árvore depois. O modelo de dados segue a ideia do
OpenTelemetry, sem a dependência: ids, pai, nome, tipo, duração, status e atributos.

Por padrão o tracer global não tem destinos (não grava nada); os pontos de entrada (CLIs) o configuram com `configure`.
"""

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("taizen_current_span", default=None)


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str  # chain | node | llm | tool | cache | eval
    start: float  # epoch (s)
    duration_ms: float = 0.0
    status: str = "ok"  # ok | error
    error: str | None = None
    attrs: dict = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False, compare=False)

    def set(self, **attrs) -> "Span":
        self.attrs.update(attrs)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        d["duration_ms"] = round(self.duration_ms, 2)
        return d


class MemorySink:
    """Guarda os spans em memória (testes e demonstrações)."""

    def __init__(self):
        self.spans: list[dict] = []
        self._lock = threading.Lock()

    def emit(self, span: dict) -> None:
        with self._lock:
            self.spans.append(span)


class JsonlSink:
    """Um arquivo por dia (`traces-AAAAMMDD.jsonl`) em `directory`; uma linha por span."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def emit(self, span: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        day = datetime.fromtimestamp(span["start"], timezone.utc).strftime("%Y%m%d")
        line = json.dumps(span, ensure_ascii=False, default=str)
        with self._lock, open(self.directory / f"traces-{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Tracer:
    def __init__(self, sinks: list | None = None):
        self.sinks = list(sinks or [])

    @property
    def enabled(self) -> bool:
        return bool(self.sinks)

    def start_span(self, name: str, kind: str = "chain", parent: Span | None = None, **attrs) -> Span:
        parent = parent if parent is not None else _current.get()
        now = time.time()
        span = Span(trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16], span_id=uuid.uuid4().hex[:16],
                    parent_id=parent.span_id if parent else None, name=name, kind=kind, start=now, attrs=dict(attrs))
        span._t0 = time.perf_counter()
        return span

    def end_span(self, span: Span, error: BaseException | str | None = None) -> None:
        span.duration_ms = (time.perf_counter() - span._t0) * 1000
        if error is not None:
            span.status, span.error = "error", f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        record = span.to_dict()
        for sink in self.sinks:
            sink.emit(record)

    @contextmanager
    def span(self, name: str, kind: str = "chain", **attrs):
        """Cronometra o bloco e grava o span ao sair (com status "error" se o bloco levantar exceção)."""
        sp = self.start_span(name, kind, **attrs)
        token = _current.set(sp)
        try:
            yield sp
        except BaseException as exc:
            self.end_span(sp, error=exc)
            raise
        else:
            self.end_span(sp)
        finally:
            _current.reset(token)


_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def configure(sinks: list | None = None) -> Tracer:
    """Define os destinos do tracer global (lista vazia = desligado)."""
    _tracer.sinks = list(sinks or [])
    return _tracer


def current_span() -> Span | None:
    return _current.get()


def traced(name: str | None = None, kind: str = "chain"):
    """Decorador: cada chamada da função vira um span."""
    def deco(fn):
        def wrapper(*a, **kw):
            with get_tracer().span(name or fn.__name__, kind):
                return fn(*a, **kw)
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


def read_spans(directory: Path, since: float | None = None) -> list[dict]:
    """Lê todos os spans gravados (opcionalmente só os iniciados depois de `since`, em epoch)."""
    out = []
    for f in sorted(Path(directory).glob("traces-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue  # linha truncada por uma execução interrompida
            if since is None or s["start"] >= since:
                out.append(s)
    return out
