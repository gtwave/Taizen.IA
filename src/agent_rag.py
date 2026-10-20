"""
Fase 1 — Agente RAG (LangChain): notícias recuperadas do índice local + cotação ao vivo.

Uso:  python src/agent_rag.py "O que saiu de notícia sobre a Vale e como está a cotação?"
      python src/agent_rag.py --cache "pergunta"     # cache semântico (Fase 3): perguntas parecidas reaproveitam a resposta
"""

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.observability.langchain_hook import TracingCallbackHandler
from src.observability.tracing import get_tracer
from src.tools.news_search import search_news as _search_news
from src.tools.stock_quote import get_stock_quote as _get_stock_quote

load_dotenv()

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_STEPS = 12  # teto do laço (passos do grafo): modelo + ferramentas contam como passos
ANSWER_TTL_S = 900  # a resposta cita cotação ao vivo: no cache, vale 15 minutos

SYSTEM_PROMPT = """Você é um analista de mercado focado em ações da B3.

Ferramentas:
- get_stock_quote: cotação e indicadores atuais de um ticker.
- search_news: trechos de notícias recentes recuperados de um índice local.

Regras:
1. Para preço ou indicador atual, use get_stock_quote. Para contexto e acontecimentos, use search_news.
2. Responda apenas com base no que as ferramentas devolveram. Nunca invente números, datas ou fatos.
3. Cite as notícias usadas como [1], [2]... e liste as URLs no final.
4. Se search_news não trouxer nada relevante, diga que não encontrou notícias sobre o assunto.
5. Não recomende compra ou venda: apresente fatos, contexto e riscos.
6. Responda em português, em até 8 linhas."""


@tool("get_stock_quote")
def get_stock_quote_tool(ticker: str) -> dict:
    """Retorna a cotação atual de uma ação da B3 dado seu ticker (ex: PETR4, VALE3, ITUB4)."""
    return _get_stock_quote(ticker)


@tool("search_news")
def search_news_tool(query: str, ticker: str | None = None, days: int | None = None) -> dict:
    """Busca trechos de notícias financeiras recentes. Use `ticker` (ex: VALE3) para filtrar por empresa
    e `days` para limitar aos últimos N dias. Devolve fonte, título, data, URL e um trecho de cada notícia."""
    return _search_news(query, ticker=ticker, days=days)


def build_agent(model=None):
    llm = model or ChatAnthropic(model=MODEL, max_tokens=1024)
    return create_agent(llm, [get_stock_quote_tool, search_news_tool], system_prompt=SYSTEM_PROMPT)


def run(question: str, agent=None, log=print, cache=None) -> tuple[str, dict]:
    """Executa o agente, imprime o rastro (tool_use/tool_result) e devolve (resposta, uso de tokens).

    Com `cache` (SemanticCache), perguntas iguais ou parecidas (e com os mesmos tickers) reaproveitam a resposta."""
    with get_tracer().span("agent_rag.run", "chain", question_chars=len(question)) as root:
        return _run(question, agent, log, cache, root)


def _run(question, agent, log, cache, root) -> tuple[str, dict]:
    from langgraph.errors import GraphRecursionError

    usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
    guard = ""
    if cache is not None:
        from src.observability.cache import question_guard

        guard = question_guard(question)
        hit = cache.get(question, namespace=MODEL, guard=guard)
        if hit is not None:
            usage["cache"] = hit.kind
            root.set(cache=hit.kind)
            log(f"\n>>> Pergunta: {question}\n[cache {hit.kind}] similaridade {hit.similarity:.3f}, resposta de {hit.age_s:.0f}s atrás "
                f"(pergunta original: {hit.matched_query!r})\n>>> Resposta:\n{hit.answer}\n")
            return hit.answer, usage
        root.set(cache="miss")
    agent = agent or build_agent()
    log(f"\n>>> Pergunta: {question}\n")
    final, interrupted = "", False
    try:
        for update in agent.stream({"messages": [{"role": "user", "content": question}]},
                                   config={"recursion_limit": MAX_STEPS, "callbacks": [TracingCallbackHandler()]}, stream_mode="updates"):
            for node, data in update.items():
                for msg in data.get("messages", []):
                    if node == "model":
                        usage["model_calls"] += 1
                        um = getattr(msg, "usage_metadata", None) or {}
                        usage["input_tokens"] += um.get("input_tokens", 0)
                        usage["output_tokens"] += um.get("output_tokens", 0)
                        for call in getattr(msg, "tool_calls", []) or []:
                            log(f"[tool_use] {call['name']}({call['args']})")
                        if not getattr(msg, "tool_calls", None):
                            final = msg.text if hasattr(msg, "text") else str(msg.content)
                    elif node == "tools":
                        log(f"[tool_result] {str(msg.content)[:300]}...\n")
    except GraphRecursionError:
        final, interrupted = f"Interrompido: o agente excedeu o limite de {MAX_STEPS} passos sem concluir.", True
    if cache is not None and final and not interrupted:
        cache.put(question, final, namespace=MODEL, guard=guard, ttl_s=ANSWER_TTL_S)  # nunca se guarda uma resposta interrompida
    log(f">>> Resposta:\n{final}\n")
    log(f"[uso] chamadas ao modelo={usage['model_calls']} | entrada={usage['input_tokens']} | saída={usage['output_tokens']} tokens")
    return final, usage


if __name__ == "__main__":
    args = sys.argv[1:]
    use_cache = "--cache" in args
    question = " ".join(a for a in args if a != "--cache") or "O que saiu de notícia sobre a Vale e como está a cotação hoje?"
    from src.config import DATA_DIR
    from src.observability import pricing
    from src.observability.report import TRACES_DIR
    from src.observability.tracing import JsonlSink, configure, read_spans

    configure([JsonlSink(TRACES_DIR)])
    pricing.assert_within_budget(read_spans(TRACES_DIR))  # FinOps: recusa a chamada paga se o orçamento do mês acabou
    semantic = None
    if use_cache:
        from src.observability.cache import SemanticCache
        from src.rag.embeddings import E5Embeddings

        semantic = SemanticCache(E5Embeddings(), DATA_DIR / "processed" / "semantic_cache.json")
    run(question, cache=semantic)
