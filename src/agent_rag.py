"""
Fase 1 — Agente RAG (LangChain): notícias recuperadas do índice local + cotação ao vivo.

Uso:  python src/agent_rag.py "O que saiu de notícia sobre a Vale e como está a cotação?"
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
from src.tools.news_search import search_news as _search_news
from src.tools.stock_quote import get_stock_quote as _get_stock_quote

load_dotenv()

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_STEPS = 12  # teto do laço (passos do grafo): modelo + ferramentas contam como passos

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


def run(question: str, agent=None, log=print) -> tuple[str, dict]:
    """Executa o agente, imprime o rastro (tool_use/tool_result) e devolve (resposta, uso de tokens)."""
    from langgraph.errors import GraphRecursionError

    agent = agent or build_agent()
    log(f"\n>>> Pergunta: {question}\n")
    usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}
    final = ""
    try:
        for update in agent.stream({"messages": [{"role": "user", "content": question}]},
                                   config={"recursion_limit": MAX_STEPS}, stream_mode="updates"):
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
        final = f"Interrompido: o agente excedeu o limite de {MAX_STEPS} passos sem concluir."
    log(f">>> Resposta:\n{final}\n")
    log(f"[uso] chamadas ao modelo={usage['model_calls']} | entrada={usage['input_tokens']} | saída={usage['output_tokens']} tokens")
    return final, usage


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "O que saiu de notícia sobre a Vale e como está a cotação hoje?"
    run(question)
