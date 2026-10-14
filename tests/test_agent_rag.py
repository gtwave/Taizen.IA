import unittest
from typing import Any
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tests import helpers  # noqa: F401
from src import agent_rag


class FakeToolModel(BaseChatModel):
    """LLM simulado: devolve, em ordem, as mensagens do roteiro (pedidos de ferramenta e resposta final)."""

    script: list[Any]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def ask_tools_then_answer():
    calls = AIMessage(content="", tool_calls=[
        {"name": "search_news", "args": {"query": "Vale notícias", "ticker": "VALE3"}, "id": "c1", "type": "tool_call"},
        {"name": "get_stock_quote", "args": {"ticker": "VALE3"}, "id": "c2", "type": "tool_call"},
    ], usage_metadata={"input_tokens": 300, "output_tokens": 40, "total_tokens": 340})
    final = AIMessage(content="A Vale esclareceu X [1]. Cotação R$ 67,90.",
                      usage_metadata={"input_tokens": 700, "output_tokens": 60, "total_tokens": 760})
    return [calls, final]


class TestAgentRag(unittest.TestCase):
    @patch("src.agent_rag._search_news", return_value={"resultados": [{"n": 1, "titulo": "t"}]})
    @patch("src.agent_rag._get_stock_quote", return_value={"symbol": "VALE3", "price": 67.9})
    def test_agent_calls_both_tools_and_reports_usage(self, quote, news):
        agent = agent_rag.build_agent(FakeToolModel(script=ask_tools_then_answer()))
        logs = []
        text, usage = agent_rag.run("Notícias e cotação da Vale?", agent=agent, log=logs.append)
        self.assertIn("A Vale esclareceu", text)
        news.assert_called_once_with("Vale notícias", ticker="VALE3", days=None)
        quote.assert_called_once_with("VALE3")
        self.assertEqual(usage, {"input_tokens": 1000, "output_tokens": 100, "model_calls": 2})
        trace = "\n".join(logs)
        self.assertIn("[tool_use] search_news", trace)
        self.assertIn("[tool_result]", trace)

    @patch("src.agent_rag._get_stock_quote", return_value={"symbol": "VALE3", "price": 67.9})
    def test_step_limit_stops_runaway_loop(self, quote):
        looping = [AIMessage(content="", tool_calls=[
            {"name": "get_stock_quote", "args": {"ticker": "VALE3"}, "id": f"c{n}", "type": "tool_call"}])
            for n in range(40)]  # ids únicos, como em um modelo real
        agent = agent_rag.build_agent(FakeToolModel(script=looping))
        text, usage = agent_rag.run("loop infinito", agent=agent, log=lambda *_: None)
        self.assertIn("excedeu o limite", text)
        self.assertLessEqual(quote.call_count, agent_rag.MAX_STEPS)


if __name__ == "__main__":
    unittest.main()
