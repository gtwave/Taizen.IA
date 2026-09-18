"""
Fase 0 — MVP de function calling.
Pergunta em linguagem natural -> LLM decide chamar get_stock_quote -> dado real da B3 -> resposta final.
"""

import os
import sys
import json

from dotenv import load_dotenv
import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.stock_quote import get_stock_quote

load_dotenv()

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Retorna a cotação atual de uma ação da B3 dado seu ticker (ex: PETR4, VALE3, ITUB4).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker da ação na B3, ex: PETR4"}
            },
            "required": ["ticker"],
        },
    }
]

SYSTEM_PROMPT = (
    "Você é um assistente de análise financeira focado em ações da B3. "
    "Sempre que precisar de uma cotação real, use a ferramenta get_stock_quote — "
    "nunca invente preços ou dados de mercado. Responda de forma direta e objetiva."
)


def run(question: str) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    print(f"\n>>> Pergunta: {question}\n")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(block.text for block in response.content if block.type == "text")
            print(f">>> Resposta:\n{final_text}\n")
            return final_text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[tool_use] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                if block.name == "get_stock_quote":
                    result = get_stock_quote(**block.input)
                    print(f"[tool_result] {json.dumps(result, ensure_ascii=False)}\n")
                else:
                    result = {"error": f"Ferramenta desconhecida: {block.name}"}
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Qual a cotação atual da PETR4 e o que você acha dela hoje?"
    run(question)
