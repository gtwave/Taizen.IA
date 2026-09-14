# Roadmap — 7 Fases, 45 dias

Estimativas assumem ~25h/semana (dev + estudo) para caber em 45 dias com alguma folga, dentro da faixa "20h+/semana" informada. Se o ritmo real ficar abaixo disso, as Fases 4 e 5 são as mais fáceis de comprimir ou adiar sem quebrar o sistema funcional.

**Total estimado: 94h de desenvolvimento + 68h de estudo = 162h**

Cada fase entrega três coisas: (1) incremento funcional no sistema, (2) material didático em `docs/`, (3) roteiro(s) de conteúdo em `social/`.

---

## Fase 0 — Fundação (Dias 1-2 · Dev 6h · Estudo 8h)
**Skills:** Prompt Engineering, System Prompts, Chain-of-Thought, Few-Shot Prompting, Function Calling básico, introdução a Token Economics.
**Entregável funcional:** script Python que consulta a cotação de uma ação da B3 via function calling e responde perguntas simples sobre ela.
**Stack:** Python, SDK Anthropic ou OpenAI, brapi.dev.
**Pasta:** `docs/fase-00-fundacao/`

## Fase 1 — MVP RAG + Agente Único (Dias 3-12 · Dev 22h · Estudo 14h)
**Skills:** Vector Databases, Embeddings, Chunking, RAG, Hybrid Search.
**Entregável funcional:** pipeline de ingestão (notícias/relatórios financeiros) → chunking → embeddings locais → Chroma → agente único (LangChain) que responde perguntas sobre ações combinando RAG + function calling em tempo real.
**Stack:** LangChain, Chroma, sentence-transformers (embeddings locais), brapi.dev.

## Fase 2 — Sistema Multi-Agente (Dias 13-19 · Dev 18h · Estudo 12h)
**Skills:** AI Agents / Agentic Workflows, Orquestradores (LangGraph e/ou CrewAI), Tool Use avançado.
**Entregável funcional:** comitê de agentes — Analista de Dados, Analista de Risco, Gestor de Carteira — que colaboram para gerar uma sugestão de carteira recomendada a partir de um conjunto de ações da B3.
**Stack:** LangGraph ou CrewAI (sobre a base LangChain da Fase 1).

## Fase 3 — LLMOps & Observabilidade (Dias 20-25 · Dev 14h · Estudo 10h)
**Skills:** LLMOps/GenAIOps, avaliação de qualidade (LangSmith/Phoenix/DeepEval), Semantic Caching, Token Economics & FinOps.
**Entregável funcional:** instrumentação do sistema da Fase 2 com tracing, métricas de latência/custo/qualidade, cache semântico para reduzir custo de chamadas repetidas.
**Stack:** LangSmith ou Phoenix, DeepEval.

## Fase 4 — Governança & Segurança (Dias 26-29 · Dev 10h · Estudo 8h)
**Skills:** Guardrails, defesa contra Prompt Injection, Data Privacy & PII Masking.
**Entregável funcional:** camada de guardrails nas entradas/saídas dos agentes e no pipeline RAG (validação de output, filtro de prompt injection, mascaramento de dados sensíveis se houver).
**Stack:** Guardrails AI ou implementação própria + regras.

## Fase 5 — Fine-Tuning & SLM (Dias 30-36 · Dev 14h · Estudo 12h)
**Skills:** Fine-Tuning, PEFT/LoRA, Small Language Models. RLHF é tratado só teoricamente (custo de prática real é proibitivo no orçamento disponível).
**Entregável funcional:** um SLM (ex: Llama 3.2 1B/3B ou Phi-3-mini) fine-tunado via LoRA para uma tarefa específica (ex: classificar sentimento de notícias financeiras em PT-BR), treinado em GPU alugada por poucas horas ou Google Colab (camada gratuita) para caber no orçamento de R$50/mês.
**Stack:** Hugging Face `peft`/`transformers`, Ollama para servir o modelo final localmente.

## Fase 6 — Deploy Final, Consolidação & Portfólio (Dias 37-45 · Dev 10h · Estudo 4h)
**Skills:** consolidação de todos os tópicos anteriores, deploy via Docker.
**Entregável funcional:** sistema completo rodando na VPS via `docker-compose` (backend + Chroma + observabilidade leve), documentação final, case de portfólio para aplicar às vagas.
**Stack:** Docker Compose na VPS.

---

## Mapeamento de skills (referência do briefing original)

| Área | Onde é coberta |
|---|---|
| a) Engenharia de Dados e Bancos Vetoriais | Fase 1 |
| b) Arquitetura de Sistemas e Agentes | Fases 1, 2 |
| c) Treinamento e Ajuste Fino | Fase 5 |
| d) Operações e Monitoramento (LLMOps) | Fase 3 |
| e) Governança, Segurança e Custos | Fases 3 (FinOps), 4 (Guardrails/Privacidade) |

Cada fase só é detalhada em `docs/` quando começamos a executá-la — evita planejar demais uma fase distante que pode mudar com o aprendizado das fases anteriores.
