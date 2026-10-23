# src/

Código do projeto, evoluindo fase a fase (ver `../ROADMAP.md`).

| Caminho | Fase | O que faz |
|---|---|---|
| `agent.py` | 0 | Laço de function calling com a API da Anthropic e a ferramenta `get_stock_quote` |
| `agent_rag.py` | 1 | Agente LangChain (`create_agent`) com `search_news` (RAG) e `get_stock_quote`; teto de passos e rastro de uso de tokens |
| `config.py` | 1 | Caminhos (`DATA_DIR`) e parâmetros do pipeline (modelo de embeddings, tamanho de chunk) |
| `rag/feeds.py` | 1 | Coleta de feeds RSS (robots.txt, User-Agent identificado, intervalo entre requisições) |
| `rag/ingest.py` | 1 | Normalização, deduplicação e gravação do acervo em JSONL |
| `rag/companies.py` | 1 | Detecção heurística de empresas (tickers) citadas em cada notícia |
| `rag/chunking.py` | 1 | Divisão recursiva em chunks com metadados |
| `rag/embeddings.py` | 1 | Embeddings locais `multilingual-e5-small` (prefixos `query:`/`passage:`) |
| `rag/store.py` | 1 | Índice Chroma persistente, indexação incremental |
| `rag/retrieval.py` | 1 | Busca vetorial, MMR, filtros por metadado, BM25 e busca híbrida (RRF) |
| `rag/search.py` | 1 | CLI de busca e a classe `Retriever` |
| `rag/evaluate.py` | 1 | Golden set e métricas (Hit@k, MRR) para comparar configurações |
| `tools/` | 0–1 | Ferramentas expostas ao agente: `get_stock_quote`, `search_news` |
| `committee/config.py` | 2 | Universo de ações (4 gratuitas + estendido com `BRAPI_TOKEN`), perfis de risco, tentativas máximas, aviso legal |
| `committee/market.py` | 2 | Histórico de preços (brapi.dev, cache em disco), validação de séries e alinhamento |
| `committee/metrics.py` | 2 | Retornos, volatilidade, drawdown, momentum, correlação, contribuição de risco |
| `committee/allocation.py` | 2 | Alocação por volatilidade inversa com tilt de score, tetos/pisos e vetos (water-filling) |
| `committee/brains.py` | 2 | `RulesBrain` (determinístico) e `LLMBrain` (limitado: ajuste de score ±1, só endurece vetos, desvio ≤15 p.p.) |
| `committee/graph.py` | 2 | Grafo LangGraph: coletar → analistas (paralelo) → gestor → validador → (retry/fallback) → relatório |
| `committee/run.py` | 2 | CLI do comitê (`--profile`, `--brain rules\|llm`, `--tickers`, `--no-save`) |
| `committee/backtest.py` | 2 | Backtest walk-forward sem lookahead, com custo de transação |
| `observability/tracing.py` | 3 | `Span`, `Tracer`, destinos (memória, JSONL), propagação de contexto e leitura dos spans |
| `observability/langchain_hook.py` | 3 | Callback do LangChain: spans de LLM e ferramentas com tokens e custo |
| `observability/pricing.py` | 3 | Tabela de preços (referência), custo por chamada, orçamento mensal, projeção e bloqueio |
| `observability/cache.py` | 3 | Cache exato (saídas validadas) e semântico (guarda de tickers, TTL) |
| `observability/report.py` | 3 | Painel de CLI: latência, tokens, custo, erros, cache e orçamento |
| `observability/evaluate.py` | 3 | Invariantes do comitê, grounding de respostas, LLM-as-judge e portão de regressão |
| `observability/golden.py` | 3 | Cenários dourados e modelos de chat simulados (`ScriptedChat`, `SimChat`) |
| `guardrails/pii.py` | 4 | Detecção e máscara de CPF/CNPJ/cartão (com dígito verificador), e-mail, telefone e segredos; achados sem o valor |
| `guardrails/injection.py` | 4 | Varredura de prompt injection (regras, Unicode, base64), saneamento e spotlighting com id aleatório |
| `guardrails/output.py` | 4 | Verificações de saída: links não autorizados, imagens com parâmetros, segredos, PII, recomendação |
| `guardrails/policy.py` | 4 | `Guard`: entrada, ferramenta e saída; modos enforce/monitor/off; auditoria sem conteúdo; spans `guard` |
| `guardrails/redteam.py` | 4 | Red team offline com modelos vulneráveis simulados; matriz de ataques x defesas e portão de CI |
| `guardrails/corpus.py` | 4 | Corpora de avaliação (injeção e PII) e métricas, com lacunas declaradas |
| `guardrails/report.py` | 4 | Painel de auditoria dos guardrails |
| `finetune/labeling.py` | 5 | Guia de rotulagem e rótulos prata do acervo (índices → positivo/negativo/neutro) |
| `finetune/data.py` | 5 | Texto de entrada, agrupamento de quase-duplicatas e divisão treino/validação/teste por grupos |
| `finetune/baselines.py` | 5 | Maioria, léxico e TF-IDF + regressão logística |
| `finetune/metrics.py` | 5 | Matriz de confusão, F1 macro, bootstrap e comparação pareada |
| `finetune/llm.py` | 5 | Prompt de chat, logits restritos aos 3 rótulos, carga do modelo e do adaptador LoRA |
| `finetune/train_lora.py` | 5 | Laço de treino LoRA em CPU (perda ponderada, acúmulo de gradiente, melhor época pela validação) |
| `finetune/evaluate_models.py` | 5 | Comparação de todos os sistemas no teste, com IC 95% e testes pareados |
| `finetune/robustness.py` | 5 | Robustez do classificador a prompt injection (taxa de virada) |
| `finetune/export.py` | 5 | Mesclar LoRA, quantizar int8 e gerar o Modelfile do Ollama |
| `tools/sentiment.py` | 5 | Ferramenta de sentimento local (backends lora/tfidf/léxico) com spans de custo zero |

Convenção planejada para as próximas fases:
- `src/api/` — camada de API exposta pelo backend (deploy na VPS)

Comandos úteis e testes: veja o "Início rápido" no `README.md` da raiz.
