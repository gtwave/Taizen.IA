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

Convenção planejada para as próximas fases:
- `src/api/` — camada de API exposta pelo backend (deploy na VPS)

Comandos úteis e testes: veja o "Início rápido" no `README.md` da raiz.
