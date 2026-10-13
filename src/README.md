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

Convenção planejada para as próximas fases:
- `src/agents/` — agentes e orquestração multiagente (Fase 2)
- `src/api/` — camada de API exposta pelo backend (deploy na VPS)

Comandos úteis e testes: veja o "Início rápido" no `README.md` da raiz.
