# src/

Código do projeto, evoluindo fase a fase (ver `../ROADMAP.md`).

Nada aqui ainda — o primeiro código (script de function calling da Fase 0) entra quando começarmos a implementação, seguindo `../docs/fase-00-fundacao/README.md`.

Convenção planejada:
- `src/agents/` — agentes e orquestração (a partir da Fase 2)
- `src/rag/` — ingestão, chunking, embeddings, retrieval (a partir da Fase 1)
- `src/tools/` — funções expostas como tools ao LLM (ex: consulta de cotação)
- `src/api/` — camada de API exposta pelo backend (a partir do deploy na VPS)
