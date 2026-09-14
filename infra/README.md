# infra/

Configuração de deploy na VPS (2 vCPU, 8GB RAM, 100GB disco).

Fica vazio até a Fase 2/3, quando o sistema passa a rodar em Docker. Planejado:
- `docker-compose.yml` — backend + Chroma (persistido em volume) + observabilidade leve (ex: Langfuse self-hosted, se couber no orçamento de RAM).
- Sem serviço de inferência de LLM pesado na VPS — geração usa API paga; a VPS só hospeda a aplicação e o vector store.
