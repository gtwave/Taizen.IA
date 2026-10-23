# Taizen.IA — Arquitetura de IA em Produção, do Zero ao Multi-Agente

Um sistema de IA sendo construído em público, do primeiro protótipo até um sistema multi-agente com observabilidade, governança e um modelo fine-tunado — rodando em hardware modesto e com orçamento real de projeto pessoal (sem GPU, ~R$50/mês). O caso de uso funcional: um comitê de agentes que analisa ações da B3 e sugere uma carteira recomendada.

Este repositório é o código e a arquitetura por trás desse processo. **Contribuições são bem-vindas** — veja [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Por que contribuir

- **Escopo real de produção, não brinquedo de tutorial.** RAG, agentes, observabilidade (LLMOps), guardrails e fine-tuning — as áreas que qualquer arquitetura de IA séria precisa cobrir, cada uma com decisões documentadas e justificadas.
- **Restrições reais.** Sem GPU dedicada, VPS de 2 vCPU/8GB, orçamento de API limitado. As soluções aqui não assumem infraestrutura ilimitada — se você trabalha com restrição de recursos parecida, as decisões deste repo são diretamente aplicáveis ao seu contexto.
- **Roadmap público e incremental.** Todo o plano de fases está em [`ROADMAP.md`](ROADMAP.md), com escopo, stack e entregável de cada etapa — dá pra entrar em qualquer ponto do processo.
- **Issues abertas por fase.** Cada fase do roadmap vira issues concretas no GitHub, com o que falta implementar, revisar ou discutir.

## Objetivos do projeto

1. **Funcional:** pipeline de dados financeiros (B3) → RAG → sistema multi-agente (análise, risco, gestão de carteira) → sugestão de carteira recomendada, com observabilidade e guardrails de produção.
2. **Arquitetural:** cobrir de ponta a ponta as áreas centrais de arquitetura de IA — engenharia de dados/vetorial, agentes/orquestração, fine-tuning, LLMOps e governança — com cada decisão de stack documentada e justificada pelas restrições reais do projeto.

## Restrições de recursos (moldam toda decisão técnica)

| Recurso | Especificação | Implicação |
|---|---|---|
| Máquina de desenvolvimento | CPU, 16GB RAM, sem GPU dedicada, SSD 256GB | Sem treinamento pesado local. Inferência local só com modelos pequenos quantizados (CPU). |
| Armazenamento de dados | Disco externo, 1TB | Datasets, modelos baixados e checkpoints ficam fora do disco principal, via variável de ambiente `DATA_DIR`. |
| Infra de deploy | VPS: 2 vCPU, 8GB RAM, 100GB disco | Hospeda backend + vector store (Chroma) via Docker. Não hospeda modelos grandes — geração usa API paga; embeddings rodam local/CPU. |
| Orçamento de API | Baixo (uso pessoal) | Fine-tuning real usa GPU alugada por poucas horas ou camada gratuita (Google Colab), não infraestrutura dedicada. |

## Decisões de arquitetura já tomadas

- **LLM:** híbrido — API paga (Claude/GPT) para geração desde o MVP; modelos locais (Ollama) entram nas fases de fine-tuning/SLM.
- **Orquestração:** framework desde o início (LangChain, evoluindo para LangGraph/CrewAI no sistema multi-agente).
- **Vector DB:** Chroma no MVP (simples, local, sem servidor separado).
- **Mercado-alvo do caso de uso:** ações da B3 (Brasil), dados via brapi.dev.
- **Embeddings:** modelo local multilíngue (`intfloat/multilingual-e5-small` via sentence-transformers) rodando em CPU — não consome orçamento de API. (Decisão revista na Fase 1: o `all-MiniLM-L6-v2` previsto inicialmente é focado em inglês e rende mal em português.)
- **Corpus da Fase 1:** notícias financeiras coletadas por feeds RSS públicos, só localmente, com respeito ao `robots.txt` — o conteúdo das notícias **não** é redistribuído neste repositório (veja a seção "Política de dados" do `CONTRIBUTING.md`).

Discussão e mudanças de decisão acontecem via issues/PRs — proponha alternativas com a justificativa de trade-off.

## Estrutura do repositório

```
├── README.md
├── ROADMAP.md      todas as fases do projeto, com escopo, stack e entregável de cada uma
├── CONTRIBUTING.md  como propor mudanças, abrir issues e enviar PRs
├── LICENSE          MIT
├── src/             código do projeto, evoluindo fase a fase
│   ├── agent.py     Fase 0 — function calling com cotação ao vivo
│   ├── agent_rag.py Fase 1 — agente LangChain: notícias (RAG) + cotação
│   ├── rag/         Fase 1 — coleta, chunking, embeddings, Chroma, recuperação e avaliação
│   ├── committee/   Fase 2 — comitê multiagente em LangGraph: dados, risco, gestor, validador, backtest
│   ├── observability/ Fase 3 — tracing, custo e orçamento, caches, avaliação e portão de regressão
│   ├── guardrails/  Fase 4 — PII, prompt injection, guardrails de saída, política, auditoria e red team
│   ├── finetune/    Fase 5 — rótulos, divisão sem vazamento, linhas de base, LoRA em CPU, avaliação com IC, quantização
│   └── tools/       ferramentas expostas ao agente (get_stock_quote, search_news)
├── tests/           testes sem rede e sem API paga (mocks e embeddings falsos)
├── infra/           docker-compose e configs de deploy na VPS
└── data/            dados locais (ignorados pelo Git); os pesados podem ficar em DATA_DIR (HD externo)
```

> Material didático (teoria/exercícios de cada fase) e roteiros de divulgação em redes sociais são produzidos em paralelo, mas não fazem parte deste repositório público — aqui fica o código e a arquitetura.

## Início rápido (Fases 1 e 2)

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; no Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # preencha ANTHROPIC_API_KEY (só o agente usa a API paga)

python -m src.rag.ingest                              # coleta notícias dos feeds RSS (local, gratuito)
python -m src.rag.index                               # chunking + embeddings locais + índice Chroma
python -m src.rag.search "juros sobre capital próprio do Santander"   # busca híbrida, sem LLM
python src/agent_rag.py "O que saiu de notícia sobre a Vale e como está a cotação?"   # usa a API paga

python -m unittest discover -s tests -t .             # 150+ testes, sem rede e sem custo
```

### Comitê de carteira (Fase 2)

```bash
python -m src.committee.run --profile moderado        # comitê de regras: gratuito, sem API paga
python -m src.committee.run --profile conservador --no-save
python -m src.committee.backtest                      # backtest walk-forward vs. igual peso e vol. inversa
python -m src.committee.run --profile moderado --brain llm   # LLM com limites (usa a API paga)
```

- **Dados de mercado:** a brapi.dev responde sem token só para `PETR4`, `VALE3`, `ITUB4` e `MGLU3` (o universo padrão). Para mais ações (BBDC4, BBAS3, ABEV3, WEGE3, SUZB3, RENT3) e índices, crie um token gratuito na brapi.dev e defina `BRAPI_TOKEN` no `.env`.
- **Princípio de projeto:** o LLM influencia, o código garante — pesos finais passam por um validador (soma 100%, tetos, vetos de risco, desvio máximo de 15 p.p. da linha de base) e, se o LLM falhar, há fallback determinístico.
- **Aviso:** projeto educacional. As carteiras geradas **não são recomendação de investimento**; backtests não garantem resultados futuros.

### Observabilidade e avaliação (Fase 3)

```bash
python -m src.committee.run --profile moderado        # já grava traces locais em data/processed/traces
python -m src.observability.report --days 7           # painel: latência p50/p95, tokens, custo, erros, cache, orçamento
python -m src.observability.evaluate                  # portão de regressão: 36 cenários offline (sai com código 1 se algum falhar)
python src/agent_rag.py --cache "pergunta"            # cache semântico (perguntas parecidas reaproveitam a resposta)
python -m src.committee.run --brain llm --cache       # cache exato de saídas validadas do LLM (usa a API paga se houver falha de cache)
```

- **Orçamento:** antes de qualquer chamada paga, o código lê os traces do mês e recusa a chamada se o gasto passar de `MONTHLY_BUDGET_BRL` (padrão R$ 50). Preços e cotação são de referência (`PRICING_JSON`, `USD_BRL`).
- **Privacidade:** os spans guardam tamanhos e um hash do prompt, não o texto; os traces ficam em `data/` (fora do Git).

### Segurança e governança (Fase 4)

```bash
python -m src.guardrails.redteam                      # red team offline: matriz de 8 ataques x 4 configurações de defesa (sai com 1 se a defesa completa falhar)
python -m src.guardrails.redteam --audit              # + grava as decisões em data/processed/guard_audit.jsonl
python -m src.guardrails.report --days 7              # painel de auditoria (tipos e regras acionadas, nunca o conteúdo)
python src/agent_rag.py --guards=monitor "pergunta"   # guardrails no agente: enforce (padrão) | monitor | off
python -m src.committee.run --guards=enforce          # manchetes do RAG em quarentena e texto entre agentes saneado
```

- **Princípio:** tudo que entra de fora (perguntas, notícias, saídas de ferramentas, texto gerado por outro agente) é **dado**; só o código decide. Camadas: varredura/quarentena → aviso no prompt → guardrail de saída → validador da carteira → limites estruturais.
- **Privacidade:** CPF, CNPJ (opcional), cartão, e-mail, telefone e segredos são mascarados antes de chegar ao LLM, ao cache e aos traces (CPF/CNPJ/cartão validados por dígito verificador). Defina `GUARD_SALT` (secreto) no `.env`: o valor padrão é público.
- **Limites honestos:** a detecção por padrões é uma peneira (paráfrase, outros idiomas e ofuscação passam; o red team registra isso). O que contém o dano são as camadas estruturais. Os "modelos vulneráveis" do red team são simulados — não provam a resistência de um modelo real.

### Fine-tuning de um SLM (Fase 5)

```bash
python -m src.finetune.labeling                       # grava data/finetune/labels.jsonl (rótulos do acervo; ver ressalva abaixo)
python -m src.finetune.train_lora --limit 12 --epochs 1   # teste de fumaça (<1 min)
python -m src.finetune.train_lora --epochs 3          # LoRA em CPU no Qwen2.5-0.5B-Instruct (~1 h; baixa ~1 GB do Hugging Face)
python -m src.finetune.evaluate_models --quant        # linhas de base x SLM zero-shot x LoRA x int8, com IC 95% por bootstrap
python -m src.finetune.robustness                     # taxa de virada sob instruções escondidas no texto (elo com a Fase 4)
python -m src.finetune.export                         # mescla o adaptador e gera o Modelfile do Ollama (não executado)
python -m src.committee.run --news --sentiment        # tom de cada manchete pelo modelo local (backend: lora > tfidf > léxico)
```

- **Tarefa:** classificar o tom (positivo/negativo/neutro) de notícias financeiras em português. Modelo: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0) + LoRA (≈ 1,75% dos parâmetros, adaptador de dezenas de MB).
- **Ressalva dos dados:** os rótulos (`src/finetune/labeling.py`) são **rótulos prata** de um único anotador (o assistente de IA da sessão, sem API paga) para o acervo RSS coletado em 23/10/2026. O **texto das notícias não é redistribuído** (pertence aos veículos): para reproduzir, colete o seu acervo (`python -m src.rag.ingest`) e rotule-o com o guia do módulo. As métricas medem concordância com esse anotador.
- **Avaliação honesta:** divisão por grupos de quase-duplicatas (sem vazamento), linhas de base (maioria, léxico, TF-IDF + regressão logística), intervalos bootstrap e comparação pareada. Com 91 notícias de teste, só diferenças grandes são detectáveis.

## Acompanhe o projeto

- YouTube: canal em construção — em breve o link aqui.
- Instagram: [instagram.com/taizen.treinamentos](https://www.instagram.com/taizen.treinamentos/)
- Site: [taizen.tech](https://taizen.tech/)

## Como navegar

Comece por [`ROADMAP.md`](ROADMAP.md) para ver o plano completo de fases. Para contribuir, veja [`CONTRIBUTING.md`](CONTRIBUTING.md).
