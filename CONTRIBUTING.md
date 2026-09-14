# Como contribuir com o Taizen.IA

Este é um projeto construído em público, evoluindo por fases documentadas em [`ROADMAP.md`](ROADMAP.md). Contribuições de qualquer tamanho são bem-vindas — de uma correção pequena a implementar uma fase inteira.

## Antes de começar

1. Leia o [`README.md`](README.md) para entender objetivo, restrições de recursos e decisões de arquitetura já tomadas.
2. Leia o [`ROADMAP.md`](ROADMAP.md) para ver em qual fase o projeto está e o que falta em cada uma.
3. Confira as *issues* abertas — cada fase do roadmap deve ter issues correspondentes com o escopo detalhado.

## Como propor uma mudança

- **Bug ou melhoria pequena:** abra um PR diretamente, descrevendo o problema e a solução.
- **Nova funcionalidade ou mudança de arquitetura:** abra uma issue primeiro, explicando a proposta e o trade-off em relação à decisão atual (lembrando das restrições de recursos do projeto — sem GPU dedicada, VPS pequena, orçamento de API baixo). Alinhar antes evita retrabalho.
- **Discordar de uma decisão já tomada:** é bem-vindo, desde que venha com justificativa técnica e considerando as restrições listadas no README. Abra uma issue para discutir.

## Padrões do projeto

- Todo código pesado em dados (datasets, modelos, checkpoints) deve ler o caminho via variável de ambiente `DATA_DIR` — nunca caminho fixo no repositório.
- Novas dependências devem justificar seu custo (de recursos e de complexidade) dado o hardware-alvo descrito no README.
- Commits e PRs em português ou inglês, à sua escolha — o time é bilíngue.

## Escopo fora deste repositório

Material didático (teoria/exercícios por fase) e roteiros de divulgação em redes sociais são produzidos à parte e não fazem parte deste repositório — aqui o foco é código e arquitetura.
