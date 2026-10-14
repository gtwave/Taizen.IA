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

## Política de dados (Fase 1)

- O corpus de notícias é coletado **localmente** por feeds RSS públicos, respeitando o `robots.txt`, com `User-Agent` identificado e intervalo entre requisições. O código não faz raspagem de HTML.
- O conteúdo das notícias pertence aos veículos de origem: **não** o publique neste repositório (as pastas de dados já são ignoradas pelo Git). Os testes usam apenas textos sintéticos.
- Ao adicionar um feed, confira os termos de uso do site e mantenha a coleta em ritmo moderado.
- Prefira, sempre que possível, fontes com licença aberta ou dados oficiais (por exemplo, dados abertos da CVM).
