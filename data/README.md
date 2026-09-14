# data/

**Regra:** só dados leves/de exemplo ficam aqui (e são ignorados pelo git via `.gitignore`, exceto este README). Dados pesados — datasets brutos, o índice do Chroma, modelos baixados, checkpoints de fine-tuning — vivem no **HD externo (1TB)**, fora do SSD de 256GB do notebook.

Configure a variável de ambiente `DATA_DIR` apontando para uma pasta no HD externo (ex: `D:\ia-data`) antes de rodar qualquer script de ingestão, treino ou persistência do Chroma. O código deve sempre ler esse caminho via `os.environ["DATA_DIR"]`, nunca com caminho fixo no SSD.
