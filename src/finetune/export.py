"""Do adaptador ao uso: mesclar o LoRA, quantizar para int8 e preparar a importação no Ollama.

* mesclar (merge): W' = W + (alpha/r)·B·A vira uma matriz só; a inferência volta a ter o custo do modelo base (sem desvio extra);
* quantizar (int8 dinâmico): guarda os pesos das camadas lineares em 8 bits (≈ 4× menores que fp32) e quantiza as ativações na hora;
  na CPU costuma reduzir memória e latência, com pequena perda de qualidade — que MEDIMOS, não presumimos;
* Ollama: servidor local de modelos. Importar o modelo mesclado exige um `Modelfile`; aqui só o geramos (o Ollama não está instalado).
"""

import io
from pathlib import Path

import torch

from src.config import ROOT


def quantize_int8(model):
    """Quantização dinâmica int8 das camadas lineares do corpo do modelo (o `lm_head` fica em fp32: só 3 linhas dele são usadas)."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    # inplace=True mantém a ligação (tie) entre embeddings e lm_head; uma cópia a quebraria e duplicaria 136 M de parâmetros
    torch.ao.quantization.quantize_dynamic(core.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tensors(o)


def model_size_mb(model) -> float:
    """Tamanho dos pesos (MB). Serializa o state_dict em memória (o tamanho "de verdade" de um arquivo); se a serialização falhar
    (ocorre em alguns estados de importação do `transformers`), soma os bytes dos tensores, contando o armazenamento compartilhado uma vez."""
    try:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.tell() / 1e6
    except Exception:
        seen, total = set(), 0
        for v in model.state_dict().values():
            for t in _tensors(v):
                key = (t.untyped_storage().data_ptr(), t.numel()) if t.device.type == "cpu" and not t.is_quantized else id(t)
                if key not in seen:
                    seen.add(key)
                    total += t.numel() * t.element_size()
        return total / 1e6


def adapter_size_mb(adapter_dir: Path) -> float:
    return sum(f.stat().st_size for f in Path(adapter_dir).glob("*.safetensors")) / 1e6


def save_merged(adapter_dir: Path, out_dir: Path, base: str | None = None) -> Path:
    """Carrega base + adaptador, mescla e salva em `out_dir` (formato Hugging Face, com o tokenizer)."""
    from src.finetune import llm

    model, tok = llm.load(base or llm.BASE_MODEL, adapter=adapter_dir)
    merged = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir


def modelfile_text(model_dir: str | Path) -> str:
    """`Modelfile` do Ollama para o modelo mesclado (ele lê pesos safetensors de uma pasta). Para gerar um GGUF quantizado
    (q4_K_M, por exemplo), use as ferramentas do llama.cpp ou `ollama create ... --quantize`. NÃO foi executado neste projeto."""
    from src.finetune.llm import SYSTEM

    return f'''FROM {Path(model_dir).as_posix()}
SYSTEM """{SYSTEM}"""
PARAMETER temperature 0
PARAMETER num_predict 4
'''


if __name__ == "__main__":
    from src.finetune import llm

    d = ROOT / "models" / "merged-sentiment"
    save_merged(llm.ADAPTER_DIR, d)
    (ROOT / "models" / "Modelfile").write_text(modelfile_text(d), encoding="utf-8")
    print("modelo mesclado em", d, "| adaptador:", round(adapter_size_mb(llm.ADAPTER_DIR), 1), "MB")
