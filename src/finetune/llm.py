"""Modelo de linguagem pequeno (SLM) como classificador: prompt, pontuação dos rótulos e carga do modelo/adaptador LoRA.

Em vez de GERAR texto e interpretá-lo, olhamos as probabilidades que o modelo atribui à PRIMEIRA palavra da resposta para
cada rótulo (positivo/negativo/neutro): a saída fica restrita a três opções (por construção, não há "resposta fora do
formato") e a classificação é determinística e rápida (uma passada para frente por exemplo).
"""

import os
import time
from pathlib import Path

import numpy as np
import torch

from src.config import ROOT
from src.finetune.data import make_text
from src.finetune.labeling import LABELS

BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = Path(os.environ.get("SLM_ADAPTER_DIR", ROOT / "models" / "lora-sentiment"))
MAX_LEN = 320  # tokens: título + abertura cabem; limita custo de memória e tempo

SYSTEM = ("Você classifica o tom de notícias financeiras brasileiras para o investidor. Responda com UMA palavra: "
          "positivo (favorece ativos, empresas ou o mercado), negativo (desfavorece) ou neutro (informativo, misto ou fora do mercado).")


def build_prompt(tokenizer, text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Notícia: {text}\nTom:"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_fewshot(tokenizer, text: str, shots: list[tuple[str, str]]) -> str:
    """Prompt com exemplos resolvidos (few-shot): cada (notícia, rótulo) vira um turno usuário/assistente antes da notícia real."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, lab in shots:
        msgs += [{"role": "user", "content": f"Notícia: {t}\nTom:"}, {"role": "assistant", "content": lab}]
    msgs.append({"role": "user", "content": f"Notícia: {text}\nTom:"})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def pick_shots(train_items: list[dict], per_class: int = 2, seed: int = 0, chars: int = 230) -> list[tuple[str, str]]:
    """Escolhe `per_class` exemplos de cada rótulo no TREINO (nunca no teste), intercalados para não enviesar a ordem."""
    import random

    rng = random.Random(seed)
    by = {lab: [x for x in train_items if x["label"] == lab] for lab in LABELS}
    chosen = {lab: rng.sample(by[lab], per_class) for lab in LABELS}
    shots = []
    for k in range(per_class):
        for lab in ("neutro", "positivo", "negativo"):
            shots.append((make_text(chosen[lab][k], chars), lab))
    return shots


def label_token_ids(tokenizer) -> list[int]:
    """Id do primeiro token de cada rótulo, na ordem de LABELS. Os três devem ser distintos."""
    ids = [tokenizer.encode(lab, add_special_tokens=False)[0] for lab in LABELS]
    if len(set(ids)) != len(ids):
        raise ValueError(f"os rótulos começam com o mesmo token: {ids}")
    return ids


def load(base: str = BASE_MODEL, adapter: Path | str | None = None, threads: int | None = None):
    """Carrega modelo (fp32, CPU) e tokenizer; com `adapter`, aplica o adaptador LoRA treinado."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads or max(os.cpu_count() // 2, 1))
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tok


def encode_batch(tok, texts: list[str], max_len: int = MAX_LEN, prompt_fn=None):
    """Tokeniza os prompts (truncando o TEXTO da notícia, não o modelo de chat) e devolve ids, máscara e posição do último token."""
    prompts = []
    for t in texts:
        body = t
        while True:
            p = (prompt_fn or build_prompt)(tok, body)
            if len(tok.encode(p)) <= max_len or len(body) < 60:
                break
            body = body[: int(len(body) * 0.85)]
        prompts.append(p)
    enc = tok(prompts, return_tensors="pt", padding=True)
    last = enc["attention_mask"].sum(dim=1) - 1
    return enc, last


def label_logits(model, enc, last, ids: list[int]) -> torch.Tensor:
    """Logits dos 3 tokens de rótulo na última posição do prompt, SEM calcular o vocabulário inteiro.

    O `lm_head` completo (151 mil palavras × todas as posições) é a parte mais cara de uma passada e aqui só importam 3 linhas
    dele na última posição: o resultado é idêntico ao do softmax restrito, com bem menos tempo e memória."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model  # PeftModel -> modelo base (com LoRA injetado)
    hidden = core.model(**enc).last_hidden_state
    h = hidden[torch.arange(len(last)), last]
    return h @ core.lm_head.weight[ids].T


@torch.no_grad()
def score_texts(model, tok, texts: list[str], batch_size: int = 8, prompt_fn=None, max_len: int = MAX_LEN) -> np.ndarray:
    """Probabilidades (n, 3) dos rótulos [negativo, neutro, positivo] — softmax restrito aos 3 tokens de rótulo."""
    ids = label_token_ids(tok)
    out = []
    for i in range(0, len(texts), batch_size):
        enc, last = encode_batch(tok, texts[i:i + batch_size], max_len, prompt_fn)
        out.append(torch.softmax(label_logits(model, enc, last, ids).float(), dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def predict_items(model, tok, items: list[dict], batch_size: int = 8, prompt_fn=None, max_len: int = MAX_LEN) -> tuple[list[str], np.ndarray]:
    probs = score_texts(model, tok, [make_text(it) for it in items], batch_size, prompt_fn, max_len)
    return [LABELS[i] for i in probs.argmax(axis=1)], probs


def time_per_item(model, tok, items: list[dict], n: int = 12) -> float:
    """Latência média (s) por notícia com lote de 1 (cenário de uso interativo)."""
    sample = [make_text(it) for it in items[:n]]
    score_texts(model, tok, sample[:1], 1)  # aquecimento
    t0 = time.perf_counter()
    for s in sample:
        score_texts(model, tok, [s], 1)
    return (time.perf_counter() - t0) / len(sample)
