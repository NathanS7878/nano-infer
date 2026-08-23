"""HuggingFace reference: load the model/tokenizer, encode prompts, and expose a
greedy generate() wrapper.

This module is the *baseline* and the *answer key* — the two things Phase 0 exists
to produce. Using `model.generate()` in here is intentional and allowed: the Phase
0 baseline row IS "HuggingFace generate(), unmodified." The Rule-1 ban on
generate() applies to *our* engine (Phase 1+), which will not import this.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def load_hf(dtype: torch.dtype = config.DTYPE, device: str = config.DEVICE,
            attn_implementation: str | None = None):
    """Load Qwen in eval mode, on the GPU, at the given dtype. Returns (model, tok).

    attn_implementation: pass "eager" to force explicit-softmax attention (the
    unambiguous math reference for per-component parity tests); None uses HF's
    default (sdpa, a fused kernel)."""
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    kwargs = {"dtype": dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **kwargs)
    model.to(device).eval()
    return model, tok


def encode_prompt(tok, prompt: str, device: str = config.DEVICE) -> torch.Tensor:
    """Apply the chat template to a single user message and tokenize.

    Returns input_ids of shape [1, seq_len]. We keep one prompt per call (no
    padding) so the captured logits are never contaminated by pad positions.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    return ids


@torch.no_grad()
def hf_generate(model, input_ids, max_new_tokens: int) -> torch.Tensor:
    """Greedy decode via HF's own generate() — the baseline the harness measures.

    Deterministic: do_sample=False, no beams, no repetition tricks. Returns the
    full sequence (prompt + continuation), shape [batch, seq_len + max_new_tokens].
    """
    attention_mask = torch.ones_like(input_ids)
    return model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=model.config.eos_token_id,
    )
