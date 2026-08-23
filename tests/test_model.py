"""Component-by-component parity for our from-scratch model (Phase 1).

Each test builds one piece and checks it against the equivalent HuggingFace
internal output. Correctness accrues one component at a time — a bug is caught the
moment the piece that has it is added, not 300 lines later in a wall of wrong logits.

Run (with prints):  python -m pytest tests/test_model.py -v -s
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.hf_ref import encode_prompt, load_hf

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def hf():
    model, tok = load_hf()
    return model, tok


@pytest.fixture(scope="module")
def weights():
    return M.load_weights()


def test_embedding_matches_hf(hf, weights):
    model, tok = hf
    ids = encode_prompt(tok, PROMPT)

    ours = M.embed_tokens(ids, weights)                 # our lookup
    ref = model.model.embed_tokens(ids)                 # HF's embedding layer

    max_diff = (ours.float() - ref.float()).abs().max().item()
    print(f"\n[embedding] input_ids shape {tuple(ids.shape)} -> "
          f"hidden {tuple(ours.shape)}")
    print(f"[embedding] max abs diff vs HF: {max_diff:.2e}")
    assert max_diff == 0.0, "same table + same gather must be bit-identical"


def test_rmsnorm_matches_hf(hf, weights):
    model, tok = hf
    cf = M.QwenConfig()
    ids = encode_prompt(tok, PROMPT)
    x = M.embed_tokens(ids, weights)                    # feed real hidden states in

    w = weights["model.layers.0.input_layernorm.weight"]
    ours = M.rms_norm(x, w, cf.rms_norm_eps)            # our RMSNorm
    ref = model.model.layers[0].input_layernorm(x)      # HF's RMSNorm, same input

    max_diff = (ours.float() - ref.float()).abs().max().item()
    print(f"\n[rmsnorm] max abs diff vs HF: {max_diff:.2e}  (fp16 tol 1e-3)")
    assert max_diff < 1e-3, f"RMSNorm diff {max_diff:.2e} exceeds fp16 tolerance"


def test_rope_matches_hf(hf, weights):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    model, tok = hf
    cf = M.QwenConfig()
    seq = 12
    torch.manual_seed(0)
    q = torch.randn(1, cf.num_q_heads, seq, cf.head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k = torch.randn(1, cf.num_kv_heads, seq, cf.head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    position_ids = torch.arange(seq, device=cfg.DEVICE).unsqueeze(0)

    # HF reference: its rotary module produces cos/sin, its apply fn rotates
    hf_cos, hf_sin = model.model.rotary_emb(q, position_ids)
    ref_q, ref_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    # ours — rope_theta moved into config.rope_parameters in transformers 5.x
    theta = float(model.config.rope_parameters["rope_theta"])
    cos, sin = M.build_rope_cache(seq, cf.head_dim, theta)
    our_q, our_k = M.apply_rope(q, k, cos, sin)

    cos_diff = (cos.float() - hf_cos.squeeze(0).float()).abs().max().item()
    q_diff = (our_q.float() - ref_q.float()).abs().max().item()
    k_diff = (our_k.float() - ref_k.float()).abs().max().item()
    print(f"\n[rope] theta={theta:g}  cos-table diff {cos_diff:.2e}")
    print(f"[rope] rotated Q diff {q_diff:.2e}   rotated K diff {k_diff:.2e}  (tol 1e-3)")
    assert q_diff < 1e-3 and k_diff < 1e-3, "RoPE output exceeds fp16 tolerance"
