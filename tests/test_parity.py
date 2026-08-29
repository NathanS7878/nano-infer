"""Parity: the correctness answer key and the tools to grade against it.

Phase 0 responsibilities:
  1. Confirm the reference fixture exists and is well-formed.
  2. Prove the reference is REPRODUCIBLE — re-run HF greedy decode and assert it
     reproduces the stored fixture exactly. This is the "same text in, same out"
     property in test form; without it the answer key would be untrustworthy.

The comparison helpers (`compare_tokens`, `compare_step0_logits`) are written to
be reused in Phase 1, where the thing under test becomes OUR engine instead of a
second run of HF.

Run:  python -m pytest tests/test_parity.py -v
(Regenerate the fixture first, if missing:  python -m tests.capture_reference)
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config
from nano_infer.hf_ref import encode_prompt, load_hf
from tests.capture_reference import greedy_capture

# fp16 logit tolerance — the Phase 1 acceptance bar ("max abs diff < 1e-3 in fp16").
LOGIT_TOL = 1e-3


def load_reference() -> dict:
    if not config.REFERENCE_FIXTURE.exists():
        pytest.skip(
            f"Missing fixture {config.REFERENCE_FIXTURE.name}. "
            f"Generate it: python -m tests.capture_reference"
        )
    return torch.load(config.REFERENCE_FIXTURE, weights_only=False)


# --- comparison helpers (reused by Phase 1) --------------------------------

def compare_tokens(ref_ids: torch.Tensor, got_ids: torch.Tensor) -> tuple[bool, int]:
    """Token-for-token. Returns (all_match, first_divergence_index or -1)."""
    n = min(len(ref_ids), len(got_ids))
    for i in range(n):
        if int(ref_ids[i]) != int(got_ids[i]):
            return False, i
    return (len(ref_ids) == len(got_ids)), -1


def compare_step0_logits(ref_logits: torch.Tensor, got_logits: torch.Tensor) -> float:
    """Max absolute difference between two logit vectors (upcast to fp32, on CPU
    so the two operands can live on different devices)."""
    return (ref_logits.float().cpu() - got_logits.float().cpu()).abs().max().item()


# --- Phase 0 tests ---------------------------------------------------------

def test_fixture_wellformed():
    ref = load_reference()
    assert ref["meta"]["model_name"] == config.MODEL_NAME
    assert len(ref["prompts"]) == len(config.PROMPTS)
    vocab = ref["meta"]["vocab_size"]
    for p in ref["prompts"]:
        assert p["greedy_ids"].shape[0] == config.PARITY_NEW_TOKENS
        assert p["step0_logits"].shape[0] == vocab
        assert p["topk_ids"].shape[0] == config.PARITY_NEW_TOKENS


@pytest.mark.slow
def test_hf_reproduces_reference():
    """Determinism / reproducibility: a fresh HF run must match the frozen fixture
    exactly (tokens identical, step-0 logits within fp16 tolerance)."""
    ref = load_reference()
    torch.manual_seed(config.SEED)
    model, tok = load_hf(attn_implementation="eager")   # match the fixture's impl

    for i, (prompt, ref_p) in enumerate(zip(config.PROMPTS, ref["prompts"])):
        ids = encode_prompt(tok, prompt)
        got = greedy_capture(model, ids, config.PARITY_NEW_TOKENS)

        ok, div = compare_tokens(ref_p["greedy_ids"], got["greedy_ids"])
        assert ok, f"prompt {i}: token divergence at step {div}"

        max_diff = compare_step0_logits(ref_p["step0_logits"], got["step0_logits"])
        assert max_diff < LOGIT_TOL, (
            f"prompt {i}: step-0 logit max abs diff {max_diff:.2e} >= {LOGIT_TOL}"
        )
