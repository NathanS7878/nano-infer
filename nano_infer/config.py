"""Shared configuration: the single source of truth for model, dtype, prompts,
and file paths. Both the benchmark harness and the parity test import from here
so they can never silently disagree about what is being measured or checked.

Architecture facts below were read directly from the model config on 2026-08-20
(see HARDWARE.md / PROGRESS.md), not assumed. Qwen2.5-0.5B-Instruct:

    layers                24
    hidden_size           896
    intermediate_size     4864
    attention heads       14 query / 2 key-value  (GQA: 7 query heads per KV head)
    head_dim              64
    vocab_size            151936
    rms_norm_eps          1e-6
    activation            silu (SwiGLU)
    tie_word_embeddings   True   (output projection reuses the embedding matrix)
    native dtype          bfloat16

SELECTING A DIFFERENT MODEL. Set NANO_INFER_MODEL before importing anything
from nano_infer, e.g.

    NANO_INFER_MODEL=Qwen/Qwen2.5-1.5B-Instruct python -m bench.decode_graph

The default is Qwen2.5-0.5B-Instruct, so every existing script, test and
recorded number is unchanged. model.QwenConfig() reads the SELECTED checkpoint's
config.json and refuses architectures the engine does not implement, rather than
running one model's weights through another model's dimensions. Results and the
parity fixture are written per model (see RESULTS_DIR, REFERENCE_FIXTURE), so a
larger model's runs can never overwrite the 0.5B numbers the README cites.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

# --- model & precision -----------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = os.environ.get("NANO_INFER_MODEL", DEFAULT_MODEL)

# "Qwen/Qwen2.5-0.5B-Instruct" -> "qwen2.5-0.5b". The 0.5B slug reproduces the
# fixture filename that predates model selection, so nothing had to be renamed.
MODEL_SLUG = MODEL_NAME.split("/")[-1].lower().replace("-instruct", "")

# PRECISION. The default 0.5B model runs and compares in fp16, deliberately:
# Phase 1's acceptance bar is a logit difference under 1e-3 *in fp16*, so the HF
# reference must be captured in the same dtype the engine uses.
#
# fp16 is NOT safe for every model. Qwen2.5-1.5B's layer-0 k_proj biases reach
# 316, which drives the raw attention product q.k -- before the 1/sqrt(d)
# scale -- to 264,115, four times fp16's 65,504 ceiling. In fp16 it overflows to
# inf, softmax turns that into NaN, and HF's own model emits `"` fifty times
# (ROADMAP #41). So:
#   * NANO_INFER_DTYPE (float16 | bfloat16), if set, always wins;
#   * otherwise the default model stays fp16, so nothing already measured moves;
#   * any other model defaults to its checkpoint's native torch_dtype, because
#     "remember to pass bf16 for 1.5B" is how the NaN failure would come back.
_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}


def _native_dtype(model_name: str) -> torch.dtype:
    import json
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    path = try_to_load_from_cache(model_name, "config.json")
    if not isinstance(path, str):
        path = hf_hub_download(model_name, "config.json")
    with open(path, encoding="utf-8") as f:
        native = json.load(f).get("torch_dtype", "float16")
    if native not in _DTYPES:
        raise ValueError(f"{model_name} ships as {native!r}; this engine runs "
                         f"float16 or bfloat16 -- set NANO_INFER_DTYPE explicitly")
    return _DTYPES[native]


def _select_dtype() -> torch.dtype:
    env = os.environ.get("NANO_INFER_DTYPE")
    if env:
        if env.lower() not in _DTYPES:
            raise ValueError(f"NANO_INFER_DTYPE={env!r}; use float16 or bfloat16")
        return _DTYPES[env.lower()]
    if MODEL_NAME == DEFAULT_MODEL:
        return torch.float16
    return _native_dtype(MODEL_NAME)


DTYPE = _select_dtype()
DTYPE_NAME = {torch.float16: "fp16", torch.bfloat16: "bf16"}[DTYPE]
DEVICE = "cuda"

# Deterministic everything. Greedy decode + fixed seed => same text in, same out,
# which is the property that makes the parity test possible at all.
SEED = 0

# --- paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
# The default configuration (0.5B, fp16) keeps writing to results/ itself and
# reading the original fixture, where every number the README cites already
# lives. Any other model OR precision gets names carrying both, so a bf16 run can
# never overwrite an fp16 number, nor a 1.5B run a 0.5B one.
_DEFAULT_CONFIG = MODEL_NAME == DEFAULT_MODEL and DTYPE == torch.float16
RUN_TAG = MODEL_SLUG if _DEFAULT_CONFIG else f"{MODEL_SLUG}-{DTYPE_NAME}"
RESULTS_DIR = (REPO_ROOT / "results" if _DEFAULT_CONFIG
               else REPO_ROOT / "results" / RUN_TAG)
REFERENCE_FIXTURE = FIXTURES_DIR / f"reference_{RUN_TAG}.pt"

# --- the canonical prompt set ----------------------------------------------
# Five prompts, intentionally varied (factual, reasoning, code, list, open).
# Phase 1 must reproduce HF's greedy continuation of each, token for token.
PROMPTS: list[str] = [
    "What is the capital of France?",
    "If a train travels 60 miles in 1.5 hours, what is its average speed?",
    "Write a one-line Python function that returns the square of a number.",
    "List three primary colors.",
    "Explain in one sentence why the sky is blue.",
]

# How many tokens of greedy continuation to capture / compare per prompt.
PARITY_NEW_TOKENS = 50

# Quiet a harmless Windows-only HF cache warning (no symlink support).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
