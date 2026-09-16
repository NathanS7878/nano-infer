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

# We run and compare in fp16, deliberately. Phase 1's acceptance bar is a logit
# difference under 1e-3 *in fp16*, so the HF reference must be captured in the
# same dtype our engine will use — otherwise the comparison is apples-to-oranges.
DTYPE = torch.float16
DEVICE = "cuda"

# Deterministic everything. Greedy decode + fixed seed => same text in, same out,
# which is the property that makes the parity test possible at all.
SEED = 0

# --- paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
# The default model keeps writing to results/ itself, where every number the
# README cites already lives; any other model gets its own subdirectory.
RESULTS_DIR = (REPO_ROOT / "results" if MODEL_NAME == DEFAULT_MODEL
               else REPO_ROOT / "results" / MODEL_SLUG)
REFERENCE_FIXTURE = FIXTURES_DIR / f"reference_{MODEL_SLUG}.pt"

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
