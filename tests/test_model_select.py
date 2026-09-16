"""Model selection: QwenConfig reads the selected checkpoint instead of assuming.

The engine's architecture constants were hard-coded for Qwen2.5-0.5B until
2026-09-16. They now come from config.MODEL_NAME's config.json. The danger in
that change is silent: every one of ~33 `QwenConfig()` call sites would quietly
run whatever weights were loaded through whatever dimensions were constructed.
So this pins three things:

  1. the default model still yields EXACTLY the old constants, so no 0.5B test,
     benchmark or published number can have shifted;
  2. a different model yields its own dimensions, read from its checkpoint;
  3. architectures the engine does not implement are refused, not run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from nano_infer import config as cfg
from nano_infer import model as M

# The constants QwenConfig hard-coded before model selection existed.
OLD_0_5B = dict(vocab_size=151936, hidden_size=896, intermediate_size=4864,
                num_layers=24, num_q_heads=14, num_kv_heads=2, head_dim=64,
                rms_norm_eps=1e-6, rope_theta=1_000_000.0, tie_word_embeddings=True)


def test_default_model_is_unchanged():
    if cfg.MODEL_NAME != cfg.DEFAULT_MODEL:
        pytest.skip("suite is running against a non-default model")
    cf = M.QwenConfig()
    got = {k: getattr(cf, k) for k in OLD_0_5B}
    assert got == OLD_0_5B, f"0.5B architecture drifted from the hard-coded values: {got}"
    assert (cf.q_dim, cf.kv_dim) == (896, 128)
    assert cfg.REFERENCE_FIXTURE.name == "reference_qwen2.5-0.5b.pt", (
        "the 0.5B fixture filename changed; the committed answer key would be orphaned")
    assert cfg.RESULTS_DIR == cfg.REPO_ROOT / "results", (
        "default-model results moved; README-cited results would stop updating in place")


def test_selecting_another_model_reads_its_checkpoint():
    """Run in a subprocess: MODEL_NAME is read at import time, by design."""
    code = (
        "from nano_infer import config as cfg, model as M\n"
        "cf = M.QwenConfig()\n"
        "print(cfg.MODEL_NAME, cfg.MODEL_SLUG, cfg.RESULTS_DIR.name, cfg.REFERENCE_FIXTURE.name,"
        " cf.hidden_size, cf.intermediate_size, cf.num_layers, cf.num_q_heads,"
        " cf.num_kv_heads, cf.head_dim)\n"
    )
    env = dict(os.environ, NANO_INFER_MODEL="Qwen/Qwen2.5-1.5B-Instruct",
               PYTHONPATH=str(cfg.REPO_ROOT))
    try:
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                             text=True, timeout=120, check=True).stdout.split()
    except subprocess.CalledProcessError as e:
        if "LocalEntryNotFoundError" in e.stderr or "offline" in e.stderr.lower():
            pytest.skip("Qwen2.5-1.5B-Instruct config not available locally")
        raise
    assert out == ["Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", "qwen2.5-1.5b",
                   "reference_qwen2.5-1.5b.pt",
                   "1536", "8960", "28", "12", "2", "128"], out


def test_unsupported_architectures_are_refused(tmp_path, monkeypatch):
    """Each feature the engine does not implement must raise, not run."""
    base = json.loads(json.dumps(M.checkpoint_config(cfg.MODEL_NAME)))
    cases = {
        "rope_scaling": {"rope_scaling": {"type": "yarn", "factor": 4.0}},
        "untied lm_head": {"tie_word_embeddings": False},
        "sliding-window": {"use_sliding_window": True},
        "model_type": {"model_type": "llama"},
    }
    import huggingface_hub
    for label, override in cases.items():
        cfg_path = tmp_path / f"{label.replace(' ', '_')}.json"
        cfg_path.write_text(json.dumps({**base, **override}), encoding="utf-8")
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda *a, _p=str(cfg_path), **k: _p)
        name = f"fake/{label}"
        with pytest.raises(ValueError, match=label):
            M.checkpoint_config(name)
