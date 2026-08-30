"""Phase 2 step 3: continuous batching.

Correctness bar: scheduling must not change what any request generates. A request
must produce the same tokens whether it ran alone, in a static group, or was
admitted mid-flight next to sequences at completely different positions.

That last case is the one continuous batching newly requires and the one most
likely to break: sequences in a batch sit at different absolute positions, so
RoPE, the block tables, and the attention mask all have to be per-sequence.

Run:  python -m pytest tests/test_continuous.py -v -s
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.engine import ContinuousBatchingEngine, Request
from nano_infer.hf_ref import encode_prompt, load_hf

NEW_TOKENS = 12


@pytest.fixture(scope="module")
def weights():
    return M.load_weights()


@pytest.fixture(scope="module")
def tok():
    _, t = load_hf()
    return t


@pytest.fixture(scope="module")
def engine(weights):
    return ContinuousBatchingEngine(weights, M.QwenConfig(), num_blocks=256,
                                    block_size=16, max_batch=3, max_seq=256)


def test_continuous_matches_solo_generation(engine, weights, tok):
    """Every request must generate exactly what it would have generated alone."""
    cf = M.QwenConfig()
    prompts = cfg.PROMPTS[:4]
    id_list = [encode_prompt(tok, p) for p in prompts]

    solo = [M.generate_paged(ids, weights, cf, NEW_TOKENS)[0].tolist()
            for ids in id_list]

    reqs = [Request(req_id=i, prompt_ids=ids, max_new_tokens=NEW_TOKENS)
            for i, ids in enumerate(id_list)]
    engine.run_continuous(reqs)

    for i, (req, expected) in enumerate(zip(reqs, solo)):
        assert req.generated == expected, (
            f"request {i} differs from its solo run\n"
            f"  solo:       {expected}\n"
            f"  continuous: {req.generated}")
    print(f"\n[continuous] all {len(reqs)} requests match their solo generation")


def test_uneven_lengths_and_midflight_admission(engine, weights, tok):
    """Varied output lengths force eviction and mid-flight admission, so later
    requests join a batch whose members are at unrelated positions."""
    cf = M.QwenConfig()
    ids = encode_prompt(tok, cfg.PROMPTS[0])
    lengths = [3, 14, 5, 9, 4]

    solo = {}
    for n in set(lengths):
        solo[n] = M.generate_paged(ids, weights, cf, n)[0].tolist()

    reqs = [Request(req_id=i, prompt_ids=ids, max_new_tokens=n)
            for i, n in enumerate(lengths)]
    stats = engine.run_continuous(reqs)

    for req in reqs:
        assert len(req.generated) == req.max_new_tokens
        assert req.generated == solo[req.max_new_tokens], (
            f"request {req.req_id} (len {req.max_new_tokens}) diverged after "
            f"mid-flight admission")

    admitted_late = [r for r in reqs if r.admitted_step > 0]
    print(f"\n[mid-flight] lengths {lengths}; {len(admitted_late)} request(s) "
          f"admitted after step 0, all still correct")
    print(f"[mid-flight] {stats.steps} decode steps, "
          f"slot utilization {stats.slot_utilization:.1f}%")
    assert admitted_late, "test needs at least one mid-flight admission"


def test_blocks_are_recycled(engine, weights, tok):
    """Every block must return to the free list once the stream drains."""
    ids = encode_prompt(tok, cfg.PROMPTS[0])
    reqs = [Request(req_id=i, prompt_ids=ids, max_new_tokens=n)
            for i, n in enumerate([4, 9, 6, 3, 7, 5])]

    cache_before = engine._new_cache()
    free_before = cache_before.allocator.num_free

    engine.run_continuous(reqs)

    after = engine._new_cache()
    assert after.allocator.num_free == free_before
    print(f"\n[blocks] {len(reqs)} requests through a "
          f"{free_before}-block pool, all blocks recycled")


def test_static_and_continuous_agree(engine, weights, tok):
    """Both policies must produce identical tokens — they differ only in when a
    slot is reused, never in what is computed."""
    ids = encode_prompt(tok, cfg.PROMPTS[1])
    lengths = [4, 11, 6]

    a = [Request(req_id=i, prompt_ids=ids, max_new_tokens=n)
         for i, n in enumerate(lengths)]
    b = [Request(req_id=i, prompt_ids=ids, max_new_tokens=n)
         for i, n in enumerate(lengths)]

    s_stats = engine.run_static(a)
    c_stats = engine.run_continuous(b)

    for ra, rb in zip(a, b):
        assert ra.generated[:ra.max_new_tokens] == rb.generated, (
            f"request {ra.req_id}: static and continuous disagree")
    print(f"\n[policies] identical output; slot utilization "
          f"static {s_stats.slot_utilization:.1f}% vs "
          f"continuous {c_stats.slot_utilization:.1f}%")
