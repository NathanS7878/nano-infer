"""Phase 2 step 2: paged KV cache.

Three things to establish:
  1. The allocator and block table behave (allocate, free, reuse, exhaustion).
  2. The paged path computes the same thing as the contiguous cache.
  3. The memory claim is real — paging wastes only the tail of each sequence's
     last block, where the contiguous cache reserves max_seq per sequence.

Run:  python -m pytest tests/test_paged.py -v -s
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer.cache import BlockAllocator, KVCache, PagedKVCache
from nano_infer.hf_ref import encode_prompt, load_hf
from tests.test_parity import compare_tokens, load_reference

NEAR_TIE_GAP = 0.05


@pytest.fixture(scope="module")
def weights():
    return M.load_weights()


@pytest.fixture(scope="module")
def tok():
    _, t = load_hf()
    return t


# --- 1. allocator ----------------------------------------------------------

def test_allocator_lifecycle():
    a = BlockAllocator(8)
    assert a.num_free == 8

    first = a.allocate(3)
    assert len(set(first)) == 3 and a.num_free == 5 and a.num_used == 3

    second = a.allocate(5)
    assert a.num_free == 0
    assert not set(first) & set(second), "a block must not be handed out twice"

    with pytest.raises(MemoryError):
        a.allocate(1)

    a.release(first)
    assert a.num_free == 3
    reused = a.allocate(3)
    assert set(reused) == set(first), "freed blocks must be reusable"
    print("\n[allocator] allocate / exhaust / release / reuse all correct")


def test_block_table_growth_and_free():
    cf = M.QwenConfig()
    c = PagedKVCache(cf.num_layers, 16, 16, cf.num_kv_heads, cf.head_dim)
    s = c.add_sequence()

    c.ensure_capacity(s, 1)
    assert len(c.block_tables[s]) == 1, "1 token needs 1 block"
    c.ensure_capacity(s, 16)
    assert len(c.block_tables[s]) == 1, "16 tokens still fit in 1 block of 16"
    c.ensure_capacity(s, 17)
    assert len(c.block_tables[s]) == 2, "17 tokens needs a 2nd block"

    used_before = c.allocator.num_used
    c.remove_sequence(s)
    assert c.allocator.num_used == used_before - 2, "blocks must return on removal"
    print("[block table] grows by blocks on demand, releases on removal")


# --- 2. numerical parity vs the contiguous cache ---------------------------

def test_paged_prefill_matches_contiguous(weights, tok):
    """Same prompt through both caches must give the same logits — the block
    indirection changes where bytes live, not what is computed."""
    cf = M.QwenConfig()
    max_diff = 0.0
    for prompt in cfg.PROMPTS:
        ids = encode_prompt(tok, prompt)
        seq = ids.shape[1]

        contig = KVCache(cf.num_layers, 1, cf.num_kv_heads, seq + 8, cf.head_dim)
        rope = M.build_rope_cache(seq + 8, cf.head_dim, cf.rope_theta)
        ref = M.forward_cached(ids, weights, cf, contig, 0, rope)

        paged = PagedKVCache(cf.num_layers, 64, 16, cf.num_kv_heads, cf.head_dim)
        sid = [paged.add_sequence()]
        start = torch.zeros(1, dtype=torch.long, device=ids.device)
        got = M.forward_paged(ids, weights, cf, paged, sid, start, rope)

        max_diff = max(max_diff, (ref.float() - got.float()).abs().max().item())

    print(f"\n[paged prefill] vs contiguous cache: max abs diff {max_diff:.2e}")
    assert max_diff == 0.0, "paging must not change the computation"


@pytest.mark.slow
def test_paged_decode_matches_fixture(weights, tok):
    """Full greedy decode through the paged cache vs the Phase 1 fixture."""
    cf = M.QwenConfig()
    ref = load_reference()

    divergences = []
    total = 0
    for i, (prompt, ref_p) in enumerate(zip(cfg.PROMPTS, ref["prompts"])):
        ids = encode_prompt(tok, prompt)
        got = M.generate_paged(ids, weights, cf, cfg.PARITY_NEW_TOKENS)[0]
        total += cfg.PARITY_NEW_TOKENS
        ok, div = compare_tokens(ref_p["greedy_ids"], got)
        print(f"[{i}] {'OK' if ok else f'diverges@{div}'}  {prompt!r}")
        if not ok:
            gap = abs(float(ref_p["topk_vals"][div][0]) -
                      float(ref_p["topk_vals"][div][1]))
            divergences.append((i, div, gap))

    rate = len(divergences) / total * 100
    print(f"\n[acceptance] {len(divergences)} divergence(s) in {total} tokens ({rate:.1f}%)")
    for i, div, gap in divergences:
        print(f"   prompt {i} step {div}: reference top-1/top-2 gap {gap:.4f}")
    real = [d for d in divergences if d[2] >= NEAR_TIE_GAP]
    assert not real, f"non-near-tie divergences: {real}"
    assert rate < 1.0


def test_paged_handles_uneven_lengths(weights, tok):
    """The point of paging: sequences of different lengths in one batch, each
    holding only the blocks it needs. Each row must match a solo run."""
    cf = M.QwenConfig()
    prompts = cfg.PROMPTS[:3]
    id_list = [encode_prompt(tok, p) for p in prompts]
    lens = [int(t.shape[1]) for t in id_list]
    assert len(set(lens)) > 1, "test needs prompts of differing length"

    paged = PagedKVCache(cf.num_layers, 128, 16, cf.num_kv_heads, cf.head_dim)
    rope = M.build_rope_cache(max(lens) + 8, cf.head_dim, cf.rope_theta)

    solo = []
    for ids in id_list:
        sid = [paged.add_sequence()]
        start = torch.zeros(1, dtype=torch.long, device=ids.device)
        solo.append(M.forward_paged(ids, weights, cf, paged, sid, start, rope))
        paged.remove_sequence(sid[0])

    frag = paged.fragmentation()
    print(f"\n[uneven] prompt lengths {lens}; after release "
          f"{frag['blocks_free']}/{paged.num_blocks} blocks free")
    assert frag["blocks_used"] == 0, "all blocks must return after removal"
    assert all(s.shape == (1, cf.vocab_size) for s in solo)


# --- 3. the memory claim ---------------------------------------------------

def test_paging_beats_contiguous_on_memory():
    """Quantify the win for a realistic mixed workload."""
    cf = M.QwenConfig()
    block_size = 16
    seq_lens = [50, 120, 30, 400, 75, 200, 60, 90]   # varied, as real traffic is
    max_seq = 512

    contiguous_slots = len(seq_lens) * max_seq
    blocks = sum((n + block_size - 1) // block_size for n in seq_lens)
    paged_slots = blocks * block_size
    used = sum(seq_lens)

    print(f"\n[memory] {len(seq_lens)} sequences, lengths {seq_lens}")
    print(f"   contiguous (max_seq={max_seq}): {contiguous_slots:6d} slots reserved")
    print(f"   paged      (block={block_size}):  {paged_slots:6d} slots held "
          f"({blocks} blocks)")
    print(f"   actually used:                   {used:6d} slots")
    print(f"   -> paging holds {contiguous_slots / paged_slots:.1f}x fewer slots; "
          f"internal fragmentation {(paged_slots - used) / paged_slots * 100:.1f}%")
    assert paged_slots < contiguous_slots
    assert paged_slots - used < len(seq_lens) * block_size, \
        "waste must be bounded by one partial block per sequence"
