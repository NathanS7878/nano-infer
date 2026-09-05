"""Phase 3: numerical parity for the custom CUDA kernels.

Rule 4 — correctness gates every optimization. A kernel that is fast and wrong is
worth zero, so parity runs before a single speed number is measured.

WHY THIS USES ULP, NOT AN ABSOLUTE TOLERANCE
--------------------------------------------
Phase 1 established that mirroring a reference op-for-op gives bit-identical
results, and that differences appear only when the STRUCTURE diverges. These
kernels are the first genuine restructuring in the project: the fused RMSNorm
sums 896 squares through a warp-tree reduction, while PyTorch uses its own
reduction order. Floating-point addition is not associative, so a different
summation order legitimately produces a different last bit.

The first version of this file used a 1e-3 absolute tolerance and failed at
0.00195. Measuring instead of loosening showed why: 99.997% of elements were
bit-identical, the rest differed by exactly ONE ULP — the smallest difference
fp16 can represent — and the failing element had magnitude 2.19, where one ULP
IS 0.00195. The tolerance was wrong, not the kernel.

So correctness here is stated in units the hardware actually uses:

    max ULP distance <= 2        (adjacent representable fp16 values)
    >= 99% of elements exact     (bit-identical)
    max relative error <= 4 eps  (eps = 9.77e-04 for fp16)

That is a stricter and more meaningful bar than any absolute number, and it does
not quietly widen when a kernel gets worse.

Run:  python -m pytest tests/test_kernels.py -v -s
"""
from __future__ import annotations

import pytest
import torch

from nano_infer import config as cfg
from nano_infer import model as M

kernels = pytest.importorskip("nano_infer.kernels")

MAX_ULP = 2
MIN_EXACT_FRAC = 0.99
MAX_REL_EPS = 4.0

SHAPES = [
    (1, 896),         # single token, the model's hidden size
    (34, 896),        # a prompt
    (32 * 128, 896),  # a full decode batch worth of rows
    (7, 4864),        # SwiGLU width, not a multiple of the block size
    (3, 128),         # narrow row (fewer elements than threads)
    (5, 1),           # degenerate width
]


def _ulp_key(t: torch.Tensor) -> torch.Tensor:
    """Map fp16 bit patterns to a monotonically ordered integer key.

    Positive floats already order correctly as signed integers; negatives are
    reversed, so they are reflected. Adjacent representable values then differ
    by exactly 1, which is what makes a ULP distance meaningful.
    """
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, (-0x8000) - i, i)


def compare(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """ULP distance, exact fraction, and relative error between two fp16 tensors."""
    ulp = (_ulp_key(ref) - _ulp_key(got)).abs()
    rel = ((ref.float() - got.float()).abs()
           / ref.float().abs().clamp(min=torch.finfo(torch.float16).tiny))
    return {
        "max_ulp": int(ulp.max()) if ulp.numel() else 0,
        "exact_frac": float((ulp == 0).float().mean()) if ulp.numel() else 1.0,
        "max_rel": float(rel.max()) if rel.numel() else 0.0,
        "max_abs": float((ref.float() - got.float()).abs().max()) if ulp.numel() else 0.0,
    }


def assert_parity(ref: torch.Tensor, got: torch.Tensor, label: str) -> dict:
    eps = torch.finfo(torch.float16).eps
    s = compare(ref, got)
    print(f"\n[{label}] max {s['max_ulp']} ulp, {s['exact_frac']*100:.3f}% exact, "
          f"rel {s['max_rel']:.2e} ({s['max_rel']/eps:.2f} eps), abs {s['max_abs']:.2e}")
    assert s["max_ulp"] <= MAX_ULP, f"{label}: {s['max_ulp']} ulp exceeds {MAX_ULP}"
    assert s["exact_frac"] >= MIN_EXACT_FRAC, f"{label}: only {s['exact_frac']:.4f} exact"
    assert s["max_rel"] <= MAX_REL_EPS * eps, f"{label}: rel {s['max_rel']:.2e} too large"
    return s


@pytest.fixture(scope="module")
def mod():
    return kernels.load()


@pytest.mark.parametrize("rows,hidden", SHAPES)
def test_rmsnorm_matches_pytorch(mod, rows, hidden):
    torch.manual_seed(0)
    x = torch.randn(rows, hidden, dtype=cfg.DTYPE, device=cfg.DEVICE)
    w = torch.randn(hidden, dtype=cfg.DTYPE, device=cfg.DEVICE)

    ref = M.rms_norm(x, w, 1e-6)
    got = mod.rmsnorm_forward(x, w, 1e-6)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert_parity(ref, got, f"rmsnorm {rows}x{hidden}")


def test_rmsnorm_3d_and_noncontiguous(mod):
    """The model calls RMSNorm on [batch, seq, hidden]; rank-3 input and a
    non-contiguous view must both work."""
    torch.manual_seed(0)
    x = torch.randn(4, 17, 896, dtype=cfg.DTYPE, device=cfg.DEVICE)
    w = torch.randn(896, dtype=cfg.DTYPE, device=cfg.DEVICE)

    got = mod.rmsnorm_forward(x, w, 1e-6)
    assert got.shape == x.shape
    assert_parity(M.rms_norm(x, w, 1e-6), got, "rmsnorm rank-3")

    sliced = x[:, ::2]                       # non-contiguous
    assert_parity(M.rms_norm(sliced, w, 1e-6),
                  mod.rmsnorm_forward(sliced, w, 1e-6), "rmsnorm non-contiguous")


def test_rmsnorm_extreme_magnitudes(mod):
    """fp16 overflows near 65504. Both reference and kernel accumulate the sum of
    squares in fp32; this catches a kernel that silently reduced in half precision,
    which would overflow to inf."""
    w = torch.ones(896, dtype=cfg.DTYPE, device=cfg.DEVICE)
    for scale in (1e-3, 1.0, 100.0):
        x = torch.full((8, 896), scale, dtype=cfg.DTYPE, device=cfg.DEVICE)
        got = mod.rmsnorm_forward(x, w, 1e-6)
        assert torch.isfinite(got).all(), f"non-finite output at scale {scale}"
        assert_parity(M.rms_norm(x, w, 1e-6), got, f"rmsnorm scale={scale:g}")


def test_rmsnorm_in_real_model_layer(mod):
    """Parity on real weights and real hidden states, not just random tensors."""
    weights = M.load_weights()
    cf = M.QwenConfig()
    torch.manual_seed(0)
    x = torch.randn(2, 34, cf.hidden_size, dtype=cfg.DTYPE, device=cfg.DEVICE)

    for layer in (0, 12, 23):
        w = weights[f"model.layers.{layer}.input_layernorm.weight"]
        assert_parity(M.rms_norm(x, w, cf.rms_norm_eps),
                      mod.rmsnorm_forward(x, w, cf.rms_norm_eps),
                      f"rmsnorm real layer {layer}")


# --- kernel 2: fused SwiGLU -------------------------------------------------
#
# RMSNorm's only source of divergence was reduction ORDER. SwiGLU has no
# reduction at all — it is purely elementwise — so in principle every element
# should be bit-identical. The risk here is different: `silu` calls `exp`, and
# an accidental `--use_fast_math` or `__expf` would silently swap in a
# lower-precision transcendental. That would still look "close enough" under an
# absolute tolerance while being a different function. The ULP bar catches it.

SWIGLU_SHAPES = [
    (1, 4864),        # decode, batch 1 — the real MLP width
    (32, 4864),       # decode, batch 32
    (34, 4864),       # prefill, batch 1
    (7, 4864),        # rows not a multiple of anything
    (3, 100),         # width not divisible by 8 -> scalar fallback path
    (5, 1),           # degenerate width
]


@pytest.mark.parametrize("rows,width", SWIGLU_SHAPES)
def test_swiglu_matches_pytorch(mod, rows, width):
    torch.manual_seed(0)
    gate = torch.randn(rows, width, dtype=cfg.DTYPE, device=cfg.DEVICE)
    up = torch.randn(rows, width, dtype=cfg.DTYPE, device=cfg.DEVICE)

    ref = torch.nn.functional.silu(gate) * up
    got = mod.swiglu_forward(gate, up)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert_parity(ref, got, f"swiglu {rows}x{width}")


def test_swiglu_3d_and_noncontiguous(mod):
    """The model calls this on [batch, seq, intermediate]."""
    torch.manual_seed(0)
    gate = torch.randn(4, 17, 4864, dtype=cfg.DTYPE, device=cfg.DEVICE)
    up = torch.randn(4, 17, 4864, dtype=cfg.DTYPE, device=cfg.DEVICE)

    got = mod.swiglu_forward(gate, up)
    assert got.shape == gate.shape
    assert_parity(torch.nn.functional.silu(gate) * up, got, "swiglu rank-3")

    g2, u2 = gate[:, ::2], up[:, ::2]        # non-contiguous views
    assert_parity(torch.nn.functional.silu(g2) * u2,
                  mod.swiglu_forward(g2, u2), "swiglu non-contiguous")


def test_swiglu_saturating_inputs(mod):
    """silu saturates in both directions: for very negative z it must go to 0,
    for very positive it must approach z. exp(-z) overflows to inf for z around
    -88 in fp32; the reference handles that by division (x/inf -> 0), and the
    kernel must reach zero the same way rather than producing NaN."""
    for val in (-60000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 60000.0):
        gate = torch.full((4, 512), val, dtype=cfg.DTYPE, device=cfg.DEVICE)
        up = torch.full((4, 512), 1.0, dtype=cfg.DTYPE, device=cfg.DEVICE)
        got = mod.swiglu_forward(gate, up)
        ref = torch.nn.functional.silu(gate) * up
        assert torch.isnan(got).sum() == torch.isnan(ref).sum(), f"NaN mismatch at {val}"
        assert_parity(ref, got, f"swiglu gate={val:g}")


def test_swiglu_in_real_mlp(mod):
    """Parity on the actual tensors the MLP produces, not random ones. Real
    activations are not standard-normal, and the exp argument range matters."""
    weights = M.load_weights()
    cf = M.QwenConfig()
    torch.manual_seed(0)
    x = torch.randn(1, 34, cf.hidden_size, dtype=cfg.DTYPE, device=cfg.DEVICE)

    for layer in (0, 12, 23):
        p = f"model.layers.{layer}.mlp."
        gate = torch.nn.functional.linear(x, weights[p + "gate_proj.weight"])
        up = torch.nn.functional.linear(x, weights[p + "up_proj.weight"])
        assert_parity(torch.nn.functional.silu(gate) * up,
                      mod.swiglu_forward(gate, up),
                      f"swiglu real layer {layer}")


# --- kernel 3: fused RoPE ---------------------------------------------------
#
# RoPE has a failure mode the other two kernels do not: it can be WRONG IN A WAY
# THAT STILL RUNS. Pairing dim i with i+head_dim/2 (HF/Llama) versus pairing
# adjacent dims (0,1),(2,3),... (the original paper's diagram) produces two
# different, equally finite, equally plausible-looking tensors. A model built on
# the wrong one still generates fluent text; it is just quietly wrong about
# position. Nothing raises. So the pairing is asserted directly below, against a
# hand-computed rotation, rather than trusted to fall out of an end-to-end check.

ROPE_SHAPES = [
    (1, 14, 1, 64),    # decode, batch 1 — query heads
    (1, 2, 1, 64),     # decode, batch 1 — KV heads (GQA: only 2)
    (32, 14, 1, 64),   # decode, batch 32
    (1, 14, 34, 64),   # prefill, batch 1
    (4, 14, 17, 64),   # prefill, batch 4
    (2, 3, 5, 8),      # tiny head_dim -> half=4 < VEC=8, scalar fallback path
    (1, 1, 1, 2),      # degenerate: one rotation pair
]


def _rope_tables(n, head_dim, batch=None):
    """cos/sin built the way the model builds them, then optionally broadcast to
    per-sequence [batch, n, head_dim] form."""
    cos, sin = M.build_rope_cache(n, head_dim, 1e6)
    if batch is None:
        return cos, sin
    # give each sequence its OWN position range, which is the case continuous
    # batching creates and a shared-position kernel would silently get wrong
    torch.manual_seed(1)
    starts = torch.randint(0, 64, (batch,))
    cos_all, sin_all = M.build_rope_cache(n + 64, head_dim, 1e6)
    cb = torch.stack([cos_all[s:s + n] for s in starts])
    sb = torch.stack([sin_all[s:s + n] for s in starts])
    return cb, sb


@pytest.mark.parametrize("batch,heads,n,head_dim", ROPE_SHAPES)
def test_rope_shared_positions(mod, batch, heads, n, head_dim):
    """Phase 1 path: one shared position range for the whole batch."""
    torch.manual_seed(0)
    x = torch.randn(batch, heads, n, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    cos, sin = _rope_tables(n, head_dim)

    ref, _ = M.apply_rope(x, x, cos, sin)
    got = mod.rope_forward(x, cos, sin)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert_parity(ref, got, f"rope shared {batch}x{heads}x{n}x{head_dim}")


@pytest.mark.parametrize("batch,heads,n,head_dim", ROPE_SHAPES)
def test_rope_per_sequence_positions(mod, batch, heads, n, head_dim):
    """Phase 2 step 3 path: every sequence sits at its own position. A kernel
    that indexed cos/sin by row instead of by (batch, position) passes the
    shared-position test above and fails here."""
    torch.manual_seed(0)
    x = torch.randn(batch, heads, n, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    cos, sin = _rope_tables(n, head_dim, batch=batch)

    ref, _ = M.apply_rope_positions(x, x, cos, sin)
    got = mod.rope_forward(x, cos, sin)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert_parity(ref, got, f"rope per-seq {batch}x{heads}x{n}x{head_dim}")


def test_rope_pairing_is_halves_not_adjacent(mod):
    """Assert the rotation convention directly.

    With cos=0 and sin=1 the transform collapses to exactly rotate_half:
        out = x*0 + rotate_half(x)*1 = [-x2, x1]
    That distinguishes the two conventions unambiguously. Under adjacent-pair
    rotation the same input would give [-x1, x0, -x3, x2, ...] instead.
    """
    head_dim = 8
    half = head_dim // 2
    x = torch.arange(1, head_dim + 1, dtype=cfg.DTYPE,
                     device=cfg.DEVICE).view(1, 1, 1, head_dim)
    cos = torch.zeros(1, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    sin = torch.ones(1, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)

    got = mod.rope_forward(x, cos, sin).flatten()
    expected = torch.cat([-x.flatten()[half:], x.flatten()[:half]])

    assert torch.equal(got, expected), (
        f"pairing is wrong: got {got.tolist()}, expected {expected.tolist()}. "
        "Halves pairing (i with i+half) is required; adjacent pairing would "
        "give a different, silently-plausible answer."
    )
    adjacent = torch.stack([-x.flatten()[1::2], x.flatten()[0::2]], dim=1).flatten()
    assert not torch.equal(got, adjacent), "kernel used adjacent-pair rotation"


def test_rope_preserves_vector_norm(mod):
    """RoPE is a rotation, so it must preserve the length of each head vector.
    This is a property test that does not depend on the reference at all — it
    would catch a kernel that matched PyTorch because both were wrong."""
    torch.manual_seed(0)
    x = torch.randn(2, 14, 33, 64, dtype=cfg.DTYPE, device=cfg.DEVICE)
    cos, sin = _rope_tables(33, 64)

    got = mod.rope_forward(x, cos, sin)
    before = x.float().norm(dim=-1)
    after = got.float().norm(dim=-1)
    rel = ((after - before).abs() / before.clamp(min=1e-6)).max().item()
    print(f"\n[rope norm preservation] max relative norm drift {rel:.2e}")
    assert rel < 2e-2, f"rotation changed vector length by {rel:.2e}"


def test_rope_noncontiguous(mod):
    """attention() produces q/k by transposing a view, so non-contiguous input
    is the normal case, not an edge case."""
    torch.manual_seed(0)
    x = torch.randn(2, 34, 14, 64, dtype=cfg.DTYPE,
                    device=cfg.DEVICE).transpose(1, 2)   # -> [2,14,34,64], non-contiguous
    assert not x.is_contiguous()
    cos, sin = _rope_tables(34, 64)

    ref, _ = M.apply_rope(x, x, cos, sin)
    assert_parity(ref, mod.rope_forward(x, cos, sin), "rope non-contiguous")


def test_rope_in_real_attention(mod):
    """Parity on the actual q/k tensors a real layer produces."""
    weights = M.load_weights()
    cf = M.QwenConfig()
    torch.manual_seed(0)
    seq = 34
    x = torch.randn(1, seq, cf.hidden_size, dtype=cfg.DTYPE, device=cfg.DEVICE)
    cos, sin = M.build_rope_cache(seq, cf.head_dim, cf.rope_theta)

    for layer in (0, 12, 23):
        p = f"model.layers.{layer}.self_attn."
        q = torch.nn.functional.linear(x, weights[p + "q_proj.weight"],
                                       weights[p + "q_proj.bias"])
        k = torch.nn.functional.linear(x, weights[p + "k_proj.weight"],
                                       weights[p + "k_proj.bias"])
        q = q.view(1, seq, cf.num_q_heads, cf.head_dim).transpose(1, 2)
        k = k.view(1, seq, cf.num_kv_heads, cf.head_dim).transpose(1, 2)

        ref_q, ref_k = M.apply_rope(q, k, cos, sin)
        assert_parity(ref_q, mod.rope_forward(q, cos, sin), f"rope real q L{layer}")
        assert_parity(ref_k, mod.rope_forward(k, cos, sin), f"rope real k L{layer}")


# --- kernel 4: fused decode attention with online softmax -------------------
#
# THIS IS THE FIRST KERNEL THAT CANNOT BE BIT-IDENTICAL, and the reason is
# structural rather than sloppy. The reference computes the QK product in fp16
# (cuBLAS rounds the matmul output to fp16), softmaxes in fp32, rounds the
# probabilities BACK to fp16, then accumulates against V. An online algorithm
# cannot round the probabilities the same way, because it does not know the
# normaliser until every key has been seen. The reduction order differs too.
#
# Gotcha #3 says: never loosen a tolerance to make a test pass — measure why it
# fails. So the bar is not loosened here, it is REPLACED with a stronger claim:
#
#     compute an fp64 ground truth, and require our error against it to be no
#     larger than the PyTorch reference's error against it.
#
# That is a harder test to pass than "close to PyTorch", because it cannot be
# satisfied by being wrong in the same direction as the reference. It says the
# divergence is the reference's rounding, not our bug.

ATTN_CASES = [
    (1, 14, 2, 64, 37),      # decode batch 1, one short sequence
    (32, 14, 2, 64, 128),    # decode batch 32, tile-aligned length
    (32, 14, 2, 64, 300),    # length spanning 3 tiles, last one partial
    (8, 14, 2, 64, 1),       # a single cached position
    (4, 4, 4, 64, 200),      # no GQA (n_rep == 1)
    (2, 8, 1, 64, 513),      # 8 query heads sharing one KV head
]


def _paged_setup(batch, kv_heads, head_dim, max_len, seed=0, shuffle=True):
    """Build a KV pool plus a slot table whose slots are deliberately scrambled.

    A contiguous slot table would let a kernel that ignored the table entirely
    still pass. Shuffling the slots means the indirection has to actually work.
    """
    torch.manual_seed(seed)
    slots_total = batch * max_len + 97          # slack, so unused slots exist
    k_pool = torch.randn(slots_total, kv_heads, head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    v_pool = torch.randn(slots_total, kv_heads, head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    if shuffle:
        perm = torch.randperm(slots_total, device=cfg.DEVICE)[:batch * max_len]
    else:
        perm = torch.arange(batch * max_len, device=cfg.DEVICE)
    slot_table = perm.reshape(batch, max_len).long()
    return k_pool, v_pool, slot_table


def _reference_decode(q, k_pool, v_pool, slot_table, lengths, scale):
    """The Phase 2 paged path, isolated: gather -> repeat_kv -> scores -> softmax
    -> blend. This is what the kernel replaces, and what it is timed against."""
    batch, q_heads, head_dim = q.shape
    kv_heads = k_pool.shape[1]
    max_len = slot_table.shape[1]
    n_rep = q_heads // kv_heads

    k = k_pool[slot_table].transpose(1, 2)       # [b, kv_heads, max_len, hd]
    v = v_pool[slot_table].transpose(1, 2)
    k = M.repeat_kv(k, n_rep)
    v = M.repeat_kv(v, n_rep)

    scores = (q.unsqueeze(2) @ k.transpose(-1, -2)).float() * scale

    pos = torch.arange(max_len, device=q.device).unsqueeze(0)
    allowed = pos < lengths.unsqueeze(1)         # [b, max_len]
    scores = scores.masked_fill(~allowed[:, None, None, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return (probs @ v).squeeze(2)                # [b, q_heads, hd]


def _ground_truth_cpu(q, k_pool, v_pool, slot_table, lengths, scale):
    """fp64 ground truth, computed ON THE CPU — deliberately, not incidentally.

    The obvious way to write this is to upcast the GPU tensors to float64 and
    reuse the reference above. That was the first version, and it was WRONG:
    `torch.softmax` in float64 on CUDA returns incorrect values on this machine
    for any tensor with more than one row. Measured on torch 2.6.0+cu124 /
    RTX 3070: at [16, 513] the elementwise error vs CPU is 1.2e-02 and rows sum
    to 0.68 instead of 1.0, while [1, 513] is exact to 1.7e-18. fp32 and fp16
    are unaffected.

    That produced a "ground truth" that disagreed with BOTH the PyTorch
    reference and this kernel by 0.18 — which is how it was caught, since two
    independent implementations agreeing with each other and not with the
    oracle indicts the oracle. test_fp64_softmax_on_cuda_is_unreliable below
    pins the bug so this never gets "simplified" back onto the GPU.
    """
    qd = q.double().cpu()
    kp = k_pool.double().cpu()
    vp = v_pool.double().cpu()
    st = slot_table.cpu()
    ln = lengths.cpu()

    batch, q_heads, head_dim = qd.shape
    n_rep = q_heads // kp.shape[1]
    out = torch.zeros(batch, q_heads, head_dim, dtype=torch.float64)
    for b in range(batch):
        L = int(ln[b])
        idx = st[b, :L]
        for h in range(q_heads):
            kvh = h // n_rep
            s = (kp[idx, kvh] @ qd[b, h]) * scale        # [L]
            out[b, h] = torch.softmax(s, dim=0) @ vp[idx, kvh]
    return out


def test_fp64_softmax_on_cuda_is_unreliable():
    """Pin the library bug that _ground_truth_cpu exists to avoid.

    If this test ever starts FAILING, torch fixed fp64 softmax on CUDA and the
    ground truth could move back to the GPU (it would be much faster). Until
    then, this documents why the slow CPU loop is there.
    """
    torch.manual_seed(0)
    x = torch.randn(16, 513, dtype=torch.float64, device=cfg.DEVICE)
    gpu = torch.softmax(x, dim=-1)
    cpu = torch.softmax(x.cpu(), dim=-1)

    row_sum_err = (gpu.sum(-1).cpu() - 1).abs().max().item()
    elem_err = (gpu.cpu() - cpu).abs().max().item()
    print(f"\n[fp64 softmax on CUDA] row-sum error {row_sum_err:.3e}, "
          f"elementwise vs CPU {elem_err:.3e} (CPU row-sum error "
          f"{(cpu.sum(-1) - 1).abs().max().item():.3e})")

    single = torch.softmax(x[:1], dim=-1)
    assert (single.sum(-1).cpu() - 1).abs().max().item() < 1e-12, \
        "single-row fp64 softmax is fine on CUDA — that part of the bug changed"
    assert row_sum_err > 1e-6, (
        "fp64 softmax on CUDA now normalises correctly for multi-row tensors. "
        "If so, _ground_truth_cpu can move back to the GPU — verify first."
    )


@pytest.mark.parametrize("batch,q_heads,kv_heads,head_dim,max_len", ATTN_CASES)
def test_decode_attention_correctness(mod, batch, q_heads, kv_heads,
                                      head_dim, max_len):
    """Correctness for a kernel that CANNOT be bit-identical to its reference.

    Kernels 1-3 were fusions: same arithmetic, fewer round trips, so 0 ULP was
    achievable and anything else was a bug. This kernel changes the algorithm,
    so a different answer is expected and the question becomes "different in
    which direction". Two claims are made, in order of importance:

      1. Against an fp64 ground truth, our error is NO LARGER than the fp16
         reference's. This is the real claim, and it cannot be satisfied by
         being wrong in the same direction as PyTorch — which is exactly what a
         plain "close to the reference" test would allow.

      2. Our divergence FROM the reference is no larger than the reference's own
         distance from the truth. That says the gap between us and PyTorch is
         accounted for by PyTorch's rounding, with nothing left over.

    Note what is NOT asserted: a relative-error bound against the reference. The
    output is a weighted average of v, so individual components pass through
    zero, and relative error against a near-zero reference value explodes while
    the absolute error stays negligible. That is the same trap that absolute
    tolerances set for RMSNorm in kernel 1, wearing the opposite hat. The
    numbers are printed for information; the bar is on the ground truth.
    """
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len)
    lengths = torch.randint(1, max_len + 1, (batch,), device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    truth = _ground_truth_cpu(q, k_pool, v_pool, slot_table, lengths, scale)
    ref = _reference_decode(q, k_pool, v_pool, slot_table, lengths, scale)
    got = mod.decode_attention_forward(q, k_pool, v_pool, slot_table, lengths, scale)

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.isfinite(got.float()).all(), "non-finite output"

    st = compare(ref, got)
    err_ref = (ref.double().cpu() - truth).abs().max().item()
    err_got = (got.double().cpu() - truth).abs().max().item()
    diff = (got.double().cpu() - ref.double().cpu()).abs().max().item()

    print(f"\n[decode attn {batch}x{q_heads}/{kv_heads}x{head_dim}, L<={max_len}] "
          f"{st['max_ulp']} ulp, {st['exact_frac']*100:.1f}% exact, "
          f"abs {st['max_abs']:.2e} | vs fp64 truth: PyTorch {err_ref:.3e}, "
          f"ours {err_got:.3e} (ratio {err_got / max(err_ref, 1e-12):.3f}), "
          f"ours-vs-PyTorch {diff:.3e}")

    assert err_got <= err_ref * 1.05 + 1e-6, (
        f"ours ({err_got:.3e}) is less accurate than the fp16 reference "
        f"({err_ref:.3e}) against fp64 ground truth")
    assert diff <= 2.5 * err_ref + 1e-5, (
        f"divergence from the reference ({diff:.3e}) is larger than the "
        f"reference's own error ({err_ref:.3e}) explains")


def test_decode_attention_correction_factor_regression(mod):
    """The online-softmax bug that produces finite, plausible, wrong output.

    One key late in the sequence is made to align hugely with q, so the running
    max jumps on the final tile and every earlier contribution must be rescaled
    by exp(m_old - m_new). A kernel that drops that correction passes every test
    above (where scores are similar in magnitude) and fails here — the Python
    prototype measured 2.881 absolute error from this bug alone.
    """
    batch, q_heads, kv_heads, head_dim, max_len = 2, 14, 2, 64, 300
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len)
    lengths = torch.full((batch,), max_len, device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    # make the LAST cached key of each sequence align strongly with query head 0
    for b in range(batch):
        slot = int(slot_table[b, max_len - 1])
        k_pool[slot, 0] = (q[b, 0].float() * 8.0).to(cfg.DTYPE)

    truth = _ground_truth_cpu(q, k_pool, v_pool, slot_table, lengths, scale)
    got = mod.decode_attention_forward(q, k_pool, v_pool, slot_table, lengths, scale)
    err = (got.double().cpu() - truth).abs().max().item()
    print(f"\n[decode attn late-max-jump] max abs error vs fp64 truth {err:.3e}")
    assert err < 1e-2, (
        f"error {err:.3e} on a late max jump — the online-softmax correction "
        "factor exp(m_old - m_new) is likely missing or misapplied"
    )


def test_decode_attention_respects_lengths(mod):
    """Positions at or beyond a sequence's length must contribute nothing.

    Checked by poisoning the slots past each length with huge values: if the
    kernel read them, they would dominate the softmax and the output would move.
    """
    batch, q_heads, kv_heads, head_dim, max_len = 4, 14, 2, 64, 256
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len)
    lengths = torch.tensor([1, 63, 128, 200], device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    clean = mod.decode_attention_forward(q, k_pool, v_pool, slot_table, lengths, scale)

    for b in range(batch):
        for p in range(int(lengths[b]), max_len):
            k_pool[slot_table[b, p]] = 300.0
            v_pool[slot_table[b, p]] = -300.0
    poisoned = mod.decode_attention_forward(q, k_pool, v_pool, slot_table, lengths, scale)

    assert torch.equal(clean, poisoned), (
        "output changed when out-of-range slots were poisoned: the kernel is "
        "reading past a sequence's length"
    )


def test_decode_attention_uses_the_slot_table(mod):
    """A kernel that ignored the slot table and read the pool contiguously would
    pass every test above if the table happened to be identity. Permuting the
    table must permute which keys are attended to."""
    batch, q_heads, kv_heads, head_dim, max_len = 2, 14, 2, 64, 64
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len,
                                              shuffle=False)
    lengths = torch.full((batch,), max_len, device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    identity = mod.decode_attention_forward(q, k_pool, v_pool, slot_table, lengths, scale)

    shuffled_table = slot_table.clone()
    shuffled_table[0] = slot_table[0].flip(0)     # reverse sequence 0's slots
    shuffled = mod.decode_attention_forward(q, k_pool, v_pool, shuffled_table,
                                            lengths, scale)

    # Attention is permutation-invariant over keys, so reversing the ORDER alone
    # must NOT change the answer — that is a real property worth asserting.
    assert_parity(identity[0], shuffled[0], "decode attn key-order invariance")

    # But pointing a sequence at a different SET of slots must change it.
    other_table = slot_table.clone()
    other_table[0] = slot_table[0] + max_len
    other = mod.decode_attention_forward(q, k_pool, v_pool, other_table,
                                         lengths, scale)
    assert not torch.equal(identity[0], other[0]), (
        "output did not change when sequence 0 was pointed at different slots: "
        "the kernel is ignoring the slot table"
    )


def test_decode_attention_block_size_invariance(mod):
    """The block size is a tuning knob (it sets the tile width), so it must not
    change the answer — only the speed.

    It is not entirely free of numerics: the tile width changes how many times
    the online-softmax correction is applied, and therefore the fp32 rounding
    order. So the requirement is agreement to within fp16 resolution, not
    bit-identity, and any drift LARGER than that means the recurrence is wrong
    for some tile count rather than merely reassociated.
    """
    batch, q_heads, kv_heads, head_dim, max_len = 8, 14, 2, 64, 700
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len)
    lengths = torch.randint(1, max_len + 1, (batch,), device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    truth = _ground_truth_cpu(q, k_pool, v_pool, slot_table, lengths, scale)
    base = None
    for bs in (64, 128, 256, 512, 1024):
        out = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                           lengths, scale, bs)
        err = (out.double().cpu() - truth).abs().max().item()
        print(f"\n[decode attn block_size={bs:4d}] err vs fp64 truth {err:.3e}")
        assert err < 5e-3, f"block_size {bs} is not just a tuning knob: err {err:.3e}"
        if base is None:
            base = out
        else:
            drift = (out.float() - base.float()).abs().max().item()
            assert drift < 5e-3, f"block_size {bs} changed the answer by {drift:.3e}"


# Head-group fusion (kernel 4b). The dispatcher picks between one block per
# (sequence, query head) and one block per (sequence, KV head) on block count,
# so BOTH paths have to be forced explicitly or the tests only ever exercise
# whichever one the heuristic happens to choose for the case's shape.
GROUPED_CASES = ATTN_CASES + [
    (4, 8, 2, 64, 700),      # n_rep 4
    (3, 16, 2, 64, 129),     # n_rep 8, block wider than the sequence
    (5, 14, 7, 64, 260),     # n_rep 2, the shallowest grouping
    (8, 14, 2, 64, 1),       # a single cached position, grouped
]


@pytest.mark.parametrize("batch,q_heads,kv_heads,head_dim,max_len", GROUPED_CASES)
def test_decode_attention_grouped_matches_per_head(mod, batch, q_heads, kv_heads,
                                                   head_dim, max_len):
    """The head-group fused kernel must be no less accurate than the one it
    replaces, on the same input.

    The two kernels compute the same recurrence; they differ only in WHO
    computes it. The per-query-head kernel gives each of the n_rep query heads
    sharing a KV head its own block, so each block re-reads the same K and V.
    The grouped kernel gives them one block between them, reads each K element
    once into a register and feeds it to n_rep dot products.

    That means the arithmetic per head is unchanged, and the bar is the strict
    one: our error against the fp64 ground truth must not exceed the fp16
    reference's — the same bar the per-query-head kernel is held to. The two
    kernels may still differ from EACH OTHER, because the dispatcher picks a
    different block size for each and the tile width sets how often the online
    correction is applied.
    """
    if q_heads == kv_heads:
        pytest.skip("n_rep == 1: nothing to group, the dispatcher cannot fuse")

    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len)
    lengths = torch.randint(1, max_len + 1, (batch,), device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    truth = _ground_truth_cpu(q, k_pool, v_pool, slot_table, lengths, scale)
    ref = _reference_decode(q, k_pool, v_pool, slot_table, lengths, scale)
    plain = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                         lengths, scale, 0, -1)
    grouped = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                           lengths, scale, 0, 1)

    assert grouped.shape == plain.shape and grouped.dtype == plain.dtype
    assert torch.isfinite(grouped.float()).all(), "non-finite grouped output"

    err_ref = (ref.double().cpu() - truth).abs().max().item()
    err_plain = (plain.double().cpu() - truth).abs().max().item()
    err_grp = (grouped.double().cpu() - truth).abs().max().item()
    drift = (grouped.double() - plain.double()).abs().max().item()

    print(f"\n[grouped {batch}x{q_heads}/{kv_heads}x{head_dim}, L<={max_len}, "
          f"n_rep {q_heads // kv_heads}] vs fp64 truth: PyTorch {err_ref:.3e}, "
          f"per-head {err_plain:.3e}, grouped {err_grp:.3e} "
          f"(ratio {err_grp / max(err_ref, 1e-12):.3f}), "
          f"grouped-vs-per-head {drift:.3e}")

    assert err_grp <= err_ref * 1.05 + 1e-6, (
        f"grouped kernel is less accurate than the fp16 reference it replaces: "
        f"{err_grp:.3e} vs {err_ref:.3e}"
    )
    assert err_grp <= err_plain * 1.05 + 1e-6, (
        f"head-group fusion cost accuracy against the per-query-head kernel: "
        f"{err_grp:.3e} vs {err_plain:.3e}. The two run the same recurrence, so "
        f"a real gap means the fusion changed the math, not just the schedule."
    )
    assert drift <= max(err_ref, err_plain) * 4 + 1e-5, (
        f"the two paths disagree by {drift:.3e}, more than either one's own "
        f"distance from the truth accounts for"
    )


def test_decode_attention_grouped_reduction_is_per_head(mod):
    """The grouped kernel reduces n_rep softmaxes in one block. Prove they stay
    SEPARATE.

    This is the failure mode the fusion invites and that an averaged error
    metric would hide: a reduction that accidentally spans the n_rep heads
    sharing a block as well as the positions still produces finite, plausible
    output — every head is still some combination of V rows, just the wrong one.
    So each query head is aimed at a DIFFERENT cached position, hard enough that
    its softmax is a delta, and its output must equal that position's V row.

    Note what this does and does not catch. A leaked running SUM (l) or a leaked
    accumulator scales or mixes the heads and shows up immediately. A leaked
    running MAX would NOT: online softmax subtracts m from the scores and
    divides by a sum computed with the same m, so an m that is too large cancels
    exactly. That is a property of the algorithm, not a gap in the test — an
    over-large m costs precision, not correctness, and the fp64-truth tests above
    are what bound the precision.

    The score gap has to be genuinely wide. An earlier version of this test used
    a gap of 7.5, which leaves exp(0)*63 / (exp(7.5) + 63) = 3.4% of the softmax
    mass spread across the non-target positions — enough to move the output by
    0.14 and fail a test that was measuring the construction, not the kernel.
    """
    batch, q_heads, kv_heads, head_dim, L = 8, 14, 2, 64, 64
    n_rep = q_heads // kv_heads
    torch.manual_seed(0)

    k_pool = torch.zeros(batch * L, kv_heads, head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    v_pool = torch.randn(batch * L, kv_heads, head_dim,
                         dtype=cfg.DTYPE, device=cfg.DEVICE)
    slot_table = torch.randperm(batch * L, device=cfg.DEVICE).reshape(batch, L).long()
    lengths = torch.full((batch,), L, device=cfg.DEVICE).long()

    # position p of every sequence gets the one-hot key 50 * e_(p % head_dim);
    # L == head_dim, so exactly one position carries each basis vector
    for b in range(batch):
        for p in range(L):
            k_pool[slot_table[b, p], :, p % head_dim] = 50.0

    # query head h is 500 * e_(target[h]), so its score at position target[h] is
    # 500*50/8 = 3125 and 0 everywhere else. exp(-3125) underflows to zero, so
    # the softmax is an exact delta and the output must be exactly one V row.
    # Both 500 and 50 are exactly representable in fp16 and their product is far
    # inside float range, where the kernel accumulates.
    target = [(h * 7 + 3) % head_dim for h in range(q_heads)]
    q = torch.zeros(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    for h in range(q_heads):
        q[:, h, target[h]] = 500.0
    scale = head_dim ** -0.5

    got = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                       lengths, scale, 0, 1).float()

    worst = 0.0
    for b in range(batch):
        for h in range(q_heads):
            kvh = h // n_rep
            want = v_pool[slot_table[b, target[h]], kvh].float()
            worst = max(worst, (got[b, h] - want).abs().max().item())
    print(f"\n[grouped per-head separation] worst deviation from the selected "
          f"V row: {worst:.3e}")
    assert worst < 1e-2, (
        f"grouped heads are contaminating each other: {worst:.3e}. A reduction "
        f"in the grouped kernel is leaking across the n_rep heads that share a "
        f"block instead of staying per-head."
    )


def test_decode_attention_grouped_block_size_invariance(mod):
    """Same knob, same requirement, on the grouped path.

    The grouped kernel picks its block size from a total-thread budget rather
    than a constant, so it lands on different widths than the per-query-head
    kernel does. That makes this the test that would catch a shared-memory
    layout that is only correct at one width — the tile arrays there are
    R*threads wide, so their offsets move with the block size.
    """
    batch, q_heads, kv_heads, head_dim, max_len = 8, 14, 2, 64, 700
    torch.manual_seed(0)
    q = torch.randn(batch, q_heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, kv_heads, head_dim, max_len)
    lengths = torch.randint(1, max_len + 1, (batch,), device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    truth = _ground_truth_cpu(q, k_pool, v_pool, slot_table, lengths, scale)
    base = None
    for bs in (64, 128, 256, 512, 1024):
        out = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                           lengths, scale, bs, 1)
        err = (out.double().cpu() - truth).abs().max().item()
        print(f"\n[grouped block_size={bs:4d}] err vs fp64 truth {err:.3e}")
        assert err < 5e-3, f"grouped block_size {bs} is not just a knob: {err:.3e}"
        if base is None:
            base = out
        else:
            drift = (out.float() - base.float()).abs().max().item()
            assert drift < 5e-3, f"grouped block_size {bs} changed the answer"


def test_decode_attention_grouped_falls_back_when_it_cannot_group(mod):
    """n_rep == 1 has nothing to fuse, and asking for it must fail loudly.

    A silent fallback would be worse than an error: the benchmark's whole job is
    to compare the two paths, and a forced flag that quietly does nothing turns
    an A/B into an A/A that reads as "the fusion bought nothing".
    """
    batch, heads, head_dim, L = 4, 4, 64, 200
    torch.manual_seed(0)
    q = torch.randn(batch, heads, head_dim, dtype=cfg.DTYPE, device=cfg.DEVICE)
    k_pool, v_pool, slot_table = _paged_setup(batch, heads, head_dim, L)
    lengths = torch.full((batch,), L, device=cfg.DEVICE).long()
    scale = head_dim ** -0.5

    with pytest.raises(RuntimeError, match="grouped path does not apply"):
        mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                     lengths, scale, 0, 1)

    # auto and forced-off must both still work and agree
    auto = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                        lengths, scale, 0, 0)
    off = mod.decode_attention_forward(q, k_pool, v_pool, slot_table,
                                       lengths, scale, 0, -1)
    assert torch.equal(auto, off), (
        "with n_rep == 1 the auto path must be the per-query-head kernel"
    )
