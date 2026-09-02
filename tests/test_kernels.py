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
