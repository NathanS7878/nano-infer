"""Phase 4: weight-only quantization — INT8 per-channel and INT4 group-wise.

WHY WEIGHT-ONLY, AND WHY IT SHOULD HELP
---------------------------------------
Phase 1 measured the forward pass costing a flat ~38-40 ms from sequence length
32 to 512. Flat, because at those lengths the time is not spent on arithmetic —
it is spent streaming 988 MB of weights from VRAM once per forward pass. Phase 2
then measured decode holding flat at ~38 ms/step across a 32x batch range, the
same story: every decode step re-reads every weight to produce one token per
sequence.

That is the thing quantization attacks. Storing a weight in 4 bits instead of 16
cuts the bytes moved by 4x, and for a memory-bound workload bytes moved is time.
Activations stay in fp16 — quantizing them buys nothing here, because they are
tiny next to the weights and quantizing them costs accuracy for no bandwidth.

WHAT IS ACTUALLY QUANTIZED, AND WHY THE HEADLINE IS NOT 4x
-----------------------------------------------------------
    all weights, fp16                          988.1 MB
    2-D projection matrices (q/k/v/o/gate/up/down)  715.7 MB   72.4%
    tied embedding + lm_head                   272.4 MB   27.6%

Only the projections are quantized. The embedding matrix is TIED to the output
head in this model, so an error there is applied twice — once when a token is
looked up and again when the logits are produced — and it is the single most
quality-sensitive tensor in the network. Leaving it in fp16 is the standard
choice, and it caps what quantization can deliver:

    INT8 projections: 988.1 -> 630.8 MB   ->  1.57x smaller (measured)
    INT4 projections: 988.1 -> 459.7 MB   ->  2.15x smaller (measured)

Not 4x. Quoting "4x smaller with INT4" would be true only of the tensors that
were quantized -- and even they come out at 3.82x, not 4x, once the group
metadata is counted. Neither is the same claim as a 4x smaller model, and this
file exists partly to keep those three numbers distinguishable.

THE TWO SCHEMES
---------------
INT8, per output channel, SYMMETRIC:

    scale[i] = max|W[i, :]| / 127
    q[i, j]  = round(W[i, j] / scale[i])         in [-127, 127]

One scale per row. Symmetric (no zero point) because weight distributions are
close to zero-centred, and a zero point would cost an extra subtract per element
in the inner loop for very little accuracy.

INT4, group-wise, ASYMMETRIC:

    for each group g of 128 consecutive inputs within a row:
        lo, hi        = min(W[g]), max(W[g])
        scale[g]      = (hi - lo) / 15
        zero[g]       = round(-lo / scale[g])    so that 0 maps inside [0, 15]
        q[j]          = clamp(round(W[j]/scale[g]) + zero[g], 0, 15)

Asymmetric here because 4 bits is only 16 levels: throwing half of them away on
a sign that the data may not use symmetrically is a real loss. Grouping by 128
along the INPUT dimension means a group is contiguous in the direction the
matmul walks, so a kernel loads one scale and one zero and then consumes 128
weights with them.

Group-wise metadata is not free, and this file counts it rather than ignoring
it: per group of 128 weights we store 64 packed bytes + one fp16 scale (2 B) +
one uint8 zero (1 B) = 67 bytes, so the effective rate is **4.19 bits/weight**,
not 4.0. `bits_per_weight()` reports the measured figure and a test pins it.

THIS FILE IS THE REFERENCE, NOT THE FAST PATH
---------------------------------------------
`quantize_dequantize` returns plain fp16 tensors that have made the round trip
through the quantized representation. That is deliberately slow and deliberately
useless for speed — its entire job is to isolate the QUALITY cost so it can be
measured before any kernel exists. The packed representation and the fused
dequant-matmul kernel come after, and are validated against this.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import config as cfg

# Layers that get quantized: the 2-D projection matrices, and nothing else.
# Matching on suffix rather than a hand-listed set so a config change cannot
# silently leave a new projection unquantized.
QUANTIZABLE_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

INT4_GROUP = 128


def is_quantizable(name: str, tensor: torch.Tensor) -> bool:
    """True for the 2-D projection weights. Excludes embeddings (tied to the
    output head), norms, and biases."""
    return tensor.dim() == 2 and name.endswith(QUANTIZABLE_SUFFIXES)


# ---------------------------------------------------------------------------
# INT8, per output channel, symmetric
# ---------------------------------------------------------------------------

@dataclass
class Int8Tensor:
    q: torch.Tensor          # [out, in]  int8
    scale: torch.Tensor      # [out, 1]   fp16
    shape: tuple

    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * self.scale.element_size()

    def dequantize(self) -> torch.Tensor:
        return (self.q.to(self.scale.dtype) * self.scale).to(cfg.DTYPE)


def _scale_to_stored_dtype(scale: torch.Tensor) -> torch.Tensor:
    """Round a positive fp32 scale UP to the nearest value representable in
    cfg.DTYPE -- never down.

    WHY UP AND NOT TO-NEAREST (Gotcha #41). The scale is stored in the model
    dtype, so that is the number dequantization multiplies by, and codes must
    be chosen against it (Gotcha #23). But the *direction* of that rounding is
    not free. Both schemes size the grid so the extreme element of a row/group
    lands exactly on the last code:

        INT8: scale = max|w| / 127        ->  max|w| / scale == 127
        INT4: scale = (hi - lo) / 15      ->  (hi - lo) / scale == 15

    If the stored scale is even slightly SMALLER than the exact one, that
    quotient exceeds the last code, round() produces 128 (or 16), clamp pulls
    it back, and the extreme element reconstructs more than half a step away --
    the one guarantee round-to-nearest is supposed to give. Rounding the scale
    up instead makes the grid a hair wider than the data, so every element
    stays inside the code range and the half-step bound holds by construction.

    The cost is bounded and tiny: the grid step grows by at most one ULP of the
    stored dtype (2^-11 relative in fp16, 2^-8 in bf16), so the half-step bound
    itself widens by the same relative amount.

    WHY IT ONLY SURFACED ON bf16. In fp16 the downward rounding is at most
    2^-11 relative, which the test's 1%-of-a-rounding slack absorbed. bf16 has
    three fewer fraction bits -- 2^-8 -- and it does not. The bug was always
    there; fp16 hid it. Found by test_int4_error_is_bounded_by_half_a_step on
    Qwen2.5-1.5B/bf16: row 34, group 3, element 37, the group maximum, whose
    exact scale 0.5562500 stored as 0.5546875 (0.281% low) pushed its code to
    16 -> clamped to 15 -> 0.0115 steps past the bound.

    Implementation: for positive finite floats the IEEE-754 bit pattern is
    monotonic, so the next representable value up is the next integer bit
    pattern. fp16 and bf16 are both 16-bit, so one int16 view serves both.
    """
    down = scale.to(cfg.DTYPE)
    up = (down.view(torch.int16) + 1).view(cfg.DTYPE)
    return torch.where(down.float() < scale, up, down)


def quantize_int8(w: torch.Tensor) -> Int8Tensor:
    """Per-row symmetric INT8. `w` is [out, in].

    NOTE THE SCALE IS ROUNDED TO THE STORED DTYPE BEFORE IT IS USED, AND
    ROUNDED UP. The scale is *stored* in the model dtype, so that is the value
    dequantization (and the kernel) will multiply by. Choosing codes against a
    more precise fp32 scale than the one that will later be used introduces
    error for free: at q = 127 an fp16 scale rounding of 2^-11 relative shifts
    the reconstruction by ~0.06 of a step, which is enough to push
    round-to-nearest past its half-step guarantee. Found by
    test_int8_error_is_bounded_by_half_a_step, which failed on 535 elements
    before this line existed. The *direction* matters too -- see
    _scale_to_stored_dtype.
    """
    assert w.dim() == 2, "expected a 2-D weight"
    wf = w.float()
    scale = wf.abs().amax(dim=1, keepdim=True) / 127.0
    # A row of exact zeros would divide by zero; give it a harmless unit scale.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Quantize with exactly what we store (Gotcha #23). The stored precision is
    # the model dtype: fp16 on 0.5B, bf16 on Qwen2.5-1.5B, whose activations the
    # dequant-matmul kernel must match.
    scale = _scale_to_stored_dtype(scale)
    q = torch.round(wf / scale.float()).clamp_(-127, 127).to(torch.int8)
    return Int8Tensor(q=q, scale=scale, shape=tuple(w.shape))


# ---------------------------------------------------------------------------
# INT4, group-wise along the input dimension, asymmetric
# ---------------------------------------------------------------------------

@dataclass
class Int4Tensor:
    packed: torch.Tensor     # [out, in/2] uint8 — two nibbles per byte
    scale: torch.Tensor      # [out, groups] fp16
    zero: torch.Tensor       # [out, groups] uint8
    shape: tuple
    group: int

    def nbytes(self) -> int:
        return (self.packed.numel()
                + self.scale.numel() * self.scale.element_size()
                + self.zero.numel())

    def bits_per_weight(self) -> float:
        return self.nbytes() * 8 / (self.shape[0] * self.shape[1])

    def dequantize(self) -> torch.Tensor:
        return dequantize_int4(self)


def quantize_int4(w: torch.Tensor, group: int = INT4_GROUP) -> Int4Tensor:
    """Group-wise asymmetric INT4. `w` is [out, in]; `in` must divide by `group`."""
    assert w.dim() == 2, "expected a 2-D weight"
    out_f, in_f = w.shape
    assert in_f % group == 0, (
        f"input dim {in_f} is not a multiple of the group size {group}")

    g = in_f // group
    wf = w.float().reshape(out_f, g, group)

    lo = wf.amin(dim=2, keepdim=True)
    hi = wf.amax(dim=2, keepdim=True)

    # A CONSTANT group has hi == lo, so the range is zero. Substituting a unit
    # scale (the obvious guard) maps every element to code 0 and reconstructs
    # the whole group as ZERO -- silently destroying it. Instead, widen the
    # range so the constant itself lands on a grid point:
    #
    #   lo > 0:  range [lo, 2lo]  -> zero=0,  code 15, (15-0)*lo/15  = lo
    #   lo < 0:  range [lo, 0]    -> zero=15, code 0,  (0-15)*|lo|/15 = lo
    #   lo == 0: range [0, 1]     -> zero=0,  code 0,               = 0
    #
    # Found by test_int4_kernel_uses_per_group_scales, which used a constant
    # group precisely because it is the degenerate case a guard tends to botch.
    degenerate = (hi == lo)
    widened = lo + torch.where(lo.abs() > 0, lo.abs(), torch.ones_like(lo))
    hi = torch.where(degenerate, widened, hi)

    scale = (hi - lo) / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # Same rule as INT8: round the scale to its stored precision BEFORE
    # choosing codes and the zero point, so quantization and dequantization
    # agree on the grid -- and round it UP, so the group's extremes stay inside
    # the 0..15 code range instead of clamping past the half-step bound.
    stored_scale = _scale_to_stored_dtype(scale)
    scale = stored_scale.float()
    zero = torch.round(-lo / scale).clamp_(0, 15)

    q = torch.round(wf / scale + zero).clamp_(0, 15).to(torch.uint8)
    q = q.reshape(out_f, in_f)

    # Pack two 4-bit values per byte: even index in the low nibble, odd in the
    # high nibble. Low-first matters — the kernel unpacks in the same order.
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()

    return Int4Tensor(
        packed=packed,
        scale=stored_scale.reshape(out_f, g),
        zero=zero.reshape(out_f, g).to(torch.uint8),
        shape=(out_f, in_f),
        group=group,
    )


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """[out, in/2] uint8 -> [out, in] uint8 in 0..15."""
    out_f = packed.shape[0]
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.empty(out_f, in_features, dtype=torch.uint8, device=packed.device)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return q


def dequantize_int4(t: Int4Tensor) -> torch.Tensor:
    out_f, in_f = t.shape
    g = in_f // t.group
    q = unpack_int4(t.packed, in_f).reshape(out_f, g, t.group).float()
    scale = t.scale.reshape(out_f, g, 1).float()
    zero = t.zero.reshape(out_f, g, 1).float()
    return ((q - zero) * scale).reshape(out_f, in_f).to(cfg.DTYPE)


# ---------------------------------------------------------------------------
# The quality-measurement path: quantize, dequantize, hand back fp16
# ---------------------------------------------------------------------------

def quantize_dequantize(w: torch.Tensor, mode: str, group: int = INT4_GROUP):
    """Round-trip one weight through a quantized representation.

    Returns (fp16 tensor, quantized object). The tensor is what the model runs
    with when measuring quality; the object carries the true storage cost.
    """
    if mode == "int8":
        t = quantize_int8(w)
    elif mode == "int4":
        t = quantize_int4(w, group)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'int8' or 'int4'")
    return t.dequantize(), t


def quantize_weights(weights: dict, mode: str, group: int = INT4_GROUP) -> tuple:
    """Return (new weights dict, stats).

    Every quantizable projection is replaced by its round-tripped fp16 value;
    everything else is passed through untouched. Nothing about the model's
    execution changes, so any difference in output is attributable to the
    quantization error and to nothing else.
    """
    if mode == "fp16":
        total = sum(v.numel() * v.element_size() for v in weights.values())
        return dict(weights), {
            "mode": "fp16", "quantized_tensors": 0,
            "bytes_original": total, "bytes_quantized": total,
            "bytes_quantizable_original": 0, "bytes_quantizable_quantized": 0,
            "compression": 1.0, "bits_per_weight": 16.0,
            "max_rel_err": 0.0, "mean_rel_err": 0.0,
        }

    out = {}
    n_quant = 0
    total_orig = 0
    total_new = 0
    quant_orig = 0
    quant_new = 0
    quant_params = 0
    max_rel = 0.0
    rel_sum = 0.0

    for name, w in weights.items():
        nbytes = w.numel() * w.element_size()
        total_orig += nbytes
        if not is_quantizable(name, w):
            out[name] = w
            total_new += nbytes
            continue

        deq, t = quantize_dequantize(w, mode, group)
        out[name] = deq
        n_quant += 1
        quant_orig += nbytes
        quant_new += t.nbytes()
        total_new += t.nbytes()
        quant_params += w.numel()

        # relative reconstruction error, scaled by the tensor's own magnitude
        # (per-element relative error is meaningless where a weight is ~0)
        err = (w.float() - deq.float()).abs()
        denom = w.float().abs().amax().clamp(min=1e-12)
        max_rel = max(max_rel, float((err / denom).max()))
        rel_sum += float((err / denom).mean())

    return out, {
        "mode": mode,
        "quantized_tensors": n_quant,
        "bytes_original": total_orig,
        "bytes_quantized": total_new,
        "bytes_quantizable_original": quant_orig,
        "bytes_quantizable_quantized": quant_new,
        "compression": total_orig / total_new if total_new else 1.0,
        "quantizable_compression": quant_orig / quant_new if quant_new else 1.0,
        "bits_per_weight": quant_new * 8 / quant_params if quant_params else 0.0,
        "max_rel_err": max_rel,
        "mean_rel_err": rel_sum / n_quant if n_quant else 0.0,
    }


# ---------------------------------------------------------------------------
# The packed path: keep the quantized form, and ONLY the quantized form
# ---------------------------------------------------------------------------

def pack_weights(weights: dict, mode: str, group: int = INT4_GROUP) -> tuple:
    """Return (weights dict holding PACKED tensors, stats).

    Unlike `quantize_weights`, which round-trips back to fp16 to isolate the
    quality question, this keeps `Int8Tensor` / `Int4Tensor` objects. Nothing
    downstream ever sees a dequantized copy, which is the only configuration in
    which the memory saving is real.

    THE DESIGN DECISION THIS ENCODES. The fused kernel beats cuBLAS at batch 1-4
    and loses above it (Gotcha #25). Two responses were available:

      (a) always quantized  -> the VRAM saving is real; batched decode is slower
      (b) quantized below the crossover, fp16 above -> best speed, but BOTH
          copies must be resident, so the memory saving evaporates

    This project takes (a), because the memory reduction is the claim INT4
    actually delivers on this hardware, and (b) would keep the headline number
    while quietly making it false. The throughput cost is reported in the
    acceptance table rather than engineered around.
    """
    if mode == "fp16":
        total = sum(v.numel() * v.element_size() for v in weights.values())
        return dict(weights), {
            "mode": "fp16", "quantized_tensors": 0,
            "bytes_original": total, "bytes_quantized": total,
            "compression": 1.0, "bits_per_weight": 16.0,
        }

    out = {}
    n_quant = 0
    total_orig = 0
    total_new = 0
    quant_params = 0
    quant_new = 0

    for name, w in weights.items():
        nbytes = w.numel() * w.element_size()
        total_orig += nbytes
        if not is_quantizable(name, w):
            out[name] = w
            total_new += nbytes
            continue
        t = quantize_int8(w) if mode == "int8" else quantize_int4(w, group)
        out[name] = t
        n_quant += 1
        total_new += t.nbytes()
        quant_new += t.nbytes()
        quant_params += w.numel()

    return out, {
        "mode": mode,
        "quantized_tensors": n_quant,
        "bytes_original": total_orig,
        "bytes_quantized": total_new,
        "compression": total_orig / total_new if total_new else 1.0,
        "bits_per_weight": quant_new * 8 / quant_params if quant_params else 0.0,
    }


def is_packed(w) -> bool:
    return isinstance(w, (Int8Tensor, Int4Tensor))
