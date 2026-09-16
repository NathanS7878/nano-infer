"""Reproduce the tensor-core (HMMA) dead end recorded in ROADMAP #40.

Not a benchmark -- an experiment with a negative result, kept runnable because
hard rule 3 applies to negative claims too. It builds
nano_infer/kernels/experimental/hmma_probe.cu as its OWN extension, separate
from the engine's kernels, and runs four checks:

  1. step 1 -- an fp16 X.W^T on tensor-core fragments vs torch, on the model's
     real projection shapes. Error against an fp64 CPU ground truth, next to
     cuBLAS's own fp16 error on the same inputs.
  2. every fragment interpretation (A row/col x B row/col x store row/col x
     aliased/separate accumulator) against every candidate product, on
     small-integer inputs whose products are exact in fp16;
  3. all-zero inputs, three calls each -- a multiply must return zero, every time;
  4. bilinearity -- a multiply must satisfy y(2X) == 2 y(X).

If NVIDIA's fragments ever start behaving as dense matmuls here, checks 2-4
will say so, and the tensor-core quantized matmul becomes worth another look.

Run:  python -m bench.hmma_probe
"""
from __future__ import annotations

import json

import torch

from nano_infer import config as cfg
from nano_infer import kernels as K


def build():
    K._ensure_build_env()
    from torch.utils.cpp_extension import load
    return load(
        name="nano_infer_hmma_probe",
        sources=[str(K._KERNEL_DIR / "experimental" / "hmma_probe.cu")],
        extra_cflags=K._CXX_FLAGS,
        extra_cuda_cflags=K._NVCC_FLAGS,
        extra_ldflags=K._cuda_link_flags(),
        verbose=False,
    )


def main():
    probe = build()
    results = {}

    # --- 1. step 1 on real projection shapes -----------------------------------
    print("\n1. fp16 X.W^T on tensor-core fragments vs fp64 truth")
    print(f"{'B':>4}{'N':>6}{'K':>6}{'max |truth|':>13}{'cuBLAS err':>12}{'fragment err':>14}{'err / scale':>13}")
    step1 = []
    for B, N, Kd in [(16, 32, 48), (16, 896, 896), (32, 128, 896), (32, 4864, 896), (32, 896, 4864)]:
        g = torch.Generator().manual_seed(B * 7 + N)
        x = torch.randn(B, Kd, generator=g).half().to(cfg.DEVICE)
        w = torch.randn(N, Kd, generator=g).half().to(cfg.DEVICE)
        truth = x.double().cpu() @ w.double().cpu().T
        err_cublas = ((x @ w.T).double().cpu() - truth).abs().max().item()
        err_frag = (probe.hmma_matmul_f16(x, w).double().cpu() - truth).abs().max().item()
        scale = truth.abs().max().item()
        print(f"{B:>4}{N:>6}{Kd:>6}{scale:>13.3e}{err_cublas:>12.3e}{err_frag:>14.3e}{err_frag / scale:>12.1%}")
        step1.append({"B": B, "N": N, "K": Kd, "max_abs_truth": scale,
                      "cublas_err": err_cublas, "fragment_err": err_frag})
    results["step1_matmul"] = step1

    # --- 2. every interpretation vs every candidate ------------------------------
    print("\n2. every fragment interpretation vs every candidate product (exact match)")
    g = torch.Generator().manual_seed(0)
    X = torch.randint(-3, 4, (16, 16), generator=g).double()
    W = torch.randint(-3, 4, (16, 16), generator=g).double()
    cands = {"X.W^T (goal)": X @ W.T, "(X.W^T)^T": (X @ W.T).T, "X.W": X @ W,
             "(X.W)^T": (X @ W).T, "X^T.W": X.T @ W, "X^T.W^T": X.T @ W.T,
             "W.X": W @ X, "W^T.X": W.T @ X, "W.X^T": W @ X.T, "W^T.X^T": W.T @ X.T}
    modes = []
    for mode in range(16):
        y = probe.hmma_layout_probe(X.half().to(cfg.DEVICE), W.half().to(cfg.DEVICE),
                                    mode).double().cpu()
        hits = [n for n, c in cands.items() if torch.equal(y, c)]
        modes.append({"mode": mode, "A_col": bool(mode & 1), "B_col": bool(mode & 2),
                      "store_col": bool(mode & 4), "separate_C": bool(mode & 8),
                      "matches": hits})
    matched = [m for m in modes if m["matches"]]
    print(f"   {len(matched)} of 16 interpretations match any candidate")
    results["interpretations"] = modes

    # --- 3. zero inputs ----------------------------------------------------------
    print("\n3. all-zero inputs (a multiply must return exactly zero, every call)")
    Z = torch.zeros(16, 16, dtype=torch.float16, device=cfg.DEVICE)
    zero = []
    for mode in (0, 8):
        outs = [probe.hmma_layout_probe(Z, Z, mode).float().cpu() for _ in range(3)]
        rec = {"mode": mode, "nonzero_cells": int((outs[0] != 0).sum()),
               "max_abs": outs[0].abs().max().item(),
               "deterministic": all(torch.equal(outs[0], o) for o in outs[1:])}
        zero.append(rec)
        print(f"   mode {mode}: {rec['nonzero_cells']} nonzero cells, max {rec['max_abs']:.3e}, "
              f"same across 3 calls: {rec['deterministic']}")
    results["zero_inputs"] = zero

    # --- 4. bilinearity ----------------------------------------------------------
    y1 = probe.hmma_layout_probe(X.half().to(cfg.DEVICE), W.half().to(cfg.DEVICE), 8).double().cpu()
    y2 = probe.hmma_layout_probe((2 * X).half().to(cfg.DEVICE), W.half().to(cfg.DEVICE), 8).double().cpu()
    bilinear = torch.equal(y2, 2 * y1)
    print(f"\n4. bilinearity: y(2X) == 2 y(X): {bilinear}")
    results["bilinear"] = bilinear

    works = (bool(matched) and all(z["nonzero_cells"] == 0 for z in zero) and bilinear)
    verdict = ("fragments behave as a matmul -- revisit ROADMAP #40" if works else
               "fragments do NOT behave as a matmul as used here (ROADMAP #40 stands)")
    print(f"\nverdict: {verdict}")
    results["verdict"] = verdict

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "hmma_probe.json").write_text(json.dumps({
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "header": "<mma.h> -> crt/mma.h, crt/mma.hpp (CUDA 12.4, conda-forge)",
        **results}, indent=2), encoding="utf-8")
    print("Saved hmma_probe.json")


if __name__ == "__main__":
    main()
