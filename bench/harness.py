"""The benchmark harness — the ruler.

Measures three things at several batch sizes, with the CUDA-correct timing that
makes the numbers real rather than fantasy:

    TTFT               time for prefill + the first decoded token
    inter-token        steady-state seconds per token during decode
    tokens/sec         (batch * new_tokens) / wall time for the full decode

Two correctness rules baked in:

  1. torch.cuda.synchronize() before EVERY timer stop. GPU kernel launches are
     asynchronous: the Python call returns immediately while the GPU works in the
     background. Stop the clock without syncing and you time the launch, not the
     compute — the classic way to report impossibly-fast fake numbers.

  2. Warmup runs, discarded. The first calls pay one-time costs (CUDA context
     creation, caching allocator warmup, kernel autotuning). Those are not what
     we want to measure, so we run a few and throw them away.

The harness is generic over a `generate_fn(input_ids, max_new_tokens) -> ids`
callable, so the exact same ruler measures HF today and our engine in Phase 5.

Run:  python -m bench.harness
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from nano_infer import config
from nano_infer.hf_ref import encode_prompt, hf_generate, load_hf

GenerateFn = Callable[[torch.Tensor, int], torch.Tensor]

# A single, fixed benchmark prompt, replicated to fill the batch. Same length
# across the batch => no padding => no pad positions to distort the timing.
BENCH_PROMPT = "Explain how a transformer neural network works, step by step."


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_generate(generate_fn: GenerateFn, input_ids: torch.Tensor,
                   max_new_tokens: int) -> float:
    """Wall-clock seconds for one generation call, CUDA-synchronized both ends."""
    _sync()
    t0 = time.perf_counter()
    generate_fn(input_ids, max_new_tokens)
    _sync()
    return time.perf_counter() - t0


@dataclass
class BenchResult:
    engine: str
    batch_size: int
    new_tokens: int
    ttft_ms: float                 # median across runs
    inter_token_ms: float          # median across runs
    tokens_per_sec: float          # median across runs
    variance_pct: float            # spread of the full-decode time across runs
    stable: bool                   # variance_pct < 3%
    runs: int
    ttft_ms_runs: list[float] = field(default_factory=list)
    full_ms_runs: list[float] = field(default_factory=list)


def _spread_pct(xs: list[float]) -> float:
    """(max - min) / median, as a percentage. Our stability metric."""
    med = statistics.median(xs)
    return (max(xs) - min(xs)) / med * 100.0 if med > 0 else 0.0


def benchmark_one(generate_fn: GenerateFn, base_ids: torch.Tensor, batch_size: int,
                  new_tokens: int, warmup: int, runs: int, engine: str) -> BenchResult:
    """Benchmark one (engine, batch_size) point."""
    input_ids = base_ids.repeat(batch_size, 1)  # [batch, seq], identical rows

    # --- warmup (discarded) ---
    for _ in range(warmup):
        _time_generate(generate_fn, input_ids, new_tokens)

    # --- measured runs ---
    ttft_s, full_s = [], []
    for _ in range(runs):
        ttft_s.append(_time_generate(generate_fn, input_ids, 1))
        full_s.append(_time_generate(generate_fn, input_ids, new_tokens))

    ttft_med = statistics.median(ttft_s)
    full_med = statistics.median(full_s)
    inter_med = (full_med - ttft_med) / max(new_tokens - 1, 1)
    tok_per_s = (batch_size * new_tokens) / full_med
    var = _spread_pct(full_s)

    return BenchResult(
        engine=engine,
        batch_size=batch_size,
        new_tokens=new_tokens,
        ttft_ms=ttft_med * 1e3,
        inter_token_ms=inter_med * 1e3,
        tokens_per_sec=tok_per_s,
        variance_pct=var,
        stable=var < 3.0,
        runs=runs,
        ttft_ms_runs=[t * 1e3 for t in ttft_s],
        full_ms_runs=[t * 1e3 for t in full_s],
    )


def run_suite(generate_fn: GenerateFn, base_ids: torch.Tensor, batch_sizes: list[int],
              new_tokens: int, warmup: int, runs: int, engine: str) -> list[BenchResult]:
    results = []
    for bs in batch_sizes:
        r = benchmark_one(generate_fn, base_ids, bs, new_tokens, warmup, runs, engine)
        results.append(r)
        flag = "ok" if r.stable else "UNSTABLE"
        print(f"  bs={bs:<3d} tok/s={r.tokens_per_sec:8.1f}  "
              f"TTFT={r.ttft_ms:7.2f}ms  inter={r.inter_token_ms:6.2f}ms  "
              f"var={r.variance_pct:4.1f}%  [{flag}]")
    return results


def format_table(results: list[BenchResult]) -> str:
    lines = [
        "| Engine | Batch | Tokens/sec | TTFT (ms) | Inter-token (ms) | Variance | Stable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.engine} | {r.batch_size} | {r.tokens_per_sec:.1f} | "
            f"{r.ttft_ms:.2f} | {r.inter_token_ms:.2f} | "
            f"{r.variance_pct:.1f}% | {'yes' if r.stable else 'NO'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    print(f"Loading {config.MODEL_NAME} ({config.DTYPE}) on {config.DEVICE} ...")
    model, tok = load_hf()
    base_ids = encode_prompt(tok, BENCH_PROMPT)

    def hf_fn(input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        return hf_generate(model, input_ids, max_new_tokens)

    print(f"Baseline: HuggingFace generate()  |  {args.new_tokens} new tokens, "
          f"{args.runs} runs, {args.warmup} warmup")
    results = run_suite(hf_fn, base_ids, args.batch_sizes, args.new_tokens,
                        args.warmup, args.runs, engine="HF generate()")

    print("\n" + format_table(results) + "\n")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "phase0_baseline.json"
    payload = {
        "hardware": "NVIDIA GeForce RTX 3070 (see HARDWARE.md)",
        "model": config.MODEL_NAME,
        "dtype": str(config.DTYPE),
        "new_tokens": args.new_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2))
    (config.RESULTS_DIR / "phase0_baseline.md").write_text(format_table(results) + "\n")
    print(f"Saved {out.relative_to(config.REPO_ROOT)} and phase0_baseline.md")

    all_stable = all(r.stable for r in results)
    print(f"\nAcceptance (variance < 3% all batch sizes): "
          f"{'PASS' if all_stable else 'FAIL'}")


if __name__ == "__main__":
    main()
