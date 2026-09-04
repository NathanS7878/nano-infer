"""Phase 4: perplexity on a held-out WikiText-2 slice, fp16 vs INT8 vs INT4.

Spec rule 5: report where it gets worse. Quantization degrades quality — that is
not a risk to be mitigated, it is a fact to be measured and stated. This is the
instrument that measures it, and Phase 0's discipline applies: build the
measuring tool first and validate it on a known-good configuration (fp16) before
using it to judge anything.

WHAT PERPLEXITY IS, IN ONE PARAGRAPH
------------------------------------
The model assigns a probability to each next token. Perplexity is
exp(mean negative log-likelihood) over a held-out corpus: roughly "how many
equally-likely options was the model effectively choosing between at each step".
Lower is better. It is the standard number for this because it is sensitive to
the whole output distribution rather than just the argmax — quantization can
leave greedy decoding unchanged while measurably flattening the distribution,
and a token-agreement metric alone would miss that.

METHOD
------
  - Non-overlapping windows of WINDOW tokens from the WikiText-2 *test* split,
    which the model was not trained on.
  - For each window: run our own Phase 1 forward (all positions), take
    log-softmax, and gather the log-probability the model assigned to the token
    that actually came next.
  - Sum NLL over all predicted tokens, divide by the count, exponentiate.
    Aggregating over tokens (not averaging per-window perplexities) is the
    correct way; averaging exponentials would bias the result.

Every configuration sees the SAME windows in the same order, so the comparison
is paired and the corpus sample cannot favour one precision over another.

WHY THIS USES THE PHASE 1 FORWARD
---------------------------------
`model.forward` returns logits at every position, which is exactly what a
perplexity sweep needs, and it is the project's reference implementation — the
answer key that Phase 1 proved token-for-token against HuggingFace. Using the
cached/paged paths here would measure the cache as well as the quantization.

Run:  python -m bench.perplexity
      python -m bench.perplexity --windows 8 --window 512
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from nano_infer import config as cfg
from nano_infer import model as M
from nano_infer import quant as Q

# The bare "wikitext" id no longer resolves on huggingface_hub >= 1.0, which
# requires a namespaced repo. This is the canonical mirror of the same data.
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")
SPLIT = "test"

WINDOW = 512          # tokens per window; the model handles far more, but the
                      # Phase 1 forward materializes an O(n^2) attention matrix
WINDOWS = 16          # 16 x 512 = 8192 predicted tokens, enough to be stable


def load_corpus_tokens(n_tokens: int) -> torch.Tensor:
    """Tokenize enough of the WikiText-2 test split to fill the sweep."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(DATASET[0], DATASET[1], split=SPLIT)
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    ids = tok(text, return_tensors="pt").input_ids[0]
    if ids.numel() < n_tokens:
        raise RuntimeError(f"corpus has {ids.numel()} tokens, need {n_tokens}")
    return ids[:n_tokens]


@torch.no_grad()
def perplexity(weights: dict, cf, tokens: torch.Tensor, window: int,
               windows: int) -> dict:
    """Aggregate NLL over non-overlapping windows, then exponentiate once."""
    chunks = []
    t0 = time.perf_counter()

    for w in range(windows):
        chunk = tokens[w * window:(w + 1) * window].unsqueeze(0).to(cfg.DEVICE)
        logits = M.forward(chunk, weights, cf)              # [1, window, vocab]

        # predict position i+1 from position i, so drop the last logit and the
        # first target
        logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = chunk[0, 1:]
        nll = -logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        chunks.append(nll.detach())

    all_nll = torch.cat(chunks)
    mean_nll = float(all_nll.mean())

    # Standard error of the mean NLL. A perplexity DELTA is not interpretable
    # without it: this corpus is a finite sample, and a difference smaller than
    # the sampling error of the estimate is not evidence of anything. It is what
    # lets INT8's -0.55% be called noise rather than an improvement.
    sem_nll = float(all_nll.std(unbiased=True) / (all_nll.numel() ** 0.5))
    ppl = float(torch.tensor(mean_nll).exp())

    return {
        "perplexity": ppl,
        "mean_nll": mean_nll,
        "sem_nll": sem_nll,
        # one standard error expressed as a percentage of perplexity
        "sem_pct": (float(torch.tensor(mean_nll + sem_nll).exp()) / ppl - 1) * 100,
        "tokens": all_nll.numel(),
        "seconds": time.perf_counter() - t0,
    }


@torch.no_grad()
def sample_generations(weights: dict, cf, n_tokens: int = 24) -> list:
    """A handful of greedy continuations, so the quality cost is legible as text
    and not only as a number. Spec: 'plus a handful of qualitative generations
    side by side'."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    out = []
    for prompt in cfg.PROMPTS[:3]:
        ids = tok(prompt, return_tensors="pt").input_ids.to(cfg.DEVICE)
        gen = M.generate_paged(ids, weights, cf, n_tokens)
        out.append({
            "prompt": prompt,
            "continuation": tok.decode(gen[0], skip_special_tokens=True),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--windows", type=int, default=WINDOWS)
    ap.add_argument("--modes", default="fp16,int8,int4")
    ap.add_argument("--int4-groups", default="128",
                    help="comma-separated INT4 group sizes to sweep, e.g. 32,64,128")
    ap.add_argument("--no-generations", action="store_true")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    int4_groups = [int(g) for g in args.int4_groups.split(",") if g.strip()]

    # (mode, group) pairs actually run; only int4 has a group size
    runs = []
    for m in modes:
        if m == "int4":
            runs.extend(("int4", g) for g in int4_groups)
        else:
            runs.append((m, None))

    print(f"\nPerplexity — {cfg.MODEL_NAME}, WikiText-2 {SPLIT} split")
    print(f"{args.windows} windows x {args.window} tokens "
          f"= {args.windows * (args.window - 1)} predicted tokens\n")

    cf = M.QwenConfig()
    base = M.load_weights()
    tokens = load_corpus_tokens(args.window * args.windows)

    print(f"{'mode':>9}{'perplexity':>13}{'delta':>10}{'model MB':>11}"
          f"{'compression':>13}{'bits/wt':>9}{'mean rel err':>14}")
    print("-" * 81)

    rows = []
    fp16_ppl = None
    for mode, group in runs:
        kwargs = {"group": group} if group else {}
        weights, stats = Q.quantize_weights(base, mode, **kwargs)
        res = perplexity(weights, cf, tokens, args.window, args.windows)
        if fp16_ppl is None and mode == "fp16":
            fp16_ppl = res["perplexity"]

        delta = (res["perplexity"] / fp16_ppl - 1) * 100 if fp16_ppl else 0.0
        label = f"{mode} g{group}" if group else mode
        rows.append({**res, **stats, "delta_pct": delta, "group": group})
        sig = "" if abs(delta) <= res["sem_pct"] else " *"
        print(f"{label:>9}{res['perplexity']:>13.4f}"
              f"{delta:>8.2f}%{sig:<2}{stats['bytes_quantized']/1e6:>10.1f}"
              f"{stats['compression']:>12.2f}x{stats['bits_per_weight']:>9.2f}"
              f"{stats['mean_rel_err']:>14.2e}")

        # free the round-tripped copy before building the next one
        del weights
        torch.cuda.empty_cache()

    print("-" * 81)
    if fp16_ppl:
        sem = rows[0].get("sem_pct", 0.0)
        print(f"fp16 baseline perplexity {fp16_ppl:.4f} +/- {sem:.2f}% (1 s.e.); "
              f"deltas are relative to it.")
        print("'*' marks a delta larger than one standard error of the "
              "estimate; an unmarked delta is not distinguishable from "
              "sampling noise.")
    print("Compression is WHOLE MODEL. The quantized tensors alone shrink more "
          "(see\nthe 'quantized tensors' figure in tests/test_quant.py) — the "
          "embedding stays fp16.")

    generations = {}
    if not args.no_generations:
        print("\nSample greedy continuations (same prompts, each precision):")
        gen_modes = []
        for m in modes:
            gen_modes.append((m, int4_groups[0] if m == "int4" else None))
        for m, g in gen_modes:
            kwargs = {"group": g} if g else {}
            weights, _ = Q.quantize_weights(base, m, **kwargs)
            generations[m] = sample_generations(weights, cf)
            del weights
            torch.cuda.empty_cache()
        for i in range(len(generations[modes[0]])):
            print(f"\n  prompt: {generations[modes[0]][i]['prompt']!r}")
            for mode in modes:
                text = generations[mode][i]["continuation"].replace("\n", " ")
                print(f"    {mode:>5}: {text[:110]}")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RESULTS_DIR / "perplexity.json").write_text(json.dumps({
        "model": cfg.MODEL_NAME,
        "dataset": f"{DATASET[0]}/{DATASET[1]}:{SPLIT}",
        "window": args.window, "windows": args.windows,
        "results": rows, "generations": generations,
    }, indent=2), encoding="utf-8")

    table = ["| Precision | Perplexity | vs fp16 | Model size | Compression | bits/weight |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        label = r["mode"]
        if r.get("group") and r["mode"] == "int4":
            label = f"int4 g{r['group']}"
        table.append(f"| {label} | {r['perplexity']:.4f} | "
                     f"{r['delta_pct']:+.2f}% | {r['bytes_quantized']/1e6:.0f} MB | "
                     f"{r['compression']:.2f}x | {r['bits_per_weight']:.2f} |")
    # Explicit encoding: Windows defaults to cp1252 and this file is UTF-8.
    (cfg.RESULTS_DIR / "perplexity.md").write_text(
        "\n".join(table) + "\n", encoding="utf-8")
    print("\nSaved perplexity.json and perplexity.md")


if __name__ == "__main__":
    main()
