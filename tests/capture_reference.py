"""Capture the HuggingFace reference fixture — the frozen answer key that every
later version of the engine is graded against.

For each canonical prompt we run HF's forward pass in a manual greedy loop (not
generate(), because we want the RAW logits, not generate()'s post-processed
scores) and record:

  - input_ids      : the tokenized prompt
  - greedy_ids     : the 50 greedily-decoded continuation tokens  (token-for-token key)
  - step0_logits   : full fp16 logit vector at the first decode step (strict numeric key)
  - topk_ids/vals  : top-5 (id, logit) at every step (cheap per-step ranking key)

Run:  python -m tests.capture_reference
"""
from __future__ import annotations

import torch
import transformers

from nano_infer import config
from nano_infer.hf_ref import encode_prompt, load_hf

TOPK = 5


@torch.no_grad()
def greedy_capture(model, input_ids, max_new_tokens: int):
    """Manual greedy decode capturing logits. Uses HF's KV cache (use_cache=True);
    the cache never changes the logits — it only avoids recomputing the frozen
    past — so the answer key is identical with or without it, just faster with."""
    device = input_ids.device
    greedy_ids: list[int] = []
    step0_logits = None
    topk_ids = torch.empty(max_new_tokens, TOPK, dtype=torch.long)
    topk_vals = torch.empty(max_new_tokens, TOPK, dtype=torch.float16)

    cur = input_ids
    past = None
    for step in range(max_new_tokens):
        out = model(cur, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :].float()  # [1, vocab], upcast for stable topk
        past = out.past_key_values

        if step == 0:
            step0_logits = logits.squeeze(0).to(torch.float16).cpu().clone()

        vals, ids = torch.topk(logits.squeeze(0), TOPK)
        topk_ids[step] = ids.cpu()
        topk_vals[step] = vals.to(torch.float16).cpu()

        next_id = int(ids[0].item())  # greedy = argmax = top-1
        greedy_ids.append(next_id)

        # feed only the new token next step; the cache holds the rest
        cur = torch.tensor([[next_id]], device=device)

    return {
        "input_ids": input_ids.squeeze(0).cpu(),
        "greedy_ids": torch.tensor(greedy_ids, dtype=torch.long),
        "step0_logits": step0_logits,
        "topk_ids": topk_ids,
        "topk_vals": topk_vals,
    }


def main():
    torch.manual_seed(config.SEED)
    config.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.MODEL_NAME} in {config.DTYPE} on {config.DEVICE} ...")
    model, tok = load_hf()

    per_prompt = []
    for i, prompt in enumerate(config.PROMPTS):
        ids = encode_prompt(tok, prompt)
        cap = greedy_capture(model, ids, config.PARITY_NEW_TOKENS)
        cap["prompt"] = prompt
        text = tok.decode(cap["greedy_ids"], skip_special_tokens=True)
        per_prompt.append(cap)
        print(f"[{i}] {prompt!r}\n    -> {text!r}\n")

    fixture = {
        "meta": {
            "model_name": config.MODEL_NAME,
            "dtype": str(config.DTYPE),
            "device": config.DEVICE,
            "new_tokens": config.PARITY_NEW_TOKENS,
            "topk": TOPK,
            "seed": config.SEED,
            "vocab_size": int(model.config.vocab_size),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "prompts": per_prompt,
    }
    torch.save(fixture, config.REFERENCE_FIXTURE)
    size_mb = config.REFERENCE_FIXTURE.stat().st_size / 1024**2
    print(f"Saved {config.REFERENCE_FIXTURE.relative_to(config.REPO_ROOT)} "
          f"({size_mb:.2f} MB, {len(per_prompt)} prompts)")


if __name__ == "__main__":
    main()
