#!/usr/bin/env python3
"""One inference pass that captures everything Stages 1-3 need.

Re-running the model once per gate variant would be wasteful and slow. Instead
this makes a single pass per split and records, for every row:

  * the greedy prediction under the frozen protocol (so it is directly scorable)
  * the full first-token distribution summary -- Stage 1 (S1 normalised rule) and
    Stage 2 (multi-signal gate) are then pure offline arithmetic
  * the final-position hidden state at several depths -- Stage 3 fits its probe
    on these without touching the model again

Every gate variant afterwards is an offline computation over this file, which is
what makes sweeping thresholds and refitting probes cheap.

Protocol is frozen to match the suite: BF16, greedy, seed 17, thinking disabled,
24 new tokens, 4096 context, truncation forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are answering a closed-context reading test. Use only the supplied "
    "context. Reply with the shortest exact answer and no explanation. If the "
    "context does not determine the answer, reply UNKNOWN."
)

# Surface forms that count as an abstention opener. The ids are derived from the
# tokenizer rather than hardcoded, then cross-checked against the id the
# published Supa-A1 gate uses so the two labs cannot silently diverge.
ABSTAIN_SURFACES = ["UNKNOWN", " UNKNOWN", "unknown", " unknown", "Unknown", " Unknown"]
SUPA_A1_UNKNOWN_TOKEN_ID = 51378


def abstention_token_ids(tok) -> list[int]:
    ids: list[int] = []
    for s in ABSTAIN_SURFACES:
        enc = tok.encode(s, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def first_token_stats(logits: torch.Tensor, abstain_ids: list[int], top_k: int) -> dict[str, Any]:
    """logits: [vocab] for the first generated position."""
    logits = logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logprobs, k=top_k)
    top1 = float(top.values[0])
    top2 = float(top.values[1])
    probs = logprobs.exp()
    entropy = float(-(probs * logprobs).sum())

    ab = torch.tensor(abstain_ids, dtype=torch.long)
    per_token = {int(i): float(logprobs[i]) for i in abstain_ids}
    variant_logprob = float(torch.logsumexp(logprobs[ab], dim=-1))
    exact_logprob = float(logprobs[SUPA_A1_UNKNOWN_TOKEN_ID])

    return {
        "top1_token_id": int(top.indices[0]),
        "top1_logprob": top1,
        "top2_logprob": top2,
        "logprob_margin": top1 - top2,
        "entropy": entropy,
        # The four signals the diagnostics ranked, recomputed here so Stage 1/2
        # never has to re-run the model.
        "unknown_exact_logprob": exact_logprob,
        "unknown_exact_minus_top1": exact_logprob - top1,
        "unknown_variant_logprob": variant_logprob,
        "unknown_variant_minus_top1": variant_logprob - top1,
        "unknown_per_token_logprob": per_token,
        "top_k": [
            {"token_id": int(i), "logprob": float(v)}
            for i, v in zip(top.indices.tolist(), top.values.tolist())
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--hidden-layers", default="-1,-8,-15",
                    help="comma-separated hidden_states indices to record")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    layers = [int(x) for x in args.hidden_layers.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code
    )
    model.eval()

    abstain_ids = abstention_token_ids(tok)
    if SUPA_A1_UNKNOWN_TOKEN_ID not in abstain_ids:
        raise RuntimeError(
            f"tokenizer does not map UNKNOWN to the published id "
            f"{SUPA_A1_UNKNOWN_TOKEN_ID}; got {abstain_ids}"
        )

    rows = [json.loads(l) for l in args.prompts.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    mode = "a" if (args.resume and done) else "w"
    eos_ids = {tok.eos_token_id}
    for extra in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(extra)
        if isinstance(tid, int) and tid >= 0:
            eos_ids.add(tid)

    started = time.time()
    with args.output.open(mode, encoding="utf-8", newline="\n") as out:
        for n, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            messages = row.get("messages") or [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            enc = tok(text, return_tensors="pt")
            if enc.input_ids.shape[-1] > args.max_context:
                raise RuntimeError(
                    f"{row['id']}: {enc.input_ids.shape[-1]} tokens exceeds "
                    f"max_context {args.max_context}; truncation is forbidden"
                )

            t0 = time.time()
            with torch.no_grad():
                pre = model(**enc, output_hidden_states=True, use_cache=True)

                # Stage 1/2 signals come from the first generated position, which
                # is the last prefill position's distribution.
                stats = first_token_stats(pre.logits[0, -1, :], abstain_ids, args.top_k)

                # Stage 3 features: last-position hidden state at several depths.
                hidden = {
                    str(li): [round(float(v), 6) for v in pre.hidden_states[li][0, -1, :].float()]
                    for li in layers
                }

                # Greedy continuation, reusing the prefill cache.
                past = pre.past_key_values
                nxt = int(torch.argmax(pre.logits[0, -1, :]))
                generated = [nxt]
                cur = torch.tensor([[nxt]])
                for _ in range(args.max_new_tokens - 1):
                    if nxt in eos_ids:
                        break
                    with torch.no_grad():
                        step = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = step.past_key_values
                    nxt = int(torch.argmax(step.logits[0, -1, :]))
                    generated.append(nxt)
                    cur = torch.tensor([[nxt]])

            text_out = tok.decode(
                [t for t in generated if t not in eos_ids], skip_special_tokens=True
            ).strip()

            out.write(json.dumps({
                "id": row["id"],
                "status": "ok",
                "prediction": text_out,
                "input_tokens": int(enc.input_ids.shape[-1]),
                "generated_tokens": len(generated),
                "latency_seconds": round(time.time() - t0, 4),
                "seed": args.seed,
                "decoding": "greedy",
                "first_token": stats,
                "hidden": hidden,
            }, ensure_ascii=False) + "\n")
            out.flush()

            if n % 16 == 0:
                rate = (time.time() - started) / max(1, n - len(done))
                print(f"  {n}/{len(rows)}  {rate:.1f}s/row", flush=True)

    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "schema_version": "1.0",
        "kind": "invisible_limit_signal_capture",
        "model": args.model,
        "prompts": str(args.prompts),
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "abstention_token_ids": abstain_ids,
        "hidden_layers": layers,
        "protocol": {
            "dtype": "bfloat16", "decoding": "greedy", "seed": args.seed,
            "thinking": False, "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context, "truncation": "forbidden",
        },
        "rows": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"captured {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
