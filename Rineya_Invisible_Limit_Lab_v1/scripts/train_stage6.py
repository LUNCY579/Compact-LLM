#!/usr/bin/env python3
"""Stage 6: contrastive LoRA training against the paraphrase/counterfactual deficit.

Stage 8 left the entire remaining gap in two columns -- paraphrase -22 and
counterfactual -21 -- and no amount of abstention engineering touches either.
This trains for them directly.

The contrastive signal is structural, not a custom loss. For each scenario the
TRAIN split carries a `base` and a `counterfactual` variant whose contexts are
near-identical but whose answers differ, because the decisive binding was
swapped. Fitting both in the same run therefore forces the model to attend to the
one sentence that distinguishes them; a model that pattern-matches on surface
familiarity cannot satisfy both examples at once. The `paraphrase` variant adds
question-form invariance over the same context.

Deliberate exclusions
---------------------
`unanswerable` rows are NOT trained on. Abstention is the probe's job, and
teaching UNKNOWN targets would move the very hidden states the probe reads. See
PREREGISTRATION.md Amendment 4.

Loss is masked to the answer tokens only. Training on the ~1000-token context
would drown the signal in language modelling of filler text.

Because training moves the hidden states, the previously frozen probe does not
transfer and must be refit on DEV afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIN_VARIANTS = {"base", "paraphrase", "counterfactual"}


def build_examples(full_path: Path, tok, max_len: int) -> list[dict]:
    rows = [json.loads(l) for l in full_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    skipped = 0
    for r in rows:
        if r["variant"] not in TRAIN_VARIANTS:
            continue
        prompt_text = tok.apply_chat_template(
            [{"role": "system", "content": r["messages"][0]["content"]},
             {"role": "user", "content": r["messages"][1]["content"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        p_ids = tok(prompt_text, add_special_tokens=False).input_ids
        a_ids = tok(r["answer"], add_special_tokens=False).input_ids
        if tok.eos_token_id is not None:
            a_ids = a_ids + [tok.eos_token_id]
        if len(p_ids) + len(a_ids) > max_len:
            skipped += 1
            continue
        ids = p_ids + a_ids
        # -100 on the prompt so loss lands only on the answer.
        labels = [-100] * len(p_ids) + a_ids
        out.append({"input_ids": ids, "labels": labels,
                    "scenario_id": r["scenario_id"], "variant": r["variant"]})
    if skipped:
        print(f"  skipped {skipped} rows over max_len={max_len}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--train-full", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)

    print("building examples ...")
    ex = build_examples(args.train_full, tok, args.max_len)
    if args.limit:
        ex = ex[: args.limit]
    import collections
    print(f"  {len(ex)} examples  {dict(collections.Counter(e['variant'] for e in ex))}")
    print(f"  scenarios: {len({e['scenario_id'] for e in ex})}")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"loading base on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    steps_per_epoch = math.ceil(len(ex) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps), pct_start=0.1)

    print(f"training: {total_steps} optimizer steps "
          f"({steps_per_epoch}/epoch x {args.epochs} epochs), grad_accum={args.grad_accum}")

    rng = torch.Generator().manual_seed(args.seed)
    started = time.time()
    step = 0
    micro = 0
    running = 0.0
    n_micro_total = int(len(ex) * args.epochs)
    done = False
    while not done:
        order = torch.randperm(len(ex), generator=rng).tolist()
        for idx in order:
            e = ex[idx]
            ids = torch.tensor([e["input_ids"]], device=device)
            lab = torch.tensor([e["labels"]], device=device)
            out = model(input_ids=ids, labels=lab)
            (out.loss / args.grad_accum).backward()
            running += out.loss.detach().item()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    el = time.time() - started
                    rate = micro / el
                    eta = (n_micro_total - micro) / max(rate, 1e-9) / 60
                    print(f"  step {step}/{total_steps}  loss={running/args.grad_accum/10:.4f}  "
                          f"{rate:.2f} ex/s  ETA {eta:.0f} min", flush=True)
                    running = 0.0
                if step >= total_steps:
                    done = True
                    break
            if micro >= n_micro_total:
                done = True
                break

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tok.save_pretrained(str(args.output))
    manifest = {
        "kind": "stage6_contrastive_lora",
        "base": args.base,
        "train_full": str(args.train_full),
        "train_variants": sorted(TRAIN_VARIANTS),
        "examples": len(ex),
        "scenarios": len({e["scenario_id"] for e in ex}),
        "epochs": args.epochs, "lr": args.lr,
        "rank": args.rank, "alpha": args.alpha,
        "grad_accum": args.grad_accum, "max_len": args.max_len,
        "seed": args.seed, "device": device,
        "optimizer_steps": step,
        "wall_seconds": round(time.time() - started, 1),
        "loss_masked_to": "answer tokens only",
        "excluded": "unanswerable variant (abstention is the probe's job)",
    }
    (args.output / "TRAIN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved adapter -> {args.output}  ({time.time()-started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
