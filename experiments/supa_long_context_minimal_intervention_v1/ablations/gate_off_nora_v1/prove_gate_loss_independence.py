"""Empirically demonstrate that the abstention gate cannot affect any loss.

CLAIM UNDER TEST
----------------
``Qwen3AbstainForCausalLM.forward`` calls ``super().forward(...)`` -- which is
where the causal-LM loss is computed -- and only afterwards adds the constant
bias to the returned ``logits`` row for token 51378. The lab consumes
``out.loss`` in both the training step (``training.py`` line 371) and the
validation pass (``training.py`` line 112), never the mutated logits.

If the claim holds, the teacher-forced loss over the real Phase 1 dev set is
BIT-IDENTICAL with the gate on and with the gate off. That in turn means:

  * every gradient the adapters ever received was gate-free, so the saved
    adapters already ARE the gate-off-trained adapters, and
  * the recorded validation-loss trajectories already ARE the gate-off
    trajectories.

A strict gate-off replication therefore does not require retraining. This
script is the evidence for that, so the conclusion does not rest on reading
source alone.

Nothing is trained and nothing is written to any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
EXPERIMENT = ABLATION_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))
sys.path.insert(0, str(EXPERIMENT / "scripts"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from slcmi.evaluation import load_model  # noqa: E402
from slcmi.provenance import CORES, checkpoint_files  # noqa: E402
from slcmi.training import _batch, base_dtype  # noqa: E402
from slcmi.training_data import build_dataset, load_packs, sample_curriculum  # noqa: E402

ABSTAIN_TOKEN_ID = 51378


def dev_loss(model, dataset, device, amp_dtype) -> tuple[float, list[float]]:
    """Teacher-forced loss over the dev set, mirroring training.evaluate_loss.

    Reimplemented here only because the frozen ``evaluate_loss`` asserts the
    gate is ON, which is precisely the condition this test must also run
    without. The arithmetic is identical: ``out.loss`` weighted by answer tokens.
    """
    was_training = model.training
    total_loss = 0.0
    total_tokens = 0
    per_example: list[float] = []
    try:
        model.eval()
        for example in dataset:
            batch = _batch(example, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                out = model(**batch)
            n = example["n_answer_tokens"]
            loss = float(out.loss.detach())
            per_example.append(loss)
            total_loss += loss * n
            total_tokens += n
            del out
    finally:
        model.train(was_training)
    return total_loss / max(1, total_tokens), per_example


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", default="a11", choices=sorted(CORES))
    ap.add_argument("--adapter-stage", default="phase1")
    ap.add_argument("--arm", default="C", choices=("A", "B", "C"))
    ap.add_argument("--limit", type=int, default=0, help="0 = the full 160-row dev set")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    core = CORES[args.core]
    before = checkpoint_files(core["path"])

    tokenizer = AutoTokenizer.from_pretrained(core["path"], trust_remote_code=True,
                                              local_files_only=True)
    packs = load_packs()
    dev_raw = sample_curriculum(packs, "dev", 40)
    dev_set, stats = build_dataset(tokenizer, dev_raw, 3840)
    if args.limit:
        dev_set = dev_set[: args.limit]

    _, model = load_model(core["path"], device="cuda", dtype=core["dtype"], gate=True)
    if args.arm != "A":
        adapter = EXPERIMENT / "results" / args.adapter_stage / f"arm_{args.arm}" / "adapter"
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    cfg = model.get_base_model().config if hasattr(model, "get_base_model") else model.config
    amp_dtype = base_dtype(model)

    canonical = float(cfg.abstain_logit_bias)
    assert canonical, "expected the canonical gate before the comparison"

    cfg.abstain_logit_bias = canonical
    loss_on, per_on = dev_loss(model, dev_set, "cuda", amp_dtype)
    cfg.abstain_logit_bias = 0.0
    loss_off, per_off = dev_loss(model, dev_set, "cuda", amp_dtype)
    cfg.abstain_logit_bias = canonical

    identical = [a == b for a, b in zip(per_on, per_off)]
    result = {
        "test": "abstention gate cannot affect teacher-forced loss",
        "core": args.core,
        "arm": args.arm,
        "adapter_stage": args.adapter_stage if args.arm != "A" else None,
        "abstain_token_id": ABSTAIN_TOKEN_ID,
        "canonical_bias": canonical,
        "n_dev_examples": len(dev_set),
        "dev_dataset_stats": stats,
        "weighted_loss_gate_on": loss_on,
        "weighted_loss_gate_off": loss_off,
        "weighted_loss_absolute_difference": abs(loss_on - loss_off),
        "weighted_loss_bit_identical": loss_on == loss_off,
        "per_example_bit_identical_count": sum(identical),
        "per_example_n": len(identical),
        "all_per_example_bit_identical": all(identical),
        "max_per_example_absolute_difference": max(
            (abs(a - b) for a, b in zip(per_on, per_off)), default=0.0
        ),
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": checkpoint_files(core["path"]),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
    }
    result["canonical_checkpoint_unchanged"] = (
        result["checkpoint_sha256_before"] == result["checkpoint_sha256_after"]
    )
    result["conclusion"] = (
        "gate is absent from the objective; saved adapters are gate-off-trained"
        if result["all_per_example_bit_identical"]
        else "GATE AFFECTS LOSS -- a strict gate-off replication REQUIRES retraining"
    )

    out = Path(args.output) if args.output else ABLATION_DIR / "results" / "gate_loss_independence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in (
        "weighted_loss_gate_on", "weighted_loss_gate_off", "weighted_loss_bit_identical",
        "per_example_bit_identical_count", "per_example_n",
        "max_per_example_absolute_difference", "canonical_checkpoint_unchanged", "conclusion",
    )}, indent=1))


if __name__ == "__main__":
    main()
