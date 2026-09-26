"""Single-variable abstention-gate ablation for the Phase 1 adapters.

WHAT THIS CHANGES, AND NOTHING ELSE
-----------------------------------
The only manipulated variable is the constant abstention logit bias that the
A-series cores apply to token 51378 (``UNKNOWN``) inside
``Qwen3AbstainForCausalLM.forward``. ``--gate off`` sets
``config.abstain_logit_bias = 0.0`` in memory. Nothing is written to any
checkpoint; the frozen backbone, the adapter weights, the curriculum, the
splits, the seeds, the decoding protocol, the suite and the scorer are all
reused unchanged, by importing the lab's own frozen modules rather than
reimplementing them.

WHY NO RETRAINING IS REQUIRED FOR THE TRAINING SIDE
---------------------------------------------------
``Qwen3AbstainForCausalLM.forward`` calls ``super().forward(...)`` first --
which is where the causal-LM loss is computed -- and only then mutates the
returned ``logits`` row for token 51378. Both the training step
(``training.py`` line 371) and the validation pass (``training.py`` line 112)
consume ``out.loss``, never the mutated logits. The bias is therefore
mathematically absent from every gradient and every recorded loss. Independently
of that, ``training.py`` already wraps the whole optimisation in
``gate_disabled(model)``, which zeroes the bias for the duration of training.

So the saved adapters ARE the gate-off-trained adapters, and the recorded
validation-loss trajectories ARE the gate-off trajectories. ``prove_gate_loss_
independence.py`` in this directory demonstrates that empirically rather than
resting on the source reading.

SELF-VALIDATION
---------------
Run with ``--gate on`` first. That must reproduce the arm's stored
``per_example.jsonl`` exactly, row for row. Only a harness that reproduces the
historical gate-ON result is trusted to report a gate-OFF one.

This file lives outside ``src/``, ``scripts/`` and ``configs/`` on purpose: those
are SHA-256 frozen into every existing run receipt and must not gain new files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
EXPERIMENT = ABLATION_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT / "src"))
sys.path.insert(0, str(EXPERIMENT / "scripts"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402

from slcmi.evaluation import evaluate_rows, load_model, render_prompt  # noqa: E402
from slcmi.preservation import build_preservation_set, evaluate_preservation  # noqa: E402
from slcmi.provenance import (  # noqa: E402
    CORES,
    checkpoint_files,
    frozen_parameter_hash,
)

ABSTAIN_TOKEN_ID = 51378
CANONICAL_BIAS = 5.156249985098839


def target_items() -> list[dict]:
    """Import the frozen target-set builder from the frozen runner."""
    import phase1_run

    return phase1_run.target_items(False)


def first_token_diagnostics(tokenizer, model, items, device="cuda", bias=CANONICAL_BIAS):
    """Raw first-generated-token logits for token 51378 versus the best alternative.

    Run with the gate OFF so the recorded logits are the model's own. The gate's
    documented rule is exactly ``z_UNKNOWN + b >= z_argmax_other`` under greedy
    decoding, so its effect is recovered analytically from these raw numbers and
    never needs a second generation pass.
    """
    records = []
    for item in items:
        prompt = render_prompt(tokenizer, item["context"], item["question"])
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0, -1, :].float()
        z_unknown = float(logits[ABSTAIN_TOKEN_ID])
        masked = logits.clone()
        masked[ABSTAIN_TOKEN_ID] = float("-inf")
        best_other = int(torch.argmax(masked))
        z_best_other = float(masked[best_other])
        records.append(
            {
                "example_id": item["example_id"],
                "answerable": item["answerable"],
                "split": item["split"],
                "variant": item["variant"],
                "task_type": item["task_type"],
                "z_unknown": round(z_unknown, 6),
                "z_best_other": round(z_best_other, 6),
                "best_other_token_id": best_other,
                "best_other_token": tokenizer.decode([best_other]),
                # margin < 0 means the model prefers a real answer on its own
                "margin_gate_off": round(z_unknown - z_best_other, 6),
                "margin_gate_on": round(z_unknown + bias - z_best_other, 6),
                "would_abstain_gate_off": bool(z_unknown >= z_best_other),
                "would_abstain_gate_on": bool(z_unknown + bias >= z_best_other),
            }
        )
        del out, logits, masked
    return records


def summarise(records, preservation):
    tgt = records
    ans = [r for r in tgt if r["answerable"]]
    una = [r for r in tgt if not r["answerable"]]

    def unknown(rows):
        return sum(1 for r in rows if (r["raw_prediction"] or "").strip().upper().startswith("UNKNOWN"))

    return {
        "target_total": {"correct": sum(r["correct"] for r in tgt), "n": len(tgt)},
        "target_answerable": {"correct": sum(r["correct"] for r in ans), "n": len(ans)},
        "target_unanswerable": {"correct": sum(r["correct"] for r in una), "n": len(una)},
        "answerable_unknown_predictions": unknown(ans),
        "unanswerable_unknown_predictions": unknown(una),
        "target_malformed": sum(r["malformed"] for r in tgt),
        "preservation_matched": {
            "matched": sum(r["matched"] for r in preservation),
            "n": len(preservation),
        },
        "preservation_by_class": {
            fam: {
                "matched": sum(r["matched"] for r in preservation if r["task_family"] == fam),
                "n": sum(1 for r in preservation if r["task_family"] == fam),
            }
            for fam in sorted({r["task_family"] for r in preservation})
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", required=True, choices=sorted(CORES))
    ap.add_argument("--adapter-stage", default=None,
                    help="stage holding the saved adapter, e.g. phase1 or phase1_a1")
    ap.add_argument("--arm", required=True, choices=("A", "B", "C"),
                    help="A evaluates the frozen core with no adapter")
    ap.add_argument("--gate", required=True, choices=("on", "off"))
    ap.add_argument("--label", required=True, help="output subdirectory name")
    ap.add_argument("--skip-diagnostics", action="store_true")
    args = ap.parse_args()

    if args.arm != "A" and not args.adapter_stage:
        ap.error("--adapter-stage is required for arms B and C")

    core = CORES[args.core]
    out_dir = ABLATION_DIR / "results" / args.label
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing evidence: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    before = checkpoint_files(core["path"])

    # gate=False zeroes the in-memory bias only; the offset lives in the config
    # and is applied in forward(), never in the weights.
    tokenizer, model = load_model(core["path"], device="cuda", dtype=core["dtype"],
                                  gate=(args.gate == "on"))
    adapter_dir = None
    if args.arm != "A":
        adapter_dir = EXPERIMENT / "results" / args.adapter_stage / f"arm_{args.arm}" / "adapter"
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        model.eval()

    # Re-assert AFTER any PEFT wrapping: the wrapper delegates .config to the base.
    base_config = model.get_base_model().config if hasattr(model, "get_base_model") else model.config
    if args.gate == "off":
        base_config.abstain_logit_bias = 0.0
        assert base_config.abstain_logit_bias == 0.0, "gate not disabled"
    else:
        assert base_config.abstain_logit_bias, "gate expected on"
    assert base_config.abstain_token_id == ABSTAIN_TOKEN_ID

    frozen_before = frozen_parameter_hash(model)
    targets = target_items()
    preservation = build_preservation_set(per_class=100)

    t0 = time.perf_counter()
    print(f"  target eval ({len(targets)} rows, gate {args.gate})...")
    target_records = evaluate_rows(tokenizer, model, targets, device="cuda")
    print(f"  preservation eval ({len(preservation)} rows)...")
    preservation_records = evaluate_preservation(tokenizer, model, preservation, device="cuda")

    diagnostics = []
    if not args.skip_diagnostics:
        print("  first-token logit diagnostics...")
        # Diagnostics always read RAW logits, independent of --gate, so the two
        # runs of an arm produce identical diagnostic numbers.
        restore = base_config.abstain_logit_bias
        try:
            base_config.abstain_logit_bias = 0.0
            diagnostics = first_token_diagnostics(tokenizer, model, targets, device="cuda")
        finally:
            base_config.abstain_logit_bias = restore

    frozen_after = frozen_parameter_hash(model)
    after = checkpoint_files(core["path"])

    summary = summarise(target_records, preservation_records)
    manifest = {
        "ablation": "gate_off_nora_v1",
        "single_manipulated_variable": "abstain_logit_bias on token 51378",
        "gate": args.gate,
        "abstain_logit_bias_used": 0.0 if args.gate == "off" else CANONICAL_BIAS,
        "core": args.core,
        "core_path": str(core["path"]),
        "dtype": core["dtype"],
        "arm": args.arm,
        "adapter_stage": args.adapter_stage,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "n_targets": len(targets),
        "n_preservation": len(preservation),
        "started_utc": started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_base_unchanged": frozen_before == frozen_after,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "canonical_checkpoint_unchanged": before == after,
        "summary": summary,
    }
    if not manifest["canonical_checkpoint_unchanged"]:
        raise RuntimeError("canonical checkpoint changed during the ablation")
    if not manifest["frozen_base_unchanged"]:
        raise RuntimeError("frozen base parameters changed during the ablation")

    (out_dir / "per_example.jsonl").write_text(
        "\n".join(
            json.dumps({"evaluation": "target", "arm": args.arm, **r}, ensure_ascii=False)
            for r in target_records
        )
        + "\n"
        + "\n".join(
            json.dumps({"evaluation": "preservation", "arm": args.arm, **r}, ensure_ascii=False)
            for r in preservation_records
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if diagnostics:
        (out_dir / "first_token_diagnostics.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in diagnostics) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    s = summary
    print(f"  target {s['target_total']['correct']}/{s['target_total']['n']}  "
          f"answerable {s['target_answerable']['correct']}/{s['target_answerable']['n']}  "
          f"unanswerable {s['target_unanswerable']['correct']}/{s['target_unanswerable']['n']}  "
          f"answerable-UNKNOWN {s['answerable_unknown_predictions']}  "
          f"preservation {s['preservation_matched']['matched']}/{s['preservation_matched']['n']}")


if __name__ == "__main__":
    main()
