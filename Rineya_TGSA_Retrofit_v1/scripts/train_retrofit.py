#!/usr/bin/env python3
"""Development training funnel for the converted Rineya retrofit.

Stages, per the registered funnel. Each is a gate: a failure here must prevent
the next stage from spending compute.

``r0``  tiny overfit. Can the trajectory path learn anything at all? Verifies the
        loss decreases, trajectory gradients are live, nothing goes non-finite,
        save/reload is identity, resume is equivalent, and generation still runs.
``r1``  trajectory-only warm start. Pretrained weights frozen.
``r2``  selective attention unfreeze, compared against r1.

Development only. This never reads the Phase-1 confirmatory bank, and its
outputs are not evidence for the Phase-1 mechanism question.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import torch

from rineya_tgsa import RineyaTGSAConfig
from rineya_tgsa.qwen3_adapter import Qwen3TGSAAttention, convert_qwen3_model
from rineya_tgsa.schedule import FreezeSchedule, apply_stage

TRAJECTORY_KEYS = ("carrier_init", "trajectory_encoders", "correction_", "carrier_norm", "conditioning")


def is_trajectory(name: str) -> bool:
    return any(key in name for key in TRAJECTORY_KEYS)


def set_stage(model, stage: str) -> dict:
    """Freeze/unfreeze by stage across the whole model."""

    for name, parameter in model.named_parameters():
        if stage == "all":
            trainable = True
        elif stage == "trajectory":
            trainable = is_trajectory(name)
        elif stage == "attention":
            trainable = is_trajectory(name) or (
                "self_attn" in name and any(p in name for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
            )
        else:
            trainable = "self_attn" in name
        parameter.requires_grad_(trainable)
    return {
        "stage": stage,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("r0", "r1", "r2"), default="r0")
    parser.add_argument("--layers", default="0,14,27")
    parser.add_argument("--trajectory-rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-impl", default="eager")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not (args.model_dir / "model.safetensors").is_file():
        print(f"ERROR: no model.safetensors under {args.model_dir}", file=sys.stderr)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    source_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    tgsa_config = RineyaTGSAConfig.from_source_config(
        source_config, trajectory_rank=args.trajectory_rank
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
        attn_implementation=args.attn_impl,
    )
    record = convert_qwen3_model(
        model, tgsa_config, layers=[int(v) for v in args.layers.split(",")]
    )

    stage = {"r0": "trajectory", "r1": "trajectory", "r2": "attention"}[args.stage]
    freeze = set_stage(model, stage)
    print(f"stage {args.stage} ({stage}): {freeze['trainable_parameters']:,} trainable / "
          f"{freeze['frozen_parameters']:,} frozen", flush=True)

    # R0 is a deliberate tiny overfit: one short batch, repeated. If the
    # trajectory path cannot drive the loss down on data it has memorised, it
    # will certainly not help on real data, and the funnel stops here.
    text = "The quick brown fox jumps over the lazy dog. " * 4
    batch = tokenizer(text, return_tensors="pt")
    ids = batch["input_ids"][:, :64]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "trajectory_state.pt"
    start_step = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    if args.resume and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        print(f"resumed from step {start_step}", flush=True)

    losses = []
    grad_norms = []
    started = time.time()
    model.train()
    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"ERROR: non-finite loss at step {step}", file=sys.stderr)
            return 1
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss))
        grad_norms.append(float(norm))
        if step % max(1, args.steps // 10) == 0:
            print(f"  step {step:4d}  loss {float(loss):.4f}  |g| {float(norm):.4f}", flush=True)

    trajectory_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if is_trajectory(name)
    }
    torch.save(
        {"model": trajectory_state, "optimizer": optimizer.state_dict(), "step": args.steps},
        checkpoint,
    )

    # Save/reload identity on the trajectory tensors.
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    identical = all(torch.equal(trajectory_state[k], reloaded[k]) for k in trajectory_state)

    module = model.model.layers[record["converted_layers"][0]].self_attn
    report = {
        "schema_version": "rineya.tgsa-train-funnel.v1",
        "stage": args.stage,
        "freeze_stage": stage,
        "model_dir": str(args.model_dir),
        "converted_layers": record["converted_layers"],
        "steps": args.steps,
        "seed": args.seed,
        "learning_rate": args.lr,
        "trainable_parameters": freeze["trainable_parameters"],
        "frozen_parameters": freeze["frozen_parameters"],
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "loss_curve": losses,
        "gradient_norms": grad_norms,
        "all_finite": bool(all(v == v for v in losses + grad_norms)),
        "trajectory_gradients_live": bool(any(n > 0 for n in grad_norms)),
        "conditioning_left_neutral": module.conditioning_is_neutral(),
        "save_reload_identity": identical,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / f"{args.stage}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"loss {report['loss_first']:.4f} -> {report['loss_last']:.4f} "
          f"(decreased={report['loss_decreased']})")
    print(f"trajectory gradients live : {report['trajectory_gradients_live']}")
    print(f"conditioning still neutral: {report['conditioning_left_neutral']} "
          f"(False is expected after training)")
    print(f"save/reload identity      : {report['save_reload_identity']}")

    failures = []
    if not report["loss_decreased"]:
        failures.append("loss did not decrease on a tiny overfit")
    if not report["trajectory_gradients_live"]:
        failures.append("trajectory gradients are dead")
    if not report["all_finite"]:
        failures.append("non-finite loss or gradient")
    if not report["save_reload_identity"]:
        failures.append("save/reload is not identity")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
