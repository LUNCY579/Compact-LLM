#!/usr/bin/env python3
"""Focused single-example trajectory trace for debugging by a human or an agent.

Emits, for one example, everything needed to reason about what the mechanism did
and to reproduce the state exactly:

* full identity — study ID, source SHA, arm, seed, dataset/config/checkpoint
  hashes, resume lineage;
* per-head carrier states, velocities, control signals, importance scores, and
  attention entropy;
* the trajectory-control statistics that distinguish the arms;
* optional gradient health for the single example.

Fails closed: an unknown example ID, a hash mismatch, or a missing checkpoint is
an error, never a warning.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

import torch

from tgsa.metrics import telemetry_summary
from tgsa.phase1_data import collate_examples, load_examples, sha256_file, validate_dataset_manifest
from tgsa.phase1_model import (
    ControlCapture,
    build_model,
    gradient_group_norms,
    masked_answer_loss,
    state_dict_sha256,
)
from tgsa.phase1_runner import (
    _model_config_from_study,
    git_snapshot,
    load_study_config,
    resolve_device,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="study root; omit to trace an untrained model")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--example-id", help="exact example ID; omit to use --index")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--gradients", action="store_true", help="also report gradient health")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    study = load_study_config(args.config.resolve())
    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "dataset_manifest.json"
    dataset_manifest = validate_dataset_manifest(manifest_path)
    splits = dataset_manifest["splits"]
    examples = load_examples(dataset_dir / str(splits[args.split]["path"]))

    if args.example_id:
        matches = [example for example in examples if example.id == args.example_id]
        if not matches:
            print(f"ERROR: no example {args.example_id!r} in split {args.split!r}", file=sys.stderr)
            return 1
        example = matches[0]
    else:
        if not 0 <= args.index < len(examples):
            print(f"ERROR: --index {args.index} out of range (0..{len(examples) - 1})", file=sys.stderr)
            return 1
        example = examples[args.index]

    max_sequence_length = max(int(entry["max_tokens"]) - 1 for entry in splits.values())
    model_config = _model_config_from_study(
        study,
        arm=args.arm,  # type: ignore[arg-type]
        vocabulary=dataset_manifest["vocabulary"],
        max_sequence_length=max_sequence_length,
    )

    lineage: dict[str, object] = {
        "study_id": study["study_id"],
        "confirmatory": study["confirmatory"],
        "arm": args.arm,
        "seed": args.seed,
        "split": args.split,
        "example_id": example.id,
        "study_config_sha256": sha256_file(args.config.resolve()),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "git": git_snapshot(args.repo_root.resolve(), str(study["expected_base_commit"])),
        "checkpoint": None,
        "checkpoint_sha256": None,
        "resume_lineage": None,
        "model_state": "untrained",
    }

    device = resolve_device(study["runtime"])
    initialization_seed = None
    run_manifest = None
    if args.output_dir is not None:
        run_dir = args.output_dir.resolve() / "runs" / args.arm / f"seed_{args.seed}"
        manifest_file = run_dir / "run_manifest.json"
        if not manifest_file.is_file():
            print(f"ERROR: no run manifest at {manifest_file}", file=sys.stderr)
            return 1
        run_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        initialization_seed = int(run_manifest["initialization_seed"])
        training = run_manifest["training"]
        checkpoint = run_dir / str(training["final_checkpoint"])
        actual = sha256_file(checkpoint)
        if actual != str(training["final_checkpoint_sha256"]):
            print(
                f"ERROR: checkpoint hash mismatch at {checkpoint}\n"
                f"  recorded={training['final_checkpoint_sha256']}\n  actual  ={actual}",
                file=sys.stderr,
            )
            return 1
        lineage.update(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual,
                "run_status": run_manifest.get("status"),
                "completed_steps": training["completed_steps"],
                "resumed_from_step": training.get("resumed_from_step"),
                "train_log_repair": training.get("train_log_repair"),
                "resume_lineage": {
                    "initialization_seed": run_manifest["initialization_seed"],
                    "data_order_seed": run_manifest["data_order_seed"],
                    "runtime_seed": run_manifest["runtime_seed"],
                    "initial_model_state_sha256": run_manifest["initial_model_state_sha256"],
                    "final_model_state_sha256": run_manifest["final_model_state_sha256"],
                },
                "failed_attempts": sorted(
                    path.name for path in (run_dir / "failures").glob("attempt_*.json")
                )
                if (run_dir / "failures").is_dir()
                else [],
                "model_state": "trained",
            }
        )

    from tgsa.phase1_data import derive_seed

    model = build_model(
        model_config,
        initialization_seed=initialization_seed
        if initialization_seed is not None
        else derive_seed(args.seed, "model-initialization"),
    ).to(device)
    if run_manifest is not None:
        payload = torch.load(lineage["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        restored = state_dict_sha256(model)
        if restored != str(run_manifest["final_model_state_sha256"]):
            print(
                "ERROR: restored model state does not match the recorded final state",
                file=sys.stderr,
            )
            return 1
        lineage["restored_model_state_sha256"] = restored

    batch = collate_examples([example]).to(device)
    model.eval()
    with torch.inference_mode():
        with ControlCapture(model) as capture:
            output = model(
                batch.input_ids,
                key_padding_mask=batch.key_padding_mask,
                telemetry_level="full",
            )
    control_statistics = [
        {
            "encoder": name,
            "shape": list(tensor.shape),
            "rms": float(tensor.float().square().mean().sqrt().item()),
            "min": float(tensor.float().min().item()),
            "max": float(tensor.float().max().item()),
            "zero_fraction": float((tensor == 0).float().mean().item()),
        }
        for name, tensor in (
            (
                {id(module): key for key, module in model.named_modules()}.get(module_id, str(module_id)),
                value,
            )
            for module_id, value in capture.captured.items()
        )
    ]

    logits = output.logits
    predicted = int(logits[0, example.answer_input_index].argmax().item())
    trace: dict[str, object] = {
        "schema_version": "tgsa.phase1-example-trace.v1",
        "lineage": lineage,
        "example": {
            "id": example.id,
            "family": example.family,
            "stratum": example.stratum,
            "token_length": len(example.tokens),
            "answer_token": example.answer_token,
            "answer_input_index": example.answer_input_index,
            "metadata": dict(example.metadata),
        },
        "prediction": {
            "predicted_token": predicted,
            "correct": predicted == example.answer_token,
        },
        "trajectory_control_statistics": control_statistics,
        "blocks": [
            {
                "block_index": index,
                "summary": None if telemetry is None else telemetry_summary(telemetry),
            }
            for index, telemetry in enumerate(output.block_telemetry)
        ],
    }

    if args.gradients:
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        loss = masked_answer_loss(output.logits, batch.labels, batch.loss_mask)
        loss.backward()
        finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        trace["gradient_health"] = {
            "loss": float(loss.detach().item()),
            "all_finite": bool(finite),
            "group_norms": gradient_group_norms(model),
        }

    payload = json.dumps(trace, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
