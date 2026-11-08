#!/usr/bin/env python3
"""Fast developer/agent preflight for the TGSA Phase-1 study."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, cast
import torch
LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
from tgsa.phase1_data import derive_seed, fixed_batches, validate_dataset_manifest
from tgsa.phase1_model import (
    ControlIntervention,
    ModelArm,
    answer_logits,
    build_model,
    masked_answer_loss,
    paired_initialization_audit,
    state_dict_sha256,
)
from tgsa.phase1_runner import (
    _model_config_from_study,
    atomic_write_json,
    configure_runtime,
    git_snapshot,
    load_study_config,
    load_study_examples,
    prepare_study_dataset,
    resolve_device,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    study = load_study_config(args.config.resolve())
    dataset_manifest = prepare_study_dataset(study, dataset_dir=args.dataset_dir.resolve())
    validate_dataset_manifest(args.dataset_dir.resolve() / "dataset_manifest.json")
    examples = load_study_examples(args.dataset_dir.resolve())
    runtime = cast(Mapping[str, object], study["runtime"])
    seed = int(study["seeds"][0])
    configure_runtime(runtime, derive_seed(seed, "doctor-runtime"))
    device = resolve_device(runtime)
    vocabulary = cast(Mapping[str, object], dataset_manifest["vocabulary"])
    max_sequence_length = max(
        int(entry["max_tokens"]) - 1
        for entry in cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"]).values()
    )
    base_config = _model_config_from_study(
        study,
        arm="learned_delta",
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    initialization_seed = derive_seed(seed, "model-initialization")
    init_audit = paired_initialization_audit(
        base_config=base_config,
        initialization_seed=initialization_seed,
    )
    batch = fixed_batches(examples["dev"][:4], batch_size=4).__next__().to(device)
    arm_checks: dict[str, object] = {}
    for arm_text in ("cascade", "history_control", "learned_delta"):
        arm = cast(ModelArm, arm_text)
        config = _model_config_from_study(
            study,
            arm=arm,
            vocabulary=vocabulary,
            max_sequence_length=max_sequence_length,
        )
        model = build_model(config, initialization_seed=initialization_seed).to(device)
        model.train()
        output = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        loss = masked_answer_loss(output.logits, batch.labels, batch.loss_mask)
        loss.backward()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        arm_checks[arm] = {
            "logits_shape": list(output.logits.shape),
            "loss": float(loss.detach().item()),
            "finite_loss": bool(torch.isfinite(loss).item()),
            "finite_gradients": bool(finite_gradients),
            "state_sha256": state_dict_sha256(model),
        }
    learned_config = _model_config_from_study(
        study,
        arm="learned_delta",
        vocabulary=vocabulary,
        max_sequence_length=max_sequence_length,
    )
    learned_model = build_model(learned_config, initialization_seed=initialization_seed).to(device).eval()
    interventions: dict[str, object] = {}
    with torch.inference_mode():
        for mode in ("identity", "zero", "negate", "roll_batch"):
            if mode == "identity":
                output = learned_model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
                records = []
            else:
                with ControlIntervention(learned_model, mode, capture_records=True, max_records=32) as hook:  # type: ignore[arg-type]
                    output = learned_model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
                    records = hook.records
            selected = answer_logits(output.logits, batch.answer_input_indices)
            interventions[mode] = {
                "finite": bool(torch.isfinite(selected).all().item()),
                "answer_logits_sha256": __import__("hashlib").sha256(
                    selected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "records": records,
            }
    lingering_hooks = sum(len(module._forward_pre_hooks) for module in learned_model.modules())
    git = git_snapshot(args.repo_root.resolve(), str(study["expected_base_commit"]))
    checks = {
        "dataset_valid": True,
        "base_commit_is_ancestor": git["expected_base_is_ancestor"],
        "history_delta_exact_initialization": init_audit["history_delta_exact_initialization"],
        "history_delta_parameter_count_equal": init_audit["history_delta_parameter_count_equal"],
        "all_forward_backward_finite": all(
            bool(value["finite_loss"]) and bool(value["finite_gradients"])
            for value in arm_checks.values()  # type: ignore[union-attr]
        ),
        "all_interventions_finite": all(
            bool(value["finite"]) for value in interventions.values()  # type: ignore[union-attr]
        ),
        "intervention_hooks_cleaned_up": lingering_hooks == 0,
    }
    passed = all(value is True or value is None for value in checks.values())
    report: dict[str, object] = {
        "schema_version": "tgsa.phase1-doctor.v1",
        "passed": passed,
        "checks": checks,
        "git": git,
        "device": str(device),
        "initialization_audit": init_audit,
        "arm_checks": arm_checks,
        "interventions": interventions,
    }
    if args.output is not None:
        atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
