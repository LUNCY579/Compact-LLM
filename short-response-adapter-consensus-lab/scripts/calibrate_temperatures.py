from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.evaluation.sampling import deterministic_stratified_sample
from consensus_lab.experiments.config import load_experiment_config
from consensus_lab.io import read_jsonl, write_json_atomic
from consensus_lab.models.committee import ExpertCommittee
from consensus_lab.paths import LabPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit scalar adapter temperatures on DEV teacher-forced NLL.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs/fusion_sweep.yaml")
    parser.add_argument("--dataset", default="v22_dev")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0],
    )
    args = parser.parse_args()
    if any(value <= 0 for value in args.temperatures):
        raise ValueError("Temperatures must be positive")
    paths = LabPaths.discover(LAB_ROOT)
    config = load_experiment_config(args.config, paths)
    committee = ExpertCommittee.load(
        paths=paths,
        backbone=config.backbone,
        experts=config.experts,
        device=args.device,
        precision=args.precision,
    )
    rows = deterministic_stratified_sample(
        read_jsonl(config.datasets[args.dataset]), limit=args.limit, seed=config.seed + 101
    )
    losses = {
        expert.name: {float(temperature): 0.0 for temperature in args.temperatures}
        for expert in committee.experts
    }
    token_counts = {expert.name: 0 for expert in committee.experts}
    calibration_ids = [str(row["id"]) for row in rows]
    for row in rows:
        messages = [dict(message) for message in row["messages"]]
        prompt_ids = committee.encode_messages(messages[:-1], max_sequence_length=config.max_sequence_length)
        gold = str(row.get("approved_answer") or row.get("answer") or messages[-1].get("content") or "")
        target_values = committee.tokenizer(gold, add_special_tokens=False)["input_ids"]
        if committee.tokenizer.eos_token_id is not None:
            target_values = [*target_values, int(committee.tokenizer.eos_token_id)]
        target = torch.tensor([target_values], device=committee.device, dtype=prompt_ids.dtype)
        full = torch.cat([prompt_ids, target], dim=1)
        model_input = full[:, :-1]
        target_ids = full[0, -len(target_values) :]
        for expert in committee.experts:
            logits = committee.teacher_forced_logits(expert, model_input, target_length=len(target_values))
            for temperature in args.temperatures:
                log_probs = torch.log_softmax(logits / temperature, dim=-1)
                loss = -log_probs.gather(dim=-1, index=target_ids[:, None]).sum()
                losses[expert.name][float(temperature)] += float(loss.item())
            token_counts[expert.name] += len(target_values)
    experts: dict[str, dict[str, object]] = {}
    for expert in committee.experts:
        count = token_counts[expert.name]
        mean_losses = {
            str(temperature): total / count for temperature, total in losses[expert.name].items()
        }
        best = min(args.temperatures, key=lambda value: (mean_losses[str(float(value))], value))
        experts[expert.name] = {
            "temperature": float(best),
            "token_count": count,
            "mean_nll_by_temperature": mean_losses,
        }
    packet = {
        "schema_version": "1.0",
        "status": "DEV_CALIBRATION_ONLY",
        "config": str(args.config.resolve()),
        "config_sha256_before_temperature_update": config.config_sha256,
        "dataset": args.dataset,
        "dataset_path": str(config.datasets[args.dataset]),
        "seed": config.seed + 101,
        "example_count": len(rows),
        "example_ids": calibration_ids,
        "objective": "teacher-forced gold-token negative log likelihood including EOS",
        "temperature_grid": args.temperatures,
        "experts": experts,
        "warning": "DEV-fitted temperatures are selection parameters; assess them finally only on frozen sealed data.",
    }
    output = paths.assert_output_path(LAB_ROOT / "artifacts/calibration/v22_temperatures.json")
    write_json_atomic(output, packet)
    print(json.dumps({"output": str(output), "experts": experts}, indent=2))


if __name__ == "__main__":
    main()
