"""Offline attachment audit and DEV proposal preparation; no model operations.

Run ``python -m experiments.rineya_generalization_dynamics_v1.attachment_cli``.
Prepared files still require a separately authorized native-run decision. This
module deliberately has no evaluate, select or confirmation command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attachment import native_spec_from_attachment, validate_attachment
from .common import Blocked, code_hash, digest, file_hash, normalized_prompt_hash, read, require, signed, verify, write_new
from .diagnostics import build_suite, suite_hash
from .export_parity import validate_parity_policy
from .pipeline import validate_config
from .protected_paths import reject_protected_path
from .registry import audit_training_overlap, freeze_registry


def audit_producer_prompt_overlap(rows: list[dict], attachment: dict,
                                 public_exposure_manifests: list[dict]) -> dict:
    """Exclude exact raw/rendered/consumed questions and declared public exposures.

    Only owner-supplied open public provenance belongs here, never a sealed bank.
    A source owner must declare every relevant exposure file before outcomes.
    Hashes and declarations cannot prove an owner omitted no file.
    """
    training = attachment["registry"]["training"]
    prompts, parents = set(training["prompt_hashes"]), set(training["parent_ids"])
    event_ref = attachment["provenance"]["events"]
    event_path = reject_protected_path(event_ref["path"], "Producer events")
    require(file_hash(event_path) == event_ref["sha256"], "Producer events changed")
    with event_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                event = json.loads(line)
                if event["event_type"] == "consume":
                    for field in ("prompt", "raw_rendered_prompt", "consumed_prompt"):
                        prompts.add(normalized_prompt_hash(event[field]))
                    parents.add(event["parent_id"])
    require(isinstance(public_exposure_manifests, list), "Declare the complete public exposure manifest list")
    refs, exposure_count = [], 0
    for ref in public_exposure_manifests:
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "Public exposure file reference required")
        path = Path(ref["path"]).expanduser()
        require(path.is_absolute() and not path.is_symlink(),
                "Retain the original open public exposure manifest")
        path = reject_protected_path(path, "Public exposure manifest")
        require(path.is_file(), "Retain the original open public exposure manifest")
        require(file_hash(path) == ref["sha256"], "Public exposure manifest changed")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                require(item["scope"] in {"public_training", "public_qualification", "public_dev"}
                        and item["public"] is True, "Only open public producer exposure provenance is admissible")
                for field in ("prompt", "rendered_prompt", "parent_id", "parent_namespace"):
                    require(isinstance(item[field], str) and bool(item[field].strip()),
                            "Public exposure prompt/parent payload is incomplete")
                prompts.update(normalized_prompt_hash(item[field]) for field in ("prompt", "rendered_prompt"))
                parents.add(item["parent_id"])
                count += 1
        require(count > 0 and file_hash(path) == ref["sha256"], "Empty or changed public exposure inventory")
        refs.append({"path": str(path.resolve()), "sha256": ref["sha256"]})
        exposure_count += count
    require(file_hash(event_path) == event_ref["sha256"], "Producer events changed during overlap audit")
    audit_training_overlap(rows, {"training": {"prompt_hashes": sorted(prompts), "parent_ids": sorted(parents)}})
    return signed({"status": "NO_NORMALIZED_EXACT_PRODUCER_PROMPT_OR_PARENT_OVERLAP",
                   "raw_rendered_consumed_prompts_included": True,
                   "public_exposure_manifests": refs, "public_exposure_count": exposure_count,
                   "excluded_prompt_count": len(prompts), "excluded_parent_count": len(parents),
                   "exclusion_inventory_sha256": digest({"prompts": sorted(prompts), "parents": sorted(parents)}),
                   "upstream_manifest_completeness": "owner-declared; not independently proven",
                   "semantic_decontamination": False})


def prepare_dev(contract_path: str, proposal_path: str, output_directory: str) -> dict:
    """Bind one reviewed source/schedule/policy before any native outcomes.

    Source and proposal are retained externally, never copied as metadata-only
    substitutes for weights or consumed-prompt payloads. A missing prerequisite
    fails before output creation.
    """
    contract_file = reject_protected_path(contract_path, "Attachment contract")
    proposal_file = reject_protected_path(proposal_path, "Reviewed DEV proposal")
    proposal = verify(read(proposal_file))
    require(proposal["status"] == "REVIEWED_PROPOSAL_NOT_EXECUTION_AUTHORIZATION",
            "An exact reviewed proposal is required before materialization")
    require(proposal["stage"] == "DEV_ONLY_NATIVE_LOADING_SCORING"
            and proposal["separate_native_authorization_required"] is True
            and proposal["confirmation_authorized"] is False,
            "Only preparation of separately authorized DEV work is supported")
    require(proposal["code_sha256"] == code_hash(), "Review must bind the current GD code")
    require(proposal["attachment_contract"] == {
        "path": str(contract_file), "sha256": file_hash(contract_file)},
        "Reviewed proposal does not bind the retained attachment contract")
    attachment = validate_attachment(str(contract_file))
    spec = native_spec_from_attachment(str(contract_file), proposal["runtime"])
    validate_config(proposal["policy"], proposal["runtime"])
    steps = proposal["checkpoint_steps"]
    require(len(steps) == 3 and steps[0] == 0
            and all(type(step) is int for step in steps)
            and steps == sorted(set(steps)), "First milestone is exact base plus two saved snapshots")
    require(steps[-1] == spec["checkpoints"][-1]["step"], "Keep the verified final control")
    available = {entry["step"]: entry for entry in spec["checkpoints"]}
    require(all(step in available for step in steps), "Reviewed checkpoint absent from original trajectory")
    spec["checkpoints"] = [available[step] for step in steps]
    require(proposal["dev_seed"] == 17 and proposal["pairs_per_family"] == 24,
            "This first milestone proposal uses DEV seed 17 and 24 pairs per family")
    rows = build_suite("dev", proposal["dev_seed"], proposal["pairs_per_family"])
    registry = freeze_registry(spec)
    audit_training_overlap(rows, registry)
    overlap = audit_producer_prompt_overlap(rows, attachment, proposal["public_exposure_manifests"])
    parity_policy = validate_parity_policy(proposal["parity_policy"])
    require(parity_policy["schema"] == "gd001_export_parity_policy_v1"
            and parity_policy["registered_before_outcomes"] is True
            and parity_policy["max_probes"] == 2,
            "Preregister two bounded DEV parity probes before outcomes")
    calls_per_prompt = proposal["runtime"]["max_new_tokens"] + 2
    dev_forwards = len(rows) * len(steps) * calls_per_prompt
    # Conservatively allow both producer and GD golden traces to make fresh calls.
    # Shared metering is a prerequisite of the later authorized supervisor plan.
    parity_forwards = 2 * parity_policy["max_probes"] * len(steps) * calls_per_prompt
    worst_case_forwards = dev_forwards + parity_forwards
    require(worst_case_forwards <= proposal["runtime"]["max_native_forward_calls"],
            "Registered aggregate forward budget cannot cover this DEV milestone")
    output = reject_protected_path(output_directory, "DEV preparation output")
    workspace = Path(__file__).resolve().parents[2]
    record_root = workspace.parent / "gd001-records"
    require(output.is_relative_to(record_root)
            and all(part.lower() != ".codex" for part in output.parts),
            "New preparations must stay in external BusinessProject/gd001-records")
    require(not output.exists(), "Keep all earlier preparations; choose a fresh output directory")
    receipt = signed({"experiment": "GD-001", "evidence_level": "OFFLINE_ATTACHMENT_PREPARATION",
        "status": "PREPARED_NOT_EXECUTION_AUTHORIZATION", "code_sha256": code_hash(),
        "reviewed_proposal": {"path": str(proposal_file), "sha256": file_hash(proposal_file)},
        "attachment_contract": {"path": str(contract_file), "sha256": file_hash(contract_file)},
        "attachment_validation": attachment, "producer_overlap_audit": overlap,
        "native_spec_sha256": digest(spec),
        "registry_sha256": digest(registry), "dev_suite_sha256": suite_hash(rows),
        "dev_seed": proposal["dev_seed"], "pairs_per_family": proposal["pairs_per_family"],
        "checkpoint_steps": steps, "policy": proposal["policy"], "runtime": proposal["runtime"],
        "parity_policy": parity_policy, "dev_worst_case_forward_calls": dev_forwards,
        "parity_worst_case_forward_calls": parity_forwards,
        "aggregate_budget_scope": "DEV scoring plus both producer/GD parity traces, all checkpoints and attempts",
        "shared_parity_dev_metering_required_before_execution": True,
        "total_prompts": len(rows) * len(steps), "worst_case_forward_calls": worst_case_forwards,
        "one_checkpoint_at_a_time": True, "native_model_loads": 0, "native_forward_calls": 0,
        "confirmation_authorized": False, "permits_model_promotion": False})
    output.mkdir(parents=True)
    write_new(output / "native-spec.json", spec)
    write_new(output / "policy.json", proposal["policy"])
    write_new(output / "parity-policy.json", parity_policy)
    write_new(output / "development-suite.json", rows)
    write_new(output / "DEV_PREPARATION.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("check")
    audit.add_argument("--contract", required=True)
    prepare = commands.add_parser("prepare-dev")
    prepare.add_argument("--contract", required=True)
    prepare.add_argument("--proposal", required=True)
    prepare.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = (validate_attachment(args.contract) if args.command == "check" else
                  prepare_dev(args.contract, args.proposal, args.output))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (Blocked, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
