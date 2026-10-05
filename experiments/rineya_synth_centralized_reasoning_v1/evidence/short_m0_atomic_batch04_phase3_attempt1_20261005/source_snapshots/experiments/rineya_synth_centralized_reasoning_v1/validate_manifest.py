"""Validate this historical design registration; never authorize a native run."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

IDENTITY = {
    "schema_version": 1,
    "experiment_id": "SYNTH-CRS-001",
    "state": "DESIGN_REGISTERED_NATIVE_UNQUALIFIED",
    "repository": "LUNCY579/ChatBubble",
    "branch": "research/rineya-synth-centralized-reasoning-v1",
    "validation_scope": "founding_registration_metadata_only",
}
CLOSED_PERMISSIONS = {
    "native_inference", "training", "paid_teacher_calls", "dataset_download",
    "protected_bank_access", "merge", "model_upload",
}
UNESTABLISHED_CLAIMS = {
    "runtime_implemented", "corpus_generated", "native_model_run",
    "quality_improvement_established", "retention_established",
    "composition_established", "release_permitted",
}
REQUIRED_BINDINGS = {
    "data_hashes", "native_qualification", "resource_caps",
    "retention_guard_enrollment", "statistical_method_and_power",
    "effect_and_retention_margins", "launch_approval",
}
TOP_KEYS = set(IDENTITY) | {
    "parent", "preservation", "student", "permissions", "claims", "scoreboards",
    "data_design", "arms", "primary_contrast", "primary_budget",
    "open_run_bindings", "source",
}


def validate(data: Any) -> list[str]:
    """Return metadata errors. An empty list says nothing about model readiness."""
    errors: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    if not isinstance(data, dict):
        return ["manifest must be a JSON object"]
    require(set(data) == TOP_KEYS, "missing or unknown top-level fields")
    for key, expected in IDENTITY.items():
        actual = data.get(key)
        require(type(actual) is type(expected) and actual == expected,
                f"invalid founding identity: {key}")

    parent = data.get("parent")
    require(parent == {
        "branch": "research/rineya-centralized-reasoning-system-v1",
        "commit": "7459b614bcb017ae4f0160b15e0ccd53473bb990",
        "tree": "9223d7b22e4bbbab867107a241c67ea7410a0ede",
    }, "founding parent identity changed")

    policy = data.get("preservation")
    expected_policy = {
        "complementary": True, "automatic_sunset": False,
        "explicit_owner_approval_for_retirement": True,
        "preserve_negative_results": True,
        "github_branch_protection_configured": False,
    }
    require(isinstance(policy, dict) and set(policy) == set(expected_policy),
            "preservation policy fields missing or unknown")
    if isinstance(policy, dict):
        for key, expected in expected_policy.items():
            require(policy.get(key) is expected, f"invalid preservation rule: {key}")

    for name, keys in (("permissions", CLOSED_PERMISSIONS),
                       ("claims", UNESTABLISHED_CLAIMS)):
        section = data.get(name)
        require(isinstance(section, dict) and set(section) == keys,
                f"{name} fields missing or unknown")
        if isinstance(section, dict):
            for key in keys:
                require(section.get(key) is False, f"{name}.{key} must remain false")

    require(data.get("student") == {
        "parameter_class": "0.6B", "checkpoint_sha256": None,
        "tokenizer_sha256": None,
    }, "historical student bindings must remain explicitly unqualified")
    require(data.get("scoreboards") == [
        "model_only", "tool_assisted_system", "distilled_model",
    ], "three independent scoreboards are required")
    require(data.get("data_design") == {
        "train_signatures": ["R", "S"], "held_out_signatures": ["RS"],
        "split_unit": "dependency_connected_parent_family",
        "target_format": "final_answer_only",
        "verifier": "independent_of_candidate_generator",
        "protected_banks": "closed",
    }, "atomic/held-out, lineage, target, or verifier design changed")
    expected_arms = [
        {"id": "M0", "trained": False, "data": "none", "target_format": None},
        {"id": "M1", "trained": True, "data": "surface_varied_verified_atomic",
         "target_format": "final_answer_only"},
        {"id": "M2", "trained": True, "data": "condition_diverse_verified_atomic",
         "target_format": "final_answer_only"},
    ]
    # JSON canonicalization distinguishes booleans from integers (True != 1 here).
    require(json.dumps(data.get("arms"), sort_keys=True) ==
            json.dumps(expected_arms, sort_keys=True), "planned arm controls changed")
    require(data.get("primary_contrast") == ["M2", "M1"],
            "primary comparison must be M2 versus M1, not the unchanged model")
    require(data.get("primary_budget") == "nonpadding_student_training_tokens",
            "primary budget must count student tokens, not just examples")
    bindings = data.get("open_run_bindings")
    require(isinstance(bindings, list) and all(isinstance(x, str) for x in bindings)
            and len(bindings) == len(REQUIRED_BINDINGS)
            and set(bindings) == REQUIRED_BINDINGS,
            "unresolved run requirements must remain explicit")
    source = data.get("source")
    require(isinstance(source, dict) and set(source) == {
        "paper", "dataset_card", "checked_on"}, "source fields missing or unknown")
    if isinstance(source, dict):
        require(source.get("paper") == "https://arxiv.org/abs/2609.37891v1",
                "paper version must be pinned")
        require(source.get("dataset_card") == "https://huggingface.co/datasets/PleIAs/SYNTH",
                "dataset source identity changed")
        require(isinstance(source.get("checked_on"), str) and bool(re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", source.get("checked_on", ""))),
            "source inspection date required")
    return errors


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path,
                        default=Path(__file__).with_name("manifest.json"))
    args = parser.parse_args(argv)
    try:
        with args.manifest.open("r", encoding="utf-8") as stream:
            text = stream.read(131073)
        if len(text) > 131072:
            raise ValueError("manifest exceeds 128 Ki characters")
        data = json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
        errors = validate(data)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        errors = [f"cannot read manifest: {exc}"]
    if errors:
        print(json.dumps({"status": "INVALID_REGISTRATION", "errors": errors}, indent=2))
        return 2
    print(json.dumps({"status": "VALID_DESIGN_REGISTRATION",
                      "native_execution_authorized": False,
                      "model_quality_evidence": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
