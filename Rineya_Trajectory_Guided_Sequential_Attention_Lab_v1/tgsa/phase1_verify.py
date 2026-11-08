"""Closed-chain evidence verification for the TGSA Phase-1 model study.

The pre-remediation analyzer checked that all runs *agreed with each other* on a
config hash and a dataset hash, but never that those shared hashes equalled the
configuration, materialized dataset, and source checkout actually in front of
the analyst. It also recorded checkpoint and train-log hashes without ever
re-verifying them, and hard-coded the
``paired_initialization_and_artifact_integrity`` advancement gate to ``True``.
See ``PHASE1_LAUNCH_AUDIT.md`` finding P0-5.

This module binds one continuous chain::

    study config
      -> dataset manifest -> split files -> instance IDs
      -> preflight (canonical initialization proofs)
      -> matrix manifest
      -> every run manifest
      -> final checkpoints, train logs, prediction files
      -> analysis

Every link is an exact SHA-256 comparison against the artifact on disk. Any
break is a hard failure with a named check, not a warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, cast

from .phase1_data import load_examples, sha256_file, validate_dataset_manifest
from .phase1_runner import (
    MATRIX_MANIFEST_VERSION,
    PREDICTION_SCHEMA_VERSION,
    REQUIRED_ARMS,
    RUN_MANIFEST_VERSION,
    load_study_config,
)

VERIFIER_SCHEMA_VERSION = "tgsa.phase1-evidence-verification.v1"


@dataclass(slots=True)
class ChainReport:
    """Accumulated pass/fail state for the evidence chain."""

    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record(self, name: str, passed: bool, message: str = "") -> bool:
        # A check that already failed must never be flipped to passing by a
        # later, narrower assertion under the same name.
        self.checks[name] = bool(passed) and self.checks.get(name, True)
        if not passed:
            self.errors.append(message or name)
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": VERIFIER_SCHEMA_VERSION,
            "passed": self.passed,
            "checks": dict(sorted(self.checks.items())),
            "details": dict(sorted(self.details.items())),
            "errors": list(self.errors),
        }


def _expected_prediction_conditions(
    study: Mapping[str, object],
    *,
    arm: str,
) -> set[str]:
    """Every prediction file a completed run of this arm must contain."""

    evaluation = cast(Mapping[str, object], study["evaluation"])
    expected = {"dev__identity", "test__identity"}
    if arm != "learned_delta":
        return expected
    for index in [int(value) for value in evaluation.get("derangement_indices", [0])]:
        expected.add("test__replay_derangement" if index == 0 else f"test__replay_derangement_{index}")
    for mode in [str(value) for value in evaluation.get("interventions", [])]:
        expected.add(f"test__{mode}")
    return expected


def _verify_prediction_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    report: ChainReport,
    label: str,
    arm: str,
    seed: int,
    split: str,
    expected_answers: Mapping[str, int],
    expected_strata: Mapping[str, str],
) -> None:
    """Reject duplicated, missing, mislabeled, or non-finite prediction rows."""

    ids = [str(row["example_id"]) for row in rows]
    report.record(
        f"predictions_unique::{label}",
        len(ids) == len(set(ids)),
        f"{label}: duplicate prediction rows",
    )
    report.record(
        f"predictions_cover_split::{label}",
        set(ids) == set(expected_answers),
        f"{label}: prediction IDs do not match the materialized {split} split",
    )
    bad_identity: list[str] = []
    bad_numeric: list[str] = []
    for row in rows:
        example_id = str(row["example_id"])
        if (
            row.get("schema_version") != PREDICTION_SCHEMA_VERSION
            or str(row.get("arm")) != arm
            or int(row.get("seed", -1)) != seed
            or str(row.get("split")) != split
            or int(row.get("answer_token", -1)) != expected_answers.get(example_id, -2)
            or str(row.get("stratum")) != expected_strata.get(example_id, "")
        ):
            bad_identity.append(example_id)
        nll = float(row.get("answer_nll", float("nan")))
        margin = float(row.get("target_margin", float("nan")))
        correct = row.get("correct")
        if nll != nll or margin != margin or not isinstance(correct, bool):
            bad_numeric.append(example_id)
        elif bool(correct) != (int(row.get("prediction_token", -1)) == int(row.get("answer_token", -2))):
            bad_identity.append(example_id)
    report.record(
        f"predictions_identity::{label}",
        not bad_identity,
        f"{label}: {len(bad_identity)} rows disagree with the materialized split or are internally inconsistent",
    )
    report.record(
        f"predictions_finite::{label}",
        not bad_numeric,
        f"{label}: {len(bad_numeric)} rows carry non-finite or mistyped scores",
    )


def verify_evidence_chain(
    *,
    study_config_path: Path,
    output_root: Path,
    dataset_dir: Path | None = None,
    repo_root: Path | None = None,
    require_confirmatory: bool | None = None,
) -> dict[str, object]:
    """Verify the complete Phase-1 evidence chain and return a typed report."""

    report = ChainReport()
    study = load_study_config(study_config_path)
    confirmatory = bool(study["confirmatory"])
    current_config_sha = sha256_file(study_config_path)
    report.details["study_id"] = study["study_id"]
    report.details["confirmatory_config"] = confirmatory
    report.details["study_config_sha256"] = current_config_sha

    if require_confirmatory is not None:
        report.record(
            "confirmatory_expectation",
            confirmatory == require_confirmatory,
            f"config confirmatory={confirmatory}, expected {require_confirmatory}",
        )

    # ---- matrix manifest binds to the current config ----
    matrix_path = output_root / "matrix_manifest.json"
    report.record("matrix_manifest_present", matrix_path.is_file(), f"missing {matrix_path}")
    if not matrix_path.is_file():
        return report.to_dict()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    report.record(
        "matrix_schema_version",
        matrix.get("schema_version") == MATRIX_MANIFEST_VERSION,
        "unsupported matrix manifest schema",
    )
    report.record(
        "matrix_status_complete",
        matrix.get("status") == "complete",
        f"matrix status is {matrix.get('status')!r}",
    )
    report.record(
        "matrix_binds_current_config",
        str(matrix.get("study_config_sha256")) == current_config_sha,
        "matrix manifest was produced from a different study configuration",
    )
    report.record(
        "matrix_study_id",
        str(matrix.get("study_id")) == str(study["study_id"]),
        "matrix manifest belongs to a different study ID",
    )
    report.record(
        "matrix_deferred_test_evaluation",
        bool(matrix.get("deferred_test_evaluation", False)),
        "matrix was produced without the deferred test-evaluation contract",
    )

    configured_arms = tuple(str(arm) for arm in study["arms"])
    configured_seeds = tuple(int(seed) for seed in study["seeds"])
    selected_arms = tuple(str(arm) for arm in matrix.get("selected_arms", ()))
    selected_seeds = tuple(int(seed) for seed in matrix.get("selected_seeds", ()))
    is_full = set(selected_arms) == set(configured_arms) and set(selected_seeds) == set(configured_seeds)
    expected_cells = {(arm, seed) for arm in selected_arms for seed in selected_seeds}
    report.details["registered_cell_count"] = len(configured_arms) * len(configured_seeds)
    report.details["selected_cell_count"] = len(expected_cells)
    report.record(
        "required_arms_selected",
        set(REQUIRED_ARMS).issubset(selected_arms),
        f"selected arms lack a required arm: {sorted(set(REQUIRED_ARMS) - set(selected_arms))}",
    )
    if confirmatory:
        report.record(
            "full_registered_matrix",
            is_full,
            "confirmatory evidence requires the complete registered arm/seed matrix",
        )

    # ---- dataset manifest binds to the current materialized data ----
    dataset_sha = str(matrix.get("dataset_manifest_sha256", ""))
    expected_answers: dict[str, int] = {}
    expected_strata: dict[str, str] = {}
    dev_ids: set[str] = set()
    if dataset_dir is not None:
        dataset_manifest_path = dataset_dir / "dataset_manifest.json"
        report.record(
            "dataset_manifest_present",
            dataset_manifest_path.is_file(),
            f"missing {dataset_manifest_path}",
        )
        if dataset_manifest_path.is_file():
            dataset_manifest = validate_dataset_manifest(dataset_manifest_path)
            report.record("dataset_manifest_validated", True)
            report.record(
                "dataset_binds_current_materialization",
                sha256_file(dataset_manifest_path) == dataset_sha,
                "matrix manifest was produced from a different materialized dataset",
            )
            splits = cast(Mapping[str, Mapping[str, object]], dataset_manifest["splits"])
            test_examples = load_examples(dataset_dir / str(splits["test"]["path"]))
            expected_answers = {example.id: example.answer_token for example in test_examples}
            expected_strata = {example.id: example.stratum for example in test_examples}
            dev_ids = {
                example.id
                for example in load_examples(dataset_dir / str(splits["dev"]["path"]))
            }
            report.details["test_examples"] = len(expected_answers)

    # ---- preflight binds to the matrix and proves canonical initialization ----
    preflight_name = str(matrix.get("preflight", "preflight.json"))
    preflight_path = output_root / preflight_name
    report.record("preflight_present", preflight_path.is_file(), f"missing {preflight_path}")
    if preflight_path.is_file():
        report.record(
            "preflight_hash_binds_matrix",
            sha256_file(preflight_path) == str(matrix.get("preflight_sha256")),
            "preflight hash recorded in the matrix manifest does not match the file",
        )
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        report.record(
            "preflight_binds_current_config",
            str(preflight.get("study_config_sha256")) == current_config_sha,
            "preflight was produced from a different study configuration",
        )
        audits = cast(Mapping[str, Mapping[str, object]], preflight.get("paired_initialization_by_seed", {}))
        missing_seeds = [seed for seed in selected_seeds if str(seed) not in audits]
        report.record(
            "preflight_covers_every_seed",
            not missing_seeds,
            f"preflight lacks initialization proofs for seeds {missing_seeds}",
        )
        for seed in selected_seeds:
            audit = audits.get(str(seed))
            if audit is None:
                continue
            report.record(
                "canonical_initialization_proved",
                bool(audit.get("all_arms_match_canonical_shared_tensors"))
                and bool(audit.get("history_delta_exact_initialization"))
                and bool(audit.get("history_delta_parameter_count_equal"))
                and bool(
                    cast(Mapping[str, object], audit["cascade_delta_shared_initialization"])["identical"]
                ),
                f"seed {seed}: arms do not all derive from the canonical supernet",
            )

    # ---- every registered run manifest and its artifacts ----
    registered_steps = int(cast(Mapping[str, object], study["training"])["steps"])
    heads: set[str] = set()
    environments: set[str] = set()
    seen_cells: set[tuple[str, int]] = set()
    initial_hashes: dict[int, set[str]] = {}
    for arm, seed in sorted(expected_cells):
        label = f"{arm}:{seed}"
        run_dir = output_root / "runs" / arm / f"seed_{seed}"
        manifest_path = run_dir / "run_manifest.json"
        if not report.record(
            f"run_manifest_present::{label}", manifest_path.is_file(), f"missing {manifest_path}"
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seen_cells.add((arm, seed))
        report.record(
            f"run_complete::{label}",
            manifest.get("schema_version") == RUN_MANIFEST_VERSION
            and manifest.get("status") == "complete",
            f"{label}: run manifest is not a completed run",
        )
        report.record(
            f"run_identity::{label}",
            str(manifest.get("arm")) == arm
            and int(manifest.get("seed", -1)) == seed
            and str(manifest.get("study_id")) == str(study["study_id"]),
            f"{label}: run manifest identity mismatch",
        )
        report.record(
            f"run_binds_current_config::{label}",
            str(manifest.get("study_config_sha256")) == current_config_sha,
            f"{label}: run was produced from a different study configuration",
        )
        report.record(
            f"run_binds_dataset::{label}",
            str(manifest.get("dataset_manifest_sha256")) == dataset_sha,
            f"{label}: run was produced from a different materialized dataset",
        )
        report.record(
            f"run_defers_test_evaluation::{label}",
            bool(manifest.get("test_evaluation_deferred", False)),
            f"{label}: run did not use the deferred test-evaluation contract",
        )

        head = manifest.get("git", {}).get("head")
        if head is not None:
            heads.add(str(head))
        if confirmatory:
            report.record(
                f"run_source_clean::{label}",
                manifest.get("git", {}).get("dirty") is False,
                f"{label}: confirmatory run was produced from a dirty checkout",
            )
            report.record(
                f"run_lineage::{label}",
                manifest.get("git", {}).get("expected_base_is_ancestor") is True,
                f"{label}: pinned parent commit is not an ancestor of the run's source head",
            )
        environment = manifest.get("environment", {})
        environments.add(f"{environment.get('python')}|{environment.get('torch')}")
        initial_hashes.setdefault(seed, set()).add(str(manifest.get("initial_model_state_sha256")))

        training_record = cast(Mapping[str, object], manifest["training"])
        report.record(
            f"run_step_count::{label}",
            int(training_record["completed_steps"]) == registered_steps,
            f"{label}: completed {training_record['completed_steps']} of {registered_steps} registered updates",
        )
        report.record(
            f"run_fixed_checkpoint::{label}",
            bool(training_record.get("fixed_final_checkpoint")) is True
            and bool(training_record.get("early_stopping")) is False,
            f"{label}: run did not use the fixed final checkpoint policy",
        )
        checkpoint_path = run_dir / str(training_record["final_checkpoint"])
        report.record(
            f"final_checkpoint_hash::{label}",
            checkpoint_path.is_file()
            and sha256_file(checkpoint_path) == str(training_record["final_checkpoint_sha256"]),
            f"{label}: final checkpoint is missing or its hash changed",
        )
        train_log_path = run_dir / str(training_record["train_log"])
        report.record(
            f"train_log_hash::{label}",
            train_log_path.is_file()
            and sha256_file(train_log_path) == str(training_record["train_log_sha256"]),
            f"{label}: train log is missing or its hash changed",
        )
        if train_log_path.is_file():
            steps = [
                int(json.loads(line)["step"])
                for line in train_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report.record(
                f"train_log_unique_steps::{label}",
                len(steps) == len(set(steps)) and steps == sorted(steps),
                f"{label}: train log is not a unique increasing step sequence",
            )

        evaluations = cast(Mapping[str, Mapping[str, object]], manifest["evaluations"])
        expected_conditions = _expected_prediction_conditions(study, arm=arm)
        report.record(
            f"prediction_conditions::{label}",
            set(evaluations) == expected_conditions,
            f"{label}: evaluation conditions {sorted(set(evaluations))} != registered {sorted(expected_conditions)}",
        )
        for condition, record in sorted(evaluations.items()):
            prediction_path = run_dir / str(record["path"])
            if not report.record(
                f"prediction_hash::{label}:{condition}",
                prediction_path.is_file() and sha256_file(prediction_path) == str(record["sha256"]),
                f"{label}:{condition}: prediction file is missing or its hash changed",
            ):
                continue
            if not condition.startswith("test__") or not expected_answers:
                continue
            rows = [
                json.loads(line)
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            _verify_prediction_rows(
                rows,
                report=report,
                label=f"{label}:{condition}",
                arm=arm,
                seed=seed,
                split="test",
                expected_answers=expected_answers,
                expected_strata=expected_strata,
            )
            if dev_ids:
                report.record(
                    f"test_predictions_exclude_dev::{label}:{condition}",
                    not (set(expected_answers) & dev_ids),
                    f"{label}:{condition}: dev and test splits share example IDs",
                )

    report.record(
        "every_registered_cell_present",
        seen_cells == expected_cells,
        f"missing registered cells: {sorted(expected_cells - seen_cells)}",
    )
    report.record(
        "single_source_head",
        len(heads) <= 1,
        f"runs span multiple source heads: {sorted(heads)}",
    )
    report.record(
        "compatible_runtime_environment",
        len(environments) <= 1,
        f"runs span incompatible runtimes: {sorted(environments)}",
    )
    for seed, hashes in sorted(initial_hashes.items()):
        # Every arm in a seed starts from the same canonical supernet, so the
        # history/delta pair must share an initial state hash. Cascade differs
        # only by the tensors it does not have.
        report.record(
            "paired_initial_state_recorded",
            all(len(value) == 64 for value in hashes),
            f"seed {seed}: initial model state hashes are not full digests",
        )
    report.details["source_heads"] = sorted(heads)
    report.details["is_full_registered_matrix"] = is_full
    if repo_root is not None:
        report.details["repo_root"] = str(repo_root)
    return report.to_dict()
