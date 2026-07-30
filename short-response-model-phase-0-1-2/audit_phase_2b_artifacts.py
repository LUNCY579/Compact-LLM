from __future__ import annotations

"""Audit Phase 2B artifacts for completeness, stale files, and interrupted runs.

Place this file in the repository root and run:

    python .\audit_phase_2b_artifacts.py

The audit is read-only. It never deletes or modifies experiment artifacts.
It validates expected file names, counts, JSON/JSONL integrity, stage lineage,
raw/recovered candidate counts, temporary work-model checkpoints, frontier
models, and final catalog consistency.

Exit codes:
    0 = structurally clean (complete or cleanly in progress)
    1 = errors/stale/orphaned content found
    2 = invalid arguments or repository/artifact root not found
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


STAGE_DIR_RE = re.compile(r"^stage_(\d{2})$")
RAW_RE = re.compile(r"^stage_(\d{2})_drop_original_(\d+)_raw$")
RECOVERED_RE = re.compile(r"^stage_(\d{2})_drop_original_(\d+)_recovered$")
FRONTIER_RE = re.compile(r"^stage_(\d{2})_drop_(\d+(?:_\d+)*)$")

IGNORED_NAMES = {"Thumbs.db", ".DS_Store", "desktop.ini"}

GENERATION_REQUIRED = {
    "results.jsonl",
    "summary.json",
    "MODEL_EVALUATION.md",
    "BASELINE_EVALUATION.md",
}
TEACHER_REQUIRED = {"results.jsonl", "summary.json"}
CANDIDATE_ROOT_REQUIRED = {
    "generation",
    "teacher_forcing",
    "combined_metrics.json",
    "footprint.json",
    "output_deviation.json",
    "OUTPUT_DEVIATION.md",
}
REFERENCE_EVIDENCE_REQUIRED = {"generation", "teacher_forcing", "footprint.json"}


@dataclass
class Finding:
    severity: str  # ERROR, WARNING, INFO
    code: str
    path: str
    message: str
    safe_to_delete: bool = False


@dataclass
class FolderCount:
    path: str
    actual_files_recursive: int
    actual_dirs_recursive: int
    bytes_recursive: int
    expected: str
    status: str


@dataclass
class AuditReport:
    repository_root: str
    artifact_root: str
    run_state: str
    base_layer_count: int | None
    probe_case_count: int | None
    max_drops: int
    recovery_top_k: int
    findings: list[Finding] = field(default_factory=list)
    folder_counts: list[FolderCount] = field(default_factory=list)
    stage_summaries: list[dict[str, Any]] = field(default_factory=list)
    safe_cleanup_candidates: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(item.severity == "ERROR" for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "WARNING" for item in self.findings)


class Auditor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = args.repo_root.resolve()
        self.artifact_root = (self.repo_root / args.artifact_root).resolve()
        self.report = AuditReport(
            repository_root=str(self.repo_root),
            artifact_root=str(self.artifact_root),
            run_state="UNKNOWN",
            base_layer_count=None,
            probe_case_count=None,
            max_drops=args.max_drops,
            recovery_top_k=args.recovery_top_k,
        )
        self.catalog: dict[str, Any] | None = None
        self.registry: dict[str, Any] | None = None
        self.frontier_by_stage: dict[int, Path] = {}
        self.evidence_by_stage: dict[int, Path] = {}

    def add(
        self,
        severity: str,
        code: str,
        path: Path | str,
        message: str,
        *,
        safe_to_delete: bool = False,
    ) -> None:
        path_text = self.relative(path)
        finding = Finding(severity, code, path_text, message, safe_to_delete)
        self.report.findings.append(finding)
        if safe_to_delete and path_text not in self.report.safe_cleanup_candidates:
            self.report.safe_cleanup_candidates.append(path_text)

    def relative(self, path: Path | str) -> str:
        item = Path(path)
        try:
            return str(item.resolve().relative_to(self.repo_root))
        except (ValueError, OSError):
            return str(item)

    @staticmethod
    def count_tree(path: Path) -> tuple[int, int, int]:
        if not path.exists():
            return 0, 0, 0
        files = 0
        dirs = 0
        size = 0
        for item in path.rglob("*"):
            if item.name in IGNORED_NAMES:
                continue
            if item.is_dir():
                dirs += 1
            elif item.is_file():
                files += 1
                try:
                    size += item.stat().st_size
                except OSError:
                    pass
        return files, dirs, size

    def record_count(self, path: Path, expected: str, status: str) -> None:
        files, dirs, size = self.count_tree(path)
        self.report.folder_counts.append(
            FolderCount(self.relative(path), files, dirs, size, expected, status)
        )

    def read_json(self, path: Path, *, required: bool = True) -> dict[str, Any] | None:
        if not path.exists():
            if required:
                self.add("ERROR", "MISSING_JSON", path, "Required JSON file is missing.")
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.add("ERROR", "INVALID_JSON", path, f"Cannot parse JSON: {exc}")
            return None
        if not isinstance(value, dict):
            self.add("ERROR", "JSON_NOT_OBJECT", path, "Expected a JSON object at the top level.")
            return None
        return value

    def jsonl_count(self, path: Path, *, required: bool = True) -> int | None:
        if not path.exists():
            if required:
                self.add("ERROR", "MISSING_JSONL", path, "Required JSONL file is missing.")
            return None
        count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        self.add(
                            "ERROR",
                            "INVALID_JSONL",
                            path,
                            f"Line {line_number} is invalid JSON: {exc}",
                        )
                        return None
                    count += 1
        except (OSError, UnicodeError) as exc:
            self.add("ERROR", "UNREADABLE_JSONL", path, f"Cannot read JSONL: {exc}")
            return None
        return count

    def immediate_names(self, path: Path) -> set[str]:
        if not path.exists() or not path.is_dir():
            return set()
        return {item.name for item in path.iterdir() if item.name not in IGNORED_NAMES}

    def check_exact_names(
        self,
        path: Path,
        required: set[str],
        *,
        allowed_extra: set[str] | None = None,
        missing_severity: str = "ERROR",
    ) -> None:
        if not path.exists():
            self.add(missing_severity, "MISSING_DIRECTORY", path, "Directory is missing.")
            return
        actual = self.immediate_names(path)
        missing = sorted(required - actual)
        if missing:
            self.add(
                missing_severity,
                "MISSING_EXPECTED_FILES",
                path,
                "Missing: " + ", ".join(missing),
            )
        extras = actual - required - (allowed_extra or set())
        if extras:
            self.add(
                "WARNING",
                "UNEXPECTED_FILES",
                path,
                "Unexpected entries: " + ", ".join(sorted(extras)),
            )

    def infer_probe_count(self) -> int | None:
        if self.args.probe_count is not None:
            return self.args.probe_count
        probe = self.repo_root / self.args.probe_file
        return self.jsonl_count(probe, required=False)

    def infer_base_layers(self) -> int | None:
        if self.args.base_layers is not None:
            return self.args.base_layers
        config = self.read_json(
            self.artifact_root / "reference" / "full_model" / "config.json",
            required=False,
        )
        if config and isinstance(config.get("num_hidden_layers"), int):
            return int(config["num_hidden_layers"])
        footprint = self.read_json(
            self.artifact_root / "reference" / "evidence" / "footprint.json",
            required=False,
        )
        if footprint and isinstance(footprint.get("num_hidden_layers"), int):
            return int(footprint["num_hidden_layers"])
        return None

    def audit_checkpoint(
        self,
        path: Path,
        *,
        manifest_name: str | None,
        context: str,
        incomplete_safe_to_delete: bool = False,
    ) -> bool:
        if not path.exists():
            self.add("ERROR", "MISSING_CHECKPOINT", path, f"{context} checkpoint directory is missing.")
            return False
        names = self.immediate_names(path)
        required = {"config.json"}
        if manifest_name:
            required.add(manifest_name)
        missing = required - names
        weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
        tokenizer_marker = any(
            (path / name).exists()
            for name in ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
        )
        complete = not missing and bool(weight_files) and tokenizer_marker
        if not complete:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if not weight_files:
                details.append("no model weight shard")
            if not tokenizer_marker:
                details.append("no tokenizer marker")
            self.add(
                "ERROR",
                "INCOMPLETE_CHECKPOINT",
                path,
                f"{context} checkpoint is incomplete: " + "; ".join(details),
                safe_to_delete=incomplete_safe_to_delete,
            )
        return complete

    def audit_generation(self, path: Path, expected_cases: int | None, *, in_progress: bool) -> bool:
        missing_severity = "INFO" if in_progress else "ERROR"
        self.check_exact_names(path, GENERATION_REQUIRED, missing_severity=missing_severity)
        if not (path / "summary.json").exists() or not (path / "results.jsonl").exists():
            return False
        summary = self.read_json(path / "summary.json")
        rows = self.jsonl_count(path / "results.jsonl")
        if summary and rows is not None:
            case_count = summary.get("case_count")
            if isinstance(case_count, int) and case_count != rows:
                self.add(
                    "ERROR",
                    "CASE_COUNT_MISMATCH",
                    path,
                    f"summary.json case_count={case_count}, but results.jsonl has {rows} rows.",
                )
            if expected_cases is not None and rows != expected_cases:
                severity = "INFO" if in_progress and rows < expected_cases else "ERROR"
                self.add(
                    severity,
                    "UNEXPECTED_PROBE_ROWS",
                    path,
                    f"Expected {expected_cases} rows, found {rows}.",
                )
        return GENERATION_REQUIRED <= self.immediate_names(path)

    def audit_teacher(self, path: Path, expected_cases: int | None, *, in_progress: bool) -> bool:
        missing_severity = "INFO" if in_progress else "ERROR"
        self.check_exact_names(path, TEACHER_REQUIRED, missing_severity=missing_severity)
        if not (path / "summary.json").exists() or not (path / "results.jsonl").exists():
            return False
        summary = self.read_json(path / "summary.json")
        rows = self.jsonl_count(path / "results.jsonl")
        if summary and rows is not None:
            case_count = summary.get("case_count")
            if isinstance(case_count, int) and case_count != rows:
                self.add(
                    "ERROR",
                    "CASE_COUNT_MISMATCH",
                    path,
                    f"summary.json case_count={case_count}, but results.jsonl has {rows} rows.",
                )
            if expected_cases is not None and rows != expected_cases:
                severity = "INFO" if in_progress and rows < expected_cases else "ERROR"
                self.add(
                    severity,
                    "UNEXPECTED_PROBE_ROWS",
                    path,
                    f"Expected {expected_cases} rows, found {rows}.",
                )
        return TEACHER_REQUIRED <= self.immediate_names(path)

    def candidate_complete(self, path: Path, expected_cases: int | None, *, recovered: bool) -> bool:
        required = set(CANDIDATE_ROOT_REQUIRED)
        if recovered:
            required.add("recovery_training")
        self.check_exact_names(path, required)
        generation_ok = self.audit_generation(path / "generation", expected_cases, in_progress=False)
        teacher_ok = self.audit_teacher(path / "teacher_forcing", expected_cases, in_progress=False)
        root_ok = required <= self.immediate_names(path)
        for json_name in ("combined_metrics.json", "footprint.json", "output_deviation.json"):
            self.read_json(path / json_name)
        if recovered:
            recovery = path / "recovery_training"
            self.read_json(recovery / "training_manifest.json")
            adapter = recovery / "adapter"
            if not (adapter / "adapter_config.json").exists():
                self.add(
                    "ERROR",
                    "INCOMPLETE_RECOVERY_ADAPTER",
                    adapter,
                    "Recovered candidate is missing adapter/adapter_config.json.",
                )
                root_ok = False
        return root_ok and generation_ok and teacher_ok

    def audit_reference(self) -> None:
        reference = self.artifact_root / "reference"
        if not reference.exists():
            self.add("ERROR", "MISSING_REFERENCE", reference, "Phase 2B reference directory is missing.")
            return
        model = reference / "full_model"
        model_ok = self.audit_checkpoint(
            model,
            manifest_name="materialization_manifest.json",
            context="Full reference",
        )
        evidence = reference / "evidence"
        self.check_exact_names(evidence, REFERENCE_EVIDENCE_REQUIRED)
        generation_ok = self.audit_generation(
            evidence / "generation", self.report.probe_case_count, in_progress=False
        )
        teacher_ok = self.audit_teacher(
            evidence / "teacher_forcing", self.report.probe_case_count, in_progress=False
        )
        self.read_json(evidence / "footprint.json")
        status = "OK" if model_ok and generation_ok and teacher_ok else "INCOMPLETE"
        self.record_count(reference, "complete reference checkpoint + 7 evidence files", status)

    def discover_stage_dirs(self) -> None:
        evidence_root = self.artifact_root / "evidence"
        if evidence_root.exists():
            for item in evidence_root.iterdir():
                if not item.is_dir():
                    continue
                match = STAGE_DIR_RE.fullmatch(item.name)
                if match:
                    stage = int(match.group(1))
                    self.evidence_by_stage[stage] = item
                else:
                    self.add(
                        "WARNING",
                        "UNEXPECTED_EVIDENCE_DIRECTORY",
                        item,
                        "Directory does not match stage_XX and may be stale.",
                        safe_to_delete=True,
                    )
        frontier_root = self.artifact_root / "frontier"
        if frontier_root.exists():
            for item in frontier_root.iterdir():
                if not item.is_dir():
                    continue
                match = FRONTIER_RE.fullmatch(item.name)
                if not match:
                    self.add(
                        "WARNING",
                        "UNEXPECTED_FRONTIER_DIRECTORY",
                        item,
                        "Directory does not match the accepted frontier naming contract.",
                    )
                    continue
                stage = int(match.group(1))
                if stage in self.frontier_by_stage:
                    self.add(
                        "ERROR",
                        "DUPLICATE_FRONTIER_STAGE",
                        item,
                        f"More than one frontier directory claims stage {stage}.",
                    )
                self.frontier_by_stage[stage] = item

    def expected_ids_for_stage(self, stage: int) -> list[int] | None:
        if self.report.base_layer_count is None:
            return None
        if stage == 1:
            return list(range(self.report.base_layer_count))
        parent = self.frontier_by_stage.get(stage - 1)
        if parent is None:
            return None
        payload = self.read_json(parent / "stage_result.json")
        if not payload:
            return None
        ids = payload.get("retained_original_layer_ids")
        if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
            self.add(
                "ERROR",
                "INVALID_FRONTIER_LINEAGE",
                parent / "stage_result.json",
                "retained_original_layer_ids is missing or invalid.",
            )
            return None
        return [int(value) for value in ids]

    def stage_is_resolved(self, stage: int) -> bool:
        return (
            stage in self.frontier_by_stage
            or stage + 1 in self.evidence_by_stage
            or self.catalog is not None
        )

    def audit_stage(self, stage: int, path: Path) -> None:
        expected_ids = self.expected_ids_for_stage(stage)
        entries = [item for item in path.iterdir() if item.name not in IGNORED_NAMES]
        raw: dict[int, Path] = {}
        recovered: dict[int, Path] = {}
        unexpected: list[Path] = []
        for item in entries:
            if not item.is_dir():
                unexpected.append(item)
                continue
            raw_match = RAW_RE.fullmatch(item.name)
            recovered_match = RECOVERED_RE.fullmatch(item.name)
            if raw_match and int(raw_match.group(1)) == stage:
                original_id = int(raw_match.group(2))
                if original_id in raw:
                    self.add("ERROR", "DUPLICATE_RAW_CANDIDATE", item, f"Duplicate raw layer {original_id}.")
                raw[original_id] = item
            elif recovered_match and int(recovered_match.group(1)) == stage:
                original_id = int(recovered_match.group(2))
                if original_id in recovered:
                    self.add(
                        "ERROR", "DUPLICATE_RECOVERED_CANDIDATE", item, f"Duplicate recovered layer {original_id}."
                    )
                recovered[original_id] = item
            else:
                unexpected.append(item)

        for item in unexpected:
            self.add(
                "WARNING",
                "UNEXPECTED_STAGE_ENTRY",
                item,
                "Entry does not match a raw or recovered Phase 2B candidate name.",
                safe_to_delete=True,
            )

        resolved = self.stage_is_resolved(stage)
        raw_complete = expected_ids is not None and set(raw) == set(expected_ids)
        stage_in_progress = not resolved and stage == max(self.evidence_by_stage, default=stage)

        if expected_ids is None:
            self.add(
                "ERROR",
                "UNKNOWN_STAGE_PARENT",
                path,
                f"Cannot determine the expected original layer IDs for stage {stage}.",
            )
        else:
            missing = sorted(set(expected_ids) - set(raw))
            extra = sorted(set(raw) - set(expected_ids))
            if missing:
                severity = "INFO" if stage_in_progress else "ERROR"
                self.add(
                    severity,
                    "MISSING_RAW_CANDIDATES",
                    path,
                    f"Expected {len(expected_ids)} raw candidates; missing original layers {missing}.",
                )
            if extra:
                self.add(
                    "ERROR",
                    "STALE_RAW_CANDIDATES",
                    path,
                    f"Raw candidates for impossible/stale original layers: {extra}.",
                    safe_to_delete=True,
                )

        for original_id, candidate in sorted(raw.items()):
            complete = self.candidate_complete(candidate, self.report.probe_case_count, recovered=False)
            if not complete and stage_in_progress:
                # A currently running candidate can be partial without being stale.
                self.add(
                    "INFO",
                    "CURRENT_CANDIDATE_INCOMPLETE",
                    candidate,
                    "Candidate appears incomplete; this can be normal while the process is active.",
                )

        recovered_without_raw = sorted(set(recovered) - set(raw))
        if recovered_without_raw:
            self.add(
                "ERROR",
                "RECOVERY_WITHOUT_RAW",
                path,
                f"Recovered candidates have no matching raw candidates: {recovered_without_raw}.",
            )

        expected_recovered = (
            min(self.args.recovery_top_k, len(expected_ids))
            if expected_ids is not None and self.args.recovery_top_k > 0
            else 0
        )
        if len(recovered) > expected_recovered:
            self.add(
                "ERROR",
                "TOO_MANY_RECOVERY_CANDIDATES",
                path,
                f"Expected at most {expected_recovered} recovered candidates, found {len(recovered)}.",
            )
        if resolved and raw_complete and len(recovered) != expected_recovered:
            self.add(
                "ERROR",
                "RECOVERY_COUNT_MISMATCH",
                path,
                f"Resolved stage should have {expected_recovered} recovered candidates, found {len(recovered)}.",
            )
        elif raw_complete and not resolved and len(recovered) < expected_recovered:
            self.add(
                "INFO",
                "RECOVERY_PENDING",
                path,
                f"Raw scan is complete; recovery is pending ({len(recovered)}/{expected_recovered}).",
            )

        for _, candidate in sorted(recovered.items()):
            self.candidate_complete(candidate, self.report.probe_case_count, recovered=True)

        expected_raw_count = len(expected_ids) if expected_ids is not None else None
        expected_text = (
            f"{expected_raw_count} raw + {expected_recovered} recovered candidates; "
            "each raw candidate has 10 evidence files, each recovered candidate adds recovery artifacts"
            if expected_raw_count is not None
            else "unknown until parent frontier lineage is valid"
        )
        if resolved and raw_complete and len(recovered) == expected_recovered:
            stage_status = "COMPLETE"
        elif stage_in_progress:
            stage_status = "IN_PROGRESS"
        else:
            stage_status = "INCOMPLETE"
        self.record_count(path, expected_text, stage_status)
        self.report.stage_summaries.append(
            {
                "stage": stage,
                "path": self.relative(path),
                "expected_original_layer_ids": expected_ids,
                "expected_raw_count": expected_raw_count,
                "actual_raw_count": len(raw),
                "expected_recovered_count": expected_recovered,
                "actual_recovered_count": len(recovered),
                "resolved": resolved,
                "status": stage_status,
            }
        )

    def audit_frontier(self) -> None:
        if not self.frontier_by_stage:
            return
        stages = sorted(self.frontier_by_stage)
        expected_sequence = list(range(1, max(stages) + 1))
        if stages != expected_sequence:
            self.add(
                "ERROR",
                "FRONTIER_STAGE_GAP",
                self.artifact_root / "frontier",
                f"Frontier stages are {stages}; expected consecutive {expected_sequence}.",
            )
        if max(stages) > self.args.max_drops:
            self.add(
                "ERROR",
                "FRONTIER_EXCEEDS_MAX_DROPS",
                self.artifact_root / "frontier",
                f"Found stage {max(stages)} but --max-drops is {self.args.max_drops}.",
            )
        for stage, path in sorted(self.frontier_by_stage.items()):
            payload = self.read_json(path / "stage_result.json")
            model = path / "model"
            self.audit_checkpoint(
                model,
                manifest_name="phase_2b_stage_manifest.json",
                context=f"Frontier stage {stage}",
            )
            if payload and self.report.base_layer_count is not None:
                expected_remaining = self.report.base_layer_count - stage
                actual_remaining = payload.get("remaining_layer_count")
                if actual_remaining != expected_remaining:
                    self.add(
                        "ERROR",
                        "FRONTIER_LAYER_COUNT_MISMATCH",
                        path / "stage_result.json",
                        f"Expected {expected_remaining} remaining layers, found {actual_remaining}.",
                    )
            self.record_count(path, "stage_result.json + complete accepted model checkpoint", "CHECKED")

    def audit_work_models(self) -> None:
        work_root = self.artifact_root / "work_models"
        if not work_root.exists():
            self.record_count(work_root, "0 after a completed stage/run", "ABSENT")
            return
        work_dirs = [item for item in work_root.iterdir() if item.is_dir()]
        completed = self.catalog is not None
        max_frontier = max(self.frontier_by_stage, default=0)
        for item in work_dirs:
            match = RAW_RE.fullmatch(item.name)
            if not match:
                self.add(
                    "WARNING",
                    "UNEXPECTED_WORK_MODEL",
                    item,
                    "Work-model name does not match a raw Phase 2B candidate.",
                    safe_to_delete=completed,
                )
                continue
            stage = int(match.group(1))
            original_id = int(match.group(2))
            evidence = self.artifact_root / "evidence" / f"stage_{stage:02d}" / item.name
            checkpoint_complete = self.audit_checkpoint(
                item,
                manifest_name="depth_reduction_manifest.json",
                context=f"Temporary raw candidate stage {stage}, original layer {original_id}",
                incomplete_safe_to_delete=True,
            )
            if not evidence.exists():
                self.add(
                    "WARNING",
                    "ORPHAN_WORK_MODEL",
                    item,
                    "Temporary checkpoint has no matching evidence directory.",
                    safe_to_delete=True,
                )
            if completed:
                self.add(
                    "WARNING",
                    "STALE_WORK_MODEL_AFTER_COMPLETION",
                    item,
                    "Final model_catalog.json exists, so this temporary checkpoint is stale.",
                    safe_to_delete=True,
                )
            elif stage <= max_frontier:
                self.add(
                    "WARNING",
                    "STALE_WORK_MODEL_AFTER_ACCEPTED_STAGE",
                    item,
                    f"Frontier stage {stage} is already accepted; its temporary checkpoint should have been cleaned.",
                    safe_to_delete=True,
                )
            elif not checkpoint_complete:
                self.add(
                    "INFO",
                    "PARTIAL_WORK_MODEL",
                    item,
                    "Interrupted/failed save detected. It is safe to delete; resume will recreate it.",
                    safe_to_delete=True,
                )
        expected = "temporary raw checkpoints only for the currently running unresolved stage; 0 after completion"
        status = "STALE" if completed and work_dirs else "PRESENT"
        self.record_count(work_root, expected, status)

    def audit_final_outputs(self) -> None:
        catalog_path = self.artifact_root / "model_catalog.json"
        registry_path = self.artifact_root / "candidate_registry.json"
        self.catalog = self.read_json(catalog_path, required=False)
        self.registry = self.read_json(registry_path, required=False)

        if self.catalog is not None:
            if self.registry is None:
                self.add("ERROR", "MISSING_REGISTRY", registry_path, "Completed catalog exists without candidate registry.")
            profiles = self.catalog.get("profiles")
            if not isinstance(profiles, dict):
                self.add("ERROR", "INVALID_CATALOG_PROFILES", catalog_path, "profiles must be a JSON object.")
            else:
                for name in ("full_quality", "balanced", "min_memory"):
                    if name not in profiles:
                        self.add("ERROR", "MISSING_PROFILE", catalog_path, f"Catalog is missing profile {name!r}.")
                        continue
                    profile = profiles[name]
                    if not isinstance(profile, dict):
                        self.add("ERROR", "INVALID_PROFILE", catalog_path, f"Profile {name!r} is not an object.")
                        continue
                    inference_model = profile.get("inference_model")
                    evaluation_summary = profile.get("evaluation_summary")
                    if isinstance(inference_model, str) and not self.resolve_recorded_path(inference_model).exists():
                        self.add(
                            "ERROR",
                            "BROKEN_PROFILE_MODEL_PATH",
                            catalog_path,
                            f"Profile {name!r} points to missing model: {inference_model}",
                        )
                    if isinstance(evaluation_summary, str) and not self.resolve_recorded_path(evaluation_summary).exists():
                        self.add(
                            "ERROR",
                            "BROKEN_PROFILE_EVALUATION_PATH",
                            catalog_path,
                            f"Profile {name!r} points to missing evaluation: {evaluation_summary}",
                        )
            for report_path in (
                self.repo_root / "reports" / "PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md",
                self.repo_root / "reports" / "PHASE_2B_STOP_GATE.md",
            ):
                if not report_path.exists():
                    self.add("ERROR", "MISSING_FINAL_REPORT", report_path, "Completed catalog exists without final report.")
            self.report.run_state = "COMPLETE"
        elif self.artifact_root.exists():
            self.report.run_state = "IN_PROGRESS"
        else:
            self.report.run_state = "NOT_STARTED"

        if self.registry is not None:
            candidates = self.registry.get("candidates")
            if not isinstance(candidates, list):
                self.add("ERROR", "INVALID_REGISTRY", registry_path, "candidates must be a list.")
            else:
                evidence_candidates = sum(
                    summary["actual_raw_count"] + summary["actual_recovered_count"]
                    for summary in self.report.stage_summaries
                )
                if len(candidates) != evidence_candidates:
                    self.add(
                        "ERROR",
                        "REGISTRY_EVIDENCE_COUNT_MISMATCH",
                        registry_path,
                        f"Registry has {len(candidates)} candidates; evidence tree has {evidence_candidates} candidate directories.",
                    )

    def resolve_recorded_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def audit_top_level(self) -> None:
        if not self.artifact_root.exists():
            return
        known = {
            "reference",
            "evidence",
            "frontier",
            "work_models",
            "profile_validation",
            "candidate_registry.json",
            "model_catalog.json",
        }
        if self.args.json_out:
            try:
                known.add(str(self.args.json_out.resolve().relative_to(self.artifact_root)))
            except ValueError:
                pass
        for item in self.artifact_root.iterdir():
            if item.name in IGNORED_NAMES or item.name in known:
                continue
            self.add(
                "WARNING",
                "UNEXPECTED_TOP_LEVEL_ENTRY",
                item,
                "Entry is not part of the current Phase 2B artifact contract and may be from an old run.",
            )

    def run(self) -> AuditReport:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Repository root does not exist: {self.repo_root}")
        if not self.artifact_root.exists():
            self.report.run_state = "NOT_STARTED"
            self.add("INFO", "NOT_STARTED", self.artifact_root, "Phase 2B artifact root does not exist yet.")
            return self.report

        # Load final state first because it affects whether temporary files are stale.
        self.catalog = self.read_json(self.artifact_root / "model_catalog.json", required=False)
        self.registry = self.read_json(self.artifact_root / "candidate_registry.json", required=False)
        self.report.base_layer_count = self.infer_base_layers()
        self.report.probe_case_count = self.infer_probe_count()
        if self.report.base_layer_count is None:
            self.add(
                "ERROR",
                "UNKNOWN_BASE_LAYER_COUNT",
                self.artifact_root,
                "Could not infer num_hidden_layers; pass --base-layers explicitly.",
            )
        if self.report.probe_case_count is None:
            self.add(
                "WARNING",
                "UNKNOWN_PROBE_COUNT",
                self.repo_root / self.args.probe_file,
                "Could not infer probe row count; pass --probe-count explicitly.",
            )

        self.discover_stage_dirs()
        self.audit_reference()
        self.audit_frontier()

        for stage, path in sorted(self.evidence_by_stage.items()):
            self.audit_stage(stage, path)

        self.audit_work_models()
        self.audit_final_outputs()
        self.audit_top_level()
        self.record_count(
            self.artifact_root,
            "reference + sequential evidence/frontier + optional current work_models + final catalog/registry",
            self.report.run_state,
        )
        return self.report


def human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def print_report(report: AuditReport, *, verbose: bool) -> None:
    print("\nPHASE 2B ARTIFACT AUDIT")
    print("=" * 78)
    print(f"Repository:      {report.repository_root}")
    print(f"Artifact root:   {report.artifact_root}")
    print(f"Run state:       {report.run_state}")
    print(f"Base layers:     {report.base_layer_count if report.base_layer_count is not None else 'unknown'}")
    print(f"Probe cases:     {report.probe_case_count if report.probe_case_count is not None else 'unknown'}")
    print(f"Errors/Warnings: {report.error_count}/{report.warning_count}")

    if report.stage_summaries:
        print("\nSTAGE COUNTS")
        print("-" * 78)
        print(f"{'Stage':<7}{'Raw':<15}{'Recovered':<15}{'Resolved':<11}{'Status'}")
        for row in report.stage_summaries:
            raw = f"{row['actual_raw_count']}/{row['expected_raw_count']}"
            recovered = f"{row['actual_recovered_count']}/{row['expected_recovered_count']}"
            print(
                f"{row['stage']:<7}{raw:<15}{recovered:<15}"
                f"{str(row['resolved']):<11}{row['status']}"
            )

    print("\nFOLDER FILE COUNTS")
    print("-" * 78)
    for row in report.folder_counts:
        print(
            f"[{row.status:<11}] {row.path}\n"
            f"  actual: {row.actual_files_recursive} files, {row.actual_dirs_recursive} dirs, "
            f"{human_bytes(row.bytes_recursive)}\n"
            f"  expected: {row.expected}"
        )

    significant = [item for item in report.findings if item.severity != "INFO" or verbose]
    print("\nFINDINGS")
    print("-" * 78)
    if not significant:
        print("No errors or warnings found.")
    else:
        order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        for item in sorted(significant, key=lambda x: (order[x.severity], x.path, x.code)):
            suffix = " [SAFE CLEANUP CANDIDATE]" if item.safe_to_delete else ""
            print(f"{item.severity:<7} {item.code}: {item.path}{suffix}")
            print(f"        {item.message}")

    print("\nSAFE CLEANUP CANDIDATES")
    print("-" * 78)
    if report.safe_cleanup_candidates:
        print("The auditor does not delete anything. Review these paths manually:")
        for path in sorted(report.safe_cleanup_candidates):
            print(f"  {path}")
    else:
        print("None detected.")

    print("\nFINAL STATUS")
    print("-" * 78)
    if report.error_count:
        print("NEEDS_ATTENTION: structural errors or stale/inconsistent artifacts were found.")
    elif report.warning_count:
        print("CLEAN_WITH_WARNINGS: no hard errors, but review the warnings.")
    elif report.run_state == "COMPLETE":
        print("CLEAN_COMPLETE: the completed Phase 2B artifact tree is internally consistent.")
    elif report.run_state == "IN_PROGRESS":
        print("CLEAN_IN_PROGRESS: the partial artifact tree is consistent with an interrupted/running job.")
    else:
        print("NOT_STARTED: no Phase 2B artifact tree exists yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of Phase 2B artifact counts, completeness, and stale files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Default: current working directory.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/phase_2_progressive_reduction"),
        help="Phase 2B artifact path relative to the repository root.",
    )
    parser.add_argument(
        "--probe-file",
        type=Path,
        default=Path("data/phase_2_structural_reduction/probe_50.jsonl"),
        help="Probe JSONL path relative to repository root.",
    )
    parser.add_argument("--probe-count", type=int, default=None, help="Override expected probe rows.")
    parser.add_argument("--base-layers", type=int, default=None, help="Override base decoder layer count.")
    parser.add_argument("--max-drops", type=int, default=6)
    parser.add_argument("--recovery-top-k", type=int, default=3)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a machine-readable audit report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print informational findings for a cleanly interrupted run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_drops < 1 or args.recovery_top_k < 0:
        print("Invalid arguments: --max-drops must be >= 1 and --recovery-top-k >= 0.", file=sys.stderr)
        return 2
    try:
        report = Auditor(args).run()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print_report(report, verbose=args.verbose)
    if args.json_out is not None:
        output = args.json_out
        if not output.is_absolute():
            output = args.repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {output}")
    return 1 if report.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
