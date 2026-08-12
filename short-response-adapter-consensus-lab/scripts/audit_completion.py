from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import LAB_ROOT  # noqa: F401

from consensus_lab.io import read_json, read_jsonl, sha256_file, write_json_atomic
from consensus_lab.paths import LabPaths


REQUIRED_FILES = (
    "README.md",
    "RESEARCH_QUESTIONS.md",
    "EXPERIMENT_PROTOCOL.md",
    "reports/CANDIDATE_DISCOVERY.md",
    "reports/candidate_inventory.json",
    "reports/BASELINE_REPRODUCTION.md",
    "reports/LIVE_RESULTS.md",
    "reports/FINAL_REPORT.md",
    "reports/plots/quality_vs_latency.png",
    "reports/plots/quality_vs_vram.png",
    "reports/plots/quality_vs_adapter_storage.png",
    "reports/plots/quality_vs_forward_passes.png",
    "reports/plots/quality_vs_committee_size.png",
    "artifacts/frozen_configs/frozen_final_config.json",
    "results/sealed/v22_frozen_finalists.jsonl",
    "results/summaries/sealed_statistics.json",
)


def _unique_result_keys(rows: list[dict[str, Any]]) -> int:
    return len(
        {
            (
                str(row.get("run_name")),
                str(row.get("dataset")),
                str(row.get("committee")),
                str(row.get("method")),
                str(row.get("example_id")),
            )
            for row in rows
        }
    )


def _tracked_status(workspace: Path, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), "status", "--short", "--untracked-files=no", "--", relative_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    paths = LabPaths.discover(LAB_ROOT)
    before = read_json(LAB_ROOT / "artifacts/manifests/official_sources_before.json")
    after = read_json(LAB_ROOT / "artifacts/manifests/official_sources_after.json")
    frozen = read_json(LAB_ROOT / "artifacts/frozen_configs/frozen_final_config.json")
    sealed_manifest = read_json(LAB_ROOT / "results/sealed/sealed_run_manifest.json")
    baseline = read_json(LAB_ROOT / "results/summaries/baseline_reproduction.json")
    baseline_by_name = {str(candidate["name"]): candidate for candidate in baseline["candidates"]}
    frozen_config = yaml.safe_load(Path(frozen["frozen_config"]).read_text(encoding="utf-8"))

    result_files = sorted((LAB_ROOT / "results/runs").glob("*.jsonl")) + [
        LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl"
    ]
    result_audits: dict[str, dict[str, Any]] = {}
    for result_file in result_files:
        rows = read_jsonl(result_file)
        unique = _unique_result_keys(rows)
        result_audits[str(result_file.relative_to(LAB_ROOT))] = {
            "rows": len(rows),
            "unique_keys": unique,
            "duplicate_keys": len(rows) - unique,
            "sha256": sha256_file(result_file),
        }

    sealed_path = LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl"
    sealed_rows = result_audits[str(sealed_path.relative_to(LAB_ROOT))]["rows"]
    expected_sealed_rows = (
        int(frozen["sealed_limit"])
        * len(frozen_config["committees"])
        * len(frozen_config["methods"])
    )
    runnable_configs = [
        path
        for path in sorted((LAB_ROOT / "configs").glob("*.yaml"))
        if path.name not in {"baseline.yaml", "mixed_peft.yaml", "paths.example.yaml"}
    ]
    cache_modes = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8")).get("cache_mode")
        for path in runnable_configs
    }
    protected_status = {
        name: _tracked_status(paths.workspace_root, name)
        for name in (
            "short-response-model-phase-0-1-2",
            "v22-model-soup-micro-experiment",
            "v22-specialist-mean-dose-sweep",
        )
    }
    missing_required = [value for value in REQUIRED_FILES if not (LAB_ROOT / value).is_file()]
    checks = {
        "required_files_present": not missing_required,
        "source_manifest_hashes_match": before["manifest_sha256"] == after["manifest_sha256"],
        "source_manifest_entry_counts_match": before["entry_count"] == after["entry_count"],
        "protected_tracked_files_clean": all(not value for value in protected_status.values()),
        "baseline_anchor_exact": baseline_by_name["frozen_anchor"]["exact_output_match_rate"] == 1.0,
        "d_greedy_exact": baseline_by_name["greedy_soup"]["exact_output_match_rate"] == 1.0,
        "frozen_before_sealed": frozen["status"] == "FROZEN_BEFORE_SEALED",
        "sealed_config_hash_matches": sealed_manifest["config_sha256"] == frozen["parsed_config_sha256"],
        "sealed_run_complete": sealed_manifest["status"] == "SEALED_RUN_COMPLETE",
        "sealed_row_count_complete": sealed_rows == expected_sealed_rows == int(sealed_manifest["rows"]),
        "all_result_keys_unique": all(value["duplicate_keys"] == 0 for value in result_audits.values()),
        "all_research_configs_use_reference_decoder": all(value == "reference" for value in cache_modes.values()),
        "no_stale_result_locks": not list((LAB_ROOT / "results").rglob("*.lock")),
    }
    packet = {
        "schema_version": "1.0",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_required_files": missing_required,
        "source_manifest_sha256": after["manifest_sha256"],
        "official_git_commit": after["git"]["commit"],
        "protected_tracked_status": protected_status,
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "sealed_dataset_sha256": frozen["sealed_datasets"]["v22_sealed"]["sha256"],
        "expected_sealed_rows": expected_sealed_rows,
        "result_files": result_audits,
        "research_config_cache_modes": cache_modes,
    }
    output = LAB_ROOT / "reports/completion_audit.json"
    write_json_atomic(output, packet)
    print(json.dumps({"status": packet["status"], "output": str(output), "checks": checks}, indent=2))
    if packet["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
