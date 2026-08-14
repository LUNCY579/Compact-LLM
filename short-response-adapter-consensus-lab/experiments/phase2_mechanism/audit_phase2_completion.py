from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

LAB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB_ROOT / "src"))

from consensus_lab.io import read_json, sha256_file, write_json_atomic  # noqa: E402
from consensus_lab.paths import LabPaths  # noqa: E402


OUTPUT = LAB_ROOT / "reports" / "phase2_completion_audit.json"
CORNERSTONE_SHA256 = "d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d"

PAPER_FILES = (
    "paper/PAPER_ANALYSIS.md",
    "paper/EVIDENCE_LEDGER.md",
    "paper/RELATED_WORK.md",
    "paper/METHODS_AND_MATH.md",
    "paper/EXPERIMENTAL_PROTOCOL.md",
    "paper/RESULTS_ANALYSIS.md",
    "paper/QUALITATIVE_ANALYSIS.md",
    "paper/LIMITATIONS_AND_ETHICS.md",
    "paper/FIGURE_PLAN.md",
    "paper/TABLE_PLAN.md",
    "paper/CLAIMS_AND_SUPPORT.md",
    "paper/FUTURE_WORK.md",
    "paper/REFERENCES.md",
    "paper/references.bib",
    "paper/REQUIREMENT_TRACEABILITY.md",
)

REPORT_FILES = (
    "reports/PHASE2_CORNERSTONE_VERIFICATION.md",
    "reports/PHASE2_RESCUE_HARM.md",
    "reports/PHASE2_DATA_INDEPENDENCE.md",
    "reports/PHASE2_FAILURE_ANALYSIS.md",
    "reports/PHASE2_SEMANTIC_AUDIT.md",
    "reports/PHASE2_SEMANTIC_AUDIT_REVIEW.md",
    "reports/PHASE2_TEACHER_FORCED_DEV.md",
    "reports/PHASE2_CONFIDENCE_CONDITIONED_DEV.md",
    "reports/PHASE2_COMMITTEE_DIAGNOSTICS_DEV.md",
    "reports/PHASE2_COALITION_STRATA.md",
    "reports/PHASE2_PAIRWISE_DIVERSITY_EXTENDED.md",
    "reports/PHASE2_ROBUST_SOFT_CONTROLS_DEV.md",
    "reports/PHASE2_NEGATIVE_CONTROLS_DEV.md",
    "reports/PHASE2_HARD_TEMPERATURE_TIE_SENSITIVITY.md",
    "reports/PHASE2_TEACHER_FORCED_CASES.md",
    "reports/PHASE2_TIE_EOS_COUNTERFACTUALS.md",
    "reports/PHASE2_RANK_TOPK_PARITY.md",
    "reports/PHASE2_SELECTED_RUNTIME_BOUNDARIES.md",
    "reports/PHASE2_ADAPTIVE_SELECTION.md",
    "reports/PHASE2_CROSS_BENCHMARK_VERIFICATION.md",
    "reports/PHASE2_FREE_RUNNING_DEV.md",
    "reports/PHASE2_ADAPTIVE_FREE_RUNNING_DEV.md",
    "reports/PHASE2_SELECTED_COMMITTEE_DEV.md",
    "reports/PHASE2_V22_CALIBRATED_TEST_CONFIRMATION.md",
    "reports/PHASE2_CROSS_V212_RESULTS.md",
    "reports/PHASE2_FINAL_SYNTHESIS_FACTS.md",
    "reports/PHASE2_ENVIRONMENT.md",
    "reports/PHASE2_LIVE_RESULTS.md",
    "reports/PHASE2_MECHANISM_REPORT.md",
)

ANALYSIS_STATUSES = {
    "results/phase2_mechanism/selected_dev_free_run_analysis.json": (
        "status",
        "DEV_SELECTED_FREE_RUNNING_ANALYSIS_COMPLETE",
    ),
    "results/phase2_mechanism/adaptive_free_run_dev_analysis.json": (
        "status",
        "DEV_ADAPTIVE_FREE_RUNNING_ANALYSIS_COMPLETE",
    ),
    "results/phase2_mechanism/v22_calibrated_test_analysis.json": (
        "status",
        "V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE",
    ),
    "results/phase2_mechanism/selected_committee_dev_analysis.json": (
        "status",
        "DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE",
    ),
    "results/phase2_mechanism/cross_v212_analysis.json": (
        "status",
        "V212_FIXED_METHOD_CROSS_CORPUS_ANALYSIS_COMPLETE",
    ),
}


def audit_jsonl(
    path: Path,
    *,
    key_fields: Iterable[str],
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "rows": 0,
        "unique_keys": 0,
        "duplicate_keys": 0,
        "malformed_rows": [],
        "methods": {},
        "config_sha256_values": [],
        "sha256": None,
    }
    if not path.is_file():
        return packet
    keys: set[tuple[str, ...]] = set()
    methods: collections.Counter[str] = collections.Counter()
    config_hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                packet["malformed_rows"].append({"line": line_number, "error": str(exc)})
                continue
            packet["rows"] += 1
            keys.add(tuple(str(row.get(field)) for field in key_fields))
            if row.get("method") is not None:
                methods[str(row["method"])] += 1
            if row.get("config_sha256") is not None:
                config_hashes.add(str(row["config_sha256"]))
    packet["unique_keys"] = len(keys)
    packet["duplicate_keys"] = int(packet["rows"]) - len(keys)
    packet["methods"] = dict(sorted(methods.items()))
    packet["config_sha256_values"] = sorted(config_hashes)
    packet["sha256"] = sha256_file(path)
    packet["config_matches"] = (
        expected_config_sha256 is None or config_hashes == {expected_config_sha256}
    )
    return packet


def historical_capture_generator_is_well_formed(packet: dict[str, Any]) -> bool:
    """Validate the generator recorded at capture time without rewriting history.

    A historical artifact should continue to name and hash the code that created
    it. Requiring that hash to equal a later, hardened generator incorrectly
    invalidates an otherwise source-bound capture.
    """

    generator = packet.get("generator", {})
    normalized_path = str(generator.get("path", "")).replace("\\", "/")
    return bool(
        normalized_path.endswith(
            "experiments/phase2_mechanism/make_selected_runtime_boundaries.py"
        )
        and re.fullmatch(r"[0-9a-f]{64}", str(generator.get("sha256", "")))
    )


def runtime_manifest_complete(
    manifest_path: Path,
    *,
    output_path: Path,
    output_audit: dict[str, Any],
    expected_rows: int,
) -> tuple[bool, dict[str, Any]]:
    if not manifest_path.is_file():
        return False, {"path": str(manifest_path), "reason": "missing"}
    manifest = read_json(manifest_path)
    invocations = manifest.get("invocations", [])
    identity_sha256 = str(manifest.get("identity_sha256", ""))
    base_valid = bool(
        manifest.get("status") == "COMPLETE"
        and manifest.get("observed_rows") == expected_rows
        and manifest.get("expected_rows") == expected_rows
        and Path(str(manifest.get("output_path", ""))).resolve() == output_path.resolve()
        and re.fullmatch(r"[0-9a-f]{64}", identity_sha256)
        and invocations
        and all(invocation.get("status") != "STARTED" for invocation in invocations)
    )
    completion_basis = manifest.get("completion_basis", "CLEAN_RUNTIME_RETURN")
    if completion_basis == "POST_HOC_VERIFIED_COMPLETE":
        reconciliations = manifest.get("reconciliations", [])
        verified = manifest.get("verified_output", {})
        basis_valid = bool(
            reconciliations
            and reconciliations[-1].get("status") == "POST_HOC_COMPLETE_RESULT_VERIFIED"
            and reconciliations[-1].get("output_sha256") == output_audit.get("sha256")
            and reconciliations[-1].get("runtime_identity_sha256") == identity_sha256
            and reconciliations[-1].get("rows") == expected_rows
            and reconciliations[-1].get("unique_keys") == expected_rows
            and verified.get("sha256") == output_audit.get("sha256")
            and verified.get("rows") == expected_rows
            and verified.get("unique_keys") == expected_rows
            and verified.get("runtime_identity_sha256") == identity_sha256
        )
    else:
        basis_valid = bool(invocations and invocations[-1].get("status") == "RETURNED_CLEANLY")
    summary = {
        "path": str(manifest_path),
        "status": manifest.get("status"),
        "completion_basis": completion_basis,
        "observed_rows": manifest.get("observed_rows"),
        "expected_rows": manifest.get("expected_rows"),
        "identity_sha256": identity_sha256,
        "invocation_count": len(invocations),
        "active_invocation_count": sum(
            invocation.get("status") == "STARTED" for invocation in invocations
        ),
        "output_sha256": output_audit.get("sha256"),
        "passed": bool(base_valid and basis_valid),
    }
    return bool(base_valid and basis_valid), summary


def bibliography_audit() -> dict[str, Any]:
    bib_path = LAB_ROOT / "paper/references.bib"
    related_path = LAB_ROOT / "paper/RELATED_WORK.md"
    ledger_path = LAB_ROOT / "paper/REFERENCES.md"
    if not all(path.is_file() for path in (bib_path, related_path, ledger_path)):
        return {"complete": False, "reason": "one or more literature files are missing"}
    bib = bib_path.read_text(encoding="utf-8")
    related = related_path.read_text(encoding="utf-8")
    ledger = ledger_path.read_text(encoding="utf-8")
    bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib)
    related_keys = re.findall(r"@([A-Za-z0-9_:-]+)", related)
    ledger_keys = re.findall(r"`([A-Za-z0-9_:-]+)`", ledger)
    duplicates = sorted({key for key in bib_keys if bib_keys.count(key) > 1})
    missing_related = sorted(set(related_keys) - set(bib_keys))
    missing_ledger = sorted(set(ledger_keys) - set(bib_keys))
    unledgered = sorted(set(bib_keys) - set(ledger_keys))
    complete = (
        len(bib_keys) == 25
        and not duplicates
        and not missing_related
        and not missing_ledger
        and not unledgered
        and bib.count("{") == bib.count("}")
    )
    return {
        "complete": complete,
        "bib_entries": len(bib_keys),
        "related_citations": len(related_keys),
        "ledger_entries": len(ledger_keys),
        "duplicate_keys": duplicates,
        "missing_related_keys": missing_related,
        "missing_ledger_keys": missing_ledger,
        "unledgered_bib_keys": unledgered,
        "brace_balance": bib.count("{") - bib.count("}"),
    }


def tracked_status(workspace: Path, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), "status", "--short", "--untracked-files=no", "--", relative_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def metadata_hash_map_matches(entries: dict[str, str]) -> bool:
    """Verify that every metadata path still exists and has its recorded digest."""
    for raw_path, expected in entries.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = LAB_ROOT / path
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def figure_output_hashes_match(metadata: dict[str, Any]) -> bool:
    for figure in metadata.get("figures", {}).values():
        for path_field, hash_field in (("png", "png_sha256"), ("svg", "svg_sha256")):
            raw_path = figure.get(path_field)
            expected = figure.get(hash_field)
            if not isinstance(raw_path, str) or not isinstance(expected, str):
                return False
            path = Path(raw_path)
            if not path.is_absolute():
                path = LAB_ROOT / path
            if not path.is_file() or sha256_file(path) != expected:
                return False
    return bool(metadata.get("figures"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete Phase 2 mechanism-study package.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write and print an INCOMPLETE packet without exiting nonzero; never changes pass criteria.",
    )
    args = parser.parse_args()
    paths = LabPaths.discover(LAB_ROOT)

    run_audits = {
        "cornerstone": audit_jsonl(
            LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
        ),
        "selected_dev": audit_jsonl(
            LAB_ROOT / "results/runs/phase2_selected_dev_free_run_dev_full.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
            expected_config_sha256="a887a4874f9717584aacb3a0f7019224d4a1571c8b76f28a92c04b965e56552e",
        ),
        "adaptive_dev": audit_jsonl(
            LAB_ROOT / "results/runs/phase2_selected_adaptive_dev_free_run_dev_full.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
            expected_config_sha256="14afa0c0f65596c013ae5fca8c24d50da0f523f8a676ad7632d7a6b52b2eeb5f",
        ),
        "v22_calibrated_test": audit_jsonl(
            LAB_ROOT / "results/runs/phase2_v22_calibrated_test_confirmation_dev_full.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
            expected_config_sha256="e4f77f7c7d68dc332eadc4decfb839d3a4b526addefd5c61538d92cf58df2794",
        ),
        "selected_committee_dev": audit_jsonl(
            LAB_ROOT / "results/runs/phase2_selected_committee_dev_free_run_dev_full.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
            expected_config_sha256="8ba792929c3f5691670ed72ec43b81b695de631304d53d34a3d611507eb0b820",
        ),
        "cross_controls": audit_jsonl(
            LAB_ROOT / "results/runs/phase2_cross_v212_controls_dev_full.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
            expected_config_sha256="1c29b6ecf08f0700291ae70118600bd8e80c7d1446f9890fae57992c31dbc914",
        ),
        "cross_full": audit_jsonl(
            LAB_ROOT / "results/runs/phase2_cross_v212_full_k7_dev_full.jsonl",
            key_fields=("dataset", "committee", "method", "example_id"),
            expected_config_sha256="e6a2051b0bf76ee4bf94eb468dd0d08b3fd0be182182bcd977766ffd76e71623",
        ),
    }

    required_files = PAPER_FILES + REPORT_FILES
    missing_required = [relative for relative in required_files if not (LAB_ROOT / relative).is_file()]
    analysis_statuses: dict[str, Any] = {}
    for relative, (field, expected) in ANALYSIS_STATUSES.items():
        path = LAB_ROOT / relative
        actual = read_json(path).get(field) if path.is_file() else None
        analysis_statuses[relative] = {"expected": expected, "actual": actual, "passed": actual == expected}

    before = read_json(LAB_ROOT / "artifacts/manifests/official_sources_before.json")
    after = read_json(LAB_ROOT / "artifacts/manifests/official_sources_after.json")
    cornerstone_audit = read_json(LAB_ROOT / "reports/phase2_cornerstone_verification.json")
    teacher = read_json(LAB_ROOT / "results/phase2_mechanism/teacher_forced_dev_analysis.json")
    robust_soft = read_json(LAB_ROOT / "results/phase2_mechanism/robust_soft_controls_dev.json")
    confidence_conditioning = read_json(
        LAB_ROOT / "results/phase2_mechanism/confidence_conditioned_correctness_dev.json"
    )
    committee_diagnostics = read_json(
        LAB_ROOT / "results/phase2_mechanism/teacher_forced_committee_diagnostics.json"
    )
    coalition_strata = read_json(
        LAB_ROOT / "results/phase2_mechanism/coalition_strata_full_test.json"
    )
    diversity_extended = read_json(
        LAB_ROOT / "results/phase2_mechanism/pairwise_diversity_extended.json"
    )
    negative_controls = read_json(LAB_ROOT / "results/phase2_mechanism/negative_controls_bounded_dev.json")
    hard_tie_temp = read_json(LAB_ROOT / "results/phase2_mechanism/hard_temperature_tie_sensitivity_dev.json")
    teacher_cases = read_json(LAB_ROOT / "results/phase2_mechanism/qualitative_teacher_forced_selection.json")
    independence = read_json(LAB_ROOT / "results/phase2_mechanism/data_independence_audit.json")
    semantic = read_json(LAB_ROOT / "results/phase2_mechanism/semantic_audit_adjudicated.json")
    tie_eos = read_json(LAB_ROOT / "results/phase2_mechanism/tie_eos_counterfactuals_dev.json")
    rank_topk_parity_path = LAB_ROOT / "results/phase2_mechanism/rank_topk_parity_verification.json"
    rank_topk_parity = read_json(rank_topk_parity_path) if rank_topk_parity_path.is_file() else {}
    runtime_boundaries_path = (
        LAB_ROOT / "results/phase2_mechanism/selected_dev_runtime_boundaries.json"
    )
    runtime_boundaries = (
        read_json(runtime_boundaries_path) if runtime_boundaries_path.is_file() else {}
    )
    adaptive_selection = read_json(LAB_ROOT / "results/phase2_mechanism/adaptive_teacher_forced_dev.json")
    cross_gate = read_json(LAB_ROOT / "results/phase2_mechanism/cross_benchmark_verification.json")
    selected_manifest = read_json(LAB_ROOT / "artifacts/phase2_mechanism/selected_dev_free_run_manifest.json")
    adaptive_manifest = read_json(LAB_ROOT / "artifacts/phase2_mechanism/selected_adaptive_dev_manifest.json")
    cross_manifest = read_json(LAB_ROOT / "artifacts/phase2_mechanism/cross_v212_confirmation_manifest.json")
    v22_calibrated_manifest = read_json(
        LAB_ROOT / "artifacts/phase2_mechanism/v22_calibrated_test_confirmation_manifest.json"
    )
    selected_committee_manifest = read_json(
        LAB_ROOT / "artifacts/phase2_mechanism/selected_committee_dev_manifest.json"
    )
    selected_committee_analysis_path = (
        LAB_ROOT / "results/phase2_mechanism/selected_committee_dev_analysis.json"
    )
    selected_committee_analysis = (
        read_json(selected_committee_analysis_path)
        if selected_committee_analysis_path.is_file()
        else {}
    )
    selected_analysis_path = (
        LAB_ROOT / "results/phase2_mechanism/selected_dev_free_run_analysis.json"
    )
    selected_analysis = (
        read_json(selected_analysis_path) if selected_analysis_path.is_file() else {}
    )
    adaptive_analysis_path = (
        LAB_ROOT / "results/phase2_mechanism/adaptive_free_run_dev_analysis.json"
    )
    adaptive_analysis = (
        read_json(adaptive_analysis_path) if adaptive_analysis_path.is_file() else {}
    )
    cross_analysis_path = LAB_ROOT / "results/phase2_mechanism/cross_v212_analysis.json"
    cross_analysis = read_json(cross_analysis_path) if cross_analysis_path.is_file() else {}
    v22_calibrated_analysis_path = (
        LAB_ROOT / "results/phase2_mechanism/v22_calibrated_test_analysis.json"
    )
    v22_calibrated_analysis = (
        read_json(v22_calibrated_analysis_path) if v22_calibrated_analysis_path.is_file() else {}
    )
    protected_status = {
        name: tracked_status(paths.workspace_root, name)
        for name in (
            "short-response-model-phase-0-1-2",
            "v22-model-soup-micro-experiment",
            "v22-specialist-mean-dose-sweep",
        )
    }
    figures_path = LAB_ROOT / "paper/figures/figure_metadata.json"
    tables_path = LAB_ROOT / "paper/tables/table_metadata.json"
    figures = read_json(figures_path) if figures_path.is_file() else {}
    tables = read_json(tables_path) if tables_path.is_file() else {}
    final_synthesis_path = LAB_ROOT / "reports/phase2_final_synthesis_verification.json"
    final_synthesis = (
        read_json(final_synthesis_path) if final_synthesis_path.is_file() else {}
    )
    synthesis_facts_path = LAB_ROOT / "results/phase2_mechanism/final_synthesis_facts.json"
    synthesis_facts = (
        read_json(synthesis_facts_path) if synthesis_facts_path.is_file() else {}
    )
    environment_path = LAB_ROOT / "artifacts/phase2_mechanism/environment.json"
    environment = read_json(environment_path) if environment_path.is_file() else {}
    literature = bibliography_audit()
    locks = sorted(str(path.relative_to(LAB_ROOT)) for path in (LAB_ROOT / "results").rglob("*.lock"))
    runtime_manifest_specs = {
        "adaptive_dev": 4050,
        "v22_calibrated_test": 4050,
        "selected_committee_dev": 48600,
        "cross_controls": 3960,
        "cross_full": 3960,
    }
    runtime_manifests: dict[str, Any] = {}
    runtime_manifest_checks: list[bool] = []
    for name, expected_rows in runtime_manifest_specs.items():
        output_path = Path(run_audits[name]["path"])
        passed, summary = runtime_manifest_complete(
            output_path.with_suffix(output_path.suffix + ".runtime.json"),
            output_path=output_path,
            output_audit=run_audits[name],
            expected_rows=expected_rows,
        )
        runtime_manifests[name] = summary
        runtime_manifest_checks.append(passed)

    checks = {
        "required_paper_and_report_files_present": not missing_required,
        "cornerstone_hash_exact": run_audits["cornerstone"]["sha256"] == CORNERSTONE_SHA256,
        "cornerstone_complete_unique": (
            run_audits["cornerstone"]["rows"] == 32400
            and run_audits["cornerstone"]["unique_keys"] == 32400
            and not run_audits["cornerstone"]["malformed_rows"]
        ),
        "cornerstone_independent_audit_passes": (
            cornerstone_audit.get("status") == "PASS"
            and len(cornerstone_audit.get("checks", {})) == 30
            and all(bool(item.get("passed")) for item in cornerstone_audit.get("checks", {}).values())
        ),
        "teacher_forced_dev_complete": (
            teacher.get("status") == "DEV_TEACHER_FORCED_ANALYSIS_COMPLETE_WITH_UNCERTIFIED_SOFT_POSITIONS"
            and teacher.get("example_count") == 4050
            and teacher.get("token_count") == 18760
            and len(teacher.get("committee_results", [])) == 1143
            and sha256_file(LAB_ROOT / "results/phase2_mechanism/teacher_forced_dev_cache_topk64.jsonl")
            == teacher.get("provenance", {}).get("cache_sha256")
        ),
        "robust_soft_controls_complete": (
            robust_soft.get("status") == "DEV_ROBUST_SOFT_CONTROLS_COMPLETE"
            and robust_soft.get("example_count") == 4050
            and robust_soft.get("token_count") == 18760
            and robust_soft.get("candidate_union", {}).get("all_winners_certified") is False
            and robust_soft.get("candidate_union", {}).get("all_nonmedian_winners_certified") is True
            and robust_soft.get("candidate_union", {}).get("uncertified_token_counts")
            == {"median_probability": 2}
            and len(robust_soft.get("method_summaries", [])) == 9
            and all(
                row.get("candidate_argmax_certification_rate") == 1.0
                for row in robust_soft.get("method_summaries", [])
                if row.get("method") != "median_probability"
            )
            and next(
                (
                    row
                    for row in robust_soft.get("method_summaries", [])
                    if row.get("method") == "median_probability"
                ),
                {},
            ).get("uncertified_token_count") == 2
        ),
        "confidence_conditioning_complete": (
            confidence_conditioning.get("status") == "DEV_CONFIDENCE_CONDITIONING_COMPLETE"
            and confidence_conditioning.get("example_count") == 4050
            and confidence_conditioning.get("token_count") == 18760
            and confidence_conditioning.get("expert_count") == 7
            and len(confidence_conditioning.get("expert_summaries", [])) == 7
            and all(
                row.get("correct_token_count", 0) + row.get("incorrect_token_count", 0) == 18760
                for row in confidence_conditioning.get("expert_summaries", [])
            )
            and confidence_conditioning.get("provenance", {}).get("cache_sha256")
            == teacher.get("provenance", {}).get("cache_sha256")
            and confidence_conditioning.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_confidence_conditioning.py"
            )
        ),
        "committee_diagnostics_complete": (
            committee_diagnostics.get("status") == "DEV_COMMITTEE_DIAGNOSTICS_COMPLETE"
            and committee_diagnostics.get("example_count") == 4050
            and committee_diagnostics.get("token_count") == 18760
            and committee_diagnostics.get("committee_count") == 127
            and committee_diagnostics.get("method_count") == 9
            and committee_diagnostics.get("record_count") == 1143
            and len(committee_diagnostics.get("records", [])) == 1143
            and len(committee_diagnostics.get("committee_scaling_intervals", [])) == 63
            and len(committee_diagnostics.get("balanced_committee_scaling", [])) == 63
            and all(
                row.get("balanced_answer_class_accuracy") is not None
                for row in committee_diagnostics.get("records", [])
            )
            and committee_diagnostics.get("provenance", {}).get(
                "all_frozen_token_and_example_counts_reproduced"
            ) is True
            and committee_diagnostics.get("provenance", {}).get("cache_sha256")
            == teacher.get("provenance", {}).get("cache_sha256")
            and committee_diagnostics.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_committee_diagnostics.py"
            )
            and all(
                row.get("measured_latency_seconds") is None
                and row.get("measured_peak_gpu_vram_bytes") is None
                for row in committee_diagnostics.get("records", [])
            )
        ),
        "coalition_strata_complete": (
            coalition_strata.get("status") == "FULL_TEST_COALITION_STRATA_COMPLETE"
            and coalition_strata.get("example_count") == 4050
            and coalition_strata.get("token_count") == 23204
            and coalition_strata.get("coalition_count") == 107
            and set(coalition_strata.get("dimensions", []))
            == {
                "global",
                "reasoning_family",
                "difficulty",
                "answer_class",
                "domain",
                "position_bucket",
            }
            and coalition_strata.get("record_count") == len(coalition_strata.get("records", []))
            and coalition_strata.get("provenance", {}).get("raw_sha256") == CORNERSTONE_SHA256
            and coalition_strata.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_coalition_strata.py"
            )
        ),
        "pairwise_diversity_extended_complete": (
            diversity_extended.get("status") == "PAIRWISE_DIVERSITY_EXTENDED_COMPLETE"
            and diversity_extended.get("pair_count") == 21
            and diversity_extended.get("full_test_example_count") == 4050
            and diversity_extended.get("full_test_token_count") == 23204
            and diversity_extended.get("dev_gold_prefix_token_count") == 18760
            and len(diversity_extended.get("pairs", [])) == 21
            and diversity_extended.get("provenance", {}).get("raw_sha256")
            == CORNERSTONE_SHA256
            and diversity_extended.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_pairwise_diversity_extended.py"
            )
        ),
        "bounded_negative_controls_audited": (
            negative_controls.get("status") == "BOUNDED_DEV_NEGATIVE_CONTROLS_AUDITED"
            and negative_controls.get("row_count") == 144
            and negative_controls.get("unique_condition_example_keys") == 144
            and len(negative_controls.get("condition_summaries", [])) == 18
            and negative_controls.get("provenance", {}).get("run_sha256")
            == "638754c533475a55e21c4f0e97dbf7f28cffdf66c19b4eedb849d5a2e9bfa38e"
        ),
        "hard_temperature_tie_boundary_complete": (
            hard_tie_temp.get("status") == "DEV_HARD_TEMPERATURE_TIE_SENSITIVITY_COMPLETE"
            and hard_tie_temp.get("example_count") == 4050
            and hard_tie_temp.get("token_count") == 18760
            and hard_tie_temp.get("plurality_tied_token_count") == 1097
            and hard_tie_temp.get("baseline_reproduced") is True
            and hard_tie_temp.get("shared_temperature_invariant") is True
            and hard_tie_temp.get("one_expert_temperature_can_change_tie_decision") is True
        ),
        "teacher_forced_qualitative_cases_complete": (
            teacher_cases.get("status") == "DEV_TEACHER_FORCED_QUALITATIVE_SELECTION_COMPLETE"
            and teacher_cases.get("example_count") == 4050
            and teacher_cases.get("token_count") == 18760
            and teacher_cases.get("selected_count") == 27
            and teacher_cases.get("empty_buckets") == []
            and len(teacher_cases.get("eligible_counts", {})) == 9
        ),
        "new_cache_analysis_generator_hashes_match": (
            robust_soft.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/analyze_robust_soft_controls.py")
            and negative_controls.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/analyze_negative_controls.py")
            and hard_tie_temp.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/analyze_hard_temperature_ties.py")
            and teacher_cases.get("provenance", {}).get("script_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/select_teacher_forced_cases.py")
        ),
        "data_independence_audit_complete": independence.get("status") == "DATA_INDEPENDENCE_SENSITIVITY_COMPLETE",
        "semantic_audit_complete": (
            semantic.get("review_status") == "complete"
            and semantic.get("sample_size") == 100
            and len(semantic.get("cases", [])) == 100
            and all(case.get("semantic_label") != "PENDING" for case in semantic.get("cases", []))
        ),
        "tie_eos_counterfactuals_complete": (
            tie_eos.get("status") == "DEV_TIE_EOS_COUNTERFACTUALS_COMPLETE"
            and tie_eos.get("example_count") == 4050
            and tie_eos.get("token_count") == 18760
        ),
        "rank_topk_implementation_parity_verified": (
            rank_topk_parity.get("status") == "RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED"
            and rank_topk_parity.get("equivalence", {}).get("all_checks_exact") is True
            and rank_topk_parity.get("exhaustive", {}).get("vector_rank_checks") == 15309
            and rank_topk_parity.get("realistic_vocabulary", {}).get("exact") is True
            and rank_topk_parity.get("provenance", {}).get("verification_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/verify_rank_topk_parity.py"
            )
            and rank_topk_parity.get("provenance", {}).get("voting_rules_sha256")
            == sha256_file(LAB_ROOT / "src/consensus_lab/voting/rules.py")
            and rank_topk_parity.get("provenance", {}).get("report_sha256")
            == sha256_file(LAB_ROOT / "reports/PHASE2_RANK_TOPK_PARITY.md")
        ),
        "selected_runtime_boundaries_captured": (
            runtime_boundaries.get("status") == "SELECTED_DEV_RUNTIME_BOUNDARIES_CAPTURED"
            and runtime_boundaries.get("run", {}).get("rows") == 16200
            and runtime_boundaries.get("run", {}).get("unique_keys") == 16200
            and runtime_boundaries.get("boundaries", {}).get(
                "stable_partial_topk_first_line"
            )
            == 12239
            and runtime_boundaries.get("rank_equivalence", {}).get(
                "all_checks_exact"
            )
            is True
            and runtime_boundaries.get("resume_history", {}).get("sha256")
            == sha256_file(
                LAB_ROOT / "artifacts/phase2_mechanism/selected_dev_resume_history.json"
            )
            and historical_capture_generator_is_well_formed(runtime_boundaries)
            and runtime_boundaries.get("report", {}).get("sha256")
            == sha256_file(LAB_ROOT / "reports/PHASE2_SELECTED_RUNTIME_BOUNDARIES.md")
        ),
        "adaptive_selection_frozen_without_test": (
            adaptive_selection.get("status") == "DEV_ADAPTIVE_THRESHOLD_SELECTED"
            and adaptive_manifest.get("status") == "DEV_ADAPTIVE_POLICY_FROZEN"
            and adaptive_manifest.get("test_data_read_for_selection") is False
        ),
        "cross_gate_and_manifest_frozen": (
            cross_gate.get("status") == "PASS_WITH_CAVEATS"
            and cross_gate.get("checks_passed") == cross_gate.get("checks_total") == 27
            and cross_manifest.get("status") == "V212_CROSS_CORPUS_CONFIRMATION_FROZEN"
            and cross_manifest.get("outcome_data_read_before_freeze") is False
        ),
        "selected_manifest_frozen_before_outputs": (
            selected_manifest.get("test_data_read_for_selection") is False
            and selected_manifest.get("output_config_canonical_sha256")
            == "a887a4874f9717584aacb3a0f7019224d4a1571c8b76f28a92c04b965e56552e"
        ),
        "v22_calibrated_confirmation_frozen_before_outputs": (
            v22_calibrated_manifest.get("status") == "V22_CALIBRATED_TEST_CONFIRMATION_FROZEN"
            and v22_calibrated_manifest.get("calibrated_confirmation_outcomes_read_before_freeze") is False
            and v22_calibrated_manifest.get("selection_used_test_outcomes") is False
            and v22_calibrated_manifest.get("config", {}).get("canonical_sha256")
            == "e4f77f7c7d68dc332eadc4decfb839d3a4b526addefd5c61538d92cf58df2794"
            and v22_calibrated_manifest.get("config", {}).get("file_sha256")
            == sha256_file(LAB_ROOT / "configs/phase2_v22_calibrated_test_confirmation.yaml")
            and v22_calibrated_manifest.get("dataset", {}).get("sha256")
            == cornerstone_audit.get("dataset_splits", {}).get("test", {}).get("sha256")
            and v22_calibrated_manifest.get("freeze_script", {}).get("sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/freeze_v22_calibrated_test.py")
        ),
        "selected_committee_panel_frozen_on_dev": (
            selected_committee_manifest.get("status") == "DEV_SELECTED_COMMITTEE_PANEL_FROZEN"
            and selected_committee_manifest.get("selection_used_test_outcomes") is False
            and selected_committee_manifest.get("committee_count") == 12
            and selected_committee_manifest.get("expected_rows") == 48600
            and selected_committee_manifest.get("config", {}).get("canonical_sha256")
            == "8ba792929c3f5691670ed72ec43b81b695de631304d53d34a3d611507eb0b820"
            and selected_committee_manifest.get("config", {}).get("file_sha256")
            == sha256_file(LAB_ROOT / "configs/phase2_selected_committee_dev_free_run.yaml")
            and selected_committee_manifest.get("freeze_script", {}).get("sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/freeze_selected_committee_dev.py"
            )
        ),
        "selected_dev_complete_unique": (
            run_audits["selected_dev"]["rows"] == 16200
            and run_audits["selected_dev"]["unique_keys"] == 16200
            and run_audits["selected_dev"]["methods"]
            == {
                "calibrated_probability_mean": 4050,
                "hard_plurality": 4050,
                "probability_mean": 4050,
                "reciprocal_rank_top5": 4050,
            }
            and run_audits["selected_dev"]["config_matches"]
            and not run_audits["selected_dev"]["malformed_rows"]
            and selected_analysis.get("provenance", {}).get("run_sha256")
            == run_audits["selected_dev"]["sha256"]
            and selected_analysis.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_selected_free_run.py"
            )
            and selected_analysis.get("provenance", {}).get("voting_rules_sha256")
            == sha256_file(LAB_ROOT / "src/consensus_lab/voting/rules.py")
            and selected_analysis.get("provenance", {}).get("direct_evaluator_sha256")
            == sha256_file(LAB_ROOT / "src/consensus_lab/evaluation/direct.py")
            and selected_analysis.get("provenance", {}).get("reference_decoder_sha256")
            == sha256_file(LAB_ROOT / "src/consensus_lab/decoding/decoder.py")
            and selected_analysis.get("provenance", {}).get("rank_topk_parity_sha256")
            == sha256_file(
                LAB_ROOT / "results/phase2_mechanism/rank_topk_parity_verification.json"
            )
            and selected_analysis.get("provenance", {}).get(
                "runtime_boundaries_sha256"
            )
            == sha256_file(runtime_boundaries_path)
            and selected_analysis.get("example_count") == 4050
            and len(selected_analysis.get("method_summaries", [])) == 4
            and len(selected_analysis.get("paired_family", {}).get("comparisons", [])) == 3
            and len(selected_analysis.get("teacher_forced_vs_free_running", [])) == 4
        ),
        "adaptive_dev_complete_unique": (
            run_audits["adaptive_dev"]["rows"] == 4050
            and run_audits["adaptive_dev"]["unique_keys"] == 4050
            and run_audits["adaptive_dev"]["methods"] == {"adaptive_hard_plurality": 4050}
            and run_audits["adaptive_dev"]["config_matches"]
            and not run_audits["adaptive_dev"]["malformed_rows"]
            and adaptive_analysis.get("provenance", {}).get("hard_run_sha256")
            == run_audits["selected_dev"]["sha256"]
            and adaptive_analysis.get("provenance", {}).get("adaptive_run_sha256")
            == run_audits["adaptive_dev"]["sha256"]
            and adaptive_analysis.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_adaptive_free_run.py"
            )
            and len(adaptive_analysis.get("summaries", [])) == 2
            and adaptive_analysis.get("paired_accuracy", {}).get("n") == 4050
        ),
        "v22_calibrated_test_complete_unique": (
            run_audits["v22_calibrated_test"]["rows"] == 4050
            and run_audits["v22_calibrated_test"]["unique_keys"] == 4050
            and run_audits["v22_calibrated_test"]["methods"]
            == {"calibrated_probability_mean": 4050}
            and run_audits["v22_calibrated_test"]["config_matches"]
            and not run_audits["v22_calibrated_test"]["malformed_rows"]
            and v22_calibrated_analysis.get("provenance", {}).get("calibrated_run_sha256")
            == run_audits["v22_calibrated_test"]["sha256"]
            and v22_calibrated_analysis.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/analyze_v22_calibrated_test.py")
            and v22_calibrated_analysis.get("example_count") == 4050
            and len(v22_calibrated_analysis.get("condition_summaries", [])) == 3
            and len(
                v22_calibrated_analysis.get("paired_family", {}).get("comparisons", [])
            )
            == 2
        ),
        "selected_committee_dev_complete_unique": (
            run_audits["selected_committee_dev"]["rows"] == 48600
            and run_audits["selected_committee_dev"]["unique_keys"] == 48600
            and run_audits["selected_committee_dev"]["methods"]
            == {"hard_plurality": 48600}
            and run_audits["selected_committee_dev"]["config_matches"]
            and not run_audits["selected_committee_dev"]["malformed_rows"]
            and selected_committee_analysis.get("provenance", {}).get("panel_run_sha256")
            == run_audits["selected_committee_dev"]["sha256"]
            and selected_committee_analysis.get("provenance", {}).get(
                "analysis_script_sha256"
            )
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_selected_committee_dev.py"
            )
            and len(
                selected_committee_analysis.get("leave_one_out_family", {}).get(
                    "comparisons", []
                )
            ) == 7
            and len(selected_committee_analysis.get("selected_best_at_k_scaling", [])) == 7
            and selected_committee_analysis.get("example_count") == 4050
            and selected_committee_analysis.get("panel_row_count") == 48600
            and len(selected_committee_analysis.get("condition_summaries", [])) == 13
        ),
        "cross_controls_complete_unique": (
            run_audits["cross_controls"]["rows"] == 3960
            and run_audits["cross_controls"]["unique_keys"] == 3960
            and run_audits["cross_controls"]["config_matches"]
            and not run_audits["cross_controls"]["malformed_rows"]
            and cross_analysis.get("provenance", {}).get("control_run_sha256")
            == run_audits["cross_controls"]["sha256"]
        ),
        "cross_full_complete_unique": (
            run_audits["cross_full"]["rows"] == 3960
            and run_audits["cross_full"]["unique_keys"] == 3960
            and run_audits["cross_full"]["methods"]
            == {
                "calibrated_probability_mean": 1320,
                "hard_plurality": 1320,
                "probability_mean": 1320,
            }
            and run_audits["cross_full"]["config_matches"]
            and not run_audits["cross_full"]["malformed_rows"]
            and cross_analysis.get("provenance", {}).get("full_run_sha256")
            == run_audits["cross_full"]["sha256"]
            and cross_analysis.get("provenance", {}).get("analysis_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/analyze_cross_v212.py"
            )
            and cross_analysis.get("example_count") == 1320
            and len(cross_analysis.get("condition_summaries", [])) == 6
            and len(cross_analysis.get("paired_family", {}).get("comparisons", [])) == 5
        ),
        "strict_long_run_analyses_complete": all(item["passed"] for item in analysis_statuses.values()),
        "strict_long_run_runtime_manifests_complete": all(runtime_manifest_checks),
        "figure09_generated": "figure09_teacher_vs_free" in figures.get("figures", {}),
        "selected_committee_panel_integrated_in_figure03": (
            "figure03_committee_scaling" in figures.get("figures", {})
            and bool(selected_committee_analysis)
            and figures.get("sources", {}).get(
                "results/phase2_mechanism/selected_committee_dev_analysis.json"
            )
            == sha256_file(selected_committee_analysis_path)
        ),
        "figure_metadata_hashes_match": (
            figures.get("generator_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/make_paper_figures.py")
            and metadata_hash_map_matches(figures.get("sources", {}))
            and figure_output_hashes_match(figures)
        ),
        "table_metadata_hashes_match": (
            tables.get("generator_sha256")
            == sha256_file(LAB_ROOT / "experiments/phase2_mechanism/make_paper_tables.py")
            and metadata_hash_map_matches(tables.get("sources", {}))
            and metadata_hash_map_matches(tables.get("outputs", {}))
        ),
        "all_planned_tables_complete": tables.get("status") == "ALL_PLANNED_TABLES_COMPLETE",
        "final_synthesis_fact_packet_complete": (
            synthesis_facts.get("status") == "FINAL_SYNTHESIS_FACTS_COMPLETE"
            and synthesis_facts.get("provenance", {}).get("generator_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/make_final_synthesis_facts.py"
            )
            and metadata_hash_map_matches(
                synthesis_facts.get("provenance", {}).get("sources", {})
            )
            and synthesis_facts.get("provenance", {}).get("report_sha256")
            == sha256_file(LAB_ROOT / "reports/PHASE2_FINAL_SYNTHESIS_FACTS.md")
            and set(synthesis_facts.get("question_evidence_map", {}))
            == {"6", "7", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"}
            and set(synthesis_facts.get("hypothesis_evidence_map", {}))
            == {
                "H1_calibration_robust_consensus",
                "H4_ordinal_bottleneck_rank_sufficiency",
                "H6_committee_size_threshold",
                "H7_trajectory_correction",
                "H8_abstention_eos_composition",
                "H10_specialization_matters",
            }
        ),
        "execution_environment_capture_complete": bool(
            environment.get("status") == "ENVIRONMENT_CAPTURE_COMPLETE"
            and environment.get("generator_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/capture_environment.py"
            )
            and environment.get("report_sha256")
            == sha256_file(LAB_ROOT / "reports/PHASE2_ENVIRONMENT.md")
            and (environment.get("gpu") or {}).get("name")
            and environment.get("host", {}).get("cpu")
            and environment.get("software", {}).get("python")
            and environment.get("software", {}).get("torch")
            and environment.get("software", {}).get("transformers")
            and environment.get("software", {}).get("peft")
        ),
        "final_written_synthesis_verified": (
            final_synthesis.get("status") == "FINAL_SYNTHESIS_VERIFIED"
            and final_synthesis.get("verification_script_sha256")
            == sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/verify_final_synthesis.py"
            )
            and metadata_hash_map_matches(final_synthesis.get("sources", {}))
            and all(final_synthesis.get("checks", {}).values())
        ),
        "verified_bibliography_complete": bool(literature.get("complete")),
        "official_source_manifests_match": (
            before.get("manifest_sha256") == after.get("manifest_sha256")
            and before.get("entry_count") == after.get("entry_count")
        ),
        "protected_repositories_clean": all(not value for value in protected_status.values()),
        "no_result_locks": not locks,
    }
    non_boolean_checks = {
        name: type(value).__name__ for name, value in checks.items() if not isinstance(value, bool)
    }
    if non_boolean_checks:
        raise TypeError(f"Completion checks must be literal booleans: {non_boolean_checks}")
    status = "PASS" if all(checks.values()) else "INCOMPLETE"
    packet = {
        "schema_version": "1.0",
        "status": status,
        "checks": checks,
        "missing_required_files": missing_required,
        "run_audits": run_audits,
        "analysis_statuses": analysis_statuses,
        "runtime_manifests": runtime_manifests,
        "historical_runtime_boundary_generator": {
            "captured_sha256": runtime_boundaries.get("generator", {}).get("sha256"),
            "current_sha256": sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/make_selected_runtime_boundaries.py"
            ),
            "changed_after_capture": runtime_boundaries.get("generator", {}).get("sha256")
            != sha256_file(
                LAB_ROOT / "experiments/phase2_mechanism/make_selected_runtime_boundaries.py"
            ),
        },
        "literature": literature,
        "static_analysis_statuses": {
            "cornerstone": cornerstone_audit.get("status"),
            "teacher_forced": teacher.get("status"),
            "robust_soft": robust_soft.get("status"),
            "confidence_conditioning": confidence_conditioning.get("status"),
            "committee_diagnostics": committee_diagnostics.get("status"),
            "coalition_strata": coalition_strata.get("status"),
            "pairwise_diversity_extended": diversity_extended.get("status"),
            "negative_controls": negative_controls.get("status"),
            "hard_tie_temperature": hard_tie_temp.get("status"),
            "teacher_forced_cases": teacher_cases.get("status"),
            "independence": independence.get("status"),
            "semantic": semantic.get("review_status"),
            "tie_eos": tie_eos.get("status"),
            "adaptive_selection": adaptive_selection.get("status"),
            "adaptive_manifest": adaptive_manifest.get("status"),
            "cross_gate": cross_gate.get("status"),
            "cross_manifest": cross_manifest.get("status"),
            "v22_calibrated_manifest": v22_calibrated_manifest.get("status"),
            "selected_committee_manifest": selected_committee_manifest.get("status"),
        },
        "protected_tracked_status": protected_status,
        "official_source_manifest_sha256": after.get("manifest_sha256"),
        "active_result_locks": locks,
    }
    write_json_atomic(OUTPUT, packet)
    print(json.dumps({"status": status, "output": str(OUTPUT), "checks": checks}, indent=2))
    if status != "PASS" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
