"""Independent stdlib verification of the frozen reference-only dispatch packet."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = Path(__file__).resolve().parent.parent
EXPECTED_ADMISSION = "4c23cbbdeb9d595e7b61b5f4f3c803051ac2bb22f4ae94793142de5cf73e498b"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def junit(path, count):
    tree = ET.parse(path)
    suites = [tree.getroot()] if tree.getroot().tag == "testsuite" else list(tree.getroot().iter("testsuite"))
    totals = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    require(totals == {"tests": count, "failures": 0, "errors": 0, "skipped": 0}, "test receipt not complete PASS: " + str(path))
    return {"path": str(Path(path).relative_to(CAMPAIGN)), "sha256": sha(path), **totals}


def main():
    out = CAMPAIGN / "reviews/reference_dispatch_receipt.json"
    require(not out.exists(), "independent review receipt already exists")
    admission_path = CAMPAIGN / "admission_v2.json"
    require(sha(admission_path) == EXPECTED_ADMISSION, "frozen admission changed")
    admission = read(admission_path)
    require(admission["status"] == "READY_FOR_REFERENCE_DISPATCH", "not ready admission")
    require(admission["id"] == "REF-V1-CPU-001" and not admission["training"] and not admission["native_test"] and not admission["gpu_lane"], "scope differs")
    require(admission["review_receipt"] == "reviews/reference_dispatch_receipt.json", "review binding differs")
    checked = {}
    for relative, expected in admission["dispatch_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        require(path.is_relative_to(ROOT.resolve()), "dispatch path leaves worktree")
        actual = sha(path)
        require(actual == expected, "dispatch source changed: " + relative)
        checked[relative] = actual
    require(len(checked) == 66, "dispatch closure count differs")
    source = Path(admission["source"]).resolve(strict=True)
    historical = read(ROOT / "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json")
    require(source == Path(historical["model_path"]).resolve(strict=True), "source not exact historical directory")
    asset_checks = {}
    for name, expected in admission["checkpoint_files"].items():
        actual = sha(source / name)
        require(actual == expected == historical["checkpoint_files_before"][name], "canonical source asset changed: " + name)
        asset_checks[name] = actual
    preflight = read(CAMPAIGN / "runs/preflight-cpu-001/result/preflight.json")
    require(preflight["status"] == "METADATA_READY_NOT_QUALIFIED" and not preflight["reasons"] and preflight["native_runs"] == 0, "historical metadata preflight not ready")
    require(sha(Path(admission["source_runtime"]["qwen_file"])) == admission["source_runtime"]["qwen_sha256"] == historical["qwen_implementation_sha256"], "Qwen bytes changed")
    require(admission["source_runtime"]["python"] == historical["python"] and admission["source_runtime"]["torch"] == historical["torch"] and admission["source_runtime"]["transformers"] == historical["transformers"], "runtime mismatch")
    tests = {
        "coordinator_reference_cpu": junit(CAMPAIGN / "runs/cpu-v1-001/junit.xml", 62),
        "coordinator_complete_inherited_armr": junit(CAMPAIGN / "runs/cpu-armr-001/junit.xml", 65),
        "independent_control_metadata": junit(CAMPAIGN / "reviews/control_binding_review_junit.xml", 15),
        "independent_final_containment": junit(CAMPAIGN / "reviews/supervision_final_independent_junit.xml", 11),
    }
    for run in ("cpu-v1-001", "cpu-armr-001", "preflight-cpu-001"):
        receipt = read(CAMPAIGN / "runs" / run / "receipt.json")
        require(receipt["status"] == "COMMAND_COMPLETE" and receipt["exit_code"] == 0 and receipt["inherited_bytes_unchanged"], "CPU/preflight process failed")
    binding_path = CAMPAIGN / "runs/binding-001/MODEL_DATA_BINDING.json"
    prompt_path = CAMPAIGN / "runs/binding-001/PROMPT_AUDIT.json"
    binding = read(binding_path)
    require(sha(binding_path) == admission["binding_sha256"], "metadata binding changed")
    require(binding["status"] == "TOKENIZER_METADATA_BOUND_NATIVE_NOT_QUALIFIED" and binding["loaded_model_state_sha256"] is None and not binding["native_qualified"], "metadata falsely qualified")
    rows = read(prompt_path)
    dev = [r for r in rows if r["split"] == "DEV"]
    smoke = [r for r in rows if r["split"] == "SMOKE"]
    require(len(dev) == 16 and len(smoke) == 8 and len(rows) == 24, "population changed")
    for row in rows:
        n, cap = len(row["input_token_ids"]), row["max_new_tokens"]
        require(n == row["input_tokens"] and n + cap - 1 == row["max_forward_prefix_tokens"] and cap == row["maximum_model_calls"], "shape formula mismatch")
        require(cap * n + cap * (cap - 1) // 2 == row["maximum_processed_prefix_positions"], "position count mismatch")
    forecast = {
        "physical_evaluations": 13 * len(smoke) + 18 * len(dev),
        "maximum_model_calls": 13 * sum(r["maximum_model_calls"] for r in smoke) + 18 * sum(r["maximum_model_calls"] for r in dev),
        "maximum_processed_prefix_positions": 13 * sum(r["maximum_processed_prefix_positions"] for r in smoke) + 18 * sum(r["maximum_processed_prefix_positions"] for r in dev),
        "max_input_tokens": max(r["input_tokens"] for r in rows),
        "max_forward_prefix_tokens": max(r["max_forward_prefix_tokens"] for r in rows),
    }
    require(forecast == {"physical_evaluations": 392, "maximum_model_calls": 6348, "maximum_processed_prefix_positions": 1248966,
                         "max_input_tokens": 2237, "max_forward_prefix_tokens": 2248}, "complete forecast differs")
    require(admission["hard_deadline_seconds"] == 18000 and admission["idle_forward_deadline_seconds"] == 950 and admission["worker_rss_ceiling_bytes"] == 6 * 1024**3, "registered hard limits differ")
    require(admission["native_finalization_reserve_seconds"] == 1800 and admission["hard_deadline_seconds"] + 1800 <= 21600, "campaign budget exceeded")
    require(admission["host_available_ram_floor_bytes"] >= 7.4 * 1024**3 and admission["disk_ceiling_bytes"] == 2 * 1024**3 and admission["disk_finalization_reserve_bytes"] == 512 * 1024**2, "resource floors/reserves differ")
    dispositions = read(CAMPAIGN / "DISPOSITIONS.json")
    require({r["id"] for r in dispositions["issues"]} == {f"R{n:02d}" for n in range(1, 12)}, "issue disposition incomplete")
    for row in dispositions["issues"]:
        if "reviewer_interpretation" in row:
            require((CAMPAIGN / row["reviewer_interpretation"]).is_file(), "broken issue receipt reference")
    policy_rows = [json.loads(line) for line in (CAMPAIGN / "RESEARCH_POLICY.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    policy = next(r for r in policy_rows if r["version"] == 2)
    require(policy["scientific_gates_weakened"] is False and policy["empirical_research_policy_improvement"] == "NOT_MEASURED", "policy improperly weakens/claims gains")
    result = {
        "schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_REFERENCE_DISPATCH", "admission": "admission_v2.json", "admission_sha256": EXPECTED_ADMISSION,
        "review_script_sha256": sha(__file__), "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "checked_dispatch_hashes": checked, "canonical_assets_independently_rehashed": asset_checks,
        "test_receipts": tests, "final_containment_test_source_sha256": sha(CAMPAIGN / "tests/test_supervision.py"),
        "final_containment_independent_pytest_seconds": 8.29, "independent_control_metadata_pytest_seconds": 7.68,
        "metadata_preflight_sha256": sha(CAMPAIGN / "runs/preflight-cpu-001/result/preflight.json"),
        "prompt_audit_sha256": sha(prompt_path), "binding_sha256": sha(binding_path), "independently_recomputed_forecast": forecast,
        "issue_dispositions_reviewed": {r["id"]: {"coordinator_disposition": r["disposition"], "reviewer_disposition": "ACCEPTED_FOR_EXACT_REFERENCE_SCOPE"} for r in dispositions["issues"]},
        "research_policy_v2_review": {"status": "ACCEPT_PROCEDURE_RULE_CPU_APPARATUS_CHECKS_PASS", "proposal": policy,
                                      "proposal_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                                      "acceptance_evidence": "15-test independent fake-host/metadata suite validates prefix RNG/reset/seed streams; future native caller must bind actual cfg.seed and qualify corrected stochastic control before its task measurements.",
                                      "empirical_native_allocation_efficiency": "NOT_MEASURED"},
        "scope": "One coordinator-owned exact historical CPU-FP32 original v1 native-dev attempt, with built-in native identity/all-arm smoke gates before complete exposed DEV. No successor/native TEST/deeper recurrence/training/GPU authorization.",
        "qualification_semantics": "Admission PASS permits the explicitly authorized original guarded runner to attempt qualification. It does not assert current loaded native identity, scaled-arm smoke success, native maximum-shape/exception PASS or efficacy.",
        "native_model_loads_by_reviewer": 0, "native_inference_by_reviewer": 0, "protected_or_native_test_reads": False,
        "remaining_limits": ["Fresh maximum-shape and forced native-exception checks NOT_RUN; mandatory before successor task measurements",
                             "Legacy A5 is repeated per-example constant alpha and duplicated nominal schedule seeds; H4 intended random-prefix comparison incomplete",
                             "Sixteen exposed DEV rows and at most thirteen provisional parent scenarios provide exploratory diagnostics only",
                             "Historical cost forecast uncertain; deadline/safety stop preserves incomplete comparison and all costs",
                             "Source/resource/lock/budget checks re-execute in supervisor and preflight; no other process may be killed",
                             "No confirmatory efficacy, deployability, speedup or empirical research-policy-efficiency claim"]}
    # Recheck the frozen admission after all reads; this receipt is exclusive.
    require(sha(admission_path) == EXPECTED_ADMISSION, "admission changed during review")
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(out), "admission_sha256": EXPECTED_ADMISSION,
                      "dispatch_files_checked": len(checked), "assets_checked": len(asset_checks)}))


if __name__ == "__main__":
    main()
