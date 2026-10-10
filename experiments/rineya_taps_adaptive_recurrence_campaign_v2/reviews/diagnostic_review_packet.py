"""Independent H3 source/hash/population/cost admission; stdlib only, no weights."""
from datetime import datetime, timezone
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
C = Path(__file__).resolve().parent.parent
EXPECTED = "69c4f0367a733ea525c46a1955ce01ddadb1f71280737c91cd38929d9426edbe"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tests(path, expected):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts == {"tests": expected, "failures": 0, "errors": 0, "skipped": 0}, "independent CPU tests differ")
    return {"counts": counts, "junit_sha256": sha(path)}


def main():
    destination = C / "reviews/diagnostic_dispatch_receipt.json"
    require(not destination.exists(), "exclusive H3 review already exists")
    packet = C / "diagnostic_admission_v3.json"
    require(sha(packet) == EXPECTED, "final H3 admission differs")
    admission = read(packet)
    require(admission["status"] == "READY_FOR_DIAGNOSTIC_FP32_DISPATCH"
            and admission["review_receipt"] == "reviews/diagnostic_dispatch_receipt.json", "H3 contract differs")
    require(len(admission["dispatch_hashes"]) == 72, "H3 source closure differs")
    for relative, digest in admission["dispatch_hashes"].items():
        path = (ROOT / relative).resolve(strict=True)
        require(path.is_relative_to(ROOT) and sha(path) == digest, "frozen H3 source differs: " + relative)
    require(sha(C / "gpu_native.py") == "4c1a05c6511ff267d790654475f2a22cb5a3b8fa02857b295aa1ab3ae6c20e6f", "qualified forward helper changed")
    require(sha(C / "diagnostic_native.py") == "7b94f88308e7afbf4836065e73d28608bdae2c6e753226aee4a428a2941cbea6"
            and sha(C / "sensitivity.py") == "5d0af45a8e91f0a00d91f83244105d06f1beba6f422cc3cd6d20c2d43d762f0d", "H3 math/worker changed")
    bindings = read(C / "runs/binding-001/MODEL_DATA_BINDING.json")
    require(sha(C / "runs/binding-001/MODEL_DATA_BINDING.json") == admission["binding_sha256"], "metadata binding differs")
    require(admission["checkpoint_files"] == bindings["checkpoint_files"]
            and admission["source"] == bindings["source_path"], "canonical assets/source differs")
    for name, digest in admission["checkpoint_files"].items():
        require(Path(name).name == name and sha(Path(admission["source"]) / name) == digest, "canonical asset changed: " + name)
    cpu = tests(C / "reviews/diagnostic_review_cpu_junit.xml", 23)
    parent = tests(C / "reviews/diagnostic_parent_final_junit.xml", 4)
    # Reproduce only the pinned pure canonical-answer function, without importing Torch.
    metric_path = ROOT / "experiments/anchor_routed_middle_recurrence_v1/src/armr/metrics.py"
    tree = ast.parse(metric_path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "canonical_answer")
    namespace = {}
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    dev_path = ROOT / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/mergedev_fast.jsonl"
    require(sha(dev_path) == bindings["datasets"]["DEV"]["sha256"], "allowlisted DEV changed")
    cases = [json.loads(line) for line in dev_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    metadata = {row["case_id"]: row for row in read(C / "runs/binding-001/PROMPT_AUDIT.json") if row["split"] == "DEV"}
    require(len(cases) == len(metadata) == 16, "complete population differs")
    population = []
    for case in cases:
        tokens = metadata[case["case_id"]]["input_tokens"]
        require(tokens == len(metadata[case["case_id"]]["input_token_ids"]), "bound token count differs")
        answer = None if tokens > 256 else namespace["canonical_answer"](case)
        status = "EXCLUDED_SHAPE" if tokens > 256 else "ELIGIBLE" if answer is not None else "NOT_APPLICABLE_INHERITED_TARGET"
        population.append({"case_id": case["case_id"], "scenario_id": case["scenario_id"], "input_tokens": tokens,
                           "status": status, "inherited_canonical_answer": answer,
                           "target_token_id": None, "target_token_status": "NATIVE_PRELOAD_BINDING_PENDING"})
    counts = Counter(row["status"] for row in population)
    require(counts == {"ELIGIBLE": 7, "NOT_APPLICABLE_INHERITED_TARGET": 7, "EXCLUDED_SHAPE": 2}, "eligibility differs")
    require(len({row["scenario_id"] for row in population if row["status"] == "ELIGIBLE"}) == 5, "provisional clusters differ")
    forecast = {"full_forward_attempts": 2 * 4 * 2 + 2 * 3 + 1 + 1 + 7 * 3,
                "expected_completed": 44, "expected_partial": 1, "cache_rejections": 2,
                "model_api_attempts": 47, "state_adjoints": 2 + 7,
                "maximum_attempted_prefix_positions": 47 * 256}
    require(all(admission["forecast"][key] == value for key, value in forecast.items()), "complete H3 forecast differs")
    require(abs(47 * .8833317 * 1.5 - admission["forecast"]["conservative_forward_seconds"]) < 1e-8
            and admission["forecast"]["adjoint_allowance_seconds"] == 90
            and admission["forecast"]["cold_load_hash_allowance_seconds"] == 200, "cost forecast differs")
    contract = {"device": "cuda:0", "dtype": "float32", "attn_implementation": "eager", "TF32": False,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True, "training": False,
                "native_test": False, "parameter_gradients_allowed": False, "generation_pool_access": False,
                "fd_step": .01, "fd_absolute_tolerance": .001, "fd_relative_tolerance": .03,
                "maximum_prefix_tokens": 256, "hard_deadline_seconds": 1800,
                "native_finalization_reserve_seconds": 1800, "infrastructure_retry_available": False,
                "parent_host_available_floor_bytes": 7 * 1024**3, "worker_host_available_floor_bytes": 6 * 1024**3,
                "worker_rss_ceiling_bytes": 6 * 1024**3, "gpu_initial_free_floor_bytes": 6 * 1024**3,
                "gpu_allocated_ceiling_bytes": 5 * 1024**3, "gpu_global_free_reserve_bytes": 1024**3}
    require(all(admission[key] == value for key, value in contract.items()), "H3 gradient/operator/resource contract differs")
    ledger = [json.loads(line) for line in (C / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    spent = sum(row["native_worker_seconds"] for row in ledger if row.get("event") == "native_completed")
    require(spent + 1800 + 1800 <= 21600 and not (C / "native_worker.lock").exists(), "native allowance/sole-worker gate differs")
    result = {"schema_version": "1.0", "reviewer": "/root/adversarial_review", "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS_DIAGNOSTIC_FP32_DISPATCH", "admission": packet.name, "admission_sha256": EXPECTED,
              "review_script_sha256": sha(__file__), "checked_dispatch_hashes": admission["dispatch_hashes"],
              "independent_CPU_tests": cpu, "final_parent_CPU_tests": parent, "population": population,
              "independently_recomputed_forecast": forecast, "native_seconds_used_at_review": spent,
              "native_model_loads_by_reviewer": 0, "native_qualification": "NOT_RUN",
              "resolved_issue": {"id": "H3-P01", "source": "diagnostic_supervisor.py:22", "disposition": "attempts/population/result journal progress now observed; final4-test regression passes"},
              "interpretation": "post-L13 fixed-state directional derivative through actual frozen suffix/custom readout; first canonical token CE surrogate",
              "limitations": ["Native256-gradient value/FD/restoration/resource qualification remains required before7measurements.",
                              "No semantic efficacy, full continuation NLL, adaptive Jacobian stability, parameter training or confirmation is established.",
                              "Seven exposed targets/five provisional clusters; counterfactual choice winner has no inherited canonical target and is not measured.",
                              "Closed-form two-observation FP32 energies can round differently from incremental v1; they are labelled offline diagnostics.",
                              "No live RAM/GPU inventory performed by reviewer; parent must enforce fresh7GiB host and6GiB GPU floors; no retry remains."],
              "unresolved_dispatch_blockers": []}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "admission_sha256": EXPECTED, "dispatch_hashes": 72, "population": counts, "native_seconds_used": spent}))


if __name__ == "__main__":
    main()
