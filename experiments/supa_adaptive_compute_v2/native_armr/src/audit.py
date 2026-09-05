"""Independent receipt reconciliation and score reconstruction; no model rerun."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

from .analysis import analyze_complete_bank, audit_rows, linked_robustness
from .protocol import DEFAULT_CONFIG, load_protocol
from shared.datasets import load_frozen_dataset
from shared.evaluation import _read_complete, aggregate_runs
from shared.hashing import canonical_sha256, sha256_file, write_json
from shared.manifests import new_run, utc_now, verify_registration
from shared.legacy import REPO
from shared.reporting import quality_cost_frontier


def audit_runs(*, config_path: Path, dataset_manifest: Path, role: str,
               run_dirs: list[Path], primary_run_dirs: list[Path] | None = None) -> tuple[Path, dict]:
    if config_path.resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("independent audit requires the canonical registered native protocol")
    config = load_protocol(config_path)
    if role == "confirmatory_diagnostics" and not primary_run_dirs:
        raise ValueError("diagnostic audit requires complete original evidence via --primary-run-list")
    if role != "confirmatory_diagnostics" and primary_run_dirs:
        raise ValueError("original companion evidence is only used for diagnostic auditing")
    registration = verify_registration("native_armr", config_path)
    dataset_manifest = Path(dataset_manifest).resolve()
    if registration["dataset_manifests"].get(dataset_manifest.relative_to(REPO).as_posix()) != sha256_file(dataset_manifest):
        raise ValueError("audited dataset must belong to current Track A registration")
    output, receipt = new_run("native_armr", "04_independent_audit", {
        "role": role, "config_sha256": canonical_sha256(config),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "registration_sha256": registration["registration_sha256"],
        "sources": registration["sources"], "model_loaded": False,
        "audit_kind": "raw_evidence_reconciliation_and_frozen_rescoring",
        "input_runs": [str(Path(path).resolve()) for path in run_dirs],
        "primary_input_runs": [str(Path(path).resolve()) for path in primary_run_dirs or []]})
    start = time.perf_counter()
    try:
        # Shared coverage and integrity gates are deliberately followed by an
        # independent reconstruction, never trusted stored analysis summaries.
        aggregate_runs(track="native_armr", config=config, dataset_manifest=dataset_manifest,
                       role=role, run_dirs=run_dirs)
        if primary_run_dirs:
            aggregate_runs(track="native_armr", config=config, dataset_manifest=dataset_manifest,
                           role="confirmatory_dev", run_dirs=primary_run_dirs)
        cases = load_frozen_dataset(dataset_manifest, track="native_armr", role=role)
        by_arm, original_by_arm, manifests, input_hashes = {}, {}, [], {}
        for path, expected_role in ([(p, role) for p in run_dirs] + [(p, "confirmatory_dev") for p in primary_run_dirs or []]):
            path = Path(path).resolve()
            manifest, rows = _read_complete(path, "native_armr")
            if manifest["role"] != expected_role:
                raise ValueError("companion evidence role mismatch")
            if manifest["registration_sha256"] != registration["registration_sha256"] or manifest["sources"] != registration["sources"]:
                raise ValueError("audited run does not match current scientific registration")
            for key in ("seed", "threads", "dtype", "device", "max_prompt_seconds"):
                if manifest.get(key) != config[key]:
                    raise ValueError(f"run runtime differs from preregistration: {key}")
            if (manifest.get("cache_mode") != "full_prefix_no_cache" or
                manifest.get("model_class") != config["model_class"] or
                manifest.get("torch") != "2.12.1+cu126" or manifest.get("transformers") != "5.14.1"):
                raise ValueError("unqualified model/runtime receipt")
            if (manifest.get("trainable_parameters") != 0 or manifest.get("backbone_gradients") != 0 or
                not manifest.get("state_sha256_before") or
                manifest["state_sha256_before"] != manifest.get("state_sha256_after") or
                manifest["checkpoint_files_before"] != manifest.get("checkpoint_files_after") or
                manifest["checkpoint_files_before"].get("model.safetensors") != config["checkpoint_sha256"]):
                raise ValueError("frozen model identity failed independent reconciliation")
            for row in rows:
                name = canonical_sha256(row["example_id"])
                stored_case = json.loads((path / "cases" / (name + ".json")).read_text(encoding="utf-8"))
                if stored_case != row:
                    raise ValueError("per-case artifact differs from aggregate raw evidence")
                logits = path / "logits" / (name + ".pt")
                if sha256_file(logits) != manifest["logits_sha256"].get(logits.name):
                    raise ValueError("first-token logit artifact changed")
            target = original_by_arm if primary_run_dirs and expected_role == "confirmatory_dev" else by_arm
            target.setdefault(manifest["arm"], []).extend(rows)
            manifests.append(manifest)
            input_hashes[str(path / "run_manifest.json")] = sha256_file(path / "run_manifest.json")
        identity_keys = ("state_sha256_before", "checkpoint_files_before", "model_class", "torch",
                         "transformers", "qwen_implementation_sha256", "parameters")
        if len({canonical_sha256({k: m.get(k) for k in identity_keys}) for m in manifests}) != 1:
            raise ValueError("model/runtime identity differs between paired arms")
        report = analyze_complete_bank(cases, by_arm, config, role)
        if primary_run_dirs:
            original_cases = load_frozen_dataset(dataset_manifest, track="native_armr", role="confirmatory_dev")
            original_rows = audit_rows(original_cases, original_by_arm, config, "confirmatory_dev")
            report["linked_original_robustness"] = linked_robustness(original_cases, original_rows, cases, by_arm, config)
        for manifest in manifests:
            arm = manifest["arm"]
            if arm != "AM":
                work = 0 if arm == "A0" else config["control"]["additional_matmul_flops_per_token"] // (1 if arm == "A3" else 2)
                if manifest["operator"] != {"added_parameter_count": 0,
                                            "added_active_parameters": 0,
                                            "extra_matmul_flops_per_token": work}:
                    raise ValueError("native recurrence work/parameter inventory differs from registration")
        # Added control weights must be identical across shards and remain frozen.
        controls = [m for m in manifests if m["arm"] == "AM"]
        states = {m.get("operator_state_sha256") for m in controls}
        if len(states) != 1 or None in states:
            raise ValueError("compute-control state identity missing or changed between shards")
        for manifest in controls:
            if any(manifest.get(key) != 0 for key in ("operator_trainable_parameters", "operator_gradients",
                                                      "operator_trainable_parameters_after", "operator_gradients_after")):
                raise ValueError("compute-control frozen inventory missing or failed")
            if manifest["operator"] != {
                "added_parameter_count": config["control"]["additional_matrix_parameters"],
                "added_active_parameters": config["control"]["additional_matrix_parameters"],
                "extra_matmul_flops_per_token": config["control"]["additional_matmul_flops_per_token"]}:
                raise ValueError("compute-control budget differs from preregistration")
        report.update(registration_sha256=registration["registration_sha256"],
                      input_manifest_sha256=input_hashes,
                      operator_inventory={arm: next(m["operator"] for m in manifests if m["arm"] == arm)
                                          for arm in by_arm},
                      limitations=["Audit independently recomputes scoring and decisions from raw responses, but does not rerun model inference.",
                                   "Bootstrap inference concerns these generated scenario families; it does not establish external benchmark generalization.",
                                   "A successful AM comparison concerns this fixed untrained control, not every possible compute allocation."])
        frontier = report["quality_cost_frontier"]["arms"]
        for arm in by_arm:
            peak = max(m["peak_memory_bytes"] for m in manifests if m["arm"] == arm and m["role"] == role)
            report["cost"][arm]["peak_run_process_memory_bytes"] = peak
            frontier[arm]["peak_memory_bytes"] = peak
        report["quality_cost_frontier"] = quality_cost_frontier(frontier)
        write_json(output / "analysis.json", report)
        receipt.update(status="COMPLETE", analysis_sha256=sha256_file(output / "analysis.json"),
                       decision=report["decision"], checked_runs=len(manifests))
    except BaseException as exc:
        receipt.update(status="FAILED", failure_receipt={"type": type(exc).__name__,
                       "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        import psutil
        memory = psutil.Process().memory_info()
        receipt.update(end_utc=utc_now(), wall_seconds=time.perf_counter()-start,
                       peak_memory_bytes=getattr(memory, "peak_wset", memory.rss))
        write_json(output / "run_manifest.json", receipt)
    return output, report
