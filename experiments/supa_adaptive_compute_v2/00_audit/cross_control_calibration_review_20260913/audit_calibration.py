"""Independent standard-library audit of B-only XM calibration JSON evidence.

No experiment imports, tensor deserialization, model execution or Git mutations.
Scalar aggregates are independently reconstructed from recorded per-case metrics.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[3]
PROGRAMME = REPO / "experiments/supa_adaptive_compute_v2"
EXPECTED_REGISTRATION = "bf99e713104573b7aafb8f5511278fd473bcd406ff1d6fece3d0ebe5787534a0"
FORMULA = "sqrt(mean_scenario_mean_ordinary_token_relative_XG_energy / mean_scenario_mean_ordinary_token_relative_XM_energy)"
FIELDS = ("teacher_energy", "control_energy", "hidden_rms", "teacher_delta_rms", "control_delta_rms")
checks = []
inputs = {}


def check(condition, description):
    checks.append({"check": description, "pass": bool(condition)})
    if not condition:
        raise AssertionError(description)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            h.update(data)
    result = h.hexdigest()
    inputs[str(Path(path).resolve())] = result
    return result


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    digest(path)
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key {key}")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def git(*args, data=None):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(REPO), *args], input=data)


def committed_at(commit, paths):
    """Check complete file bytes at the actual pre-execution commit in one call."""
    names = sorted(set(paths))
    result = git("cat-file", "--batch", data="".join(f"{commit}:{name}\n" for name in names).encode())
    offset = 0
    for name in names:
        end = result.index(b"\n", offset)
        header = result[offset:end].split()
        check(len(header) == 3 and header[1] == b"blob", f"committed before execution: {name}")
        size = int(header[2])
        blob = result[end + 1:end + 1 + size]
        current = (REPO / name).read_bytes()
        check(blob.replace(b"\r\n", b"\n") == current.replace(b"\r\n", b"\n"),
              f"execution commit bytes (newline-normalized): {name}")
        offset = end + 1 + size + 1
    check(offset == len(result), "complete Git batch stream consumed")


def audit(report_path):
    report_path = report_path.resolve()
    allowed = PROGRAMME / "cross_scale/generation_stage/v1/evidence/00_control_calibration"
    check(report_path.is_relative_to(allowed) and report_path.name == "report.json", "allowed completed B calibration report")
    report = read(report_path)
    directory = report_path.parent
    run, started = read(directory / "run_manifest.json"), read(directory / "started_manifest.json")
    registration_path = REPO / report["registration_path"]
    registration = read(registration_path)
    contract = registration["contract"]
    config = contract["config"]
    check(report["registration_sha256"] == EXPECTED_REGISTRATION == registration["registration_sha256"] == canonical(contract), "fixed preregistration digest")
    check(registration["status"] == "FROZEN" and report["status"] == run["status"] == "COMPLETE" and started["status"] == "STARTED", "successful immutable lifecycle")
    check(run["report_sha256"] == digest(report_path), "final manifest binds report bytes")
    check(run["failure_receipt"] is None and started["failure_receipt"] is None, "no hidden failed status")
    for key, value in contract.items():
        check(started.get(key) == value == run.get(key), f"started/final contract: {key}")
        if key not in {"schema_version", "track", "config"}:
            check(report.get(key) == value, f"report contract: {key}")
    check(report["formula"] == config["formula"] == FORMULA, "registered equal-scenario energy formula")
    boundaries = {"track": "cross_scale", "label_free": True, "answer_labels_used": False,
                  "generated_tokens": 0, "holdout_selects_alpha": False, "alpha_search": False,
                  "alpha_clipping": False, "directional_fit": False, "trainable_parameters": 0,
                  "backbone_gradients": 0, "canonical_unchanged": True, "activation_mmap": True,
                  "accessed_supa_layers": [19], "accessed_donor_activation_layers": [],
                  "full_supa_model_loaded": False, "donor_layer_index": 24, "donor_hidden_size": 2560,
                  "device": "cpu", "dtype": "float32", "threads": 4, "use_cache": False,
                  "additional_parameters": 106168320, "additional_active_parameters": 106168320,
                  "additional_matmul_flops_per_token": 212336640}
    for key, expected in boundaries.items():
        check(type(report.get(key)) is type(expected) and report[key] == expected, f"science/runtime record: {key}")
    runtime = {"device": "cpu", "dtype": "float32", "threads": 4, "seed": 2701,
               "torch": "2.12.1+cu126", "transformers": "5.14.1", "attention": "eager", "use_cache": False}
    for key, expected in runtime.items():
        check(report["runtime_identity"].get(key) == expected, f"runtime identity: {key}")
    committed_paths = [report["registration_path"]]
    for name, expected in contract["source_files"].items():
        check(digest(REPO / name) == expected, f"current frozen source bytes: {name}")
        # Preserved historical graft.py is intentionally untracked; its exact bytes
        # remain bound by the preregistration instead of an invented source commit.
        if name.startswith("experiments/supa_adaptive_compute_v2/"):
            committed_paths.append(name)
    for stem in ("calibration_config", "dataset_manifest", "calibration_data", "export_manifest", "export_tensor", "capture", "activation"):
        name = contract[stem + "_path"]
        check(digest(REPO / name) == contract[stem + "_sha256"], f"current input bytes: {stem}")
        if stem not in {"activation", "export_tensor"}:
            committed_paths.append(name)
    check(read(REPO / contract["calibration_config_path"]) == config, "frozen calibration config contents")
    committed_at(started["commit_sha"], committed_paths)
    check(started["commit_sha"] == run["commit_sha"], "execution commit stable")
    check(datetime.fromisoformat(registration["registered_at_utc"]) <= datetime.fromisoformat(started["start_utc"]) <= datetime.fromisoformat(run["end_utc"]), "registration/start/end timestamp order")
    manifest = read(REPO / contract["dataset_manifest_path"])
    manifest_id = manifest.pop("manifest_sha256")
    check(canonical(manifest) == manifest_id == "6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46", "frozen B bank canonical identity")
    check(manifest["track"] == "cross_scale" and manifest["sealed_access"] is False, "B-only unsealed bank")
    prompt_fields = ("example_id", "scenario_id", "family", "template_id", "prompt_sha256", "alignment_partition")
    rows = []
    with (REPO / contract["calibration_data_path"]).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            raw = json.loads(line)
            check((raw["track"], raw["split"], raw["variant"]) == ("cross_scale", "calibration", "original"), "B original calibration role")
            check(canonical(raw["prompt"]) == raw["prompt_sha256"], "prompt identity")
            rows.append({key: raw[key] for key in prompt_fields})
            del raw
    check(len(rows) == len({r["scenario_id"] for r in rows}) == len({r["example_id"] for r in rows}) == 128, "128 unique B scenarios")
    check(Counter(r["alignment_partition"] for r in rows) == {"fit": 96, "holdout": 32}, "96 fit / 32 holdout coverage")
    expected_names = {canonical(r["example_id"]) + ".json" for r in rows}
    check(set(report["case_files"]) == expected_names == {p.name for p in (directory / "cases").glob("*.json")}, "exact immutable case inventory")
    special = report["special_control_token_ids"]
    check(special == sorted(set(special)) and all(type(v) is int and v >= 0 for v in special), "special token inventory")
    cases = []
    for row in rows:
        name = canonical(row["example_id"]) + ".json"
        path = directory / "cases" / name
        check(digest(path) == report["case_files"][name], f"case bytes: {row['example_id']}")
        case = read(path)
        check(all(case[key] == value for key, value in row.items()), f"case identity: {row['example_id']}")
        expected_partition = "holdout" if int(row["scenario_id"].rsplit("_", 1)[1]) % 4 == 3 else "fit"
        check(case["alignment_partition"] == expected_partition, "independent scenario partition")
        tokens = case["token_ids"]
        check(all(type(v) is int and v >= 0 for v in tokens) and 0 < len(tokens) <= 256, "whole valid token sequence")
        check(case["full_prompt_tokens"] == case["retained_prompt_tokens"] == len(tokens), "no prompt truncation")
        ordinary = sum(v not in special for v in tokens)
        counts = {"all": len(tokens), "ordinary": ordinary, "special_control": len(tokens) - ordinary, "final_prompt": 1}
        check(ordinary > 0 and set(case["metrics"]) == set(counts), "complete token strata")
        for stratum, count in counts.items():
            metric = case["metrics"][stratum]
            check((metric is None) == (count == 0), f"stratum missingness: {stratum}")
            if metric is not None:
                check(metric["token_count"] == count, f"independent token count: {stratum}")
                check(all(type(metric[key]) in (int, float) and math.isfinite(metric[key]) and metric[key] >= 0 for key in FIELDS), f"finite nonnegative metrics: {stratum}")
        check(case["hidden_sha256_before"] == case["hidden_sha256_after"], "activation preservation record")
        hash_names = ("hidden_sha256_before", "hidden_sha256_after", "teacher_delta_sha256", "control_delta_sha256", "original_positions_sha256", "original_mask_sha256", "projected_hidden_sha256")
        hashes = [case[key] for key in hash_names] + case["donor_rope_sha256"]
        check(len(case["donor_rope_sha256"]) == 2 and all(re.fullmatch("[0-9a-f]{64}", h) for h in hashes), "tensor/position/mask/RoPE hash inventory")
        check(.05 <= case["donor_output_input_rms_ratio"] <= 10, "registered donor RMS guard")
        cases.append(case)
    reconstructed = {}
    for partition, n in (("fit", 96), ("holdout", 32)):
        subset = [case for case in cases if case["alignment_partition"] == partition]
        totals = {key: math.fsum(case["metrics"]["ordinary"][key] for case in subset) / n for key in FIELDS}
        aggregate = {**totals, "scenarios": n, "example_ids": sorted(c["example_id"] for c in subset),
                     "scenario_ids": sorted(c["scenario_id"] for c in subset),
                     "ordinary_tokens": sum(c["metrics"]["ordinary"]["token_count"] for c in subset)}
        check(aggregate["example_ids"] == contract[partition + "_example_ids"], f"preregistered {partition} IDs")
        if partition == "fit":
            alpha = math.sqrt(totals["teacher_energy"] / totals["control_energy"])
            check(math.isfinite(alpha) and alpha > 0 and alpha == report["alpha"], "independent fit-only alpha")
        aggregate["scaled_control_energy"] = alpha * alpha * totals["control_energy"]
        check(aggregate == report[partition], f"independent equal-scenario {partition} aggregation")
        reconstructed[partition] = aggregate
    check(not set(reconstructed["fit"]["scenario_ids"]) & set(reconstructed["holdout"]["scenario_ids"]), "no fit/holdout scenario overlap")
    fit = read(directory / "fit_scale.json")
    check(digest(directory / "fit_scale.json") == report["fit_scale_sha256"], "fit scale marker bytes")
    check(fit == {"alpha": alpha, "formula": FORMULA, "heldout_transformations_at_write": 0,
                  "fit": {k: v for k, v in reconstructed["fit"].items() if k != "scaled_control_energy"}}, "fit-only marker contents")
    modules = report["module_state_receipts"]
    check(set(modules) == {"donor", "rotary", "bridges", "control"}, "complete frozen module records")
    for name, receipt in modules.items():
        check(receipt["trainable_parameters"] == receipt["backbone_gradients"] == 0, f"frozen/no-gradient record: {name}")
        check(receipt["state_sha256_before"] == receipt["state_sha256_after"] and bool(receipt["state_sha256_before"]), f"state preservation record: {name}")
        check(receipt["buffer_sha256_before"] == receipt["buffer_sha256_after"], f"buffer preservation record: {name}")
    check(modules["control"]["parameters"] == 106168320 and modules["control"]["state_sha256_before"] == report["control_state_sha256"], "control parameter and identity record")
    context = contract["upstream_context"]
    check(report["supa_checkpoint_files_before"] == report["supa_checkpoint_files_after"] == context["supa_files"], "Supa checkpoint preservation records")
    check(report["donor_identity_before"] == report["donor_identity_after"] == context["donor_identity"], "donor checkpoint preservation records")
    export = read(REPO / contract["export_manifest_path"])
    check(report["bridge_source_hashes"] == report["bridge_copy_hashes"] == export["tensor_sha256"]["XG"], "teacher uses selected XG bridge records")
    check(0 < report["peak_rss_bytes"] <= 6 * 2**30 and 0 < report["wall_seconds"] <= 172800, "recorded stage resource bounds")
    return {"alpha": alpha, "fit": reconstructed["fit"], "holdout": reconstructed["holdout"],
            "control_state_sha256": report["control_state_sha256"], "execution_commit": started["commit_sha"],
            "report_sha256": digest(report_path), "source_file_count": len(contract["source_files"]),
            "preexecution_committed_file_count": len(set(committed_paths)),
            "peak_rss_bytes": report["peak_rss_bytes"], "wall_seconds": report["wall_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(BASE) or output == BASE:
        raise ValueError("audit output must be a new child directory of the audit root")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "independent_b_control_calibration_audit_v1", "started_utc": datetime.now(timezone.utc).isoformat(),
               "report_path": str(args.report.resolve()), "auditor_sha256": digest(__file__),
               "independent_method": "Python standard library; no experiment imports or model execution",
               "limitations": ["Per-token energies are recorded measurements; this audit recomputes scenario aggregation and alpha, not model tensor operations.",
                               "Before/after checkpoint and module preservation are checked as hash-bound records; no independent live tensor snapshot is available.",
                               "Fit-before-holdout chronology follows the frozen source and fit marker; there are no independent per-case timestamps proving execution order.",
                               "Historical sources outside the programme may be hash-bound rather than committed; current bytes are checked against preregistration.",
                               "Calibration is not CPU/GPU qualification, generation evidence or a capability claim."]}
    code = 0
    try:
        receipt["results"] = audit(args.report)
        receipt["status"] = "PASS"
    except Exception as exc:
        receipt.update(status="FAIL", error={"type": type(exc).__name__, "message": str(exc)})
        code = 1
    receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), check_count=len(checks), checks=checks, input_sha256=inputs)
    with (output / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "receipt": str(output / "receipt.json"),
                      "error": receipt.get("error"), "results": receipt.get("results")}, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
