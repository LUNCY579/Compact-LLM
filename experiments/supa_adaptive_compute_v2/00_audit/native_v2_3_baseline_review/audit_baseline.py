"""Independent, score-blind A0 evidence audit; standard library only.

Run from any directory. Incomplete coverage prints an inventory and exits 2
without creating a receipt. --write creates receipt.json and audit.md exclusively.
No model imports, checkpoints, answer oracle, scorer, or Git commands are used.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / "experiments/supa_adaptive_compute_v2"
OUTPUT = Path(__file__).resolve().parent
REGISTRATIONS = {
    "v2.2": "7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5",
    "v2.3": "86e768e08bf3576dd8ce80ec939b7e876fb5fb7449d41a3aa51c497ce6607b44",
}
META = ("example_id", "scenario_id", "family", "track", "split", "variant",
        "template_id", "template_unseen_in_engineering")
RAW = (*META, "arm", "token_ids", "gold_nll", "safety_stopped")
CHECKPOINT = "a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026"
STATE = "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"
QWEN = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
checks = Counter()
file_hashes = {}


def require(condition, category, detail):
    if not condition:
        raise AssertionError(f"{category}: {detail}")
    checks[category] += 1


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def sha(path):
    path = Path(path).resolve()
    key = relative(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    value = digest.hexdigest()
    require(key not in file_hashes or file_hashes[key] == value,
            "read_stability", key)
    file_hashes[key] = value
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def projected(text, allowed):
    # Answers, responses and scores are discarded rather than returned to callers.
    return json.loads(text, object_pairs_hook=lambda pairs:
                      {key: value for key, value in pairs if key in allowed})


def compact_json(text):
    """Remove JSON whitespace lexically, preserving all opaque scalar bytes."""
    output = []
    quoted = escaped = False
    for char in text:
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            output.append(char)
        elif char not in " \t\r\n":
            output.append(char)
    require(not quoted, "json_structure", "unterminated string")
    return "".join(output)


def raw_array_objects(text):
    """Split serialized row objects without decoding their answer/score fields."""
    compact = compact_json(text)
    require(compact.startswith("[") and compact.endswith("]"),
            "json_structure", "results must be an array")
    rows = []
    depth = 0
    quoted = escaped = False
    start = None
    for index, char in enumerate(compact[1:-1], 1):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            if depth == 0:
                require(char == "{", "json_structure", "row must be an object")
                start = index
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth < 0:
                raise AssertionError("json_structure: negative nesting")
            if depth == 0:
                rows.append(compact[start:index + 1])
        else:
            if depth == 0 and char != ",":
                raise AssertionError("json_structure: unexpected array text")
    require(depth == 0 and not quoted, "json_structure", "unclosed array object")
    return rows


def read_expected():
    manifest_path = ROOT / "data/generated/v2/native_armr/manifest.json"
    manifest = load(manifest_path)
    copy = dict(manifest)
    claimed = copy.pop("manifest_sha256")
    require(canonical(copy) == claimed, "dataset", "manifest canonical digest")
    require(manifest["track"] == "native_armr" and manifest["sealed_access"] is False,
            "dataset", "track/sealed identity")
    role = manifest["datasets"]["architecture_dev"]
    bank = manifest_path.parent / role["path"]
    require(bank.resolve().parent == manifest_path.parent.resolve(), "dataset", "bank containment")
    require(sha(bank) == role["sha256"], "dataset", "frozen architecture bank digest")
    expected = [projected(line, META) for line in bank.read_text(encoding="utf-8").splitlines() if line]
    require(len(expected) == role["examples"] == 128, "dataset", "128 expected rows")
    for row in expected:
        require(set(row) == set(META), "dataset", "all permitted metadata present")
        require(row["track"] == "native_armr" and row["split"] == "architecture_dev"
                and row["variant"] == "original", "dataset", "fixed architecture role")
    require(len({row["example_id"] for row in expected}) == 128, "dataset", "unique IDs")
    require(len({row["scenario_id"] for row in expected}) == 128, "dataset", "unique scenarios")
    require(dict(Counter(row["family"] for row in expected)) == role["families"],
            "dataset", "family coverage")
    return expected, manifest_path, sha(manifest_path)


def inventory():
    result = {version: [] for version in REGISTRATIONS}
    excluded = {version: [] for version in REGISTRATIONS}
    for path in sorted((ROOT / "native_armr/evidence/01_architecture_dev").glob("*/run_manifest.json")):
        manifest = load(path)
        if manifest.get("arm") != "A0":
            continue
        for version, digest in REGISTRATIONS.items():
            if manifest.get("registration_sha256") != digest:
                continue
            entry = (path, manifest)
            if manifest.get("status") == "COMPLETE":
                result[version].append(entry)
            else:
                excluded[version].append({"path": relative(path), "status": manifest.get("status"),
                                          "sha256": sha(path)})
    return result, excluded


def qualification(path, registration, dataset_sha):
    final_path = path / "run_manifest.json"
    final = load(final_path)
    report_path = path / "report.json"
    report = load(report_path)
    require(final["status"] == "COMPLETE" and final["qualification_pass"] is True
            and final["canonical_unchanged"] is True and final["failure_receipt"] is None,
            "qualification", "successful final receipt")
    require(final["registration_sha256"] == registration["registration_sha256"]
            and final["sources"] == registration["sources"]
            and final["config_sha256"] == canonical(registration["config"])
            and final["dataset_manifest_sha256"] == dataset_sha,
            "qualification", "source/config/data registration binding")
    require(sha(report_path) == final["report_sha256"]
            and report["qualification_pass"] is True
            and report["same_device_disabled_restored_identity"] is True,
            "qualification", "report digest and declared pass")
    require(report["thresholds"] == registration["config"]["qualification"],
            "qualification", "frozen thresholds")
    require(len(report["comparisons"]) == 40 and all(row["pass"] is True
            and all(value is True for value in row["gates"].values())
            for row in report["comparisons"]), "qualification", "all 40 pass receipts")
    require(len(report["artifacts"]) == 80, "qualification", "80 saved logit artifacts")
    for name, digest in report["artifacts"].items():
        artifact = (path / name).resolve()
        require(artifact.is_relative_to(path.resolve()), "qualification", "artifact containment")
        require(sha(artifact) == digest, "qualification", "artifact digest")
    return sha(final_path), report["gpu_identity"]


def inspect_version(version, runs, expected, dataset_path, dataset_sha):
    reg_path = ROOT / "native_armr/evidence/00_preregistration" / REGISTRATIONS[version] / "registration.json"
    registration = load(reg_path)
    copy = {key: registration[key] for key in
            ("track", "config_sha256", "config_path", "config", "sources", "dataset_manifests")}
    claimed = registration["registration_sha256"]
    require(canonical(copy) == claimed == REGISTRATIONS[version], "registration", "canonical digest")
    require(registration["status"] == "FROZEN" and registration["track"] == "native_armr",
            "registration", "frozen track")
    require(registration["dataset_manifests"][relative(dataset_path)] == dataset_sha,
            "registration", "dataset binding")
    config = registration["config"]
    require(sha(REPO / registration["config_path"]) == registration["config_sha256"],
            "registration", "config file bytes")
    require(load(REPO / registration["config_path"]) == config, "registration", "config contents")
    sha(reg_path)
    if version == "v2.3":
        for name, digest in registration["sources"].items():
            require(sha(REPO / name) == digest, "current_sources", name)

    expected_by_id = {row["example_id"]: row for row in expected}
    shards = {index: [row["example_id"] for row in expected
                     if int(canonical(row["scenario_id"]), 16) % 32 == index]
              for index in range(32)}
    populated = {index: ids for index, ids in shards.items() if ids}
    require(len(populated) == 30 and set(shards) - set(populated) == {15, 23},
            "coverage", "30 populated of 32 deterministic shards")
    require(len(runs) == 30 and {v["shard_index"] for _, v in runs} == set(populated),
            "coverage", f"{version} complete shard set")
    qualification_dirs = {v["runtime_qualification_directory"] for _, v in runs}
    require(len(qualification_dirs) == 1, "qualification", "one qualification per version")
    qpath = Path(next(iter(qualification_dirs))).resolve()
    require(qpath.parent == (ROOT / "native_armr/evidence/00_runtime_qualification").resolve(),
            "qualification", "qualification containment")
    qdigest, identity = qualification(qpath, registration, dataset_sha)
    rows_by_id, logits_by_id, manifest_records = {}, {}, []
    identity_keys = ("model_path", "model_class", "torch", "transformers", "parameters",
                     "qwen_implementation_sha256", "state_sha256_before", "checkpoint_files_before",
                     "runtime_identity", "trainable_parameters", "backbone_gradients")
    for path, manifest in runs:
        directory = path.parent
        name = relative(path)
        require(manifest["track"] == "native_armr" and manifest["arm"] == "A0"
                and manifest["role"] == "architecture_dev" and manifest["stage"] == "01_architecture_dev",
                "run_contract", name)
        require(manifest["branch"] == "codex/supa-adaptive-compute-v2"
                and len(manifest["commit_sha"]) == 40
                and all(char in "0123456789abcdef" for char in manifest["commit_sha"]),
                "run_contract", "recorded branch and commit identity")
        require(Path(manifest["config_path"]).resolve() == (REPO / registration["config_path"]).resolve()
                and Path(manifest["dataset_manifest"]).resolve() == dataset_path.resolve(),
                "run_contract", "canonical config and dataset paths")
        require(manifest["registration_sha256"] == claimed
                and manifest["sources"] == registration["sources"]
                and manifest["config_sha256"] == canonical(config)
                and manifest["dataset_manifest_sha256"] == dataset_sha,
                "run_contract", "registration/source/config/data identity")
        require(manifest["shard_count"] == 32
                and manifest["example_ids"] == populated[manifest["shard_index"]],
                "coverage", "deterministic ordered shard ownership")
        require(manifest["runtime_qualification_receipt_sha256"] == qdigest,
                "qualification", "run bound to qualification receipt")
        require(all(manifest[key] == identity[key] for key in identity_keys),
                "model_identity", "identical to qualified GPU model/runtime")
        require(manifest["canonical_unchanged"] is True
                and manifest["state_sha256_before"] == manifest["state_sha256_after"] == STATE
                and manifest["checkpoint_files_before"] == manifest["checkpoint_files_after"]
                and manifest["checkpoint_files_before"]["model.safetensors"] == CHECKPOINT
                and manifest["qwen_implementation_sha256"] == QWEN,
                "model_identity", "canonical hashes preserved")
        require(manifest["model_class"] == "Qwen3AbstainForCausalLM"
                and manifest["parameters"] == 596049920
                and manifest["trainable_parameters"] == manifest["backbone_gradients"] == 0,
                "model_identity", "canonical frozen class and parameter count")
        require(all(manifest[key] == config[key] for key in ("seed", "threads", "dtype", "device", "max_prompt_seconds"))
                and manifest["torch"] == "2.12.1+cu126" and manifest["transformers"] == "5.14.1"
                and manifest["cache_mode"] == "full_prefix_no_cache", "runtime", "frozen execution settings")
        runtime = manifest["runtime_identity"]
        require(runtime["attention"] == "eager" and runtime["use_cache"] is False
                and runtime["tf32_matmul"] is False and runtime["tf32_cudnn"] is False
                and runtime["deterministic_algorithms"] is True
                and runtime["matmul_precision"] == "highest"
                and runtime["cublas_workspace_config"] == ":4096:8", "runtime", "FP32 deterministic eager backend")
        require(manifest["operator"] == {"added_active_parameters": 0, "added_parameter_count": 0,
                                         "extra_matmul_flops_per_token": 0}
                and manifest["baseline_run_directory"] is None and manifest["baseline_manifest_sha256"] is None,
                "run_contract", "intact baseline without extra operator")
        require(manifest["status"] == "COMPLETE" and manifest["failure_receipt"] is None,
                "run_contract", "complete failure-free receipt")
        require(datetime.fromisoformat(manifest["end_utc"]) >= datetime.fromisoformat(manifest["start_utc"])
                and math.isfinite(manifest["wall_seconds"]) and manifest["wall_seconds"] > 0,
                "runtime", "finite timing and chronological endpoints")
        require(0 < manifest["peak_memory_bytes"] <= config["max_rss_gib"] * 1024**3
                and 0 < manifest["peak_cuda_reserved_bytes"] <= config["max_cuda_memory_gib"] * 1024**3,
                "runtime", "recorded memory within bounds")
        initial_identity = load(directory / "model_identity.json")
        cuda_memory_fields = {"cuda_allocated_bytes", "cuda_reserved_bytes",
                              "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}
        require(all(manifest[key] == value for key, value in initial_identity.items()
                    if key not in cuda_memory_fields),
                "model_identity", "model_identity file reconciles with final receipt")
        require(all(0 <= initial_identity[key] <= manifest[key]
                    for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")),
                "runtime", "peak CUDA memory monotonically includes loading")
        sha(directory / "model_identity.json")
        started = load(directory / "started_manifest.json")
        require(started["status"] == "STARTED"
                and all(manifest[key] == value for key, value in started.items() if key != "status"),
                "run_contract", "initial contract reconciles with final receipt")
        sha(directory / "started_manifest.json")
        sha(path)
        result_path = directory / "results.json"
        require(sha(result_path) == manifest["results_sha256"], "results_hash", relative(result_path))
        raw_rows = raw_array_objects(result_path.read_text(encoding="utf-8"))
        rows = [projected(raw, RAW) for raw in raw_rows]
        require([row["example_id"] for row in rows] == manifest["example_ids"],
                "coverage", "result order equals manifest")
        expected_names = {canonical(identifier) for identifier in manifest["example_ids"]}
        require({p.name for p in (directory / "cases").iterdir()} == {name + ".json" for name in expected_names},
                "artifact_inventory", "exact case file inventory")
        require({p.name for p in (directory / "logits").iterdir()} == {name + ".pt" for name in expected_names}
                and set(manifest["logits_sha256"]) == {name + ".pt" for name in expected_names},
                "artifact_inventory", "exact logit file inventory")
        for raw, row in zip(raw_rows, rows):
            identifier = row["example_id"]
            require(identifier not in rows_by_id and identifier in expected_by_id,
                    "coverage", "unique expected example")
            require({key: row[key] for key in META} == expected_by_id[identifier]
                    and row["arm"] == "A0", "metadata", identifier)
            require(row["safety_stopped"] is False, "validity_receipt", "no safety stop")
            require(isinstance(row["token_ids"], list) and row["token_ids"]
                    and all(type(token) is int and token >= 0 for token in row["token_ids"])
                    and type(row["gold_nll"]) in (int, float) and math.isfinite(row["gold_nll"]),
                    "raw_outputs", "finite NLL and integer token IDs")
            stem = canonical(identifier)
            case_path = directory / "cases" / (stem + ".json")
            require(compact_json(case_path.read_text(encoding="utf-8")) == raw,
                    "case_results_reconciliation", "opaque full case JSON equals hash-bound result row")
            sha(case_path)
            logit_path = directory / "logits" / (stem + ".pt")
            digest = sha(logit_path)
            require(digest == manifest["logits_sha256"][logit_path.name], "logit_hash", identifier)
            rows_by_id[identifier] = row
            logits_by_id[identifier] = digest
        manifest_records.append({"path": name, "sha256": file_hashes[name],
                                 "shard_index": manifest["shard_index"], "n": len(rows),
                                 "commit_sha": manifest["commit_sha"]})
    require(set(rows_by_id) == set(expected_by_id), "coverage", "full 128-example baseline")
    return rows_by_id, logits_by_id, {
        "registration_sha256": claimed, "complete_cases": len(rows_by_id), "complete_shards": len(runs),
        "qualified_runtime": identity["runtime_identity"], "model_state_sha256": STATE,
        "checkpoint_files": identity["checkpoint_files_before"], "qualification_directory": relative(qpath),
        "qualification_receipt_sha256": qdigest, "runs": sorted(manifest_records, key=lambda item: item["shard_index"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    expected, dataset_path, dataset_sha = read_expected()
    runs, excluded = inventory()
    counts = {version: {"complete_cases": sum(len(value["example_ids"]) for _, value in entries),
                        "complete_shards": len(entries)} for version, entries in runs.items()}
    if any(item != {"complete_cases": 128, "complete_shards": 30} for item in counts.values()):
        print(json.dumps({"status": "INCOMPLETE", "inventory": counts, "receipt_written": False}))
        return 2
    inspected = {version: inspect_version(version, runs[version], expected, dataset_path, dataset_sha)
                 for version in REGISTRATIONS}
    old_rows, old_logits, old_summary = inspected["v2.2"]
    new_rows, new_logits, new_summary = inspected["v2.3"]
    differences = {key: [] for key in ("token_ids", "gold_nll", "logit_file_bytes")}
    nll_max_abs_difference = 0.0
    for identifier in old_rows:
        for key in ("token_ids", "gold_nll"):
            if old_rows[identifier][key] != new_rows[identifier][key]:
                differences[key].append(identifier)
        nll_max_abs_difference = max(nll_max_abs_difference,
                                     abs(old_rows[identifier]["gold_nll"] - new_rows[identifier]["gold_nll"]))
        if old_logits[identifier] != new_logits[identifier]:
            differences["logit_file_bytes"].append(identifier)
    require(old_summary["qualified_runtime"] == new_summary["qualified_runtime"]
            and old_summary["checkpoint_files"] == new_summary["checkpoint_files"],
            "cross_version_identity", "same runtime and canonical checkpoint receipts")
    sha(Path(__file__))
    comparison = {"n": 128, "identical_counts": {key: 128 - len(value) for key, value in differences.items()},
                  "difference_example_ids": differences, "maximum_absolute_nll_difference": nll_max_abs_difference}
    receipt = {
        "status": "PASS", "scope": "Complete A0 architecture baseline evidence and raw-output reproducibility only",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "wall_seconds": time.perf_counter() - start, "checks_passed": sum(checks.values()),
        "checks_by_category": dict(checks), "dataset_manifest_sha256": dataset_sha,
        "baselines": {"v2.2": old_summary, "v2.3": new_summary}, "comparison": comparison,
        "excluded_noncomplete_receipts": excluded, "file_sha256": file_hashes,
        "boundaries": {"scoring_or_oracle_imported": False, "accuracy_computed": False,
                       "response_or_answer_fields_consulted": False, "torch_or_numpy_imported": False,
                       "model_or_checkpoint_loaded": False, "gpu_used": False,
                       "confirmation_bank_opened": False, "cross_scale_data_opened": False,
                       "git_used": False, "old_evidence_modified": False},
        "limitations": [
            "This is a saved-evidence audit, not a fresh model forward or checkpoint byte reread.",
            "State preservation and runtime facts are reconciled execution receipts; no live tensors are inspected.",
            "Every case SHA is newly inventoried; case content is bound indirectly by exact opaque JSON equality to its SHA-bound results row.",
            "Logit archives are compared bytewise by SHA256, not deserialized or numerically reconstructed.",
            "All current v2.3 registered source files are rehashed; v2.2 source identity is checked against its preserved registration, not historical Git blobs.",
            "Qualification report/artifact hashes and all 40 pass/gate receipts are verified; qualification metrics are not recomputed here.",
            "A0 reproducibility neither qualifies another arm nor completes the five-arm architecture gate or establishes capability.",
            "Bank/results JSON parsing discards unpermitted fields; opaque byte reconciliation includes their serialized bytes without semantic consultation.",
        ],
    }
    if args.write:
        for name in ("receipt.json", "audit.md"):
            if (OUTPUT / name).exists():
                raise FileExistsError(f"refusing to replace {name}")
        note = (
            "# Independent v2.3 A0 baseline review\n\n"
            "Both native GPU versions have all 128 architecture cases across the same 30 populated "
            "deterministic shards; shard indices 15 and 23 are empty in the frozen bank. "
            "All expected IDs are unique, metadata and shard order match the frozen bank, and every "
            "results, case and saved logit artifact was hashed and reconciled.\n\n"
            f"Raw comparison: {comparison['identical_counts']['token_ids']}/128 token ID sequences, "
            f"{comparison['identical_counts']['gold_nll']}/128 saved NLL values, and "
            f"{comparison['identical_counts']['logit_file_bytes']}/128 logit archives are identical. "
            f"Maximum absolute NLL difference: {nll_max_abs_difference}. "
            "Difference IDs, if present, are listed in receipt.json. No accuracy or candidate-selection analysis was performed.\n\n"
            "Each version binds its own configuration, registration, sources and successful GPU qualification. "
            "Canonical class, checkpoint/state identities, zero trainable parameters/gradients, before/after "
            "preservation receipts and FP32 eager deterministic runtime match throughout and across versions. "
            "The current v2.3 source bytes were rehashed. Original failed receipts remain excluded and preserved.\n\n"
            f"Validation: {receipt['checks_passed']} checks; {len(file_hashes)} distinct files hashed. "
            "Run `python audit_baseline.py` for a read-only repeat, or `python audit_baseline.py --write` "
            "only when the exclusive outputs do not exist. The script uses only the standard library.\n\n"
            "Limits: this audit checks saved evidence, not fresh model execution or live checkpoint bytes. "
            "Case files have no independent original manifest SHA, so full opaque JSON equality binds them "
            "to the SHA-verified results. Logit archive bytes are compared without tensor deserialization. "
            "Qualification metric values and historical Git source blobs were not recomputed. "
            "Bank and result field projection discards answers/responses/scores; these are only traversed as "
            "opaque serialized bytes for integrity checks. Confirmation and Track B data remain unopened. "
            "This baseline audit does not open the five-arm architecture or capability gate.\n"
        )
        with (OUTPUT / "receipt.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        with (OUTPUT / "audit.md").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(note)
    print(json.dumps({"status": receipt["status"], "checks_passed": receipt["checks_passed"],
                      "files_hashed": len(file_hashes), "comparison": comparison,
                      "receipt_written": args.write, "wall_seconds": receipt["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
