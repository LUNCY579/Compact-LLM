"""Independent standard-library audit of one immutable v2.4 calibration run.

No programme modules, tensor libraries, checkpoints, model or GPU are loaded.
Scalar evidence is recomputed from recorded per-scenario ordinary-token means.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT.parents[1]
RUN = ROOT / "native_armr/evidence/00_learned_control_calibration/20260906T014443_419834Z_b790e5e7"
FORMULA = "sqrt(mean_scenario_mean_ordinary_token_relative_A3_energy / mean_scenario_mean_ordinary_token_relative_M_energy)"
checks, inputs = [], {}


def check(condition, label):
    checks.append({"check": label, "pass": bool(condition)})
    if not condition:
        raise AssertionError(label)


def digest(path):
    path = Path(path).resolve()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    result = value.hexdigest()
    inputs[path.relative_to(REPO).as_posix()] = result
    return result


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    digest(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolved(relative):
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO) or path.relative_to(REPO).as_posix() != relative:
        raise ValueError("noncanonical repository-relative input")
    return path


def sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def state(value, label, parameters=None):
    check(sha(value["state_sha256_before"]) and value["state_sha256_before"] == value["state_sha256_after"], label + ": state before/after")
    check(value["buffer_sha256_before"] == value["buffer_sha256_after"] and all(sha(v) for v in value["buffer_sha256_before"].values()), label + ": buffer before/after")
    check(value["trainable_parameters"] == value["backbone_gradients"] == 0, label + ": zero trainable/gradient parameters")
    if parameters is not None:
        check(value["parameters"] == parameters, label + ": parameter inventory")


def main():
    started = time.perf_counter()
    result = {"audit_kind": "independent_standard_library_recorded_scalar_and_provenance_reconstruction",
              "run_directory": RUN.relative_to(REPO).as_posix(),
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "model_executed": False, "checkpoint_weights_read": False,
              "tensor_libraries_imported": False, "source_implementation_imported": False,
              "scope_limitations": [
                  "Per-scenario scalar means are input evidence. Raw hidden/native/control arrays are represented only by hashes; their per-token values and forward computations are not independently reproduced.",
                  "Module and checkpoint before/after digests are reconciled as recorded execution metadata; weights are not reread or rehashed by this audit.",
                  "Token counts and exclusions are reconstructed from recorded token IDs and special IDs; tokenizer encoding is not rerun.",
                  "Fit-before-holdout chronology is checked for consistency using file modification times and fit-only artifact contents, not an independently authenticated event log.",
                  "Git commitment is not checked here; execution admission separately requires committed artifacts. This audit does not license efficacy claims."]}
    try:
        report = read(RUN / "report.json")
        final, start = read(RUN / "run_manifest.json"), read(RUN / "started_manifest.json")
        registration = read(resolved(report["registration_path"]))
        contract = registration["contract"]
        config = read(resolved(report["calibration_config_path"]))
        check(registration["status"] == "FROZEN" and canonical(contract) == registration["registration_sha256"] == report["registration_sha256"], "registration canonical digest")
        check(config == contract["config"], "exact saved calibration config")
        check(config["formula"] == report["formula"] == FORMULA, "fixed scenario-balanced energy formula")
        check(report["control"] == config["control"] == {"kind": "pretrained_parallel_ffn", "insertion_layer": 13, "source_layers": [12, 13, 14], "summation_order": "(F12+F13)+F14", "local_frozen_copies": True}, "fixed three-branch parallel control")
        check(report["status"] == final["status"] == "COMPLETE" and start["status"] == "STARTED" and final["failure_receipt"] is None, "completed execution and failure metadata")
        check(final["report_sha256"] == digest(RUN / "report.json"), "final report byte binding")
        for key in ("branch", "commit_sha", "start_utc", "registration_sha256", "registration_path"):
            check(final[key] == start[key], "run identity: " + key)
        for label, receipt in (("start", start), ("final", final)):
            check(all(receipt.get(k) == v for k, v in contract.items()), label + " exact contract binding")
            check(receipt["track"] == "native_armr" and receipt["stage"] == "00_learned_control_calibration", label + " track/stage")
        check(report["source_files"] == contract["source_files"], "source-map registration binding")
        for relative, expected in report["source_files"].items():
            check(digest(resolved(relative)) == expected, "source bytes: " + relative)
        for prefix in ("calibration_config", "dataset_manifest", "calibration_data"):
            check(all(report[prefix + suffix] == contract[prefix + suffix] for suffix in ("_path", "_sha256")), prefix + " saved binding")
            check(digest(resolved(report[prefix + "_path"])) == report[prefix + "_sha256"], prefix + " actual bytes")
        manifest = read(resolved(report["dataset_manifest_path"]))
        self_digest = manifest.pop("manifest_sha256")
        check(canonical(manifest) == self_digest and manifest["track"] == "native_armr" and manifest["sealed_access"] is False, "frozen native unsealed manifest")
        info = manifest["datasets"]["calibration"]
        check(info["sha256"] == report["calibration_data_sha256"] and info["path"] == "calibration.jsonl", "calibration-only file binding")
        fields = ("example_id", "scenario_id", "family", "template_id", "prompt", "prompt_sha256", "alignment_partition")
        rows = []
        with resolved(report["calibration_data_path"]).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                full = json.loads(line)
                check((full["track"], full["split"], full["variant"]) == ("native_armr", "calibration", "original"), "calibration role: " + full["example_id"])
                rows.append({key: full[key] for key in fields})
                del full
        rows.sort(key=lambda row: row["example_id"])
        check(len(rows) == len({r["example_id"] for r in rows}) == len({r["scenario_id"] for r in rows}) == 128, "128 unique independent calibration scenarios")
        check(dict(Counter(r["family"] for r in rows)) == info["families"] and dict(Counter(r["template_id"] for r in rows)) == info["templates"], "frozen family/template coverage")
        expected_files = {canonical(row["example_id"]) + ".json" for row in rows}
        check(set(report["case_files"]) == expected_files == {p.name for p in (RUN / "cases").glob("*.json")}, "128 exact case files")
        special = report["special_control_token_ids"]
        check(special == sorted(set(special)) and all(type(i) is int and i >= 0 for i in special), "special/control token identity metadata")
        cases, mtimes = [], {"fit": [], "holdout": []}
        metric_keys = ("native_energy", "control_energy", "hidden_rms", "native_delta_rms", "control_delta_rms")
        for row in rows:
            identity = row["example_id"]
            check(canonical(row["prompt"]) == row["prompt_sha256"], "prompt bytes: " + identity)
            partition = "holdout" if int(row["scenario_id"].rsplit("_", 1)[1]) % 4 == 3 else "fit"
            check(row["alignment_partition"] == partition, "fixed scenario partition: " + identity)
            path = RUN / "cases" / (canonical(identity) + ".json")
            check(digest(path) == report["case_files"][path.name], "case bytes: " + identity)
            case = read(path)
            check(all(case[k] == row[k] for k in fields if k != "prompt"), "case identity fields: " + identity)
            tokens = case["token_ids"]
            check(all(type(t) is int and t >= 0 for t in tokens) and 0 < len(tokens) <= config["max_input_tokens"] and len(tokens) == case["full_prompt_tokens"] == case["retained_prompt_tokens"], "full token coverage: " + identity)
            ordinary = sum(t not in special for t in tokens)
            check(ordinary > 0, "ordinary-token presence: " + identity)
            counts = {"all": len(tokens), "ordinary": ordinary, "special_control": len(tokens) - ordinary, "final_prompt": 1}
            check(set(case["metrics"]) == set(counts), "token stratum coverage: " + identity)
            for stratum, count in counts.items():
                value = case["metrics"][stratum]
                check((value is None if count == 0 else value["token_count"] == count and set(value) == {*metric_keys, "token_count"} and all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] >= 0 for k in metric_keys)), "token count/finite metrics: " + identity + ": " + stratum)
            check(all(sha(case[k]) for k in ("embedding_rows_sha256", "hidden_sha256", "native_delta_sha256", "control_delta_sha256", "original_positions_sha256", "original_mask_sha256")) and len(case["original_rope_sha256"]) == 2 and all(sha(v) for v in case["original_rope_sha256"]), "tensor digest metadata: " + identity)
            cases.append(case)
            mtimes[partition].append(path.stat().st_mtime_ns)
        reconstructed = {}
        for partition, count in (("fit", 96), ("holdout", 32)):
            values = [c for c in cases if c["alignment_partition"] == partition]
            check(len(values) == count, partition + " scenario count")
            summary = {"scenario_ids": sorted(c["scenario_id"] for c in values), "example_ids": sorted(c["example_id"] for c in values), "scenarios": count, "ordinary_tokens": sum(c["metrics"]["ordinary"]["token_count"] for c in values), **{key: math.fsum(c["metrics"]["ordinary"][key] for c in values) / count for key in metric_keys}}
            check(summary["example_ids"] == contract[partition + "_example_ids"], partition + " saved ID allocation")
            reconstructed[partition] = summary
        check(not set(reconstructed["fit"]["scenario_ids"]) & set(reconstructed["holdout"]["scenario_ids"]), "fit/holdout disjointness")
        fit = reconstructed["fit"]
        check(fit["native_energy"] > 0 and fit["control_energy"] > 0, "positive independently aggregated fit energies")
        alpha = math.sqrt(fit["native_energy"] / fit["control_energy"])
        check(math.isfinite(alpha) and alpha > 0 and alpha == report["alpha"], "exact independently reconstructed alpha")
        scale = read(RUN / "fit_scale.json")
        check(digest(RUN / "fit_scale.json") == report["fit_scale_sha256"], "fit-scale byte binding")
        check(scale == {"alpha": alpha, "fit": fit, "formula": FORMULA}, "fit-only scale contents; no holdout input")
        for partition, summary in reconstructed.items():
            summary["scaled_control_energy"] = alpha ** 2 * summary["control_energy"]
            check(summary == report[partition], partition + " exact independent scalar summary")
        scale_time = (RUN / "fit_scale.json").stat().st_mtime_ns
        chronology = {"latest_fit_case_mtime_ns": max(mtimes["fit"]), "fit_scale_mtime_ns": scale_time, "earliest_holdout_case_mtime_ns": min(mtimes["holdout"])}
        check(max(mtimes["fit"]) <= scale_time < min(mtimes["holdout"]), "filesystem chronology consistent with fit scale preceding holdout residual records")
        layer_records = report["layer_state_receipts"]
        check([(r["stage"], r["layer"]) for r in layer_records] == [("ordinary_prefix", i) for i in range(14)] + [("replay", 12), ("replay", 13), ("control_source", 14)], "exact ordinary/replay/source decoder coverage")
        for index, value in enumerate(layer_records):
            state(value, "decoder receipt " + str(index))
        for i, j in ((12, 14), (13, 15)):
            check(layer_records[i]["state_sha256_before"] == layer_records[j]["state_sha256_before"] and layer_records[i]["buffer_sha256_before"] == layer_records[j]["buffer_sha256_before"], "ordinary/replayed decoder identity: " + str(i))
        state(report["rotary_state_receipt"], "rotary state", 0)
        state(report["control_state_receipt"], "composite control", 28314624)
        check(report["control_state_sha256"] == report["control_state_receipt"]["state_sha256_before"], "composite creation/current-state binding")
        inventory = {"matrix_parameters": 9 * 1024 * 3072, "norm_parameters": 3 * 1024, "resident_copied_parameters": 9 * 1024 * 3072 + 3 * 1024, "active_copied_parameters": 9 * 1024 * 3072 + 3 * 1024, "resident_bytes": (9 * 1024 * 3072 + 3 * 1024) * 4, "newly_learned_parameters": 0, "extra_matmul_flops_per_token": 18 * 1024 * 3072}
        check(inventory == report["parameter_inventory"] == config["parameter_counts"], "independent geometry/count/FP32 byte/FLOP inventory")
        for kind in ("branch_provenance", "source_hashes", "copy_hashes", "source_branch_receipts", "copy_branch_receipts"):
            check(set(report[kind]) == {"12", "13", "14"}, "complete three-branch metadata: " + kind)
        for layer in ("12", "13", "14"):
            provenance = report["branch_provenance"][layer]
            check((provenance["source_layer"], provenance["hidden_size"], provenance["intermediate_size"], provenance["norm_epsilon"], provenance["mlp_class"], provenance["norm_class"], provenance["activation_class"]) == (int(layer), 1024, 3072, 1e-6, "Qwen3MLP", "Qwen3RMSNorm", "SiLUActivation"), "canonical branch geometry/norm/activation: " + layer)
            for field, short, count in (("mlp", "mlp", 3 * 1024 * 3072), ("post_attention_layernorm", "norm", 1024)):
                digests = []
                for kind in ("source", "copy"):
                    branch = report[kind + "_branch_receipts"][layer]
                    check(branch["source_layer"] == int(layer), kind + " source layer identity: " + layer)
                    state(branch[field], kind + " " + layer + " " + field, count)
                    digests.extend((branch[field]["state_sha256_before"], report[kind + "_hashes"][layer][field], provenance[kind + "_" + short + "_sha256"]))
                check(len(set(digests)) == 1, "source/copy and provenance identity: " + layer + " " + field)
        reference = read(REPO / "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json")
        check(report["checkpoint_files_before"] == report["checkpoint_files_after"] == reference["checkpoint_files_before"], "checkpoint before/after/historical identity metadata")
        check(report["checkpoint_files_before"]["model.safetensors"] == config["checkpoint_sha256"] and report["checkpoint_files_before"]["config.json"] == config["model_config_sha256"] == report["model_config_sha256"] and report["qwen_implementation_sha256"] == config["qwen_implementation_sha256"], "registered checkpoint/model-config/HF identity")
        check(report["runtime_identity"] == {"device": "cpu", "dtype": "float32", "threads": 4, "seed": 17, "torch": "2.12.1+cu126", "transformers": "5.14.1", "attention": "eager", "use_cache": False}, "registered execution runtime metadata")
        check(report["label_free"] is True and report["answer_labels_used"] is False and report["generated_tokens"] == 0 and report["holdout_selects_alpha"] is False and report["alpha_search"] is False and report["alpha_clipping"] is False and report["backbone_gradients"] == report["trainable_parameters"] == 0 and report["canonical_unchanged"] is True, "label-free/frozen/no-search execution metadata")
        check(report["peak_rss_bytes"] <= config["max_rss_gib"] * 2 ** 30 and report["wall_seconds"] <= config["max_stage_seconds"], "recorded execution resource bounds")
        result.update(status="PASS", alpha=alpha, reconstructed=reconstructed, chronology=chronology,
                      registration_sha256=report["registration_sha256"], control_state_sha256=report["control_state_sha256"], parameter_inventory=inventory,
                      source_files_checked=len(report["source_files"]), case_files_checked=len(cases))
    except BaseException as exc:
        result.update(status="FAILED", failure={"type": type(exc).__name__, "message": str(exc)})
    result.update(checks=checks, checks_passed=sum(c["pass"] for c in checks), input_sha256=inputs,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), wall_seconds=time.perf_counter() - started)
    with (OUT / "receipt.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result.get(key) for key in ("status", "checks_passed", "alpha", "source_files_checked", "case_files_checked", "wall_seconds", "failure")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
