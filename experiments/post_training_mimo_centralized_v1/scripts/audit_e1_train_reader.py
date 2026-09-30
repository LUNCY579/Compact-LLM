"""Committed actual scheduled TRAIN reader audit; never a model/training run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
PREFIX = "experiments/post_training_mimo_centralized_v1/"
CONFIG_RELATIVE = PREFIX + "configs/E1_TRAINING_COMPONENTS_V1.json"
CONFIG_SHA256 = "d40669540100f0208e67f856fc70ffbe5e08fbf0d782194a84399a6a8f797f20"
PRODUCER_RELATIVE = PREFIX + "scripts/audit_e1_train_reader.py"
SCOPE_RELATIVE = PREFIX + "scripts/train_scope.py"
SCOPE_SHA256 = "7d24b694571a65855021ad5e68ff6c1c66a4e8c278077dc9f8d5241653fbc3ef"
RECEIPT_RELATIVE = PREFIX + "receipts/train_scope_v1/TRAIN_SCOPE_RECEIPT.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def pin(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": Path(path).stat().st_size}


def write(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def source_snapshot(cfg):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True, timeout=15).strip(), "commit all source before production audit")
    expected = dict(cfg["source_pins"])
    for relative in (CONFIG_RELATIVE, PRODUCER_RELATIVE):
        expected[relative] = pin(ROOT / relative)
    for relative, wanted in expected.items():
        require(pin(ROOT / relative) == wanted, "audit component source changed:" + relative)
        committed = subprocess.run(["git", "show", head + ":" + relative], cwd=ROOT,
            capture_output=True, check=False, timeout=15)
        require(committed.returncode == 0 and hashlib.sha256(committed.stdout).hexdigest() == wanted["sha256"]
            and len(committed.stdout) == wanted["bytes"], "audit source not committed:" + relative)
    return {"producer_commit": head, "files": expected}


def recheck_sources(snapshot):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15).strip()
            == snapshot["producer_commit"], "audit HEAD changed")
    require(all(pin(ROOT / name) == wanted for name, wanted in snapshot["files"].items()), "audit component source changed")


def load_scope_budget():
    path = ROOT / SCOPE_RELATIVE
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SCOPE_SHA256, "pinned scope budget source changed")
    spec = importlib.util.spec_from_file_location("_e1_reader_audit_scope", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module.Budget()


def audit(config_path, output):
    require(config_path.resolve() == (ROOT / CONFIG_RELATIVE).resolve(), "registered audit config required")
    raw = config_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "audit registration changed")
    cfg = json.loads(raw)
    require(Path(sys.executable).resolve() == Path(cfg["cpu_audit"]["python_executable"]).resolve()
            and list(sys.version_info[:3]) == cfg["cpu_audit"]["python_version"], "historical shuffle requires pinned Python3.11.15")
    snapshot = source_snapshot(cfg)
    require(not output.exists(), "reader audit output must be new")
    output.mkdir(parents=True)
    write(output / "CONFIG.json", cfg)
    write(output / "SOURCE_RECEIPT.json", snapshot)
    budget = load_scope_budget()
    result = {"status": "RUNNING_ACTUAL_READER_AUDIT", "config_sha256": CONFIG_SHA256,
        "producer_commit": snapshot["producer_commit"], "runtime_python": sys.version,
        "runtime_executable": sys.executable, "scheduled_rows_returned": 0,
        "scheduled_nonpadding_positions": 0, "scheduled_supervised_positions": 0,
        "model_calls": 0, "training_updates": 0, "generation_tokens": 0, "teacher_calls": 0,
        "native_qualified": False, "native_backend_connected": False, "native_launch_authorized": False,
        "native_resume_qualified": False, "E1_authorized": False, "CHECK_FINAL_access": False,
        "resource_check_stride_rows": 32,
        "resource_enforcement": "cooperative OS high-water process RSS and wall checks; no hard CPU process containment"}
    trace = hashlib.sha256()
    try:
        budget.check()
        from experiments.post_training_mimo_centralized_v1.scripts.e1_train_reader import BoundTrainReader
        plan_rel = "artifacts/mimo_central/e1_plan_seed101_v2/PLAN.json"
        binding_rel = "artifacts/mimo_central/e1_plan_seed101_v2/DATA_BINDING.json"
        with BoundTrainReader.open_production(ROOT, plan_relative=plan_rel, plan_pin=cfg["metadata_pins"][plan_rel],
                binding_relative=binding_rel, binding_pin=cfg["metadata_pins"][binding_rel],
                scope_receipt_relative=RECEIPT_RELATIVE, scope_receipt_pin=cfg["metadata_pins"][RECEIPT_RELATIVE]) as reader:
            budget.check()
            before = reader.receipt()
            require(before["fixture_only"] is False and before["plan_sha256"] == cfg["historical_plan_sha256"], "production plan/reader binding differs")
            require(pin(ROOT / plan_rel) == cfg["metadata_pins"][plan_rel], "frozen schedule bytes changed")
            independent_plan = json.loads((ROOT / plan_rel).read_bytes())
            expected_keys = [key for step in independent_plan["steps"] for key in step["row_keys"]]
            del independent_plan
            component_rows = dict.fromkeys(("acquisition", "deterministic_replay", "language_replay"), 0)
            prior = 0
            with (output / "SCHEDULED_ROWS.jsonl").open("xb") as stream:
                for item in reader.iter_plan():
                    row, schedule = item["row"], item["schedule"]
                    index = result["scheduled_rows_returned"]
                    result["scheduled_rows_returned"] += 1
                    result["scheduled_nonpadding_positions"] += row["nonpadding_tokens"]
                    result["scheduled_supervised_positions"] += row["completion_tokens"]
                    # Count the returned row before any later guard/IO failure.
                    # High-water sampling every32 rows catches earlier peaks;
                    # this is cooperative validation, not hard containment.
                    if index % 32 == 0:
                        budget.check()
                    require(schedule["sequence_index"] == index and schedule["row_key"] == expected_keys[index], "scheduled lookup order drift")
                    require(schedule["optimizer_step"] == index // 20 + 1 and schedule["microbatch_index"] == index % 20 + 1
                        and schedule["sequences_in_step"] == 20, "optimizer/microbatch schedule drift")
                    component_rows[schedule["component"]] += 1
                    prior += row.get("prior_assistant_turns", 0)
                    record = {"sequence_index": index, "optimizer_step": schedule["optimizer_step"],
                        "microbatch_index": schedule["microbatch_index"], "row_key": schedule["row_key"],
                        "component": schedule["component"], "learning_rate": schedule["learning_rate"],
                        "nonpadding_tokens": row["nonpadding_tokens"], "supervised_tokens": row["completion_tokens"],
                        "input_token_ids_sha256": row["input_token_ids_sha256"],
                        "labels_sha256": hashlib.sha256(canonical(row["labels"])).hexdigest(),
                        "source_row_sha256": row["source_row_sha256"]}
                    encoded = canonical(record) + b"\n"
                    stream.write(encoded); trace.update(encoded)
                    del item, row, schedule, record, encoded
            reader.verify_unchanged(full_hash=True)
            result["reader_receipt"] = reader.receipt()
            result["component_rows"] = component_rows
            result["language_prior_assistant_turns_masked"] = prior
            require(component_rows == {"acquisition": 4096, "deterministic_replay": 768, "language_replay": 256}, "full component dose changed")
            require((result["scheduled_rows_returned"], result["scheduled_nonpadding_positions"], result["scheduled_supervised_positions"], prior)
                == (5120, 680532, 63546, 81), "scheduled consumer dose differs")
            require(result["reader_receipt"]["totals"]["maximum_sequence_tokens"] == 398, "full targets no longer fit")
        require(not set(("torch", "transformers", "tokenizers", "safetensors")).intersection(sys.modules), "unexpected framework/tokenizer import")
        recheck_sources(snapshot)
        budget.check()
        result["source_files_unchanged"] = True
        result["scheduled_trace_sha256"] = trace.hexdigest()
        result["scheduled_trace_file"] = pin(output / "SCHEDULED_ROWS.jsonl")
        result["status"] = "PASS_REGISTERED_SCHEDULED_TRAIN_READER_DATA_ONLY"
    except Exception as error:
        result["status"] = "FAILED_ACTUAL_READER_AUDIT_DATA_ONLY"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        result["partial_trace_sha256"] = trace.hexdigest()
        write(output / "FAILURE_RECEIPT.json", result)
    finally:
        result["wall_seconds"] = time.monotonic() - budget.started
        result["peak_observed_process_RSS_bytes"] = budget.peak_rss_bytes
        write(output / "READER_AUDIT_RESULT.json", result)
        print(json.dumps({name: result[name] for name in ("status", "scheduled_rows_returned", "model_calls", "wall_seconds", "peak_observed_process_RSS_bytes")}), flush=True)
    return 0 if result["status"] == "PASS_REGISTERED_SCHEDULED_TRAIN_READER_DATA_ONLY" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return audit(args.config, args.output.resolve())


if __name__ == "__main__":
    sys.exit(main())
