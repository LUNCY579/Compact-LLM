"""Read-only, stdlib audit of the completed disposable two-update qualification.

Only the original control's checkpoint-00002 tensor payloads are permitted.
No model runtime, full base weights, active checkpoint or later active step is
read. The actual disposable tensors no longer exist: this audit recomputes all
reference component hashes and checks the recorded actual hashes against them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

BASE = Path("C:/Users/clclc/BusinessProject")
ROOT = BASE / "rineya-dataset-retention-records/binding-integration001-20260920"
HERE = Path(__file__).resolve().parent
CONTROL = BASE / "rineya-dataset-retention-records/highdose-lora001-resume-20260920"
REFERENCE = CONTROL / "journal/runs/train-001/checkpoint-00002"
REFERENCE_MANIFEST_SHA = "afc09301260af14ea033f42e4ccb006982f8a36f7b31db4807d521e7399c5376"


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(path):
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "missing/nonregular audit input: " + str(path))
    if path.suffix == ".safetensors":
        require(path.parent == REFERENCE.resolve() and re.fullmatch(r"tensor-\d{4}\.safetensors", path.name),
                "only original control checkpoint-00002 tensor payloads may be read")
    else:
        require(path.suffix.lower() in (".py", ".json", ".jsonl", ".md", ".txt"), "unregistered audit input type")
    return path


def file_sha(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_components(path, name, expected_shape):
    """Hash exact F32 payload bytes using the qualified trainer's component salt."""
    path = regular(path)
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "truncated safetensor length")
        header_length = struct.unpack("<Q", length_bytes)[0]
        require(0 < header_length <= 1024**2, "safetensor header exceeds audit bound")
        header = json.loads(stream.read(header_length).decode("utf-8"))
        require(set(header) == {"parameter", "exp_avg", "exp_avg_sq"}, "reference tensor component inventory differs")
        data_start = 8 + header_length; ranges = []; result = {}
        for component in ("parameter", "exp_avg", "exp_avg_sq"):
            value = header[component]; lo, hi = value["data_offsets"]
            require(value["dtype"] == "F32" and value["shape"] == expected_shape and
                    type(lo) is int and type(hi) is int and 0 <= lo < hi and
                    hi - lo == 4 * math.prod(expected_shape), "reference component dtype/shape/length differs")
            stream.seek(data_start + lo)
            hashed = hashlib.sha256(canonical([name, component, expected_shape, "torch.float32"]))
            remaining = hi - lo
            while remaining:
                chunk = stream.read(min(1024**2, remaining))
                require(bool(chunk), "reference tensor payload truncated")
                hashed.update(chunk); remaining -= len(chunk)
            result[component] = hashed.hexdigest(); ranges.append((lo, hi))
        ranges.sort()
        require(ranges[0][0] == 0 and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])) and
                data_start + ranges[-1][1] == path.stat().st_size, "reference payload has gaps, overlaps or trailing bytes")
    return result


def first_line(path):
    """Read the one permitted stable prefix line; never consume later steps."""
    with regular(path).open("rb") as stream:
        value = stream.readline(1024**2)
    require(value.endswith(b"\n") and len(value) < 1024**2, "first active step is incomplete or exceeds bound")
    return value


def audit(root=ROOT):
    root = Path(root).resolve(); run = root / "journal/runs/train-001"; pins = {}
    def read(path):
        path = regular(path); pins[str(path)] = file_sha(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))
    freeze = read(root / "PRELAUNCH_BINDING.json")
    require(freeze["status"] == "PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION" and
            len(freeze["code_files_sha256"]) == 14 and len(freeze["external_code_files_sha256"]) == 178,
            "registered code freeze geometry differs")
    for name, expected in freeze["code_files_sha256"].items():
        path = (HERE / name).resolve(); require(path.parent == HERE.resolve(), "code pin escapes integration module")
        require(file_sha(path) == expected, "frozen integration code changed: " + name); pins[str(path)] = expected
    for name, expected in freeze["external_code_files_sha256"].items():
        path = regular(name); require(path.suffix == ".py" and file_sha(path) == expected, "external qualified code changed")
        pins[str(path)] = expected
    for name, expected in freeze["artifact_files_sha256"].items():
        path = (root / name).resolve(); require(path.is_relative_to(root), "frozen artifact pin escapes root")
        require(file_sha(path) == expected, "registered artifact changed: " + name); pins[str(path)] = expected
    plan, grant = read(root / "PLAN.json"), read(root / "GRANT.json")
    record = read(run / "zero_addend_qualification.json")
    binding = read(run / "binding.json")["binding"]
    require(record["schema"] == 1 and record["record"] == "BINDING_INTEGRATION_ZERO_ADDEND_QUALIFICATION" and
            record["experiment"] == "BINDING-INTEGRATION-001" and record["status"] == "PASS" and record["complete"] is True and
            record["failure"] is None and record["scope"] == "EXACT_TWO_UPDATE_PREFIX_ONLY" and
            record["full_193_update_trajectory_equality_measured"] is False and record["quality_claim"] is False,
            "qualification is incomplete, failed, or overstates measured scope")
    require(record["binding"] == binding and binding["plan_sha256"] == digest(plan) == grant["plan_sha256"] and
            binding["grant_sha256"] == digest(grant) and binding["execution_code"] == freeze["training_execution_identity"] and
            binding["code_sha256"] == digest(binding["execution_code"]) and
            binding["source"]["assets"] == freeze["source_assets_sha256"] and
            binding["source"]["sha256"] == digest(freeze["source_assets_sha256"]), "qualification source/plan/grant/code binding differs")
    require(binding["data_contract_sha256"] == file_sha(root / "prepared/DATA_CONTRACT.json") and
            binding["data_build_sha256"] == file_sha(root / "prepared/DATA_BUILD.json") and
            binding["native_clock_tokens"] == plan["native_processed_tokens"] == 380921,
            "qualification data contract or native clock differs")
    reference = plan["zero_addend_reference"]
    require(Path(reference["checkpoint_path"]).resolve() == REFERENCE.resolve() == Path(record["reference_checkpoint_path"]).resolve() and
            reference["manifest_sha256"] == record["reference_manifest_sha256"] == REFERENCE_MANIFEST_SHA,
            "qualification points to another reference checkpoint")
    manifest_path = REFERENCE / "manifest.json"; manifest = read(manifest_path)
    require(file_sha(manifest_path) == REFERENCE_MANIFEST_SHA and manifest["sha256"] == digest({k: v for k, v in manifest.items() if k != "sha256"}) and
            manifest["record"] == "DOSELORA_TRAINING_CHECKPOINT" and manifest["experiment"] == "DOSELORA-001" and manifest["status"] == "COMPLETE",
            "original reference manifest seal/identity differs")
    expected_progress = {"updates": 2, "next_group": 2, "processed_tokens": 4032, "supervised_tokens": 388}
    require(manifest["progress"] == record["reference_progress"] == expected_progress and
            record["optimizer_updates"] == record["optimizer_step_count"] == 2 and record["binding_forward_calls"] == record["generation_calls"] == 0,
            "qualification prefix/update/generation scope differs")
    require(record["budget_charges"] == {"processed_training_tokens_all_attempts": 4032, "evaluation_and_qualification_model_calls": 28} and
            freeze["expected_charged_training_and_qualification_tokens"] == 590245 and freeze["expected_qualification_calls"] == 76 and
            grant["limits"]["processed_training_tokens_all_attempts"] == 600000 and grant["limits"]["evaluation_and_qualification_model_calls"] == 128,
            "registered qualification costs or ceiling changed")
    require(manifest["binding"]["source"] == binding["source"] and manifest["binding"]["runtime"] == binding["runtime"] and
            manifest["binding"]["design"] == binding["design"] and manifest["binding"]["parameterization"] == binding["parameterization"],
            "reference/current source/runtime/design/adapter scope differ")
    entries = manifest["entries"]; checks = record["checks"]
    require(len(entries) == len(checks) == record["checked_trainable_tensors"] == 360 and record["checked_component_tensors"] == 1080 and
            [item["name"] for item in entries] == [item["name"] for item in checks] and len({item["name"] for item in entries}) == 360,
            "reference/actual component inventory differs")
    tensor_pins = {entry["file"]: entry["sha256"] for entry in entries}
    require(tensor_pins == reference["tensor_files_sha256"] == record["reference_tensor_files_sha256"] and
            {path.name for path in REFERENCE.iterdir()} == {"manifest.json", *tensor_pins}, "reference file pin inventory differs")
    component_audits = []; prefix_trainable = hashlib.sha256()
    for entry, observed in zip(entries, checks):
        filename = entry["file"]
        require(re.fullmatch(r"tensor-\d{4}\.safetensors", filename) and observed["shape"] == entry["shape"] and
                observed["reference_file"] == filename and observed["reference_file_sha256"] == entry["sha256"], "recorded component provenance differs")
        path = REFERENCE / filename
        require(file_sha(path) == entry["sha256"], "reference safetensor bytes changed")
        hashes = tensor_components(path, entry["name"], entry["shape"])
        for component, expected in hashes.items():
            require(observed[component]["equal"] is True and observed[component]["actual_sha256"] == observed[component]["expected_sha256"] == expected,
                    "independent component bytes disagree with qualification actual/reference hash: " + entry["name"] + "/" + component)
        prefix_trainable.update(bytes.fromhex(hashes["parameter"]))
        component_audits.append({"name": entry["name"], "shape": entry["shape"], "reference_file": filename,
                                 "reference_file_sha256": entry["sha256"], "component_sha256": hashes,
                                 "recorded_actual_hashes_match_independent_reference": True})
        require(file_sha(path) == entry["sha256"], "reference tensor changed during audit")
        pins[str(path.resolve())] = entry["sha256"]
    for key in ("all_trainable_parameters_bitwise_equal", "all_first_moments_bitwise_equal", "all_second_moments_bitwise_equal",
                "reference_rng_bitwise_equal", "rng_restored_exactly", "trainable_state_restored_exactly", "frozen_base_unchanged",
                "qualification_optimizer_discarded", "initial_model_training_mode_restored"):
        require(record[key] is True, "qualification/restoration flag failed: " + key)
    reference_rng = digest(manifest["rng"])
    require(record["actual_prefix_rng_sha256"] == record["reference_rng_sha256"] == reference_rng and
            record["rng_before_sha256"] == record["rng_after_restore_sha256"] and
            record["initial_trainable_sha256"] == record["restored_trainable_sha256"] and
            record["frozen_base_sha256_before"] == record["frozen_base_sha256_after"], "recorded RNG/restoration hash equality differs")
    for key in ("initial_trainable_sha256", "restored_trainable_sha256", "rng_before_sha256", "rng_after_restore_sha256",
                "frozen_base_sha256_before", "frozen_base_sha256_after"):
        require(re.fullmatch(r"[0-9a-f]{64}", record[key]) is not None, "malformed recorded restoration digest")
    prefix_state_sha = prefix_trainable.hexdigest()
    require(prefix_state_sha != record["initial_trainable_sha256"], "recorded initial state unexpectedly equals trained two-update reference")
    native_indices = [index for group in plan["native_update_groups"][:2] for index in group]
    first_group = plan["update_groups"][0]
    wanted = set(native_indices) | set(first_group); prefix_rows = {}
    with regular(root / "prepared/microfit-train.jsonl").open(encoding="utf-8-sig") as stream:
        for index in range(max(wanted) + 1):
            raw = stream.readline(); require(bool(raw), "prepared prefix truncated")
            if index in wanted: prefix_rows[index] = json.loads(raw)
    native_tokens = sum(len(prefix_rows[index]["ids"]) for index in native_indices)
    native_supervised = sum(sum(value != -100 for value in prefix_rows[index]["labels"][1:]) for index in native_indices)
    require(len(native_indices) == 28 and native_tokens == 4032 and native_supervised == 388 and
            record["native_row_ids"] == [prefix_rows[index]["id"] for index in native_indices] and
            record["native_denominators"] == plan["native_denominators"][:2] == [213, 175] and
            record["applied_native_learning_rates"] == plan["applied_native_learning_rates"][:2] == [2.672470144728172e-06, 5.2924359644125685e-06],
            "actual registered prefix rows/tokens/coefficients differ")
    # Read only the first active row; the remainder continues to grow elsewhere.
    steps_path = run / "steps.jsonl"; line = first_line(steps_path); first = json.loads(line)
    native_first, binding_first = plan["native_update_groups"][0], plan["binding_update_groups"][0]
    combined_tokens = sum(len(prefix_rows[index]["ids"]) for index in first_group)
    combined_supervised = sum(sum(value != -100 for value in prefix_rows[index]["labels"][1:]) for index in first_group)
    native_first_tokens = sum(len(prefix_rows[index]["ids"]) for index in native_first)
    lr = plan["optimizer"]["peak_lr"] * (native_first_tokens / 380921) / 0.2
    require(first["updates"] == first["next_group"] == 1 and first_group == native_first + binding_first and
            first["row_ids"] == [prefix_rows[index]["id"] for index in first_group] and
            first["processed_tokens"] == combined_tokens == 2821 and first["supervised_tokens"] == combined_supervised == 224 and
            first["native_clock_tokens"] == native_first_tokens == plan["native_processed_cumulative"][0] == 2036 and
            first["native_forward_calls"] == len(native_first) == 14 and first["binding_forward_calls"] == len(binding_first) == 3 and
            first["native_supervised_denominator"] == first["binding_loss_divisor"] == 213 and
            first["lr"] == lr == plan["applied_native_learning_rates"][0], "first active handoff data/LR/coefficient differs")
    events = []; previous = None
    for index in range(5):
        value = read(root / "journal/events" / f"{index:06d}.json")
        body = {key: item for key, item in value.items() if key != "sha256"}
        require(value["sha256"] == digest(body) and body["index"] == index and body["previous"] == previous,
                "initial journal charge chain differs")
        events.append(body); previous = value["sha256"]
    require(events[0]["kind"] == "start" and events[3]["kind"] == "charge" and events[3]["charges"] == record["budget_charges"] and
            events[4]["kind"] == "charge" and events[4]["charges"] == {"processed_training_tokens_all_attempts": 2821},
            "qualification charge does not precede the first active update")
    require(events[1]["charges"] == {"processed_training_tokens_all_attempts": 2203, "evaluation_and_qualification_model_calls": 16} and
            events[2]["charges"] == {"processed_training_tokens_all_attempts": 2203, "evaluation_and_qualification_model_calls": 26},
            "retained source/zero-B qualification charge geometry differs")
    require(first_line(steps_path) == line and all(file_sha(path) == expected for path, expected in pins.items()), "audited immutable evidence or stable first line changed")
    require(not any(name in sys.modules for name in ("torch", "transformers", "safetensors", "numpy")), "audit imported a tensor/model runtime")
    return {"schema": 1, "record": "BINDING_INTEGRATION_ZERO_ADDEND_AUDIT", "status": "PASS_INDEPENDENT_ZERO_ADDEND_RECEIPT_AUDIT",
            "created_utc": datetime.now(timezone.utc).isoformat(), "auditor_path": str(Path(__file__).resolve()), "auditor_sha256": file_sha(Path(__file__)),
            "qualification_receipt_sha256": file_sha(run / "zero_addend_qualification.json"),
            "verified_frozen_code_files": 14, "verified_external_code_files": 178,
            "verified_frozen_artifact_files": len(freeze["artifact_files_sha256"]), "reference_manifest_sha256": REFERENCE_MANIFEST_SHA,
            "reference_tensor_files_verified": 360, "reference_component_hashes_independently_recomputed": 1080,
            "recorded_actual_prefix_component_hashes_match": True, "component_audits": component_audits,
            "prefix_costs": {"native_rows": 28, "processed_tokens": 4032, "supervised_tokens": 388, "qualification_forward_calls": 28,
                             "optimizer_updates": 2, "binding_forward_calls": 0, "generation_calls": 0},
            "reference_rng_sha256_independently_recomputed": reference_rng,
            "recorded_rng_control_match_and_restoration": True, "recorded_initial_restored_trainable_sha256": record["initial_trainable_sha256"],
            "reference_prefix_trainable_sha256_independently_recomputed": prefix_state_sha,
            "initial_recorded_state_differs_from_trained_prefix": True, "recorded_frozen_base_hash_unchanged": True,
            "first_active_step": {"line_number": 1, "line_bytes": len(line), "line_sha256": hashlib.sha256(line).hexdigest(),
                "processed_tokens": 2821, "supervised_tokens": 224, "native_clock_tokens": 2036, "native_rows": 14, "binding_rows": 3,
                "native_supervised_denominator": 213, "binding_loss_divisor": 213, "applied_lr": lr,
                "native_and_binding_per_supervised_token_coefficient": lr / 213, "execution_order_native_then_binding": True},
            "initial_journal_events_verified": 5, "qualification_reserved_before_first_active_charge": True,
            "scope": "EXACT_TWO_UPDATE_PREFIX_ONLY; first active update handoff checked separately",
            "limitations": ["Disposable actual prefix tensors were intentionally not persisted. This audit independently hashes reference bytes and compares them to the actual hashes recorded by the frozen supervised worker; it does not remeasure live tensors.",
                "Initial parameter restoration and frozen-base immutability are corroborated by recorded equal hashes, boolean checks and the frozen code's pre-active gates. Initial/current model tensors and full base weights were not reread.",
                "Only steps.jsonl line1 and journal events000000–000004 were inspected. No later active step, current active checkpoint, endpoint score or complete-training claim is included.",
                "The result qualifies a two-update native-only recovery prefix. It does not establish equality of the full193-update trajectory, preservation, model quality, selection or release."],
            "source_artifact_sha256": pins, "model_calls": 0, "gpu_calls": 0, "tensor_runtime_imports": False,
            "full_base_weights_read": False, "active_checkpoints_read": False, "later_active_steps_read": False,
            "full_193_update_trajectory_equality_claimed": False, "training_completion_audited": False,
            "quality_claim": False, "release_permitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(); destination = args.root / "ZERO_ADDEND_AUDIT.json"
    require(not destination.exists(), "audit output is create-only")
    value = audit(args.root)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": value["status"], "out": str(destination), "reference_files": 360, "component_hashes": 1080,
                      "first_active_step_only": True, "quality_claim": False}))


if __name__ == "__main__":
    main()
