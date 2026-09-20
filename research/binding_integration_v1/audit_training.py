"""Independent stdlib audit of the completed integration training evidence.

Reads the closed native run, its two optimizer checkpoints and frozen training
inputs/code. It never opens source base weights, export/evaluation artifacts or
model runtimes, and reports no task-quality endpoint or training loss values.
"""
from __future__ import annotations

import argparse
from array import array
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
FINAL_SHA = "be57a94106a76f8a2039b0a463447403b8e717c7e25bc0ec0962efa2137ed3d6"
ZERO_SHA = "1202e83e70a080d838ce93c5bd1a180e94a2ddc6c1fb07b2166e3563c7901255"
WALL_KEYS = ("model_worker_wall_seconds_and_conservative_gpu_seconds", "total_owned_work_wall_seconds")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_path(path, root):
    path = Path(path).resolve(); run = root / "journal/runs/train-001"
    require(path.is_file() and not path.is_symlink(), "missing/nonregular audit input: " + str(path))
    if path.is_relative_to(root):
        require(path.relative_to(root).parts[0] not in ("eval", "posttrain", "arms"), "export/evaluation access is forbidden")
    if path.suffix == ".safetensors":
        require(path.parent in (run / "checkpoint-00002", run / "checkpoint-00193") and re.fullmatch(r"tensor-\d{4}\.safetensors", path.name),
                "only the two completed native checkpoints may be opened")
    else:
        require(path.suffix.lower() in (".py", ".json", ".jsonl", ".md", ".txt", ".csv"), "unregistered audit input type")
    return path


def sha(path, root):
    with safe_path(path, root).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unseal(value):
    body = {key: item for key, item in value.items() if key != "sha256"}
    require(value.get("sha256") == digest(body), "native manifest/journal seal differs")
    return body


def inspect_tensor_file(path, name, shape, root):
    """Validate all stored F32 values and nonnegative second moments in chunks."""
    path = safe_path(path, root); values_count = math.prod(shape)
    require(shape and all(type(n) is int and n > 0 for n in shape), "invalid trainable shape")
    result = {}; ranges = []
    with path.open("rb") as stream:
        length = stream.read(8); require(len(length) == 8, "truncated safetensor")
        header_length = struct.unpack("<Q", length)[0]
        require(0 < header_length <= 1024**2, "unexpected safetensor header size")
        header = json.loads(stream.read(header_length).decode("utf-8"))
        require(set(header) == {"parameter", "exp_avg", "exp_avg_sq"}, "optimizer tensor components differ")
        offset = 8 + header_length
        for component in ("parameter", "exp_avg", "exp_avg_sq"):
            item = header[component]; lo, hi = item["data_offsets"]
            require(item["dtype"] == "F32" and item["shape"] == shape and type(lo) is int and type(hi) is int and
                    0 <= lo < hi and hi - lo == 4 * values_count, "component dtype/shape/byte geometry differs")
            stream.seek(offset + lo); remaining = hi - lo
            hashed = hashlib.sha256(canonical([name, component, shape, "torch.float32"]))
            scanned = 0
            while remaining:
                chunk = stream.read(min(1024**2, remaining))
                require(bool(chunk) and len(chunk) % 4 == 0, "truncated/non-F32 payload")
                hashed.update(chunk); remaining -= len(chunk)
                numbers = array("f"); numbers.frombytes(chunk)
                require(numbers.itemsize == 4, "stdlib float array is not F32")
                if sys.byteorder != "little":
                    numbers.byteswap()
                require(all(map(math.isfinite, numbers)), "nonfinite stored " + component + ": " + name)
                if component == "exp_avg_sq":
                    require(min(numbers) >= 0, "negative Adam second moment: " + name)
                scanned += len(numbers)
            require(scanned == values_count, "float-value count differs")
            result[component] = {"sha256": hashed.hexdigest(), "values": scanned, "finite": True,
                                 **({"nonnegative": True} if component == "exp_avg_sq" else {})}
            ranges.append((lo, hi))
        ranges.sort()
        require(ranges[0][0] == 0 and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])) and
                offset + ranges[-1][1] == path.stat().st_size, "safetensor payload has gaps/overlap/trailing bytes")
    return result


def learning_rate(peak, completed, total, design):
    fraction = completed / total
    if fraction < design["warmup_fraction"]:
        return peak * fraction / design["warmup_fraction"]
    start = 1.0 - design["decay_fraction"]
    if fraction <= start:
        return peak
    phase = min(1.0, (fraction - start) / design["decay_fraction"])
    return peak * (design["minimum_lr_fraction"] + (1.0 - design["minimum_lr_fraction"]) * (1.0 - math.sqrt(phase)))


def audit(root=ROOT):
    root = Path(root).resolve(); run = root / "journal/runs/train-001"; pins = {}
    def pin(path):
        path = safe_path(path, root); value = sha(path, root); pins[str(path)] = value; return value
    def read(path):
        pin(path); return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    def jsonl(path):
        pin(path)
        with Path(path).open(encoding="utf-8-sig") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    freeze = read(root / "PRELAUNCH_BINDING.json")
    require(freeze["status"] == "PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION" and len(freeze["code_files_sha256"]) == 14 and
            len(freeze["external_code_files_sha256"]) == 178, "unexpected frozen execution code inventory")
    for name, expected in freeze["code_files_sha256"].items():
        require(Path(name).name == name and pin(HERE / name) == expected, "frozen integration code changed")
    for path, expected in freeze["external_code_files_sha256"].items():
        require(Path(path).suffix == ".py" and pin(path) == expected, "qualified external code changed")
    for name, expected in freeze["artifact_files_sha256"].items():
        path = (root / name).resolve(); require(path.is_relative_to(root) and pin(path) == expected, "frozen training artifact changed")
    plan, grant = read(root / "PLAN.json"), read(root / "GRANT.json")
    require(read(root / "journal/grant.json") == grant, "native journal grant differs from registration")
    result = read(run / "result.json"); summary = read(run / "run_summary.json"); supervisor = read(run / "supervisor.json")
    require(result["status"] == "COMPLETE" and result["complete"] is True and result["record"] == "BINDING_INTEGRATION_TRAINING_RESULT" and
            result["experiment"] == summary["experiment"] == "BINDING-INTEGRATION-001" and summary["run_id"] == "train-001" and
            type(summary["returncode"]) is int and summary["returncode"] == 0 and summary["worker_failed"] is False and
            supervisor["status"] == "EXITED" and type(supervisor["returncode"]) is int and supervisor["returncode"] == 0 and
            supervisor["cleanup"] is None and summary["supervisor"] == supervisor,
            "native worker/controller did not complete successfully")
    require(not any((run / name).exists() for name in ("failure.json", "supervisor_failure.json")), "native failure receipt exists")
    binding = read(run / "binding.json")["binding"]
    require(result["binding"] == binding and binding["plan_sha256"] == digest(plan) == grant["plan_sha256"] and
            binding["grant_sha256"] == digest(grant) and binding["execution_code"] == freeze["training_execution_identity"] and
            binding["code_sha256"] == digest(binding["execution_code"]) and binding["source"]["assets"] == freeze["source_assets_sha256"],
            "native source/plan/grant/code binding differs")
    launch, spawn, claim = (read(run / name) for name in ("launch.json", "spawn.json", "worker_claim.json"))
    argv = launch["argv"]
    def option(name):
        require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "launch argument absent/duplicated")
        return argv[argv.index(name) + 1]
    require(Path(launch["out"]).resolve() == run and launch["timeout_seconds"] == 2400 and
            Path(launch["cwd"]).resolve() == BASE / "ChatBubble-binding-integration-pilot-20260920" and
            Path(argv[0]).resolve() == BASE / ".venv-mq001/Scripts/python.exe" and Path(argv[1]).resolve() == HERE / "train.py" and
            Path(option("--plan")).resolve() == root / "PLAN.json" and Path(option("--prepared")).resolve() == root / "prepared" and
            Path(option("--grant")).resolve() == root / "GRANT.json" and Path(option("--journal")).resolve() == root / "journal" and
            Path(option("--out")).resolve() == run and option("--run-id") == "train-001" and
            option("--expected-code-digest") == digest(freeze["training_execution_identity"]) and
            argv.count("--_worker") == argv.count("--confirm-model-execution") == 1, "native supervised command changed")
    require(spawn["launch_sha256"] == claim["launch_sha256"] == digest(launch) and
            spawn["supervisor_pid"] == claim["supervisor_pid"] == launch["supervisor_pid"] and
            spawn["child_pid"] == supervisor["child_pid"] and
            (claim["pid"] == spawn["child_pid"] or (claim.get("launcher_redirected") is True and claim["parent_pid"] == spawn["child_pid"])),
            "native launch/spawn/worker ownership differs")
    zero_path = run / "zero_addend_qualification.json"; zero = read(zero_path); zero_audit = read(root / "ZERO_ADDEND_AUDIT.json")
    require(pin(zero_path) == result["zero_addend_qualification_sha256"] == ZERO_SHA and Path(result["zero_addend_qualification_path"]).resolve() == zero_path and
            zero["status"] == "PASS" and zero["complete"] is True and zero["scope"] == "EXACT_TWO_UPDATE_PREFIX_ONLY" and
            zero["binding"] == binding and zero_audit["status"] == "PASS_INDEPENDENT_ZERO_ADDEND_RECEIPT_AUDIT" and
            zero_audit["qualification_receipt_sha256"] == ZERO_SHA, "zero-addend qualification/audit pin differs")
    require(result["active_optimizer_started_fresh"] is True and
            result["active_initial_trainable_sha256"] == zero["initial_trainable_sha256"] == zero["restored_trainable_sha256"] and
            result["active_initial_rng_sha256"] == zero["rng_before_sha256"] == zero["rng_after_restore_sha256"] and
            zero["trainable_state_restored_exactly"] is True and zero["rng_restored_exactly"] is True and
            zero["qualification_optimizer_discarded"] is True and zero["initial_model_training_mode_restored"] is True and
            result["frozen_base_sha256_before_after"] == zero["frozen_base_sha256_before"] == zero["frozen_base_sha256_after"] and
            zero["frozen_base_unchanged"] is True and zero["full_193_update_trajectory_equality_measured"] is False,
            "zero-addend restoration/fresh-active/frozen-base handoff differs")
    rows = jsonl(root / "prepared/microfit-train.jsonl"); steps = jsonl(run / "steps.jsonl")
    require(pin(run / "steps.jsonl") == result["steps_sha256"] and pin(run / "train.csv") == result["train_csv_sha256"] and
            len(rows) == 3465 and len(steps) == len(plan["update_groups"]) == len(plan["native_update_groups"]) == len(plan["binding_update_groups"]) == 193,
            "completed step/data inventory differs")
    native_total = processed_total = supervised_total = 0; per_update = []
    original_plan_path = BASE / "rineya-dataset-retention-records/highdose-lora001-resume-20260920/PLAN.json"
    original_plan = read(original_plan_path)
    require(pin(original_plan_path) == plan["parent_control_plan_file_sha256"], "original native plan changed")
    for index, step in enumerate(steps):
        native = plan["native_update_groups"][index]; added = plan["binding_update_groups"][index]; combined = plan["update_groups"][index]
        require(combined == native + added and [rows[i]["id"] for i in native] ==
                [original_plan["row_order_per_exposure"][i] for i in original_plan["update_groups"][index]], "native group or native-first ordering changed")
        n_tokens = sum(len(rows[i]["ids"]) for i in native)
        all_tokens = sum(len(rows[i]["ids"]) for i in combined)
        n_supervised = sum(sum(token != -100 for token in rows[i]["labels"][1:]) for i in native)
        all_supervised = sum(sum(token != -100 for token in rows[i]["labels"][1:]) for i in combined)
        native_total += n_tokens; processed_total += all_tokens; supervised_total += all_supervised
        rate = learning_rate(plan["optimizer"]["peak_lr"], native_total, 380921, binding["design"])
        require(step["updates"] == step["next_group"] == index + 1 and step["row_ids"] == [rows[i]["id"] for i in combined] and
                step["processed_tokens"] == processed_total and step["supervised_tokens"] == supervised_total and
                step["native_clock_tokens"] == native_total == plan["native_processed_cumulative"][index] and
                step["native_supervised_denominator"] == step["binding_loss_divisor"] == n_supervised == plan["native_denominators"][index] and
                step["lr"] == rate == plan["applied_native_learning_rates"][index] and
                step["native_forward_calls"] == len(native) and step["binding_forward_calls"] == len(added),
                "saved update/data/LR/native-coefficient accounting differs at update " + str(index + 1))
        require(type(step["grad_norm"]) in (int, float) and math.isfinite(step["grad_norm"]) and step["grad_norm"] >= 0,
                "nonfinite gradient-norm execution telemetry")
        per_update.append({"update": index + 1, "native_rows": len(native), "binding_rows": len(added),
                           "processed_tokens_this_update": all_tokens, "native_clock_tokens": native_total,
                           "native_supervised_denominator": n_supervised, "binding_loss_divisor": n_supervised,
                           "applied_learning_rate": rate, "per_supervised_token_loss_coefficient_times_lr": rate / n_supervised})
    final_progress = {"updates": 193, "next_group": 193, "processed_tokens": 577417, "supervised_tokens": 39502}
    require(native_total == result["native_clock_tokens"] == 380921 and processed_total == 577417 and supervised_total == 39502 and
            result["progress"] == final_progress and [i for group in plan["update_groups"] for i in group] == list(range(3465)),
            "complete active trajectory totals differ")
    paths = sorted((root / "journal/events").iterdir()); events = []; previous = None
    used = {key: 0 for key in grant["limits"]}; active = {}; started = set()
    for index, path in enumerate(paths):
        require(path.name == f"{index:06d}.json", "journal index gap or unregistered file")
        raw = read(path); event = unseal(raw)
        require(event["index"] == index and event["previous"] == previous, "journal chain differs")
        charges = event.get("charges", {})
        require(set(charges) <= set(used) and all(type(value) is int and value >= 0 for value in charges.values()), "invalid native budget charge")
        for key, value in charges.items(): used[key] += value
        if event["kind"] == "start":
            require(not active and event["run_id"] not in started, "overlapping/reused native attempt")
            active[event["run_id"]] = event; started.add(event["run_id"])
        elif event["kind"] == "settle":
            require(event["run_id"] in active and event["receipt_sha256"] == pin(run / "supervisor.json"), "settlement lacks its observed supervisor")
            start = active.pop(event["run_id"])
            for key in WALL_KEYS:
                require(type(event["actual_wall"][key]) is int and event["actual_wall"][key] >= 0, "invalid settled wall amount")
                used[key] += event["actual_wall"][key] - start["charges"][key]
        else:
            require(event["kind"] == "charge", "unknown journal event kind")
        events.append(event); previous = raw["sha256"]
    require(started == {"train-001"} and not active and all(used[key] <= limit for key, limit in grant["limits"].items()), "native budget is active or breached")
    state = {"used_or_reserved": used, "active_reservations": [], "budget_breached": False,
             "remaining": {key: grant["limits"][key] - value for key, value in used.items()}}
    require(state == summary["budget_after"], "run summary does not reproduce settled journal")
    q_charges = [{"processed_training_tokens_all_attempts": 2203, "evaluation_and_qualification_model_calls": 16},
                 {"processed_training_tokens_all_attempts": 2203, "evaluation_and_qualification_model_calls": 26},
                 {"processed_training_tokens_all_attempts": 4032, "evaluation_and_qualification_model_calls": 28}]
    checkpoint_reserve = 7255040 * 12 + 2**20
    expected_events = list(q_charges)
    for update in per_update:
        expected_events.append({"processed_training_tokens_all_attempts": update["processed_tokens_this_update"]})
        if update["update"] in (2, 193): expected_events.append({"new_disk_bytes": checkpoint_reserve})
    expected_events.append({"processed_training_tokens_all_attempts": 4390, "evaluation_and_qualification_model_calls": 6})
    require(len(events) == 201 and events[0]["kind"] == "start" and events[-1]["kind"] == "settle" and
            all(event["kind"] == "charge" for event in events[1:-1]) and [event["charges"] for event in events[1:-1]] == expected_events,
            "all-step/qualification/checkpoint charge order differs")
    wall = supervisor["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0 and used[WALL_KEYS[0]] == math.ceil(wall) and
            used[WALL_KEYS[1]] >= math.ceil(wall) and used["processed_training_tokens_all_attempts"] == result["charged_processed_tokens"] == 590245 and
            used["evaluation_and_qualification_model_calls"] == result["qualification_model_calls"] == 76 and
            used["model_worker_launches"] == 1 and used["retries"] == used["generated_tokens_all_candidates"] == used["on_policy_rl_rollout_tokens"] == used["cloud_or_teacher_spend"] == 0,
            "registered completed native accounting differs")
    checkpoint_audits = {}; total_values = 0; total_files = 0
    require(set(result["checkpoint_manifests_sha256"]) == {"checkpoint-00002", "checkpoint-00193"} and
            result["checkpoint_manifest_sha256"] == result["checkpoint_manifests_sha256"]["checkpoint-00193"] == FINAL_SHA,
            "final checkpoint identity differs")
    for name, expected_manifest in result["checkpoint_manifests_sha256"].items():
        directory = run / name; manifest_path = directory / "manifest.json"; manifest = read(manifest_path); body = unseal(manifest)
        update = 2 if name == "checkpoint-00002" else 193
        expected_progress = {key: steps[update - 1][key] for key in ("updates", "next_group", "processed_tokens", "supervised_tokens")}
        require(pin(manifest_path) == expected_manifest and body["schema"] == 1 and body["record"] == "BINDING_INTEGRATION_TRAINING_CHECKPOINT" and
                body["experiment"] == "BINDING-INTEGRATION-001" and body["status"] == "COMPLETE" and body["binding"] == binding and
                body["progress"] == expected_progress and body["adapter_spec"] == result["adapter_spec"] and
                body["zero_addend_qualification_sha256"] == ZERO_SHA, "checkpoint scope/progress/qualification differs")
        entries = body["entries"]; spec = body["adapter_spec"]
        require(len(entries) == 360 and [entry["name"] for entry in entries] == spec["trainable_names"] and
                len({entry["name"] for entry in entries}) == 360 and sum(math.prod(entry["shape"]) for entry in entries) == spec["trainable_parameters"] == 7255040 and
                [entry["file"] for entry in entries] == [f"tensor-{i:04d}.safetensors" for i in range(360)] and
                {path.name for path in directory.iterdir()} == {"manifest.json", *[entry["file"] for entry in entries]}, "checkpoint tensor inventory differs")
        tensors = []; logical = manifest_path.stat().st_size
        for entry in entries:
            path = directory / entry["file"]; require(pin(path) == entry["sha256"], "saved optimizer tensor file hash differs")
            components = inspect_tensor_file(path, entry["name"], entry["shape"], root)
            require(sha(path, root) == entry["sha256"], "checkpoint changed while inspected")
            tensors.append({"name": entry["name"], "file": entry["file"], "shape": entry["shape"], "file_sha256": entry["sha256"], "components": components})
            logical += path.stat().st_size; total_files += 1; total_values += sum(value["values"] for value in components.values())
        require(logical <= checkpoint_reserve, "checkpoint exceeds its native disk reservation")
        checkpoint_audits[name] = {"manifest_sha256": expected_manifest, "progress": expected_progress, "native_clock_tokens": plan["native_processed_cumulative"][update - 1],
            "trainable_tensors": 360, "component_tensors": 1080, "all_stored_values_finite": True, "all_second_moments_nonnegative": True,
            "logical_checkpoint_bytes": logical, "registered_checkpoint_reservation_bytes": checkpoint_reserve, "tensors": tensors}
    merge = read(run / "merge_qualification.json")
    require(pin(run / "merge_qualification.json") == result["merge_qualification_sha256"] and merge["status"] == "PASS" and merge["complete"] is True and
            merge["binding"] == binding and merge["checkpoint_manifest_sha256"] == FINAL_SHA and
            merge["frozen_base_sha256_before_after"] == result["frozen_base_sha256_before_after"] and
            merge["atol"] == merge["rtol"] == 2e-4 and merge["argmax_equality_required"] is True and merge["original_modules_restored"] is True and
            len(merge["dense_tensor_sha256"]) == 220 and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in merge["dense_tensor_sha256"].values()),
            "final qualified CPU merge receipt differs")
    fixture_lengths = {"reply_7": 31, "first_training_row": 116, "synthetic_2048_arange": 2048}
    require({value["input"] for value in merge["inputs"]} == set(fixture_lengths) and len(merge["inputs"]) == 3 and
            merge["budget_charges"] == expected_events[-1], "final merge fixture/cost scope differs")
    for fixture in merge["inputs"]:
        require(fixture["input_token_count"] == fixture_lengths[fixture["input"]] and fixture["passed"] is True and
                fixture["checked_last_positions"] == 16 and fixture["checked_logits_shape"] == [1, 16, 151936] and
                fixture["tolerance_violations"] == fixture["argmax_mismatches"] == 0 and
                type(fixture["max_abs_difference"]) in (int, float) and math.isfinite(fixture["max_abs_difference"]), "merge fixture did not pass fixed tolerance/argmax gates")
    require(not any(name in sys.modules for name in ("torch", "transformers", "safetensors", "numpy")), "audit imported a model/tensor runtime")
    require(all(sha(path, root) == expected for path, expected in pins.items()), "closed training evidence changed during audit")
    return {"schema": 1, "record": "BINDING_INTEGRATION_TRAINING_AUDIT", "status": "PASS_INDEPENDENT_COMPLETED_TRAINING_AUDIT",
            "created_utc": datetime.now(timezone.utc).isoformat(), "auditor_path": str(Path(__file__).resolve()), "auditor_sha256": sha(Path(__file__), root),
            "frozen_code_files_verified": 14, "external_code_files_verified": 178, "sealed_journal_events_replayed": 201,
            "active_updates_replayed": 193, "training_inventory": {"views": 3465, "processed_tokens": 577417, "supervised_tokens": 39502, "native_clock_tokens": 380921},
            "all_native_groups_denominators_and_learning_rates_match": True, "all_binding_losses_use_original_native_denominator": True,
            "native_first_then_binding_order_verified": True, "per_update_execution_audit": per_update,
            "zero_addend_receipt_sha256": ZERO_SHA, "zero_addend_prior_independent_audit_sha256": pin(root / "ZERO_ADDEND_AUDIT.json"),
            "recorded_initial_state_and_rng_handoff_match": True, "active_optimizer_started_fresh": True, "recorded_frozen_base_hash_unchanged": True,
            "observed_supervisor_wall_seconds": wall, "rounded_worker_seconds": used[WALL_KEYS[0]], "rounded_owned_seconds": used[WALL_KEYS[1]],
            "native_budget_state_reproduced": state, "checkpoint_tensor_files_verified": total_files,
            "checkpoint_components_verified": total_files * 3, "finite_fp32_values_scanned": total_values,
            "checkpoints": checkpoint_audits, "final_manifest_sha256": FINAL_SHA,
            "merge_qualification_receipt_verified": True, "merge_dense_tensor_hash_count": 220,
            "merge_qualification_reexecuted": False, "source_artifact_sha256": pins,
            "limitations": ["Checkpoint parameters and Adam moments were independently inspected as raw F32 bytes. Gradients and optimizer updates were not reexecuted.",
                "Frozen-base preservation and initial state/RNG restoration rely on matching recorded hashes, the earlier independent prefix receipt audit, and the frozen executed code's gates; full base weights were not reread.",
                "The final merge qualification receipt was checked for fixed fixtures, tolerances, argmax agreement and220 dense identities. CPU merge tensors and active export artifacts were not opened or recomputed here.",
                "Observed supervisor wall is not measured GPU-active time. No evaluation result, VIEW quality endpoint, preservation or acquisition outcome was inspected or inferred.",
                "The measured zero-addend equality applies only to its two-update prefix; full193-update trajectory equality with the original control is not claimed."],
            "model_calls": 0, "gpu_calls": 0, "full_base_weights_read": False, "export_artifacts_read": False, "evaluation_artifacts_read": False,
            "full_193_update_control_equality_claimed": False, "quality_claim": False, "release_permitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(); output = args.root / "TRAINING_AUDIT.json"
    require(not output.exists(), "training audit output is create-only")
    result = audit(args.root)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(output), "steps": 193,
                      "tensor_files": result["checkpoint_tensor_files_verified"], "component_tensors": result["checkpoint_components_verified"],
                      "finite_fp32_values_scanned": result["finite_fp32_values_scanned"], "quality_claim": False}))


if __name__ == "__main__":
    main()
