"""Standard-library study-dose registration, never a training/launch permit."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
BUNDLE = PACKAGE / "data" / "crossed_source2_v1_20261003"
UPDATES, ACCUMULATION, CHUNK_UPDATES = 27, 20, 9
EOS_ID, PAD_ID = 151645, 151643
EXPECTED_TOTALS = {"M1": (130562, 5614), "M2": (130574, 5615)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def filehash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_jsonl(path):
    return [json.loads(line, object_pairs_hook=_pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def learning_rates():
    """One-based update LR; ceil(27*.05)=2 warmup updates; update27 LR=0."""
    peak = Fraction(1, 10000)
    return [float(peak / 2), float(peak)] + [float(peak * Fraction(27 - step, 25)) for step in range(3, 28)]


def validate_token_row(row):
    for name in ("prompt_ids", "completion_ids", "input_ids", "labels", "attention_mask"):
        require(type(row.get(name)) is list and all(type(x) is int for x in row[name]), "typed array " + name)
    prompt, completion, ids = row["prompt_ids"], row["completion_ids"], row["input_ids"]
    require(prompt and completion and 1 < len(ids) <= 512, "complete bounded sequence")
    require(all(x >= 0 for x in prompt + completion), "negative token")
    require(completion[-1] == EOS_ID and EOS_ID not in completion[:-1], "exact final completion EOS")
    require(PAD_ID not in ids, "padding token forbidden")
    require(ids == prompt + completion, "no packing, truncation or target alteration")
    require(row["labels"] == [-100] * len(prompt) + completion, "prompt mask or target/EOS labels")
    require(row["attention_mask"] == [1] * len(ids), "padding or attention-mask alteration")
    require(type(row.get("prompt_tokens")) is int and row["prompt_tokens"] == len(prompt), "prompt token count")
    require(type(row.get("total_tokens")) is int and row["total_tokens"] == len(ids), "input token count")
    require(type(row.get("supervised_tokens")) is int and row["supervised_tokens"] == len(completion), "supervised token count")


def input_paths():
    names = ("preparation_receipt.json", "plan.json", "enrollment.json", "candidate_records.jsonl",
             "pilot_decisions.json", "matching_policy.json", "schedule_result.json", "training_occurrences.json",
             "token_pool_M1.jsonl", "token_pool_M2.jsonl")
    paths = [BUNDLE / name for name in names]
    paths += [PACKAGE / "PILOT_DECISIONS.json", PACKAGE / "pilot_trainable_inventory.json", Path(__file__),
              PACKAGE / "evidence" / "nq001_attempt7_independent_audit_20261003.json",
              PACKAGE / "evidence" / "nq001_attempt7_20261003" / "SUPERVISOR_RECEIPT.json",
              PACKAGE / "evidence" / "nq001_attempt7_20261003" / "worker_result.json"]
    return paths


def _inputs():
    decisions = read_json(PACKAGE / "PILOT_DECISIONS.json")
    native_receipt = read_json(PACKAGE / "evidence" / "nq001_attempt7_20261003" / "SUPERVISOR_RECEIPT.json")
    require(native_receipt["status"] == "SUPERVISED_WORKER_EXITED_ZERO", "native7 supervised closure")
    for filename, expected in native_receipt["preflight"]["data_pins"].items():
        require(filehash(Path(filename)) == expected, "source bundle differs from native7 byte binding")
    require(native_receipt["preflight"]["model_pins"]["model.safetensors"] == decisions["student"]["weight_sha256"], "student decision/native7 weight binding")
    require(decisions == read_json(BUNDLE / "pilot_decisions.json"), "root/bundle decisions differ")
    inventory = read_json(PACKAGE / "pilot_trainable_inventory.json")
    require(filehash(PACKAGE / "pilot_trainable_inventory.json") == decisions["adaptation"]["inventory_sha256"], "inventory pin")
    require(inventory["tensor_count"] == 112 and inventory["trainable_parameter_count"] == 1146880
            and len(inventory["tensors"]) == 112 and all(item["dtype"] == "float32" for item in inventory["tensors"]), "full FP32 qv8 inventory")
    require(sum(item["parameters"] for item in inventory["tensors"]) == 1146880, "inventory parameter count")
    require(decisions["seeds"]["first_development_seed"] == 101 and decisions["adaptation"]["rank"] == 8
            and decisions["adaptation"]["gradient_accumulation_sequences"] == 20, "frozen seed/rank/accumulation")
    schedule = read_json(BUNDLE / "schedule_result.json")
    occurrences = read_json(BUNDLE / "training_occurrences.json")
    require(schedule["status"] == "MATCHED_CPU_SCHEDULE" and schedule["optimizer_updates_per_arm"] == 27, "matched source schedule")
    require(occurrences["row_ids"] == schedule["schedules"] and occurrences["schedule_sha256"] == schedule["sha256"], "exact occurrence source order")
    plan = read_json(BUNDLE / "plan.json")
    enrollment = read_json(BUNDLE / "enrollment.json")
    candidates = read_jsonl(BUNDLE / "candidate_records.jsonl")
    records = {item["row_id"]: item for item in candidates}
    require(len(records) == len(candidates), "duplicate candidate ID")
    pools = {}
    for arm in ("M1", "M2"):
        rows = read_jsonl(BUNDLE / ("token_pool_" + arm + ".jsonl"))
        pools[arm] = {row["row_id"]: row for row in rows}
        require(len(pools[arm]) == len(rows), "duplicate token row ID")
        for row in rows:
            validate_token_row(row)
            record = records[row["row_id"]]
            require(record["arm"] == arm and record["split"] == "train" and record["skill"] in ("R", "S"), "atomic TRAIN arm only")
            require(record["verification"]["accepted"] is True and record["verification"]["reason"] == "verified", "unverified row")
            require(record["parent_id"] in enrollment["parents"] and plan["assignments"][record["parent_id"]]["split"] == "train", "unexposed source-parent enrollment")
            require(record["component_id"] == plan["assignments"][record["parent_id"]]["component_id"], "dependency component binding")
    return decisions, schedule, pools, records


def _stats(ids, pool, records):
    return {"sequences": len(ids), "input_tokens": sum(len(pool[row]["input_ids"]) for row in ids),
            "supervised_tokens": sum(len(pool[row]["completion_ids"]) for row in ids),
            "parent_exposures": dict(sorted(Counter(records[row]["parent_id"] for row in ids).items()))}


def build_registration():
    decisions, source, pools, records = _inputs()
    rates = learning_rates()
    arms = {"M0": {"role": "unchanged_base_evaluation_only", "adaptation_sequences": 0,
                    "optimizer_updates": 0, "training_input_tokens": 0, "trainable_parameters": 0}}
    for arm in ("M1", "M2"):
        ids = source["schedules"][arm]
        require(len(ids) == 540 and all(row in pools[arm] for row in ids), "exact540 whole occurrences")
        totals = _stats(ids, pools[arm], records)
        require((totals["input_tokens"], totals["supervised_tokens"]) == EXPECTED_TOTALS[arm], "actual dose drift")
        require(totals["parent_exposures"] == source["metrics"]["parent_exposures"][arm], "source-parent exposure vector")
        updates, chunks = [], []
        for ordinal in range(1, 28):
            start = (ordinal - 1) * 20
            selected = ids[start:start + 20]
            updates.append({"update_ordinal": ordinal, "learning_rate": rates[ordinal - 1],
                            "first_occurrence_ordinal": start + 1, "last_occurrence_ordinal": start + 20,
                            "occurrence_row_ids": selected, **_stats(selected, pools[arm], records)})
        for ordinal in range(1, 4):
            start = (ordinal - 1) * 180
            selected = ids[start:start + 180]
            chunks.append({"chunk_ordinal": ordinal, "first_update_ordinal": (ordinal - 1) * 9 + 1,
                           "last_update_ordinal": ordinal * 9, "first_occurrence_ordinal": start + 1,
                           "last_occurrence_ordinal": start + 180, "occurrence_row_ids": selected,
                           "occurrence_order_sha256": digest(selected), **_stats(selected, pools[arm], records)})
        arms[arm] = {"intervention": "verified_surface_varied_atomic_R_S" if arm == "M1" else "verified_condition_diverse_atomic_R_S",
                     "occurrence_order_sha256": digest(ids), "optimizer_updates": 27, "totals": totals,
                     "updates": updates, "resume_chunks": chunks}
    require(arms["M1"]["totals"]["parent_exposures"] == arms["M2"]["totals"]["parent_exposures"], "matched parent vector")
    audit = read_json(PACKAGE / "evidence" / "nq001_attempt7_independent_audit_20261003.json")
    require(audit["status"] == "PASS_SHORT_SCOPED_NATIVE_MECHANICS" and audit["short_scoped_native_mechanics_verified"] is True
            and audit["supervisor_status"] == "SUPERVISED_WORKER_EXITED_ZERO", "native7 short-mechanics evidence")
    result = {"schema_version": 1, "id": "SYNTH-STUDY-DEV-DRAFT-001", "experiment": "SYNTH-CRS-001",
        "status": "CPU_SCHEDULE_PREPARATION_NOT_EXECUTION_PERMIT", "native_launch_authorized": False,
        "permissions": {"native_inference": False, "training": False, "protected_bank_access": False,
                        "paid_services": False, "model_upload": False, "merge": False},
        "seed": 101, "student": {key: value for key, value in decisions["student"].items()
                                    if key not in ("native_load_performed", "static_asset_hashes_refreshed")},
        "adaptation": decisions["adaptation"],
        "optimizer": {**decisions["optimizer"], "foreach": False, "fused": False, "capturable": False,
                      "differentiable": False, "learning_rates_by_1based_update": rates, "warmup_updates": 2,
                      "schedule_definition": "ceil(27*0.05)=2; updates1,2 use5e-5,1e-4; updates3..27 use1e-4*(27-update)/25",
                      "update27_zero_LR": "Consumes all20 whole sequences, loss/backward/clip/AdamW step and moments still count; no dropped final batch"},
        "dose": {"full_updates_per_adapted_arm": 27, "sequences_per_update": 20, "whole_sequences_per_arm": 540,
                 "nonpadding_input_token_ceiling": 131072, "M1_actual_input_tokens": 130562, "M2_actual_input_tokens": 130574,
                 "loss": "mean_completion_plus_EOS_per_sequence_then_mean_over20_sequences",
                 "padding": False, "packing": False, "truncation": False, "dropping": False,
                 "replay_added": False, "warmstart": False, "RS_training": False},
        "resume_contract": {"chunks_per_arm": 3, "full_updates_per_chunk": 9,
                 "chunk1_start": "unchanged pinned base and fresh identically seeded zero-B qv8 adapters; fresh optimizer",
                 "later_chunk_start": "only successful previous study chunk of samearm/seed/source/recipe",
                 "qualification_STATE1_warmstart_allowed": False,
                 "required_state": ["112FP32adapter_tensors", "FP32AdamWmoments_and_step", "Python_CPU_CUDA0_RNG",
                      "exact_occurrence_cursor", "global_LR_update_ordinal", "empty_gradient_accumulation",
                      "source_decision_tokenizer_recipe_pins"]},
        "native7_binding": {"scope": "short_mechanics_only_40TRAIN_rows_2native_updates_not_sustained_study_fit",
                    "audit_sha256": filehash(PACKAGE / "evidence" / "nq001_attempt7_independent_audit_20261003.json"),
                    "source_commit": audit["source_commit"], "config_sha256": audit["hash_closure"]["config_sha256_verified"],
                    "quality_transfer_retention_claims": False, "STATE1_use": "qualification_evidence_only_never_study_initialization"},
        "pending_gates": ["sustained27_update_and9_update_chunk_runtime_resource_fit", "fresh_priorcorrect_guard_enrollment",
            "language_and_long_context_guard_scope", "fresh_atomic_and_RS_evaluation_freeze", "scorer_and_answer_format_qualification",
            "effect_retention_margins_statistics_power_and_seed_rule", "measured_cost_and_dispatch_caps",
            "current_clean_committed_source_runtime_data_host_admission", "complete_run_specific_launch_authorization"],
        "confirmation_authorized": False, "arms": arms,
        "source_and_decision_pins": {str(path.relative_to(ROOT)).replace("\\", "/"): filehash(path) for path in input_paths()}}
    result["sha256"] = digest(result)
    return result


def audit_chunks(arm, specification, source_ids, pool, records):
    """Independent ordinal walk: detect missing/duplicate/reordered chunks."""
    seen, cursor = [], 1
    require(len(specification["resume_chunks"]) == 3, "three resume chunks")
    for expected, chunk in enumerate(specification["resume_chunks"], 1):
        ids = chunk["occurrence_row_ids"]
        require(chunk["chunk_ordinal"] == expected and chunk["first_occurrence_ordinal"] == cursor, "chunk ordinal coverage")
        require(chunk["last_occurrence_ordinal"] == cursor + 179 and len(ids) == 180, "nine full updates per chunk")
        require(chunk["first_update_ordinal"] == (expected - 1) * 9 + 1 and chunk["last_update_ordinal"] == expected * 9, "global update cursor")
        require(ids == source_ids[cursor - 1:cursor + 179], "chunk source occurrence order")
        for row_id in ids:
            validate_token_row(pool[row_id])
        stats = _stats(ids, pool, records)
        require(all(chunk[key] == value for key, value in stats.items()), "chunk tokens or parent vector")
        require(chunk["occurrence_order_sha256"] == digest(ids), "chunk order binding")
        seen.extend(ids)
        cursor += 180
    require(cursor == 541 and seen == source_ids, "whole dose coverage without dropped/replayed occurrence")
    require(len(specification["updates"]) == 27, "27 actual updates")
    for ordinal, update in enumerate(specification["updates"], 1):
        ids = source_ids[(ordinal - 1) * 20:ordinal * 20]
        require(update["update_ordinal"] == ordinal and update["learning_rate"] == learning_rates()[ordinal - 1], "global LR ordinal")
        require(update["occurrence_row_ids"] == ids and update["first_occurrence_ordinal"] == (ordinal - 1) * 20 + 1
                and update["last_occurrence_ordinal"] == ordinal * 20, "full batch occurrence cursor")
        require(all(update[key] == value for key, value in _stats(ids, pool, records).items()), "update tokens/parent exposure")
    return _stats(seen, pool, records)


def validate_registration(registration):
    require(type(registration) is dict and registration.get("sha256") == digest({k: v for k, v in registration.items() if k != "sha256"}), "registration checksum")
    expected = build_registration()
    require(registration["source_and_decision_pins"] == expected["source_and_decision_pins"], "source/decision byte pins changed")
    _, source, pools, records = _inputs()
    for arm in ("M1", "M2"):
        totals = audit_chunks(arm, registration["arms"][arm], source["schedules"][arm], pools[arm], records)
        require(totals == registration["arms"][arm]["totals"], "whole arm dose")
    require(registration == expected, "closed draft/shared study contract changed")
    return {"status": "PASS_CPU_STUDY_SCHEDULE_AUDIT_NOT_LAUNCH_PERMISSION", "arms": {arm: registration["arms"][arm]["totals"] for arm in ("M1", "M2")},
            "native_launch_authorized": False, "qualification_STATE1_warmstart_allowed": False}


def write_draft(path):
    path = Path(path).resolve()
    require(path.parent == PACKAGE and path.name == "STUDY_DEV_REGISTRATION_DRAFT.json", "owned draft filename")
    body = build_registration()
    validate_registration(body)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(body, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return body
