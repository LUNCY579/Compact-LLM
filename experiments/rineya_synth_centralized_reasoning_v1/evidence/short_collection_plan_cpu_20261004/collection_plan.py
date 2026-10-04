"""CPU batching plan; no collector, model, scorer or native permission."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE))
from inference_component import validate_row
from rs_contract import canonical, digest, require

SOURCE_PINS = {
    "atomic_RS_DEV": ("evidence/fresh_dev_compact002_20261003/prompt_token_inputs_cpu.json",
        "4664c1770a88a6345c6bfa02cad3c7bdf667c3037f8811364049b503a6b90a4f", 384),
    "task_guard_DEV": ("evidence/task_guards_root1_20261004/policy/prompts.json",
        "780fefe724554e3a071129e2159d7fc18fee061e9957b449073363acc8c92a78", 768)}
ROWS_PER_BATCH, CONTEXT_CAP, GENERATION_CAP = 64, 512, 32
TASK_FIELDS = {"row_id", "messages", "input_ids", "attention_mask", "prompt_sha256",
               "prompt_token_sha256", "prompt_tokens"}


def cpu_only():
    require(not any(name.split(".", 1)[0] in {"torch", "transformers", "jax", "tensorflow", "flax"}
                    for name in sys.modules), "CPU plan cannot import model frameworks")


def read_source(kind):
    cpu_only()
    relative, pin, count = SOURCE_PINS[kind]
    path = PACKAGE / relative
    require(path.is_file() and not path.is_symlink(), "pinned public CPU input source required")
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    require(len(raw) <= 16 * 1024**2 and hashlib.sha256(raw).hexdigest() == pin, "CPU input source byte drift")
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    require(type(value) is dict and value.get("native_launch_authorized") is False
            and type(value.get("rows")) is list and len(value["rows"]) == count, "complete closed CPU source")
    if kind == "atomic_RS_DEV":
        require(value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"})
                and value.get("status") == "CPU_INPUTS_PREPARED_NATIVE_UNADMITTED"
                and value.get("context_cap") == CONTEXT_CAP and value.get("maximum_generated_tokens") == GENERATION_CAP
                and value.get("oracle_inputs_read") is False and value.get("model_outputs_used") is False
                and value.get("overlong_row_ids") == [] and value.get("rows_dropped") == 0
                and value.get("rows_truncated") == 0, "complete prompt-only DEV preparation required")
        rows = deepcopy(value["rows"])
    else:
        rows = []
        for row in value["rows"]:
            require(type(row) is dict and set(row) == TASK_FIELDS
                    and row["prompt_sha256"] == digest(row["messages"])
                    and row["prompt_token_sha256"] == digest(row["input_ids"]), "task prompt/message/token binding")
            rows.append({"row_id": row["row_id"], "prompt_sha256": row["prompt_sha256"],
                "prompt_ids": deepcopy(row["input_ids"]), "prompt_tokens": row["prompt_tokens"],
                "attention_mask": deepcopy(row["attention_mask"])})
    for row in rows:
        validate_row(row)
    require(len({row["row_id"] for row in rows}) == count, "duplicate source rows")
    return rows


def batches(rows, kind):
    """Fixed whole-row boundaries, independent of outputs and token lengths."""
    cpu_only()
    require(type(rows) is list and bool(rows) and kind in SOURCE_PINS, "explicit source population")
    for row in rows:
        validate_row(row)
    require(len({row["row_id"] for row in rows}) == len(rows), "duplicate rows")
    result = []
    for start in range(0, len(rows), ROWS_PER_BATCH):
        packet = deepcopy(rows[start:start + ROWS_PER_BATCH])
        # Each step sees L+t input tokens, t=0..31. This includes cap-truncated
        # outputs; EOS may reduce actual cost but never increases the frozen cap.
        positions = sum(GENERATION_CAP * row["prompt_tokens"]
                        + GENERATION_CAP * (GENERATION_CAP - 1) // 2 for row in packet)
        result.append({"ordinal": len(result) + 1, "source_kind": kind, "start": start,
            "stop": start + len(packet), "rows": packet, "source_packet_sha256": digest(packet),
            "maximum_capture_forwards": len(packet) * GENERATION_CAP,
            "maximum_capture_input_positions": positions,
            "maximum_zero_parity_forwards": 2,
            "maximum_zero_parity_input_positions": 2 * packet[0]["prompt_tokens"],
            "maximum_total_forwards": len(packet) * GENERATION_CAP + 2,
            "maximum_total_input_positions": positions + 2 * packet[0]["prompt_tokens"],
            "native_launch_authorized": False, "scoring_authorized": False})
    require([row["row_id"] for batch in result for row in batch["rows"]]
            == [row["row_id"] for row in rows], "whole source order/denominator preserved")
    return result


def build():
    populations = {kind: read_source(kind) for kind in SOURCE_PINS}
    all_ids = [row["row_id"] for rows in populations.values() for row in rows]
    require(len(set(all_ids)) == len(all_ids), "populations share row identities")
    plan = {"schema_version": 1, "status": "CPU_COLLECTION_PLAN_NATIVE_UNAUTHORIZED",
        "scope": "fresh short DEV and task prompts; no language or long-context substitution",
        "source_pins": deepcopy(SOURCE_PINS), "context_cap": CONTEXT_CAP,
        "maximum_generated_tokens": GENERATION_CAP, "rows_per_batch": ROWS_PER_BATCH,
        "initial_model": "fresh unchanged M0 only; no qualification or study checkpoint",
        "batches": {kind: batches(rows, kind) for kind, rows in populations.items()},
        "rows": len(all_ids), "row_order_sha256": digest(all_ids),
        "work_bounds_are_not_runtime_or_resource_fit": True,
        "native_launch_authorized": False, "scoring_authorized": False,
        "production_training_authorized": False, "language_qualification_established": False,
        "long_context_qualification_established": False, "prior_correct_enrollment_performed": False,
        "protected_bank_access": False, "confirmation_eligible": False}
    plan["sha256"] = digest(plan)
    cpu_only()
    return plan


if __name__ == "__main__":
    result = build()
    target = Path(__file__).resolve().parent / "short_M0_collection_plan_cpu.json"
    with target.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(canonical(result) + "\n")
    print(canonical({"status": result["status"], "rows": result["rows"],
        "batches": {kind: len(items) for kind, items in result["batches"].items()},
        "maximum_batch_forwards": max(b["maximum_total_forwards"] for items in result["batches"].values() for b in items),
        "native_launch_authorized": False}))
