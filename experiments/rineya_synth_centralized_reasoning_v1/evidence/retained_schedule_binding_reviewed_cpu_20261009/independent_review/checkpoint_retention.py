"""Closed SYNTH retained-step metadata binding; no payload I/O or native authority.

The caller supplies an exact 27-step cumulative schedule and its separately
frozen canonical SHA256. This module never derives per-step doses from totals,
opens descriptor paths, imports producer code, or executes an exporter/loader.
Producer checkpoint kind/schema are exact external pins, not a new native
protocol. An accepted result establishes only consistency of supplied metadata.
File hashes and native-looking labels cannot attest original execution or parity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEDULE_SCHEMA = "synth_checkpoint_retention_schedule_v1"
MANIFEST_SCHEMA = "synth_checkpoint_retention_binding_v1"
CHECKPOINT_SCHEMA = "synth_retained_step_descriptor_v1"
EXPORT_SCHEMA = "synth_gd_peft_export_descriptor_v1"
PARITY_SCHEMA = "synth_gd_parity_receipt_join_v1"
SCIENTIFIC_SCHEMA = "synth_scientific_boundary_descriptor_join_v1"
ARMS = ("M1", "M2")
STEPS = list(range(1, 28))
RESUME_STEPS = [9, 18, 27]
GD_STEPS = [9, 27]
MAX_STUDY_STATE_BYTES = 32 * 1024 * 1024
OWNER_RETAINED_KIND = "SYNTH_STUDY_RETAINED_UPDATE_V1"
OWNER_RECORD_KIND = "SYNTH_STUDY_RETAINED_UPDATE_RECORD_V1"
OWNER_SCIENTIFIC_KIND = "SYNTH_STUDY_TORCH_BOUNDARY_V1"
OWNER_COST_SCOPE = "SYNTH_STUDY_CHUNK_PROTOCOL_V1"
HASH_FIELDS = {
    "recipe_sha256", "source_sha256", "runtime_sha256", "data_sha256",
    "base_content_sha256", "tokenizer_content_sha256",
    "cohort_sha256", "backend_source_sha256",
}
COUNTER_FIELDS = {
    "completed_updates", "consumed_occurrences", "input_tokens",
    "supervised_tokens", "consumed_order_sha256", "next_update_ordinal",
    "learning_rate",
}
METADATA_FIELDS = {
    "schema_version", "kind", "experiment", "scope", "arm", "seed",
    "recipe_sha256", "cohort_sha256", "backend_source_sha256",
    "fresh_start_attestation_sha256", "adapter_parameter_order",
    "adapter_inventory_sha256", "native_launch_authorized", "training_authorized",
    "native_resume_qualified", "qualification_STATE1_used", "producer_policy_sha256", "resumable",
}
PROTOCOL_COUNTS = {
    "gradient_reset": 27, "forward": 540, "completion_loss": 540, "backward": 540,
    "finite_gradient_clip": 27, "optimizer_step": 27, "checkpoint_save": 3,
    "checkpoint_reload": 2, "checkpoint_publish": 3, "terminal_costs_publish": 3,
    "checkpoint_promote": 3,
}
REQUIRED_EXTERNAL_VERIFICATION = (
    "Verify every retained original payload byte, full state, and owner lineage.",
    "Verify actual standard-PEFT export files and exact source tensor parity.",
    "Independently verify strict native loading of the exact base and steps9/27.",
    "Verify actual tokenizer/prompt, greedy generation, and logit parity traces.",
    "Run GD attachment/parity checks with exact original consumed/exposure provenance.",
    "Obtain separate owner authorization and qualification for native operations.",
)


class ValidationError(ValueError):
    """An incomplete, stale, failed, or nonclosed metadata binding was refused."""


def _require(condition, label):
    if not condition:
        raise ValidationError(label)


def canonical(value):
    """Sidecar/rs_contract canonical bytes (ASCII escapes); exact JSON types only."""
    try:
        _json_shape(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError) as error:
        raise ValidationError("metadata must be finite JSON") from error


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _json_shape(value):
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "JSON object keys must be strings")
        for child in value.values():
            _json_shape(child)
    elif type(value) is list:
        for child in value:
            _json_shape(child)
    else:
        _require(value is None or type(value) in (str, bool, int, float),
                 "only exact JSON container/scalar types are permitted")


def producer_digest(value):
    """Exact study_schedule/backend digest, whose canonical encoding uses UTF-8."""
    canonical(value)  # Shared closed-JSON validation, with no normalization.
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError) as error:
        raise ValidationError("producer metadata must be finite UTF-8 JSON") from error
    return hashlib.sha256(data).hexdigest()


def _same(actual, expected, label):
    _require(canonical(actual) == canonical(expected), label)


def _object(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys), label + ": exact fields required")
    return value


def _text(value, label):
    _require(type(value) is str and bool(value.strip()) and value == value.strip()
             and all(ord(c) >= 32 for c in value), label + ": nonempty text required")
    _require(value.casefold() not in {"pending", "tbd", "unresolved", "unknown", "placeholder", "null", "none"},
             label + ": pending/unresolved textual pins refused")
    return value


def _hash(value, label):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             label + ": exact lowercase SHA256 pin required; pending is refused")
    return value


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label + ": integer required")
    return value


def _path(value, label, original):
    value = _text(value, label)
    normalized = value.replace("\\", "/")
    _require(re.match(r"^[A-Za-z]:/", normalized) is not None,
             label + ": stable absolute Windows path required")
    parts = normalized[3:].split("/")
    _require(parts and all(p and p not in (".", "..") and not p.endswith((".", " "))
                           and not any(c in p for c in ':*?"<>|') for p in parts),
             label + ": aliased/relative/device path refused")
    reserved = {"con", "prn", "aux", "nul", "conin$", "conout$"}
    reserved |= {"com" + str(i) for i in range(1, 10)} | {"lpt" + str(i) for i in range(1, 10)}
    reserved |= {"com" + suffix for suffix in "¹²³"} | {"lpt" + suffix for suffix in "¹²³"}
    _require(all(p.split(".", 1)[0].casefold() not in reserved for p in parts),
             label + ": Windows device path refused")
    low = normalized.casefold()
    _require("state1" not in low and "full27" not in low and "qualification" not in low,
             label + ": qualification STATE1/FULL27 artifacts refused")
    if original:
        _require(not any(token in low for token in ("fixture", "toy", "stub")),
                 label + ": fixture artifact cannot masquerade as original lineage")
    return normalized


def _file(value, label, seen_paths, original):
    _object(value, {"path", "sha256", "bytes"}, label)
    path = _path(value["path"], label + ".path", original)
    _hash(value["sha256"], label + ".sha256")
    _integer(value["bytes"], label + ".bytes", 1)
    key = path.casefold()
    _require(key not in seen_paths, label + ": duplicate retained file path")
    seen_paths.add(key)
    return path


def _identity(value, label):
    _object(value, HASH_FIELDS | {"trajectory_id"}, label)
    _text(value["trajectory_id"], label + ".trajectory_id")
    for key in HASH_FIELDS:
        _hash(value[key], label + "." + key)


def _counter(value, step, previous, label):
    _object(value, COUNTER_FIELDS, label)
    for key in COUNTER_FIELDS - {"consumed_order_sha256", "learning_rate"}:
        _integer(value[key], label + "." + key)
    _hash(value["consumed_order_sha256"], label + ".consumed_order_sha256")
    _require(value["completed_updates"] == step and value["consumed_occurrences"] == step * 20
             and value["next_update_ordinal"] == step + 1,
             label + ": exact completed update/20-occurrence cursor required")
    _require(type(value["learning_rate"]) is float and math.isfinite(value["learning_rate"])
             and value["learning_rate"] >= 0, label + ": finite float learning_rate required")
    inputs, targets = value["input_tokens"], value["supervised_tokens"]
    _require(previous["input_tokens"] < inputs <= 131072
             and previous["supervised_tokens"] < targets <= inputs,
             label + ": cumulative token counters invalid")
    _require(targets - previous["supervised_tokens"] <= inputs - previous["input_tokens"],
             label + ": per-step supervised tokens exceed per-step input tokens")
    _require(value["consumed_order_sha256"] != previous.get("consumed_order_sha256"),
             label + ": stale occurrence-order prefix hash")


def validate_schedule(frozen_schedule, expected_schedule_sha256):
    """Validate an externally pinned, fully resolved descriptor-only schedule.

    A pending prospective draft is deliberately not an executable schedule.
    No native permission is issued, even for frozen_owner_schedule_metadata.
    """
    _hash(expected_schedule_sha256, "expected_schedule_sha256")
    _require(digest(frozen_schedule) == expected_schedule_sha256,
             "frozen schedule differs from external SHA256 pin")
    _object(frozen_schedule, {
        "schema", "evidence_kind", "experiment", "seed", "occurrences_per_arm",
        "updates_per_arm", "occurrences_per_update", "retained_steps",
        "scientific_resume_boundaries", "gd_original_steps", "producer_checkpoint", "scientific_checkpoint",
        "producer_record", "producer_policy_sha256", "cost_scope", "learning_rates", "arms",
    }, "schedule")
    fixed = {"schema": SCHEDULE_SCHEMA, "experiment": "SYNTH-CRS-001", "seed": 101,
             "occurrences_per_arm": 540, "updates_per_arm": 27, "occurrences_per_update": 20,
             "retained_steps": STEPS, "scientific_resume_boundaries": RESUME_STEPS,
             "gd_original_steps": GD_STEPS}
    for key, value in fixed.items():
        _same(frozen_schedule[key], value, "schedule." + key + ": accepted plan changed")
    kind = frozen_schedule["evidence_kind"]
    _require(kind in ("frozen_owner_schedule_metadata", "mechanical_fixture"),
             "schedule.evidence_kind: unsupported lineage")
    for format_name in ("producer_checkpoint", "producer_record", "scientific_checkpoint"):
        format_pin = frozen_schedule[format_name]
        _object(format_pin, {"kind", "schema_version"}, "schedule." + format_name)
        _integer(format_pin["schema_version"], "schedule." + format_name + ".schema_version", 1)
        value = _text(format_pin["kind"], "schedule." + format_name + ".kind")
        low = value.casefold()
        _require(not any(t in low for t in ("state1", "full27", "qualification")),
                 "qualification checkpoint/record kind refused")
        if kind != "mechanical_fixture":
            _require(not any(t in low for t in ("fixture", "toy", "stub")),
                     "fixture checkpoint/record kind cannot bind original lineage")
    _hash(frozen_schedule["producer_policy_sha256"], "schedule.producer_policy_sha256")
    _text(frozen_schedule["cost_scope"], "schedule.cost_scope")
    _require(frozen_schedule["scientific_checkpoint"]["kind"] != frozen_schedule["producer_checkpoint"]["kind"],
             "scientific and nonresumable retained checkpoint kinds must be distinct")
    if kind != "mechanical_fixture":
        for name, required_kind in (("producer_checkpoint", OWNER_RETAINED_KIND),
                                    ("producer_record", OWNER_RECORD_KIND),
                                    ("scientific_checkpoint", OWNER_SCIENTIFIC_KIND)):
            _same(frozen_schedule[name], {"kind": required_kind, "schema_version": 1},
                  "schedule." + name + ": exact frozen owner kind/schema_version1 required")
        _same(frozen_schedule["cost_scope"], OWNER_COST_SCOPE,
              "original native cost scope differs from frozen owner contract")
    rates = frozen_schedule["learning_rates"]
    _require(type(rates) is list and len(rates) == 27, "schedule.learning_rates: exact27 rows required")
    for step, rate in enumerate(rates, 1):
        _object(rate, {"update", "numerator", "denominator", "float_value"}, "schedule.learning_rates")
        _integer(rate["update"], "learning rate update", 1)
        _integer(rate["numerator"], "learning rate numerator")
        _integer(rate["denominator"], "learning rate denominator", 1)
        _require(rate["update"] == step and type(rate["float_value"]) is float
                 and math.isfinite(rate["float_value"])
                 and rate["float_value"] == rate["numerator"] / rate["denominator"],
                 "exact externally frozen learning rate fraction/float mismatch")
    _object(frozen_schedule["arms"], ARMS, "schedule.arms")
    trajectories = set()
    for arm in ARMS:
        spec = frozen_schedule["arms"][arm]
        label = "schedule.arms." + arm
        _object(spec, {"identity", "metadata_pins", "scientific_metadata_pins", "gd_parity", "counters"}, label)
        _identity(spec["identity"], label + ".identity")
        metadata = spec["metadata_pins"]
        _object(metadata, METADATA_FIELDS, label + ".metadata_pins")
        expected = {"schema_version": frozen_schedule["producer_checkpoint"]["schema_version"],
                    "kind": frozen_schedule["producer_checkpoint"]["kind"],
                    "experiment": "SYNTH-CRS-001", "arm": arm, "seed": 101,
                    "recipe_sha256": spec["identity"]["recipe_sha256"],
                    "cohort_sha256": spec["identity"]["cohort_sha256"],
                    "backend_source_sha256": spec["identity"]["backend_source_sha256"],
                    "producer_policy_sha256": frozen_schedule["producer_policy_sha256"],
                    "resumable": False, "native_launch_authorized": False,
                    "training_authorized": False, "native_resume_qualified": False,
                    "qualification_STATE1_used": False}
        for key, value in expected.items():
            _same(metadata[key], value, label + ".metadata_pins." + key + ": source/permission pin mismatch")
        scientific_metadata = spec["scientific_metadata_pins"]
        _object(scientific_metadata, METADATA_FIELDS - {"producer_policy_sha256", "resumable"},
                label + ".scientific_metadata_pins")
        expected_scientific = {k: v for k, v in metadata.items() if k not in ("producer_policy_sha256", "resumable")}
        expected_scientific.update(frozen_schedule["scientific_checkpoint"])
        _same(scientific_metadata, expected_scientific,
              label + ": scientific source/recipe/fresh-start pins differ from retained lineage")
        scope = _text(metadata["scope"], label + ".metadata_pins.scope")
        if kind != "mechanical_fixture":
            _same(scope, "UNQUALIFIED_NATIVE_IMPLEMENTATION",
                  "fixture/unknown metadata scope cannot masquerade as original native lineage")
        for key in ("fresh_start_attestation_sha256", "adapter_inventory_sha256"):
            _hash(metadata[key], label + ".metadata_pins." + key)
        order = metadata["adapter_parameter_order"]
        _require(type(order) is list and bool(order) and all(type(n) is str and n for n in order)
                 and len(set(order)) == len(order), "exact unique adapter parameter-order pins required")
        trajectory = spec["identity"]["trajectory_id"]
        _require(trajectory not in trajectories, "arm trajectories must be distinct")
        trajectories.add(trajectory)
        _object(spec["gd_parity"], {"policy_sha256", "probe_registration_sha256"}, label + ".gd_parity")
        for key, value in spec["gd_parity"].items():
            _hash(value, label + ".gd_parity." + key)
        counters = spec["counters"]
        _require(type(counters) is list and len(counters) == 27,
                 label + ".counters: exact externally supplied 27-step schedule required")
        previous = {"input_tokens": 0, "supervised_tokens": 0}
        order_hashes = set()
        for step, counter in enumerate(counters, 1):
            _counter(counter, step, previous, label + ".counters[" + str(step) + "]")
            _same(counter["learning_rate"], rates[step - 1]["float_value"],
                  label + ".counters: learning rate differs from externally frozen27-row schedule")
            _require(counter["consumed_order_sha256"] not in order_hashes,
                     label + ": duplicate/stale cumulative occurrence-order prefix hash")
            order_hashes.add(counter["consumed_order_sha256"])
            previous = counter
    return {"schedule_sha256": expected_schedule_sha256, "structural_only": True,
            "native_admissible": False, "payloads_read": False}


def roles_for_step(step):
    _require(type(step) is int and step in STEPS, "retained step must be1..27")
    roles = ["retained_step"]
    if step in RESUME_STEPS:
        roles.append("scientific_boundary_step")
    if step in GD_STEPS:
        roles.append("gd_original_snapshot_source")
    return roles


def _counts(inventory, label):
    _require(type(inventory) is dict and bool(inventory), label + ": nonempty operation inventory required")
    for operation, counts in inventory.items():
        _text(operation, label + ".operation")
        _object(counts, {"attempted", "returned", "failed", "pending"}, label + "." + operation)
        for key, value in counts.items():
            _integer(value, label + "." + operation + "." + key)
        _require(counts["failed"] == counts["pending"] == 0
                 and counts["attempted"] == counts["returned"],
                 label + ": failed/pending/incomplete operation lineage refused")


def _costs(costs, counter, step, cost_scope, previous):
    label = "retained posted-save costs[" + str(step) + "]"
    _object(costs, {"schema_version", "scope", "protocol", "positions", "primitive_operations"}, label)
    _same(costs["schema_version"], 1, label + ": schema_version")
    _same(costs["scope"], cost_scope, label + ": scope differs from external pin")
    _object(costs["protocol"], PROTOCOL_COUNTS, label + ".protocol")
    for inventory in ("protocol", "primitive_operations"):
        _counts(costs[inventory], label + "." + inventory)
        if previous is not None:
            for name, counts in previous[inventory].items():
                _require(name in costs[inventory]
                         and all(costs[inventory][name][key] >= value for key, value in counts.items()),
                         label + ": cumulative operation lineage cannot regress/disappear")
    chunks = (step - 1) // 9
    expected_counts = {
        "gradient_reset": step, "forward": step * 20, "completion_loss": step * 20,
        "backward": step * 20, "finite_gradient_clip": step, "optimizer_step": step,
        "checkpoint_save": chunks, "checkpoint_reload": chunks,
        "checkpoint_publish": chunks, "terminal_costs_publish": chunks, "checkpoint_promote": chunks,
    }
    for name, number in expected_counts.items():
        _same(costs["protocol"][name], {"attempted": number, "returned": number, "failed": 0, "pending": 0},
              label + ": original step/protocol cost binding " + name)
    expected_positions = {
        "input_attempted": counter["input_tokens"], "input_returned": counter["input_tokens"],
        "supervised_attempted": counter["supervised_tokens"], "supervised_returned": counter["supervised_tokens"],
    }
    _same(costs["positions"], expected_positions, label + ": token counter/cost mismatch")
    for name in ("retained_update_path", "retained_update_save", "retained_update_verify",
                 "retained_update_publish_write", "retained_update_publish_verify"):
        number = step - int(name.endswith(("publish_write", "publish_verify")))
        _same(costs["primitive_operations"].get(name), {
            "attempted": number, "returned": number, "failed": 0, "pending": 0,
        }, label + ": incomplete retained-save/publication prefix " + name)
    requests = sum(s < step for s in GD_STEPS)
    for name in ("GD_export_parity_request_write", "GD_export_parity_request_verify"):
        zero = {"attempted": 0, "returned": 0, "failed": 0, "pending": 0}
        _same(costs["primitive_operations"].get(name, zero), {
            "attempted": requests, "returned": requests, "failed": 0, "pending": 0,
        }, label + ": GD request cost binding")


def _publication(value, record, label, seen_paths, original):
    _object(value, {"path", "bytes", "sha256", "payload_sha256", "durable"}, label)
    _same(value["durable"], True, label + ": completed publication required")
    _same(value["payload_sha256"], producer_digest(record), label + ": stale publication record digest")
    _file({k: value[k] for k in ("path", "sha256", "bytes")}, label, seen_paths, original)


def _checkpoint(record, arm, step, spec, schedule, evidence, seen_paths, checkpoint_hashes, previous_costs):
    label = arm + ".checkpoint[" + str(step) + "]"
    _object(record, {"schema", "evidence_kind", "artifact_role", "arm", "seed", "step",
                     "producer_checkpoint", "identity", "roles", "counters", "save_status",
                     "lineage_status", "native_descriptor", "producer_record", "publication",
                     "GD_export_parity_request_publication"}, label)
    original = evidence == "producer_metadata"
    expected = {"schema": CHECKPOINT_SCHEMA, "evidence_kind": evidence, "arm": arm,
                "seed": 101, "step": step, "producer_checkpoint": schedule["producer_checkpoint"],
                "identity": spec["identity"], "roles": roles_for_step(step),
                "counters": spec["counters"][step - 1], "save_status": "SAVE_RETURNED",
                "lineage_status": "COMPLETE",
                "artifact_role": "original_study_checkpoint" if original else "mechanical_fixture_checkpoint"}
    for key, value in expected.items():
        _same(record[key], value, label + "." + key + ": stale/failed/foreign binding")
    descriptor = record["native_descriptor"]
    _object(descriptor, {"schema_version", "kind", "path", "bytes", "sha256", "metadata", "metadata_sha256"},
            label + ".native_descriptor")
    for key in ("kind", "schema_version"):
        _same(descriptor[key], schedule["producer_checkpoint"][key],
              label + ": native descriptor format differs from external owner pin")
    expected_metadata = dict(spec["metadata_pins"], binding=spec["counters"][step - 1])
    _same(descriptor["metadata"], expected_metadata, label + ": native metadata source/counter/permission pins mismatch")
    _same(descriptor["metadata_sha256"], producer_digest(expected_metadata), label + ": stale native metadata digest")
    _file({k: descriptor[k] for k in ("path", "sha256", "bytes")}, label + ".payload", seen_paths, original)
    _require(descriptor["bytes"] <= MAX_STUDY_STATE_BYTES,
             label + ": frozen backend32MiB checkpoint byte ceiling")
    payload_hash = descriptor["sha256"]
    _require(payload_hash not in checkpoint_hashes, label + ": duplicate original checkpoint payload hash")
    checkpoint_hashes.add(payload_hash)
    native_record = record["producer_record"]
    _object(native_record, {"schema_version", "kind", "arm", "seed", "recipe_sha256",
                           "producer_policy_sha256", "binding", "backend_descriptor", "posted_save_costs",
                           "GD_export_parity_required", "GD_export_parity_qualified", "sha256"},
            label + ".producer_record")
    expected_record = {
        "schema_version": schedule["producer_record"]["schema_version"],
        "kind": schedule["producer_record"]["kind"], "arm": arm, "seed": 101,
        "recipe_sha256": spec["identity"]["recipe_sha256"],
        "producer_policy_sha256": schedule["producer_policy_sha256"],
        "binding": spec["counters"][step - 1], "backend_descriptor": descriptor,
        "GD_export_parity_required": step in GD_STEPS, "GD_export_parity_qualified": False,
    }
    for key, value in expected_record.items():
        _same(native_record[key], value, label + ": native retained record binding " + key)
    _same(native_record["sha256"], producer_digest({k: v for k, v in native_record.items() if k != "sha256"}),
          label + ": stale native retained record digest")
    _costs(native_record["posted_save_costs"], native_record["binding"], step,
           schedule["cost_scope"], previous_costs)
    _publication(record["publication"], native_record, label + ".publication", seen_paths, original)
    request = record["GD_export_parity_request_publication"]
    if step in GD_STEPS:
        _publication(request, native_record, label + ".GD_request", seen_paths, original)
    else:
        _same(request, None, label + ": GD requests only at original9/27")
    return native_record["posted_save_costs"]


def _scientific(record, arm, step, spec, schedule, retained_checkpoint, evidence, seen_paths, payload_hashes):
    label = arm + ".scientific_boundary[" + str(step) + "]"
    _object(record, {"schema", "evidence_kind", "arm", "step", "identity", "counters",
                     "role", "retained_step_descriptor_sha256", "native_descriptor"}, label)
    expected = {
        "schema": SCIENTIFIC_SCHEMA, "evidence_kind": evidence, "arm": arm, "step": step,
        "identity": spec["identity"], "counters": spec["counters"][step - 1],
        "role": "registered_scientific_resume_boundary",
        "retained_step_descriptor_sha256": digest(retained_checkpoint),
    }
    for key, value in expected.items():
        _same(record[key], value, label + ": distinct scientific boundary join " + key)
    descriptor = record["native_descriptor"]
    _object(descriptor, {"schema_version", "kind", "path", "bytes", "sha256", "metadata", "metadata_sha256"},
            label + ".native_descriptor")
    for key in ("kind", "schema_version"):
        _same(descriptor[key], schedule["scientific_checkpoint"][key],
              label + ": retained archives cannot substitute for original scientific boundary kind/schema")
    metadata = dict(spec["scientific_metadata_pins"], binding=spec["counters"][step - 1])
    _same(descriptor["metadata"], metadata, label + ": scientific source/recipe/cursor/fresh-start mismatch")
    _same(descriptor["metadata_sha256"], producer_digest(metadata), label + ": scientific metadata digest mismatch")
    _file({k: descriptor[k] for k in ("path", "sha256", "bytes")}, label + ".payload",
          seen_paths, evidence == "producer_metadata")
    _require(descriptor["bytes"] <= MAX_STUDY_STATE_BYTES,
             label + ": frozen backend32MiB scientific checkpoint byte ceiling")
    _require(descriptor["sha256"] not in payload_hashes,
             label + ": scientific payload must be distinct from retained payloads")
    payload_hashes.add(descriptor["sha256"])


def _export(record, arm, step, spec, checkpoint, evidence, seen_paths, directories):
    label = arm + ".gd_export[" + str(step) + "]"
    _object(record, {"schema", "evidence_kind", "artifact_role", "arm", "step", "identity",
                     "source_checkpoint_sha256", "format", "directory", "files", "inventory_sha256",
                     "snapshot_receipt", "export_receipt", "status"}, label)
    original = evidence == "producer_metadata"
    expected = {"schema": EXPORT_SCHEMA, "evidence_kind": evidence, "arm": arm, "step": step,
                "identity": spec["identity"], "source_checkpoint_sha256": digest(checkpoint),
                "format": "standard_peft_lora" if original else "inert_fixture",
                "artifact_role": "standard_peft_payload_metadata" if original else "mechanical_fixture_peft_payload",
                "status": "EXPORT_REPORTED_COMPLETE"}
    for key, value in expected.items():
        _same(record[key], value, label + "." + key + ": stale/foreign export binding")
    directory = _path(record["directory"], label + ".directory", original)
    _require(directory.casefold() not in directories, label + ": reused export directory")
    directories.add(directory.casefold())
    files = record["files"]
    _require(type(files) is list and len(files) == 2, label + ": exact config/weights inventory required")
    for entry, role, name in zip(files, ("adapter_config", "adapter_weights"),
                                 ("adapter_config.json", "adapter_model.safetensors")):
        _object(entry, {"role", "file"}, label + ".files")
        _same(entry["role"], role, label + ": ordered standard-PEFT roles required")
        path = _file(entry["file"], label + "." + role, seen_paths, original)
        _same(path.casefold(), (directory + "/" + name).casefold(), label + ": payload outside unique PEFT directory")
    _hash(record["inventory_sha256"], label + ".inventory_sha256")
    _same(record["inventory_sha256"], digest(files), label + ": PEFT inventory metadata digest changed")
    for key in ("snapshot_receipt", "export_receipt"):
        _file(record[key], label + "." + key, seen_paths, original)


def _parity(join, arm, spec, checkpoints, exports, evidence, seen_paths):
    label = arm + ".parity_join"
    _object(join, {"schema", "evidence_kind", "arm", "identity", "parity_policy_sha256",
                   "probe_registration_sha256", "joins", "receipts", "receipt_schema", "evidence_level"}, label)
    expected = {"schema": PARITY_SCHEMA, "evidence_kind": evidence, "arm": arm,
                "identity": spec["identity"], "parity_policy_sha256": spec["gd_parity"]["policy_sha256"],
                "probe_registration_sha256": spec["gd_parity"]["probe_registration_sha256"],
                "receipt_schema": "gd001_export_parity_result_v1",
                "evidence_level": "native_pilot" if evidence == "producer_metadata" else "mechanical_fixture",
                "joins": [{"step": step, "source_checkpoint_sha256": digest(checkpoints[step - 1]),
                           "peft_export_sha256": digest(export)} for step, export in zip(GD_STEPS, exports)]}
    for key, value in expected.items():
        _same(join[key], value, label + "." + key + ": stale/foreign parity receipt binding")
    receipts = join["receipts"]
    _object(receipts, {"parity_result", "producer_trace", "gd_trace", "strict_load"}, label + ".receipts")
    for key, value in receipts.items():
        _file(value, label + ".receipts." + key, seen_paths, evidence == "producer_metadata")


def validate_retention(manifest, frozen_schedule, expected_schedule_sha256):
    """Return closed structural validation or raise ValidationError.

    All file descriptors remain unverified assertions. Even native_pilot parity
    receipt metadata is only joined, never executed, read, compared or admitted.
    Unknown keys, including capability/admission/parity-success booleans, refuse.
    """
    validate_schedule(frozen_schedule, expected_schedule_sha256)
    _object(manifest, {"schema", "evidence_kind", "schedule_sha256", "arms"}, "manifest")
    evidence = ("mechanical_fixture" if frozen_schedule["evidence_kind"] == "mechanical_fixture"
                else "producer_metadata")
    for key, value in {"schema": MANIFEST_SCHEMA, "evidence_kind": evidence,
                       "schedule_sha256": expected_schedule_sha256}.items():
        _same(manifest[key], value, "manifest." + key + ": stale/foreign schedule binding")
    _object(manifest["arms"], ARMS, "manifest.arms")
    seen_paths, checkpoint_hashes, directories = set(), set(), set()
    for arm in ARMS:
        body, spec = manifest["arms"][arm], frozen_schedule["arms"][arm]
        label = "manifest.arms." + arm
        _object(body, {"identity", "lineage_status", "completed_updates", "failed_operations",
                       "pending_operations", "incomplete_saves", "completion_receipt", "checkpoints",
                       "scientific_boundaries", "gd_exports", "parity_join"}, label)
        for key, value in {"identity": spec["identity"], "lineage_status": "COMPLETED",
                           "completed_updates": 27, "failed_operations": 0, "pending_operations": 0,
                           "incomplete_saves": 0}.items():
            _same(body[key], value, label + "." + key + ": failed/incomplete lineage refused")
        _file(body["completion_receipt"], label + ".completion_receipt", seen_paths, evidence == "producer_metadata")
        checkpoints = body["checkpoints"]
        _require(type(checkpoints) is list and len(checkpoints) == 27,
                 label + ".checkpoints: retain every original step1..27; boundaries alone are insufficient")
        previous_costs = None
        for step, record in enumerate(checkpoints, 1):
            previous_costs = _checkpoint(record, arm, step, spec, frozen_schedule, evidence,
                                         seen_paths, checkpoint_hashes, previous_costs)
        scientific = body["scientific_boundaries"]
        _require(type(scientific) is list and len(scientific) == 3,
                 label + ": all three distinct scientific boundary descriptors9/18/27 required")
        for step, record in zip(RESUME_STEPS, scientific):
            _scientific(record, arm, step, spec, frozen_schedule, checkpoints[step - 1],
                        evidence, seen_paths, checkpoint_hashes)
        exports = body["gd_exports"]
        _require(type(exports) is list and len(exports) == 2,
                 label + ".gd_exports: both original steps9/27 required")
        for step, record in zip(GD_STEPS, exports):
            _export(record, arm, step, spec, checkpoints[step - 1], evidence, seen_paths, directories)
        _parity(body["parity_join"], arm, spec, checkpoints, exports, evidence, seen_paths)
    return {
        "schema": "synth_checkpoint_retention_validation_v1",
        "status": "STRUCTURAL_METADATA_BINDING_VALIDATED",
        "evidence_kind": evidence, "schedule_sha256": expected_schedule_sha256,
        "manifest_sha256": digest(manifest), "retained_steps_per_arm": list(STEPS),
        "scientific_resume_boundaries": list(RESUME_STEPS), "gd_original_steps": list(GD_STEPS),
        "checkpoint_descriptors": 54, "scientific_checkpoint_descriptors": 6,
        "peft_export_descriptors": 4, "parity_receipt_joins": 2,
        "structural_only": True, "payloads_read": False, "receipt_contents_verified": False,
        "native_admissible": False, "native_execution_verified": False,
        "native_export_parity_verified": False, "native_resume_qualified": False,
        "retained_archives_resumable": False, "gd_request_publications_are_parity_evidence": False,
        "training_authorized": False, "capability_claims_admissible": False,
        "required_external_owner_verification": list(REQUIRED_EXTERNAL_VERIFICATION),
        "integrity_limit": "Descriptor consistency and content digests do not prove files exist, original lineage, or native execution.",
    }
