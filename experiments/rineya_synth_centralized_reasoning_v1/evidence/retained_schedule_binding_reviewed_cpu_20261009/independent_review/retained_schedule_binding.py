"""Exact per-update registration metadata bridge; never corpus or native admission."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import re

ARMS = ("M1", "M2")
TOTALS = {"M1": (130562, 5614), "M2": (130574, 5615)}


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate metadata JSON key"); value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite metadata")))


def reference(path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fractions():
    peak = Fraction(1, 10000)
    return [peak / 2, peak] + [peak * Fraction(27 - ordinal, 25) for ordinal in range(3, 28)]


def _exposures(value, total):
    require(type(value) is dict and bool(value)
            and all(type(key) is str and key and type(n) is int and n > 0 for key, n in value.items())
            and sum(value.values()) == total, "exact typed registered parent exposure marginal")
    return Counter(value)


def validate_metadata(registration, lr_metadata):
    require(type(registration) is dict and type(registration.get("schema_version")) is int
            and registration["schema_version"] == 1 and registration.get("experiment") == "SYNTH-CRS-001"
            and registration.get("seed") == 101
            and registration.get("sha256") == digest({k: v for k, v in registration.items() if k != "sha256"}),
            "exact sealed historical registration metadata")
    rates = [{"update": ordinal, "numerator": rate.numerator, "denominator": rate.denominator,
              "float_value": float(rate)} for ordinal, rate in enumerate(fractions(), 1)]
    require(lr_metadata.get("learning_rates") == rates
            and registration["optimizer"]["learning_rates_by_1based_update"] == [row["float_value"] for row in rates],
            "exact registered/frozen27 learning-rate fractions")
    result = {}
    for arm in ARMS:
        specification = registration["arms"][arm]
        updates = specification["updates"]
        require(type(updates) is list and len(updates) == 27 and specification["optimizer_updates"] == 27,
                "all original27 registered update rows required")
        occurrence_ids, cumulative_inputs, cumulative_targets = [], 0, 0
        exposures = Counter(); counters = []; update_groups = []
        for ordinal, update in enumerate(updates, 1):
            require(type(update["update_ordinal"]) is int and update["update_ordinal"] == ordinal
                    and update["first_occurrence_ordinal"] == (ordinal - 1) * 20 + 1
                    and update["last_occurrence_ordinal"] == ordinal * 20 and update["sequences"] == 20,
                    "registered update/occurrence ordinal coverage")
            ids = update["occurrence_row_ids"]
            require(type(ids) is list and len(ids) == 20 and all(type(identity) is str
                    and re.fullmatch(r"[0-9a-f]{64}", identity) for identity in ids), "exact20 registered occurrence IDs")
            inputs, targets = update["input_tokens"], update["supervised_tokens"]
            require(type(inputs) is int and type(targets) is int and 0 < targets <= inputs <= 20 * 512,
                    "typed positive per-update registered token counts")
            require(type(update["learning_rate"]) is float and update["learning_rate"] == rates[ordinal - 1]["float_value"],
                    "per-update global learning rate")
            occurrence_ids.extend(ids); cumulative_inputs += inputs; cumulative_targets += targets
            exposures.update(_exposures(update["parent_exposures"], 20))
            counters.append({"completed_updates": ordinal, "consumed_occurrences": len(occurrence_ids),
                "input_tokens": cumulative_inputs, "supervised_tokens": cumulative_targets,
                "consumed_order_sha256": digest(occurrence_ids), "next_update_ordinal": ordinal + 1,
                "learning_rate": update["learning_rate"]})
            update_groups.append({"update_ordinal": ordinal, "occurrence_row_ids": list(ids),
                "input_tokens": inputs, "supervised_tokens": targets,
                "parent_exposures": dict(sorted(update["parent_exposures"].items()))})
        require(len(occurrence_ids) == 540 and digest(occurrence_ids) == specification["occurrence_order_sha256"],
                "registered full cohort hash and540 occurrence coverage")
        require((cumulative_inputs, cumulative_targets) == TOTALS[arm]
                and specification["totals"]["input_tokens"] == cumulative_inputs
                and specification["totals"]["supervised_tokens"] == cumulative_targets
                and specification["totals"]["sequences"] == 540
                and dict(exposures) == specification["totals"]["parent_exposures"],
                "exact registered final token totals and parent exposure vector")
        chunks = specification["resume_chunks"]
        require(type(chunks) is list and len(chunks) == 3, "all original three scientific chunks")
        chunk_joins = []
        for ordinal, chunk in enumerate(chunks, 1):
            first, last = (ordinal - 1) * 9, ordinal * 9
            selected = updates[first:last]
            ids = [identity for update in selected for identity in update["occurrence_row_ids"]]
            chunk_exposures = Counter()
            for update in selected: chunk_exposures.update(update["parent_exposures"])
            require(type(chunk["chunk_ordinal"]) is int and chunk["chunk_ordinal"] == ordinal
                    and chunk["first_update_ordinal"] == first + 1 and chunk["last_update_ordinal"] == last
                    and chunk["first_occurrence_ordinal"] == first * 20 + 1
                    and chunk["last_occurrence_ordinal"] == last * 20 and chunk["sequences"] == 180
                    and chunk["occurrence_row_ids"] == ids and chunk["occurrence_order_sha256"] == digest(ids)
                    and chunk["input_tokens"] == sum(u["input_tokens"] for u in selected)
                    and chunk["supervised_tokens"] == sum(u["supervised_tokens"] for u in selected)
                    and chunk["parent_exposures"] == dict(chunk_exposures),
                    "exact registered9/18/27 chunk order/token/exposure joins")
            chunk_joins.append({"chunk_ordinal": ordinal, "last_update_ordinal": last,
                "registered_chunk_order_sha256": digest(ids), "cumulative_binding": deepcopy(counters[last - 1]),
                "chunk_input_tokens": chunk["input_tokens"], "chunk_supervised_tokens": chunk["supervised_tokens"],
                "chunk_parent_exposures": deepcopy(chunk["parent_exposures"])})
        result[arm] = {"counters": counters, "update_groups": update_groups, "chunk_joins": chunk_joins,
                       "parent_exposures": dict(sorted(exposures.items())),
                       "cohort_sha256": specification["occurrence_order_sha256"]}
    require(result["M1"]["parent_exposures"] == result["M2"]["parent_exposures"], "matched source-parent marginal vectors")
    return result, rates


def producer_api_join(registration, derived, producer_source):
    tree = ast.parse(Path(producer_source).read_bytes())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("retained_binding", "producer_policy")]
    require(len(definitions) == 2, "selected pure producer functions")
    namespace = {"require": require, "schedule": SimpleNamespace(digest=digest,
                 learning_rates=lambda: [float(rate) for rate in fractions()])}
    policy_kind = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "PRODUCER_POLICY_KIND" for target in node.targets))
    namespace["PRODUCER_POLICY_KIND"] = policy_kind
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(producer_source), "exec"), namespace)
    for arm in ARMS:
        for ordinal, counter in enumerate(derived[arm]["counters"], 1):
            require(namespace["retained_binding"](registration, arm, ordinal) == counter,
                    "actual producer retained_binding metadata API differs")
    return namespace["producer_policy"]()


def build(registration_path, lr_path, producer_paths, structural):
    registration, lr_metadata = read(registration_path), read(lr_path)
    derived, rates = validate_metadata(registration, lr_metadata)
    policy = producer_api_join(registration, derived, producer_paths["study_native_chunk.py"])
    pins = {name: reference(path) for name, path in producer_paths.items()}
    pending_identity = {name: None for name in structural.HASH_FIELDS | {"trajectory_id"}}
    candidate = {"schema": structural.SCHEDULE_SCHEMA, "evidence_kind": "frozen_owner_schedule_metadata",
        "experiment": "SYNTH-CRS-001", "seed": 101, "occurrences_per_arm": 540, "updates_per_arm": 27,
        "occurrences_per_update": 20, "retained_steps": list(range(1, 28)), "scientific_resume_boundaries": [9, 18, 27],
        "gd_original_steps": [9, 27], "producer_checkpoint": {"kind": "SYNTH_STUDY_RETAINED_UPDATE_V1", "schema_version": 1},
        "scientific_checkpoint": {"kind": "SYNTH_STUDY_TORCH_BOUNDARY_V1", "schema_version": 1},
        "producer_record": {"kind": "SYNTH_STUDY_RETAINED_UPDATE_RECORD_V1", "schema_version": 1},
        "producer_policy_sha256": digest(policy), "cost_scope": "SYNTH_STUDY_CHUNK_PROTOCOL_V1",
        "learning_rates": rates, "arms": {}}
    for arm in ARMS:
        identity = {**pending_identity, "cohort_sha256": derived[arm]["cohort_sha256"],
                    "backend_source_sha256": pins["study_torch_backend.py"]["sha256"]}
        metadata = {"schema_version": 1, "kind": candidate["producer_checkpoint"]["kind"],
            "experiment": "SYNTH-CRS-001", "scope": "UNQUALIFIED_NATIVE_IMPLEMENTATION", "arm": arm, "seed": 101,
            "recipe_sha256": None, "cohort_sha256": identity["cohort_sha256"],
            "backend_source_sha256": identity["backend_source_sha256"], "fresh_start_attestation_sha256": None,
            "adapter_parameter_order": None, "adapter_inventory_sha256": None,
            "native_launch_authorized": False, "training_authorized": False, "native_resume_qualified": False,
            "qualification_STATE1_used": False, "producer_policy_sha256": candidate["producer_policy_sha256"], "resumable": False}
        scientific = {k: v for k, v in metadata.items() if k not in ("resumable", "producer_policy_sha256")}
        scientific.update(candidate["scientific_checkpoint"])
        candidate["arms"][arm] = {"identity": identity, "metadata_pins": metadata,
            "scientific_metadata_pins": scientific, "gd_parity": {"policy_sha256": None, "probe_registration_sha256": None},
            "counters": derived[arm]["counters"]}
    try: structural.validate_schedule(candidate, structural.digest(candidate))
    except structural.ValidationError as error:
        denial = {"accepted": False, "error_type": type(error).__name__, "error": str(error)}
    else: raise AssertionError("pending native identity incorrectly accepted")
    ref = reference(registration_path); ref.update(value_sha256=digest(registration), body_sha256=registration["sha256"])
    value = {"kind": "CLOSED_PARTIAL_SCHEDULE_001", "status": "COMPLETE_REGISTERED_COUNTERS_NATIVE_IDENTITY_PENDING",
        "registration_reference": ref, "lr_metadata_reference": reference(lr_path), "current_producer_pins": pins,
        "registration_data_freeze": "Historical registration metadata only; source pins are not requalified corpus or native evidence.",
        "arms": derived, "schedule_candidate": candidate, "structural_validation": denial,
        "native_launch_authorized": False, "actual_export_parity_established": False,
        "pending": ["Exact owned STUDY trajectory/recipe/source/runtime/data/base/tokenizer identities",
                    "Actual fresh start and full adapter inventory/order attestation", "GD export policy and probe registration",
                    "Current resource/native qualification and actual retained archives/export parity9/27"]}
    value["sha256"] = digest(value)
    return value
