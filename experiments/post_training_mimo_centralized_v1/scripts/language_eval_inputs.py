"""Stdlib-only language DEV/CAL source, blind-template and rating-import mechanics.

No file, corpus, tokenizer, model or native-runtime access occurs here. Documents
are supplied as exact {source_path, raw_bytes} wrappers. Paths are fixed labels,
not filesystem instructions. validate_specs returns ordered original worlds plus
source/hash bindings; validate_payloads consumes those already-validated bindings
and verifies their internal identities again. Such provenance is not native or
human attestation, semantic freshness proof, or permission to score a model.

validate_ratings requires rows freshly obtained by validating both source-document
sets. It rebuilds the exact packet from those rows before accepting any import;
rehashed packet self-consistency alone is insufficient. This core cannot establish
filesystem freshness for the caller and never treats supplied hashes as attestation.
Rating inputs have exact keys schema_version, packet_sha256, provenance, ratings.
packet_sha256 binds the public rating_view. Each supplied rating is {id, scores},
with all four dimensions containing an integer 0..2, 'missing' or 'unjudgeable'.
Absent items are retained as all-missing. No imported claim can qualify a grader.
The coordinator mapping must remain separate from the rating view in workflow;
opaque IDs and deterministic shuffling are not cryptographic access controls.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import random
import re


STRATA = (
    "explanation", "meaning_preserving_rewrite", "synthesis", "clarification",
    "supportive_conversation", "disagreement_and_compromise",
    "contextual_recommendation", "followup_and_closure",
)
DIMENSIONS = ("adequacy", "relevance", "fluency", "role_consistency")
CONTROL_IDS = ("positive_a", "positive_b") + DIMENSIONS
AUTHORS = {"A": "/root/baseline_enrollment", "B": "/root/e1_review"}
TEMPLATES = {"TEMPLATE_A": 20260928, "TEMPLATE_B": 20260929}
SOURCE_DIRECTORY = "sources/language_dev_cal_v1/"
WORLD_FIELDS = {"parent_id", "role", "stratum", "conversation_form", "semantic_world", "author_id"}
SEMANTIC_LIST_FIELDS = {"actors", "facts", "timeline", "constraints", "potential_confusions", "exclusion_tags"}
SEMANTIC_FIELDS = SEMANTIC_LIST_FIELDS | {"task_intent", "decisive_structure"}
SPEC_BINDING_FIELDS = {"world_spec_sha256", "spec_file_sha256", "spec_source_path"}
PAYLOAD_FIELDS = {"parent_id", "world_spec_sha256", "messages", "semantic_contract", "dev_exemplar",
                  "cal_controls", "authorship"}
ROW_FIELDS = WORLD_FIELDS | SPEC_BINDING_FIELDS | PAYLOAD_FIELDS | {
    "payload_file_sha256", "payload_source_path", "payload_world_sha256", "row_sha256",
}
MAPPING_FIELDS = SPEC_BINDING_FIELDS | {
    "parent_id", "control_id", "stratum", "author_id", "author_intent", "intent_rationale", "is_repeat", "repeat_of_id",
    "rating_item_sha256", "control_sha256", "payload_source_path", "payload_file_sha256", "payload_world_sha256", "row_sha256",
}
PROVENANCE_FIELDS = {"grader_id", "human_or_model", "rubric_revision", "calibration_receipt_hash",
                     "assignment_and_blinding_receipt", "conflicts_or_prior_exposure", "timestamp"}
INSTRUCTIONS = (
    "Unassigned diagnostic rating template; rubric revision PROPOSED-LANG-RUBRIC-001. "
    "Rate adequacy, relevance, fluency, and role_consistency separately. "
    "Use 0 for material failure, 1 for a meaningful defect, and 2 when the dimension is met. "
    "Adequacy: fulfills the request with supplied facts and appropriate uncertainty. "
    "Relevance: serves the current conversational purpose without material distraction. "
    "Fluency: natural, readable, coherent language in the requested register. "
    "Role consistency: preserves speaker roles, context, and honest capability boundaries. "
    "Multiple wordings can be acceptable; exact reference matching is insufficient. "
    "Use 'unjudgeable' when a dimension cannot be judged, and 'missing' for an absent rating. "
    "Prior assistant messages are authored context. This template assigns no reviewer or grading authority."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _fields(value, expected, label):
    _require(type(value) is dict and set(value) == set(expected), label + " field inventory")


def _string(value, label):
    _require(type(value) is str and bool(value.strip()), label + " must be a nonempty string")


def _strings(value, label):
    _require(type(value) is list and bool(value), label + " must be a nonempty string list")
    for child in value:
        _string(child, label)


def _hash(value, label):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, label + " SHA256")


def _finite(value):
    if type(value) is dict:
        for child in value.values():
            _finite(child)
    elif type(value) is list:
        for child in value:
            _finite(child)
    elif type(value) is float:
        _require(math.isfinite(value), "nonfinite JSON")


def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON: " + value)

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    _finite(value)
    return value


def _source_documents(documents, kind):
    _require(type(documents) in (list, tuple) and len(documents) == 2, "exactly two author source documents required")
    expected = {SOURCE_DIRECTORY + kind + "_" + part + ".json": part for part in AUTHORS}
    seen, parsed = set(), []
    for wrapper in documents:
        _fields(wrapper, {"source_path", "raw_bytes"}, "source wrapper")
        label, raw = wrapper["source_path"], wrapper["raw_bytes"]
        _require(type(label) is str and label in expected and label not in seen, "unregistered or duplicate source path label")
        _require(type(raw) is bytes and bool(raw), "source raw_bytes required")
        seen.add(label)
        parsed.append((expected[label], label, hashlib.sha256(raw).hexdigest(), _parse(raw)))
    return sorted(parsed)


def _slots():
    slots = {}
    for index, stratum in enumerate(STRATA):
        author = AUTHORS["A" if index < 4 else "B"]
        for role, prefix, count in (("DEV", "LDEV", 8), ("CAL", "LCAL", 2)):
            for ordinal in range(1, count + 1):
                parent = f"{prefix}-{stratum}-{ordinal:02d}"
                form = "single" if ordinal <= (4 if role == "DEV" else 1) else "multi"
                slots[parent] = (role, stratum, form, author)
    return slots


SLOTS = _slots()


def _world(world):
    _fields(world, WORLD_FIELDS, "world specification")
    parent = world["parent_id"]
    _require(type(parent) is str and parent in SLOTS, "unregistered DEV/CAL parent ID")
    _require(tuple(world[name] for name in ("role", "stratum", "conversation_form", "author_id")) == SLOTS[parent],
             "parent role/stratum/conversation form/author drift")
    semantic = world["semantic_world"]
    _fields(semantic, SEMANTIC_FIELDS, "semantic world")
    for field in SEMANTIC_LIST_FIELDS:
        _strings(semantic[field], field)
    for field in ("task_intent", "decisive_structure"):
        _string(semantic[field], field)


def _population(worlds):
    parents = [world["parent_id"] for world in worlds]
    _require(len(parents) == 80 and len(set(parents)) == 80 and set(parents) == set(SLOTS),
             "fixed population requires all 64 DEV and 16 CAL worlds exactly once")
    semantic_hashes = [digest(world["semantic_world"]) for world in worlds]
    _require(len(set(semantic_hashes)) == 80, "exact duplicate semantic world under different parent IDs")
    by_parent = {world["parent_id"]: world for world in worlds}
    return [by_parent[parent] for parent in SLOTS]


def validate_specs(spec_documents):
    """Validate the fixed pre-render bank; hash original six-field worlds canonically."""
    worlds = []
    for part, label, raw_hash, document in _source_documents(spec_documents, "SPECS"):
        _fields(document, {"schema_version", "stage", "author_id", "worlds"}, "spec document")
        _require(type(document["schema_version"]) is int and document["schema_version"] == 1
                 and document["stage"] == "SPEC_ONLY_NO_RENDERED_TEXT", "spec document stage/version")
        _require(document["author_id"] == AUTHORS[part] and type(document["worlds"]) is list
                 and len(document["worlds"]) == 40, "author scope requires exactly 40 worlds")
        for original in document["worlds"]:
            _world(original)
            _require(original["author_id"] == document["author_id"], "world author differs from owned source")
            worlds.append({**deepcopy(original), "world_spec_sha256": digest(original),
                           "spec_file_sha256": raw_hash, "spec_source_path": label})
    return _population(worlds)


def _validated_specs(specs):
    _require(type(specs) in (list, tuple), "validated specifications must be ordered records")
    worlds = []
    for spec in specs:
        _fields(spec, WORLD_FIELDS | SPEC_BINDING_FIELDS, "bound world")
        original = {name: spec[name] for name in WORLD_FIELDS}
        _world(original)
        _require(spec["world_spec_sha256"] == digest(original), "bound semantic world hash drift")
        _hash(spec["spec_file_sha256"], "spec file")
        part = "A" if original["author_id"] == AUTHORS["A"] else "B"
        _require(spec["spec_source_path"] == SOURCE_DIRECTORY + "SPECS_" + part + ".json", "bound source scope drift")
        worlds.append(deepcopy(spec))
    worlds = _population(worlds)
    for part in AUTHORS:
        _require(len({world["spec_file_sha256"] for world in worlds if world["author_id"] == AUTHORS[part]}) == 1,
                 "one exact raw spec-file identity per author required")
    return worlds


def _conversation(messages, conversation_form):
    _require(conversation_form in ("single", "multi"), "unregistered conversation form")
    _require(type(messages) is list and len(messages) in ((1,) if conversation_form == "single" else (3, 5, 7)),
             "single requires one user turn; multi requires 2..4 user turns")
    for index, message in enumerate(messages):
        _fields(message, {"role", "content"}, "conversation message")
        _require(message["role"] == ("user" if index % 2 == 0 else "assistant"), "conversation must alternate and end user")
        _string(message["content"], "message content")


def _payload_world(payload, spec):
    _fields(payload, PAYLOAD_FIELDS, "payload world")
    _require(payload["parent_id"] == spec["parent_id"] and payload["world_spec_sha256"] == spec["world_spec_sha256"],
             "payload world/spec binding drift")
    _require(payload["authorship"] == "assistant_source_candidate", "unqualified assistant authorship marker required")
    _conversation(payload["messages"], spec["conversation_form"])
    contract = payload["semantic_contract"]
    _fields(contract, {"must_include", "must_avoid", "acceptable_variation", "dimensions"}, "semantic contract")
    for field in ("must_include", "must_avoid", "acceptable_variation"):
        _strings(contract[field], field)
    _fields(contract["dimensions"], DIMENSIONS, "dimension guidance")
    for guidance in contract["dimensions"].values():
        _string(guidance, "dimension guidance, never a score")
    controls = payload["cal_controls"]
    _require(type(controls) is list, "cal_controls list required")
    if spec["role"] == "DEV":
        _string(payload["dev_exemplar"], "DEV illustrative exemplar")
        _require(not controls, "DEV cannot contain calibration controls")
    else:
        _require(payload["dev_exemplar"] is None and len(controls) == 6, "CAL requires six controls and no DEV exemplar")
        _require([control.get("control_id") for control in controls if type(control) is dict] == list(CONTROL_IDS),
                 "CAL controls must be the six registered IDs in canonical order")
        responses = []
        for control in controls:
            _fields(control, {"control_id", "response", "author_intent", "intent_rationale", "qualified_reference_scores"}, "CAL control")
            expected = "intended_acceptable" if control["control_id"].startswith("positive_") else control["control_id"]
            _require(control["author_intent"] == expected, "CAL author intent does not match control ID")
            _require(control["qualified_reference_scores"] is None, "reference score qualification is unavailable")
            _string(control["response"], "CAL candidate response")
            _string(control["intent_rationale"], "unqualified author-intent rationale")
            responses.append(control["response"])
        _require(len(set(responses)) == 6, "CAL candidate responses must be distinct")


def validate_payloads(specs, payload_documents):
    """Join rendered candidates only to already validated, unchanged specifications."""
    bound_specs = {spec["parent_id"]: spec for spec in _validated_specs(specs)}
    rows, seen = [], set()
    for part, label, raw_hash, document in _source_documents(payload_documents, "PAYLOADS"):
        _fields(document, {"schema_version", "stage", "spec_file_sha256", "author_id", "worlds"}, "payload document")
        _require(type(document["schema_version"]) is int and document["schema_version"] == 1
                 and document["stage"] == "RENDERED_CANDIDATES_UNQUALIFIED", "payload stage/version")
        _require(document["author_id"] == AUTHORS[part] and type(document["worlds"]) is list
                 and len(document["worlds"]) == 40, "payload author scope requires 40 worlds")
        for payload in document["worlds"]:
            _require(type(payload) is dict and type(payload.get("parent_id")) is str, "payload parent missing")
            parent = payload["parent_id"]
            _require(parent in bound_specs and parent not in seen, "unregistered or duplicate payload parent")
            seen.add(parent)
            spec = bound_specs[parent]
            _require(spec["author_id"] == document["author_id"] and document["spec_file_sha256"] == spec["spec_file_sha256"],
                     "payload raw spec-file hash or author scope drift")
            _payload_world(payload, spec)
            row = {**deepcopy(spec), **deepcopy(payload), "payload_file_sha256": raw_hash,
                   "payload_source_path": label, "payload_world_sha256": digest(payload)}
            row["row_sha256"] = digest(row)
            rows.append(row)
    return _population(rows)


def _validated_rows(rows):
    _require(type(rows) in (list, tuple), "bound rows must be an ordered collection")
    specs = []
    for row in rows:
        _fields(row, ROW_FIELDS, "normalized row")
        _require(row["row_sha256"] == digest({key: value for key, value in row.items() if key != "row_sha256"}), "normalized row hash drift")
        specs.append({name: row[name] for name in WORLD_FIELDS | SPEC_BINDING_FIELDS})
        payload = {name: row[name] for name in PAYLOAD_FIELDS}
        _require(row["payload_world_sha256"] == digest(payload), "normalized payload hash drift")
        _payload_world(payload, specs[-1])
        _hash(row["payload_file_sha256"], "payload file")
        part = "A" if row["author_id"] == AUTHORS["A"] else "B"
        _require(row["payload_source_path"] == SOURCE_DIRECTORY + "PAYLOADS_" + part + ".json", "payload source scope drift")
    _validated_specs(specs)
    for part in AUTHORS:
        _require(len({row["payload_file_sha256"] for row in rows if row["author_id"] == AUTHORS[part]}) == 1,
                 "one exact payload-file identity per author required")
    return _population(deepcopy(rows))


def build_cal_packet(rows, *, seed, template_id):
    """Create 96 candidate records plus one hidden repeat per CAL world; unassigned."""
    _require(type(template_id) is str and template_id in TEMPLATES
             and type(seed) is int and seed == TEMPLATES[template_id], "unregistered packet template/seed")
    cal_rows = [row for row in _validated_rows(rows) if row["role"] == "CAL"]
    rng = random.Random(seed)
    entries = []
    originals = {}
    for row in cal_rows:
        for control in row["cal_controls"]:
            entry = {"row": row, "control": control, "repeat": False}
            entries.append(entry)
            originals[(row["parent_id"], control["control_id"])] = entry
        entries.append({"row": row, "control": rng.choice(row["cal_controls"]), "repeat": True})
    used_ids = set()
    for entry in entries:
        opaque = "item-" + format(rng.getrandbits(128), "032x")
        _require(opaque not in used_ids, "opaque ID collision")
        used_ids.add(opaque)
        entry["id"] = opaque
    rng.shuffle(entries)
    view, mapping = [], {}
    for entry in entries:
        row, control = entry["row"], entry["control"]
        item = {"id": entry["id"], "messages": deepcopy(row["messages"]),
                "response": control["response"], "instructions": INSTRUCTIONS}
        view.append(item)
        mapping[entry["id"]] = {
            "parent_id": row["parent_id"], "control_id": control["control_id"],
            "stratum": row["stratum"], "author_id": row["author_id"],
            "author_intent": control["author_intent"], "intent_rationale": control["intent_rationale"],
            "is_repeat": entry["repeat"],
            "repeat_of_id": originals[(row["parent_id"], control["control_id"])]["id"] if entry["repeat"] else None,
            "rating_item_sha256": digest(item), "control_sha256": digest(control),
            **{field: row[field] for field in SPEC_BINDING_FIELDS | {
                "payload_source_path", "payload_file_sha256", "payload_world_sha256", "row_sha256"}},
        }
    coordinator = {"schema_version": 1, "status": "UNASSIGNED_CAL_TEMPLATE_NO_QUALIFIED_LABELS",
                   "template_id": template_id, "seed": seed, "rating_view_sha256": digest(view),
                   "unique_controls": 96, "hidden_repeats": 16, "entries": mapping}
    coordinator["mapping_sha256"] = digest(coordinator)
    return {"rating_view": view, "coordinator_mapping": coordinator}


def _packet(packet):
    _fields(packet, {"rating_view", "coordinator_mapping"}, "packet")
    view, coordinator = packet["rating_view"], packet["coordinator_mapping"]
    _fields(coordinator, {"schema_version", "status", "template_id", "seed", "rating_view_sha256",
                          "unique_controls", "hidden_repeats", "entries", "mapping_sha256"}, "coordinator mapping")
    _require(coordinator["mapping_sha256"] == digest({k: v for k, v in coordinator.items() if k != "mapping_sha256"}), "coordinator mapping drift")
    _require(type(coordinator["schema_version"]) is int and coordinator["schema_version"] == 1
             and coordinator["status"] == "UNASSIGNED_CAL_TEMPLATE_NO_QUALIFIED_LABELS"
             and type(coordinator["template_id"]) is str and coordinator["template_id"] in TEMPLATES
             and type(coordinator["seed"]) is int and coordinator["seed"] == TEMPLATES[coordinator["template_id"]], "packet contract drift")
    _require(type(view) is list and len(view) == 112 and coordinator["rating_view_sha256"] == digest(view), "rating view count/hash drift")
    entries = coordinator["entries"]
    _require(type(entries) is dict and len(entries) == 112, "packet mapping count")
    identifiers = [item.get("id") for item in view if type(item) is dict]
    _require(len(identifiers) == 112 and all(type(key) is str for key in identifiers)
             and len(set(identifiers)) == 112, "packet IDs must be distinct strings")
    originals, repeats, parent_sources = {}, [], {}
    for item in view:
        _fields(item, {"id", "messages", "response", "instructions"}, "blind rating item")
        _require(type(item["id"]) is str and re.fullmatch(r"item-[0-9a-f]{32}", item["id"]) is not None
                 and item["id"] in entries, "opaque packet ID drift")
        entry = entries[item["id"]]
        _fields(entry, MAPPING_FIELDS, "coordinator entry")
        _require(entry.get("rating_item_sha256") == digest(item)
                 and item["instructions"] == INSTRUCTIONS, "rating item/instructions drift")
        _require(type(entry.get("is_repeat")) is bool, "repeat marker must be boolean")
        key = (entry.get("parent_id"), entry.get("control_id"))
        _require(type(key[0]) is str and key[0] in SLOTS and SLOTS[key[0]][0] == "CAL"
                 and type(key[1]) is str and key[1] in CONTROL_IDS, "coordinator parent/control drift")
        _require(entry["stratum"] == SLOTS[key[0]][1] and entry["author_id"] == SLOTS[key[0]][3], "coordinator author/stratum drift")
        _conversation(item["messages"], SLOTS[key[0]][2])
        _string(item["response"], "displayed candidate response")
        expected_intent = "intended_acceptable" if key[1].startswith("positive_") else key[1]
        _require(entry["author_intent"] == expected_intent, "coordinator unqualified intent drift")
        _string(entry["intent_rationale"], "coordinator intent rationale")
        control = {"control_id": key[1], "response": item["response"], "author_intent": entry["author_intent"],
                   "intent_rationale": entry["intent_rationale"], "qualified_reference_scores": None}
        _require(entry["control_sha256"] == digest(control), "displayed response/control source hash drift")
        for field in MAPPING_FIELDS:
            if field.endswith("_sha256"):
                _hash(entry[field], field)
        part = "A" if entry["author_id"] == AUTHORS["A"] else "B"
        _require(entry["spec_source_path"] == SOURCE_DIRECTORY + "SPECS_" + part + ".json"
                 and entry["payload_source_path"] == SOURCE_DIRECTORY + "PAYLOADS_" + part + ".json", "coordinator source label drift")
        source = {"messages": item["messages"], **{field: entry[field] for field in SPEC_BINDING_FIELDS | {
            "payload_source_path", "payload_file_sha256", "payload_world_sha256", "row_sha256"}}}
        _require(key[0] not in parent_sources or parent_sources[key[0]] == source,
                 "same-parent displayed conversation or source provenance differs")
        parent_sources[key[0]] = source
        if entry["is_repeat"]:
            repeats.append((item, entry))
        else:
            _require(key not in originals and entry.get("repeat_of_id") is None, "duplicate original control")
            originals[key] = item
    _require(len(originals) == 96 and len(repeats) == 16 and type(coordinator["unique_controls"]) is int
             and coordinator["unique_controls"] == 96 and type(coordinator["hidden_repeats"]) is int
             and coordinator["hidden_repeats"] == 16, "packet dose drift")
    _require(Counter(entry["parent_id"] for _, entry in repeats) == Counter({parent: 1 for parent, slot in SLOTS.items() if slot[0] == "CAL"}),
             "one hidden repeat per CAL world required")
    for item, entry in repeats:
        original = originals[(entry["parent_id"], entry["control_id"])]
        _require(entry["repeat_of_id"] == original["id"]
                 and {k: v for k, v in item.items() if k != "id"} == {k: v for k, v in original.items() if k != "id"},
                 "hidden repeat content/source mismatch")
        source_fields = MAPPING_FIELDS - {"rating_item_sha256", "is_repeat", "repeat_of_id"}
        _require({field: entry[field] for field in source_fields} ==
                 {field: entries[original["id"]][field] for field in source_fields}, "hidden repeat provenance differs from original")
    return view, coordinator


def validate_ratings(packet, ratings, *, rows):
    """Import against freshly validated source rows; retain every missing/unjudgeable case.

    The caller must re-read the frozen spec and payload document bytes and use
    validate_specs/validate_payloads immediately before this call. No source-less
    import exists. This function rebuilds the registered packet and requires exact
    agreement, including source bindings, displayed text, order and hidden repeats.
    """
    view, coordinator = _packet(packet)
    expected = build_cal_packet(rows, seed=coordinator["seed"], template_id=coordinator["template_id"])
    _require(canonical(packet) == canonical(expected), "packet differs from freshly validated source rows")
    _fields(ratings, {"schema_version", "packet_sha256", "provenance", "ratings"}, "rating import")
    _finite(ratings)
    _require(type(ratings["schema_version"]) is int and ratings["schema_version"] == 1
             and ratings["packet_sha256"] == coordinator["rating_view_sha256"], "rating packet identity drift")
    provenance = ratings["provenance"]
    _require(type(provenance) is dict and set(provenance) in (PROVENANCE_FIELDS, PROVENANCE_FIELDS | {"qualification_claim"}),
             "rating provenance field inventory")
    for name in PROVENANCE_FIELDS:
        _require(provenance[name] is None or (type(provenance[name]) is str and bool(provenance[name].strip())), "provenance must preserve strings or explicit nulls")
    _require(type(ratings["ratings"]) is list, "rating records must be a list")
    supplied = {}
    for record in ratings["ratings"]:
        _fields(record, {"id", "scores"}, "rating record")
        key = record["id"]
        _require(type(key) is str and key in coordinator["entries"] and key not in supplied, "extra/duplicate/drifted rating ID")
        _fields(record["scores"], DIMENSIONS, "four dimension scores")
        for score in record["scores"].values():
            _require((type(score) is int and 0 <= score <= 2) or (type(score) is str and score in {"missing", "unjudgeable"}),
                     "dimension score must be integer 0..2 or explicit missing/unjudgeable")
        supplied[key] = deepcopy(record["scores"])
    normalized = [{"id": item["id"], "supplied": item["id"] in supplied,
                   "scores": supplied.get(item["id"], dict.fromkeys(DIMENSIONS, "missing"))} for item in view]
    by_id = {record["id"]: record["scores"] for record in normalized}
    counts = {dimension: {str(value): sum(record["scores"][dimension] == value for record in normalized)
                          for value in (0, 1, 2, "missing", "unjudgeable")} for dimension in DIMENSIONS}
    comparable = agreed = 0
    for key, entry in coordinator["entries"].items():
        if not entry["is_repeat"]:
            continue
        first, second = by_id[key], by_id[entry["repeat_of_id"]]
        if all(type(scores[dimension]) is int for scores in (first, second) for dimension in DIMENSIONS):
            comparable += 1
            agreed += first == second
    return {"schema_version": 1, "status": "UNQUALIFIED_RATING_IMPORT_DIAGNOSTIC_ONLY",
            "packet_sha256": coordinator["rating_view_sha256"], "mapping_sha256": coordinator["mapping_sha256"],
            "ratings_sha256": digest(ratings), "provenance": deepcopy(provenance), "ratings": normalized,
            "expected_items": 112, "supplied_items": len(supplied), "absent_items": 112 - len(supplied),
            "dimension_counts_including_repeats": counts,
            "repeat_diagnostic": {"pairs": 16, "fully_scored_pairs": comparable, "exact_vector_agreements": agreed,
                                  "missing_or_unjudgeable_pairs": 16 - comparable,
                                  "agreement_fraction": agreed / comparable if comparable else None},
            "grading_qualified": False, "source_correct_membership": False, "e1_authorized": False,
            "native_calls": 0, "reference_labels_qualified": False,
            "limitations": "Author intent is unqualified; no calibration threshold or acceptance claim is evaluated. Repeats are not independent worlds."}
