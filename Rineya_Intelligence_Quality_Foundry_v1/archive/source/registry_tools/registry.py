"""Append-only split/lineage registry, independent of the frozen generator.

Only TRAIN, TRAIN_SELECTION, CAL and exposed DEV are accepted. No API opens an
evaluation bank. Registration detects declared lineage and the fingerprint
coverage below; it cannot prove semantic or pretraining decontamination.

Public API: Registry(path).register(batch, cycle_id=...), snapshot(), summary();
empty_batch(), make_source(), make_structure(), make_parent(), make_row(),
make_release(), make_exclusion(), and pilot_v3_batch(foundry_root).
Use a separate registration for prospective parents before submitting rows.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import tempfile
from datetime import datetime, timezone
from contextlib import contextmanager
import unicodedata

SCHEMA_VERSION = 1
SPLITS = frozenset({"TRAIN", "TRAIN_SELECTION", "CAL", "DEV"})
PROSPECTIVE = "PROSPECTIVE_PRE_GENERATION"
RETROSPECTIVE = "POST_GENERATION_REGISTRATION_NOT_PROSPECTIVE"
NORMALIZER = "NFKC_CASEFOLD_WHITESPACE_V1"
TOKENIZER = "NFKC_CASEFOLD_REGEX_LEXICAL_V1_NOT_MODEL_TOKEN_IDS"
COVERAGE = {
    "normalizer": NORMALIZER,
    "lexical_tokenizer": TOKENIZER,
    "covered": ["exact canonical JSON", "Unicode NFKC and casefold",
                "collapsed whitespace", "ordered lexical words and punctuation",
                "declared parent and source ancestry", "source revision families"],
    "not_covered": ["entity-renaming isomorphism", "paraphrase semantics",
                    "undeclared ancestry", "pretraining overlap", "sealed payloads"],
    "semantic_decontamination": "NOT_PROVEN",
    "prospective_timing": "LOCAL_COMMIT_ORDER_ONLY_NOT_EXTERNAL_ATTESTATION",
}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT = re.compile(r"([0-9]{8})-([0-9a-f]{64})\.json\Z")
_COLLECTIONS = ("sources", "structures", "parents", "rows", "releases", "exclusions")
_FORBIDDEN_SPEC_KEYS = frozenset({"target", "messages", "rendered", "completion",
                                  "solution", "model_response"})
_DOMAINS = frozenset({"source_content", "parent_specification", "parent_normalized",
                     "row_exact", "row_normalized", "row_lexical_tokens",
                     "prompt_exact", "prompt_normalized", "prompt_lexical_tokens",
                     "native_titan_prompt_token_ids"})
IDENTITY_SPACES = {
    "source_content": "SHA256_SOURCE_CONTENT_BYTES_V1",
    "parent_specification": "SHA256_CANONICAL_FULL_CONSTRUCTION_SPECIFICATION_V1",
    "parent_normalized": NORMALIZER + "_SHA256_CANONICAL_SPECIFICATION",
    "row_exact": "SHA256_CANONICAL_MESSAGES_AND_TARGET_V1",
    "row_normalized": NORMALIZER + "_SHA256_CANONICAL_MESSAGES_AND_TARGET",
    "row_lexical_tokens": TOKENIZER + "_SHA256_CANONICAL_MESSAGES_AND_TARGET",
    "prompt_exact": "SHA256_CANONICAL_MESSAGES_V1",
    "prompt_normalized": NORMALIZER + "_SHA256_CANONICAL_MESSAGES",
    "prompt_lexical_tokens": TOKENIZER + "_SHA256_CANONICAL_MESSAGES",
    "native_titan_prompt_token_ids": "NATIVE_TITAN_PROMPT_TOKEN_IDS_SHA256_CANONICAL_ORDERED_IDS_V1",
}


class RegistryError(ValueError):
    """Malformed input, identity drift or split/duplicate conflict."""


class RegistryLocked(RegistryError):
    """Another cycle owns the exclusive registry lock; never steal it."""


class CommittedRegistrationError(RuntimeError):
    """A complete snapshot committed, then lease integrity failed.

    receipt identifies the committed snapshot. This is never a rejected batch;
    verify the receipt and retained lock rather than retrying as fresh data.
    """
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = copy.deepcopy(receipt)


def _json_value(value, location="value"):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for child in value:
            _json_value(child, location)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _json_value(child, location)
        return
    raise RegistryError(f"{location}: requires finite JSON values and string keys")


def canonical(value):
    _json_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            result.update(chunk)
    return result.hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RegistryError("duplicate JSON key: " + key)
            result[key] = value
        return result
    try:
        result = json.loads(text, object_pairs_hook=pairs,
                            parse_constant=lambda x: (_ for _ in ()).throw(RegistryError(x)))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise RegistryError("invalid JSON") from error
    _json_value(result)
    return result


def normalize(value):
    """Declared mechanical normalization; retains JSON keys and list order."""
    if type(value) is str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if type(value) is list:
        return [normalize(child) for child in value]
    if type(value) is dict:
        return {key: normalize(child) for key, child in value.items()}
    return value


def fingerprints(payload):
    """Canonical payload hashes; lexical tokens are explicitly not model IDs."""
    normalized = normalize(payload)
    tokens = re.findall(r"\w+|[^\w\s]", canonical(normalized), flags=re.UNICODE)
    return {"exact_sha256": digest(payload), "normalized_sha256": digest(normalized),
            "lexical_tokens_sha256": digest(tokens), "normalizer": NORMALIZER,
            "lexical_tokenizer": TOKENIZER}


def _keys(value, keys, location):
    if type(value) is not dict or set(value) != set(keys):
        raise RegistryError(f"{location}: expected exactly {sorted(keys)}")


def _string(value, location):
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise RegistryError(location + ": expected nonempty string")


def _id(value, location):
    if type(value) is not str or not _ID.fullmatch(value):
        raise RegistryError(location + ": invalid stable ID")


def _hash(value, location):
    if type(value) is not str or not _HASH.fullmatch(value):
        raise RegistryError(location + ": expected lowercase SHA-256")


def _ids(value, location, allow_empty=True):
    if type(value) is not list or (not allow_empty and not value):
        raise RegistryError(location + ": expected list of IDs")
    for identifier in value:
        _id(identifier, location)
    if len(set(value)) != len(value):
        raise RegistryError(location + ": duplicate IDs")


def _split(value):
    if type(value) is not str or value not in SPLITS:
        raise RegistryError("closed or unknown split; only TRAIN/TRAIN_SELECTION/CAL/DEV")


def _specification(specification):
    if type(specification) is not dict or not specification:
        raise RegistryError("construction specification must be a complete nonempty JSON object")
    _json_value(specification, "specification")
    def walk(value):
        if type(value) is dict:
            if set(value) & _FORBIDDEN_SPEC_KEYS:
                raise RegistryError("construction specification contains target/rendering fields")
            for child in value.values():
                walk(child)
        elif type(value) is list:
            for child in value:
                walk(child)
    walk(specification)


def empty_batch():
    return {"schema_version": SCHEMA_VERSION,
            **{name: [] for name in _COLLECTIONS}}


def make_source(source_id, source_family_id, revision, content_sha256, *, origin,
                owner, licence_evidence, use_permission, split_scope="CONTENT_FAMILY",
                safety="TRAIN_SAFE", ancestor_source_ids=()):
    """Licence/use strings are evidence assertions, not automated rights approval."""
    return {"source_id": source_id, "source_family_id": source_family_id,
            "revision": revision, "content_sha256": content_sha256, "origin": origin,
            "owner": owner, "licence_evidence": licence_evidence,
            "use_permission": use_permission, "split_scope": split_scope,
            "safety": safety, "ancestor_source_ids": list(ancestor_source_ids)}


def make_structure(specification, *, description):
    return {"structural_cluster_id": "sc-" + digest(specification),
            "specification": specification, "specification_sha256": digest(specification),
            "description": description}


def make_parent(specification, *, construction_schema, split, source_ids,
                structural_cluster_id, registration_mode=PROSPECTIVE,
                ancestor_parent_ids=(), aliases=(), instance_fingerprints=(),
                historical_overlap="NOT_AUDITED"):
    """Keep derivatives on the original parent or explicitly declare ancestry."""
    return {"semantic_parent_id": "sp-" + digest(specification),
            "specification": specification, "specification_sha256": digest(specification),
            "normalized_specification_sha256": digest(normalize(specification)),
            "construction_schema": construction_schema, "split": split,
            "source_ids": list(source_ids), "structural_cluster_id": structural_cluster_id,
            "ancestor_parent_ids": list(ancestor_parent_ids),
            "aliases": list(aliases), "instance_fingerprints": list(instance_fingerprints),
            "registration_mode": registration_mode,
            "prospective": registration_mode == PROSPECTIVE,
            "historical_overlap": historical_overlap}


def make_row(row_id, semantic_parent_id, split, payload, *, source_ids,
             ancestor_row_ids=(), native_prompt_tokens=None):
    """payload has messages plus target; metadata stays outside content identity."""
    return {"row_id": row_id, "semantic_parent_id": semantic_parent_id, "split": split,
            "source_ids": list(source_ids), "ancestor_row_ids": list(ancestor_row_ids),
            "payload": payload, "fingerprints": fingerprints(payload),
            "prompt_fingerprints": fingerprints(payload.get("messages")),
            "native_prompt_tokens": native_prompt_tokens}


def make_release(release_id, row_ids, *, manifest_sha256, canonical_sha256,
                 registration_mode=RETROSPECTIVE):
    return {"release_id": release_id, "row_ids": list(row_ids),
            "manifest_sha256": manifest_sha256, "canonical_sha256": canonical_sha256,
            "registration_mode": registration_mode, "model_gain": "UNMEASURED",
            "semantic_decontamination": "NOT_PROVEN"}


def make_exclusion(exclusion_id, identities, *, evidence_reference,
                   coverage_statement, qualification="KNOWN_EXCLUSION_HASHES",
                   forbidden_roles=tuple(sorted(SPLITS))):
    # The helper's default is an explicit assertion of this module's identity
    # space. Supply identity_space when adapting an external inventory; unknown
    # normalizers must remain unqualified and cannot be mislabeled as this one.
    bound_identities = []
    for identity in identities:
        item = copy.deepcopy(identity)
        if type(item) is dict and "identity_space" not in item:
            item["identity_space"] = IDENTITY_SPACES.get(item.get("domain"))
        bound_identities.append(item)
    return {"exclusion_id": exclusion_id, "identities": bound_identities,
            "evidence_reference": evidence_reference, "coverage_statement": coverage_statement,
            "qualification": qualification, "forbidden_roles": list(forbidden_roles)}


def _initial_state():
    return {name: {} for name in _COLLECTIONS} | {"parent_reserved_sequence": {}}


def _validate_record(kind, record):
    keysets = {
        "sources": {"source_id", "source_family_id", "revision", "content_sha256", "origin",
                    "owner", "licence_evidence", "use_permission", "split_scope", "safety",
                    "ancestor_source_ids"},
        "structures": {"structural_cluster_id", "specification", "specification_sha256", "description"},
        "parents": {"semantic_parent_id", "specification", "specification_sha256",
                    "normalized_specification_sha256", "construction_schema", "split", "source_ids",
                    "structural_cluster_id", "ancestor_parent_ids", "registration_mode",
                    "prospective", "historical_overlap"},
        "rows": {"row_id", "semantic_parent_id", "split", "source_ids", "ancestor_row_ids",
                 "payload", "fingerprints", "prompt_fingerprints"},
        "releases": {"release_id", "row_ids", "manifest_sha256", "canonical_sha256",
                     "registration_mode", "model_gain", "semantic_decontamination"},
        "exclusions": {"exclusion_id", "identities", "evidence_reference", "coverage_statement",
                       "qualification", "forbidden_roles"},
    }
    keysets["parents"].update({"aliases", "instance_fingerprints"})
    keysets["rows"].add("native_prompt_tokens")
    _keys(record, keysets[kind], kind)
    primary = {"sources": "source_id", "structures": "structural_cluster_id",
               "parents": "semantic_parent_id", "rows": "row_id",
               "releases": "release_id", "exclusions": "exclusion_id"}[kind]
    _id(record[primary], primary)
    for key, value in record.items():
        if key.endswith("sha256"):
            _hash(value, key)
    if kind == "sources":
        _id(record["source_family_id"], "source_family_id")
        _ids(record["ancestor_source_ids"], "ancestor_source_ids")
        for key in ("revision", "origin", "owner", "licence_evidence", "use_permission"):
            _string(record[key], key)
        _string(record["split_scope"], "split_scope")
        _string(record["safety"], "safety")
        if record["split_scope"] not in {"CONTENT_FAMILY", "PRODUCER_ONLY"}:
            raise RegistryError("invalid source split_scope")
        if record["safety"] not in {"TRAIN_SAFE", "EXCLUDED", "UNQUALIFIED"}:
            raise RegistryError("invalid source safety")
    elif kind == "structures":
        _specification(record["specification"])
        _string(record["description"], "description")
        if record["specification_sha256"] != digest(record["specification"]) or record[primary] != "sc-" + record["specification_sha256"]:
            raise RegistryError("structural identity drift")
    elif kind == "parents":
        _specification(record["specification"])
        _split(record["split"])
        _ids(record["source_ids"], "source_ids", False)
        _ids(record["ancestor_parent_ids"], "ancestor_parent_ids")
        _ids(record["aliases"], "aliases")
        if type(record["instance_fingerprints"]) is not list:
            raise RegistryError("instance_fingerprints must be a list")
        algorithms = set()
        for fingerprint in record["instance_fingerprints"]:
            _keys(fingerprint, {"algorithm", "sha256", "coverage"}, "instance fingerprint")
            _id(fingerprint["algorithm"], "instance fingerprint algorithm")
            _hash(fingerprint["sha256"], "instance fingerprint sha256")
            _string(fingerprint["coverage"], "instance fingerprint coverage")
            if fingerprint["algorithm"] in algorithms:
                raise RegistryError("duplicate instance fingerprint algorithm")
            algorithms.add(fingerprint["algorithm"])
        _id(record["construction_schema"], "construction_schema")
        _id(record["structural_cluster_id"], "structural_cluster_id")
        if record["specification_sha256"] != digest(record["specification"]) or record[primary] != "sp-" + record["specification_sha256"]:
            raise RegistryError("parent specification/ID drift")
        if record["normalized_specification_sha256"] != digest(normalize(record["specification"])):
            raise RegistryError("parent normalized fingerprint drift")
        _string(record["registration_mode"], "registration_mode")
        _string(record["historical_overlap"], "historical_overlap")
        if record["registration_mode"] not in {PROSPECTIVE, RETROSPECTIVE} or type(record["prospective"]) is not bool or record["prospective"] != (record["registration_mode"] == PROSPECTIVE):
            raise RegistryError("invalid prospective/retrospective metadata")
        if record["historical_overlap"] not in {"NOT_AUDITED", "UNQUALIFIED", "KNOWN_HASHES_CHECKED_SEMANTICS_NOT_PROVEN"}:
            raise RegistryError("historical overlap cannot claim semantic decontamination")
    elif kind == "rows":
        _split(record["split"])
        _id(record["semantic_parent_id"], "semantic_parent_id")
        _ids(record["source_ids"], "source_ids", False)
        _ids(record["ancestor_row_ids"], "ancestor_row_ids")
        _keys(record["payload"], {"messages", "target"}, "row payload")
        messages = record["payload"]["messages"]
        if type(messages) is not list or not messages:
            raise RegistryError("messages must be a nonempty list")
        for message in messages:
            _keys(message, {"role", "content"}, "message")
            _string(message["role"], "message role")
            if message["role"] not in {"system", "user", "assistant", "tool"}:
                raise RegistryError("invalid message role")
            _string(message["content"], "content")
        if type(record["payload"]["target"]) is not str:
            raise RegistryError("target must be an exact string; an empty EOS-only target is permitted")
        if record["fingerprints"] != fingerprints(record["payload"]) or record["prompt_fingerprints"] != fingerprints(messages):
            raise RegistryError("row fingerprint drift")
        native = record["native_prompt_tokens"]
        if native is not None:
            _keys(native, {"kind", "prompt_ids_sha256", "tokenizer_identity_sha256",
                           "chat_template_sha256", "template_args_sha256", "profile_rows_sha256",
                           "profile_summary_sha256", "row_payload_sha256"}, "native prompt tokens")
            if native["kind"] != "NATIVE_TITAN_PROMPT_TOKEN_IDS":
                raise RegistryError("unknown native prompt token kind")
            for key in set(native) - {"kind"}:
                _hash(native[key], key)
            if native["row_payload_sha256"] != record["fingerprints"]["exact_sha256"]:
                raise RegistryError("native token row binding drift")
    elif kind == "releases":
        _ids(record["row_ids"], "row_ids", False)
        _string(record["registration_mode"], "registration_mode")
        if record["registration_mode"] not in {PROSPECTIVE, RETROSPECTIVE} or record["model_gain"] != "UNMEASURED" or record["semantic_decontamination"] != "NOT_PROVEN":
            raise RegistryError("release cannot claim model gain or semantic decontamination")
    elif kind == "exclusions":
        for key in ("evidence_reference", "coverage_statement"):
            _string(record[key], key)
        _string(record["qualification"], "qualification")
        if record["qualification"] not in {"KNOWN_EXCLUSION_HASHES", "UNQUALIFIED_HISTORICAL_INVENTORY"}:
            raise RegistryError("invalid exclusion qualification")
        _ids(record["forbidden_roles"], "forbidden_roles", False)
        for role in record["forbidden_roles"]:
            _split(role)
        if type(record["identities"]) is not list:
            raise RegistryError("exclusion identities must be a list")
        if not record["identities"] and record["qualification"] == "KNOWN_EXCLUSION_HASHES":
            raise RegistryError("known exclusion must contain at least one identity")
        seen = set()
        for identity in record["identities"]:
            _keys(identity, {"domain", "sha256", "identity_space"}, "exclusion identity")
            _string(identity["domain"], "exclusion domain")
            if identity["domain"] not in _DOMAINS:
                raise RegistryError("invalid exclusion fingerprint domain")
            if identity["identity_space"] != IDENTITY_SPACES[identity["domain"]]:
                raise RegistryError("exclusion hash uses an unknown/mismatched identity space")
            _hash(identity["sha256"], "exclusion sha256")
            pair = (identity["domain"], identity["sha256"])
            if pair in seen:
                raise RegistryError("duplicate exclusion identity")
            seen.add(pair)
    return primary


def _closure(records, identifier, edge):
    result, active = set(), set()
    def visit(current):
        if current not in records:
            raise RegistryError("unknown ancestry ID: " + current)
        if current in active:
            raise RegistryError("ancestry cycle")
        if current in result:
            return
        active.add(current)
        for ancestor in records[current][edge]:
            visit(ancestor)
        active.remove(current)
        result.add(current)
    visit(identifier)
    return result


def _check_state(state):
    sources, parents, rows = state["sources"], state["parents"], state["rows"]
    content_families, revisions, family_scopes, family_safety = {}, {}, {}, {}
    for source_id, source in sources.items():
        closure = _closure(sources, source_id, "ancestor_source_ids")
        family = source["source_family_id"]
        pair = (family, source["revision"])
        if pair in revisions and revisions[pair] != source["content_sha256"]:
            raise RegistryError("same source family/revision changed content")
        revisions[pair] = source["content_sha256"]
        if family in family_scopes and family_scopes[family] != source["split_scope"]:
            raise RegistryError("source family split-scope laundering")
        family_scopes[family] = source["split_scope"]
        if family in family_safety and family_safety[family] != source["safety"]:
            raise RegistryError("source family/revision safety laundering; clearance is not supported by this registry")
        family_safety[family] = source["safety"]
        content = source["content_sha256"]
        if content in content_families and content_families[content] != family:
            raise RegistryError("source content reidentified as another family")
        content_families[content] = family
        if source["split_scope"] == "PRODUCER_ONLY" and any(sources[a]["split_scope"] != "PRODUCER_ONLY" for a in closure):
            raise RegistryError("content derivative cannot become producer-only")
        if source["safety"] == "TRAIN_SAFE" and any(sources[a]["safety"] != "TRAIN_SAFE" for a in closure):
            raise RegistryError("unsafe source ancestry laundering")
    family_roles, normalized_parents, aliases, scoped_instances = {}, {}, {}, {}
    for parent_id, parent in parents.items():
        closure = _closure(parents, parent_id, "ancestor_parent_ids")
        if parent["structural_cluster_id"] not in state["structures"]:
            raise RegistryError("unknown structural cluster")
        normalized = parent["normalized_specification_sha256"]
        if normalized in normalized_parents and normalized_parents[normalized] != parent_id:
            raise RegistryError("duplicate normalized semantic instance")
        normalized_parents[normalized] = parent_id
        for alias in parent["aliases"]:
            if alias in parents and alias != parent_id:
                raise RegistryError("alias shadows another parent ID")
            if alias in aliases and aliases[alias] != parent_id:
                raise RegistryError("immutable parent alias reassignment")
            aliases[alias] = parent_id
        for fingerprint in parent["instance_fingerprints"]:
            key = (fingerprint["algorithm"], fingerprint["sha256"])
            if key in scoped_instances and scoped_instances[key] != parent_id:
                raise RegistryError("duplicate scoped semantic instance")
            scoped_instances[key] = parent_id
        required_sources = set()
        for ancestor in closure:
            if parents[ancestor]["split"] != parent["split"]:
                raise RegistryError("semantic parent ancestry crosses roles")
            if ancestor != parent_id:
                required_sources.update(parents[ancestor]["source_ids"])
        source_closure = set()
        for source_id in parent["source_ids"]:
            source_closure.update(_closure(sources, source_id, "ancestor_source_ids"))
        if not required_sources <= source_closure:
            raise RegistryError("parent derivative dropped source ancestry")
        for source_id in source_closure:
            source = sources[source_id]
            if source["safety"] != "TRAIN_SAFE":
                raise RegistryError("parent uses excluded or unqualified source")
            if source["split_scope"] == "CONTENT_FAMILY":
                family = source["source_family_id"]
                if family in family_roles and family_roles[family] != parent["split"]:
                    raise RegistryError("source family/revision ancestry crosses roles")
                family_roles[family] = parent["split"]
    indexes = {name: {} for name in ("exact_sha256", "normalized_sha256", "lexical_tokens_sha256")}
    prompt_indexes = {name: {} for name in indexes}
    native_instances = {}
    for row_id, row in rows.items():
        parent_id = row["semantic_parent_id"]
        if parent_id not in parents:
            raise RegistryError("row has unknown semantic parent")
        parent = parents[parent_id]
        if row["split"] != parent["split"] or set(row["source_ids"]) != set(parent["source_ids"]):
            raise RegistryError("row changed parent split or source membership")
        for ancestor_id in _closure(rows, row_id, "ancestor_row_ids"):
            ancestor = rows[ancestor_id]
            if ancestor["semantic_parent_id"] not in _closure(parents, parent_id, "ancestor_parent_ids") or ancestor["split"] != row["split"]:
                raise RegistryError("row derivative launders parent ancestry")
        for name, index in indexes.items():
            fingerprint = row["fingerprints"][name]
            if fingerprint in index:
                raise RegistryError("duplicate row content/normalized/lexical instance")
            index[fingerprint] = row_id
        for name, index in prompt_indexes.items():
            fingerprint = row["prompt_fingerprints"][name]
            if fingerprint in index and index[fingerprint] != parent_id:
                raise RegistryError("same prompt reidentified under another parent")
            index[fingerprint] = parent_id
        native = row["native_prompt_tokens"]
        if native is not None:
            key = (native["tokenizer_identity_sha256"], native["chat_template_sha256"],
                   native["template_args_sha256"], native["prompt_ids_sha256"])
            if key in native_instances and native_instances[key] != parent_id:
                raise RegistryError("same native token prompt reidentified under another parent")
            native_instances[key] = parent_id
    for release in state["releases"].values():
        for row_id in release["row_ids"]:
            if row_id not in rows:
                raise RegistryError("release refers to unknown row")
        modes = {parents[rows[r]["semantic_parent_id"]]["registration_mode"] for r in release["row_ids"]}
        if release["registration_mode"] == PROSPECTIVE and modes != {PROSPECTIVE}:
            raise RegistryError("retrospective rows cannot become prospective via a new release")
    for exclusion in state["exclusions"].values():
        # Unqualified inventories still reject any explicitly known matching hash;
        # their lack of other hashes never implies absence of overlap.
        identities = {(item["domain"], item["sha256"]) for item in exclusion["identities"]}
        for parent in parents.values():
            if parent["split"] not in exclusion["forbidden_roles"]:
                continue
            candidates = {("parent_specification", parent["specification_sha256"]),
                          ("parent_normalized", parent["normalized_specification_sha256"])}
            for source_id in parent["source_ids"]:
                for ancestor in _closure(sources, source_id, "ancestor_source_ids"):
                    candidates.add(("source_content", sources[ancestor]["content_sha256"]))
            if candidates & identities:
                raise RegistryError("known exclusion overlap in parent/source")
        for row in rows.values():
            if row["split"] not in exclusion["forbidden_roles"]:
                continue
            candidates = set()
            for prefix, field in (("row", "fingerprints"), ("prompt", "prompt_fingerprints")):
                for domain, key in (("exact", "exact_sha256"), ("normalized", "normalized_sha256"), ("lexical_tokens", "lexical_tokens_sha256")):
                    candidates.add((prefix + "_" + domain, row[field][key]))
            if row["native_prompt_tokens"] is not None:
                candidates.add(("native_titan_prompt_token_ids", row["native_prompt_tokens"]["prompt_ids_sha256"]))
            if candidates & identities:
                raise RegistryError("known exclusion overlap in row/prompt")


def _apply(state, batch, sequence):
    _keys(batch, {"schema_version", *_COLLECTIONS}, "batch")
    if type(batch["schema_version"]) is not int or batch["schema_version"] != SCHEMA_VERSION:
        raise RegistryError("invalid schema_version")
    candidate = copy.deepcopy(state)
    additions = 0
    new_parents = set()
    for kind in _COLLECTIONS:
        if type(batch[kind]) is not list:
            raise RegistryError(kind + ": expected list")
        seen = set()
        for record in batch[kind]:
            primary = _validate_record(kind, record)
            identifier = record[primary]
            if identifier in seen:
                raise RegistryError("duplicate ID within batch: " + identifier)
            seen.add(identifier)
            previous = candidate[kind].get(identifier)
            if previous is not None:
                if canonical(previous) != canonical(record):
                    raise RegistryError("immutable identity reassignment/drift: " + identifier)
                continue  # Exact version reuse is not a new independent parent.
            candidate[kind][identifier] = copy.deepcopy(record)
            additions += 1
            if kind == "parents":
                candidate["parent_reserved_sequence"][identifier] = sequence
                new_parents.add(identifier)
    if not additions:
        raise RegistryError("no new registration; byte-identical content already registered")
    for row in batch["rows"]:
        parent = candidate["parents"].get(row["semantic_parent_id"])
        if parent and parent["prospective"] and candidate["parent_reserved_sequence"][row["semantic_parent_id"]] >= sequence:
            raise RegistryError("prospective parent must commit before row generation/registration")
    _check_state(candidate)
    return candidate


def _summary(state):
    role_counts = {split: 0 for split in sorted(SPLITS)}
    structural = {}
    for parent in state["parents"].values():
        role_counts[parent["split"]] += 1
        structural.setdefault(parent["structural_cluster_id"], set()).add(parent["split"])
    # Registration IDs are not an independence denominator: derivative parent
    # specifications stay grouped with every declared ancestor.
    group_roots = {identifier: identifier for identifier in state["parents"]}
    def group(identifier):
        while group_roots[identifier] != identifier:
            identifier = group_roots[identifier]
        return identifier
    for identifier, parent in state["parents"].items():
        for ancestor in parent["ancestor_parent_ids"]:
            group_roots[group(identifier)] = group(ancestor)
    lineage_groups = {group(identifier) for identifier in group_roots}
    return {"unique_sources": len(state["sources"]),
            "source_families": len({s["source_family_id"] for s in state["sources"].values()}),
            "unique_semantic_parents": len(state["parents"]),
            "declared_parent_lineage_groups": len(lineage_groups),
            "derivative_parent_specifications": sum(bool(p["ancestor_parent_ids"]) for p in state["parents"].values()),
            "parents_by_split": role_counts, "unique_rows": len(state["rows"]),
            "release_versions": len(state["releases"]), "structural_clusters": len(structural),
            "structural_roles": {key: sorted(value) for key, value in sorted(structural.items())},
            "shared_structure_disclosure": "SHARED_STRUCTURE_PERMITS_INSTANCE_DEV_NOT_UNSEEN_STRUCTURE",
            "prospective_parents": sum(p["prospective"] for p in state["parents"].values()),
            "retrospective_parents": sum(not p["prospective"] for p in state["parents"].values()),
            "immutable_parent_aliases": sum(len(p["aliases"]) for p in state["parents"].values()),
            "native_prompt_token_fingerprints": sum(r["native_prompt_tokens"] is not None for r in state["rows"].values()),
            "scoped_instance_fingerprint_algorithms": sorted({f["algorithm"] for p in state["parents"].values() for f in p["instance_fingerprints"]}),
            "known_exclusion_inventories": sum(e["qualification"] == "KNOWN_EXCLUSION_HASHES" for e in state["exclusions"].values()),
            "unqualified_historical_inventories": sum(e["qualification"] == "UNQUALIFIED_HISTORICAL_INVENTORY" for e in state["exclusions"].values()),
            "historical_semantic_overlap": "NOT_AUDITED", "model_gain": "UNMEASURED",
            "coverage": copy.deepcopy(COVERAGE)}


class Registry:
    """Single-host write-once snapshots; external backups anchor against deletion.

    The API detects edits, gaps and invalid replay. It cannot protect a directory
    from its owner deleting/reconstructing all snapshots. Preserve tail hashes in
    external cycle receipts. A lock has no timeout or automatic stale takeover.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()

    def _read(self):
        directory = self.path / "snapshots"
        state, tail, count = _initial_state(), None, 0
        if not directory.exists():
            return state, tail, count
        paths = sorted(directory.iterdir(), key=lambda p: p.name)
        for path in paths:
            if path.name.startswith(".pending-"):
                continue  # Never a committed snapshot; retained after a crash.
            match = _SNAPSHOT.fullmatch(path.name)
            if not match or not path.is_file() or path.is_symlink():
                raise RegistryError("unexpected snapshot entry: " + path.name)
            sequence = int(match.group(1))
            if sequence != count + 1:
                raise RegistryError("snapshot sequence gap or collision")
            data = path.read_bytes()
            sha256 = hashlib.sha256(data).hexdigest()
            if sha256 != match.group(2):
                raise RegistryError("snapshot byte hash drift")
            record = strict_json(data.decode("utf-8"))
            _keys(record, {"schema_version", "sequence", "previous_snapshot_sha256",
                           "created_utc", "cycle_id", "batch", "summary"}, "snapshot")
            if type(record["sequence"]) is not int or record["sequence"] != sequence or type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
                raise RegistryError("snapshot metadata drift")
            if record["previous_snapshot_sha256"] != tail:
                raise RegistryError("broken snapshot hash chain")
            _id(record["cycle_id"], "cycle_id")
            _string(record["created_utc"], "created_utc")
            try:
                timestamp = datetime.fromisoformat(record["created_utc"])
            except ValueError as error:
                raise RegistryError("invalid snapshot UTC timestamp") from error
            if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
                raise RegistryError("snapshot timestamp must be UTC")
            if data != (canonical(record) + "\n").encode("utf-8"):
                raise RegistryError("noncanonical snapshot serialization")
            state = _apply(state, record["batch"], sequence)
            if canonical(record["summary"]) != canonical(_summary(state)):
                raise RegistryError("snapshot summary drift")
            tail, count = sha256, sequence
        return state, tail, count

    def snapshot(self):
        state, tail, count = self._read()
        return {"schema_version": SCHEMA_VERSION, "sequence": count,
                "tail_sha256": tail, "state": state, "summary": _summary(state)}

    def summary(self):
        return self.snapshot()["summary"]

    @contextmanager
    def _lock(self, cycle_id):
        _id(cycle_id, "cycle_id")
        self.path.mkdir(parents=True, exist_ok=True)
        lock = self.path / ".registry-cycle.lock"
        token = secrets.token_hex(32)
        payload = {"cycle_id": cycle_id, "token": token, "pid": os.getpid(),
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "stale_policy": "NEVER_AUTOMATICALLY_STEAL_OR_DELETE"}
        assert_owned = None
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise RegistryLocked("another cycle owns the registry lock") from error
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((canonical(payload) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            def assert_owned():
                try:
                    current = strict_json(lock.read_text(encoding="utf-8"))
                except (FileNotFoundError, RegistryError) as error:
                    raise RegistryLocked("registry cycle lock disappeared or changed") from error
                if canonical(current) != canonical(payload):
                    raise RegistryLocked("registry lock ownership changed; lock retained")
            assert_owned.committed_receipt = None
            yield assert_owned
        finally:
            # Never remove a lock whose ownership changed while work was running.
            receipt = None if assert_owned is None else assert_owned.committed_receipt
            try:
                if lock.exists():
                    current = strict_json(lock.read_text(encoding="utf-8"))
                    if canonical(current) != canonical(payload):
                        raise RegistryLocked("registry lock ownership changed; lock retained")
                    lock.unlink()
                elif receipt is not None:
                    raise RegistryLocked("registry lock disappeared after publication")
            except OSError as error:
                if receipt is None:
                    raise
                receipt.setdefault("cleanup_warnings", []).append("Registry lock cleanup failed; committed snapshot is intact: " + str(error))
            except RegistryError as error:
                if receipt is None:
                    raise
                raise CommittedRegistrationError("Complete registration committed, then lock ownership/integrity failed", receipt) from error

    def register(self, batch, *, cycle_id):
        """Validate all records before one atomic append; conflicts leave no commit.

        Hard-link publication is atomic and refuses overwrite. If the filesystem
        cannot support it, registration fails safely; it never falls back to a
        truncating overwrite. Directory fsync is best effort on Windows.
        After publication cleanup errors return a committed receipt with warnings;
        changed lease ownership raises CommittedRegistrationError with that receipt.
        """
        batch = strict_json(canonical(batch))  # Detach caller-owned mutable objects.
        with self._lock(cycle_id) as assert_owned:
            state, tail, count = self._read()
            candidate = _apply(state, batch, count + 1)
            record = {"schema_version": SCHEMA_VERSION, "sequence": count + 1,
                      "previous_snapshot_sha256": tail,
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "cycle_id": cycle_id, "batch": batch, "summary": _summary(candidate)}
            data = (canonical(record) + "\n").encode("utf-8")
            sha256 = hashlib.sha256(data).hexdigest()
            directory = self.path / "snapshots"
            directory.mkdir(exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=directory)
            pending = Path(name)
            committed = None
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                final = directory / f"{count + 1:08d}-{sha256}.json"
                assert_owned()
                os.link(pending, final)
                committed = {"sequence": count + 1, "snapshot_sha256": sha256,
                             "snapshot_path": str(final), "summary": record["summary"]}
                assert_owned.committed_receipt = committed
                try:
                    directory_fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass
            finally:
                try:
                    pending.unlink(missing_ok=True)
                except OSError as error:
                    if committed is None:
                        raise
                    committed.setdefault("cleanup_warnings", []).append("Pending hard-link cleanup failed; committed snapshot is intact: " + str(error))
            return committed


def pilot_v3_batch(foundry_root, *, scoped_fingerprints=None):
    """Retrospective import of only pilot-v3; never invokes/imports its producer.

    The old parent IDs are aliases recorded in immutable metadata,
    not evidence of pre-generation registration. Version reuse should reference
    these registered row IDs in another make_release() without reimporting rows.
    """
    root = Path(foundry_root).resolve()
    canonical_path = root / "candidates" / "pilot-v3" / "canonical.jsonl"
    manifest_path = root / "releases" / "pilot-v3" / "release_manifest.json"
    manifest = strict_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_sha256") != file_digest(canonical_path):
        raise RegistryError("pilot canonical bytes do not match release manifest")
    if manifest.get("historical_semantic_overlap") != "NOT_AUDITED" or manifest.get("training_ready") is not False:
        raise RegistryError("pilot retrospective evidence has unexpected admission claims")
    batch, structures = empty_batch(), {}
    profile_rows_path = root / "profiles" / "pilot-v3" / "rows.jsonl"
    profile_summary_path = root / "profiles" / "pilot-v3" / "summary.json"
    profile_summary = strict_json(profile_summary_path.read_text(encoding="utf-8"))
    profile_rows_sha256 = file_digest(profile_rows_path)
    if (profile_summary.get("canonical_sha256") != manifest["canonical_sha256"] or
            profile_summary.get("rows_sha256") != profile_rows_sha256 or
            file_digest(profile_summary_path) != manifest.get("tokenization_sha256")):
        raise RegistryError("pilot native profile byte binding drift")
    native_rows = {}
    with profile_rows_path.open(encoding="utf-8") as stream:
        for line in stream:
            profile_row = strict_json(line)
            identifier = profile_row.get("row_id")
            _id(identifier, "profile row_id")
            if identifier in native_rows:
                raise RegistryError("duplicate profile row ID")
            native_rows[identifier] = profile_row
    _hash(profile_summary.get("chat_template_sha256"), "pilot chat_template_sha256")
    if type(profile_summary.get("tokenizer_files")) is not dict or not profile_summary["tokenizer_files"]:
        raise RegistryError("pilot native tokenizer identity missing")
    for tokenizer_hash in profile_summary["tokenizer_files"].values():
        _hash(tokenizer_hash, "pilot tokenizer file hash")
    # A producer bundle is distinct from source-content families; it may support
    # independently specified parents in multiple roles.
    source_identity = manifest.get("source_identity")
    if type(source_identity) is not dict or not source_identity:
        raise RegistryError("missing pilot producer identities")
    for relative, expected in source_identity.items():
        _hash(expected, "pilot producer hash")
        if type(relative) is not str or not relative.startswith("scripts/"):
            raise RegistryError("pilot producer path is outside scripts")
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "scripts") or file_digest(path) != expected:
            raise RegistryError("pilot producer drift or path escape")
    producer = make_source("pilot-v3-producer", "iqf-pilot-producer", "v3",
                           digest(source_identity), origin="releases/pilot-v3/release_manifest.json#source_identity",
                           owner="Rineya foundry", licence_evidence="Original local synthetic producer; imported asset rights not established",
                           use_permission="Local research registry only; no training authorization",
                           split_scope="PRODUCER_ONLY")
    batch["sources"].append(producer)
    original_rows = []
    with canonical_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise RegistryError("blank pilot canonical row")
            row = strict_json(line)
            _split(row.get("split"))
            _id(row.get("row_id"), "legacy row_id")
            _id(row.get("parent_id"), "legacy parent_id")
            expected = row.get("row_sha256")
            _hash(expected, "legacy row_sha256")
            unhashed = {key: value for key, value in row.items() if key != "row_sha256"}
            if digest(unhashed) != expected:
                raise RegistryError("pilot row digest drift")
            original_rows.append(row)
    if type(manifest.get("rows")) is not int or manifest["rows"] != len(original_rows):
        raise RegistryError("pilot row count differs from release manifest")
    for row in original_rows:
        family = row.get("family")
        _id(family, "pilot family")
        scoped = None if scoped_fingerprints is None else scoped_fingerprints.get(row["row_id"])
        if scoped_fingerprints is not None and (type(scoped) is not dict or scoped.get("row_sha256") != row["row_sha256"] or scoped.get("split") != row["split"]):
            raise RegistryError("scoped pilot fingerprint missing or row binding drift")
        structure_spec = ({"producer_family": family, "template_version": "pilot-v3"}
                          if scoped is None else scoped["structural_specification"])
        structure_key = digest(structure_spec)
        if scoped is not None and structure_key != scoped.get("structural_sha256"):
            raise RegistryError("scoped structure specification hash drift")
        if structure_key not in structures:
            structure = make_structure(structure_spec,
                                       description="Shared pilot algorithm/writing template; instance DEV is exposed and not unseen-structure DEV")
            structures[structure_key] = structure
            batch["structures"].append(structure)
        world = row.get("world")
        _specification(world)
        provenance = row.get("provenance")
        if type(provenance) is not dict or provenance.get("kind") != "original_synthetic" or provenance.get("source_id") != row["parent_id"]:
            raise RegistryError("pilot origin/source identity mismatch")
        source_id = "world-" + digest(world)
        world_source = make_source(source_id, "sf-" + digest(world), "pilot-world-v0", digest(world),
                                   origin="candidates/pilot-v3/canonical.jsonl#" + row["row_id"],
                                   owner="Rineya foundry", licence_evidence="Original synthetic world; declared " + str(provenance.get("license")),
                                   use_permission="Reviewed research pilot; training_ready=false; historical overlap NOT_AUDITED")
        batch["sources"].append(world_source)
        specification = {"family": family,
                         "world": world, "full_public_rules": row["messages"][0]["content"]}
        parent = make_parent(specification, construction_schema="iqf-pilot-v3-retrospective-full-world-rules-v1",
                             split=row["split"], source_ids=[producer["source_id"], source_id],
                             structural_cluster_id=structures[structure_key]["structural_cluster_id"],
                             aliases=[row["parent_id"]],
                             instance_fingerprints=[] if scoped is None else [
                                 {"algorithm": "PILOT_V3_ACTIVE_TASK_EXACT_V1", "sha256": scoped["active_world_sha256"],
                                  "coverage": "Pilot family scoped; ignores archives, notes and record IDs; not universal semantics"},
                                 {"algorithm": "PILOT_V3_SCOPED_NORMALIZED_INSTANCE_V1", "sha256": scoped["scoped_normalized_instance_sha256"],
                                  "coverage": "Limited entity-group normalization; graph isomorphism structural only; no historical comparison"}],
                             registration_mode=RETROSPECTIVE)
        batch["parents"].append(parent)
        profile_row = native_rows.get(row["row_id"])
        if profile_row is None or profile_row.get("row_sha256") != row["row_sha256"] or profile_row.get("split") != row["split"]:
            raise RegistryError("native profile row binding drift")
        native = {"kind": "NATIVE_TITAN_PROMPT_TOKEN_IDS", "prompt_ids_sha256": profile_row["prompt_ids_sha256"],
                  "tokenizer_identity_sha256": digest(profile_summary["tokenizer_files"]),
                  "chat_template_sha256": profile_summary["chat_template_sha256"],
                  "template_args_sha256": digest(profile_summary["template_args"]),
                  "profile_rows_sha256": profile_rows_sha256,
                  "profile_summary_sha256": file_digest(profile_summary_path),
                  "row_payload_sha256": digest({"messages": row["messages"], "target": row["target"]})}
        batch["rows"].append(make_row(row["row_id"], parent["semantic_parent_id"], row["split"],
                                       {"messages": row["messages"], "target": row["target"]},
                                       source_ids=parent["source_ids"], native_prompt_tokens=native))
    if set(native_rows) != {row["row_id"] for row in original_rows}:
        raise RegistryError("native profile row coverage mismatch")
    if scoped_fingerprints is not None and set(scoped_fingerprints) != set(native_rows):
        raise RegistryError("scoped fingerprint row coverage mismatch")
    batch["releases"].append(make_release("pilot-v3", [r["row_id"] for r in original_rows],
                                           manifest_sha256=file_digest(manifest_path),
                                           canonical_sha256=file_digest(canonical_path)))
    return batch
