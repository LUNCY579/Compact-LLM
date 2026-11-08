"""Hash-bound, TRAIN-only language replay candidate selection, without models.

V3's recorded source-version label is not trusted. Exact original V2 membership
is established after removing only the two documented V3 metadata additions.
Exports retain each chosen V3 record's original bytes, with separate lineage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

VERSION = "rineya-train-language-candidates-v1"
ADDED_FIELDS = ("preservation_replay_for", "preservation_source_version")
CLASS_QUOTAS = {
    "acknowledgment_update": 12, "apology_repair": 12, "boundary_setting": 16,
    "clarification_request": 16, "communication_rewrite": 12, "comparative_recommendation": 12,
    "conversation_closure": 16, "decision_explanation": 12, "deescalation": 16,
    "empathy_validation": 16, "evidence_grounded_dialogue": 12, "informal_texting": 16,
    "multi_turn_context_update": 12, "natural_follow_up": 12, "negotiation_compromise": 16,
    "plan_revision": 12, "practical_next_step": 12, "preference_opinion": 12,
    "tactful_disagreement": 12,
}


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant " + value)


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _integer(value, low=0, high=1000000):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid bounded integer")
    return value


def _string(value):
    if type(value) is not str or not value or len(value) > 512:
        raise ValueError("missing/invalid identifier")
    return value


def resolve_bound_path(root: Path, relative: str) -> Path:
    relative = _string(relative).replace("\\", "/")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source paths must stay within the supplied source root")
    root, resolved = root.resolve(), (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("source path resolves outside its bound root")
    return resolved


def _validate_binding(binding):
    if type(binding) is not dict or not re.fullmatch("[0-9a-f]{64}", binding.get("sha256", "")):
        raise ValueError("source requires a SHA256 binding")
    _string(binding.get("path"))
    _integer(binding.get("bytes"), 1, 1024**3)
    _integer(binding.get("rows"), 1)
    if binding.get("all_split_train") is not True:
        raise ValueError("bound source must be TRAIN only")


def validate_registration(registration: dict, audit: dict) -> None:
    if registration.get("status") != "FROZEN_TRAIN_LANGUAGE_CANDIDATES_ONLY":
        raise ValueError("registration is not frozen for TRAIN candidate selection")
    authorization = {"read_bound_TRAIN_sources": True, "candidate_export": True,
                     "native_execution": False, "training": False, "CHECK": False, "FINAL": False}
    if any(registration.get("authorization", {}).get(key) is not value for key, value in authorization.items()):
        raise ValueError("authorization must be TRAIN-source reads and candidate export only")
    for key in ("native_model_outcome_selection", "source_correct_training_filter", "training_eligibility", "e1_launch_ready"):
        if registration.get(key) is not False:
            raise ValueError("model selection and training admission are forbidden")
    if registration.get("class_quotas") != CLASS_QUOTAS:
        raise ValueError("the exact frozen 19-class, 256-row allocation is required")
    if any(type(value) is not int for value in registration["class_quotas"].values()):
        raise ValueError("quota counts must be integers")
    if (registration.get("parent_cluster") != "v2.0:metadata.base_template_family"
            or registration.get("one_primary_row_per_cluster") is not True
            or registration.get("original_rows_immutable") is not True
            or type(registration.get("bounded_alternatives_per_selected_cluster")) is not int
            or registration["bounded_alternatives_per_selected_cluster"] != 3):
        raise ValueError("invalid cluster, immutable-row or alternatives contract")
    _integer(registration.get("selection_seed"), 0, 2**63 - 1)
    _integer(registration.get("maximum_candidate_rows_read"), 1)
    _integer(registration.get("expected_eligible_rows"), 256)
    _integer(registration.get("expected_eligible_clusters"), 256)
    for key in ("source", "original_v2_source"):
        _validate_binding(registration.get(key))
    if registration["source"] != audit.get("source"):
        raise ValueError("preservation source differs from bound audit")
    originals = [item for item in audit.get("original_train_sources", []) if item.get("version") == "v2.0"]
    if len(originals) != 1 or registration["original_v2_source"] != originals[0]:
        raise ValueError("original V2 binding differs from bound audit")
    if audit.get("origin_comparison_removed_metadata_fields") != list(ADDED_FIELDS):
        raise ValueError("audit does not bind the exact two removable metadata fields")
    if audit.get("counterfactual_base_group_conflicts") != {}:
        raise ValueError("audit reports unresolved counterfactual parent conflicts")
    if registration["source"]["rows"] > registration["maximum_candidate_rows_read"]:
        raise ValueError("source exceeds frozen candidate-read ceiling")


def _record(raw: bytes) -> dict:
    row = parse_json(raw)
    if type(row) is not dict or set(row) != {"id", "messages", "metadata", "system_policy_version"}:
        raise ValueError("unexpected source row schema")
    _string(row["id"])
    _string(row["system_policy_version"])
    metadata = row["metadata"]
    if type(metadata) is not dict or metadata.get("split") != "train":
        raise ValueError("non-TRAIN row in authorized TRAIN-only source")
    messages = row["messages"]
    if type(messages) is not list or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError("invalid conversation structure")
    for i, message in enumerate(messages):
        expected = "system" if i == 0 else ("user" if i % 2 else "assistant")
        if (type(message) is not dict or set(message) != {"role", "content"} or message["role"] != expected
                or type(message["content"]) is not str or not message["content"].strip()):
            raise ValueError("invalid role order or empty message")
    return row


def _stream(path: Path, binding: dict):
    """Hash every byte while streaming; consumers must exhaust before export."""
    digest, byte_count, count = hashlib.sha256(), 0, 0
    with path.open("rb") as handle:
        for count, raw in enumerate(handle, 1):
            if count > binding["rows"]:
                raise ValueError("bound source row ceiling exceeded")
            digest.update(raw)
            byte_count += len(raw)
            if byte_count > binding["bytes"] or not raw.strip():
                raise ValueError("bound source byte ceiling exceeded or blank row")
            yield count, raw, _record(raw)
    if (count, byte_count, digest.hexdigest()) != (binding["rows"], binding["bytes"], binding["sha256"]):
        raise ValueError("source byte/hash/row binding mismatch")


@dataclass(frozen=True)
class Origin:
    row_id: str
    line_number: int
    canonical_sha256: str
    raw_line_sha256: str
    base_family: str
    reply_class: str
    capability_group: str
    counterfactual_group: str | None


@dataclass(frozen=True)
class Candidate:
    row: dict
    raw_line: bytes
    line_number: int
    origin: Origin

    @property
    def cluster(self):
        return "v2.0:" + self.origin.base_family


def scan_original(path: Path, binding: dict) -> dict[str, Origin]:
    inventory, counterfactual_clusters = {}, {}
    for number, raw, row in _stream(path, binding):
        if row["id"] in inventory:
            raise ValueError("duplicate original TRAIN row ID")
        metadata = row["metadata"]
        if any(key in metadata for key in ADDED_FIELDS):
            raise ValueError("original source unexpectedly contains V3 metadata additions")
        base = _string(metadata.get("base_template_family"))
        cls, capability = _string(metadata.get("reply_class")), _string(metadata.get("capability_group"))
        counterfactual = metadata.get("counterfactual_group_id")
        if counterfactual is not None:
            _string(counterfactual)
            if counterfactual_clusters.setdefault(counterfactual, base) != base:
                raise ValueError("counterfactual group crosses recovered base families")
        inventory[row["id"]] = Origin(row["id"], number, sha256(json_bytes(row)), sha256(raw),
                                      base, cls, capability, counterfactual)
    return inventory


def exclusions(audit: dict) -> tuple[set[str], set[str]]:
    families, ids = set(), set()
    flags = audit.get("quality_flags")
    if type(flags) is not list:
        raise ValueError("audit quality flags missing")
    for flag in flags:
        if type(flag) is not dict or type(flag.get("affected_row_ids")) is not list:
            raise ValueError("invalid audited quality flag")
        ids.update(_string(row_id) for row_id in flag["affected_row_ids"])
        if flag.get("scope") == "entire_family":
            families.add(_string(flag.get("family")))
    return families, ids


def scan_candidates(path: Path, binding: dict, inventory: dict[str, Origin], audit: dict) -> tuple[list[Candidate], dict]:
    families, affected_ids = exclusions(audit)
    seen, candidates, counters = set(), [], Counter()
    for number, raw, row in _stream(path, binding):
        row_id = row["id"]
        if row_id in seen:
            raise ValueError("duplicate preservation row ID")
        seen.add(row_id)
        counters["preservation_rows"] += 1
        origin = inventory.get(row_id)
        if origin is None:
            counters["not_original_v2_members"] += 1
            continue
        metadata = row["metadata"]
        if any(key not in metadata for key in ADDED_FIELDS):
            raise ValueError("V3 record lacks one of the two documented additions")
        stripped = {**row, "metadata": {key: value for key, value in metadata.items() if key not in ADDED_FIELDS}}
        if sha256(json_bytes(stripped)) != origin.canonical_sha256:
            raise ValueError("preservation member differs from exact original V2 record")
        counters["exact_original_v2_members"] += 1
        if origin.capability_group not in {"communication", "integrated"}:
            counters["excluded_nonlanguage_capability"] += 1
        elif origin.reply_class == "roleplay_reasoning":
            counters["excluded_roleplay_reasoning"] += 1
        elif origin.base_family in families or row_id in affected_ids:
            counters["excluded_audit_quality_flags"] += 1
        else:
            candidates.append(Candidate(row, raw, number, origin))
    counters["eligible_rows"] = len(candidates)
    counters["eligible_clusters"] = len({candidate.cluster for candidate in candidates})
    return candidates, dict(counters)


def cluster_order(seed: int, cls: str, cluster: str) -> str:
    return sha256(json_bytes([seed, cls, cluster]))


def row_order(seed: int, candidate: Candidate) -> str:
    return sha256(json_bytes([seed, candidate.origin.row_id, candidate.origin.canonical_sha256]))


def select_candidates(candidates: list[Candidate], quotas: dict[str, int], seed: int, alternatives: int = 3):
    """Fixed class/cluster and row hash order; no quality/model outcome inputs."""
    _integer(seed, 0, 2**63 - 1)
    _integer(alternatives, 0, 3)
    groups, cluster_classes = defaultdict(lambda: defaultdict(list)), {}
    for candidate in candidates:
        cls = candidate.origin.reply_class
        if cls not in quotas:
            raise ValueError("eligible class has no frozen quota; no implicit substitution")
        if cluster_classes.setdefault(candidate.cluster, cls) != cls:
            raise ValueError("base cluster crosses reply classes")
        groups[cls][candidate.cluster].append(candidate)
    primary, backups, ranks = [], [], {}
    for cls, quota in sorted(quotas.items()):
        _integer(quota, 1)
        if len(groups[cls]) < quota:
            raise ValueError(f"class {cls} has insufficient distinct clusters; quota substitution forbidden")
        clusters = sorted(groups[cls], key=lambda cluster: (cluster_order(seed, cls, cluster), cluster))
        for family_rank, cluster in enumerate(clusters[:quota], 1):
            ordered = sorted(groups[cls][cluster], key=lambda item: (row_order(seed, item), item.origin.row_id))
            for within_rank, candidate in enumerate(ordered[:alternatives + 1], 1):
                (primary if within_rank == 1 else backups).append(candidate)
                ranks[candidate.origin.row_id] = {"family_rank": family_rank, "within_family_row_rank": within_rank,
                                                 "family_order_sha256": cluster_order(seed, cls, cluster),
                                                 "row_order_sha256": row_order(seed, candidate)}
    return primary, backups, ranks


def _lineage(candidate: Candidate, registration: dict, audit: dict, ranks: dict, primary: bool) -> dict:
    row, original = candidate.row, candidate.origin
    metadata = row["metadata"]
    prompt = row["messages"][:-1]
    normalized_prompt = [{"role": message["role"], "content": " ".join(message["content"].split())} for message in prompt]
    return {
        "schema_version": 1, "row_id": row["id"], "original_row_id": original.row_id,
        "role": "TRAIN", "candidate_kind": "PRIMARY" if primary else "RANKED_ALTERNATIVE_SAME_CLUSTER",
        "audit_id": audit["audit_id"], "audit_sha256": registration["source_audit_sha256"],
        "source_file_sha256": registration["source"]["sha256"], "source_line_number": candidate.line_number,
        "source_raw_line_sha256": sha256(candidate.raw_line),
        "original_train_file_sha256": registration["original_v2_source"]["sha256"],
        "original_line_number": original.line_number, "original_raw_line_sha256": original.raw_line_sha256,
        "canonical_original_record_sha256": original.canonical_sha256,
        "recovered_source_version": "v2.0", "source_version_evidence": "exact original TRAIN membership and structural digest",
        "recorded_preservation_source_version": metadata["preservation_source_version"],
        "base_template_family": original.base_family, "template_family": metadata.get("template_family"),
        "parent_cluster": candidate.cluster, "counterfactual_group_id": original.counterfactual_group,
        "reply_class": original.reply_class, "capability_group": original.capability_group,
        "system_policy_version": row["system_policy_version"], "source_provenance": metadata.get("source_provenance"),
        "human_review_status": metadata.get("human_review_status"), "semantic_target_review": "PENDING",
        "target": row["messages"][-1]["content"], "target_sha256": sha256(row["messages"][-1]["content"].encode()),
        "normalized_prompt_hash": sha256(json_bytes(normalized_prompt)),
        "prompt_normalization": "preserve role sequence; collapse Unicode whitespace within each content",
        "tokenized_prompt_hash": None, "tokenizer_hash": None, "chat_template_hash": None,
        "full_context_tokens": None, "target_tokens": None, "target_quality_flags": [],
        "exclusion_inventory_revision": "bound audit quality_flags only; broader historical exclusions remain unqualified",
        "selection_seed": registration["selection_seed"], **ranks[row["id"]],
        "training_eligibility": False, "semantic_rubric_revision": None,
        "source_correct_membership": None, "model_outcome_selection": False,
        "alternatives_policy": "review candidates only; no automatic replacement based on native model outcomes",
        "statistical_independence_claim": False,
    }


def build_selection(registration_path: Path, source_root: Path, output: Path, source_revision: str) -> dict:
    if type(source_revision) is not str or not re.fullmatch("[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase Git SHA")
    if output.exists():
        raise FileExistsError("selection export is write-once; output already exists")
    started = time.perf_counter()
    raw_registration = registration_path.read_bytes()
    registration = parse_json(raw_registration)
    audit_path = resolve_bound_path(source_root, registration.get("source_audit_path"))
    raw_audit = audit_path.read_bytes()
    if sha256(raw_audit) != registration.get("source_audit_sha256"):
        raise ValueError("source audit hash mismatch")
    audit = parse_json(raw_audit)
    validate_registration(registration, audit)
    original = scan_original(resolve_bound_path(source_root, registration["original_v2_source"]["path"]),
                             registration["original_v2_source"])
    candidates, counts = scan_candidates(resolve_bound_path(source_root, registration["source"]["path"]),
                                         registration["source"], original, audit)
    if counts.get("exact_original_v2_members", 0) != registration["original_v2_source"].get("preservation_members"):
        raise ValueError("exact original V2 membership count differs from frozen audit")
    if (counts["eligible_rows"], counts["eligible_clusters"]) != (
            registration["expected_eligible_rows"], registration["expected_eligible_clusters"]):
        raise ValueError("eligible pool differs from frozen row/cluster counts")
    primary, alternatives, ranks = select_candidates(candidates, registration["class_quotas"], registration["selection_seed"])
    if len(primary) != 256 or len({item.cluster for item in primary}) != 256:
        raise ValueError("exact 256 primary rows and distinct conservative clusters required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "REGISTRATION.json").write_bytes(raw_registration)
    for name, selected in (("TRAIN_CANDIDATES.jsonl", primary), ("ALTERNATIVE_CANDIDATES.jsonl", alternatives)):
        with (output / name).open("xb") as handle:
            for item in selected:
                # Include a missing terminal newline only as a JSONL separator;
                # the exact source raw-line hash remains in lineage.
                handle.write(item.raw_line)
                if not item.raw_line.endswith(b"\n"):
                    handle.write(b"\n")
    with (output / "LINEAGE.jsonl").open("xb") as handle:
        for is_primary, selected in ((True, primary), (False, alternatives)):
            for item in selected:
                handle.write(json_bytes(_lineage(item, registration, audit, ranks, is_primary)) + b"\n")
    files = {}
    for name in ("REGISTRATION.json", "TRAIN_CANDIDATES.jsonl", "ALTERNATIVE_CANDIDATES.jsonl", "LINEAGE.jsonl"):
        raw = (output / name).read_bytes()
        files[name] = {"sha256": sha256(raw), "bytes": len(raw)}
    receipt = {
        "schema_version": 1, "status": "MEASURED_TRAIN_LANGUAGE_CANDIDATES_REVIEW_PENDING",
        "generator_version": VERSION, "generator_file_sha256": sha256(Path(__file__).read_bytes()),
        "source_revision": source_revision, "source_revision_verification": "caller-supplied Git SHA; executed generator bytes bound separately",
        "registration_sha256": sha256(raw_registration), "source_audit_sha256": sha256(raw_audit),
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": registration["source"],
        "original_v2_source": registration["original_v2_source"], "files": files, "counts": counts,
        "primary_rows": len(primary), "primary_conservative_clusters": len({item.cluster for item in primary}),
        "class_counts": dict(sorted(Counter(item.origin.reply_class for item in primary).items())),
        "alternative_rows": len(alternatives), "maximum_alternatives_per_selected_cluster": 3,
        "ordering": "cluster SHA256(canonical JSON [seed,class,parent_cluster]); row SHA256(canonical JSON [seed,row_id,canonical_original_record_sha256]); ascendinghex then identifier",
        "selection_seed": registration["selection_seed"], "source_version_label_used_for_membership": False,
        "removed_for_origin_comparison_only": list(ADDED_FIELDS), "original_row_payloads_modified": False,
        "all_exported_derivatives_role": "TRAIN", "counterfactual_cross_cluster_conflicts": 0,
        "semantic_target_review": "PENDING", "human_review_fabricated": False,
        "source_human_review_status_counts": dict(Counter(item.row["metadata"].get("human_review_status", "MISSING") for item in primary)),
        "native_model_outcome_selection": False, "source_correct_training_filter": False,
        "tokenizer_context_qualified": False, "native_baseline_qualified": False,
        "historical_exclusions_qualified": False, "training_eligibility": False, "e1_launch_ready": False,
        "fresh_language_dev_qualified": False, "language_preservation_qualified": False,
        "statistical_independence_claim": False,
        "limitations": ["conservative template clusters are training coverage labels, not independent held-out parents",
                        "exact historical membership does not establish semantic target quality",
                        "no CHECK/FINAL payloads or fresh language DEV produced",
                        "alternatives are review-only; no native-outcome-based substitutions"],
        "license_provenance": audit.get("license"),
        "cost": {"wall_seconds": time.perf_counter() - started, "model_calls": 0, "teacher_calls": 0,
                 "gpu_seconds": 0, "actual_training_tokens": 0, "protected_content_reads": 0},
        "runtime": {"python": sys.version, "platform": sys.platform, "dependencies": "Python standard library only"},
    }
    (output / "DATA_RECEIPT.json").write_bytes(json_bytes(receipt) + b"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    receipt = build_selection(args.registration, args.source_root, args.output, args.source_revision)
    path = args.output / "DATA_RECEIPT.json"
    print(json.dumps({"receipt": str(path.resolve()), "receipt_sha256": sha256(path.read_bytes()),
                      "primary_rows": receipt["primary_rows"], "alternative_rows": receipt["alternative_rows"],
                      "training_eligibility": False}, indent=2))


if __name__ == "__main__":
    main()
