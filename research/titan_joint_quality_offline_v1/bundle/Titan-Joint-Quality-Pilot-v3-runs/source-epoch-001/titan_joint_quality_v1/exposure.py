"""Offline, whole-row exposure planning; this module grants no model execution."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import random
import re

from .common import CONTRACT, DOMAINS, DOMAIN_ROLE, PARENT_SHA, ROLE_SHARES, digest, file_sha

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = ("case_id", "parent_id", "family_id", "domain", "role", "messages",
           "canonical_answer", "target", "requested_format")
_ENCODED_FIELDS = frozenset(("source_record", "ids", "response_start", "response_end",
    "eos_position", "eos_token_id", "label_mask", "processed_tokens", "response_eos_tokens",
    "row_sha256", "tokenization_sha256", "candidate_ordinal"))


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a strict integer >= {minimum}")
    return value


def _validate_row(row):
    if not isinstance(row, dict) or any(name not in row for name in _FIELDS):
        raise ValueError("row is missing required schema fields")
    for name in ("case_id", "parent_id", "family_id", "requested_format"):
        if not isinstance(row[name], str) or not row[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if row["domain"] not in DOMAINS or row["role"] != DOMAIN_ROLE[row["domain"]]:
        raise ValueError("domain and role do not match the fixed contract")
    if not isinstance(row["target"], str) or not row["target"].strip():
        raise ValueError("target must be nonempty verified text")
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in
                {"system", "user", "assistant"} or
                not isinstance(message.get("content"), str) or not message["content"]):
            raise ValueError("message role/content is missing or invalid")
    if messages[-1]["role"] != "user":
        raise ValueError("the final prompt message must be a user message")
    # Also checks that all provenance/answer fields can be bound as strict JSON.
    digest(row)


def _validate_encoding(value):
    if not isinstance(value, dict):
        raise ValueError("encode_case must return an encoding dictionary")
    ids = value.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("tokenization is missing")
    for token in ids:
        _integer(token, "token ID")
    if len(ids) > CONTRACT["max_row_tokens"]:
        raise ValueError("whole row exceeds max_row_tokens; truncation is forbidden")
    start = _integer(value.get("response_start"), "response_start", 1)
    end = _integer(value.get("response_end"), "response_end", 1)
    eos_position = _integer(value.get("eos_position"), "eos_position", 1)
    eos = _integer(value.get("eos_token_id"), "eos_token_id")
    if len(ids) < 4 or not start + 2 <= end < len(ids):
        raise ValueError("prompt, nonempty response, EOS and masked trailing newline are required")
    if eos_position != end - 1 or ids[eos_position] != eos or eos in ids[start:eos_position]:
        raise ValueError("response must have exactly one supervised EOS at response_end-1")
    if value.get("truncated", False) is not False:
        raise ValueError("truncated encodings are forbidden")
    if value.get("padded", False) is not False:
        raise ValueError("padded encodings are forbidden")
    return list(ids), start, end, eos_position, eos


def encode_rows(rows, tokenizer_adapter):
    """Encode all candidates and bind response+EOS labels without dropping failures.

    ``encode_case(messages, target)`` must independently establish its response
    boundary. On any refusal, ``encoded_rows`` is empty so a caller cannot silently
    turn the valid prefix into a different training population.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        return {"status": "BLOCKED_TOKENIZATION", "encoded_rows": [],
                "rejections": [{"case_id": None, "reason": "nonempty rows required"}],
                "native_authorized": False, "training_authorized": False}
    encoded, rejected, seen = [], [], set()
    method = getattr(tokenizer_adapter, "encode_case", None)
    for ordinal, row in enumerate(rows):
        case_id = row.get("case_id") if isinstance(row, dict) else None
        try:
            _validate_row(row)
            if _ENCODED_FIELDS.intersection(row):
                raise ValueError("original source fields collide with encoded bookkeeping")
            if row["case_id"] in seen:
                raise ValueError("duplicate case_id")
            seen.add(row["case_id"])
            if not callable(method):
                raise ValueError("a tokenizer_adapter.encode_case implementation is required")
            # Freeze the COMPLETE original record, including future metadata.
            # Keep an independent projection so mutation cannot change both views.
            source_record = deepcopy(row)
            ids, start, end, eos_position, eos = _validate_encoding(
                method(deepcopy(source_record["messages"]), source_record["target"]))
            labels = [start <= position < end for position in range(len(ids))]
            encoded.append({**deepcopy(source_record), "source_record": source_record,
                            "ids": ids, "response_start": start,
                            "response_end": end, "eos_position": eos_position,
                            "eos_token_id": eos, "label_mask": labels,
                            "processed_tokens": len(ids),
                            "response_eos_tokens": end - start,
                            "row_sha256": digest(source_record),
                            "tokenization_sha256": digest({"ids": ids,
                                "response_start": start, "response_end": end,
                                "eos_position": eos_position, "eos_token_id": eos}),
                            "candidate_ordinal": ordinal})
        except Exception as error:
            rejected.append({"case_id": case_id, "candidate_ordinal": ordinal,
                             "reason": str(error)})
    summary = {"candidate_rows": len(rows), "validated_rows": len(encoded),
               "rejected_rows": len(rejected),
               "validated_case_ids": [row["case_id"] for row in encoded],
               "validated_encoding_sha256": digest(encoded),
               "truncated_rows": 0, "padded_rows": 0}
    return {"status": "BLOCKED_TOKENIZATION" if rejected else "ENCODED_OFFLINE_ONLY",
            "encoded_rows": [] if rejected else encoded, "summary": summary,
            "rejections": rejected, "native_authorized": False,
            "training_authorized": False, "capability_evidence": False}


def _check_encoded(row):
    _validate_row(row)
    original = row.get("source_record")
    if not isinstance(original, dict):
        raise ValueError("complete original source_record is missing")
    _validate_row(original)
    if _ENCODED_FIELDS.intersection(original):
        raise ValueError("original source fields collide with encoded bookkeeping")
    if row.get("row_sha256") != digest(original):
        raise ValueError("row content/provenance binding changed")
    if set(row) != set(original) | _ENCODED_FIELDS:
        raise ValueError("encoded source field membership changed")
    # Canonical digests distinguish strict JSON types (e.g. True from integer 1)
    # and bind nested trace/provenance fields, not just the minimal schema.
    projection = {name: row[name] for name in original}
    if digest(projection) != digest(original):
        raise ValueError("encoded projection differs from complete original source_record")
    ids, start, end, eos_position, eos = _validate_encoding(row)
    expected = [start <= position < end for position in range(len(ids))]
    if row.get("label_mask") != expected or any(type(x) is not bool for x in row["label_mask"]):
        raise ValueError("response+EOS label mask is missing or changed")
    if row.get("processed_tokens") != len(ids) or row.get("response_eos_tokens") != end - start:
        raise ValueError("token census differs from complete row encoding")
    if row.get("tokenization_sha256") != digest({"ids": ids, "response_start": start,
                                                "response_end": end, "eos_position": eos_position,
                                                "eos_token_id": eos}):
        raise ValueError("tokenization binding changed")


def _pack_groups(units):
    """Pack complete rows; a fully selected parent may cross update boundaries."""
    remaining = [row for unit in units for row in unit["rows"]]
    groups = []
    while remaining:
        reasoning = [unit for unit in remaining if unit["role"] == "reasoning"]
        if not reasoning:
            raise ValueError("remaining preservation units cannot form a group with R labels")
        anchor = reasoning[0]
        remaining.remove(anchor)
        group = [anchor]
        tokens = anchor["processed_tokens"]
        while True:
            non_r = [row for row in remaining if row["role"] != "reasoning"]
            # An R anchor is a scarce group prerequisite. While non-R rows
            # remain, use fitting non-R rows first; close this group if none
            # fit, rather than spending another anchor and starving them.
            candidates = non_r if non_r else remaining
            available = [row for row in candidates if tokens + row["processed_tokens"]
                         <= CONTRACT["max_update_tokens"]]
            if not available:
                break
            selected = available[0]
            remaining.remove(selected)
            group.append(selected)
            tokens += selected["processed_tokens"]
        rows = group
        denominator = sum(row["response_eos_tokens"] for row in rows)
        r_tokens = sum(row["response_eos_tokens"] for row in rows
                       if row["role"] == "reasoning")
        if denominator <= 0 or r_tokens <= 0:
            raise ValueError("every update must have a common positive denominator and R targets")
        groups.append({"group_index": len(groups), "rows": rows,
                       "case_ids": [row["case_id"] for row in rows],
                       "parent_ids": list(dict.fromkeys(row["parent_id"] for row in rows)),
                       "processed_tokens": tokens, "planned_denominator": denominator,
                       "reasoning_label_tokens": r_tokens,
                       "joint_label_tokens": denominator,
                       "reasoning_only_label_tokens": r_tokens})
    return groups


def build_exposure_plan(encoded_rows, source_bindings):
    """Select before outcomes and use exactly the same exposure in both arms.

    Source bindings require the exact parent, data and tokenizer SHA256 values.
    Paired rows sharing a parent are selected together; complete rows may cross
    update groups. All nonselected whole parents are recorded; infeasible
    mixture/packing stops rather than editing the bank.
    """
    receipt = {"status": "BLOCKED_EXPOSURE_PLAN", "native_authorized": False,
               "training_authorized": False, "source_objective_qualified": False,
               "exposure_source_sha256": file_sha(Path(__file__).resolve()),
               "common_source_sha256": file_sha(Path(__file__).resolve().with_name("common.py")),
               "selection_seed": CONTRACT["seed"], "rejections": [],
               "selected_case_ids": [], "groups": [], "unselected": []}
    try:
        if not isinstance(source_bindings, dict):
            raise ValueError("source_bindings dictionary required")
        for name in ("parent_sha256", "data_sha256", "tokenizer_sha256"):
            if not isinstance(source_bindings.get(name), str) or not _HASH.fullmatch(source_bindings[name]):
                raise ValueError(f"missing/invalid {name}")
        if source_bindings["parent_sha256"] != PARENT_SHA:
            raise ValueError("parent must be the exact pinned Titan-Sparse artifact")
        if isinstance(encoded_rows, dict):
            if encoded_rows.get("status") != "ENCODED_OFFLINE_ONLY" or encoded_rows.get("rejections"):
                raise ValueError("the complete tokenization receipt did not pass")
            encoded_rows = encoded_rows["encoded_rows"]
        if not isinstance(encoded_rows, (list, tuple)) or not encoded_rows:
            raise ValueError("complete nonempty encoded rows required")
        by_parent, case_ids = defaultdict(list), set()
        for row in encoded_rows:
            _check_encoded(row)
            if row["case_id"] in case_ids:
                raise ValueError("duplicate encoded case_id")
            case_ids.add(row["case_id"])
            by_parent[row["parent_id"]].append(row)
        queues = {role: [] for role in ROLE_SHARES}
        for parent_id, rows in sorted(by_parent.items()):
            roles = {row["role"] for row in rows}
            if len(roles) != 1:
                raise ValueError("a whole-parent selection unit must have one contract role")
            tokens = sum(row["processed_tokens"] for row in rows)
            role = next(iter(roles))
            queues[role].append({"parent_id": parent_id, "role": role,
                                 "processed_tokens": tokens,
                                 "rows": sorted(rows, key=lambda row: row["case_id"])})
        rng = random.Random(CONTRACT["seed"])
        for queue in queues.values():
            rng.shuffle(queue)
        counts = {role: 0 for role in ROLE_SHARES}
        selected, unselected, total = [], [], 0
        while any(queues.values()):
            roles = sorted((role for role in ROLE_SHARES if queues[role]),
                key=lambda role: (-(ROLE_SHARES[role] * CONTRACT["processed_tokens_cap_per_arm"]
                    - counts[role]) / ROLE_SHARES[role], list(ROLE_SHARES).index(role)))
            added = False
            for role in roles:
                unit = queues[role].pop(0)
                if total + unit["processed_tokens"] <= CONTRACT["processed_tokens_cap_per_arm"]:
                    selected.append(unit)
                    counts[role] += unit["processed_tokens"]
                    total += unit["processed_tokens"]
                    added = True
                    break
                unselected.append({"parent_id": unit["parent_id"],
                    "case_ids": [row["case_id"] for row in unit["rows"]],
                    "processed_tokens": unit["processed_tokens"],
                    "reason": "not selected: complete parent would exceed fixed token cap"})
            if not added and not any(queues.values()):
                break
        if total <= 0:
            raise ValueError("no whole rows fit the exposure budget")
        shares = {role: counts[role] / total for role in ROLE_SHARES}
        invalid = {role: shares[role] for role in shares
                   if abs(shares[role] - ROLE_SHARES[role]) > CONTRACT["mixture_tolerance"] + 1e-12}
        if invalid:
            receipt["observed_role_shares"] = shares
            raise ValueError(f"processed-token role mixture outside five-point tolerance: {invalid}")
        groups = _pack_groups(selected)
        ordered = [row for group in groups for row in group["rows"]]
        receipt.update({"status": "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN",
            "source_bindings": source_bindings, "source_bindings_sha256": digest(source_bindings),
            "candidate_encoded_rows_sha256": digest(encoded_rows),
            "candidate_rows": len(encoded_rows), "selected_rows": len(ordered),
            "selected_parent_count": len(selected),
            "selected_case_ids": [row["case_id"] for row in ordered], "groups": groups,
            "parent_sha256": source_bindings["parent_sha256"],
            "sequence_digest": digest([{"case_id": row["case_id"], "ids": row["ids"]} for row in ordered]),
            "reasoning_labels_digest": digest([{"case_id": row["case_id"], "ids": row["ids"],
                "label_mask": row["label_mask"]} for row in ordered if row["role"] == "reasoning"]),
            "unselected": unselected, "processed_tokens_per_arm": total,
            "processed_token_counts_by_role": counts, "processed_token_shares_by_role": shares,
            "planned_denominators": [group["planned_denominator"] for group in groups],
            "joint_label_tokens": sum(group["planned_denominator"] for group in groups),
            "reasoning_only_label_tokens": sum(group["reasoning_label_tokens"] for group in groups),
            "same_sequences_order_groups": True, "same_reasoning_targets": True,
            "control_extra_reasoning_rows": 0,
            "updates": len(groups), "processed_tokens": total,
            "denominator_rule": "all planned response+EOS labels in the common update group",
            "selection_rule": "seed-17 whole-parent budget selection before any model outcome",
            "truncation": False, "padding": False, "capability_evidence": False})
        receipt["plan_sha256"] = digest(receipt)
    except (ValueError, TypeError, KeyError) as error:
        receipt["rejections"].append({"reason": str(error)})
        receipt["source_bindings"] = source_bindings if isinstance(source_bindings, dict) else None
        receipt["groups"] = []
        receipt["selected_case_ids"] = []
    return receipt
