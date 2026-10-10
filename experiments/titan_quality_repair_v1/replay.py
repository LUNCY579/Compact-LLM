"""Verified historical transition replay; no generated predictions become targets."""
from __future__ import annotations
import json
from .schema import SCHEMA_VERSION, digest, validate_row, typed_equal
from .exclusion import T_TRAIN, T_TRAIN_SHA, arithmetic_key
from .custody import read_pinned


def replay_oracle(family, payload):
    before, e = dict(payload["before"]), payload["event"]
    if family == "ledger":
        if e["status"] != "posted":
            return before
        account, v = e["account"], e["amount"]
        if e["kind"] == "add":
            before[account] += v
        elif e["kind"] == "scale":
            before[account] *= v
        elif e["kind"] == "transfer":
            before[account] -= v
            before[e["other"]] += v
        else:
            raise ValueError("STOP_REPLAY_UNSUPPORTED_OPERATION")
        return before
    target, v = e["register"], e["value"]
    if e["kind"] == "add":
        before[target] = (before[target] + v) % 97
    elif e["kind"] == "conditional":
        before[target] = (before[target] + v if before[e["other"]] > e["threshold"] else before[target] - v) % 97
    else:
        raise ValueError("STOP_REPLAY_UNSUPPORTED_OPERATION")
    return before


def load_replay():
    data = read_pinned(T_TRAIN, T_TRAIN_SHA)
    rows, skipped = [], {}
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        raw = json.loads(line)
        family = {"ledger_replay": "ledger", "register_machine": "register"}.get(raw["family"])
        if family is None:
            skipped[raw["family"]] = skipped.get(raw["family"], 0) + 1
            continue
        if raw.get("source_split") != "train":
            raise ValueError("STOP_REPLAY_NONTRAIN_SOURCE")
        messages = raw["prompt"]
        if [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("STOP_REPLAY_PROMPT_ROLES")
        payload = json.loads(messages[1]["content"])
        if family == "register" and payload["event"]["kind"] not in {"add", "conditional"}:
            skipped[payload["event"]["kind"]] = skipped.get(payload["event"]["kind"], 0) + 1
            continue
        expected = json.loads(raw["reference"])
        if not typed_equal(replay_oracle(family, payload), expected):
            raise ValueError(f"STOP_REPLAY_GOLD_MISMATCH:{raw['id']}")
        # Independent algebraic invariant additionally checks each changed field.
        key = arithmetic_key(family, payload)
        payload = {**payload, "expected_kind": "object"}
        row = {"schema_version": SCHEMA_VERSION, "row_id": "replay:" + raw["id"],
               "world_id": "replay:" + raw["parent_id"], "pair_id": None, "split": "replay",
               "family": family, "variant_id": "canonical", "system": messages[0]["content"],
               "user": messages[1]["content"], "answerable": True, "expected": expected,
               "contract": {"kind": "json_object"}, "payload": payload, "operative_key": key,
               "support_expansion": False, "strata": {"event_kind": payload["event"]["kind"], "historical_replay": True},
               "replay": True, "reference": raw["reference"], "replay_target": raw["reference"],
               "provenance": {"source_path": str(T_TRAIN), "source_sha256": T_TRAIN_SHA,
                              "source_line": line_no, "source_id": raw["id"],
                              "target_type": "HUMAN_OR_DETERMINISTIC_EXISTING_TRAIN_GOLD", "gold_verified": True,
                              "status": "INDEPENDENT_ARITHMETIC_VERIFIED",
                              "verification": "INDEPENDENT_ARITHMETIC_RECOMPUTATION",
                              "source_split": "train"}}
        rows.append(validate_row(row))
    if not all(any(r["family"] == f for r in rows) for f in ("ledger", "register")):
        raise ValueError("STOP_REPLAY_FAMILY_UNAVAILABLE")
    return rows, {"source": str(T_TRAIN), "sha256": T_TRAIN_SHA, "eligible_rows": len(rows),
        "eligible_families": {f: sum(r["family"] == f for r in rows) for f in ("ledger", "register")},
        "skipped": skipped, "source_targets_recomputed": True,
        "limitation": "Replay covers historical transition training only; broader conversation targets retain pending human review and are not substituted."}
