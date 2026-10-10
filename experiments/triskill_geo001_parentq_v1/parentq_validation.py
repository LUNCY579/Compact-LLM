"""Additional registration checks; payload/oracle arithmetic only, no model outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPOSED_T_PINS = {
    "experiments/composition_confirmation_v1/T_BANK_V3X.json":
        "173aa372466d5e3cc8efbebe4cc7e7ba09a8eb34de4598ce8f9d96a2b8e6557c",
    "experiments/composition_confirmation_v2/T_BANK_V4.json":
        "82f0a6423dd6b58b81319708e7fbda7c5880b9734742323ea11574ff354c93c0",
}
REQUIRED_DECISIONS = frozenset({"estimand_and_calibration", "gate_reliability", "representativeness_and_collision",
    "canary_threshold_and_route", "padding_failure_route", "exposed_row_policy", "q2_diagnostics_and_ceiling",
    "retry_and_nondeterminism", "geo_supervision_amendment", "source_backup_and_hardware_slot"})


def operative_number_key(benchmark, family, payload):
    """Touched numerical inputs and their outputs, preserving operative roles.

    Account/register names, tick and untouched values cannot hide reused arithmetic.
    The conditional includes both comparison operands and the chosen branch.
    """
    before, event = payload["before"], payload["event"]
    kind = event["kind"]
    after = benchmark.transition_oracle(family, payload)
    if family == "ledger_replay":
        account = event["account"]
        if kind in ("scale", "add"):
            return (family, kind, event["status"], before[account], event["amount"], after[account])
        other = event["other"]
        return (family, kind, event["status"], before[account], before[other], event["amount"],
                after[account], after[other])
    if family == "register_machine":
        register = event["register"]
        if kind == "add":
            return (family, kind, before[register], event["value"], after[register])
        if kind == "swap":
            other = event["other"]
            return (family, kind, before[register], before[other], after[register], after[other])
        if kind == "mix":
            return (family, kind, before[register], before[event["other"]], event["value"], after[register])
        return (family, kind, before[register], before[event["other"]], event["threshold"],
                event["value"], before[event["other"]] > event["threshold"], after[register])
    raise ValueError(f"unsupported operative key family: {family}")


def pinned_exposed_payloads(repo: Path, benchmark, delta_path: Path, delta_pin: str):
    for relative, digest in EXPOSED_T_PINS.items():
        data = (repo / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"STOP_EXPOSED_T_BANK_DIGEST_MISMATCH:{relative}")
        for row in json.loads(data)["rows"]:
            if row["family"] not in benchmark.T_SUPPORT:
                continue
            content = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else row["prompt"]
            yield relative, row["family"], json.loads(content)
    data = delta_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != delta_pin:
        raise ValueError("STOP_DELTA_T_TRAINING_DIGEST_MISMATCH")
    for line in data.decode("utf-8").splitlines():
        row = json.loads(line)
        if row["family"] in benchmark.T_SUPPORT:
            content = next(m["content"] for m in row["prompt"] if m["role"] == "user")
            yield "delta_T_training", row["family"], json.loads(content)


def operative_collisions(benchmark, t_rows, geo_rows, exposed_payloads):
    lookup = {}
    for row in t_rows:
        key = operative_number_key(benchmark, row.metadata["transition_family"], json.loads(row.user))
        lookup.setdefault(key, []).append(row.row_id)
    hits = {row.row_id: [] for row in t_rows}
    within = [ids for ids in lookup.values() if len(ids) > 1]
    historical = [(row.row_id, row.metadata["transition_family"], json.loads(row.user))
                  for row in geo_rows if row.task_kind == "transition_parent"]
    scanned = 0
    for source, family, payload in (*historical, *exposed_payloads):
        scanned += 1
        for row_id in lookup.get(operative_number_key(benchmark, family, payload), ()):
            hits[row_id].append(source)
    return {"key": "role_preserving_touched_inputs_and_outputs_v1", "scanned_payloads": scanned,
            "hits": hits, "within_bank": within, "collision_free": not any(hits.values()) and not within,
            "policy": "DISCLOSED_DESIGN_BLOCKER_NO_AUTOMATIC_ROW_REPLACEMENT"}


def readiness(here: Path):
    data = json.loads((here / "READINESS_PARENTQ.json").read_bytes())
    ready = (data.get("status") == "READY_FOR_REVIEWED_PROPOSAL"
             and data.get("native_execution_authorized") is False
             and data.get("independent_review", {}).get("status") == "PASS"
             and set(data.get("decisions", {})) == REQUIRED_DECISIONS
             and all(d.get("status") == "RESOLVED" for d in data["decisions"].values()))
    return data, ready
