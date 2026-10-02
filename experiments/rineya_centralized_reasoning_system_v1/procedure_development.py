"""Bounded supplied-procedure lifecycle, not Rineya proposal discovery.

Coordinator/scorer side only. Public workers receive case+restricted snapshot,
never latent gold, scorer feedback or revision history. Same-account filesystem
privileges remain shared; this is interface separation, not OS secrecy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

from .core import Limits, ProcedureSnapshot, canonical, digest, validate_graph
from .data import audit_splits, build_dataset
from .operations import OperationsError, OperationsStore
from .runner import binding, execute_case
from .scoring import score

PACKAGE = Path(__file__).resolve().parent
ACTOR = {"kind": "CODEX_ENGINEERING_AGENT", "id": "/root/runtime_audit"}
SELECTOR = {"kind": "DETERMINISTIC_POLICY", "id": "ALL_FIXED_DEV_CASES_CORRECT_NO_OPERATIONAL_FAILURE"}
VERIFIER = {"kind": "LATENT_ORACLE_PLUS_EXTERNAL_SCORER", "id": "data.oracle+scoring.score"}
CONTRACT_PATH = PACKAGE / "contracts/PROCEDURE_DEVELOPMENT_V1.json"


class DevelopmentError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supplied_candidates():
    base = {"id": "ownership_development", "origin": "CODEX_SUPPLIED",
            "proposer": dict(ACTOR), "applicability": "ownership_remove_event_v1", "proposal_evidence": None}
    first = {**base, "version": 1, "reviser": None, "parent_sha256": None,
             "graph": [{"id": "read", "op": "LOOKUP", "deps": [], "args": {}},
                       {"id": "premature", "op": "FILTER", "deps": ["read"], "args": {"until": 0, "modality": "ALL"}},
                       {"id": "answer", "op": "TRACE_STATE", "deps": ["premature"], "args": {}},
                       {"id": "check", "op": "CHECK", "deps": ["answer"], "args": {}}]}
    second = {**base, "version": 2, "reviser": dict(ACTOR), "parent_sha256": digest(first),
              "graph": [{"id": "read", "op": "LOOKUP", "deps": [], "args": {}},
                        {"id": "answer", "op": "TRACE_STATE", "deps": ["read"], "args": {}},
                        {"id": "check", "op": "CHECK", "deps": ["answer"], "args": {}}]}
    return [first, second]


def validate_candidate(candidate, *, parent=None, store=None):
    fields = {"id", "version", "origin", "proposer", "reviser", "parent_sha256", "graph", "applicability", "proposal_evidence"}
    if type(candidate) is not dict or set(candidate) != fields:
        raise DevelopmentError("candidate schema rejects executable/unknown fields")
    if candidate["origin"] in {"MODEL_PROPOSED", "MODEL_REVISED"}:
        evidence = candidate["proposal_evidence"]
        if store is None or type(evidence) is not dict or set(evidence) != {"proposal_receipt", "qualification_receipt"}:
            raise DevelopmentError("model origin requires identified actual native proposal and exact qualification evidence")
        # Hash verification is necessary but does not make a status dictionary a
        # native proposal permit. This CPU successor registers supplied graphs
        # only; a real native-proposal adapter needs its own reviewed contract.
        for ref in evidence.values():
            store.verify_receipt(ref)
        raise DevelopmentError("native proposal adapter unregistered; supplied CPU lifecycle cannot claim MODEL_PROPOSED")
    if candidate["origin"] != "CODEX_SUPPLIED" or candidate["proposal_evidence"] is not None:
        raise DevelopmentError("only explicitly supplied Codex candidates registered here")
    proposer = candidate["proposer"]
    if (type(proposer) is not dict or set(proposer) != {"kind", "id"}
            or proposer["kind"] != "CODEX_ENGINEERING_AGENT" or type(proposer["id"]) is not str or not proposer["id"]):
        raise DevelopmentError("identified Codex proposer required")
    if type(candidate["version"]) is not int or candidate["version"] < 1:
        raise DevelopmentError("positive candidate version required")
    if parent is None:
        if candidate["version"] != 1 or candidate["parent_sha256"] is not None or candidate["reviser"] is not None:
            raise DevelopmentError("initial proposal cannot invent revision ancestry")
    else:
        if (candidate["parent_sha256"] != digest(parent) or candidate["id"] != parent["id"]
                or candidate["version"] != parent["version"] + 1 or candidate["reviser"] != ACTOR):
            raise DevelopmentError("revision must bind exact prior candidate and identified reviser")
    validate_graph(candidate["graph"], Limits())
    # Snapshot enforces valid IDs/types and a restricted operation graph.
    candidate_snapshot(candidate)
    return digest(candidate)


def candidate_snapshot(candidate):
    return ProcedureSnapshot([{"id": candidate["id"], "version": candidate["version"],
        "origin": candidate["origin"], "status": "FROZEN", "applicability": candidate["applicability"],
        "graph": candidate["graph"], "validation": "IMMUTABLE_DEV_TEST_INPUT_NOT_PROMOTION_OR_NATIVE_DISCOVERY"}])


def _history(store, key, index, previous, candidate, stage, references):
    event = {"schema_version": 1, "index": index, "previous_history_sha256": previous,
             "candidate_sha256": digest(candidate), "procedure_id": candidate["id"],
             "procedure_version": candidate["version"], "stage": stage,
             "proposer": candidate["proposer"], "reviser": candidate["reviser"],
             "selector": SELECTOR, "verifier": VERIFIER, "evidence": references,
             "origin": "CODEX_SUPPLIED", "native_discovery": False}
    return store.write_receipt("procedure-history", key + ":" + str(index), event)


def verify_nested_references(store, value, seen=None):
    """A cached parent receipt cannot hide missing/corrupt evidence children."""
    seen = set() if seen is None else seen
    if type(value) is dict and set(value) == {"path", "sha256"}:
        identity = (value["path"], value["sha256"])
        if identity not in seen:
            seen.add(identity)
            child = store.verify_receipt(value)
            verify_nested_references(store, child["payload"], seen)
    elif type(value) is dict:
        for item in value.values():
            verify_nested_references(store, item, seen)
    elif type(value) is list:
        for item in value:
            verify_nested_references(store, item, seen)


def run_development(store, *, epoch="CRS-PROCEDURE-DEV-001", count=32, seed=20261001):
    if count not in {16, 32} or type(count) is not int or type(seed) is not int or seed < 0:
        raise DevelopmentError("registered pair-complete DEV count16/32 and nonnegative seed required")
    candidates = supplied_candidates()
    if len(candidates) != 2:
        raise DevelopmentError("exactly two supplied candidates required")
    for index, candidate in enumerate(candidates):
        validate_candidate(candidate, parent=candidates[index - 1] if index else None, store=store)
    train, dev = build_dataset("TRAIN", 16, seed), build_dataset("DEV", count, seed)
    split = audit_splits(train, dev)
    source = binding()
    contract = json.loads(CONTRACT_PATH.read_bytes())
    registration = {"schema_version": 1, "study": "CRS-SUPPLIED-PROCEDURE-DEVELOPMENT-001",
        "binding": source, "contract_sha256": sha(CONTRACT_PATH), "count": count, "seed": seed,
        "datasets": [train["manifest"], dev["manifest"]], "split_audit": split,
        "candidates": candidates, "candidate_sha256": [digest(c) for c in candidates],
        "budget": {"candidate_count": 2, "dev_cases_per_candidate": count,
                   "maximum_public_worker_attempts": 2 * count, "per_worker_seconds": 15,
                   "coordinator_wall_seconds": contract["limits"]["coordinator_wall_seconds"], "native_seconds": 0},
        "exposure": "EXISTING_OPEN_DEV_CONSUMED_FOR_SUPPLIED_REVISION_NO_CONFIRMATION",
        "claim_scope": "Codex supplied restricted-graph engineering lifecycle; not autonomous Rineya discovery"}
    key = "procedure-development-" + digest(registration)
    job = store.register_job(key, registration, epoch)
    if job["state"] in {"SUCCEEDED", "FAILED"}:
        cached = store.verify_receipt(job["result"]["evidence"]["result_receipt"])["payload"]
        verify_nested_references(store, cached)
        return cached
    if job["state"] == "PENDING":
        store.ready(key)
    elif job["state"] != "READY":
        raise OperationsError("prior lifecycle outcome unknown; no automatic retry")
    owner = "PROCEDURE-CPU-" + uuid.uuid4().hex
    lease = store.acquire_lease(owner, 600, kind="coordinator")
    started = time.monotonic()
    history, trials, previous = [], [], None
    result_ref = None
    try:
        registration_ref = store.write_receipt("procedure-registration", key, registration)
        for index, candidate in enumerate(candidates):
            if time.monotonic() - started > contract["limits"]["coordinator_wall_seconds"]:
                raise DevelopmentError("controller wall ceiling; partial receipts retained")
            proposal_ref = store.write_receipt("procedure-proposal", key + ":" + str(index + 1), candidate)
            stage = "SUPPLIED_PROPOSAL" if index == 0 else "CODEX_SUPPLIED_REVISION"
            refs = {"proposal_receipt": proposal_ref, "registration_receipt": registration_ref}
            if index:
                refs["prior_counterexamples"] = trials[-1]["counterexample_receipt"]
            event = _history(store, key, len(history), previous, candidate, stage, refs)
            history.append(event); previous = event["sha256"]
            snapshot = candidate_snapshot(candidate)
            predictions, raw, case_references = {}, [], []
            for case in dev["cases"]:
                if time.monotonic() - started > contract["limits"]["coordinator_wall_seconds"]:
                    raise DevelopmentError("controller wall ceiling; partial cases retained")
                case_key = key + ":" + digest(candidate) + ":" + case["id"]
                output = execute_case(store, case_key, case, "C", snapshot, epoch)
                raw.append(output)
                case_references.append(store.get_job(case_key)["result"]["evidence"]["case_receipt"])
                predictions[case["id"]] = output.get("answer") if output.get("error") is None else {"error": output.get("error")}
                store.heartbeat_lease(owner, lease["fence"], 600, kind="coordinator")
            grades = score(predictions, dev["gold"])
            counterexamples = [{"case": case, "public_case_sha256": digest(case), "gold": dev["gold"][case["id"]],
                               "prediction": predictions[case["id"]], "grade": grades["rows"][case["id"]],
                               "purpose": "EVALUATOR_SIDE_DEV_COUNTEREXAMPLE_NOT_RUNTIME_INPUT"}
                              for case in dev["cases"] if not grades["rows"][case["id"]]["semantic_exact"]]
            counter_ref = store.write_receipt("procedure-counterexamples", key + ":" + str(index + 1),
                       {"candidate_sha256": digest(candidate), "counterexamples": counterexamples,
                        "scheduled_cases": count, "all_scheduled_accounted": True,
                        "scorer_sha256": source["scorer_sha256"], "verifier": VERIFIER})
            unknown = sum(o.get("stop_reason") == "RECONCILIATION_REQUIRED" for o in raw)
            known_seconds = sum(o.get("worker_wall_seconds", 0) for o in raw)
            valid_semantic_failures = sum(not row["semantic_exact"] and row["format_valid"] and not row["operational_error"]
                                          for row in grades["rows"].values())
            accepted = grades["counts"]["semantic_exact"] == count and grades["counts"]["operational_error"] == 0
            trial = {"candidate_sha256": digest(candidate), "version": candidate["version"],
                     "status": "DEV_VALIDATED_CPU_SUPPLIED" if accepted else "REJECTED_DEV" if valid_semantic_failures else "QUARANTINED_OPERATIONAL",
                     "acceptance": accepted, "scores": grades, "counterexample_receipt": counter_ref,
                     "valid_semantic_counterexamples": valid_semantic_failures,
                     "public_worker_receipts": case_references, "test_snapshot_sha256": snapshot.sha256,
                     "test_snapshot_status": "IMMUTABLE_INPUT_NOT_ACCEPTED_LIBRARY",
                     "cost": {"scheduled_attempts": count, "observed_case_receipts": len(raw),
                         "known_cpu_worker_seconds_lower_bound": known_seconds,
                         "unknown_cpu_attempts": unknown, "cpu_worker_seconds_total": None if unknown else known_seconds,
                         "primitive_operations": sum(o.get("cost", {}).get("operations", 0) for o in raw),
                         "source_bytes_scanned": sum(o.get("cost", {}).get("source_bytes_scanned", 0) for o in raw),
                         "source_bytes_verification_scanned": sum(o.get("cost", {}).get("source_bytes_verification_scanned", 0) for o in raw),
                         "model_calls": 0, "training_updates": 0, "native_worker_seconds": 0}}
            trial_ref = store.write_receipt("procedure-trial", key + ":" + str(index + 1), trial)
            event = _history(store, key, len(history), previous, candidate, trial["status"],
                             {"trial_receipt": trial_ref, "counterexamples": counter_ref})
            history.append(event); previous = event["sha256"]
            trials.append({**trial, "trial_receipt": trial_ref})
        demonstrated = trials[0]["valid_semantic_counterexamples"] > 0 and trials[1]["acceptance"]
        frozen_ref = None
        if demonstrated:
            entry = candidate_snapshot(candidates[1]).entries()[0]
            entry["validation"] = "CPU_DEV_VALIDATED_CODEX_SUPPLIED_NOT_RINEYA_DISCOVERY_OR_EFFICACY"
            frozen = ProcedureSnapshot([entry])
            frozen_ref = store.write_receipt("procedure-frozen-snapshot", key,
                {"entries": frozen.entries(), "snapshot_sha256": frozen.sha256,
                 "accepted_candidate_sha256": digest(candidates[1]), "validation_trial": trials[1]["trial_receipt"],
                 "proposer": ACTOR, "reviser": ACTOR, "selector": SELECTOR, "verifier": VERIFIER,
                 "evaluation_memory_writes_allowed": False, "validation_scope": "CONSUMED_DEV_CPU_FIXTURES_ONLY"})
            event = _history(store, key, len(history), previous, candidates[1], "FROZEN_CPU_SUPPLIED_SNAPSHOT",
                             {"snapshot_receipt": frozen_ref})
            history.append(event); previous = event["sha256"]
        result = {"job_key": key, "epoch": epoch, "status": "CPU_SUPPLIED_LIFECYCLE_COMPLETED" if demonstrated else "INCONCLUSIVE_LIFECYCLE",
                  "registration_receipt": registration_ref, "binding": source,
                  "trials": trials, "immutable_history_receipts": history, "history_tip_sha256": previous,
                  "frozen_snapshot_receipt": frozen_ref, "planned_cases": 2 * count,
                  "accounted_case_receipts": sum(len(t["public_worker_receipts"]) for t in trials),
                  "cost": {"known_cpu_worker_seconds_lower_bound": sum(t["cost"]["known_cpu_worker_seconds_lower_bound"] for t in trials),
                           "unknown_cpu_attempts": sum(t["cost"]["unknown_cpu_attempts"] for t in trials),
                           "primitive_operations": sum(t["cost"]["primitive_operations"] for t in trials),
                           "model_calls": 0, "model_loads": 0, "training_updates": 0, "native_worker_seconds": 0},
                  "stop_reason": "REGISTERED_TWO_CANDIDATE_DEV_BUDGET_COMPLETED",
                  "claims": {"Rineya_procedure_discovery": False, "system_efficacy": False, "weight_learning": False,
                             "transfer_confirmed": False, "retention_confirmed": False, "fresh_confirmation_used": False},
                  "limitations": ["Both graphs and revision supplied by Codex before outcomes.",
                                  "Revised graph reuses the already supplied strong TRACE_STATE primitive.",
                                  "DEV is exposed development evidence, not fresh confirmation.",
                                  "Process boundary is public-input separation, not same-account OS access isolation."]}
        result_ref = store.write_receipt("procedure-development-result", key, result)
        store.finish_cpu(key, "SUCCEEDED" if demonstrated else "FAILED", {"result_receipt": result_ref})
        return result
    except BaseException as exc:
        # A resumed study may fail again for a different reason. The immutable
        # study key is stable; each actual coordinator attempt has its own ID.
        try:
            store.write_receipt("procedure-development-failure", key + ":attempt:" + owner,
                               {"attempt_id": owner, "study_key": key,
                                "exception": type(exc).__name__ + ": " + str(exc), "history": history,
                                "partial_case_receipts_retained": True, "native_seconds": 0})
        except BaseException as publication_error:
            exc.add_note("Failure receipt publication also failed: " + type(publication_error).__name__)
        raise
    finally:
        store.release_lease(owner, lease["fence"], kind="coordinator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["accept"])
    parser.add_argument("--cases", type=int, choices=[16, 32], default=32)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, default=PACKAGE / "outputs/procedure_development_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PACKAGE / "outputs"):
        raise DevelopmentError("development output must be an ignored CRS outputs child")
    store = OperationsStore(output / "operations.sqlite3", output / "receipts")
    try:
        result = run_development(store, count=args.cases, seed=args.seed)
        print(json.dumps({k: result[k] for k in ("status", "job_key", "planned_cases", "accounted_case_receipts", "cost", "frozen_snapshot_receipt", "claims")}, indent=2))
        return 0 if result["status"] == "CPU_SUPPLIED_LIFECYCLE_COMPLETED" else 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
