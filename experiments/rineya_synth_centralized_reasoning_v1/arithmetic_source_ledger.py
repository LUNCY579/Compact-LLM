"""Additive complete arithmetic candidate provenance; never model consumption.

Uses only the exact previously reviewed pure generator/policy. No arithmetic
target solver, scorer, central population, model, tokenizer or native API runs.
The selected parent population and its order are unchanged.
"""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import hashlib
import json
import random

ORIGINAL_SOURCE_COMMIT = "e0e00272a4ae8f9dea45712dc286bd97150ebe3d"
GUARD_SOURCE_SHA256 = "02025fecbddd0fa882e67501c19556aad01795b608a33821f14201757fdc25b9"
POLICY_FILE_SHA256 = "42ee9c26f0fe1ab42c368618a5c25385fe09dac0f5777bab92df5c34c5fa05b7"
POLICY_BODY_SHA256 = "b0351aa3e6ed9a4e099ac3180cd273403a6bbf4dc68e4694f0608ea5c3d85f16"
ORIGINAL_POPULATION_FILE_SHA256 = "2992454925eecfabda4a6046d1d59ef78e718c1639d58714aa1404f90888f5c3"


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reference(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "selected regular absolute source path")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _selected(ref, expected):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
            and type(ref["bytes"]) is int and 0 < ref["bytes"] <= 1024 ** 2
            and ref["sha256"] == expected, "exact externally pinned original source descriptor")
    path = Path(ref["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "selected regular absolute source path")
    raw = path.read_bytes()
    require(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"], "selected original source bytes changed")
    return raw


def load_guard(source_ref, policy_ref):
    raw = _selected(source_ref, GUARD_SOURCE_SHA256)
    _selected(policy_ref, POLICY_FILE_SHA256)
    value = ModuleType("selected_original_arithmetic_guard")
    value.__file__ = source_ref["path"]
    exec(compile(raw, value.__file__, "exec"), value.__dict__)
    value.POLICY = Path(policy_ref["path"])
    frozen = value.policy()
    require(frozen["sha256"] == POLICY_BODY_SHA256, "exact original policy value pin")
    return value, frozen


def _candidate_id(depth, index, raw_sha256):
    return digest({"kind": "SYNTH_ARITHMETIC_SOURCE_CANDIDATE001", "policy_sha256": POLICY_BODY_SHA256,
                   "depth": depth, "candidate_index": index, "raw_expression_sha256": raw_sha256})


def reconstruct(guard, frozen, source_ref, policy_ref):
    require(frozen == guard.policy() and frozen["sha256"] == POLICY_BODY_SHA256, "unchanged original policy before replay")
    rng = random.Random(frozen["seed"]); records = []; aggregates = []
    selected_ids = []; selected_parents = []
    # Store all finite SOURCE rows and structural FILTER facts before selection;
    # no reference targets, CPU solvers or model outcomes are consulted.
    for depth in frozen["depths"]:
        candidates = {}; per_depth = []; equal_count = duplicate_count = 0
        for index in range(frozen["selection"]["finite_candidates_per_depth"]):
            raw = {"op": "add", "args": [guard._tree(rng, depth - 1), guard._tree(rng, depth - 1)]}
            normalized = guard.normalize(raw); raw_sha = digest(raw); expression_sha = digest(normalized)
            candidate_id = _candidate_id(depth, index, raw_sha)
            row = {"kind": "SYNTH_ARITHMETIC_SOURCE_CANDIDATE001", "schema_version": 1,
                "candidate_ordinal": len(records) + len(per_depth) + 1, "candidate_id": candidate_id,
                "depth": depth, "candidate_index": index, "raw_expression": raw,
                "raw_expression_sha256": raw_sha, "canonical_expression": normalized,
                "canonical_expression_sha256": expression_sha, "policy_sha256": POLICY_BODY_SHA256,
                "canonical_parent_id": digest({"source_id": frozen["source_id"], "canonical_expression": normalized}),
                "selection_state": None, "filter_reason": None, "duplicate_of_candidate_id": None,
                "eligible_rank": None, "selected_parent": None,
                "consume": {"native_model_occurrences": 0, "native_training_occurrences": 0,
                            "future_occurrence_reference": None}}
            if normalized["args"][0] == normalized["args"][1]:
                equal_count += 1; row["selection_state"] = "FILTERED"
                row["filter_reason"] = "equal_root_operands_prevent_distinct_equivalent_layout"
            elif expression_sha in candidates:
                duplicate_count += 1; row["selection_state"] = "FILTERED"
                row["filter_reason"] = "duplicate_canonical_content"
                row["duplicate_of_candidate_id"] = candidates[expression_sha]["candidate_id"]
            else:
                candidates[expression_sha] = row
            per_depth.append(row)
        require(len(candidates) >= 32, "original finite pool cannot yield32 parents")
        for rank, expression_sha in enumerate(sorted(candidates), 1):
            row = candidates[expression_sha]; row["eligible_rank"] = rank
            if rank <= 32:
                row["selection_state"] = "SELECTED_PARENT"
                parent = {"kind": "SYNTH_ARITHMETIC_GUARD_PARENT001", "domain_id": "arithmetic", "domain_class": "task",
                    "source_id": frozen["source_id"], "parent_id": row["canonical_parent_id"],
                    "policy_sha256": POLICY_BODY_SHA256, "canonical_expression": deepcopy(row["canonical_expression"]),
                    "expression_sha256": expression_sha, "depth": depth, "candidate_index": row["candidate_index"]}
                row["selected_parent"] = parent; selected_parents.append(parent); selected_ids.append(row["candidate_id"])
            else: row["selection_state"] = "ELIGIBLE_UNUSED"
        records.extend(per_depth)
        aggregates.append({"depth": depth, "candidates": 128, "unique_eligible_canonical": len(candidates),
            "duplicate_canonical": duplicate_count, "equal_operand_layout": equal_count, "selected": 32,
            "output_or_answer_filter_performed": False})
    original_parents, original_aggregates = guard.allocate(frozen)
    require(selected_parents == original_parents and aggregates == original_aggregates,
            "all original selected parent/order and aggregate counts remain unchanged")
    counts = {state: sum(row["selection_state"] == state for row in records)
              for state in ("SELECTED_PARENT", "ELIGIBLE_UNUSED", "FILTERED")}
    require(len(records) == 512 and len({(row["depth"], row["candidate_index"]) for row in records}) == 512
            and counts["SELECTED_PARENT"] == 128 and sum(counts.values()) == 512, "complete SOURCE/FILTER/UNUSED closure")
    value = {"kind": "SYNTH_ARITHMETIC_COMPLETE_SOURCE_LEDGER001", "schema_version": 1,
        "domain_id": "arithmetic", "domain_class": "task", "source_id": "SYNTHARITHGUARD001",
        "original_source_commit": ORIGINAL_SOURCE_COMMIT, "guard_source": deepcopy(source_ref),
        "policy_file": deepcopy(policy_ref), "policy_sha256": POLICY_BODY_SHA256,
        "original_population_file_sha256": ORIGINAL_POPULATION_FILE_SHA256,
        "candidate_order_rule": "depth1..4 then original candidate_index0..127",
        "candidates": records, "candidate_order_sha256": digest([r["candidate_id"] for r in records]),
        "selection_order_rule": "original perdepth canonical expression SHA256 sorted cutoff32",
        "selected_candidate_ids": selected_ids, "selected_parents": selected_parents,
        "selected_parent_order_sha256": digest([p["parent_id"] for p in selected_parents]),
        "original_allocation_audit": aggregates, "counts": counts,
        "consume": {"native_model_occurrences": 0, "native_training_occurrences": 0,
                    "future_occurrence_reference": None, "CPU_solver_calls": 0},
        "CPU_candidate_reconstruction_performed": True, "M0_enrollment_performed": False,
        "native_scoring_authorized": False, "model_quality_established": False,
        "pending": ["Future actual model input/consume/exposure occurrence receipts",
                    "Available training/evaluation-lineage disjointness and M0 qualification"]}
    value["sha256"] = digest(value)
    return value


def build(source_ref, policy_ref):
    guard, frozen = load_guard(source_ref, policy_ref)
    return reconstruct(guard, frozen, source_ref, policy_ref)


def validate(ledger, source_ref, policy_ref):
    require(type(ledger) is dict and type(ledger.get("schema_version")) is int and ledger["schema_version"] == 1
            and ledger.get("sha256") == digest({k: v for k, v in ledger.items() if k != "sha256"}),
            "exact sealed complete candidate ledger")
    expected = build(source_ref, policy_ref)
    require(canonical(ledger) == canonical(expected), "complete original candidate/selection/consume ledger replay differs")
    return {"kind": "SYNTH_ARITHMETIC_SOURCE_LEDGER_CPU_VALIDATION001", "candidate_count": 512,
            "selected_parent_count": 128, "native_model_occurrences": 0,
            "original_selected_population_unchanged": True, "native_scoring_authorized": False}
