"""Fresh, oracle-verified candidate data for one prospectively fixed tiny screen.

This module writes nothing and never calls a model.  ``admitted_for`` names the
proposed destination split required by the existing row schema; it is not an
admission grant.  The returned manifest remains CANDIDATE_NOT_ADMITTED until an
independent reviewer admits the exact bytes and checks native tokenizer budgets.
Teacher-gate rows are returned separately and must never enter update/eval rows.
"""
from __future__ import annotations

import hashlib
import json
import random

from .data import FIELDS, score


STATUS = "CANDIDATE_NOT_ADMITTED"
MODULUS = 17
TASK = "modular_state"
FROZEN_GENERATION_SEEDS = {
    "train": 91017,
    "iid_acquisition": 91029,
    "harder_transfer": 91043,
    "retention_arithmetic": 91059,
    "retention_spatial": 91071,
    "teacher_gate": 91083,
}
COUNTS = {"train": 16, "iid_acquisition": 12, "harder_transfer": 12,
          "retention_arithmetic": 8, "retention_spatial": 8, "teacher_gate": 8}
COHORT_ROLES = {
    "train": "student_updates_only",
    "iid_acquisition": "unprivileged_iid_acquisition_evaluation",
    "harder_transfer": "unprivileged_harder_transfer_evaluation_separate_endpoint",
    "retention_arithmetic": "independent_unprivileged_retention_replay",
    "retention_spatial": "independent_unprivileged_retention_replay",
    "teacher_gate": "qualification_only_never_student_updates_or_evaluation",
}


def _validate_affine_input(initial, operations, modulus):
    if type(initial) is not int or type(modulus) is not int or modulus < 2:
        raise ValueError("Integer initial state and modulus >= 2 required")
    if not operations or any(len(pair) != 2 or any(type(v) is not int for v in pair)
                             for pair in operations):
        raise ValueError("Nonempty integer affine operations required")


def sequential_states(initial, operations, modulus=MODULUS):
    """Oracle A: evaluate each update directly, reducing after every operation."""
    _validate_affine_input(initial, operations, modulus)
    states = [initial % modulus]
    for multiplier, offset in operations:
        states.append((multiplier * states[-1] + offset) % modulus)
    return tuple(states)


def solve_composed_affine(initial, operations, modulus=MODULUS):
    """Oracle B: compose affine maps, then apply the single map to the input.

    This algorithm does not use the sequential states or the first oracle's
    answer.  Coefficients are reduced during composition, before application.
    """
    _validate_affine_input(initial, operations, modulus)
    slope, intercept = 1, 0
    for multiplier, offset in operations:
        slope = (multiplier * slope) % modulus
        intercept = (multiplier * intercept + offset) % modulus
    return (slope * initial + intercept) % modulus


def _affine_semantic_digest(initial, operations, modulus=MODULUS):
    # Equal composed maps at equal normalized inputs are also excluded across
    # cohorts, even when their visible operation sequences differ in length.
    intercept = solve_composed_affine(0, operations, modulus)
    slope = (solve_composed_affine(1, operations, modulus) - intercept) % modulus
    return _sha({"kind": "composed_modular_affine", "modulus": modulus,
                 "initial_residue": initial % modulus,
                 "slope": slope, "intercept": intercept})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def validate_disjoint_cohorts(cohorts, case_records):
    """Check parent and underlying-input custody across all candidate roles.

    A parent is the digest of the mathematical input, without cohort or row
    labels.  Merely relabeling an identical input therefore cannot pass.
    """
    records = {r["id"]: r for r in case_records}
    if len(records) != len(case_records):
        raise ValueError("Duplicate candidate case record")
    row_ids, parents, signatures, affine_semantics = set(), set(), set(), set()
    for cohort, rows in cohorts.items():
        for row in rows:
            if set(row) != FIELDS:
                raise ValueError("Candidate row has incompatible schema")
            if row["id"] in row_ids or row["parent_id"] in parents:
                raise ValueError("Candidate row/parent reused within or across cohorts")
            record = records.get(row["id"])
            if record is None or record["cohort"] != cohort or record["parent_id"] != row["parent_id"]:
                raise ValueError("Candidate case custody changed")
            signature = _sha(record["canonical_input"])
            if signature != record["input_sha256"] or signature != row["parent_id"]:
                raise ValueError("Candidate parent does not identify its underlying input")
            if signature in signatures:
                raise ValueError("Canonical underlying input reused across cohorts")
            case = record["canonical_input"]
            if case["kind"] == "signed_modular_affine":
                semantic_digest = _affine_semantic_digest(case["initial"], case["operations"], case["modulus"])
                if semantic_digest != record["affine_semantic_sha256"]:
                    raise ValueError("Candidate affine equivalence custody changed")
                if semantic_digest in affine_semantics:
                    raise ValueError("Equivalent normalized input/affine map reused across cohorts")
                affine_semantics.add(semantic_digest)
            if not score(row["reference"], row["answer"], row["scorer"]):
                raise ValueError("Candidate reference fails its exact scorer")
            row_ids.add(row["id"])
            parents.add(row["parent_id"])
            signatures.add(signature)
    if row_ids != records.keys():
        raise ValueError("Missing/extra candidate custody record")
    return {"unique_rows": len(row_ids), "unique_parents": len(parents),
            "unique_underlying_inputs": len(signatures),
            "unique_affine_input_map_pairs": len(affine_semantics), "all_disjoint": True}


def build_candidate_bank():
    """Return exactly 56 screen rows plus 8 disjoint qualification-only rows.

    IID and harder rows deliberately share TASK for the existing loader.  Their
    ID lists must remain separate in prospective analysis; its legacy pooled
    acquisition endpoint is not this screen's primary acquisition endpoint.
    """
    cohorts = {name: [] for name in COUNTS}
    case_records, used_inputs, used_affine_semantics = [], set(), set()

    def add(cohort, canonical_input, task, prompt, reference, answer, scorer,
            oracle_agreement=True):
        parent = _sha(canonical_input)
        if parent in used_inputs:
            return False
        affine_digest = None
        if canonical_input["kind"] == "signed_modular_affine":
            affine_digest = _affine_semantic_digest(canonical_input["initial"], canonical_input["operations"],
                                                    canonical_input["modulus"])
            if affine_digest in used_affine_semantics:
                return False
        if not oracle_agreement or not score(reference, answer, scorer):
            raise ValueError("Independent candidate solution verification failed")
        used_inputs.add(parent)
        if affine_digest is not None:
            used_affine_semantics.add(affine_digest)
        split = "train" if cohort in {"train", "teacher_gate"} else "dev"
        purpose = "retention" if cohort.startswith("retention_") else "acquisition"
        row_id = f"{cohort}-{parent}"
        cohorts[cohort].append({
            "id": row_id, "parent_id": parent, "task": task, "split": split,
            "purpose": purpose, "source": "DSD-001 fresh deterministic candidate; review pending",
            "admitted_for": split, "prompt": prompt, "reference": reference,
            "answer": answer, "scorer": scorer,
        })
        case_records.append({"id": row_id, "parent_id": parent, "cohort": cohort,
                             "input_sha256": parent, "canonical_input": canonical_input,
                             "affine_semantic_sha256": affine_digest,
                             "oracle_agreement": oracle_agreement,
                             "reference_verified": True})
        return True

    for cohort in ("train", "iid_acquisition", "harder_transfer", "teacher_gate"):
        rng = random.Random(FROZEN_GENERATION_SEEDS[cohort])
        while len(cohorts[cohort]) < COUNTS[cohort]:
            index = len(cohorts[cohort])
            length = 5 + index % 3 if cohort == "harder_transfer" else 2
            initial = rng.randrange(-16, 17)
            operations = tuple((rng.choice((-3, -2, -1, 1, 2, 3)), rng.randrange(-8, 9))
                               for _ in range(length))
            inputs = {"kind": "signed_modular_affine", "modulus": MODULUS,
                      "initial": initial % MODULUS, "operations": [list(pair) for pair in operations]}
            states = sequential_states(initial, operations)
            independent = solve_composed_affine(initial, operations)
            op_text = "; ".join(f"({a}, {b})" for a, b in operations)
            prompt = (
                f"Start with s = {initial}. Apply these ordered (a, b) operations: {op_text}. "
                f"For each operation set s = (a*s + b) mod {MODULUS}, where mod means the "
                f"unique residue from 0 through {MODULUS - 1}. Give the final s as a boxed integer."
            )
            # A compact verified solution, not only an unverified final answer.
            reference = " -> ".join(map(str, states)) + f"; \\boxed{{{states[-1]}}}"
            add(cohort, inputs, TASK, prompt, reference, str(states[-1]), "boxed_int",
                oracle_agreement=states[-1] == independent)

    rng = random.Random(FROZEN_GENERATION_SEEDS["retention_arithmetic"])
    while len(cohorts["retention_arithmetic"]) < COUNTS["retention_arithmetic"]:
        left, right = rng.randrange(100, 2000), rng.randrange(1, 200)
        inputs = {"kind": "integer_addition", "left": left, "right": right}
        answer = left + right
        # Independently sum decimal place contributions; no model is consulted.
        place_answer = sum(int(digit) * 10**i for value in (left, right)
                           for i, digit in enumerate(str(value)[::-1]))
        add("retention_arithmetic", inputs, "retention_arithmetic",
            f"Compute {left} + {right}. Give a boxed integer.",
            f"{left}+{right}={answer}; \\boxed{{{answer}}}", str(answer), "boxed_int",
            oracle_agreement=answer == place_answer)

    rng = random.Random(FROZEN_GENERATION_SEEDS["retention_spatial"])
    directions = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
    while len(cohorts["retention_spatial"]) < COUNTS["retention_spatial"]:
        anchor = (rng.randrange(-9, 10), rng.randrange(-9, 10))
        movements = tuple((rng.choice(tuple(directions)), rng.randrange(1, 5))
                          for _ in range(rng.randrange(1, 4)))
        x, y = anchor
        for direction, distance in movements:
            vx, vy = directions[direction]
            x, y = x + vx * distance, y + vy * distance
        independent = (anchor[0] + sum(directions[d][0] * n for d, n in movements),
                       anchor[1] + sum(directions[d][1] * n for d, n in movements))
        inputs = {"kind": "ordinary_spatial", "anchor": list(anchor),
                  "movements": [list(move) for move in movements]}
        prompt = f"A marker starts at {anchor}. " + " Then ".join(
            f"move {n} cells {d}." for d, n in movements)
        prompt += " Give its final coordinates as a boxed integer pair."
        answer = str((x, y))
        add("retention_spatial", inputs, "retention_spatial", prompt,
            f"\\boxed{{{answer}}}", answer, "boxed_pair", oracle_agreement=(x, y) == independent)

    audit = validate_disjoint_cohorts(cohorts, case_records)
    audit.update(status=STATUS, deterministic_without_model_filtering=True,
                 modular_oracles=["sequential_reduction", "composed_affine_map"],
                 canonical_initial_states_normalized=True,
                 equivalent_initial_state_composed_map_pairs_excluded=True,
                 all_references_verified=True, native_token_budgets_verified=False,
                 case_records=case_records)
    manifest = {
        "status": STATUS, "admission_granted": False, "launch_authorized": False,
        "frozen_generation_seeds": dict(FROZEN_GENERATION_SEEDS),
        "paired_row_order_seeds": [17, 29], "screen_row_count": 56,
        "teacher_gate_row_count": 8, "modulus": MODULUS,
        "harder_intervention": "operation_count_only: 2 versus 5/6/7; same generator and explicit rule",
        "cohorts": {name: {"role": COHORT_ROLES[name], "count": len(rows),
                           "ids": [r["id"] for r in rows],
                           "parents": [r["parent_id"] for r in rows],
                           "operation_counts": ([5, 6, 7] if name == "harder_transfer"
                                                else [2]) if name in {
                                                    "train", "iid_acquisition", "harder_transfer", "teacher_gate"}
                                                else None}
                    for name, rows in cohorts.items()},
        "teacher_gate_excluded_from_updates_and_student_dev": True,
        "retention_is_evaluation_replay_only": True,
        "legacy_pooled_acquisition_endpoint_is_primary": False,
        "limitations": ["fresh small synthetic cohorts", "two retention domains only",
                        "harder compositional transfer within one explicit task",
                        "shared task and algorithm families are intentional; no broad family independence",
                        "candidate review and exact native tokenizer limits still pending",
                        "oracle/fixture success is not Rineya capability evidence"],
    }
    screen_names = [name for name in COUNTS if name != "teacher_gate"]
    return {"status": STATUS,
            "screen_rows": [row for name in screen_names for row in cohorts[name]],
            "teacher_gate_rows": list(cohorts["teacher_gate"]),
            "cohort_manifest": manifest, "audit": audit}
