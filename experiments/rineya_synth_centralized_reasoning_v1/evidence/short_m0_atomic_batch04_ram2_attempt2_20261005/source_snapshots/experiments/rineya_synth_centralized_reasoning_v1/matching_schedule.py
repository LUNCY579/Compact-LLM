"""TRAIN-only integer allocation and independent exact CPU schedule checks.

The MILP proposes weights. The standard-library checker independently counts
actual arrays, certificates, exposures and strata; solver flags are not proof.
No student result, held-out feedback, framework, native worker or training call.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import random
import sys

from balanced_corpus import _chain, compile_balanced_records, validate_plan
from rs_contract import canonical, digest, isint, parse_answer, parse_visible, require, validate_spec
from rs_verifier import verify

VERSION = "synth-paired-schedule-1"
PACKAGE = Path(__file__).resolve().parent
ARRAY_FIELDS = {"prompt_ids", "completion_ids", "input_ids", "labels", "attention_mask", "prompt_tokens",
                "supervised_tokens", "total_tokens", "prompt_text_sha256", "full_template_sha256"}
POSITION_RESIDUAL = Fraction(1, 25)
POSITION_DIFFERENCE = Fraction(1, 50)


def _policy() -> dict:
    return json.loads((PACKAGE / "MATCHING_POLICY.json").read_text(encoding="utf-8"))


def freeze_enrollment(plan: dict, parents_per_stratum: int = 2) -> dict:
    """Choose shared parents by source hashes before any token measurement."""
    validate_plan(plan)
    require(isint(parents_per_stratum, 1, 8), "enrollment stratum count")
    chosen, strata = [], {}
    for skill in ("R", "S"):
        for stratum in range(1, 6):
            candidates = []
            for pid, parent in plan["parents"].items():
                spec = parent["spec"]
                if plan["assignments"][pid]["split"] != "train" or validate_spec(spec) != skill:
                    continue
                value = len(_chain(spec, spec["lookup"]["requested"][0])[0]) if skill == "R" else spec["selection"]["k"]
                if value == stratum:
                    candidates.append(pid)
            candidates.sort(key=lambda pid: digest({"seed": plan["seed"], "pool": pid}))
            require(len(candidates) >= parents_per_stratum, "insufficient TRAIN source stratum; no replacement")
            for pid in candidates[:parents_per_stratum]:
                chosen.append(pid)
                strata[pid] = {"skill": skill, "stratum": stratum}
    all_rows = compile_balanced_records(plan)
    record_ids = {arm: sorted(r["row_id"] for r in all_rows if r["parent_id"] in chosen and r["arm"] == arm)
                  for arm in ("M1", "M2")}
    enrollment = {"version": VERSION, "plan_sha256": plan["sha256"], "seed": plan["seed"],
                  "parents": chosen, "parent_strata": strata, "parents_per_stratum": parents_per_stratum,
                  "rule": "first_hash_ordered_TRAIN_parents_per_R_first_depth_and_S_k_stratum_before_tokenization",
                  "record_ids": record_ids, "policy_sha256": digest(_policy()), "training_admitted": False}
    enrollment["sha256"] = digest(enrollment)
    return enrollment


def _budget(decisions: dict) -> dict:
    require(type(decisions) is dict and type(decisions.get("budget")) is dict and
            type(decisions.get("adaptation")) is dict, "decision sections")
    b = decisions["budget"]
    names = ("ceiling_per_trained_arm_per_seed", "minimum_consumed_per_arm_per_seed",
             "maximum_between_arm_residual_tokens", "maximum_per_skill_budget_difference_from_half")
    require(all(isint(b.get(n), 0, 2**31 - 1) for n in names), "integer token budgets")
    require(0 < b[names[1]] <= b[names[0]], "token budget interval")
    accumulation = decisions["adaptation"].get("gradient_accumulation_sequences")
    require(isint(accumulation, 1, 1024), "accumulation count")
    return {**{n: b[n] for n in names}, "accumulation": accumulation}


def _array_check(array: dict) -> None:
    require(type(array) is dict and set(array) == ARRAY_FIELDS, "token array fields")
    for name in ("prompt_ids", "completion_ids", "input_ids"):
        require(type(array[name]) is list and bool(array[name]) and
                all(isint(v, 0, 2**31 - 1) for v in array[name]), "integer token IDs")
    prompt, completion = array["prompt_ids"], array["completion_ids"]
    require(completion[-1] == 151645 and 151645 not in completion[:-1], "completion EOS")
    require(array["input_ids"] == prompt + completion and
            type(array["labels"]) is list and all(type(v) is int for v in array["labels"]) and
            array["labels"] == [-100] * len(prompt) + completion and
            type(array["attention_mask"]) is list and all(type(v) is int for v in array["attention_mask"]) and
            array["attention_mask"] == [1] * len(array["input_ids"]), "position masks or padding")
    for key, value in (("prompt_tokens", len(prompt)), ("supervised_tokens", len(completion)),
                       ("total_tokens", len(prompt) + len(completion))):
        require(type(array[key]) is int and array[key] == value, "token count drift")
    require(array["total_tokens"] <= 512, "context limit; no truncation")
    for name in ("prompt_text_sha256", "full_template_sha256"):
        require(type(array[name]) is str and len(array[name]) == 64 and
                all(c in "0123456789abcdef" for c in array[name]), "rendered text hash")


def _pool(rows: list[dict], arrays: dict, enrollment: dict) -> tuple[dict, list[tuple[str, int]], dict]:
    require(type(enrollment) is dict and type(enrollment.get("sha256")) is str and
            enrollment["sha256"] == digest({k: v for k, v in enrollment.items() if k != "sha256"}) and
            enrollment.get("version") == VERSION and enrollment.get("policy_sha256") == digest(_policy()) and
            enrollment.get("training_admitted") is False, "frozen enrollment binding")
    require(type(rows) is list and type(arrays) is dict, "pool containers")
    expected_ids = set(enrollment["record_ids"]["M1"] + enrollment["record_ids"]["M2"])
    require(len(expected_ids) == len(enrollment["parents"]) * 32 and set(arrays) == expected_ids,
            "complete paired token pool required")
    pool, cells = {}, {}
    for row in rows:
        require(type(row) is dict and type(row.get("row_id")) is str, "row identity")
        if row["row_id"] not in expected_ids:
            continue
        require(row["row_id"] not in pool and row["row_id"] == digest({k: v for k, v in row.items() if k != "row_id"}), "row hash/duplicate")
        require(row["split"] == "train" and row["parent_id"] in enrollment["parents"] and
                row["arm"] in ("M1", "M2") and row["row_id"] in enrollment["record_ids"][row["arm"]], "TRAIN arm/parent isolation")
        spec = parse_visible(row["messages"])
        skill = validate_spec(spec)
        require(skill in ("R", "S") and row["skill"] == skill and
                verify(row["messages"], row["target"])["accepted"], "visible atomic certificate")
        require(isint(row["condition_index"], 0, 3) and isint(row["template"], 0, 3), "crossing cell")
        key = (row["parent_id"], row["condition_index"], row["template"], row["arm"])
        require(key not in cells, "duplicate crossing cell")
        cells[key] = row["row_id"]
        _array_check(arrays[row["row_id"]])
        pool[row["row_id"]] = row
    require(set(pool) == expected_ids, "missing registered source rows")
    quads = [(pid, ci) for pid in enrollment["parents"] for ci in range(4)]
    require(all((pid, ci, t, arm) in cells for pid, ci in quads for t in range(4) for arm in ("M1", "M2")), "full condition/render crossing")
    return pool, quads, cells


def _metrics(schedule_ids: dict, pool: dict, arrays: dict) -> dict:
    totals, exposures, strata, positions, expected_positions, slots = {}, {}, {}, {}, {}, {}
    for arm in ("M1", "M2"):
        counts = Counter(schedule_ids[arm])
        total = supervised = 0
        exposures[arm] = Counter()
        strata[arm] = {"S_k": Counter(), "R_first_depth": Counter()}
        positions[arm] = {"R": Counter(), "S": Counter()}
        expected_positions[arm] = {"R": defaultdict(Fraction), "S": defaultdict(Fraction)}
        slots[arm] = Counter()
        skill_tokens = Counter()
        for rid, count in counts.items():
            row, array = pool[rid], arrays[rid]
            require(row["arm"] == arm, "schedule arm crossing")
            spec = parse_visible(row["messages"])
            skill = validate_spec(spec)
            total += count * array["total_tokens"]
            supervised += count * array["supervised_tokens"]
            skill_tokens[skill] += count * array["total_tokens"]
            exposures[arm][row["parent_id"]] += count
            answer = parse_answer(row["target"], len(spec["initial"]))
            require(type(answer) is list, "only solved acquisition examples")
            slots[arm][skill] += count * len(answer)
            positions[arm][skill].update({j: count for j in answer})
            for j in range(len(spec["initial"])):
                expected_positions[arm][skill][j] += Fraction(count * len(answer), len(spec["initial"]))
            if skill == "S":
                strata[arm]["S_k"][spec["selection"]["k"]] += count
            else:
                depth = len(_chain(spec, spec["lookup"]["requested"][0])[0])
                strata[arm]["R_first_depth"][depth] += count
        totals[arm] = {"rows": len(schedule_ids[arm]), "nonpadding_tokens": total,
                       "supervised_tokens": supervised, "skill_tokens": dict(skill_tokens)}
    return {"totals": totals, "parent_exposures": {a: dict(c) for a, c in exposures.items()},
            "strata": {a: {k: {str(i): n for i, n in c.items()} for k, c in v.items()} for a, v in strata.items()},
            "answer_position_counts": {a: {s: {str(j): n for j, n in c.items()} for s, c in v.items()} for a, v in positions.items()},
            "answer_slots": {a: dict(c) for a, c in slots.items()},
            "position_expectations": {a: {s: {str(j): [n.numerator, n.denominator] for j, n in c.items()}
                                         for s, c in v.items()} for a, v in expected_positions.items()}}


def _check_metrics(metrics: dict, budget: dict) -> None:
    totals = metrics["totals"]
    require(metrics["parent_exposures"]["M1"] == metrics["parent_exposures"]["M2"], "parent exposure mismatch")
    require(metrics["strata"]["M1"] == metrics["strata"]["M2"], "marginal skill-condition strata mismatch")
    for arm in ("M1", "M2"):
        t = totals[arm]
        require(t["rows"] % budget["accumulation"] == 0, "partial accumulation")
        require(budget["minimum_consumed_per_arm_per_seed"] <= t["nonpadding_tokens"] <= budget["ceiling_per_trained_arm_per_seed"], "token dose interval")
        require(abs(t["skill_tokens"].get("R", 0) - t["skill_tokens"].get("S", 0)) <=
                2 * budget["maximum_per_skill_budget_difference_from_half"], "R/S token allocation")
        for skill in ("R", "S"):
            slots = metrics["answer_slots"][arm].get(skill, 0)
            require(slots > 0, "missing skill")
            observed = metrics["answer_position_counts"][arm][skill]
            for j, pair in metrics["position_expectations"][arm][skill].items():
                require(abs(Fraction(observed.get(j, 0)) - Fraction(*pair)) <= POSITION_RESIDUAL * slots,
                        "answer position opportunity imbalance")
    require(totals["M1"]["rows"] == totals["M2"]["rows"], "optimizer update mismatch")
    require(abs(totals["M1"]["nonpadding_tokens"] - totals["M2"]["nonpadding_tokens"]) <=
            budget["maximum_between_arm_residual_tokens"], "between-arm token residual")
    for skill in ("R", "S"):
        require(metrics["answer_slots"]["M1"][skill] == metrics["answer_slots"]["M2"][skill], "answer-slot count mismatch")
        a, b = metrics["answer_position_counts"]["M1"][skill], metrics["answer_position_counts"]["M2"][skill]
        limit = POSITION_DIFFERENCE * metrics["answer_slots"]["M1"][skill]
        require(all(abs(a.get(j, 0) - b.get(j, 0)) <= limit for j in set(a) | set(b)), "between-arm answer positions")


def _make_schedule(weights: dict, pool: dict, quads: list, cells: dict, arrays: dict,
                   enrollment: dict, decisions: dict, solver: dict) -> dict:
    occurrences = []
    for pid, ci in quads:
        repeat = weights[f"{pid}:{ci}"]
        require(isint(repeat, 1, 6), "quad repeat limit")
        occurrences.extend((pid, ci, t, occurrence) for occurrence in range(repeat) for t in range(4))
    random.Random(enrollment["seed"]).shuffle(occurrences)
    schedule_ids = {a: [cells[(pid, ci, t, a)] for pid, ci, t, _ in occurrences] for a in ("M1", "M2")}
    metrics = _metrics(schedule_ids, pool, arrays)
    _check_metrics(metrics, _budget(decisions))
    result = {"version": VERSION, "status": "MATCHED_CPU_SCHEDULE", "schedules": schedule_ids,
              "quad_weights": weights, "enrollment_sha256": enrollment["sha256"],
              "decisions_sha256": digest(decisions), "policy_sha256": digest(_policy()),
              "arrays_sha256": digest(arrays), "metrics": metrics,
              "optimizer_updates_per_arm": metrics["totals"]["M1"]["rows"] // _budget(decisions)["accumulation"],
              "solver": solver, "training_admitted": False, "native_launch_authorized": False}
    result["sha256"] = digest(result)
    return result


def validate_schedule(schedule: dict, rows: list[dict], arrays: dict, enrollment: dict, decisions: dict) -> None:
    """Exact standard-library checker; no optimizer imports or trusted flags."""
    require(type(schedule) is dict and schedule.get("status") == "MATCHED_CPU_SCHEDULE", "no matched schedule")
    pool, quads, cells = _pool(rows, arrays, enrollment)
    require(type(schedule.get("quad_weights")) is dict and set(schedule["quad_weights"]) ==
            {f"{p}:{c}" for p, c in quads}, "complete quad weights")
    expected = _make_schedule(schedule["quad_weights"], pool, quads, cells, arrays, enrollment, decisions, schedule.get("solver"))
    require(canonical(schedule) == canonical(expected), "schedule counts, binding or order drift")


def match_schedule(rows: list[dict], arrays: dict, enrollment: dict, decisions: dict) -> dict:
    pool, quads, cells = _pool(rows, arrays, enrollment)
    budget = _budget(decisions)
    require(not any(k.split(".")[0] in {"torch", "transformers", "tensorflow", "jax"} for k in sys.modules), "CPU allocation process has model framework")
    dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
    require(dependencies == {"numpy": "2.4.6", "scipy": "1.17.1"}, "CPU solver dependency version drift")
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = len(quads) + 1  # Last integer is optimizer update count.
    constraints, lower, upper = [], [], []

    def constraint(vector: list, lo: float, hi: float) -> None:
        constraints.append(vector)
        lower.append(lo)
        upper.append(hi)

    def vector(function) -> list:
        return [sum(function(pool[cells[(pid, ci, t, arm)]], arrays[cells[(pid, ci, t, arm)]], arm)
                    for t in range(4) for arm in ("M1", "M2")) for pid, ci in quads] + [0]

    costs = {}
    for arm in ("M1", "M2"):
        costs[arm] = vector(lambda row, arr, a: arr["total_tokens"] if a == arm else 0)
        constraint(costs[arm], budget["minimum_consumed_per_arm_per_seed"], budget["ceiling_per_trained_arm_per_seed"])
        difference = vector(lambda r, arr, a: arr["total_tokens"] * (1 if r["skill"] == "R" else -1) if a == arm else 0)
        margin = 2 * budget["maximum_per_skill_budget_difference_from_half"]
        constraint(difference, -margin, margin)
    residual = [a - b for a, b in zip(costs["M1"], costs["M2"])]
    constraint(residual, -budget["maximum_between_arm_residual_tokens"], budget["maximum_between_arm_residual_tokens"])
    constraint([4] * len(quads) + [-budget["accumulation"]], 0, 0)
    for skill in ("R", "S"):
        for value in range(1, 6):
            def indicator(row, arr, arm):
                spec = parse_visible(row["messages"])
                if row["skill"] != skill:
                    return 0
                actual = spec["selection"]["k"] if skill == "S" else len(_chain(spec, spec["lookup"]["requested"][0])[0])
                return (1 if arm == "M1" else -1) if actual == value else 0
            constraint(vector(indicator), 0, 0)
        # Answer opportunity residuals are normalized within arm x skill.
        for arm in ("M1", "M2"):
            for index in range(15):
                for sign in (-1, 1):
                    def position(row, arr, a):
                        if row["skill"] != skill or a != arm:
                            return 0
                        spec = parse_visible(row["messages"])
                        answer = parse_answer(row["target"], len(spec["initial"]))
                        expected = len(answer) / len(spec["initial"]) if index < len(spec["initial"]) else 0
                        return sign * ((1 if index in answer else 0) - expected) - float(POSITION_RESIDUAL) * len(answer)
                    constraint(vector(position), -np.inf, 0)
        for index in range(15):
            for sign in (-1, 1):
                def between(row, arr, arm):
                    if row["skill"] != skill:
                        return 0
                    answer = parse_answer(row["target"], len(row["spec"]["initial"]))
                    return sign * (1 if arm == "M1" else -1) * (1 if index in answer else 0) - (float(POSITION_DIFFERENCE) * len(answer) if arm == "M1" else 0)
                constraint(vector(between), -np.inf, 0)
    objective = [1 + int(digest({"parent": pid, "condition": ci})[:8], 16) / 2**32 * 1e-6 for pid, ci in quads] + [0]
    result = milp(np.array(objective), integrality=np.ones(n),
                  bounds=Bounds(np.array([1] * len(quads) + [1]), np.array([6] * len(quads) + [8192])),
                  constraints=LinearConstraint(np.array(constraints), np.array(lower), np.array(upper)),
                  options={"time_limit": 60, "mip_rel_gap": 0.0, "presolve": True})
    solver = {"backend": "scipy_milp_HiGHS", "dependencies": dependencies,
              "status_code": int(result.status), "message": str(result.message), "optimal": result.status == 0}
    if result.x is None:
        return {"status": "BLOCKED_MATCHING", "reason": "solver_reported_infeasible" if result.status == 2 else "solver_did_not_establish_feasibility",
                "solver": solver, "enrollment_sha256": enrollment["sha256"], "training_admitted": False}
    require(all(abs(float(value) - round(float(value))) <= 1e-5 for value in result.x), "noninteger allocation witness")
    weights = {f"{pid}:{ci}": round(float(result.x[j])) for j, (pid, ci) in enumerate(quads)}
    try:
        schedule = _make_schedule(weights, pool, quads, cells, arrays, enrollment, decisions, solver)
        validate_schedule(schedule, rows, arrays, enrollment, decisions)
        return schedule
    except ValueError as exc:
        return {"status": "BLOCKED_MATCHING", "reason": "allocation_failed_independent_exact_checks", "detail": str(exc),
                "solver": solver, "enrollment_sha256": enrollment["sha256"], "training_admitted": False}
