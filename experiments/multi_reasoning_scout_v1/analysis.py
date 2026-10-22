"""Gold is read here only after supervised generation; oracle never enters inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import math
from pathlib import Path
import random

from .core import (RECORDS, ROOT, PublicItem, decode, digest, file_hash, git, load_frozen, new_dir,
                   read_json, validate_receipt, verify_files, write_new)
from .parsing import answer, majority, parse
from . import prompts


def fraction(n, d):
    return n / d if d else None


def paired_counts(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("nonempty aligned pairs required")
    both = sum(x and y for x, y in zip(a, b))
    repairs = sum(not x and y for x, y in zip(a, b))
    damage = sum(x and not y for x, y in zip(a, b))
    n = repairs + damage
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(repairs, damage) + 1)) / 2**n) if n else 1.
    return {"n": len(a), "both_correct": both, "repairs": repairs, "damage": damage,
            "both_wrong": len(a)-both-repairs-damage, "net_gain": repairs-damage,
            "delta_accuracy": (repairs-damage)/len(a), "exact_McNemar_two_sided_p": p,
            "p_status": "descriptive exploratory; not multiplicity-adjusted or confirmation"}


def cluster_ci(values, worlds, seed=1978, draws=2000):
    if not values or len(values) != len(worlds):
        raise ValueError("clustered observations required")
    groups = defaultdict(list)
    for v, world in zip(values, worlds):
        groups[world].append(v)
    samples, rng, keys = [], random.Random(seed), sorted(groups)
    for _ in range(draws):
        selected = [v for k in rng.choices(keys, k=len(keys)) for v in groups[k]]
        samples.append(sum(selected)/len(selected))
    samples.sort()
    return {"low": samples[int(.025*draws)], "high": samples[int(.975*draws)-1],
            "world_clusters": len(groups), "draws": draws, "seed": seed,
            "method": "paired world-cluster percentile bootstrap; descriptive 95%"}


def arm_score(record, name):
    return answer(record["outputs"][record["arms"][name]])


def state_outcome(value, gold, family):
    if family not in ("register_machine", "counterfactual_state", "prerequisite_schedule"):
        return {"applicable": False, "valid": None, "category": None}
    category = ("valid_final_state" if value == gold["answer"] else
                "format_or_termination" if value is None else
                "ignored_missing_premise" if not gold["answerable"] else
                "false_abstention" if value == "UNKNOWN" else "wrong_final_state_or_schedule")
    return {"applicable": True, "valid": value == gold["answer"], "category": category}


def complementarity(rows, labels, keys=("R0", "R1", "R2", "R3")):
    result = {}
    for first, second in combinations(keys, 2):
        a = [answer(r["outputs"][first]) for r in rows]
        b = [answer(r["outputs"][second]) for r in rows]
        c = paired_counts([x == labels[r["item"]["id"]]["answer"] for x, r in zip(a, rows)],
                          [x == labels[r["item"]["id"]]["answer"] for x, r in zip(b, rows)])
        result[f"{first}:{second}"] = {"n": len(rows), "both_correct": c["both_correct"],
            "first_only_correct": c["damage"], "second_only_correct": c["repairs"],
            "both_wrong": c["both_wrong"], "disagreement_rate": sum(x != y for x, y in zip(a, b))/len(rows),
            "mean_unique_valid_answers": sum(len({x, y}-{None}) for x, y in zip(a, b))/len(rows)}
    return result


def summarize(rows, labels):
    if not rows:
        raise ValueError("no completed cases")
    ids = [r["item"]["id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) - set(labels):
        raise ValueError("duplicate or unlabelled items")
    correctness, stats, scored = {}, {}, []
    for arm in ("A0", "A1", "A2", "A3", "S1"):
        correct = [arm_score(r, arm) == labels[r["item"]["id"]]["answer"] for r in rows]
        correctness[arm] = correct
        selected = [r["outputs"][r["arms"][arm]] for r in rows]
        ans = [i for i, r in enumerate(rows) if labels[r["item"]["id"]]["answerable"]]
        unans = [i for i in range(len(rows)) if i not in ans]
        logical = [r["outputs"][k] for r in rows for k in r["logical_calls"][arm]]
        stats[arm] = {"n": len(rows), "correct": sum(correct), "accuracy": sum(correct)/len(rows),
            "answerable_accuracy": fraction(sum(correct[i] for i in ans), len(ans)),
            "unanswerable_accuracy": fraction(sum(correct[i] for i in unans), len(unans)),
            "false_abstention_rate": fraction(sum(answer(selected[i]) == "UNKNOWN" for i in ans), len(ans)),
            "unsupported_answer_rate": fraction(sum(answer(selected[i]) not in (None, "UNKNOWN") for i in unans), len(unans)),
            "formatting_failure_rate": sum(r["format_failure"] for r in selected)/len(rows),
            "nontermination_rate": sum(not r["terminated"] for r in selected)/len(rows),
            "parser_repairs": sum(r["repair_count"] for r in logical),
            "logical_cost": {k: sum(r[k] for r in logical) for k in
                             ("prompt_tokens", "output_tokens", "total_tokens", "generation_calls", "forward_calls", "wall_seconds")}}
        states = [state_outcome(arm_score(r, arm), labels[r["item"]["id"]], r["item"]["family"]) for r in rows]
        state_n = sum(s["applicable"] for s in states)
        stats[arm]["state_validity"] = {"n": state_n, "valid": sum(s["valid"] is True for s in states),
            "rate": fraction(sum(s["valid"] is True for s in states), state_n),
            "categories": dict(Counter(s["category"] for s in states if s["applicable"])),
            "scope": "Final-state/earliest-time correctness only; intermediate prose trajectories are not verified"}
    paired = {}
    worlds = [r["item"]["scenario_id"] for r in rows]
    for a, b in (("A0", "A1"), ("A0", "A2"), ("A0", "A3"), ("A1", "A3"), ("A2", "A3"),
                 ("A2", "S1"), ("S1", "A3")):
        values = [int(y)-int(x) for x, y in zip(correctness[a], correctness[b])]
        paired[f"{b}-{a}"] = {**paired_counts(correctness[a], correctness[b]), "CI": cluster_ci(values, worlds)}
    ceiling, repeated_ceiling, any_all, diversity = [], [], [], []
    selector = {k: Counter() for k in ("S0", "S1", "S2")}
    state_counts = Counter()
    for i, row in enumerate(rows):
        gold = labels[row["item"]["id"]]
        candidates = [answer(row["outputs"][r]) for r in ("R0", "R1", "R2", "R3")]
        ceiling.append(gold["answer"] in candidates)
        repeated_ceiling.append(gold["answer"] in [answer(row["outputs"][k]) for k in ("R0", "D1", "D2", "D3")])
        all_answers = [answer(row["outputs"][r]) for r in ("R0", "R1", "R2", "R3", "D1", "D2", "D3", "A1")]
        any_all.append(gold["answer"] in all_answers)
        diversity.append(len(set(candidates)-{None}))
        for key, chosen in (("S0", row["S0"]), ("S1", arm_score(row, "S1")), ("S2", arm_score(row, "A3"))):
            selector[key]["correct"] += chosen == gold["answer"]
            selector[key]["correct_when_candidate_exists"] += ceiling[-1] and chosen == gold["answer"]
            selector[key]["correct_candidate_existed_selector_wrong"] += ceiling[-1] and chosen != gold["answer"]
            selector[key]["no_candidate_correct"] += not ceiling[-1]
            selector[key]["novel_correct_synthesis"] += not ceiling[-1] and chosen == gold["answer"]
            wrong_longer = any(c != gold["answer"] and c == chosen and
                               row["outputs"][k]["output_tokens"] > min(
                                   (row["outputs"][j]["output_tokens"] for j in ("R0", "R1", "R2", "R3")
                                    if answer(row["outputs"][j]) == gold["answer"]), default=10**9)
                               for c, k in zip(candidates, ("R0", "R1", "R2", "R3")))
            selector[key]["selected_longer_wrong_when_correct_available_association"] += wrong_longer
        final = arm_score(row, "A3")
        if row["item"]["family"] in ("register_machine", "counterfactual_state", "prerequisite_schedule"):
            state_counts["cases"] += 1
            state_counts["valid_final_state"] += final == gold["answer"]
            if final != gold["answer"]:
                category = ("format_or_termination" if final is None else "ignored_missing_premise" if not gold["answerable"]
                            else "false_abstention" if final == "UNKNOWN" else "wrong_final_state_or_schedule")
                state_counts[category] += 1
        scored.append({"id": row["item"]["id"], "scenario_id": row["item"]["scenario_id"],
                       "family": row["item"]["family"], "split": row["item"]["split"],
                       "difficulty": gold.get("difficulty"), "answerable": gold["answerable"],
                       "answers": {k: arm_score(row, k) for k in correctness},
                       "role_answers": {k: answer(row["outputs"][k]) for k in ("R0", "R1", "R2", "R3", "D1", "D2", "D3")},
                       "role_correct": {k: answer(row["outputs"][k]) == gold["answer"] for k in ("R0", "R1", "R2", "R3", "D1", "D2", "D3")},
                       "role_disagreement": diversity[-1] > 1,
                       "state_validity": {k: state_outcome(arm_score(row, k), gold, row["item"]["family"]) for k in correctness},
                       "selector_outcomes": {k: {"answer": value, "correct": value == gold["answer"],
                           "candidate_correct_existed": ceiling[-1],
                           "missed_correct_candidate": ceiling[-1] and value != gold["answer"],
                           "novel_correct_synthesis": not ceiling[-1] and value == gold["answer"]}
                           for k, value in (("S0", row["S0"]), ("S1", arm_score(row, "S1")), ("S2", final))},
                       "correct": {k: v[i] for k, v in correctness.items()}, "candidate_correct": ceiling[-1],
                       "repair": not correctness["A0"][i] and correctness["A3"][i],
                       "damage": correctness["A0"][i] and not correctness["A3"][i],
                       "unique_candidate_answers": diversity[-1]})
    selectors = {k: {**v, "accuracy": v["correct"]/len(rows),
                     "selector_efficiency": fraction(v["correct_when_candidate_exists"], sum(ceiling)),
                     "efficiency_denominator": sum(ceiling)} for k, v in selector.items()}
    delta_cost = {k: stats["A3"]["logical_cost"][k]-stats["A0"]["logical_cost"][k]
                  for k in stats["A3"]["logical_cost"]}
    comparison = paired["A3-A0"]
    per_role = {}
    for role in ("R0", "R1", "R2", "R3", "D1", "D2", "D3"):
        outputs = [r["outputs"][role] for r in rows]
        correct = [answer(o) == labels[r["item"]["id"]]["answer"] for o, r in zip(outputs, rows)]
        per_role[role] = {"n": len(rows), "correct": sum(correct), "accuracy": sum(correct)/len(rows),
                          "format_failures": sum(o["format_failure"] for o in outputs),
                          "nontermination": sum(not o["terminated"] for o in outputs)}
    return {"arms": stats, "paired": paired, "per_item": scored,
            "A4_oracle_analysis_only": {"A3_candidate_ceiling": sum(ceiling)/len(rows),
                                       "A2_candidate_ceiling": sum(repeated_ceiling)/len(rows),
                                       "all_solver_candidates_ceiling": sum(any_all)/len(rows), "deployable": False},
            "selectors": selectors, "per_role": per_role, "candidate_diversity_mean": sum(diversity)/len(rows),
            "role_disagreement_rate": sum(n > 1 for n in diversity)/len(rows),
            "complementarity": complementarity(rows, labels),
            "repeated_direct_complementarity": complementarity(rows, labels, ("R0", "D1", "D2", "D3")),
            "state_validity": {**state_counts, "rate": fraction(state_counts["valid_final_state"], state_counts["cases"]),
                               "scope": "Task-native final-state/earliest-time validity; intermediate prose states not scored as verified trajectories"},
            "cost_per_gross_repair": {k: fraction(v, comparison["repairs"]) for k, v in delta_cost.items()},
            "cost_per_net_recovered_case": {k: v/comparison["net_gain"] if comparison["net_gain"] > 0 else None for k, v in delta_cost.items()},
            "selector_diagnostic_limits": {"incorrect_checker_rejection": "Not identifiable in M1: independent checkers never see another candidate",
                                           "misled_by_verbosity": "Only association measured; no causal claim without randomized verbosity intervention"},
            "checker_errors": {
                "R2_false_insufficiency_on_answerable": sum(labels[r["item"]["id"]]["answerable"] and
                    r["outputs"]["R2"]["parsed"].get("sufficient_evidence") is False for r in rows),
                "R3_rejects_own_correct_answer": sum(answer(r["outputs"]["R3"]) == labels[r["item"]["id"]]["answer"] and
                    r["outputs"]["R3"]["parsed"].get("candidate_valid") is False for r in rows)}}


def factorial(old, new):
    native = "model" in old or "model" in new
    if not native and (old.get("native_execution_pass") or new.get("native_execution_pass")):
        raise ValueError("native claim requires registered model identities")
    if native:
        load_frozen()
        if old.get("native_execution_pass") is not True or new.get("native_execution_pass") is not True:
            raise ValueError("native factorial results require completed native runs")
        registry = read_json(RECORDS / "MODEL_MANIFEST.json")
        primary = next(m for m in registry["selected"]
                       if m["id"] == registry["primary_old_candidate_for_future"])
        if old.get("model") != primary:
            raise ValueError("old model must match the exact frozen primary D1 comparator")
        for key, filename in (("freeze_sha256", "FREEZE.json"), ("bank_sha256", "BANK_MANIFEST.json"),
                              ("prompt_sha256", "PROMPT_MANIFEST.json"), ("protocol_sha256", "protocol.json")):
            if old[key] != file_hash(RECORDS / filename) or new[key] != old[key]:
                raise ValueError("native factorial does not match current frozen artifact: " + key)
        current, future = old["model"], new["model"]
        if (current["id"] == future["id"] or
                Path(current["path"]).resolve() == Path(future["path"]).resolve() or
                current["assets"]["model.safetensors"] == future["assets"]["model.safetensors"]):
            raise ValueError("old/new checkpoint identities must differ")
        source_keys = ("source_repository", "source_branch", "source_sha")
        if (any(not isinstance(future.get(k), str) or not future[k] for k in source_keys) or
                tuple(current[k] for k in source_keys) == tuple(future[k] for k in source_keys)):
            raise ValueError("future training source identity must be recorded and distinct")
        if {k: v for k, v in current["assets"].items() if k != "model.safetensors"} != {
                k: v for k, v in future["assets"].items() if k != "model.safetensors"}:
            raise ValueError("future config/tokenizer assets differ")
        for key in ("policy", "runtime_dtype", "storage_dtype", "attention_backend", "decoding"):
            if old["model"][key] != new["model"][key]:
                raise ValueError("future serving policy differs: " + key)
        if old["runtime"] != new["runtime"]:
            raise ValueError("future runtime differs; qualify a new experiment")
    for key in ("freeze_sha256", "bank_sha256", "prompt_sha256", "protocol_sha256"):
        if old[key] != new[key]:
            raise ValueError("incompatible future experiment: " + key)
    arows, brows = old["per_item"], new["per_item"]
    a, b = {r["id"]: r for r in arows}, {r["id"]: r for r in brows}
    if not a or len(a) != len(arows) or len(b) != len(brows) or set(a) != set(b):
        raise ValueError("future comparison requires exact paired IDs, no partial intersection")
    per_item = []
    for item in sorted(a):
        x, y = a[item], b[item]
        if (x["scenario_id"], x["family"], x["split"], x.get("difficulty")) != (
                y["scenario_id"], y["family"], y["split"], y.get("difficulty")):
            raise ValueError("future item metadata differs")
        A, B, C, D = x["correct"]["A0"], x["correct"]["A3"], y["correct"]["A0"], y["correct"]["A3"]
        if any(type(v) is not bool for v in (A, B, C, D)):
            raise ValueError("paired outcomes must be boolean")
        per_item.append({"id": item, "scenario_id": x["scenario_id"], "family": x["family"],
            "difficulty": x.get("difficulty"), "split": x["split"],
            "A": A, "B": B, "C": C, "D": D,
            "pattern": ''.join(str(int(v)) for v in (A, B, C, D)), "all_four_correct": A and B and C and D,
            "training_only_repair": not A and C, "multi_only_repair": not A and B,
            "both_available": not A and B and C, "posttrain_multi_extra": not C and D,
            "multi_lost_after_training": B and not D, "harm_from_training": A and not C,
            "harm_from_multi_pretrain": A and not B, "harm_from_multi_posttrain": C and not D,
            "training_makes_multi_newly_useful": not (not A and B) and not C and D,
            "training_makes_multi_unnecessary": not A and B and C and D,
            "training_makes_multi_harmful": not (A and not B) and C and not D,
            "delta_training_direct": int(C)-int(A), "delta_multi_old": int(B)-int(A),
            "delta_multi_new": int(D)-int(C), "interaction": (int(D)-int(C))-(int(B)-int(A))})
    def aggregate(rs):
        deltas = ("delta_training_direct", "delta_multi_old", "delta_multi_new", "interaction")
        events = ("training_only_repair", "multi_only_repair", "both_available", "posttrain_multi_extra",
                  "multi_lost_after_training", "harm_from_training", "harm_from_multi_pretrain", "harm_from_multi_posttrain",
                  "all_four_correct", "training_makes_multi_newly_useful", "training_makes_multi_unnecessary",
                  "training_makes_multi_harmful")
        return {"n": len(rs), "cells": {k: sum(r[k] for r in rs)/len(rs) for k in "ABCD"},
                "patterns_ABCD": {''.join(bits): sum(r["pattern"] == ''.join(bits) for r in rs)
                                   for bits in product("01", repeat=4)},
                "counts": {k: sum(r[k] for r in rs) for k in events},
                "deltas": {k: sum(r[k] for r in rs)/len(rs) for k in deltas},
                "clustered_CI": {k: cluster_ci([r[k] for r in rs], [r["scenario_id"] for r in rs]) for k in deltas}}
    result = {"evidence_class": "OPEN_DEV_NATIVE_PAIRED" if native else "SOFTWARE_FIXTURE",
            "native_execution_pass": native, "open_dev_result": native,
            "confirmatory_claim": False, "release_claim": False,
            "overall": aggregate(per_item), "per_family": {f: aggregate([r for r in per_item if r["family"] == f])
            for f in sorted({r["family"] for r in per_item})},
            "per_difficulty": {str(d): aggregate([r for r in per_item if r["difficulty"] == d])
                               for d in sorted({r["difficulty"] for r in per_item}, key=str)},
            "per_family_difficulty": {f: {str(d): aggregate([r for r in per_item if r["family"] == f and r["difficulty"] == d])
                for d in sorted({r["difficulty"] for r in per_item if r["family"] == f}, key=str)}
                for f in sorted({r["family"] for r in per_item})}, "per_item": per_item,
            "event_definitions": {
                "training_makes_multi_newly_useful": "B-A <= 0 and D-C > 0",
                "training_makes_multi_unnecessary": "A wrong, B correct, C correct, D correct",
                "training_makes_multi_harmful": "B-A >= 0 and D-C < 0",
                "patterns_ABCD": "Exhaustive mutually exclusive Boolean patterns, in A/B/C/D order"},
            "significance_claim": False, "event_names_follow_user_definitions": "training_only and multi_only are not exclusive; both_available overlaps them"}
    # Preserve the already registered role/selection/cost diagnostics in the future paired report.
    result["before_after_diagnostics"] = {k: {"old": old.get(k), "new": new.get(k)} for k in (
        "per_role", "complementarity", "repeated_direct_complementarity", "A4_oracle_analysis_only",
        "selectors", "checker_errors", "candidate_diversity_mean", "role_disagreement_rate",
        "paired", "cost_per_gross_repair", "cost_per_net_recovered_case", "actual_unique_cost_all_partitions",
        "actual_unique_cost_including_qualification", "qualification_cost", "resources")}
    if "model" in old:
        result["identities"] = {"old": old["model"], "new": new["model"],
                                "freeze_sha256": old["freeze_sha256"]}
    return result


def validate_supervision(run, binding):
    supervisor = read_json(run / "supervisor.json")
    launch = read_json(run / "launch.json")
    spawn = read_json(run / "spawn.json")
    claim = read_json(run / "worker_claim.json")
    if (supervisor.get("status") != "EXITED" or supervisor.get("returncode") != 0 or
            (run / "failure.json").exists() or (run / "supervisor_failure.json").exists()):
        raise ValueError("successful supervision required for complete model result")
    if (launch.get("schema") != 1 or Path(launch["out"]).resolve() != run.resolve() or
            Path(launch["cwd"]).resolve() != ROOT.resolve() or
            type(launch.get("timeout_seconds")) is not int or launch["timeout_seconds"] <= 0):
        raise ValueError("supervisor launch does not match run")
    argv = launch["argv"]
    if (argv[1:4] != ["-m", "experiments.multi_reasoning_scout_v1.runner", "_worker"] or
            "--out" not in argv or "--model" not in argv or
            Path(argv[argv.index("--out")+1]).resolve() != run.resolve() or
            argv[argv.index("--model")+1] != binding["model"]["id"]):
        raise ValueError("supervisor command does not match model/run")
    if (spawn.get("launch_sha256") != digest(launch) or claim.get("launch_sha256") != digest(launch) or
            spawn.get("supervisor_pid") != launch["supervisor_pid"] or
            claim.get("supervisor_pid") != launch["supervisor_pid"] or
            supervisor.get("child_pid") != spawn.get("child_pid") or claim.get("schema") != 1):
        raise ValueError("unlinked supervisor receipts")
    direct = claim.get("pid") == spawn["child_pid"] and claim.get("parent_pid") == launch["supervisor_pid"]
    redirected = claim.get("launcher_redirected") is True and claim.get("parent_pid") == spawn["child_pid"]
    if not (direct or redirected):
        raise ValueError("worker claim does not match launched child")


def validate_item(row, public, binding, raw):
    if row["item"] != public or row["binding"] != binding:
        raise ValueError("item differs from frozen public bank or binding")
    independent = {k: (k, (), 64) for k in ("R0", "D1", "D2", "D3", "R1", "R2", "R3")}
    calls = {**independent, "A1": ("R0", (), 320),
             "A2_selector": ("S1", ("R0", "D1", "D2", "D3"), 64),
             "A3_selector": ("S2", ("R0", "R1", "R2", "R3"), 64),
             "S1_diagnostic": ("S1", ("R0", "R1", "R2", "R3"), 64)}
    arms = {"A0": "R0", "A1": "A1", "A2": "A2_selector", "A3": "A3_selector", "S1": "S1_diagnostic"}
    logical = {"A0": ["R0"], "A1": ["A1"], "A2": ["R0", "D1", "D2", "D3", "A2_selector"],
               "A3": ["R0", "R1", "R2", "R3", "A3_selector"], "S1": ["R0", "R1", "R2", "R3", "S1_diagnostic"]}
    if set(row["outputs"]) != set(calls) or row["arms"] != arms or row["logical_calls"] != logical:
        raise ValueError("unregistered role/arm/call mapping")
    item = PublicItem(**public)
    for key, (role, context, cap) in calls.items():
        record = row["outputs"][key]
        if record != read_json(raw / f"{item.id}.{key}.json"):
            raise ValueError("item subrecord differs from raw call")
        if (record["case_id"] != item.id or record["scenario_id"] != item.scenario_id or
                record["binding"] != binding or record["key"] != key or record["role"] != role or
                record["max_new_tokens"] != cap):
            raise ValueError("cross-model/item or changed role budget")
        prompt = prompts.messages(item, role, [row["outputs"][k] for k in context])
        if record["prompt"] != prompt or record["prompt_sha256"] != digest(prompt):
            raise ValueError("raw prompt differs from frozen role information")
        parsed = parse(record["raw_text"], role, record["terminated"])
        if any(record.get(k) != v for k, v in parsed.items()):
            raise ValueError("raw output and scored parser fields disagree")
        if (record["generation_calls"] != 1 or record["output_tokens"] != len(record["generated_ids"]) or
                record["forward_calls"] != record["output_tokens"] or
                record["total_tokens"] != record["prompt_tokens"] + record["output_tokens"] or
                not 0 < record["output_tokens"] <= cap or record["wall_seconds"] < 0):
            raise ValueError("inconsistent raw generation cost")
    if (row["S0"] != majority([row["outputs"][k] for k in ("R0", "R1", "R2", "R3")]) or
            row["A2_S0"] != majority([row["outputs"][k] for k in ("R0", "D1", "D2", "D3")])):
        raise ValueError("changed majority selector result")


def analyze_run(run, out):
    freeze = load_frozen()
    generation = read_json(run / "generation.json")
    binding = read_json(run / "binding.json")
    validate_receipt(generation, binding, "FROZEN_SCOUT_GENERATION")
    if binding["freeze_sha256"] != file_hash(RECORDS / "FREEZE.json"):
        raise ValueError("stale generation freeze")
    frozen_source, source = freeze["source"], binding["source"]
    if (any(source[k] != frozen_source[k] for k in ("repository", "branch", "files")) or
            source["files"] != freeze["source_files"] or
            git("merge-base", frozen_source["sha"], source["sha"]) != frozen_source["sha"]):
        raise ValueError("generation source differs from frozen source lineage")
    if binding["model_identity"] != digest(binding["model"]):
        raise ValueError("model identity digest mismatch")
    for name, stage in (("qualification.json", "NATIVE_QUALIFICATION"), ("smoke.json", "SCHEMA_SMOKE")):
        validate_receipt(read_json(run / name), binding, stage)
    qualification = read_json(run / "qualification.json")
    if (qualification.get("causality") is not True or qualification.get("repeat_ids_identical") is not True or
            qualification.get("training_steps") != 0 or generation.get("training_steps") != 0 or
            generation.get("state_unchanged") is not True):
        raise ValueError("incomplete inference-only qualification")
    validate_supervision(run, binding)
    rows = [read_json(p) for p in sorted((run / "raw").glob("*.item.json"))]
    manifest = read_json(RECORDS / "BANK_MANIFEST.json")
    keys = ("R0", "A1", "D1", "D2", "D3", "A2_selector", "R1", "R2", "R3", "A3_selector", "S1_diagnostic", "item")
    expected = {f"{item}.{key}.json" for item in manifest["case_ids"] for key in keys}
    if (set(generation["raw_files"]) != expected or {p.name for p in (run / "raw").iterdir()} != expected):
        raise ValueError("raw manifest/file set differs from complete registered run")
    verify_files(run / "raw", generation["raw_files"])
    if (len(rows) != manifest["n"] or generation.get("items") != manifest["n"] or
            {r["item"]["id"] for r in rows} != set(manifest["case_ids"])):
        raise ValueError("incomplete generation set")
    public = {r["id"]: r for r in map(decode, (RECORDS / "bank/public.jsonl").read_text().splitlines())}
    for row in rows:
        validate_item(row, public[row["item"]["id"]], binding, run / "raw")
    from .runner import smoke_pass
    smoke = [r for r in rows if r["item"]["split"] == "SMOKE"]
    if (not smoke or len(smoke) != read_json(run / "smoke.json")["items"] or not smoke_pass(smoke)):
        raise ValueError("schema smoke receipt differs from raw outputs")
    # First semantic gold read in the inference-to-analysis workflow.
    labels = {g["id"]: g for g in map(decode, (RECORDS / "bank/gold.jsonl").read_text().splitlines())}
    if set(labels) != set(public) or any(
            (g["scenario_id"], g["family"], g["split"]) !=
            (public[item]["scenario_id"], public[item]["family"], public[item]["split"]) or
            type(g["difficulty"]) is not int or g["difficulty"] < 0
            for item, g in labels.items()):
        raise ValueError("gold metadata differs from frozen public bank")
    check = [r for r in rows if r["item"]["split"] == "CHECK"]
    result = summarize(check, labels)
    result["per_family"] = {f: summarize([r for r in check if r["item"]["family"] == f], labels)
                            for f in manifest["families"]}
    result["per_difficulty"] = {str(d): summarize([r for r in check if labels[r["item"]["id"]]["difficulty"] == d], labels)
                                for d in sorted({labels[r["item"]["id"]]["difficulty"] for r in check})}
    result["per_family_difficulty"] = {f: {str(d): summarize([r for r in check if r["item"]["family"] == f and
        labels[r["item"]["id"]]["difficulty"] == d], labels)
        for d in sorted({labels[r["item"]["id"]]["difficulty"] for r in check if r["item"]["family"] == f})}
        for f in manifest["families"]}
    result.update(freeze_sha256=binding["freeze_sha256"], bank_sha256=file_hash(RECORDS / "BANK_MANIFEST.json"),
                  prompt_sha256=file_hash(RECORDS / "PROMPT_MANIFEST.json"), protocol_sha256=file_hash(RECORDS / "protocol.json"),
                  model=binding["model"], runtime=binding["runtime"], native_execution_pass=True, open_dev_result=True,
                  evidence_class="OPEN_DEV_NATIVE",
                  confirmatory_claim=False, release_claim=False, resources=generation["resources"])
    result["source"] = binding["source"]
    result["generation_binding"] = binding
    result["actual_unique_cost_all_partitions"] = {k: sum(r[k] for row in rows for r in row["outputs"].values())
        for k in ("prompt_tokens", "output_tokens", "total_tokens", "generation_calls", "forward_calls", "wall_seconds")}
    qualification_generations = qualification.get("qualification_generations", [])
    qualification_cost = {k: sum(r[k] for r in qualification_generations)
                          for k in result["actual_unique_cost_all_partitions"]}
    numerical_forwards = qualification.get("qualification_forward_calls", qualification_cost["forward_calls"]) - qualification_cost["forward_calls"]
    if numerical_forwards < 0:
        raise ValueError("inconsistent qualification forward-call accounting")
    qualification_cost["forward_calls"] += numerical_forwards
    result["actual_unique_cost_including_qualification"] = {
        k: value + qualification_cost[k] for k, value in result["actual_unique_cost_all_partitions"].items()}
    result["qualification_cost"] = {**qualification_cost, "numerical_forward_calls": numerical_forwards,
        "cost_scope": "Token counts and summed generation wall time cover generation calls; numerical qualification forwards are counted separately. Full loaded-worker elapsed time is resources.wall_seconds."}
    out = new_dir(out)
    write_new(out / "CURRENT_CANDIDATE_RESULTS.json", result)
    write_new(out / "COMPLEMENTARITY_MATRIX.json", {"all": result["complementarity"],
              "per_family": {f: v["complementarity"] for f, v in result["per_family"].items()}})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("score", "factorial"))
    p.add_argument("--run", type=Path)
    p.add_argument("--old", type=Path)
    p.add_argument("--new", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "score":
        analyze_run(args.run, args.out)
    else:
        result = factorial(read_json(args.old), read_json(args.new))
        new_dir(args.out)
        write_new(args.out / "FUTURE_2X2_RESULTS.json", result)


if __name__ == "__main__":
    main()
