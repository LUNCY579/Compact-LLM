"""Privileged, offline evaluation. Never imported by the native worker."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import random
import time

from .contracts import (digest, exact, file_sha, parse_answer, read_json, read_jsonl,
                        require, seed_for, source_identity, write_jsonl, write_new)
from .datasets import load_study
from .runner import Journal, slots, validate_bank
from .selection import public_candidate, select, pass_at_k


def paired_interval(rows: list[dict], treatment: str, baseline: str,
                    replicates: int, seed: int) -> list[float] | None:
    if not replicates:
        return None
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent"]].append(row[treatment]-row[baseline])
    blocks = list(groups.values())
    rng, values = random.Random(seed), []
    for _ in range(replicates):
        chosen = [rng.choice(blocks) for _ in blocks]
        values.append(sum(sum(b) for b in chosen)/sum(len(b) for b in chosen))
    values.sort()
    return [values[int(0.025*(replicates-1))], values[int(0.975*(replicates-1))]]


def evaluate(study_dir: Path, run: Path, oracle_path: Path, out: Path) -> dict:
    study, public = load_study(study_dir)
    require(file_sha(oracle_path) == study["dataset"]["oracle_sha256"], "oracle binding drift")
    private = read_jsonl(oracle_path)
    require(len(private) == len(public), "oracle population mismatch")
    public_by_id = {r["id"]: r for r in public}
    oracle, parents = {}, {}
    for r in private:
        exact(r, {"id", "parent", "family", "role", "expected", "messages_digest", "row_digest"}, "private oracle")
        require(r["id"] in public_by_id and r["id"] not in oracle, "oracle ID mismatch")
        require(r["role"] == study["dataset"]["role"] and r["messages_digest"] == digest(public_by_id[r["id"]]["messages"]),
                "oracle split/prompt drift")
        require(parse_answer(r["expected"]) == r["expected"], "invalid reference envelope")
        require(type(r["parent"]) is str and type(r["family"]) is str, "invalid parent/family")
        oracle[r["id"]] = r
        if r["parent"] in parents:
            require(parents[r["parent"]] == r["role"], "parent role collision")
        parents[r["parent"]] = r["role"]
    manifest = read_json(run / "run.json")
    require(manifest["binding"]["study_id"] == study["study_id"], "run does not match study")
    require(manifest["binding"]["code_digest"] == source_identity()["code_digest"],
            "evaluation source differs from frozen run; use its exact implementation")
    config = study["config"]
    journal = Journal(run / "events.jsonl", config["max_output_bytes"])
    inventory = slots(study, public)
    validate_bank(journal, inventory, config)
    complete = len(journal.records) == len(inventory) and all(r["state"] == "returned" for r in journal.records.values())
    all_records, correct = {}, {}
    for item in inventory:
        r = journal.records.get(item["slot"], {**item, "state": "unstarted", "token_ids": []})
        all_records[item["slot"]] = r
        target = oracle[item["task"]]["expected"]
        correct[item["slot"]] = int(r["state"] == "returned" and r.get("stop") == "eos"
                                     and parse_answer(r.get("text", "")) == target)
    grouped = defaultdict(list)
    for item in inventory:
        grouped[item["task"]].append(all_records[item["slot"]])
    curves, case_rows = [], []
    for k in config["budgets"]:
        measurements = []
        selector_seconds = 0.0
        pool_tokens, pool_positions, pool_seconds = 0, 0, 0.0
        for task in public:
            group = grouped[task["id"]]
            greedy, first, bank, pool = group[0], group[1], group[1:], group[1:k+1]
            public_pool = [public_candidate(r) for r in pool]
            started = time.perf_counter()
            choices = {policy: select(public_pool, policy, seed_for(config["seed"], task["id"], policy, k))
                       for policy in ("vote", "random")}
            selector_seconds += time.perf_counter()-started
            row = {"id": task["id"], "parent": oracle[task["id"]]["parent"],
                   "family": oracle[task["id"]]["family"], "k": k,
                   "greedy": correct[greedy["slot"]], "first": correct[first["slot"]],
                   "coverage": int(any(correct[r["slot"]] for r in pool)),
                   "coverage_upper": int(any(correct[r["slot"]] for r in pool)
                                         or any(r["state"] in {"unstarted", "pending", "failed"} for r in pool)),
                   "vote": correct.get(choices["vote"], 0), "random": correct.get(choices["random"], 0),
                   "vote_slot": choices["vote"], "random_slot": choices["random"],
                   "pass_estimate": pass_at_k(config["samples"], sum(correct[r["slot"]] for r in bank), k)
                       if all(r["state"] == "returned" for r in bank) else None}
            chosen = all_records.get(choices["vote"])
            abstains = chosen is not None and parse_answer(chosen.get("text", "")) == '{"answer":"UNKNOWN"}'
            expects_abstention = oracle[task["id"]]["expected"] == '{"answer":"UNKNOWN"}'
            row["false_abstention"] = int(abstains and not expects_abstention)
            row["correct_abstention"] = int(abstains and expects_abstention)
            row["invalid_or_unfinished_in_pool"] = sum(r["state"] != "returned" or r.get("stop") != "eos"
                                                        or parse_answer(r.get("text", "")) is None for r in pool)
            measurements.append(row)
            pool_tokens += sum(len(r["token_ids"]) for r in pool)
            pool_positions += sum(r.get("forward_positions", 0) for r in pool)
            pool_seconds += sum(r.get("seconds", 0.0) for r in pool)
        n = len(measurements)
        mean = lambda key: sum(r[key] for r in measurements)/n
        vote_count = sum(r["vote"] for r in measurements)
        coverage_count = sum(r["coverage"] for r in measurements)
        curve = {"k": k, "rows": n, "parents": len(parents), "greedy_accuracy": mean("greedy"),
                 "first_sample_accuracy": mean("first"), "prefix_coverage_lower": mean("coverage"),
                 "prefix_coverage_upper": mean("coverage_upper"), "vote_accuracy": mean("vote"),
                 "random_accuracy": mean("random"), "matched_selection_gap": mean("coverage")-mean("vote"),
                 "vote_conditional_recovery": vote_count/coverage_count if coverage_count else None,
                 "random_subset_pass_estimate": mean("pass_estimate")
                    if all(r["pass_estimate"] is not None for r in measurements) else None,
                 "vote_delta_vs_first": mean("vote")-mean("first"),
                 "vote_delta_vs_greedy": mean("vote")-mean("greedy"),
                 "coverage_delta_vs_first": mean("coverage")-mean("first"),
                 "vote_repairs_vs_first": sum(r["vote"] == 1 and r["first"] == 0 for r in measurements),
                 "vote_damages_vs_first": sum(r["vote"] == 0 and r["first"] == 1 for r in measurements),
                 "vote_retained_from_first": sum(r["vote"] == r["first"] == 1 for r in measurements),
                 "vote_delta_parent_bootstrap_95pct": paired_interval(measurements, "vote", "first", config["bootstrap_replicates"], config["seed"]),
                 "coverage_delta_parent_bootstrap_95pct": paired_interval(measurements, "coverage", "first", config["bootstrap_replicates"], config["seed"]),
                 "pool_durably_recorded_decode_tokens": pool_tokens, "returned_forward_positions": pool_positions,
                 "returned_generation_seconds": pool_seconds, "offline_selector_seconds": selector_seconds,
                 "tokens_per_correct_selected": pool_tokens/vote_count if vote_count else None,
                 "invalid_or_unfinished_rate": sum(r["invalid_or_unfinished_in_pool"] for r in measurements)/(n*k),
                 "false_abstentions": sum(r["false_abstention"] for r in measurements),
                 "correct_abstentions": sum(r["correct_abstention"] for r in measurements)}
        curves.append(curve)
        case_rows.extend(measurements)
    families = {}
    for family in sorted({r["family"] for r in private}):
        panel = [r for r in case_rows if r["family"] == family and r["k"] == config["samples"]]
        families[family] = {"rows": len(panel), "first": sum(r["first"] for r in panel),
                            "greedy": sum(r["greedy"] for r in panel), "coverage": sum(r["coverage"] for r in panel),
                            "vote": sum(r["vote"] for r in panel)}
    segments = [read_json(p) for p in sorted((run / "segments").glob("*/SUPERVISOR.json"))]
    result = {"schema": study["schema"], "study_id": study["study_id"], "run_binding": manifest["binding"],
              "evidence": manifest["binding"]["backend"]["evidence"],
              "status": "COMPLETE_BANK" if complete else "INCOMPLETE_BANK",
              "scheduled_attempts": len(inventory), "started_attempts": len(journal.records),
              "returned_attempts": sum(r["state"] == "returned" for r in journal.records.values()),
              "failed_attempts": sum(r["state"] == "failed" for r in journal.records.values()),
              "pending_attempts": int(journal.pending is not None),
              "durably_recorded_tokens": journal.tokens, "supervised_wall_seconds": sum(s["elapsed_seconds"] for s in segments),
              "cost_accounting_incomplete": not complete, "curves": curves, "families_at_max_k": families,
              "interpretation": "synthetic plumbing, not Titan efficacy" if study["dataset"]["kind"] == "synthetic_fixture"
                else "exploratory fixed-policy DEV evidence, not a release/preservation/confirmation result",
              "inference": "HEADROOM_OBSERVED" if complete and curves[-1]["coverage_delta_vs_first"] > 0
                else ("NO_HEADROOM_OBSERVED_WITHIN_BUDGET" if complete else "INCOMPLETE_DO_NOT_DECIDE"),
              "caveats": ["prefix pools and random-subset estimator are different estimands",
                          "bootstrap resamples whole parents; intervals are descriptive and not multiplicity-adjusted",
                          "repeated samples do not create independent problem observations",
                          "candidate availability does not prove reasoning or justify a weight improvement claim",
                          "pool costs are retrospective; total bank and qualification costs are additional",
                          "failed/pending GPU work can exceed durably logged tokens; reserved upper bounds remain binding",
                          "no learned verifier, production selection, long-context promotion, or training is implemented"]}
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "report.json", result)
    write_jsonl(out / "cases.private.jsonl", case_rows)
    with (out / "curves.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    lines = ["# RVS-001 experiment report", "", f"**{result['evidence']} / {result['status']}**", "",
             result["interpretation"], "", "| k | Greedy | First sample | Pool coverage (lower) | Vote | Selection gap |",
             "|---:|---:|---:|---:|---:|---:|"]
    for c in curves:
        lines.append(f"| {c['k']} | {c['greedy_accuracy']:.3f} | {c['first_sample_accuracy']:.3f} | "
                     f"{c['prefix_coverage_lower']:.3f} | {c['vote_accuracy']:.3f} | {c['matched_selection_gap']:.3f} |")
    lines += ["", f"Inference diagnostic: **{result['inference']}**.", "", "## Interpretation limits", ""]
    lines += ["- " + text for text in result["caveats"]]
    with (out / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    return result
