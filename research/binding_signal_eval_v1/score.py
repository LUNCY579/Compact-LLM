"""Offline exploratory DEV baseline scoring at the independent-world level.

Primary correctness is canonical_complete_exact: EOS AND NFKC/whitespace/casefold
exact text. Content-only canonical/strict metrics remain separate. No substring,
punctuation stripping, answer extraction or model judging occurs. Scoring never
authorizes training, even when baseline repair separation is demonstrated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys
import unicodedata

if __package__:
    from . import common as c
else:
    import common as c

CATEGORIES = ("person_answerable", "person_missing", "person_vacant", "role_answerable")
UNKNOWN_CATEGORIES = ("person_missing", "person_vacant")
SCORE_KEYS = {"schema", "record", "experiment", "baseline_plan_path", "baseline_plan_sha256",
              "grant_path", "grant_sha256", "gold_path", "gold_sha256", "scorer_sha256",
              "generator_path", "generator_sha256"}


def normalize(text, *, casefold=True):
    c.require(isinstance(text, str), "scorer accepts strings only")
    result = " ".join(unicodedata.normalize("NFKC", text).split())
    return result.casefold() if casefold else result


def exact_pair(source, lora):
    c.require(source and source.keys() == lora.keys() and
              all(type(v) is bool for m in (source, lora) for v in m.values()), "paired world maps differ")
    wins = sum(source[k] and not lora[k] for k in source)
    losses = sum(lora[k] and not source[k] for k in source)
    nd = wins + losses
    def probability(w, l):
        n = w + l; term = tail = 1
        for k in range(1, min(w, l) + 1):
            term = term * (n - k + 1) // k; tail += term
        return min(Fraction(1), Fraction(2 * tail, 1 << n))
    minimum = next((net for net in range(nd % 2, nd + 1, 2)
                    if probability((nd + net) // 2, (nd - net) // 2) <= Fraction(1, 20)), None)
    return {"worlds": len(source), "source_true": sum(source.values()), "lora_true": sum(lora.values()),
            "source_wins": wins, "lora_wins": losses, "net_source_minus_lora": wins - losses,
            "nd": nd, "exact_p": float(probability(wins, losses)),
            "minimum_significant_abs_net_at_observed_nd": minimum}


def holm_two(contrasts):
    c.require(set(contrasts) == set(UNKNOWN_CATEGORIES), "primary multiplicity family is exactly missing and vacant")
    running = 0.0
    for rank, (_, value) in enumerate(sorted(contrasts.items(), key=lambda item: item[1]["exact_p"])):
        running = max(running, min(1.0, (2 - rank) * value["exact_p"]))
        value.update(holm_adjusted_p=running, holm_family_size=2)


def validate_gold(inputs, gold):
    c.require(len(gold) == 384, "complete DEV gold required")
    source = {row["id"]: row for row in inputs}; result = {}; worlds = defaultdict(Counter)
    for row in gold:
        c.require(set(row) == {"id", "parent_id", "category", "reference"}, "unregistered gold fields")
        key = row["id"]
        c.require(key in source and key not in result and row["parent_id"] == source[key]["parent_id"], "gold input/world alignment differs")
        c.require(row["category"] in CATEGORIES and isinstance(row["reference"], str) and row["reference"].strip(), "invalid category/reference")
        if row["category"] in UNKNOWN_CATEGORIES:
            c.require(row["reference"] == "UNKNOWN", "unknown-category gold must be literal UNKNOWN")
        else:
            c.require(normalize(row["reference"]) != "unknown", "positive-control reference cannot be UNKNOWN")
        result[key] = row; worlds[row["parent_id"]][row["category"]] += 1
    c.require(result.keys() == source.keys() and len(worlds) == 32 and
              all(dict(counts) == {category: 3 for category in CATEGORIES} for counts in worlds.values()),
              "gold requires exactly three views per category in each of 32 worlds")
    return result


def score_rows(inputs, gold, predictions, decode_policy_input):
    source = {row["id"]: row for row in inputs}
    c.require(len(predictions) == 384, "partial generations cannot be scored")
    raw = {row["id"]: row for row in predictions}
    c.require(len(raw) == 384 and raw.keys() == gold.keys() == source.keys(), "prediction IDs differ from DEV")
    details = {}; category_rows = defaultdict(list); world_rows = defaultdict(lambda: defaultdict(list))
    for key, target in gold.items():
        pred = raw[key]
        c.require(pred["parent_id"] == target["parent_id"] and type(pred.get("terminated")) is bool and
                  isinstance(pred.get("prediction"), str), "invalid prediction/world/termination record")
        text = pred["prediction"]; terminated = pred["terminated"]
        canonical = normalize(text) == normalize(target["reference"])
        strict = normalize(text, casefold=False) == normalize(target["reference"], casefold=False)
        unknown = normalize(text) == "unknown"
        parsed = decode_policy_input(source[key]["prompt"][1]["content"])
        people = parsed["world"]["people"]
        c.require(isinstance(people, list) and len(people) == len(set(people)) == 6 and
                  all(isinstance(person, str) for person in people), "independent policy parser did not recover six roster people")
        c.require(parsed["answer"] == target["reference"], "independent policy-input oracle disagrees with DEV gold")
        unanswerable = target["category"] in UNKNOWN_CATEGORIES
        row = {"id": key, "parent_id": target["parent_id"], "category": target["category"],
               "canonical_text_exact": canonical, "strict_text_exact": strict, "terminated": terminated,
               "canonical_complete_exact": canonical and terminated, "strict_complete_exact": strict and terminated,
               "canonical_exact_unknown": unknown, "false_unknown_on_positive": unknown and not unanswerable,
               "exact_roster_person_on_unanswerable": unanswerable and normalize(text) in {normalize(p) for p in people}}
        details[key] = row; category_rows[target["category"]].append(row)
        world_rows[target["parent_id"]][target["category"]].append(row)
    fields = ("canonical_text_exact", "strict_text_exact", "canonical_complete_exact", "strict_complete_exact", "terminated")
    categories = {cat: {"rows": 96, **{field: sum(row[field] for row in rows) for field in fields}}
                  for cat, rows in category_rows.items()}
    maps = {policy: {category: {} for category in CATEGORIES} for policy in ("canonical_complete_exact", "strict_complete_exact")}
    for policy in maps:
        maps[policy].update(unknown_combined_all6={}, no_false_unknown_all6_positive_controls={})
    roster = {}
    for world, cats in world_rows.items():
        negatives = cats["person_missing"] + cats["person_vacant"]
        positives = cats["person_answerable"] + cats["role_answerable"]
        for policy in maps:
            for category in CATEGORIES:
                maps[policy][category][world] = all(row[policy] for row in cats[category])
            maps[policy]["unknown_combined_all6"][world] = all(row[policy] for row in negatives)
            maps[policy]["no_false_unknown_all6_positive_controls"][world] = not any(row["false_unknown_on_positive"] for row in positives)
        roster[world] = any(row["exact_roster_person_on_unanswerable"] for row in negatives)
    totals = {field: sum(row[field] for row in details.values()) for field in fields}
    totals.update(rows=384, worlds=32, positive_control_rows=192, unanswerable_rows=192,
                  false_unknown_on_positive=sum(row["false_unknown_on_positive"] for row in details.values()),
                  exact_roster_person_on_unanswerable=sum(row["exact_roster_person_on_unanswerable"] for row in details.values()))
    return {"totals": totals, "categories": categories, "world_counts": {
                policy: {endpoint: sum(worlds.values()) for endpoint, worlds in endpoints.items()} for policy, endpoints in maps.items()},
            "roster_person_world_any_count": sum(roster.values()), "rows": details}, maps, roster


def sensitivity(source_maps, lora_maps):
    comparisons = {policy: {endpoint: exact_pair(source_maps[policy][endpoint], lora_maps[policy][endpoint])
                           for endpoint in source_maps[policy]} for policy in source_maps}
    for policy in comparisons:
        holm_two({category: comparisons[policy][category] for category in UNKNOWN_CATEGORIES})
    primary = comparisons["canonical_complete_exact"]
    passed = any(primary[category]["net_source_minus_lora"] >= 6 and primary[category]["holm_adjusted_p"] <= .05
                 for category in UNKNOWN_CATEGORIES)
    failures = {category: 32 - sum(lora_maps["canonical_complete_exact"][category].values()) for category in UNKNOWN_CATEGORIES}
    headroom = any(count >= 7 for count in failures.values())
    return {"status": "BASELINE_REPAIR_SEPARATION_ESTABLISHED" if passed else "BASELINE_REPAIR_SEPARATION_NOT_ESTABLISHED",
            "primary_policy": "canonical_complete_exact", "repair_separation_passed": passed,
            "rule": "at least one unknown category: source-minus-LoRA net >=6 worlds and Holm(2) adjusted exact two-sided p <=0.05",
            "minimum_initial_Holm2_all_direction_wins": 7,
            "lora_unknown_category_world_failures": failures,
            "new_skill_headroom": "NEW_SKILL_HEADROOM_PRESENT" if headroom else "INSUFFICIENT_HEADROOM_FOR_7_WORLD_HOLM2_GAIN",
            "new_skill_route_requires_separate_registration": True, "training_permission": False,
            "world_contrasts_source_minus_lora": comparisons}


def verify_worker(plan, plan_path, grant_path, role):
    root = c.absolute(plan["output_root"]); out = root / role
    c.require(not (out / "FAILURE.json").exists() and not (out / "supervisor_failure.json").exists(), "baseline attempt has a failure receipt")
    binding = c.read(out / "BASELINE_BINDING.json")
    c.require(binding.get("status") == "COMPLETE" and binding.get("complete") is True and
              binding.get("model_role") == role and type(binding.get("controller_returncode")) is int and
              binding["controller_returncode"] == 0, "baseline parent did not accept a successful worker")
    for stem, filename in (("supervisor", "supervisor.json"), ("worker_terminal", "worker_terminal.json"), ("generations", "GENERATIONS.json")):
        c.require(c.sha(out / filename) == binding[stem + "_sha256"], "baseline execution artifact hash differs")
    supervisor, terminal, generations = (c.read(out / name) for name in ("supervisor.json", "worker_terminal.json", "GENERATIONS.json"))
    c.verify_terminal(supervisor, terminal)
    launch_identity = c.verify_launch(out, plan_path, grant_path, plan, role, supervisor)
    c.require(all(binding.get(k) == v for k, v in launch_identity.items()), "baseline launch custody differs")
    c.require(binding["plan_sha256"] == terminal["plan_sha256"] == generations["plan_sha256"] == c.sha(plan_path) and
              binding["grant_sha256"] == terminal["grant_sha256"] == generations["grant_sha256"] == c.sha(grant_path),
              "baseline output belongs to another plan/grant")
    reservation_path = root / f"RESERVATION_{role}.json"; reservation = c.read(reservation_path)
    c.require(c.sha(reservation_path) == binding["reservation_sha256"] and reservation["model_role"] == role and
              reservation["plan_sha256"] == c.sha(plan_path) and reservation["grant_sha256"] == c.sha(grant_path) and
              reservation["reserved_worker_seconds"] == 900 and reservation["reserved_rows"] == 384 and
              reservation["reserved_generated_tokens"] == 9216 and reservation["reserved_forward_calls"] == 9600 and reservation["retries"] == 0,
              "baseline reservation is missing or changed")
    before = generations["before"]
    c.require(before == generations["after"] == terminal["before"] == terminal["after"] and
              before["input_sha256"] == plan["input_sha256"] and before["code"] == c.code_snapshot(plan) and
              before["runtime_versions"] == plan["runtime_versions"] and before["model"] == c.model_snapshot(plan, role),
              "baseline model/input/code/runtime custody differs")
    c.require(terminal["generations_sha256"] == c.sha(out / "GENERATIONS.json") and
              generations["model_role"] == role and generations["rows"] == 384 and generations["worlds"] == 32 and
              generations["gold_file_read"] is False and generations["numerics"] == c.NUMERICS, "baseline generation scope differs")
    eos = generations["eos_ids"]
    c.require(isinstance(eos, list) and bool(eos) and all(type(t) is int for t in eos), "generation EOS IDs missing")
    for row in generations["predictions"]:
        tokens = row["generated_ids_including_eos"]
        c.require(isinstance(tokens, list) and 1 <= len(tokens) == row["generated_tokens"] <= 24 and
                  all(type(t) is int and 0 <= t < 151936 for t in tokens) and
                  row["terminated"] == (tokens[-1] in eos) and not any(t in eos for t in tokens[:-1]),
                  "generation exceeded its token bound or EOS custody differs")
        c.require(row["forward_calls"] == len(tokens) and
                  row["processed_input_tokens"] == row["input_tokens"] + len(tokens) - 1,
                  "generation forward-call/input-token accounting differs")
    actual = {"rows": len(generations["predictions"]),
              "generated_tokens_including_eos": sum(r["generated_tokens"] for r in generations["predictions"]),
              "forward_calls": sum(r["forward_calls"] for r in generations["predictions"]),
              "processed_input_tokens": sum(r["processed_input_tokens"] for r in generations["predictions"])}
    c.require(generations["actual_budget_charges"] == actual and generations["extra_warmup_or_qualification_calls"] == 0 and
              actual["rows"] == 384 and actual["generated_tokens_including_eos"] <= 9216 and
              actual["forward_calls"] <= reservation["reserved_forward_calls"] and
              actual["processed_input_tokens"] <= reservation["reserved_processed_input_tokens"],
              "baseline actual calls/tokens exceed or disagree with reservations")
    return generations, {"binding_sha256": c.sha(out / "BASELINE_BINDING.json"),
                         "generations_sha256": c.sha(out / "GENERATIONS.json"), "supervisor_sha256": c.sha(out / "supervisor.json"),
                         "worker_seconds": supervisor["wall_seconds"], "actual_budget_charges": actual,
                         "reserved_processed_input_tokens": reservation["reserved_processed_input_tokens"]}


def build_report(score_plan_path):
    c.require("torch" not in sys.modules, "offline scorer must run without model imports")
    config = c.read(score_plan_path)
    c.require(set(config) == SCORE_KEYS and config["schema"] == 1 and config["record"] == "BINDING_DEV_SCORE_PLAN" and
              config["experiment"] == c.EXPERIMENT, "unregistered scoring plan")
    paths = {key: c.absolute(config[key + "_path"], file=True) for key in ("baseline_plan", "grant", "gold", "generator")}
    for key, path in paths.items():
        c.require(c.sha(path) == config[key + "_sha256"], f"scoring {key} identity changed")
    c.require(c.sha(Path(__file__)) == config["scorer_sha256"], "scorer differs from the frozen normalization/statistics code")
    plan = c.validate_plan(c.read(paths["baseline_plan"])); grant = c.read(paths["grant"])
    c.validate_grant(paths["baseline_plan"], grant)
    c.phase_guard(plan, grant)
    c.require(c.sha(plan["input_path"]) == plan["input_sha256"], "admitted DEV input bytes differ")
    inputs = c.validate_inputs(c.read_jsonl(plan["input_path"]))
    gold = validate_gold(inputs, c.read_jsonl(paths["gold"]))
    bundle, custody = {}, {}
    # Establish both successful supervised legs before reading any score value.
    for role in c.MODEL_SHAS:
        bundle[role], custody[role] = verify_worker(plan, paths["baseline_plan"], paths["grant"], role)
        by_id = {row["id"]: row for row in bundle[role]["predictions"]}
        c.require(len(by_id) == 384 and all(row["id"] in by_id and by_id[row["id"]]["input_tokens"] == len(row["prompt_ids"])
                                          for row in inputs), "recorded model-input token counts differ from admitted prompts")
    c.require(bundle["source"]["numerics"] == bundle["lora"]["numerics"] and
              sum(v["worker_seconds"] for v in custody.values()) <= 1800, "baseline numerical settings or total time differ")
    charges = {key: sum(v["actual_budget_charges"][key] for v in custody.values())
               for key in ("rows", "generated_tokens_including_eos", "forward_calls", "processed_input_tokens")}
    c.require(charges["rows"] == 768 and charges["generated_tokens_including_eos"] <= 18432 and
              charges["forward_calls"] <= c.LIMITS["forward_calls"] and charges["processed_input_tokens"] <= c.LIMITS["processed_input_tokens"] and
              sum(v["reserved_processed_input_tokens"] for v in custody.values()) <= c.LIMITS["processed_input_tokens"],
              "aggregate baseline reservations/actual calls/tokens exceed grant")
    reservations = list(Path(plan["output_root"]).glob("RESERVATION_*.json"))
    c.require({p.name for p in reservations} == {"RESERVATION_source.json", "RESERVATION_lora.json"}, "unexpected baseline workers/retries")
    spec = importlib.util.spec_from_file_location("frozen_binding_v2_policy_decoder", paths["generator"])
    c.require(spec is not None and spec.loader is not None, "independent policy-input decoder missing")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    c.require("torch" not in sys.modules, "policy decoder imported a model runtime")
    endpoints, maps, roster = {}, {}, {}
    for role in c.MODEL_SHAS:
        endpoints[role], maps[role], roster[role] = score_rows(inputs, gold, bundle[role]["predictions"], module.decode_policy_input)
    decision = sensitivity(maps["source"], maps["lora"])
    for key, path in paths.items():
        c.require(c.sha(path) == config[key + "_sha256"], "scoring input changed while being read")
    c.phase_guard(plan, grant)
    return {"schema": 1, "record": "BINDING_DEV_BASELINE_REPORT", "experiment": c.EXPERIMENT, **decision,
            "created_utc": datetime.now(timezone.utc).isoformat(), "score_plan_sha256": c.sha(score_plan_path),
            "source_files_sha256": {str(path): config[key + "_sha256"] for key, path in paths.items()},
            "input_sha256": plan["input_sha256"], "scorer_sha256": config["scorer_sha256"], "custody": custody,
            "actual_budget_charges": charges, "registered_limits": dict(c.LIMITS),
            "endpoints": endpoints, "world_maps": maps,
            "roster_person_world_any_maps": roster,
            "roster_person_world_any_source_minus_lora": exact_pair(roster["source"], roster["lora"]),
            "normalization": {"canonical": "NFKC + whitespace collapse + casefold, exact equality",
                              "strict": "NFKC + whitespace collapse, case-sensitive exact equality",
                              "primary": "EOS termination AND canonical exact text", "punctuation_stripped": False,
                              "UNKNOWN": "canonical exact content only; no substring/extraction",
                              "roster_diagnostic": "on unanswerable rows, canonical reply equals one of six independently decoded roster names; descriptive only"},
            "limitations": ["Fresh exploratory DEV baseline; not independent confirmation or a repair result.",
                "The 32 parent worlds are the paired units; 384 row-level counts are descriptive, not 384 independent trials.",
                "SOURCE-minus-LoRA separation qualifies restoration sensitivity only. Failure does not establish inability to learn new binding skills.",
                "At least seven control-world failures is only headroom for a separately registered new-skill route, never permission or a prediction of success.",
                "Exact UNKNOWN and exact roster-person diagnostics retain punctuation. UNKNOWN. is not UNKNOWN; a non-match does not prove absence of semantic abstention or person emission.",
                "No corpus adaptation or case selection may follow these baseline outcomes. No baseline status authorizes training."],
            "training_permission": False, "quality_claim": False, "release_permitted": False, "selection_permitted": False}


def markdown(report):
    lines = ["# Binding DEV baseline", "", f"**{report['status']}**", "", "Primary correctness requires canonical exact text and EOS termination.", "",
             "| Model | Category | Canonical text /96 | Strict text /96 | Canonical + EOS /96 | Strict + EOS /96 | Terminated /96 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for role, endpoint in report["endpoints"].items():
        for category in CATEGORIES:
            v = endpoint["categories"][category]
            lines.append(f"| {role} | {category} | {v['canonical_text_exact']} | {v['strict_text_exact']} | {v['canonical_complete_exact']} | {v['strict_complete_exact']} | {v['terminated']} |")
    lines += ["", "## Descriptive output diagnostics", "",
              "False UNKNOWN uses canonical exact content regardless of EOS. Roster-person emission requires exact equality to one of the six independently decoded names; a non-match is not a semantic judgment.", "",
              "| Model | Terminated /384 | False UNKNOWN on positive /192 | Exact roster person on unanswerable /192 | Any such person in world /32 |",
              "|---|---:|---:|---:|---:|"]
    for role, endpoint in report["endpoints"].items():
        v = endpoint["totals"]
        lines.append(f"| {role} | {v['terminated']} | {v['false_unknown_on_positive']} | {v['exact_roster_person_on_unanswerable']} | {endpoint['roster_person_world_any_count']} |")
    lines += ["", "## World-level contrasts", "", "Positive net means SOURCE exceeds LoRA. Holm families contain only missing and vacant within each scoring policy; the registered gate uses canonical + EOS.", "",
              "| Policy | Endpoint | SOURCE /32 | LoRA /32 | net | nd | exact p | Holm p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for policy, endpoints in report["world_contrasts_source_minus_lora"].items():
        for name, v in endpoints.items():
            lines.append(f"| {policy} | {name} | {v['source_true']} | {v['lora_true']} | {v['net_source_minus_lora']:+d} | {v['nd']} | {v['exact_p']:.8g} | {v.get('holm_adjusted_p', 'descriptive only')} |")
    lines += ["", f"New-skill headroom: **{report['new_skill_headroom']}**. A separate registration is required; training permission remains false.", "", "## Limits", ""]
    lines += [f"- {line}" for line in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = c.absolute(args.out); md = output.with_suffix(".md")
    c.require(not output.exists() and not md.exists(), "baseline score outputs are create-only")
    try:
        report = build_report(args.score_plan)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        print({"status": "BLOCKED_ON_BASELINE_COMPARABILITY", "reason": str(exc), "training_permission": False})
        return 2
    c.write_new(output, report)
    with md.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown(report))
    print({"status": report["status"], "outputs": [str(output), str(md)], "training_permission": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
