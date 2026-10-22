"""Reproduce historical patterns and counterexamples; no new model runs or gains."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .pattern_statistics import clopper_pearson, critical_count, is_floor_clear, upper_tail
from .queue import BUSINESS, ROOT, external_output, read_evidence, require, sha_file, write_new

EVIDENCE = ROOT / "experiments/architecture_benchmark_001/evidence/retention-design-v1"
RECORDS = BUSINESS / "rineya-dataset-retention-records"
SOURCE_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"


def registry():
    rows = []
    for seed in (17, 23, 29):
        rows += [("d0", seed, f"rel_d0s{seed}", f"rel-d0s{seed}"),
                 ("a1", seed, f"rel_a1s{seed}", f"rel-a1s{seed}")]
    rows += [("d1", 17, "rrp008_D1", "dose-D1"),
             ("d1", 23, "d1r_s23", "d1r-s23"), ("d1", 29, "d1r_s29", "d1r-s29")]
    for seed in (41, 47, 53):
        rows.append(("d1", seed, f"rel_d1s{seed}", f"rel-d1s{seed}"))
    rows += [("d3", 17, "supa_rrp007_a1", "process-a1-prepared-002"),
             ("d3", 23, "rrp007_s23", "process-a1-prepared-s23"),
             ("d3", 29, "rrp007_s29", "process-a1-prepared-s29")]
    rows += [("m1", seed, "rrp009_M1" + (f"_s{seed}" if seed != 17 else ""),
              "mix-M1" + (f"-s{seed}" if seed != 17 else "")) for seed in (17, 23, 29)]
    return rows


def process_summary(value):
    """Retally existing recorded flags; never regrade or run model output."""
    predictions = value["predictions"]
    require(type(predictions) is list and len(predictions) == value["aggregate"]["n"], "prediction inventory")
    ids = [row["id"] for row in predictions]
    require(len(set(ids)) == len(ids), "duplicate evaluation identity")
    for row in predictions:
        require(all(type(row[key]) is bool for key in ("canonical", "strict", "terminated")), "typed recorded flags")
    total = sum(row["canonical"] for row in predictions)
    require(total == value["aggregate"]["canonical_correct"]
            and sum(row["strict"] for row in predictions) == value["aggregate"]["strict_correct"], "aggregate tally drift")
    families = {}
    for family, reported in value["per_family"].items():
        subset = [row for row in predictions if row["family"] == family]
        k, n = sum(row["canonical"] for row in subset), len(subset)
        require((k, n) == (reported["canonical_correct"], reported["n"]), "family tally drift")
        require(sum(row["strict"] for row in subset) == reported["strict_correct"], "strict family tally drift")
        floor = reported["binding_floor_count"]
        low, high = clopper_pearson(k, n)
        clear = is_floor_clear(k, n, floor)
        require(clear is reported["clears_floor"], "recorded confidence verdict disagrees")
        require(abs(low - reported["clopper_pearson_95"][0]) <= .000051
                and abs(high - reported["clopper_pearson_95"][1]) <= .000051, "recorded confidence interval drift")
        families[family] = {"correct": k, "n": n, "floor_count": floor,
            "point_estimate_exceeds_floor": k > floor, "implemented_floor_cleared": clear,
            "ci95": [low, high], "one_sided_p": upper_tail(k, n, floor / n),
            "one_sided_critical_05": critical_count(n, floor / n, .05),
            "two_sided_lower_bound_critical_95": critical_count(n, floor / n, .025)}
    return {"correct": total, "n": len(ids), "families": families,
            "all_family_confidence_criteria": all(item["implemented_floor_cleared"] for item in families.values()),
            "views_sha256": value["views_sha256"], "scorer_digest": value["implementation_digest"],
            "quality_claim": value.get("quality_claim"),
            "prediction_id_digest": hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()}


def retained_summary(value):
    results = {}
    for policy in ("permissive", "strict"):
        item = value["by_bare_policy"][policy]
        source = item["source_correct_total"]
        retained = source - item["lost_total"]
        require(retained == item["arm_correct_total"] - item["arm_correct_outside_source"], "source-cohort retention identities disagree")
        results[policy] = {"source_correct": source, "retained": retained,
                          "lost_original": item["lost_total"], "panel_correct": item["arm_correct_total"],
                          "union_reference_size": item["reference_size"],
                          "union_per_class_retained_sum": sum(row["retained"] for row in item["per_class"].values()),
                          "strict_zero_loss_status": item["status"]}
    return {"policies": results, "probe_rows": value["probe_rows"], "probe_sha256": value["probe_sha256"],
            "reference_digest": value["reference_digest"], "illustrative_sample_count": len(value.get("samples", [])),
            "complete_retention_identity_rederivation": False}


def paired_process(left, right):
    require(left["views_sha256"] == right["views_sha256"]
            and left["implementation_digest"] == right["implementation_digest"], "acquisition bank/scorer mismatch")
    a, b = ({row["id"]: row["canonical"] for row in value["predictions"]} for value in (left, right))
    require(set(a) == set(b), "paired acquisition identities differ")
    repairs = sum(not a[key] and b[key] for key in a)
    damages = sum(a[key] and not b[key] for key in a)
    return {"repairs": repairs, "damages": damages, "net": repairs - damages, "n": len(a),
            "scope": "paired exposed historical evaluation identities; not new confirmation"}


def parse_pruning_table(text):
    rows = {}
    for line in text.splitlines():
        fields = [part.strip().replace("**", "") for part in line.strip().strip("|").split("|")]
        if len(fields) != 10 or fields[0] not in ("17", "23", "29") or fields[1] not in ("ctrl", "k25"):
            continue
        seed, arm = int(fields[0]), fields[1]
        require((seed, arm) not in rows, "duplicate pruning table row")
        rows[seed, arm] = {"acquisition": int(fields[2]), "arithmetic": int(fields[4]),
                          "strict_arithmetic": int(fields[5]), "retained": int(fields[6]),
                          "long_context": int(fields[7]), "literal_unknown": int(fields[9])}
    require(len(rows) == 6, "complete six-arm paired pruning table required")
    pairs = []
    for seed in (17, 23, 29):
        left, right = rows[seed, "ctrl"], rows[seed, "k25"]
        pairs.append({"seed": seed, "replication_role": "selection_anchor" if seed == 17 else "prospective_new_draw",
                      "full": left, "k25": right, "delta": {key: right[key] - left[key] for key in left},
                      "remaining_original_losses": 47 - right["retained"],
                      "dominates_frozen_source": False})
    return pairs


def training_parents(build, plan):
    """Use explicit identities where recorded; otherwise label ID derivation."""
    if "selected_parent_ids" in build:
        ids = build["selected_parent_ids"]
        seed = build["selection_seed"]
        method = "EXPLICIT_BUILD_PARENT_IDS"
    else:
        rows = [row for row in plan["row_order_per_exposure"] if not row.startswith("rehearsal:")]
        require(all(re.fullmatch(r"RRFR2-[0-9a-f]{24}-step[0-9]+-valid", row) for row in rows),
                "unrecognized transition identity; no heuristic grouping")
        ids = [row.split("-step", 1)[0] for row in rows]
        seed = build["transition_draw"]["selection_seed"] if "transition_draw" in build else build["seed"]
        method = "PLAN_TRANSITION_ID_PREFIX_DERIVATION"
    require(len(set(ids)) == len(ids), "duplicate training parent")
    if "row_order_per_exposure" in plan:
        rows = [row for row in plan["row_order_per_exposure"] if not row.startswith("rehearsal:")]
        require(all(re.fullmatch(r"RRFR2-[0-9a-f]{24}-step[0-9]+-valid", row) for row in rows), "plan transition identity format")
        planned = [row.split("-step", 1)[0] for row in rows]
        require(len(set(planned)) == len(planned) and set(planned) == set(ids), "build/plan training parent roster mismatch")
    return set(ids), seed, method


def evaluation_parents(process):
    rows = [row["id"] for row in process["predictions"]]
    require(all(re.fullmatch(r"RRP-process:RRFR2-[0-9a-f]{24}-step[0-9]+-valid", row) for row in rows),
            "unrecognized evaluation parent identity")
    parents = {row.split(":", 1)[1].split("-step", 1)[0] for row in rows}
    require(len(parents) == len(rows), "evaluation has multiple views per parent")
    return parents


def audit():
    snapshots = {}
    def load(key, path, kind="json"):
        require(key not in snapshots, "duplicate audit source")
        snap = read_evidence({"id": key, "path": str(path), "format": kind,
                              "classification": "historical", "role": "receipt" if kind == "json" else "protocol"})
        snapshots[key] = snap
        return snap["json"] if kind == "json" else snap["text"]
    rows, raw_process, parent_sets = [], {}, {}
    for condition, seed, receipt_stem, plan_stem in registry():
        key = condition + "_" + str(seed)
        process = load(key + "_process", EVIDENCE / ("PROCESS_" + receipt_stem + ".json"))
        retention_stem = "rrp007_a1" if condition == "d3" and seed == 17 else receipt_stem
        retention = load(key + "_retention", EVIDENCE / ("RETENTION_PROBE_" + retention_stem + ".json"))
        plan = load(key + "_plan", RECORDS / (plan_stem + "-PLAN.json"))
        build_dir = plan_stem if plan_stem.startswith("process-a1-prepared") else plan_stem + "-prepared"
        build = load(key + "_build", RECORDS / build_dir / "ARM_BUILD.json")
        raw_process[key] = process
        parent_sets[key], actual_seed, identity_method = training_parents(build, plan)
        require(actual_seed == seed, "actual build selection seed mismatch")
        eval_parents = evaluation_parents(process)
        require(not parent_sets[key] & eval_parents, "historical acquisition training/evaluation parent collision")
        proc, ret = process_summary(process), retained_summary(retention)
        require(proc["quality_claim"] is False, "historical receipt not a quality authorization")
        journal_name = {"process-a1-prepared-002": "process-a1-journal-002",
                        "process-a1-prepared-s23": "process-a1-journal-s23",
                        "process-a1-prepared-s29": "process-a1-journal-s29"}.get(plan_stem, plan_stem + "-journal")
        attempts, training = [], {"status": "NOT_MEASURED", "reason": "No completed source-bound training journal was supplied by known metadata paths."}
        for attempt in ("train-001", "train-002"):
            directory = RECORDS / journal_name / "runs" / attempt
            if not (directory / "result.json").is_file():
                continue
            recorded = load(key + "_" + attempt.replace("-", "_") + "_result", directory / "result.json")
            attempts.append({"attempt": attempt, "status": recorded.get("status"), "complete": recorded.get("complete"),
                             "synthetic": recorded.get("synthetic")})
            if recorded.get("complete") is True and recorded.get("synthetic") is False:
                binding = load(key + "_" + attempt.replace("-", "_") + "_binding", directory / "binding.json")["binding"]
                require(recorded.get("binding") == binding, "embedded result binding differs from separate source binding")
                require(binding["source"]["assets"]["model.safetensors"] == SOURCE_SHA, "historical training source mismatch")
                plan_digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                require(binding["plan_sha256"] == plan_digest, "historical canonical training plan mismatch")
                require(recorded["progress"]["processed_tokens"] == plan["processed_tokens"]
                        and recorded["progress"]["supervised_tokens"] == plan["supervised_tokens"]
                        and recorded["progress"]["updates"] == len(plan["update_groups"]), "historical completion/exposure mismatch")
                training = {"status": "COMPLETED_HISTORICAL_NATIVE_MICROFIT_NOT_QUALITY_CONFIRMATION",
                            "attempt": attempt, "source_weight_sha256": binding["source"]["assets"]["model.safetensors"],
                            "runtime": binding["runtime"], "progress": recorded["progress"],
                            "scored_model_lineage": "ARCHIVAL_RUN_ATTRIBUTION_NOT_FULL_CHECKPOINT_EXPORT_VERIFICATION",
                            "plan_digest": plan_digest, "plan_digest_semantics": "CANONICAL_SORTED_JSON_NOT_FILE_BYTES",
                            "source_code_sha256": binding["code_sha256"], "recipe_code_sha256": binding["mq_code_sha256"]}
        rows.append({"condition": condition, "seed": seed, "acquisition": proc, "retention": ret,
                     "soft_arithmetic_gate": ret["policies"]["permissive"]["panel_correct"] >= 62
                         and ret["policies"]["permissive"]["retained"] >= 37,
                     "planned_processed_positions": plan["processed_tokens"],
                     "planned_supervised_positions": plan["supervised_tokens"],
                     "planned_updates": len(plan["update_groups"]), "optimizer": plan["optimizer"],
                     "order_seed": plan["order_seed"], "plan_selection_seed": plan.get("selection_seed"),
                     "actual_selection_seed": actual_seed, "training_identity_method": identity_method,
                     "plan_selection_seed_disagrees": plan.get("selection_seed") is not None and plan["selection_seed"] != actual_seed,
                     "training_parent_count": len(parent_sets[key]), "source_evidence_prefix": key,
                     "exact_training_eval_parent_overlap": 0, "semantic_template_decontamination_established": False,
                     "historical_training_evidence": training, "historical_attempts": attempts})
    by_key = {row["condition"] + "_" + str(row["seed"]): row for row in rows}
    nested, allocation_pairs, replay_pairs = [], [], []
    for seed in (17, 23, 29):
        d0, a1, d1 = (parent_sets[condition + "_" + str(seed)] for condition in ("d0", "a1", "d1"))
        require(d0 < a1 < d1, "nested draw identity mismatch")
        nested.append({"seed": seed, "counts": [len(d0), len(a1), len(d1)], "outsiders": [len(d0-a1), len(a1-d1)]})
        d, a = by_key[f"d1_{seed}"], by_key[f"a1_{seed}"]
        allocation_pairs.append({"seed": seed, "acquisition": paired_process(raw_process[f"d1_{seed}"], raw_process[f"a1_{seed}"]),
            "source_retention_delta": a["retention"]["policies"]["permissive"]["retained"] - d["retention"]["policies"]["permissive"]["retained"],
            "planned_position_ratio": a["planned_processed_positions"] / d["planned_processed_positions"],
            "updates": [d["planned_updates"], a["planned_updates"]], "compute_matched": False})
        d, m = by_key[f"d3_{seed}"], by_key[f"m1_{seed}"]
        replay_pairs.append({"seed": seed, "acquisition": paired_process(raw_process[f"d3_{seed}"], raw_process[f"m1_{seed}"]),
            "source_retention_delta": m["retention"]["policies"]["permissive"]["retained"] - d["retention"]["policies"]["permissive"]["retained"],
            "planned_position_ratio": m["planned_processed_positions"] / d["planned_processed_positions"],
            "updates": [d["planned_updates"], m["planned_updates"]], "measured_flops_matched": False})
    m, d = by_key["m1_17"], by_key["d1_17"]
    equal_retention = {"seed": 17, "acquisition": paired_process(raw_process["d1_17"], raw_process["m1_17"]),
        "retained_counts": [d["retention"]["policies"]["permissive"]["retained"], m["retention"]["policies"]["permissive"]["retained"]],
        "planned_position_ratio": m["planned_processed_positions"] / d["planned_processed_positions"],
        "updates": [d["planned_updates"], m["planned_updates"]], "compute_matched": False}
    floor_doc = load("floor_registration", RECORDS / "forge-v2-prereg-001/PROCESS_PROBE_FLOORS.md", "text")
    require("two-sided exact" in floor_doc and "one-sided exact" in floor_doc, "historical threshold ambiguity absent")
    prune_doc = ROOT.parent / "ChatBubble-rineya-pruning-retention-confirm-v1/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/evidence/REP001_RESULTS.md"
    pruning = parse_pruning_table(load("rep001", prune_doc, "text"))
    sparse_doc = load("sparse001", prune_doc.with_name("SPARSE001_RESULTS.md"), "text")
    require("| α = 0.50 — shrink all elements by half | 219 | 67 | **42** | **164** | 12 |" in sparse_doc
            and "| k = 0.50 — keep top half at full strength | **235** | 68 | 39 | 159 | 9 |" in sparse_doc,
            "nominal-scale counterexample source drift")
    result = {"schema": 1, "status": "HISTORICAL_PATTERN_REPRODUCED_NO_DETERMINISTIC_BALANCED_GAIN",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "source_weight_sha256": SOURCE_SHA,
        "scope": "Supa1.7 historical FP32 ordinary adaptation on exposed process and arithmetic instruments; not current Titan/RSD qualification",
        "runs": rows, "nested_parent_checks": nested,
        "asymmetric_minus_balanced_pairs": allocation_pairs, "replay_minus_transition_only_pairs": replay_pairs,
        "equal_retained_count_comparison": equal_retention, "pruning_paired_draws": pruning,
        "floor_correction": {"d0_point_estimates_exceed_floor": True,
            "d0_implemented_confidence_criteria": [by_key[f"d0_{seed}"]["acquisition"]["families"]["ledger_replay"] for seed in (17,23,29)],
            "interpretation": "Failure of the confidence criterion is not below-shortcut point performance or proof of no learning.",
            "historical_verdicts_changed": False},
        "counterexamples": [
            "Pruning strict arithmetic drops1 case in seeds17 and23; seed23 permissive arithmetic drops5. Never-hurts broad quality is false.",
            "Every pruned donor still loses6/8/7 original source successes; all literal UNKNOWN counts remain below source13.",
            "Nominal half-scaling retains42 versus39 for half-magnitude pruning, while acquisition219 versus235 differs; neither control is matched acquisition/norm.",
            "A1 retains more than matched D1 in3/3 draws but loses26/9/32 acquisition cases and changes updates/processed exposure.",
            "M1 repairs narrow arithmetic retention relative to D3 but sacrifices acquisition; broad long-context/abstention damage remains in historical reports.",
            "The equal-retention+29 acquisition contrast costs approximately4x processed positions and is one draw, not a compute-matched replicated win."],
        "research_next": {"primary": "Preserve the existing one-donor magnitude-versus-acquisition-matched-scalar proposal; qualify its exact runtime and independently reviewed fresh banks through its owner.",
            "allocation_followup": "Prospectively compare asymmetric and balanced256-parent schedules with matched exposure/updates and fresh reviewed family/source-retention instruments; do not retrofit old gates.",
            "replay_followup": "Independently vary replay draws and effective supervised dose; do not infer a universal share threshold from two single-seed mixture points.",
            "native_execution_authorized": False, "protected_bank_access_authorized": False},
        "goal_completion_audit": {"reproducible_historical_direction": "PROVED_IN_RECORDED_SCOPE",
            "isolated_magnitude_mechanism": "NOT_MEASURED_MATCHED_SCALAR_REQUIRED",
            "historical_never_hurts_overclaim": "CONTRADICTED_BY_RECORDED_COUNTEREXAMPLES_NOT_AN_EXTRA_ACCEPTANCE_RULE",
            "registered_balanced_preservation": "NOT_PROVED_ON_CURRENT_RUNTIME_AND_FRESH_REVIEWED_BANKS",
            "independent_fresh_confirmation": "NOT_MEASURED",
            "current_source_runtime_and_preservation": "UNQUALIFIED",
            "full_goal_achieved": False},
        "limitations": ["Recorded flags are retallied, not independently semantically regraded.",
            "Source/plan-bound completed journals are verified, but registry joins do not prove every scored HF export/research checkpoint matches the final training-state bytes.",
            "Some later HF retention captures also retain verified pre-load file identities; this is distinct from loaded-state identity and does not retroactively qualify path-only acquisition/long-context captures.",
            "Retention receipts expose only illustrative samples; cohort counts are arithmetically checked but full identity-level retention cannot be rederived.",
            "Repeated training draws share exposed evaluation parents; they are not new independent evaluation samples.",
            "Planned input positions are a cost proxy, not measured FLOPs; fixed token budget does not fix supervised objective dose.",
            "The seed17 pruning anchor was selected earlier; only23/29 are new prospective draws.",
            "Historical long-context/abstention is report-only and cannot select or confirm a winner."],
        "activity": {"native_loads": 0, "native_updates": 0, "native_evaluations": 0, "protected_bank_reads": 0},
        "promotion_permitted": False, "quality_improvement_established": False}
    return result, snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = external_output(args.output)
    result, snapshots = audit()
    output.mkdir(parents=True, exist_ok=False)
    bindings = []
    for key, snap in snapshots.items():
        target = output / "evidence" / (key + Path(snap["binding"]["path"]).suffix)
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(snap["raw"])
        bindings.append(dict(snap["binding"], capsule=target.relative_to(output).as_posix()))
    result["source_bindings"] = bindings
    write_new(output / "PATTERN_AUDIT.json", result)
    write_new(output / "MANIFEST.json", {"files": {path.relative_to(output).as_posix(): sha_file(path)
                                                  for path in output.rglob("*") if path.is_file()}})
    print(json.dumps({"status": result["status"], "historical_runs_retallied": len(result["runs"]),
                      "source_files": len(bindings), "goal_achieved": False, "report": str(output / "PATTERN_AUDIT.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
