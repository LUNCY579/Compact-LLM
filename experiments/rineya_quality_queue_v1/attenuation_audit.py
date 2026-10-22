"""Historical attenuation/masking controls; no prospective gate or native permit."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .paired_statistics import exact_mcnemar, paired_bootstrap_interval, stratified_paired_bootstrap_interval
from .pattern_audit import paired_process, process_summary, retained_summary
from .queue import BUSINESS, ROOT, external_output, read_evidence, require, sha_file, write_new

RECORDS = BUSINESS / "rineya-dataset-retention-records"


def lost_source_ids(retention, policy="permissive"):
    item = retention["by_bare_policy"][policy]
    ids = [key for group in item["per_class"].values() for key in group["lost_ids"]]
    require(len(ids) == len(set(ids)) == item["lost_total"], "complete unique source-loss inventory required")
    require(item["source_correct_total"] - len(ids) == item["arm_correct_total"] - item["arm_correct_outside_source"],
            "source loss/retention arithmetic disagreement")
    return set(ids)


def compare_records(left, right, left_ret, right_ret):
    require(left["max_new_tokens"] == right["max_new_tokens"]
            and left["runtime"] == right["runtime"] and left["numerics"] == right["numerics"], "scoring budget/runtime mismatch")
    overall = paired_process(left, right)
    a, b = ({row["id"]: row for row in item["predictions"]} for item in (left, right))
    require(all("reference" in a[key] and "reference" in b[key]
                and a[key]["reference"] == b[key]["reference"] for key in a), "paired reference targets differ or are absent")
    families, strata = {}, []
    for family in sorted(left["per_family"]):
        ids = [key for key in a if a[key]["family"] == family]
        require(all(b[key]["family"] == family for key in ids), "paired family membership differs")
        benefit = sum(not a[key]["canonical"] and b[key]["canonical"] for key in ids)
        harm = sum(a[key]["canonical"] and not b[key]["canonical"] for key in ids)
        interval = paired_bootstrap_interval(benefit, harm, len(ids), .90)
        families[family] = {"n": len(ids), "left_correct": sum(a[key]["canonical"] for key in ids),
            "right_correct": sum(b[key]["canonical"] for key in ids), "benefits": benefit, "harms": harm,
            "difference": benefit-harm, "conditional_empirical_ci90": interval,
            "point_gap_within_05": abs(interval["net_rate"]) <= .05,
            "empirical_ci_inside_05": interval["lower"] > -.05 and interval["upper"] < .05}
        strata.append((benefit, harm, len(ids)))
    macro = stratified_paired_bootstrap_interval(strata, .90)
    require(left_ret["probe_sha256"] == right_ret["probe_sha256"]
            and left_ret["reference_digest"] == right_ret["reference_digest"], "source-retention instrument mismatch")
    require(type(left_ret.get("reference_receipt")) is str and bool(left_ret["reference_receipt"])
            and left_ret["reference_receipt"] == right_ret.get("reference_receipt"), "source-reference receipt identity mismatch")
    source_n = left_ret["by_bare_policy"]["permissive"]["source_correct_total"]
    require(source_n == right_ret["by_bare_policy"]["permissive"]["source_correct_total"], "retention cohort denominator mismatch")
    left_lost, right_lost = lost_source_ids(left_ret), lost_source_ids(right_ret)
    repairs, damages = len(left_lost-right_lost), len(right_lost-left_lost)
    retained = {"source_n": source_n, "left_retained": source_n-len(left_lost), "right_retained": source_n-len(right_lost),
        "repairs": repairs, "damages": damages, "difference": repairs-damages,
        "exact_mcnemar_two_sided": exact_mcnemar(repairs, damages),
        "pooled_case_empirical_ci90": paired_bootstrap_interval(repairs, damages, source_n, .90),
        "endpoint_scope": "historical pooled source-correct cases; not current registered family-macro retention",
        "unseen_case_independence_or_no_future_harm_established": False}
    return {"acquisition": overall, "acquisition_by_family": families, "source_retention": retained,
            "fixed_family_macro_empirical_ci90": macro,
            "point_overall_gap_within_03": abs(macro["net_rate"]) <= .03,
            "empirical_macro_ci_inside_03": macro["lower"] > -.03 and macro["upper"] < .03,
            "all_empirical_family_ci_inside_05": all(value["empirical_ci_inside_05"] for value in families.values()),
            "diagnostics_only_not_registered_gate_application": True, "native_execution_authorized": False,
            "confirmation_or_magnitude_specific_superiority_established": False}


def build_audit():
    snapshots = {}
    def load(key, path, kind="json", classification="historical", role="receipt"):
        snap = read_evidence({"id": key, "path": str(path), "format": kind,
                              "classification": classification, "role": role})
        snapshots[key] = snap
        return snap["json"] if kind == "json" else snap["text"]
    records, raw = {}, {}
    definitions = [("scalar025", "int001-20260917", "a025"), ("scalar050", "int001-20260917", "a050"),
                   ("scalar075", "int001-20260917", "a075"), ("full", "int001-20260917", "a100"),
                   ("mag10", "sparse001-20260917", "s10d"), ("mag25", "sparse001-20260917", "s25d"),
                   ("mag50", "sparse001-20260917", "s50d")]
    definitions += [(f"rep{seed}_{arm}", "rep001-20260917", f"rep_s{seed}_{arm}")
                    for seed in (23,29) for arm in ("ctrl","k25")]
    for name, directory, suffix in definitions:
        proc = load(name+"_process", RECORDS/directory/"evidence"/("PROCESS_int001_"+suffix+".json"))
        ret = load(name+"_retention", RECORDS/directory/"evidence"/("RETENTION_PROBE_int001_"+suffix+".json"))
        records[name] = {"process": process_summary(proc), "retention": retained_summary(ret)}
        raw[name] = (proc, ret)
    comparisons = {}
    for name, left, right in (("near_scalar075_mag25", "scalar075", "mag25"),
                               ("equal_aggregate_full_mag50", "full", "mag50"),
                               ("equal_nominal_half", "scalar050", "mag50"),
                               ("rep23_full_mag25", "rep23_ctrl", "rep23_k25"),
                               ("rep29_full_mag25", "rep29_ctrl", "rep29_k25")):
        comparisons[name] = dict(compare_records(raw[left][0], raw[right][0], raw[left][1], raw[right][1]), left=left, right=right)
    scalar_arm = load("scalar075_construction", RECORDS/"int001-20260917/arm-a075/ARM.json")
    mask_arm = load("mag25_construction", RECORDS/"sparse001-20260917/arm-s25d/ARM.json")
    mask50 = load("mag50_construction", RECORDS/"sparse001-20260917/arm-s50d/ARM.json")
    require(scalar_arm["source_weight_sha256"] == mask_arm["source_weight_sha256"]
            and scalar_arm["trained_fp32_sha256"] == mask_arm["trained_fp32_sha256"], "archival scalar/mask source/donor mismatch")
    bf16 = load("bf16_norm_census", RECORDS/"int001-20260917/BF16_EFFECTIVE_ALPHA.json")
    load("scalar_results", RECORDS/"int001-20260917/INT001_RESULTS.md", "text")
    load("mask_results", RECORDS/"sparse001-20260917/SPARSE001_RESULTS.md", "text")
    load("replication_results", RECORDS/"rep001-20260917/REP001_RESULTS.md", "text")
    protocol = load("prospective_protocol", ROOT.parent/"ChatBubble-rineya-pruning-retention-confirm-v1/research/rineya_pruning_retention_confirm_v1/PROTOCOL.md", "text", "preparation", "protocol")
    require("±.03 overall and ±.05 in each acquisition family" in protocol, "prospective acquisition margins changed")
    norms = {"mag25_L2_fraction": mask_arm["delta_norm_fraction_retained"],
             "mag50_L2_fraction": mask50["delta_norm_fraction_retained"],
             "scalar050_L2_fraction_ideal": .5, "scalar075_L2_fraction_ideal": .75,
             "bf16_L2_fraction_recorded": bf16["deployed_titan_effective_alpha"],
             "interpretation": "Coordinate share, L2 fraction, squared-L2 energy, direction and acquisition are distinct quantities."}
    norms["mag25_energy_fraction"] = norms["mag25_L2_fraction"]**2
    norms["mag50_energy_fraction"] = norms["mag50_L2_fraction"]**2
    result = {"schema": 1, "status": "EXPOSED_HISTORICAL_CAUSAL_CONTROL_AUDIT_NOT_CONFIRMED_GAIN",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "historical_points": records,
        "comparisons": comparisons, "norms": norms,
        "archival_source_weight_sha256": scalar_arm["source_weight_sha256"],
        "archival_seed17_donor_sha256": scalar_arm["trained_fp32_sha256"],
        "new_decision_evidence": [
            "A near-acquisition scalar comparison already exists at seed17: k25 versus scalar.75; it is a posthoc historical diagnostic, not fresh equivalence qualification.",
            "Seed23 aggregate acquisition differs by only1 case while ledger loses13/128, so totals can hide family-specific acquisition damage.",
            "Equal aggregate acquisition235/384 for full and k50 coexists with one fewer original source success retained; pruning advantage is not monotone/universal.",
            "Magnitude/scalar coordinate or nominal-coefficient matching does not match update norm, direction or effective task-specific dose."],
        "next_test": "Keep prospective acquisition-selected scalar calibration with every family-equivalence requirement; do not add a retrospective alpha search or infer recipe superiority from one old donor.",
        "limitations": ["All outcomes are exposed historical source reports and case flags, not new native measurements.",
            "Empirical bootstrap PMFs are deterministic computations, not exact frequentist coverage or independent confirmation.",
            "Retention intervals resample pooled original-source cases, not the current registered family/answerability macro endpoint.",
            "No scalar arms were evaluated for replication draws23/29; scalar superiority comparisons remain one donor.",
            "Historical HF retention receipts retain pre-load source/candidate/sidecar file identities; no loaded-state or post-load digest is captured, and acquisition/long-context no-checkpoint paths have weaker weight identity.",
            "Long-context/abstention remain report-only; no winner is selected on them."],
        "activity": {"native_loads": 0, "native_evaluations": 0, "native_updates": 0, "protected_bank_reads": 0},
        "full_goal_achieved": False, "native_execution_authorized": False, "promotion_permitted": False}
    return result, snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = external_output(args.output)
    result, snapshots = build_audit()
    output.mkdir(parents=True, exist_ok=False)
    bindings = []
    for key, snap in snapshots.items():
        target = output/"evidence"/(key+Path(snap["binding"]["path"]).suffix)
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(snap["raw"])
        bindings.append(dict(snap["binding"], capsule=target.relative_to(output).as_posix()))
    result["source_bindings"] = bindings
    write_new(output/"ATTENUATION_AUDIT.json", result)
    write_new(output/"MANIFEST.json", {"files": {path.relative_to(output).as_posix(): sha_file(path)
                                               for path in output.rglob("*") if path.is_file()}})
    near = result["comparisons"]["near_scalar075_mag25"]
    print(json.dumps({"status": result["status"], "historical_points": len(result["historical_points"]),
        "sources": len(bindings), "near_match_acquisition_ci90": near["fixed_family_macro_empirical_ci90"],
        "near_match_retention": near["source_retention"], "full_goal_achieved": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
