"""Reviewed, terminal-only exposed DEV response selection; never loads a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .. import bulk_bank
from .opposed_gate_policy import POLICIES, GUARD, decide, header_cue, matched_random_decisions
from .probe_analysis import validate_export, READ_BODY, TASK_CONTEXT, POP_KEYS

V3 = Path(__file__).resolve().parent
CAMPAIGN = V3.parent
ROOT = CAMPAIGN.parents[1]
STATE = "b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551"
PINS = {
    "quality_pattern_v3/data/dev_prefix_probe_001/DEVELOPMENT_BANK.json": "681a3e77e3d797e2faebcff3971dbb365a4479ebdac017bf3dcfcb48081ba03f",
    "quality_pattern_v3/data/dev_prefix_probe_001/DEVELOPMENT_TOKEN_BINDING.json": "c34948026ebaf332111ce635a473b1489ef0d2f3932100f9e0d3cdd6a08ce6d2",
    "quality_pattern_v3/runs/native-prefix-probe-001/result/readouts.jsonl": "b61150bb60b4721c97fa2750b29d1b4d023ec7d0edd5531a4d70d596efbec652",
    "quality_pattern_v3/runs/native-prefix-probe-001/terminal_receipt.json": "ef292863a3c6050175fac115715546cc542343023c430ccaed5ed2699f493f48",
    "quality_pattern_v3/runs/native-prefix-probe-001/result/receipt.json": "b0d2dede48274ca7f4e7a4e5969aa4ab6bc3f230648f27b3c0c79695a1b78817",
    "quality_pattern_v3/runs/native-prefix-probe-001/result/qualification.json": "a4282d8bd76d018323c2b1701e07b705bd2e1245a82dd3fc02733eb06c1f62b8",
    "quality_pattern_v3/reviews/probe_execution_review_001.json": "fffb9d699b5f4458e73cbdae8dd192b34e4fed17e80ae7e7f705b2eab386f99e",
    "quality_pattern_v3/analyses/prefix-probe-001/REPORT.json": "6bf9dd67f2f6f2b4a5bcaec0738b84058212cd28883ecda237a309532fd930c7",
    "runs/bulk-bank-apparatus-001/bank.json": "862a63648caf7efb3d287c11bc5caaefc884e0ec6cb3130ce8f196526efc40cd",
    "runs/native-bulk-restart-001/result/outcomes.jsonl": "858ba6c1b5105b460e600c93872575c78e2c3e43545538847e3d9eba30e01cd8",
    "runs/native-bulk-restart-001/terminal_receipt.json": "dc4d8c7a5895e846aee3c1b1b19b9e4f4909386779b313aa5693599ec9f2ded6",
    "runs/native-bulk-restart-001/result/receipt.json": "d9c34a822d9bdaaac39d33820eaacf05d083974cd9191e43559307bd2258ff19",
    "bulk_bank.py": "3242439231064fc81991e157c0da62833f4e4223ec71d60cf9ffd2851815b5eb",
    "open_bank.py": "25db8ad7b8bd4166188eccab4be73c9a430f696610dea5b8c83b380883f9d42a",
    "quality_pattern_v3/probe_analysis.py": "2d6c8e598df0a0caab29875ff95cecfb6b7eef1e8eff595efabb3322ebf70a27",
    "quality_pattern_v3/OPPOSED_GATE_CONTRACT_DRAFT_002.md": "067426a54cc2836b558401ee528dfd2f63008297d91ae41bb95f3e8491e734cd",
    "quality_pattern_v3/reviews/opposed_gate_design_review_001.json": "e613a48955a639bc1e873ccd8fb1f19e1f7293da88664f1b918f12aa589f5d26",
    "quality_pattern_v3/reviews/opposed_replay_input_review_001.json": "7c356efb50e56b2047e43645ce914472adb4dc7cda934cd4581b247523ce661b",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((CAMPAIGN / relative).read_text(encoding="utf-8"))


def read_lines(relative):
    with (CAMPAIGN / relative).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def decisions_before_labels(population, readouts):
    """The only selection phase: prompt/p input objects, no scored histories."""
    if any(set(row) != POP_KEYS for row in population):
        raise ValueError("exact prompt-only decision population required")
    ids = [row["case_id"] for row in population]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt population")
    by = {row["case_id"]: row for row in population}
    indexed = {}
    for row in readouts:
        key = row["case_id"], row["mode"]
        if (set(row) != READ_BODY | TASK_CONTEXT or key in indexed or row["case_id"] not in by
                or row["phase"] != "TASK" or row["split"] != "DEVELOPMENT"
                or row["mode"] not in ("OBSERVED_K0", "OBSERVED_UNIT")
                or row["schema"] != "PASSIVE_INPUT_FEATURES_V1"
                or row["vocabulary_rows"] != 151936 or row["UNKNOWN_token_id"] != 51378
                or row["finite"] is not True or row["target_independent"] is not True
                or row["teacher_conditioned"] is not False or row["preceding_gold_tokens"] != 0
                or row["prefix_kind"] != "ORIGINAL_INPUT_ONLY" or row["logits_changed"] is not False):
            raise ValueError("exact target-free original-input readout differs")
        ref = by[row["case_id"]]
        if any(row[k] != ref[k] for k in ("prompt_sha256", "rendered_prompt_sha256", "prefix_ids_sha256")):
            raise ValueError("readout/prompt/token binding differs")
        indexed[key] = row
    expected = {(case_id, mode) for case_id in ids for mode in ("OBSERVED_K0", "OBSERVED_UNIT")}
    if set(indexed) != expected:
        raise ValueError("complete two-mode readouts required")
    decisions = {policy: [] for policy in POLICIES}
    header = []
    for ref in population:
        p = indexed[ref["case_id"], "OBSERVED_K0"]["top1_probability"]
        for policy in POLICIES:
            choice = decide(policy, ref["prompt"], None if policy == GUARD else p)
            decisions[policy].append(asdict(choice))
        header.append(asdict(header_cue(ref["prompt"])))
    rates = {policy: sum(row["k"] for row in rows) / len(ids) for policy, rows in decisions.items()}
    randoms = {policy: {str(seed): matched_random_decisions(rate, len(ids), seed) for seed in (17, 29)}
               for policy, rate in rates.items()}
    return {"case_order": ids, "decisions": decisions, "header": header,
            "rates": rates, "randoms": randoms, "label_phase_started": False}, indexed


def grade_history(population, raw_bank, history, readouts):
    """Labels arrive only after frozen decisions; VAL/aliases are never graded."""
    ids = {row["case_id"] for row in population}
    labels = [row for row in raw_bank if row["split"] == "DEVELOPMENT"]
    cases = {row["case_id"]: row for row in labels}
    if len(cases) != len(labels) or set(cases) != ids:
        raise ValueError("exact DEV label membership differs")
    physical = {arm: {} for arm in ("A0", "A1", "FIXED1")}
    for row in history:
        if row["split"] != "DEVELOPMENT":
            continue
        arm = row["arm"] if row["arm"] in ("A0", "A1") else (
            "FIXED1" if row["arm"] == "A2" and row["fixed_alpha"] == 1. else None)
        if arm is None:
            continue
        if (row["case_id"] not in ids or row["case_id"] in physical[arm]
                or row["physical_run"] is not True or row["configured_seed"] != 17 or row["seed"] != 17
                or row["extra_loops"] != int(arm != "A0") or row["safety_stopped"] is not False):
            raise ValueError("DEV physical baseline membership/failure differs")
        tokens = row["token_ids"]
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= 8
                or row["output_tokens"] != len(tokens) or row["full_model_calls_including_gold_nll"] != len(tokens)
                or any(type(t) is not int or not 0 <= t < 151936 for t in tokens)
                or row["termination"] not in ("eos", "token_cap")
                or row["eos_emitted"] is not (row["termination"] == "eos")
                or row["truncated"] is not (row["termination"] == "token_cap")
                or (row["eos_emitted"] and tokens[-1] not in (151643, 151645))
                or (row["truncated"] and len(tokens) != 8)):
            raise ValueError("EOS-inclusive generated length/termination differs")
        case = cases[row["case_id"]]
        score = bulk_bank.score_response(row["response"], case)
        if float(score["score"]) != row["score"]:
            raise ValueError("independent strict scorer disagrees with history")
        physical[arm][row["case_id"]] = {**row, **score}
    if any(set(rows) != ids for rows in physical.values()):
        raise ValueError("complete intact/unit/fixed1 required")
    for ref in population:
        case_id = ref["case_id"]
        if ref["prompt"] != cases[case_id]["prompt"]:
            raise ValueError("original user-content differs from history bank")
        if any(physical["A1"][case_id][k] != physical["FIXED1"][case_id][k]
               for k in ("response", "token_ids", "score", "termination", "eos_emitted", "truncated")):
            raise ValueError("unit/fixed1 response identity differs")
        for arm, mode in (("A0", "OBSERVED_K0"), ("A1", "OBSERVED_UNIT")):
            if physical[arm][case_id]["token_ids"][0] != readouts[case_id, mode]["top1_token_id"]:
                raise ValueError("passive/historical first-token identity differs")
    return cases, physical


def summarize(cases, rows):
    families = defaultdict(lambda: defaultdict(list))
    for case_id, case in cases.items():
        families[case["category"]][case["parent_id"]].append(float(rows[case_id]["correct"]))
    family_q = {name: sum(sum(values) / len(values) for values in parents.values()) / len(parents)
                for name, parents in families.items()}
    endpoints = ("correct", "format_valid", "literal_UNKNOWN", "false_UNKNOWN", "unsupported_canonical_answer",
                 "eos_emitted", "truncated")
    return {"cases": len(cases), "parent_units": len({c["parent_id"] for c in cases.values()}),
            **{name: sum(float(row[name]) for row in rows.values()) for name in endpoints},
            "equal_family_parent_accuracy": sum(family_q.values()) / len(family_q),
            "family_accuracy": family_q,
            "answerable_cases": sum(c["gold"] != "UNKNOWN" for c in cases.values()),
            "unanswerable_cases": sum(c["gold"] == "UNKNOWN" for c in cases.values())}


def comparison(cases, candidate, baseline):
    gains = [i for i in cases if candidate[i]["correct"] and not baseline[i]["correct"]]
    losses = [i for i in cases if baseline[i]["correct"] and not candidate[i]["correct"]]
    c, b = summarize(cases, candidate), summarize(cases, baseline)
    return {"gains": gains, "losses": losses, "gain_parents": sorted({cases[i]["parent_id"] for i in gains}),
            "loss_parents": sorted({cases[i]["parent_id"] for i in losses}),
            "retained_baseline_correct": b["correct"] - len(losses), "baseline_correct": b["correct"],
            "delta": c["equal_family_parent_accuracy"] - b["equal_family_parent_accuracy"],
            "family_delta": {k: c["family_accuracy"][k] - b["family_accuracy"][k] for k in b["family_accuracy"]},
            "format_delta": c["format_valid"] - b["format_valid"],
            "unsupported_delta": c["unsupported_canonical_answer"] - b["unsupported_canonical_answer"],
            "false_UNKNOWN_delta": c["false_UNKNOWN"] - b["false_UNKNOWN"]}


def nominate(policy_reports, unit_score, header_score):
    eligible = []
    for policy in POLICIES:
        entry = policy_reports[policy]
        contrast = entry["vs_unit"]
        checks = {"strictly_better_than_unit": entry["summary"]["equal_family_parent_accuracy"] > unit_score,
                  "zero_unit_correct_losses": not contrast["losses"],
                  "no_family_decline": all(x >= 0 for x in contrast["family_delta"].values()),
                  "no_format_decline": contrast["format_delta"] >= 0,
                  "no_unsupported_increase": contrast["unsupported_delta"] <= 0,
                  "no_false_UNKNOWN_increase": contrast["false_UNKNOWN_delta"] <= 0,
                  "strictly_better_than_random": entry["summary"]["equal_family_parent_accuracy"] > entry["random_mean"]["equal_family_parent_accuracy"],
                  "guard_beats_header": policy != GUARD or entry["summary"]["equal_family_parent_accuracy"] > header_score}
        entry["mechanical_nomination_checks"] = checks
        entry["mechanical_nomination_eligible"] = all(checks.values())
        if all(checks.values()):
            eligible.append(policy)
    return max(eligible, key=lambda p: (policy_reports[p]["summary"]["equal_family_parent_accuracy"], -POLICIES.index(p))) if eligible else None


def analyze_frozen_selection(selection, population, cases, physical):
    ids = selection["case_order"]
    def choose(bits):
        return {i: physical["A1" if k else "A0"][i] for i, k in zip(ids, bits, strict=True)}
    header_rows = choose(row["k"] for row in selection["header"])
    baseline = {name: summarize(cases, physical[arm]) for name, arm in (("intact", "A0"), ("unit", "A1"))}
    header_summary = summarize(cases, header_rows)
    reports = {}
    for policy in POLICIES:
        selected = choose(row["k"] for row in selection["decisions"][policy])
        random_rows = {seed: choose(bits) for seed, bits in selection["randoms"][policy].items()}
        averaged = {i: {name: sum(float(random_rows[seed][i][name]) for seed in ("17", "29")) / 2.
                       for name in ("correct", "format_valid", "literal_UNKNOWN", "false_UNKNOWN",
                                    "unsupported_canonical_answer", "eos_emitted", "truncated")} for i in ids}
        summary = summarize(cases, selected)
        contrast = comparison(cases, selected, physical["A1"])
        reports[policy] = {"summary": summary, "vs_unit": contrast,
                           "vs_intact": comparison(cases, selected, physical["A0"]),
                           "K1_rate": selection["rates"][policy], "K1_count": sum(r["k"] for r in selection["decisions"][policy]),
                           "random_seed_summaries": {seed: summarize(cases, rows) for seed, rows in random_rows.items()},
                           "random_K1_counts": {seed: sum(bits) for seed, bits in selection["randoms"][policy].items()},
                           "random_mean": summarize(cases, averaged),
                           "useful_five_point_DEV_gain": False}
    nomination = nominate(reports, baseline["unit"]["equal_family_parent_accuracy"], header_summary["equal_family_parent_accuracy"])
    certificate_confusion = Counter()
    recognized = 0
    for ref, decision in zip(population, selection["decisions"][GUARD], strict=True):
        if not decision["recognized"]:
            continue
        recognized += 1
        _, status = bulk_bank.parse_prompt(ref["prompt"])
        reference = {"MISSING": "ABSENT", "CONFLICT": "CONFLICT", "ANSWER": "CONSISTENT"}[status]
        certificate_confusion[reference + "->" + decision["certificate"]] += 1
    # Reference solver and structural floors are separate ordinary controls.
    floor_rows = defaultdict(dict)
    for i, case in cases.items():
        predictions = {"positive_prompt_solver": bulk_bank.solve_prompt(case["prompt"]), **bulk_bank.blind_predictions(case["prompt"])}
        for name, response in predictions.items():
            floor_rows[name][i] = {**bulk_bank.score_response(response, case), "eos_emitted": False, "truncated": False}
    return {"baselines": baseline, "header_cue": header_summary, "policies": reports,
            "mechanical_nominee": nomination, "nomination_is_efficacy": False,
            "guard_certificate_classifier": {"recognized": recognized, "unrecognized": len(ids) - recognized,
                "coverage": recognized / len(ids), "confusion": dict(sorted(certificate_confusion.items())),
                "false_certificates": sum(n for pair, n in certificate_confusion.items() if len(set(pair.split("->"))) != 1),
                "reference": "Independently implemented historical bulk_bank.parse_prompt(raw prompt)"},
            "ordinary_control_scores": {name: {**summarize(cases, rows), "eos_emitted": None,
                "truncated": None, "generation_status": "NOT_APPLICABLE_SYMBOLIC_CONTROL"}
                for name, rows in floor_rows.items()},
            "binary_replay_oracle_correct": sum(physical["A0"][i]["correct"] or physical["A1"][i]["correct"] for i in ids),
            "practical_margin": .05, "useful_DEV_superiority_possible": False,
            "quality_gain_established": False, "native_model_calls": 0,
            "limitations": ["Exposed DEV response selection; no new generation or latency measured.",
                            "Useful five-point replay benefit is impossible under the finite binary oracle.",
                            "Mechanical nomination does not establish A0/source retention or independent source benefit.",
                            "Symbolic classifier/solver scores are separate from neural reasoning."]}


def verify_admission(path):
    if path.resolve() != (V3 / "opposed_replay_admission_v1.json").resolve():
        raise PermissionError("exact V3 replay admission required")
    admission = json.loads(path.read_text(encoding="utf-8"))
    if (admission.get("status") != "FROZEN_EXPOSED_DEV_REPLAY_SOURCE"
            or admission.get("input_pins") != PINS or admission.get("candidate_revisions_used") != 3):
        raise PermissionError("replay source/data/candidate freeze differs")
    required = {"opposed_gate_policy.py", "opposed_gate_replay.py", "tests/test_opposed_gate_policy.py",
                "tests/test_opposed_gate_replay.py", "OPPOSED_REPLAY_ANALYSIS_PLAN_001.md"}
    hashes = admission.get("source_hashes", {})
    if set(hashes) != required or any(sha(V3 / name) != h for name, h in hashes.items()):
        raise PermissionError("actual replay source membership differs")
    for relative, expected in PINS.items():
        if sha(CAMPAIGN / relative) != expected:
            raise PermissionError("historical binding changed:" + relative)
    review = V3 / admission["review_receipt"]
    if V3 not in review.resolve().parents:
        raise PermissionError("review outside V3")
    receipt = json.loads(review.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_OPPOSED_EXPOSED_DEV_REPLAY_SOURCE" or receipt.get("admission_sha256") != sha(path):
        raise PermissionError("independent exact admission review required")
    terminal = read_json("quality_pattern_v3/runs/native-prefix-probe-001/terminal_receipt.json")
    worker = read_json("quality_pattern_v3/runs/native-prefix-probe-001/result/receipt.json")
    bulk_terminal = read_json("runs/native-bulk-restart-001/terminal_receipt.json")
    bulk_worker = read_json("runs/native-bulk-restart-001/result/receipt.json")
    if (terminal["status"] != "WORKER_TERMINAL" or terminal["exit_code"] != 0 or not worker["complete"]
            or worker["after_state_sha256"] != STATE or worker["before_state_sha256"] != STATE
            or bulk_terminal["status"] != "WORKER_TERMINAL" or bulk_terminal["exit_code"] != 0
            or bulk_worker["status"] != "BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY"
            or bulk_worker["after_state_sha256"] != STATE or bulk_worker["loaded_state_sha256"] != STATE):
        raise PermissionError("complete terminal/state bindings required")
    return admission


def run(admission_path, out):
    out = out.resolve()
    if V3 not in out.parents or out.exists():
        raise PermissionError("exclusive new V3 replay output required")
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "INVALID_EXPOSED_DEV_REPLAY", "errors": [], "native_model_calls": 0}
    try:
        admission = verify_admission(admission_path)
        raw = read_json("quality_pattern_v3/data/dev_prefix_probe_001/DEVELOPMENT_BANK.json")
        packet = read_json("quality_pattern_v3/data/dev_prefix_probe_001/DEVELOPMENT_TOKEN_BINDING.json")
        errors, population = validate_export(raw, packet["rows"])
        if errors:
            raise ValueError(";".join(errors))
        selection, indexed = decisions_before_labels(population, read_lines("quality_pattern_v3/runs/native-prefix-probe-001/result/readouts.jsonl"))
        (out / "decisions_before_labels.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        bank = read_json("runs/bulk-bank-apparatus-001/bank.json")
        cases, physical = grade_history(population, bank, read_lines("runs/native-bulk-restart-001/result/outcomes.jsonl"), indexed)
        if len(cases) != 192 or len({c["parent_id"] for c in cases.values()}) != 96 or Counter(c["category"] for c in cases.values()) != Counter({name: 32 for name in bulk_bank.CATEGORIES}):
            raise ValueError("192cases/96parents/equal-six-family membership differs")
        parent_groups = defaultdict(list)
        for case in cases.values():
            parent_groups[case["parent_id"]].append(case)
        if (any(len(group) != 2 or {c["variant_id"] for c in group} != {0, 1}
                or len({c["category"] for c in group}) != 1 for group in parent_groups.values())
                or Counter(group[0]["category"] for group in parent_groups.values()) != Counter({name: 16 for name in bulk_bank.CATEGORIES})):
            raise ValueError("paired parent/variant/family units differ")
        findings = analyze_frozen_selection(selection, population, cases, physical)
        if findings["guard_certificate_classifier"]["false_certificates"]:
            raise ValueError("independent certificate verifier disagrees")
        report.update(status="COMPLETE_EXPOSED_DEV_POLICY_REPLAY", **findings,
                      admission_sha256=sha(admission_path), source_hashes=admission["source_hashes"], input_pins=PINS,
                      decisions_sha256=sha(out / "decisions_before_labels.json"), timestamp_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report["errors"].append(type(error).__name__ + ": " + str(error))
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": report["errors"], "mechanical_nominee": report.get("mechanical_nominee")}))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.admission, args.out))
