"""Offline recount of already exposed historical behavior outputs; no model imports.

Run from the research worktree. Default prints JSON; --output creates a new file.
It never writes historical records, opens fresh banks, or qualifies a runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unicodedata

sys.dont_write_bytecode = True
RECORDS = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
HERE = Path(__file__).resolve().parent
PINS = {}


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    PINS[str(path)] = value
    return value


def read(path):
    path = Path(path)
    sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    path = Path(path)
    sha(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def pair(reference, candidate):
    assert reference.keys() == candidate.keys() and reference
    wins = [p for p in reference if not reference[p] and candidate[p]]
    losses = [p for p in reference if reference[p] and not candidate[p]]
    nd = len(wins) + len(losses)
    p = min(1, 2 * sum(math.comb(nd, k) for k in range(min(len(wins), len(losses)) + 1)) / 2**nd)
    return dict(n=len(reference), wins=len(wins), losses=len(losses), net=len(wins)-len(losses),
                nd=nd, exact_two_sided_p=p, win_ids=wins, loss_ids=losses)


def holm_two(contrasts):
    last = 0
    for rank, (_, row) in enumerate(sorted(contrasts.items(), key=lambda item: item[1]["exact_two_sided_p"])):
        last = max(last, min(1, (2-rank) * row["exact_two_sided_p"]))
        row["holm2_p"] = last


def binding(generations, gold):
    counts, groups, false_ids = Counter(), defaultdict(list), []
    assert len(generations["predictions"]) == len(gold) == 384
    assert {p["id"] for p in generations["predictions"]} == set(gold)
    for row in generations["predictions"]:
        target = gold[row["id"]]
        assert row["parent_id"] == target["parent_id"]
        correct = row["terminated"] and norm(row["prediction"]) == norm(target["reference"])
        counts[target["category"]] += correct
        groups[target["category"], target["parent_id"]].append(correct)
        if target["category"] in ("person_answerable", "role_answerable") and norm(row["prediction"]) == "unknown":
            false_ids.append(row["id"])
    worlds = {}
    for (category, parent), values in groups.items():
        assert len(values) == 3
        worlds.setdefault(category, {})[parent] = all(values)
    assert all(len(values) == 32 for values in worlds.values())
    return dict(rows_correct=dict(counts), complete_worlds={k:sum(v.values()) for k,v in worlds.items()},
                false_unknown=len(false_ids), false_unknown_ids=false_ids,
                terminated=sum(p["terminated"] for p in generations["predictions"])), worlds


def historical_typed_parser():
    # This is an exact byte copy, not permission to relocate or execute a native worker.
    path = Path("C:/Users/clclc/BusinessProject/ChatBubble-small-vocab-depth/experiments/architecture_benchmark_001/retention.py")
    assert sha(path) == "c52b6a48cb811e3c272793a66765e82152fe170e8307eea153c3b1b6c5c58322"
    spec = importlib.util.spec_from_file_location("historical_typed_recount", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert not ({"torch", "transformers", "tokenizers", "safetensors"} & sys.modules.keys())
    return module


def retention(raw, typed, predictions):
    reference_path = RECORDS/"rescued-compose-lane-outputs-20260922/retention_B28_a329eb7b.json"
    assert sha(reference_path) == "ec3fca1e8b1eb418a6b848dadfea06235cf7c4317ddf6930061c84147afa9a8f"
    reference = read(reference_path)
    candidate = {p["id"]:p for p in predictions["predictions"]}
    assert len(candidate) == len(predictions["predictions"]) == 150
    assert candidate.keys() == {r["id"] for r in reference["rows"]}
    samples = [dict(id=r["id"],target=r["target"],source=r["source"]["raw"],
                    source_terminated=r["source"]["terminated"],arm=candidate[r["id"]]["text"],
                    arm_terminated=candidate[r["id"]]["terminated"]) for r in reference["rows"]]
    assert len(samples) == 150 and len({r["id"] for r in samples}) == 150
    result = {}
    source_maps = {}
    for policy in ("permissive", "strict"):
        refs, predictions = {}, {}
        for row in samples:
            target = dict(row["target"])
            if target["kind"] not in ("clock", "duration", "number"):
                target["bare_allowed"] = policy == "permissive"
            refs[row["id"]] = typed.score_prediction(row["source"], row["source_terminated"], target)["correct"]
            predictions[row["id"]] = typed.score_prediction(row["arm"], row["arm_terminated"], target)["correct"]
        assert sum(refs.values()) == raw["frozen_reference_sizes"][policy]
        assert {p for p,v in refs.items() if v} == set(reference["reference_ids_by_policy"][policy])
        assert sum(predictions.values()) == raw["by_bare_policy"][policy]["arm_correct_total"]
        cohort = [p for p,v in refs.items() if v]
        result[policy] = dict(panel_correct=sum(predictions.values()), source_reference_n=len(cohort),
                              retained=sum(predictions[p] for p in cohort),
                              lost_ids=[p for p in cohort if not predictions[p]])
        source_maps[policy] = refs
    fixed47 = [p for p,v in source_maps["permissive"].items() if v]
    strict = {}
    for row in samples:
        target = dict(row["target"])
        if target["kind"] not in ("clock", "duration", "number"):
            target["bare_allowed"] = False
        strict[row["id"]] = typed.score_prediction(row["arm"],row["arm_terminated"],target)["correct"]
    result["strict_on_fixed47"] = sum(strict[p] for p in fixed47)
    return result


def recount_integration():
    root = RECORDS / "binding-integration001-20260920"
    legs = ("dose", "select", "retention", "long_context", "binding", "diagnostic")
    terminal = read(root / "posttrain/TERMINAL.json")
    dispatch = read(root / "posttrain/DISPATCH_RESERVATION.json")
    freeze = read(root / "PRELAUNCH_BINDING.json")
    assert terminal["status"] == "FAILED_OR_BLOCKED"
    assert terminal["failure"] == {"type":"ValueError", "message":"six-probe attempt receipt inventory differs"}
    assert dispatch["legs"] == list(legs) and dispatch["automatic_retries"] == 0
    assert dispatch["owner"]["prelaunch_binding_sha256"] == sha(root / "PRELAUNCH_BINDING.json")
    expected_files = {"TERMINAL.json", "DISPATCH_RESERVATION.json", *("integrated-"+leg+".json" for leg in legs)}
    assert {p.name for p in (root/"posttrain").glob("*.json")} == expected_files
    arm = read(root / "arms/integrated/MATERIALIZATION.json")
    arm_sha = arm["arm_weight_sha256"]
    custody, raw, attempts = {}, {}, []
    rows, reserved_tokens, seconds = 0, 0, 0.0
    for leg in legs:
        out = root / "eval" / ("integrated-" + leg)
        attempt = read(root / "posttrain" / ("integrated-" + leg + ".json"))
        assert attempt in terminal["attempts"] and attempt["returncode"] == 0
        aux = leg in ("binding", "diagnostic")
        supervisor = read(out / ("supervisor.json" if aux else "worker/supervisor.json"))
        assert supervisor["status"] == "EXITED" and supervisor["returncode"] == 0
        assert abs(supervisor["wall_seconds"]-attempt["worker_seconds_charged"]) < 1e-6
        receipt_path = out / ("AUX_BINDING.json" if aux else "PROBE_WEIGHT_BINDING.json")
        receipt = read(receipt_path)
        assert receipt["status"] == "COMPLETE" and receipt["complete"] is True
        if aux:
            for filename, field in (("GENERATIONS.json","generations_sha256"),("supervisor.json","supervisor_sha256"),
                                    ("worker_terminal.json","worker_terminal_sha256"),("launch.json","launch_sha256"),
                                    ("spawn.json","spawn_sha256"),("worker_claim.json","worker_claim_sha256")):
                assert sha(out/filename) == receipt[field]
            assert receipt["controller_returncode"] == receipt["worker_returncode"] == 0
            assert receipt["before"] == receipt["after"]
            assert receipt["before"]["weight_sha256"] == arm_sha
            raw[leg] = read(out / "GENERATIONS.json")
        else:
            assert sha(out/"result.json") == receipt["raw_result_sha256"]
            assert sha(out/"worker/supervisor.json") == receipt["supervisor_sha256"]
            assert receipt["weight_sha256_before"] == receipt["weight_sha256_after"] == arm_sha
            assert receipt["returncode"] == 0
            raw[leg] = read(out/"result.json")
            if leg == "retention":
                assert sha(out/"worker/retention_predictions.json") == receipt["retention_predictions_sha256"]
        custody[leg] = dict(receipt_sha256=sha(receipt_path), worker_rc=0, result_sha256=sha(out/("GENERATIONS.json" if aux else "result.json")))
        rows += attempt["reserved_rows"]; reserved_tokens += attempt["reserved_generation_tokens"]
        seconds += attempt["worker_seconds_charged"]; attempts.append(attempt)
    assert rows == 2454 and reserved_tokens == 138768 and len(attempts) == 6
    gold = {r["id"]:r for r in lines(RECORDS / "binding-signal-prep-20260920/candidate-v2/compiled-v1/DEV_GOLD.jsonl")}
    bind, worlds = binding(raw["binding"], gold)
    baseline = read(RECORDS/"binding-dev-baseline001-20260920/BASELINE_SCORES.json")
    contrasts = {c:pair(baseline["world_maps"]["lora"]["canonical_complete_exact"][c],worlds[c]) for c in ("person_missing","person_vacant")}
    holm_two(contrasts)
    typed = historical_typed_parser()
    ret = retention(raw["retention"], typed, read(root/"eval/integrated-retention/worker/retention_predictions.json"))
    # Pure historical scorer; no build_report(), worker verifier, model, or old-output writer runs.
    code = RECORDS/"binding-diagnostic001-20260920/code_before_inference"
    for name in ("score.py","common.py","arithmetic.py","binding.py"):
        sha(code/name)
    sys.path.insert(0,str(code))
    import score as diagnostic_score
    diagnostic_score.no_model_imports()
    diagroot = RECORDS/"binding-diagnostic001-20260920"
    inputs = lines(diagroot/"candidate-v1/DIAGNOSTIC_INPUTS.jsonl")
    diag_gold = {r["id"]:r for r in lines(diagroot/"candidate-v1/GOLD.jsonl")}
    endpoint, maps = diagnostic_score.score_rows(inputs,diag_gold,raw["diagnostic"]["predictions"],typed)
    old = read(diagroot/"RESULTS.json")
    cohort = old["source_conditioned_retention_descriptive"]["parent_ids"]
    assert len(cohort)==4 and old["source_conditioned_retention_descriptive"]["source_only_eligibility"]
    diagnostic = dict(all16=sum(maps["binding"]["canonical_complete_exact/all16"].values()),
        false_unknown=endpoint["totals"]["binding"]["false_exact_unknown"],
        false_unknown_by_template=dict(Counter(str(r["template"]) for r in endpoint["rows"].values() if r.get("false_exact_unknown"))),
        strict_anchors=sum(maps["arithmetic"]["strict_anchor"].values()),
        original_source_cohort_n=4, original_source_cohort_ids=cohort,
        original_source_cohort_retained=sum(maps["arithmetic"]["strict_anchor"][p] for p in cohort),
        original_source_cohort_status="INSUFFICIENT_REFERENCE_COHORT", totals=endpoint["totals"])
    for leg in ("dose","select"):
        preds=raw[leg]["predictions"]
        assert len(preds)==384 and len({p["id"] for p in preds})==384
        n=0
        for pred in preds:
            try: ok=json.dumps(json.loads(pred["prediction"]),sort_keys=True,separators=(",",":"))==pred["reference"]
            except (ValueError,TypeError): ok=False
            assert ok==pred["canonical"]
            n+=ok
        assert n==raw[leg]["aggregate"]["canonical_correct"]
    criteria=read(root/"CRITERIA.json")
    values=dict(dose=raw["dose"]["aggregate"]["canonical_correct"],select=raw["select"]["aggregate"]["canonical_correct"],
                references_retained=ret["permissive"]["retained"],arithmetic_panel=ret["permissive"]["panel_correct"],
                long_context_report_only=raw["long_context"]["all256_hits"],
                literal_unknown_report_only=raw["long_context"]["says_unknown_on_unanswerable"],
                false_unknown_on_positive=bind["false_unknown"],**bind["rows_correct"])
    bands={name:dict(observed=values[name],band=bounds,pass_=bounds[0]<=values[name]<=bounds[1]) for name,bounds in criteria["prediction_bands"].items()}
    rule=criteria["primary_engineering"]
    failures=[]
    for bank in ("dose","select"):
        if values[bank]<rule["each_acquisition_at_least"]:
            failures.append(f"{bank}{values[bank]}<{rule['each_acquisition_at_least']}")
    if values["references_retained"]<rule["fixed47_retained_at_least"]:
        failures.append(f"fixed47 retention{values['references_retained']}<{rule['fixed47_retained_at_least']}")
    if values["arithmetic_panel"]<rule["arithmetic150_at_least"]:
        failures.append(f"arithmetic{values['arithmetic_panel']}<{rule['arithmetic150_at_least']}")
    for name in (*bind["rows_correct"],"false_unknown_on_positive"):
        if not bands[name]["pass_"]:
            failures.append(f"{name}={values[name]} outside {bands[name]['band']}")
    engineering=not failures
    control_pass=all(bands[k]["pass_"] for k in ("person_answerable","role_answerable","false_unknown_on_positive"))
    efficacy=control_pass and any(v["net"]>0 and v["holm2_p"]<=.05 for v in contrasts.values())
    # Published historical regression expectations, separate from the derived verdict.
    assert values == dict(dose=294,select=297,references_retained=38,arithmetic_panel=71,
        long_context_report_only=166,literal_unknown_report_only=32,false_unknown_on_positive=16,
        person_answerable=76,role_answerable=93,person_missing=83,person_vacant=96)
    assert not engineering and not efficacy and len(failures)==2
    return dict(status="RETROSPECTIVE_OFFLINE_RECOUNT_ONLY",original_wrapper_status=terminal["status"],
        historical_accounting_complete_receipt_exists=(root/"posttrain/ACCOUNTING_COMPLETE.json").exists(),
        all_six_receipts_and_success_rcs_rechecked=True,custody=custody,
        accounting=dict(workers=6,retries=0,reserved_rows=rows,reserved_generation_tokens=reserved_tokens,worker_seconds=seconds),
        source_weight_sha256=freeze["source_assets_sha256"]["model.safetensors"],arm_weight_sha256=arm_sha,
        training_regime=dict(seed=17,updates=193,processed_tokens=577417,supervised_tokens=39502,
                             native_clock_tokens=380921,zero_addend_scope="TWO_UPDATES_ONLY"),
        endpoint_values=values,retention=ret,binding=bind,binding_world_contrasts_vs_lora=contrasts,diagnostic=diagnostic,bands=bands,
        reconstructed_engineering_verdict="PASS" if engineering else "FAILED_PREDICTION",reconstructed_binding_efficacy="ESTABLISHED" if efficacy else "NOT_ESTABLISHED",
        failure_reasons=failures,
        authorization_or_quality_claim=False,
        limitations=["Historical wrapper failure remains unchanged; no original COMPLETE_SIX_PROBES or ACCOUNTING_COMPLETE receipt exists.",
            "No native runtime requalification or model-weight reading; receipt hashes attest to saved outputs only.",
            "All historical scoring policies/cohorts preserved; old source-anchor N4 remains inadequate.",
            "Exposed DEV/selection/report-only banks are spent evidence, not unseen confirmation.",
            "Same native denominator/LR coefficients do not imply unchanged gradients under clipping/Adam."])


def build():
    integration=recount_integration()
    docs=("binding-mixture001-20260920/INTERPRETATION.md","binding-mixture001-20260920/RESULTS.json",
          "binding-diagnostic001-20260920/INTERPRETATION_v2.md","binding-diagnostic001-20260920/BINDING_ERROR_ANALYSIS.md",
          "binding-diagnostic001-20260920/NATIVE_COEFFICIENT_AUDIT.json","binding-integration001-20260920/ACCOUNTING_CORRECTION.md",
          "binding-integration001-20260920/PREREG.md","binding-integration001-20260920/DATA_USE_ADMISSION.json",
          "binding-integration001-20260920/TRAINING_AUDIT.json","binding-integration001-20260920/ZERO_ADDEND_AUDIT.json",
          "clockfix001-20260923/CLOCKFIX001_RESULT.md","clockfix001-20260923/C1_GATE_RESULT.md",
          "clockfix002-20260926/CLOCKFIX002_PREREG.md","abst001-20260925/ABST_RESULT_COLD_READ.md",
          "abst001-20260925/result/ABST_PRIMARY.json","abst001-20260925/result/ABST_H3.json",
          "hlcc001-20260925/HLCC_RESULT_COLD_READ.md","hlcc001-20260925/result/HLCC_RESULT.json",
          "doseparam001-20260922/DOSEPARAM001_RESULT.md","modcoef001-20260923/MODCOEF001_RESULT.md")
    for path in docs: sha(RECORDS/path)
    sha(Path(__file__))
    return dict(schema=1,record="HISTORICAL_BEHAVIOR_RECOUNT",model_calls=0,training_calls=0,
                fresh_or_sealed_banks_opened=False,quality_claim=False,release_permitted=False,
                integration=integration,evidence_files_sha256=dict(sorted(PINS.items())))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=build()
    assert not ({"torch","transformers","tokenizers","safetensors"}&sys.modules.keys())
    text=json.dumps(result,indent=2,ensure_ascii=False)+"\n"
    if args.output:
        assert args.output.resolve().is_relative_to(HERE.resolve())
        with args.output.open("x",encoding="utf-8",newline="\n") as stream: stream.write(text)
        print(json.dumps(dict(output=str(args.output.resolve()),sha256=sha(args.output),status=result["integration"]["status"])))
    else: print(text)
