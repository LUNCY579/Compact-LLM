"""Audit completed candidate runs after an isolated excluded-control failure.

Never resume or relabel the failed campaign. Preserve individual frozen gates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
from armr.provenance import HISTORY, sha256, write_json
from armr.reporting import csv_file, screen_report, summarize


def audit(run: Path, diagnosis: Path) -> Path:
    run, diagnosis = run.resolve(), diagnosis.resolve()
    if not run.is_relative_to((ROOT/"evidence/01_middle_reapplication_screen").resolve()) or not diagnosis.is_relative_to((ROOT/"evidence/01b_late_failure_diagnosis").resolve()):
        raise ValueError("unexpected input evidence paths")
    m=json.loads((run/"run_manifest.json").read_text())
    dm=json.loads((diagnosis/"run_manifest.json").read_text())
    d=json.loads((diagnosis/"diagnosis.json").read_text())
    config=json.loads((ROOT/"configs/reapplication_screen.json").read_text())
    if m["status"] != "FAIL" or m.get("error") != "SafetyError('hidden norm emergency bound')":
        raise ValueError("only the documented isolated norm failure is supported")
    if not m["canonical_unchanged"] or dm["status"] != "PASS" or not all(d["checks"].values()):
        raise ValueError("canonical/restoration diagnosis failed")
    if m["config"] != config or dm["config"] != config or d["layer"] != 27 or d["applications"] != 2:
        raise ValueError("configuration/control changed")
    if m["state_hash_before"] != dm["state_hash_before"] or m["checkpoint_files_before"] != dm["checkpoint_files_before"]:
        raise ValueError("diagnosis checkpoint changed")
    source=[json.loads(s) for s in (run/"raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped=defaultdict(list)
    for row in source:
        grouped[row["arm"]].append(row)
    baseline=grouped["intact"]
    expected={r["case_id"] for r in baseline}
    results, hidden_rows, logits_rows, coverage=[],[],[],[]
    for layer in config["candidate_layers"]:
        for apps in config["applications"]:
            name=f"layer_{layer:02d}_apps_{apps}"
            rows=grouped.get(name,[])
            if layer == 27:
                if rows:
                    raise ValueError("unexpected completed L27 run")
                coverage.append({"layer":layer,"applications":apps,"completed_cases":0,
                                 "status":"FAIL_NORM_FIRST_PREFILL" if apps==2 else "NOT_RUN_AFTER_STOP"})
                continue
            if len(rows)!=8 or {r["case_id"] for r in rows}!=expected:
                raise ValueError(f"incomplete candidate/control {name}")
            if any(len(r["hidden"])!=apps for r in rows):
                raise ValueError("missing hidden recurrence records")
            result={"layer":layer,"applications":apps,**summarize(rows,baseline,config)}
            historical=json.loads((run/f"{name}_summary.json").read_text())
            if result != historical:
                raise ValueError("recomputed gates differ from archived receipt")
            results.append(result)
            coverage.append({"layer":layer,"applications":apps,"completed_cases":8,"status":"COMPLETE"})
            for r in rows:
                key={"layer":layer,"applications":apps,"case_id":r["case_id"]}
                hidden_rows.extend({**key,**h} for h in r["hidden"])
                logits_rows.append({**key,**{k:r[k] for k in ("logit_cosine","kl_base_to_arm","top1_agreement","topk_overlap","margin_change","gold_rank","baseline_gold_rank","gold_nll")}})
    out=ROOT/"evidence/01c_completed_candidate_atlas"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True,exist_ok=False)
    note=("The campaign stopped at excluded late control L27; it remains FAIL and is not resumed. "
          "All ten prespecified middle layers (both depths, eight cases each) and L25 completed before the stop. "
          "The isolated diagnostic reproduced a 12.1768x RMS ratio under unchanged bounds and verified bit-identical restoration. "
          "These completed per-candidate gates are recomputed exactly from the immutable raw evidence; no threshold, "
          "candidate, score, or selection rule changed. The original per-candidate protocol licenses fixed development "
          "qualification for passing middle arms, independently of the already-excluded late control. "
          "This is an explicit scope decision after diagnosis, not a passing label for the failed campaign.")
    decision=screen_report(out,results,json.loads((HISTORY/"ranking.json").read_text()),config,
                           campaign_status="FAIL_L27_CONTROL",qualification_note=note)
    csv_file(out/"coverage.csv",coverage)
    csv_file(out/"hidden_state_metrics.csv",hidden_rows)
    csv_file(out/"logit_metrics.csv",logits_rows)
    with (out/"raw_results.jsonl").open("xb") as f:
        f.write((run/"raw_results.jsonl").read_bytes())
    write_json(out/"failed_late_control.json",d)
    with (out/"ranking.md").open("a",encoding="utf-8") as f:
        f.write("\nL27 repeat-2: stopped at the first prefill (RMS ratio 12.1768; hidden cosine -0.1663). "
                "No downstream logits or generation collected. L27 repeat-3: NOT RUN after safety stop. "
                "These missing observations are not imputed as zero scores.\n")
    write_json(out/"run_manifest.json",{
        "status":decision["status"],"kind":"derived_completed_candidate_gate","config":config,
        "source_run":str(run),"source_manifest_sha256":sha256(run/"run_manifest.json"),
        "source_raw_sha256":sha256(run/"raw_results.jsonl"),
        "source_summary_hashes":{p.name:sha256(p) for p in run.glob("*summary.json")},
        "diagnosis":str(diagnosis),"diagnosis_manifest_sha256":sha256(diagnosis/"run_manifest.json"),
        "canonical_unchanged":True,"decision":decision,"model_loaded":False,
        "script_sha256":sha256(Path(__file__)),"scope_decision":note})
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("diagnosis",type=Path)
    args=parser.parse_args()
    print(audit(args.run,args.diagnosis))
