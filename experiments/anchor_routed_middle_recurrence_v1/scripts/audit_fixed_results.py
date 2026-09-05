"""Conservative quality audit: censored intact controls cannot license routing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parents[1]/"Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
from armr.provenance import sha256, write_json
from run_fixed_recurrence import development_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    run=args.run.resolve()
    if not run.is_relative_to((ROOT/"evidence/02_fixed_recurrence").resolve()):
        raise ValueError("only fixed recurrence evidence may be audited")
    manifest=json.loads((run/"run_manifest.json").read_text())
    base=json.loads((run/"outputs/A0_dev.json").read_text(encoding="utf-8"))
    results={}
    coverage={}
    for path in sorted((run/"outputs").glob("A*_dev.json")):
        rows=json.loads(path.read_text(encoding="utf-8"))
        complete=[r for r in rows if not r["safety_stopped"]]
        coverage[path.stem]={"completed_cases":len(complete),"planned_cases":len(rows),
                             "correct_among_completed":sum(r["score"] for r in complete),
                             "censored_cases":[{"case_id":r["case_id"],"family":r["family"],"reason":r["termination"],
                                                "seconds":r["wall_seconds"]} for r in rows if r["safety_stopped"]}
        if path.stem!="A0_dev":
            results[path.stem]=development_result(rows,base)
    intact_censored=bool(coverage["A0_dev"]["censored_cases"])
    router=not intact_censored and manifest["canonical_unchanged"] and any(r["router_scaffold_licensed"] for r in results.values())
    status="INCONCLUSIVE" if intact_censored else "PASS" if router else "FAIL"
    out=ROOT/"evidence/02b_fixed_quality_audit"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True,exist_ok=False)
    decision={"status":status,"adaptive_routing":router,"peft_recovery":False,"sealed_evaluation":False,
              "reason":"intact development baseline censored by the frozen time budget" if intact_censored else "completed fixed development gates applied",
              "coverage":coverage,"comparisons":results,"canonical_unchanged":manifest["canonical_unchanged"]}
    write_json(out/"decision.json",decision)
    write_json(out/"source_manifest.json",{"source_run":str(run),"source_manifest_sha256":sha256(run/"run_manifest.json"),
                "source_outputs_sha256":{p.name:sha256(p) for p in (run/"outputs").glob("*.json")},
                "script_sha256":sha256(Path(__file__)),"model_loaded":False})
    table="\n".join(f"| {arm} | {r['completed_cases']}/{r['planned_cases']} | {r['correct_among_completed']:.0f} | {len(r['censored_cases'])} |" for arm,r in coverage.items())
    (out/"report.md").write_text(f"""# Fixed recurrence quality audit

Status: {status}.

Question: are the fixed-arm quality comparisons interpretable when the intact
baseline encounters the runtime stopping rule? This is an offline audit of the
immutable 16-case MERGEDEV-fast run; no model, new dataset, or retry was used.

| Arm | Completed / planned | Correct among completed | Censored by safety stop |
|---|---:|---:|---:|
{table}

Interpretation: {decision['reason']}. Censored baseline cases cannot be treated
as demonstrated language failures or credited as recurrence improvements.
The original run's raw scores and FAIL/PASS receipt are preserved. This audit
adds an explicit baseline-completion prerequisite to prevent an invalid routing
claim; it tightens interpretation and does not change a selection threshold,
rerun an example, or turn a failed candidate into a passing one.

The corrected comparator has a regression test for an apparent gain against a
timed-out baseline. Complete-case counts above are descriptive; they are not a
new selection/evaluation subset or a post-hoc efficacy test.

Non-claims: no demonstrated capability gain, adaptive routing benefit, PEFT
recovery, or sealed efficacy. Fixed recurrence's short smoke qualification and
canonical-state restoration are separate from this censored quality comparison.

Next experiment: preregister and rerun the identical A0/A1/A3 development matrix
with an adequate per-prompt CPU time budget; retain the same examples, weights,
recurrence counts, decoding and scoring. No other mechanism should change.
""",encoding="utf-8")
    print(out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
