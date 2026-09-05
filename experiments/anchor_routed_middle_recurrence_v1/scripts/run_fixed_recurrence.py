"""Qualify selected fixed middle arms on the frozen unsealed development set."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parents[1]/"Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from armr.hooks import Reapplication, SafetyError, frozen_inference
from armr.evaluation import encode, forward_logits
from armr.provenance import DATA, cases_from, checkpoint_files, new_run, sha256, state_hash, write_json
from armr.recurrence import FixedBlock
from armr.reporting import csv_file, mean, summarize
from run_experiment import resolve_model, run_arm


def development_result(rows, baseline):
    base={r["case_id"]:r for r in baseline}
    family=defaultdict(list)
    wins,losses=[],[]
    for row in rows:
        old=base[row["case_id"]]
        delta=row["score"]-old["score"]
        family[row["family"]].append(delta)
        if delta>0:
            wins.append(row["case_id"])
        if delta<0:
            losses.append(row["case_id"])
    deltas={k:mean(v) for k,v in family.items()}
    valid_delta=mean([r["invalid"] for r in rows])-mean([r["invalid"] for r in baseline])
    checks={"baseline_completed_without_safety_stop":not any(r["safety_stopped"] for r in baseline),
            "overall_no_loss":len(wins)>=len(losses),
            "no_family_loss_over_0_10":all(d>=-.10 for d in deltas.values()),
            "no_invalid_increase":valid_delta<=0,
            "no_safety_stops":not any(r["safety_stopped"] for r in rows),
            "complete":bool(rows) and len(rows)==len(baseline) and len(base)==len(rows)
                       and {r["case_id"] for r in rows}==set(base)}
    preserved=all(checks.values())
    return {"items":len(rows),"comparison_valid":checks["baseline_completed_without_safety_stop"],
            "correct":sum(r["score"] for r in rows),
            "baseline_correct":sum(r["score"] for r in baseline),"wins":wins,"losses":losses,
            "family_score_deltas":deltas,"checks":checks,"preservation_pass":preserved,
            "router_scaffold_licensed":preserved and bool(wins),
            "mean_logit_cosine":mean([r["logit_cosine"] for r in rows]),
            "mean_kl":mean([r["kl_base_to_arm"] for r in rows]),
            "invalid_rate":mean([r["invalid"] for r in rows]),
            "wall_seconds":sum(r["wall_seconds"] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas",type=Path,required=True)
    args=parser.parse_args()
    atlas_path=args.atlas.resolve()
    if not atlas_path.is_relative_to((ROOT/"evidence/01c_completed_candidate_atlas").resolve()):
        raise ValueError("audited completed-candidate atlas required")
    receipt=json.loads((atlas_path/"run_manifest.json").read_text())
    decision=json.loads((atlas_path/"decision.json").read_text())
    atlas=json.loads((atlas_path/"atlas.json").read_text())
    config=json.loads((ROOT/"configs/reapplication_screen.json").read_text())
    if receipt["status"]!="PASS" or not decision["fixed_recurrence"] or receipt["config"]!=config:
        raise SafetyError("matching fixed-recurrence gate required")
    layer=decision["selected_layer"]
    singles={r["layer"]:r for r in atlas if r["applications"]==2 and r["go_fixed_smoke"] and r["layer"]<25}
    pairs=[(i,i+1) for i in singles if i+1 in singles]
    pair=max(pairs,key=lambda p:(mean([singles[i]["mean_logit_cosine"] for i in p]),-p[0])) if pairs else None
    a2=any(r["layer"]==layer and r["applications"]==3 and r["go_fixed_smoke"] for r in atlas)
    selected={"A1_layer":layer,"A2_licensed":a2,"A3_block":pair,
              "selection":"frozen mean-cosine ranking among passing middle arms; adjacent-pair constituent mean"}
    model_path=resolve_model(None)
    smoke=cases_from(DATA/config["smoke_file"])
    dev=cases_from(DATA/config["development_file"])
    out,manifest=new_run("02_fixed_recurrence",config,model_path,DATA/config["development_file"])
    write_json(out/"selected_configuration.json",selected)
    manifest.update(atlas_receipt=str(atlas_path),atlas_receipt_sha256=sha256(atlas_path/"run_manifest.json"),selected_configuration=selected)
    print(f"RUN_DIR={out}",flush=True)
    start=time.perf_counter()
    model=None
    try:
        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        before_files=checkpoint_files(model_path)
        if before_files["model.safetensors"]!=config["checkpoint_sha256"]:
            raise SafetyError("checkpoint hash mismatch")
        manifest["checkpoint_files_before"]=before_files
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=True,
                  dtype=torch.float32,attn_implementation="eager",low_cpu_mem_usage=True).eval().requires_grad_(False)
        if type(model).__name__!="Qwen3AbstainForCausalLM":
            raise SafetyError("custom Supa class required")
        manifest["state_hash_before"]=state_hash(model)
        smoke_base,smoke_logits=run_arm(model,tokenizer,smoke,config,out,"A0_smoke")
        check_rows,_=run_arm(model,tokenizer,smoke,config,out,"A1_smoke",smoke_logits,layer=layer,applications=2)
        source_rows=[json.loads(s) for s in (atlas_path/"raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
        old={r["case_id"]:r for r in source_rows if r["arm"]==f"layer_{layer:02d}_apps_2"}
        reproduced=all(r["token_ids"]==old[r["case_id"]]["token_ids"] and abs(r["logit_cosine"]-old[r["case_id"]]["logit_cosine"])<1e-7 for r in check_rows)
        if not reproduced or not summarize(check_rows,smoke_base,config)["go_fixed_smoke"]:
            raise SafetyError("selected A1 smoke failed to reproduce")
        block_go=False
        if pair:
            block_rows,_=run_arm(model,tokenizer,smoke,config,out,"A3_smoke",smoke_logits,
                                  wrapper_type=FixedBlock,layers=list(pair),applications=2)
            block_summary=summarize(block_rows,smoke_base,config)
            write_json(out/"A3_smoke_gate.json",block_summary)
            block_go=block_summary["go_fixed_smoke"]
        baseline,bases=run_arm(model,tokenizer,dev,config,out,"A0_dev")
        results={}
        arms=[("A1_dev",Reapplication,{"layer":layer,"applications":2})]
        if a2:
            arms.append(("A2_dev",Reapplication,{"layer":layer,"applications":3}))
        if block_go:
            arms.append(("A3_dev",FixedBlock,{"layers":list(pair),"applications":2}))
        for name,kind,kwargs in arms:
            rows,_=run_arm(model,tokenizer,dev,config,out,name,bases,wrapper_type=kind,**kwargs)
            results[name]=development_result(rows,baseline)
            write_json(out/f"{name}_comparison.json",results[name])
        with Reapplication(model),frozen_inference(model):
            restored=all(torch.equal(bases[c["case_id"]],forward_logits(model,encode(tokenizer,c))) for c in dev)
        if not restored:
            raise SafetyError("fixed-stage restoration failed")
        final={"status":"PASS" if any(r["preservation_pass"] for r in results.values()) else "FAIL",
               "mechanical_qualification":"PASS","A1_smoke_reproduced":reproduced,
               "restored_logits_bit_identical":restored,"A2_run":a2,"A3_smoke_pass":block_go,
               "adaptive_routing":any(r["router_scaffold_licensed"] for r in results.values()),
               "peft_recovery":False,"sealed_evaluation":False,"development_results":results}
        write_json(out/"decision.json",final)
        table="\n".join(f"| {name} | {r['correct']:.0f}/{r['items']} | {len(r['wins'])} | {len(r['losses'])} | {r['mean_logit_cosine']:.4f} | {r['preservation_pass']} | {r['router_scaffold_licensed']} |" for name,r in results.items())
        (out/"report.md").write_text(f"""# Fixed middle recurrence development qualification

Status: {final['status']} for development preservation; mechanical qualification PASS.

Question: does the frozen selected recurrence preserve the existing unsealed
MERGEDEV-fast suite, and does it license an adaptive-router scaffold?
Setup: CPU FP32/eager Supa-A1.1, seed 17, four threads, full-prefix no-KV;
16 development items. A1: L{layer}, two total applications. A2 licensed: {a2}.
A3 block: {pair}; independent smoke gate passed: {block_go}.
Selection and gates were frozen before development outputs. Source provenance,
dataset hash, parameter/checkpoint hashes and resources are in run_manifest.json.

Mechanical qualification: A1 reproduced the original completed smoke outputs
and logits; block execution/order has separate unit tests and an eight-case
real-model smoke gate; all development baseline logits restore bit-identically.

Intact development baseline: {sum(r['score'] for r in baseline):.0f}/{len(baseline)}.

| Arm | Correct | Paired wins | Paired losses | Logit cosine | Preservation | Router licensed |
|---|---:|---:|---:|---:|---|---|
{table}

Interpretation: each arm must preserve overall accuracy, lose no family by more
than 0.10, add no invalid outputs and correct at least one previously missed
answer to license routing. Family deltas and paired case IDs are recorded in
the comparison JSON files. A frozen operator can be mechanically stable yet
fail this development gate. The original L27 campaign failure remains FAIL;
only completed, audited middle operators were eligible here.

Non-claims: no training, router, PEFT, long-context reasoning or sealed efficacy
result. Sixteen development items and mostly one item per family are a small
screen. Serial CPU timing includes instrumentation and is not deployment speed.
No compute-matched efficacy claim is made.

Next gate: {'router scaffold mechanical qualification, without an efficacy claim' if final['adaptive_routing'] else 'STOP adaptive escalation; retain the fixed-operator evidence and preregister one further causal development test'}.
""",encoding="utf-8")
        manifest.update(status=final["status"],decision=final)
    except BaseException as exc:
        manifest.update(status="FAIL",error=repr(exc))
        write_json(out/"failure.json",{"error":repr(exc),"traceback":traceback.format_exc()})
        if not (out/"report.md").exists():
            (out/"report.md").write_text(f"# Fixed recurrence failure\n\nStatus: FAIL.\n\n{exc!r}\n\nNo adaptive gate is licensed.\n",encoding="utf-8")
        traceback.print_exc()
    finally:
        if model is not None:
            manifest["state_hash_after"]=state_hash(model)
            manifest["checkpoint_files_after"]=checkpoint_files(model_path)
            manifest["canonical_unchanged"]=manifest["state_hash_after"]==manifest.get("state_hash_before") and manifest["checkpoint_files_after"]==manifest.get("checkpoint_files_before")
            manifest["trainable_parameter_count"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
            manifest["parameters_with_gradients"]=sum(p.grad is not None for p in model.parameters())
            if not manifest["canonical_unchanged"] or manifest["parameters_with_gradients"] or manifest["trainable_parameter_count"]:
                manifest["status"]="FAIL"
        manifest.update(wall_seconds=time.perf_counter()-start,peak_rss_bytes=psutil.Process().memory_info().peak_wset)
        write_json(out/"run_manifest.json",manifest)
    print(f"STATUS={manifest['status']} RUN_DIR={out}",flush=True)
    return 0 if manifest["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
