"""Publish exact finite-group tables and original mathematical control cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
from pathlib import Path
import platform

from . import exact_math as m

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P_GRID = (0, .0005, .005, .01, .025, .05, .1, .25, .5, .75, .9, .99, 1)
N_GRID = (2,4,8,16)
CURRENT_SOURCE = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-comp-rs-amendment-1/experiments/maxrl_comp002_pretraining_v1/evidence")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binomial_tail(n, rate, minimum):
    return sum(comb(n,k)*rate**k*(1-rate)**(n-k) for k in range(minimum,n+1))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise SystemExit("audit directory exists (create-only)")
    rows=[]
    for n in N_GRID:
        for p in P_GRID:
            for method in m.METHODS:
                epsilons=(0.0,1e-6) if method in ("grpo_sample","grpo_population","maxrl_practical") else (0.0,)
                for epsilon in epsilons:
                    rows.append(m.gradient_statistics(method,p,n,epsilon))
    max_error=max(abs(r["expected_logit_gradient"]-m.maxrl_closed_gradient(r["p"],r["n"],practical=r["method"]=="maxrl_practical"))
                  for r in rows if r["method"] in ("maxrl_success","maxrl_practical") and r["epsilon"]==0)
    if max_error>1e-12 or max(abs(r["probability_sum"]-1) for r in rows)>1e-12:
        raise SystemExit("exact-enumeration identity failed")
    stopped=[m.bounded_failure_continuation(p,n,blocks,gamma)
             for p in (.0005,.01,.05,.1,.5,.9) for n in (4,8) for blocks in (2,4) for gamma in (0,.5,1)]
    if max(abs(r["probability_sum"]-1) for r in stopped)>1e-12 or max(abs(r["expected_draws"]-r["expected_draws_closed"]) for r in stopped)>1e-10:
        raise SystemExit("stopping-cost identity failed")
    source={}
    for name in ("S0_Q1_FP32.json","S0_COMPETENCE.json"):
        path=CURRENT_SOURCE/name
        record=json.loads(path.read_text(encoding="utf-8"))
        source[name]={"path":str(path),"sha256":sha(path),"completed":record["completed"],
                      "stage":record["stage"]}
        if name=="S0_COMPETENCE.json":
            source[name]["outcome"]=record["outcome"]
            source[name]["by_skill"]={k:{"strict_successes":v["successes"],"samples":v["samples"],
                                          "mixed_groups":v["groups"]["mixed"],"prompts":v["prompts"]}
                                     for k,v in record["summary"].items()}
    signal=[{"p":p,"mixed_probability_n8":m.mixed_probability(p,8),
             "expected_mixed_of_150":150*m.mixed_probability(p,8),
             "probability_at_least_15_of_150_constant_p":binomial_tail(150,m.mixed_probability(p,8),15)}
            for p in P_GRID]
    result={"experiment":"SKILL-ACQUISITION-METHOD-LAB-001","kind":"exact scalar enumeration and original quadratic controls",
            "python":platform.python_version(),"native_models_run":0,"source_evidence":source,
            "identity_max_abs_error":max_error,"fixed_group_rows":rows,"stopped_sampling":stopped,
            "rejection_mixture":m.rejection_mixture([.02,.5],[.5,.5],8),
            "rejection_uniform_prompt":m.rejection_mixture([0,.5],[.5,.5],8),
            "signal_window":signal,"sequential_counterexample":m.sequential_counterexample(),
            "composition_counterexample":m.composition_counterexample(),
            "dual_fisher_proxy":m.dual_fisher_allocation([1]*4,[9,8,4,1],[10,1,0,0],2,1),
            "row_fisher_ranking_invariance":m.row_ranking_invariance([[4,3,2,1],[1,3,2,4]],[100,.01],2),
            "lora_gauge":m.lora_gauge_counterexample(),"mastered_replay":m.replay_signal(),
            "limits":"Exact for these defined toys; does not establish native acquisition, composition, retention, or superiority."}
    args.out.mkdir(parents=True)
    (args.out/"EXACT_AUDIT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    with (args.out/"fixed_group_expectations.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (args.out/"stopped_sampling_cost.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(stopped[0]));writer.writeheader();writer.writerows(stopped)
    binding={"code_sha256":{name:sha(HERE/name) for name in ("exact_math.py","run_audit.py")},
             "outputs_sha256":{p.name:sha(p) for p in args.out.iterdir() if p.is_file()},
             "enumeration_rows":len(rows),"stopping_cases":len(stopped),"status":"EXACT_TOY_IDENTITIES_PASS"}
    (args.out/"RECEIPT.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":binding["status"],"cases":len(rows),"stopping_cases":len(stopped),"max_error":max_error}))


if __name__=="__main__":
    main()
