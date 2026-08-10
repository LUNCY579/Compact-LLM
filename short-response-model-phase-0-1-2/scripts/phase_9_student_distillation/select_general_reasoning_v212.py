from __future__ import annotations

"""Preservation-aware selector for v2.1.2 general-reasoning training."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ORIGINAL_PROTECTED = [
    "procedure_correctness",
    "supported_answer_correctness",
    "evidence_faithfulness",
    "correct_abstention_rate",
    "conflict_safe_handling_rate",
    "social_usable_rate",
]
REASONING_BENEFIT = ["overall_correctness", "macro_family_correctness", "hard_correctness"]
REASONING_PROTECTED = ["abstention_correctness", "short_usable_rate"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def delta(a: float | None, b: float | None) -> float:
    return 0.0 if a is None or b is None else b - a


def rel_loss(a: float | None, b: float | None) -> float:
    if a is None or b is None or a <= 0:
        return 0.0
    return max(0.0, 1.0 - b / a)


def metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize eval packets for selection metrics.

    `evaluate_rag.py` stores protected metrics under ``conditions.real_rag``.
    General-reasoning evals and unit-test stubs keep metrics at the top level.
    Always prefer Real-RAG when that condition packet exists.
    """
    if not isinstance(summary, dict):
        return {}
    conditions = summary.get("conditions")
    if isinstance(conditions, dict):
        real = conditions.get("real_rag")
        if isinstance(real, dict):
            view = dict(real)
            view.setdefault("evaluation_valid", summary.get("evaluation_valid", True))
            return view
    return summary


@dataclass
class Candidate:
    name: str
    adapter: str
    original: dict[str, Any]
    challenge: dict[str, Any]
    reasoning: dict[str, Any]


def packet(c: Candidate, bo: dict[str, Any], bc: dict[str, Any], br: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    # RAG evals nest metrics under conditions.real_rag; reasoning evals stay flat.
    bo_m = metric_view(bo)
    bc_m = metric_view(bc)
    co_m = metric_view(c.original)
    cc_m = metric_view(c.challenge)

    original = {}
    for key in ORIGINAL_PROTECTED:
        before = f(bo_m, key)
        after = f(co_m, key)
        original[key] = {
            "baseline": before,
            "candidate": after,
            "relative_loss": rel_loss(before, after),
            "delta": delta(before, after),
        }
    reasoning = {}
    gains = []
    for key in REASONING_BENEFIT:
        before = f(br, key)
        after = f(c.reasoning, key)
        g = max(0.0, delta(before, after))
        gains.append(g)
        reasoning[key] = {"baseline": before, "candidate": after, "delta": delta(before, after), "positive_gain": g}
    for key in REASONING_PROTECTED:
        before = f(br, key)
        after = f(c.reasoning, key)
        reasoning[key] = {
            "baseline": before,
            "candidate": after,
            "delta": delta(before, after),
            "relative_loss": rel_loss(before, after),
        }
    general_gain = sum(gains) / len(gains) if gains else 0.0
    cb = f(bc_m, "procedure_correctness")
    ca = f(cc_m, "procedure_correctness")
    challenge_delta = delta(cb, ca)
    challenge_positive = max(0.0, challenge_delta)
    challenge_loss = rel_loss(cb, ca)
    invalid = []
    if not bool(c.original.get("evaluation_valid", True)):
        invalid.append("original_invalid")
    if not bool(c.challenge.get("evaluation_valid", True)):
        invalid.append("challenge_invalid")
    if not bool(c.reasoning.get("evaluation_valid", True)):
        invalid.append("reasoning_invalid")
    policies = {}
    for name, pcfg in (cfg.get("selection", {}).get("policies", {}) or {}).items():
        ow = pcfg.get("original_preservation_weights", {})
        rp = pcfg.get("reasoning_preservation_weights", {})
        original_cost = sum(float(ow.get(k, 0.0)) * original[k]["relative_loss"] for k in ORIGINAL_PROTECTED)
        reasoning_cost = sum(float(rp.get(k, 0.0)) * reasoning[k]["relative_loss"] for k in REASONING_PROTECTED)
        ch_cost = float(pcfg.get("challenge_regression_weight", 0.0)) * challenge_loss
        benefit = (
            float(pcfg.get("general_reasoning_gain_weight", 1.0)) * general_gain
            + float(pcfg.get("challenge_gain_weight", 0.0)) * challenge_positive
        )
        cost = original_cost + reasoning_cost + ch_cost
        policies[name] = {"benefit": benefit, "preservation_cost": cost, "opportunity_margin": benefit - cost}
    return {
        "name": c.name,
        "adapter": c.adapter,
        "technical_invalidity": invalid,
        "original_preservation": original,
        "reasoning": reasoning,
        "general_reasoning_gain": general_gain,
        "challenge": {
            "baseline": cb,
            "candidate": ca,
            "delta": challenge_delta,
            "positive_gain": challenge_positive,
            "relative_loss": challenge_loss,
        },
        "policies": policies,
    }


def select(cfg:dict[str,Any],baseline:Candidate,trained:list[Candidate])->dict[str,Any]:
    packets=[packet(baseline,baseline.original,baseline.challenge,baseline.reasoning,cfg)]
    packets += [packet(c,baseline.original,baseline.challenge,baseline.reasoning,cfg) for c in trained]
    official=str(cfg.get("selection",{}).get("official_policy","preservation_guarded"))
    policy_names=list((cfg.get("selection",{}).get("policies",{}) or {}).keys())
    selections={}
    for pol in policy_names:
        eligible=[p for p in packets if not p["technical_invalidity"] and p["name"]!="input_student" and p["policies"][pol]["opportunity_margin"]>0]
        if eligible:
            w=max(eligible,key=lambda p:(p["policies"][pol]["opportunity_margin"],p["general_reasoning_gain"]))
        else:
            w=packets[0]
        selections[pol]=w["name"]
    winner_name=selections[official]
    winner=next(p for p in packets if p["name"]==winner_name)
    disagreement=len(set(selections.values()))>1
    status="GENERAL_REASONING_CANDIDATE_SELECTED" if winner_name!="input_student" else "INPUT_STUDENT_RETAINED"
    if disagreement: status="POLICY_SENSITIVITY_REVIEW_REQUIRED"
    return {"schema_version":"2.1.2","status":status,"official_policy":official,"selected_candidate":winner_name,"selected_adapter":winner["adapter"],"policy_selections":selections,"policy_sensitivity_disagreement":disagreement,"candidates":packets,"interpretation":{"general_reasoning_gain":"Mean positive absolute gain across overall, family-macro, and hard general-reasoning correctness.","preservation_cost":"Nonnegative relative losses on original Phase 9 behavior, reasoning abstention/shortness, and v2.1.1 procedural challenge where applicable.","fallback":"The input student remains a first-class candidate; training is kept only when measured generalization benefit exceeds preservation cost."}}


def markdown(x:dict[str,Any])->str:
    lines=["# v2.1.2 General Reasoning DEV Selection","",f"**Status:** `{x['status']}`",f"**Selected:** `{x['selected_candidate']}`",f"**Official policy:** `{x['official_policy']}`","","| Candidate | Gen. gain | Reasoning overall | Reasoning hard | Original procedure | Supported | Faithfulness | Abstention | v2.1.1 challenge | Margin |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in x["candidates"]:
        o=c["original_preservation"]; r=c["reasoning"]; pol=c["policies"].get(x["official_policy"],{})
        lines.append("| {n} | {g:.4f} | {ro} | {rh} | {op} | {sup} | {fa} | {ab} | {ch} | {m:.6f} |".format(n=c["name"],g=c["general_reasoning_gain"],ro=r["overall_correctness"]["candidate"],rh=r["hard_correctness"]["candidate"],op=o["procedure_correctness"]["candidate"],sup=o["supported_answer_correctness"]["candidate"],fa=o["evidence_faithfulness"]["candidate"],ab=o["correct_abstention_rate"]["candidate"],ch=c["challenge"]["candidate"],m=float(pol.get("opportunity_margin",0.0))))
    lines += ["",f"Policy selections: `{x['policy_selections']}`",f"Policy sensitivity disagreement: `{x['policy_sensitivity_disagreement']}`","","The selector does not require a trained checkpoint to win. If broad reasoning training does not earn its preservation cost, the input student is retained."]
    return "\n".join(lines)+"\n"


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True)
    for stem in ["baseline","stage-a","stage-b1","stage-b2"]:
        p.add_argument(f"--{stem}-original",type=Path,required=True); p.add_argument(f"--{stem}-challenge",type=Path,required=True); p.add_argument(f"--{stem}-reasoning",type=Path,required=True); p.add_argument(f"--{stem}-adapter",default="")
    p.add_argument("--output-json",type=Path,required=True); p.add_argument("--output-md",type=Path,required=True); a=p.parse_args()
    def c(stem,label): return Candidate(label,getattr(a,f"{stem.replace('-','_')}_adapter"),load(getattr(a,f"{stem.replace('-','_')}_original")),load(getattr(a,f"{stem.replace('-','_')}_challenge")),load(getattr(a,f"{stem.replace('-','_')}_reasoning")))
    cfg=load(a.config); base=c("baseline","input_student"); trained=[c("stage-a","stage_a_reasoning"),c("stage-b1","stage_b1_mixed"),c("stage-b2","stage_b2_mixed")]
    out=select(cfg,base,trained); a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8"); a.output_md.write_text(markdown(out),encoding="utf-8"); print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__": main()
