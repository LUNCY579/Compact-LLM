#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path

def read(path):
    with path.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def norm(t): return " ".join(re.findall(r"[a-z0-9_.$%+\-]+",t.lower()))
def latest(row):
    if "messages" in row:
        us=[m["content"] for m in row["messages"] if m["role"]=="user"]; return norm(us[-1]) if us else ""
    return norm(row.get("prompt") or row.get("question") or row.get("input") or "")
def validate_sft(rows):
    ids=set(); prompts=set(); errs=[]
    for i,r in enumerate(rows,1):
        roles=[m.get("role") for m in r.get("messages",[])]; expected=["system"]+["user","assistant"]*((len(roles)-1)//2)
        if roles!=expected: errs.append(f"line {i}: role order {roles}")
        if not roles or roles[-1]!="assistant": errs.append(f"line {i}: final role")
        if r.get("metadata",{}).get("split")!="train": errs.append(f"line {i}: split")
        if "question" in r: errs.append(f"line {i}: top-level question")
        if r.get("id") in ids: errs.append(f"line {i}: duplicate id")
        ids.add(r.get("id")); p=norm("\n".join(f"{m['role']}:{m['content']}" for m in r.get("messages",[])[:-1]))
        if p in prompts: errs.append(f"line {i}: duplicate prompt")
        prompts.add(p)
    return errs,{"rows":len(rows),"unique_ids":len(ids),"unique_prompts":len(prompts)}
def validate_dpo(rows):
    ids=set(); prompts=set(); errs=[]
    for i,r in enumerate(rows,1):
        if r.get("id") in ids: errs.append(f"line {i}: duplicate dpo id")
        ids.add(r.get("id")); p=r.get("prompt","")
        if p in prompts: errs.append(f"line {i}: duplicate dpo prompt")
        prompts.add(p)
        if not p.endswith("<|im_start|>assistant\n"): errs.append(f"line {i}: bad prefix")
        if r.get("chosen","").strip()==r.get("rejected","").strip(): errs.append(f"line {i}: same choice")
    return errs,{"rows":len(rows),"unique_ids":len(ids),"unique_prompts":len(prompts)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sft",type=Path,action="append",default=[]); ap.add_argument("--dpo",type=Path,action="append",default=[]); ap.add_argument("--forbidden-eval-file",type=Path,action="append",default=[]); a=ap.parse_args(); report={"sft":{},"dpo":{},"leakage":{}}; errors=[]
    eval_text=set()
    for p in a.forbidden_eval_file:
        for r in read(p): eval_text.add(latest(r))
    for p in a.sft:
        rows=read(p); e,s=validate_sft(rows); errors+=e; report["sft"][str(p)]=s
        if eval_text: report["leakage"][str(p)]={"exact_latest_user_overlap":sum(latest(r) in eval_text for r in rows)}
    for p in a.dpo:
        rows=read(p); e,s=validate_dpo(rows); errors+=e; report["dpo"][str(p)]=s
    report["status"]="PASS" if not errors and all(v["exact_latest_user_overlap"]==0 for v in report["leakage"].values()) else "FAIL"; report["errors"]=errors[:100]; print(json.dumps(report,indent=2)); raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__": main()
