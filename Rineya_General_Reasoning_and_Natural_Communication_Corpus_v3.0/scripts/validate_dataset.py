#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,statistics
from pathlib import Path

def read(p):
    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def norm(t): return " ".join(re.findall(r"[a-z0-9]+",t.lower()))
def latest(r): return [m["content"] for m in r["messages"] if m["role"]=="user"][-1]
def toks(t): return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\w\s]",t))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sft",type=Path,action="append",default=[]); ap.add_argument("--dpo",type=Path,action="append",default=[]); ap.add_argument("--forbidden",type=Path,action="append",default=[]); a=ap.parse_args(); errors=[]; forbidden=set()
    for p in a.forbidden:
        for r in read(p):
            if isinstance(r.get("messages"),list):
                us=[m.get("content","") for m in r["messages"] if m.get("role")=="user"]; forbidden.add(norm(us[-1])) if us else None
            for k in ("prompt","question","input","query","instruction"):
                if isinstance(r.get(k),str): forbidden.add(norm(r[k]))
    report={"sft":{},"dpo":{},"leakage":{}}
    for p in a.sft:
        rows=read(p); ids=set(); prompts=set(); users=set(); lens=[]; overlap=0
        for i,r in enumerate(rows,1):
            roles=[m.get("role") for m in r.get("messages",[])]; exp=["system"]+["user","assistant"]*((len(roles)-1)//2)
            if roles!=exp: errors.append(f"{p}:{i}:roles")
            if r.get("metadata",{}).get("split")!="train": errors.append(f"{p}:{i}:split")
            if "question" in r: errors.append(f"{p}:{i}:question")
            if r.get("id") in ids: errors.append(f"{p}:{i}:id")
            ids.add(r.get("id")); pr=norm("\n".join(f"{m['role']}:{m['content']}" for m in r["messages"][:-1])); u=norm(latest(r)); overlap+=int(u in forbidden)
            if pr in prompts: errors.append(f"{p}:{i}:prompt")
            if u in users: errors.append(f"{p}:{i}:latest")
            prompts.add(pr); users.add(u); lens.append(toks(r["messages"][-1]["content"]))
        report["sft"][str(p)]={"rows":len(rows),"unique_ids":len(ids),"median":statistics.median(lens),"max":max(lens)}; report["leakage"][str(p)]={"exact_latest_user_overlap":overlap}
    for p in a.dpo:
        rows=read(p); report["dpo"][str(p)]={"rows":len(rows),"bad_suffix":sum(not r.get("prompt","").endswith("<|im_start|>assistant\n") for r in rows),"same_choice":sum(r.get("chosen")==r.get("rejected") for r in rows)}
    report["errors"]=errors[:100]; report["status"]="PASS" if not errors and all(v["exact_latest_user_overlap"]==0 for v in report["leakage"].values()) else "FAIL"; print(json.dumps(report,indent=2)); raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__": main()
