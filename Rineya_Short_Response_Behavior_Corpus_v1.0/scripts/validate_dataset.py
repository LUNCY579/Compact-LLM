#!/usr/bin/env python3
"""Validate Rineya train-only SFT and DPO JSONL files.

The script intentionally has no dependency on Transformers. jsonschema is
required for schema validation; rapidfuzz is optional for fuzzy leakage checks.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:
    raise SystemExit("Install jsonschema first: pip install jsonschema") from exc

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():
                continue
            try:
                row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows

def norm(text: str) -> str:
    text=text.lower().replace("’","'").replace("“",'"').replace("”",'"')
    return " ".join(re.findall(r"[a-z0-9_.$%+-]+",text))

def prompt_norm(row: dict[str,Any]) -> str:
    return norm("\n".join(
        f"{m['role']}:{m['content']}" for m in row["messages"][:-1]
    ))

def latest_user_norm(row: dict[str,Any]) -> str:
    users=[m["content"] for m in row["messages"] if m["role"]=="user"]
    return norm(users[-1])

def validate_sft(path: Path, schema: dict[str,Any]) -> dict[str,Any]:
    rows=read_jsonl(path)
    v=Draft202012Validator(schema)
    ids=set()
    prompts=set()
    errors=[]
    for i,row in enumerate(rows,1):
        for err in sorted(v.iter_errors(row),key=lambda e:list(e.path)):
            errors.append(f"line {i}: schema: {err.message}")
        rid=row.get("id")
        if rid in ids:
            errors.append(f"line {i}: duplicate id {rid}")
        ids.add(rid)
        messages=row.get("messages",[])
        if messages:
            roles=[m.get("role") for m in messages]
            expected=["system"]+["user","assistant"]*((len(messages)-1)//2)
            if roles != expected:
                errors.append(f"line {i}: invalid role order {roles}")
            if messages[-1].get("role")!="assistant":
                errors.append(f"line {i}: final message is not assistant")
        if row.get("metadata",{}).get("split")!="train":
            errors.append(f"line {i}: metadata.split is not train")
        if "question" in row:
            errors.append(f"line {i}: top-level question would trigger Phase 9 prompt rebuilding")
        p=prompt_norm(row) if messages else ""
        if p in prompts:
            errors.append(f"line {i}: duplicate rendered prompt")
        prompts.add(p)
    if errors:
        raise ValueError("\n".join(errors[:100]))
    return {"rows":len(rows),"unique_ids":len(ids),"unique_prompts":len(prompts)}

def validate_dpo(path: Path, schema: dict[str,Any]) -> dict[str,Any]:
    rows=read_jsonl(path)
    v=Draft202012Validator(schema)
    ids=set()
    prompts=set()
    errors=[]
    for i,row in enumerate(rows,1):
        for err in sorted(v.iter_errors(row),key=lambda e:list(e.path)):
            errors.append(f"line {i}: schema: {err.message}")
        rid=row.get("id")
        if rid in ids:
            errors.append(f"line {i}: duplicate id {rid}")
        ids.add(rid)
        prompt=row.get("prompt","")
        if not prompt.endswith("<|im_start|>assistant\n"):
            errors.append(f"line {i}: prompt lacks Qwen assistant generation prefix")
        if row.get("chosen","").strip()==row.get("rejected","").strip():
            errors.append(f"line {i}: chosen equals rejected")
        if prompt in prompts:
            errors.append(f"line {i}: duplicate DPO prompt")
        prompts.add(prompt)
    if errors:
        raise ValueError("\n".join(errors[:100]))
    return {"rows":len(rows),"unique_ids":len(ids),"unique_prompts":len(prompts)}

def leakage_check(train_paths: list[Path], eval_paths: list[Path], threshold: float) -> dict[str,Any]:
    train=[]
    eval_rows=[]
    for p in train_paths:
        train.extend(read_jsonl(p))
    for p in eval_paths:
        eval_rows.extend(read_jsonl(p))
    eval_full={prompt_norm(r) for r in eval_rows}
    eval_user={latest_user_norm(r) for r in eval_rows}
    exact_full=sum(prompt_norm(r) in eval_full for r in train)
    exact_user=sum(latest_user_norm(r) in eval_user for r in train)
    fuzzy=0
    fuzzy_available=False
    try:
        from rapidfuzz import fuzz, process
        fuzzy_available=True
        choices=list(eval_user)
        for row in train:
            q=latest_user_norm(row)
            match=process.extractOne(q,choices,scorer=fuzz.ratio,score_cutoff=threshold)
            if match and q != match[0]:
                fuzzy+=1
    except ImportError:
        pass
    return {
        "train_rows":len(train),
        "eval_rows":len(eval_rows),
        "exact_full_prompt_overlap":exact_full,
        "exact_latest_user_overlap":exact_user,
        "fuzzy_threshold":threshold,
        "fuzzy_near_overlap":fuzzy if fuzzy_available else None,
        "rapidfuzz_available":fuzzy_available,
    }

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--sft",type=Path,action="append",default=[])
    ap.add_argument("--dpo",type=Path,action="append",default=[])
    ap.add_argument("--sft-schema",type=Path,required=True)
    ap.add_argument("--dpo-schema",type=Path,required=True)
    ap.add_argument("--forbidden-eval-file",type=Path,action="append",default=[])
    ap.add_argument("--fuzzy-threshold",type=float,default=94.0)
    args=ap.parse_args()
    sft_schema=json.loads(args.sft_schema.read_text(encoding="utf-8"))
    dpo_schema=json.loads(args.dpo_schema.read_text(encoding="utf-8"))
    report={"sft":{},"dpo":{}}
    for p in args.sft:
        report["sft"][str(p)]=validate_sft(p,sft_schema)
    for p in args.dpo:
        report["dpo"][str(p)]=validate_dpo(p,dpo_schema)
    if args.forbidden_eval_file:
        report["leakage"]=leakage_check(args.sft,args.forbidden_eval_file,args.fuzzy_threshold)
        leak=report["leakage"]
        if leak["exact_full_prompt_overlap"] or leak["exact_latest_user_overlap"] or (leak["fuzzy_near_overlap"] or 0):
            raise SystemExit(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
