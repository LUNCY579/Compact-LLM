#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path

def read(p):
    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,action="append",required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--seed",type=int,default=20260818); a=ap.parse_args(); rows=[]
    for p in a.input: rows.extend(read(p))
    ids=[r.get("id") for r in rows]
    if len(ids)!=len(set(ids)): raise SystemExit("Duplicate IDs detected; do not append complete v1/v2 to the recommended v3 combined file.")
    random.Random(a.seed).shuffle(rows); a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8",newline="\n") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps({"rows":len(rows),"unique_ids":len(set(ids)),"output":str(a.output)},indent=2))
if __name__=="__main__": main()
