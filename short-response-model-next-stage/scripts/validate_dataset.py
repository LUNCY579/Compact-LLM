from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with p.open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--schema",type=Path,default=ROOT/"data/schemas/sft_record.schema.json");a=ap.parse_args()
    v=Draft202012Validator(json.loads(a.schema.read_text(encoding="utf-8")));ids=set();errs=[]
    for split in ("train","dev","sealed"):
        rows=read(ROOT/"data"/split/f"{split}.jsonl"); c=Counter()
        for i,r in enumerate(rows,1):
            errs += [f"{split}:{i}: {e.message}" for e in v.iter_errors(r)]
            if r["id"] in ids: errs.append(f"duplicate id {r['id']}")
            ids.add(r["id"])
            if r["metadata"]["split"]!=split:errs.append(f"{r['id']}: wrong split metadata")
            if r["messages"][0]["role"]!="system" or r["messages"][-1]["role"]!="assistant":errs.append(f"{r['id']}: invalid message boundary")
            c[r["metadata"]["reply_class"]]+=1
        print(split,len(rows),dict(sorted(c.items())))
    if errs:
        print("VALIDATION FAILED");print("\n".join(errs[:100]));raise SystemExit(1)
    print("VALIDATION PASSED")
if __name__=="__main__":main()
