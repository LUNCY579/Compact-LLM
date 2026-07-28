from __future__ import annotations
# This is the structural gate for the generated SFT corpus. It checks
# schema compliance and project-specific invariants before training.
import argparse,json
from collections import Counter
from pathlib import Path
from jsonschema import Draft202012Validator
# Use stable repository-relative paths.
ROOT=Path(__file__).resolve().parents[2]
# Parse a JSONL file into records while ignoring blank lines.
def read(p):
    with p.open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
# Validate train, development, and sealed data in one run.
def main():
    # The schema path is configurable, but defaults to the project's
    # versioned SFT-record contract.
    ap=argparse.ArgumentParser();ap.add_argument("--schema",type=Path,default=ROOT/"data/schemas/sft_record.schema.json");a=ap.parse_args()
    # Keep one global ID set so duplicates are caught even when they occur
    # in different splits.
    v=Draft202012Validator(json.loads(a.schema.read_text(encoding="utf-8")));ids=set();errs=[]
    # Apply the same validation rules to every split.
    for split in ("train","dev","sealed"):
        rows=read(ROOT/"data"/split/f"{split}.jsonl"); c=Counter()
        for i,r in enumerate(rows,1):
            # Collect every schema error instead of stopping at the first
            # one, making a single repair pass more practical.
            errs += [f"{split}:{i}: {e.message}" for e in v.iter_errors(r)]
            if r["id"] in ids: errs.append(f"duplicate id {r['id']}")
            ids.add(r["id"])
            # The record's declared split must agree with its directory.
            if r["metadata"]["split"]!=split:errs.append(f"{r['id']}: wrong split metadata")
            # Training examples must begin with policy context and end with
            # the assistant response that will be used as the target.
            if r["messages"][0]["role"]!="system" or r["messages"][-1]["role"]!="assistant":errs.append(f"{r['id']}: invalid message boundary")
            # Category counts provide a quick balance/coverage diagnostic.
            c[r["metadata"]["reply_class"]]+=1
        print(split,len(rows),dict(sorted(c.items())))
    # Fail the process when any invariant is violated so downstream
    # training cannot accidentally consume malformed records.
    if errs:
        print("VALIDATION FAILED");print("\n".join(errs[:100]));raise SystemExit(1)
    print("VALIDATION PASSED")
# Standard script entry point.
if __name__=="__main__":main()
