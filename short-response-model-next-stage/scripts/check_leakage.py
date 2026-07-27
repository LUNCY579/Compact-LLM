from __future__ import annotations
import json,re
from itertools import combinations
from pathlib import Path
from rapidfuzz.fuzz import ratio
ROOT=Path(__file__).resolve().parents[1]
def read(s):
    with (ROOT/"data"/s/f"{s}.jsonl").open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
def norm(x):
    x=x.lower();x=re.sub(r"quick (update|question):|just to let you know,|can you answer briefly:|in one sentence,|i am wondering:|based on what i said,|remind me:|one quick check:","",x);return re.sub(r"\s+"," ",x).strip(" .!?,")
def prompt(r):return next(m["content"] for m in reversed(r["messages"]) if m["role"]=="user")
def main():
    d={s:read(s) for s in ("train","dev","sealed")};bad=[]
    for a,b in combinations(d,2):
        fa={r["metadata"]["template_family"] for r in d[a]};fb={r["metadata"]["template_family"] for r in d[b]}
        bad += [f"family overlap {x} between {a}/{b}" for x in sorted(fa&fb)]
        pa=[(r["id"],norm(prompt(r))) for r in d[a]];pb=[(r["id"],norm(prompt(r))) for r in d[b]]
        for ia,xa in pa:
            for ib,xb in pb:
                if ratio(xa,xb)>=94:bad.append(f"near overlap {ia} <-> {ib}")
    if bad:
        print("LEAKAGE CHECK FAILED");print("\n".join(bad[:100]));raise SystemExit(1)
    print("LEAKAGE CHECK PASSED")
if __name__=="__main__":main()
