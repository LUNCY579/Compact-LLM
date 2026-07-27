from __future__ import annotations
import json,random
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(s):
    with (ROOT/"data"/s/f"{s}.jsonl").open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
def user(r):return next(m["content"] for m in reversed(r["messages"][:-1]) if m["role"]=="user")
def main():
    rng=random.Random(20260727);out=["# HUMAN DATA AUDIT PACKET","","Mark each item `ACCEPT`, `REPAIR`, or `REJECT`.",""]
    n=0
    for split in ("train","dev","sealed"):
        by=defaultdict(list)
        for r in read(split):by[r["metadata"]["reply_class"]].append(r)
        k=3 if split!="sealed" else 1
        for cat,rows in sorted(by.items()):
            for r in rng.sample(rows,k):
                n+=1;out += [f"## {n}. `{r['id']}`","",f"- Split: `{split}`",f"- Category: `{cat}`",f"- User: {user(r)}",f"- Assistant: {r['messages'][-1]['content']}","- Decision: `PENDING`","- Notes:",""]
    p=ROOT/"reports/HUMAN_DATA_AUDIT_PACKET.md";p.write_text("\n".join(out),encoding="utf-8");print(p)
if __name__=="__main__":main()
