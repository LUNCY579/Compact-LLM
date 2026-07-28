from __future__ import annotations
# This script protects evaluation integrity by detecting prompts or
# template families that appear across train, dev, and sealed splits.
import json,re
from itertools import combinations
from pathlib import Path
from rapidfuzz.fuzz import ratio
# Anchor file access to the repository root.
ROOT=Path(__file__).resolve().parents[2]
# Read one named JSONL split and skip blank lines.
def read(s):
    with (ROOT/"data"/s/f"{s}.jsonl").open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
# Remove superficial wrapper phrases and normalize whitespace/punctuation.
# This makes the comparison focus on the underlying prompt rather than
# harmless wording such as 'Quick question:' or 'In one sentence,'.
def norm(x):
    x=x.lower();x=re.sub(r"fyi|btw|by the way|now that i think of it|check this out|tell me|just tell me|quick (update|question):|just to let you know,|can you answer briefly:|in one sentence,|i am wondering:|based on what i said,|remind me:|one quick check:","",x);return re.sub(r"\s+"," ",x).strip(" .!?,")
# Extract the most recent user message, which is the prompt answered by
# the final assistant target.
def prompt(r):return next(m["content"] for m in reversed(r["messages"]) if m["role"]=="user")
# Compare every pair of splits; records within the same split are outside
# this script's cross-split leakage check.
def main():
    # Load all three splits once and collect every violation before
    # deciding whether the check passes.
    d={s:read(s) for s in ("train","dev","sealed")};bad=[]
    for a,b in combinations(d,2):
        # Template-family overlap is an explicit project-level failure,
        # even when the rendered wording is not exactly identical.
        fa={r["metadata"]["template_family"] for r in d[a]};fb={r["metadata"]["template_family"] for r in d[b]}
        bad += [f"family overlap {x} between {a}/{b}" for x in sorted(fa&fb)]
        # Store normalized prompts with IDs so any detected overlap can be
        # traced back to the source records.
        pa=[(r["id"],norm(prompt(r))) for r in d[a]];pb=[(r["id"],norm(prompt(r))) for r in d[b]]
        for ia,xa in pa:
            for ib,xb in pb:
                # RapidFuzz returns a 0-100 similarity score. The threshold
                # flags very close paraphrases while tolerating normal variation.
                if ratio(xa,xb)>=94:bad.append(f"near overlap {ia} <-> {ib}")
    # A nonzero exit status stops automated training workflows instead of
    # allowing contaminated data to continue silently.
    if bad:
        print("LEAKAGE CHECK FAILED");print("\n".join(bad[:100]));raise SystemExit(1)
    print("LEAKAGE CHECK PASSED")
# Standard script entry point.
if __name__=="__main__":main()
