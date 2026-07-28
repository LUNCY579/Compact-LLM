from __future__ import annotations
# This utility creates a deterministic, human-readable sample of the
# dataset so a reviewer can judge qualities that schemas cannot detect,
# such as naturalness, usefulness, and appropriate brevity.
import json,random
from collections import defaultdict
from pathlib import Path
# Resolve all project-relative paths from the repository root rather
# than from the terminal's current working directory.
ROOT=Path(__file__).resolve().parents[2]
# Load one JSONL split. Blank lines are ignored so harmless spacing in
# the file does not create empty records.
def read(s):
    with (ROOT/"data"/s/f"{s}.jsonl").open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
# Walk backward through the conversation (excluding the final assistant
# target) to find the user message that the target is responding to.
def user(r):return next(m["content"] for m in reversed(r["messages"][:-1]) if m["role"]=="user")
# Build the Markdown audit packet from balanced category samples.
def main():
    # A fixed seed makes the same records appear every time, which keeps
    # reviews reproducible across machines and repeated runs.
    rng=random.Random(20260727);out=["# HUMAN DATA AUDIT PACKET","","Mark each item `ACCEPT`, `REPAIR`, or `REJECT`.",""]
    n=0
    # Sample each split separately so reviewers can notice split-specific
    # problems without combining all records into one pool.
    for split in ("train","dev","sealed"):
        # Group records by reply class to prevent large categories from
        # dominating the human-review packet.
        by=defaultdict(list)
        for r in read(split):by[r["metadata"]["reply_class"]].append(r)
        # Inspect fewer sealed examples to reduce unnecessary exposure to
        # the final held-out set while still checking its data quality.
        k=3 if split!="sealed" else 1
        for cat,rows in sorted(by.items()):
            # Randomly sample within each category using the fixed seed.
            for r in rng.sample(rows,k):
                n+=1;out += [f"## {n}. `{r['id']}`","",f"- Split: `{split}`",f"- Category: `{cat}`",f"- User: {user(r)}",f"- Assistant: {r['messages'][-1]['content']}","- Decision: `PENDING`","- Notes:",""]
    # Write one reviewable Markdown artifact and print its path for the
    # calling PowerShell workflow.
    p=ROOT/"reports/HUMAN_DATA_AUDIT_PACKET.md";p.write_text("\n".join(out),encoding="utf-8");print(p)
# Run only when executed as a script, not when imported by tests/tools.
if __name__=="__main__":main()
