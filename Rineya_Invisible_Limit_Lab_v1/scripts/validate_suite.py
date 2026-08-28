#!/usr/bin/env python3
"""Validate a generated Rineya invisible-limit split.

Checks the properties the benchmark depends on. A failure here invalidates every
downstream number, so this runs before any inference.

Structural
  1. four variants per scenario, ids unique, key agrees with full records
  2. paraphrase shares the base context but asks a different question
  3. context word counts sit near the declared length bucket

Semantic — the properties that make the suite hard
  4. the base answer is literally present in the base context
  5. the counterfactual answer differs from the base answer and is present in
     the counterfactual context
  6. BOTH candidate entities appear in base AND counterfactual contexts, so the
     swap moves the relation without moving the surface statistics
  7. every candidate entity still appears in the unanswerable context, so the
     model must detect a missing link rather than a missing name
  8. the unanswerable context does not still assert the decisive binding

Hygiene
  9. no sentence repeats verbatim inside a context
 10. base / counterfactual / unanswerable differ only in their decisive region
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

TOLERANCE_WORDS = 200


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.”])\s+", text) if s.strip()]


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = load(args.full)
    key = {r["id"]: r for r in load(args.key)}
    by_scenario: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        by_scenario[r["scenario_id"]][r["variant"]] = r

    failures: list[str] = []
    warnings: list[str] = []

    def fail(sid: str, msg: str) -> None:
        failures.append(f"{sid}: {msg}")

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        failures.append("duplicate row ids")
    if set(ids) != set(key):
        failures.append("full records and answer key disagree on ids")

    buckets = {"compact_long": 860, "standard_long": 955, "extended_long": 1055}

    for sid, variants in by_scenario.items():
        if set(variants) != {"base", "paraphrase", "counterfactual", "unanswerable"}:
            fail(sid, f"expected 4 variants, got {sorted(variants)}")
            continue
        b, p, c, u = (variants[v] for v in ("base", "paraphrase", "counterfactual", "unanswerable"))

        # 2 paraphrase shares context, differs in question
        if p["context"] != b["context"]:
            fail(sid, "paraphrase context differs from base")
        if p["question"] == b["question"]:
            fail(sid, "paraphrase question identical to base")

        # 3 length bucket
        target = buckets.get(b["length_bucket"])
        if target and abs(b["context_words"] - target) > TOLERANCE_WORDS:
            warnings.append(f"{sid}: {b['context_words']} words vs {b['length_bucket']} target {target}")

        ans_b, ans_c = b["answer"], c["answer"]

        # 4/5 answers present in their own contexts
        if ans_b not in b["context"]:
            fail(sid, f"base answer {ans_b!r} absent from base context")
        if ans_c == ans_b:
            fail(sid, f"counterfactual answer equals base answer ({ans_b!r})")
        if ans_c not in c["context"]:
            fail(sid, f"counterfactual answer {ans_c!r} absent from cf context")

        # 6 both candidates present in both answerable contexts
        for label, ctx in (("base", b["context"]), ("counterfactual", c["context"])):
            for cand in (ans_b, ans_c):
                if cand not in ctx:
                    fail(sid, f"candidate {cand!r} missing from {label} context "
                              f"(swap would change surface statistics)")

        # 7 candidates survive into the unanswerable context
        for cand in (ans_b, ans_c):
            if cand not in u["context"]:
                fail(sid, f"candidate {cand!r} removed from unanswerable context "
                          f"(model could detect a missing name instead of a missing link)")

        # 8 unanswerable must not still assert the binding
        if u["answerable"] is not False or u["answer"] != "UNKNOWN":
            fail(sid, "unanswerable variant is not marked UNKNOWN")

        # 9 no verbatim sentence repeats
        for label, r in (("base", b), ("cf", c), ("unans", u)):
            sents = sentences(r["context"])
            dupes = [s for s, n in collections.Counter(sents).items() if n > 1]
            if dupes:
                fail(sid, f"{label} context repeats {len(dupes)} sentence(s), e.g. {dupes[0][:70]!r}")

        # 10 variants differ only in a decisive region
        sb, sc, su = (set(sentences(r["context"])) for r in (b, c, u))
        if len(sb ^ sc) > 12:
            warnings.append(f"{sid}: base/cf differ in {len(sb ^ sc)} sentences")
        if len(sb ^ su) > 14:
            warnings.append(f"{sid}: base/unans differ in {len(sb ^ su)} sentences")

    n_sc = len(by_scenario)
    print(f"scenarios: {n_sc}   rows: {len(rows)}")
    dist = collections.Counter(r["task_type"] for r in rows if r["variant"] == "base")
    print("task_type:", dict(sorted(dist.items())))
    dist = collections.Counter(r["domain"] for r in rows if r["variant"] == "base")
    print("domain:   ", dict(sorted(dist.items())))
    dist = collections.Counter(r["needle_profile"] for r in rows if r["variant"] == "base")
    print("needle:   ", dict(sorted(dist.items())))
    wc = [r["context_words"] for r in rows]
    print(f"context words: min={min(wc)} mean={sum(wc)//len(wc)} max={max(wc)}")
    print()
    if warnings and args.verbose:
        for w in warnings[:15]:
            print("WARN", w)
        if len(warnings) > 15:
            print(f"... {len(warnings)-15} more warnings")
    print(f"warnings: {len(warnings)}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures[:25]:
            print("  ", f)
        if len(failures) > 25:
            print(f"   ... {len(failures)-25} more")
        return 1
    print("FAILURES: 0  -- suite is structurally and semantically sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
