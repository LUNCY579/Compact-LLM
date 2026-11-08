"""Measured structural capacity per (family, tier).

Quotas must be derived from what the generative space can actually supply, not chosen and
then discovered to be infeasible. This probe samples each cell hard, deduplicates by
canonical fingerprint, excludes the evaluation bank's structures, and reports how many
distinct structures exist per answer and per information-gap class.

The binding numbers it surfaces are properties of the task definitions, not preferences:
a 1-2 operation register can only be made answerable-despite-a-gap in a handful of ways,
and only two 4-job schedules with answer "3" remain once the eval bank has taken six of
the eight that exist.
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict

from . import families as F
from . import sampling
from .corpus import scout_exclusions

DRAWS = 60000


def probe(scout_worlds, draws=DRAWS, seed=7717):
    forbidden_fp, _ = scout_exclusions(scout_worlds)
    result = {}
    for family in F.FAMILIES:
        for tier in sampling.TIERS[family]:
            rng = random.Random(seed)
            seen, buckets = set(), Counter()
            outside = 0
            for _ in range(draws):
                world = sampling.sample(rng, family, tier)
                try:
                    answer = F.solve(world)
                except ValueError:
                    continue
                if answer not in F.ANSWERS:
                    outside += 1
                    continue
                fingerprint = F.fingerprint(world)
                if fingerprint in forbidden_fp or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                buckets[(answer, bool(F.answerability_cue(world)))] += 1
            per_answer = defaultdict(int)
            gap_answerable = 0
            for (answer, gap), count in buckets.items():
                per_answer[answer] += count
                if gap and answer != "UNKNOWN":
                    gap_answerable += count
            from .corpus import GAP_ANSWERABLE_SHARE
            uniform_cap = 5 * min(per_answer.get(a, 0) for a in F.ANSWERS)
            gap_cap = int(gap_answerable / GAP_ANSWERABLE_SHARE)
            result[f"{family}/{tier}"] = {
                "draws": draws, "distinct_structures": len(seen),
                "per_answer": dict(sorted(per_answer.items())),
                "gap_answerable_structures": gap_answerable,
                "answers_outside_vocabulary": outside,
                "uniform_balance_cap": uniform_cap,
                "gap_pool_cap": gap_cap,
                "recommended_cell_cap": min(uniform_cap, gap_cap),
                "enumerated_total": sampling.capacity_note(family, tier),
            }
    return result


def main():
    import json
    from experiments.multi_reasoning_scout_v1.core import RECORDS, decode
    worlds = [decode(line)["world"] for line
              in (RECORDS / "bank/worlds.jsonl").read_text(encoding="utf-8").splitlines()]
    report = probe(worlds)
    for cell, row in report.items():
        print(f"{cell:38s} distinct {row['distinct_structures']:>6d}  "
              f"uniform_cap {row['uniform_balance_cap']:>6d}  "
              f"gap_cap {row['gap_pool_cap']:>6d}  "
              f"-> cap {row['recommended_cell_cap']:>6d}  {row['per_answer']}")
    print()
    print(json.dumps({k: v["recommended_cell_cap"] for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
