"""The two evaluation suites Phase 0 is run on, in a common row schema.

**demo** -- ``Rineya_Long_Context_Demo_v1`` (256 rows). Primary, because it is
the suite Supa-A1.1's published 45/125/170 is defined on, so condition A has an
exact expected value and the whole pipeline can be validity-gated against it.
Its ``supporting_facts`` are abstracted: 0 of 424 appear verbatim in their
context, so condition D must be localised by lexical overlap.

**lcr_dev** -- the dev split of ``Rineya_LCR_SFT_v1`` (800 rows, sampled).
Confirmatory, and it fixes the primary suite's two weaknesses: contexts are
longer (median 1409 words against 1020) and, decisively, all 2688 of its
supporting-facts entries are exact substrings of their context, so condition D
can be localised exactly rather than approximately.

The dev split is disjoint from the Phase 1 training split by construction, and
was verified disjoint from the demo suite (0 shared contexts, questions or
titles).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .provenance import DEMO_SUITE, REPO

LCR_SFT = (
    REPO.parent
    / "Chat-Bubble"
    / "Rineya_LCR_SFT_v1"
    / "Rineya_LCR_SFT_v1"
    / "data"
    / "lcr_sft_v1.jsonl"
)

LCR_DEV_SAMPLE = 200
LCR_DEV_SEED = 20260905


@dataclass
class Suite:
    key: str
    path: Path
    oracle_mode: str
    role: str
    note: str
    loader: Callable[[bool], list[dict]]


def _load_demo(smoke: bool) -> list[dict]:
    rows = [
        json.loads(line)
        for line in DEMO_SUITE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:32] if smoke else rows


def _load_lcr_dev(smoke: bool) -> list[dict]:
    """Sample whole scenario packs from the LCR dev split, seeded.

    Packs are kept intact so all four variants of a scenario are evaluated,
    matching the demo suite's paired structure.
    """
    packs: dict[str, list[dict]] = {}
    with LCR_SFT.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "dev":
                continue
            packs.setdefault(row["scenario_id"], []).append(row)

    ids = sorted(packs)
    rng = random.Random(LCR_DEV_SEED)
    rng.shuffle(ids)
    target = 8 if smoke else LCR_DEV_SAMPLE
    chosen: list[dict] = []
    for sid in ids:
        if len(chosen) >= target:
            break
        chosen.extend(packs[sid])

    from .training_data import UNKNOWN_SURFACE, short_target

    out = []
    for row in chosen[:target]:
        # LCR ships verbose sentence answers ("The final verified location of
        # navigation wafer ST-000-020 was ..."), but this lab always prompts with
        # the demo suite's brevity system message. Score against the same short
        # span the Phase 1 curriculum trains on, so prompt, target and scorer
        # agree; otherwise strict exact match would measure answer length.
        if row["answerable"]:
            accepted = [short_target(row)]
        else:
            accepted = [UNKNOWN_SURFACE, "unknown", "not determinable", "cannot be determined"]
        out.append(
            {
                **row,
                "difficulty": row.get("difficulty", "hard"),
                "answer": accepted[0],
                "accepted_answers": accepted,
                "lcr_original_answer": row["answer"],
            }
        )
    out.sort(key=lambda r: r["id"])
    return out


SUITES: dict[str, Suite] = {
    "demo": Suite(
        key="demo",
        path=DEMO_SUITE,
        oracle_mode="lexical",
        role="primary",
        note=(
            "Canonical suite for Supa-A1.1; condition A must reproduce "
            "45/64 Showcase, 125/192 Stability, 170/256 combined. "
            "supporting_facts are abstracted (0/424 verbatim), so D is "
            "localised by lexical overlap and is approximate."
        ),
        loader=_load_demo,
    ),
    "lcr_dev": Suite(
        key="lcr_dev",
        path=LCR_SFT,
        oracle_mode="exact",
        role="confirmatory",
        note=(
            "Held-out dev split of the Phase 1 curriculum, disjoint from its "
            "train split and from the demo suite. Longer contexts and 2688/2688 "
            "verbatim supporting_facts, so D is localised exactly. No published "
            "baseline exists for this split, so it carries no validity gate."
        ),
        loader=_load_lcr_dev,
    ),
}
