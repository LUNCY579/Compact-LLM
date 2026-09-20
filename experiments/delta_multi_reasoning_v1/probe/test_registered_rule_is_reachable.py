"""A registered rule that no scoring path can reach governs nothing. This test proves it.

**Why this file exists.** GATEFIX-001 was registered at 2026-09-20T14:11:46Z and replaced GATE
v2's `references_retained` conjunct with a three-valued verdict. DOSELORA-001's arms were scored
at 17:49:46Z — **3h38m later, under the two-valued rule it replaced** — because the prereg
changed prose and nothing changed code.

The prereg's own banner said the arms *"would otherwise be the first scored under it"*, which
reads as though registering conferred protection. It did not.

> **A prereg cannot assert its own implementation. Something has to test it.**

Registering before the run was correct and is what makes the result recoverable at all. The
defect is that **nothing in the loop checked whether the registered rule was reachable from the
scoring path** — the same family as a battery printing COMPLETE over a failed leg, and as a
verification that leaves the worktree dirty: *the check nobody runs is the one after apparent
success.*

## What this asserts

Every scoring module that applies the `references_retained` conjunct must also carry the
three-valued vocabulary. A module that thresholds the endpoint without ever mentioning an
interval is applying the superseded rule.

## This test is EXPECTED TO FAIL until the rule is wired in

That is its purpose. A reachability test that passes on the day it is written tests nothing.
While it fails, the suite is correctly red: **a registered rule is inert and results are being
produced under a superseded one.** Do not skip or xfail it to green the suite — fix the scoring
path, or withdraw the registration.

Companion to `test_refuse_empty.py`, which pins a property invisible in normal use for the same
reason: every passing run exercises the path where the thing works.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BASE = Path("C:/Users/clclc/BusinessProject")

# Roots that may contain a module applying the conjunct. Missing roots are skipped, not failed —
# this test must not depend on which worktrees happen to exist on a given host.
SEARCH_ROOTS = (
    BASE / "ChatBubble-dose-param" / "research",
    BASE / "ChatBubble-dmrf" / "experiments",
)

# A module APPLIES the conjunct if it thresholds the endpoint, rather than merely printing it.
APPLIES = re.compile(r"references_retained\"?\]?\s*>=", re.I)

# GATEFIX-001's verdict is three-valued and interval-based. Any faithful implementation must
# name at least one of these; none of them can appear in a two-valued threshold comparison.
THREE_VALUED = ("CERTIFIED_PASS", "CERTIFIED_FAIL", "UNCERTIFIABLE",
                "clopper", "cp_lower", "interval_contains")


def _scoring_modules() -> list[Path]:
    out: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if p.name.startswith("test_"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if APPLIES.search(text):
                out.append(p)
    return sorted(out)


def test_a_scoring_module_that_applies_the_conjunct_exists():
    """If nothing applies it, this test is watching the wrong thing and must say so."""
    mods = _scoring_modules()
    assert mods, (
        "No module was found applying the references_retained conjunct. Either the scoring path "
        "moved or SEARCH_ROOTS is stale. A reachability test that cannot find its subject is a "
        "check whose scope cannot reach its claim — fix the roots rather than trusting the pass.")


@pytest.mark.parametrize("module", _scoring_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_registered_three_valued_rule_is_reachable(module: Path):
    """GATEFIX-001 is registered. Every module applying the conjunct must implement it."""
    text = module.read_text(encoding="utf-8", errors="replace")
    found = [tok for tok in THREE_VALUED if tok.lower() in text.lower()]
    assert found, (
        f"\n{module} applies the references_retained conjunct but carries NONE of the "
        f"three-valued vocabulary {THREE_VALUED}.\n\n"
        f"GATEFIX-001 (probe/GATEFIX001_PREREG.md, committed 2026-09-20T14:11:46Z) replaced the "
        f"binary verdict with CERTIFIED_PASS / UNCERTIFIABLE / CERTIFIED_FAIL. This module still "
        f"applies the superseded two-valued rule, so results it produces report a verdict the "
        f"registration withdrew.\n\n"
        f"This is not a style complaint. DOSELORA-001 was scored under the old rule 3h38m after "
        f"the new one was registered, and reported FAILED_PREDICTION where the registered rule "
        f"reports UNCERTIFIABLE.\n\n"
        f"Fix the scoring path or withdraw the registration. Do NOT xfail this to green the "
        f"suite — a registered rule that no scoring path can reach governs nothing.")
