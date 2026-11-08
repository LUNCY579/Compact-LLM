"""GATEFIX-001's three-valued retention verdict. ONE implementation, imported by every report.

**Why this module exists and why it is not four copies.** GATEFIX-001 was registered at
`ffc8079d` (2026-09-20T14:11:46Z) and replaced GATE v2's `references_retained` conjunct with a
three-valued verdict. `report.py` never implemented it, so DOSELORA-001's arms were scored at
17:49:46Z — 3h38m later — under the superseded two-valued rule.

`probe/test_registered_rule_is_reachable.py` in the dmrf worktree found **four** modules applying
the conjunct, and the three `report.py` copies had **already diverged from each other** (distinct
sha256). Adding a fourth and fifth copy of a rule is how the rule drifts, which is the failure
this whole episode is about. So the rule lives here once.

## What is registered, and what this module deliberately does NOT do

Per `SCOREFIX001_PREREG.md` §4, which reconciles code to `GATEFIX001_PREREG.md`:

  1. Emit CERTIFIED_PASS / UNCERTIFIABLE / CERTIFIED_FAIL **alongside** the existing two-valued
     field, which is retained unchanged so prior outputs stay comparable.
  2. **UNCERTIFIABLE is never aggregated into a pass.** A conjunction with an UNCERTIFIABLE limb
     is reported as not adjudicable on that limb.
  3. `RESULTS.json` is **not** rewritten. Re-scoring a completed arm is out of scope.
  4. The verdict is emitted against **37** (the GATE v2 conjunct). `report.py`'s `>= 40` is
     DOSELORA's own primary target — a different rule on the same endpoint which GATEFIX-001
     does not govern and which this does **not** redefine.

**No threshold moves here.** This module adds a verdict; it changes no criterion.

## The interval

Exact Clopper-Pearson, computed by bisection on the exact binomial tail — pure stdlib, because
these reports are required to run in a fresh process with no third-party imports. At n = 47 this
is two bisections of ~200 iterations over a 48-term sum: microseconds.

`self_test()` reproduces GATEFIX-001's published band (CERTIFIED_FAIL <= 30, UNCERTIFIABLE
31-42, CERTIFIED_PASS >= 43 at T = 37) and is called on import. **A rule module that cannot
reproduce its own registration refuses to load** — the canary rule applied to the implementation
rather than to an extractor.
"""
from __future__ import annotations

from math import comb

GATE_V2_THRESHOLD = 37
CERTIFIED_PASS = "CERTIFIED_PASS"
UNCERTIFIABLE = "UNCERTIFIABLE"
CERTIFIED_FAIL = "CERTIFIED_FAIL"


def _binom_tail_ge(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _binom_tail_le(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval on a binomial proportion, by tail inversion."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"clopper_pearson: k={k} n={n} out of range")
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if _binom_tail_ge(k, n, m) > alpha / 2:
                b = m
            else:
                a = m
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if _binom_tail_le(k, n, m) < alpha / 2:
                b = m
            else:
                a = m
        hi = (a + b) / 2
    return lo, hi


def three_valued(k: int, n: int, threshold: int = GATE_V2_THRESHOLD) -> dict:
    """GATEFIX-001's verdict: where does the exact 95% interval sit relative to the threshold?"""
    lo, hi = clopper_pearson(k, n)
    lo_rows, hi_rows = lo * n, hi * n
    if lo_rows > threshold:
        verdict = CERTIFIED_PASS
    elif hi_rows < threshold:
        verdict = CERTIFIED_FAIL
    else:
        verdict = UNCERTIFIABLE
    return {
        "verdict": verdict, "observed": k, "n": n, "threshold": threshold,
        "interval_rows": [round(lo_rows, 2), round(hi_rows, 2)],
        "registration": "GATEFIX001_PREREG.md (ffc8079d, 0bce03e3); "
                        "reconciled by SCOREFIX001_PREREG.md",
        "note": ("UNCERTIFIABLE is NOT a pass. The gate abstains: the threshold lies inside the "
                 "arm's own exact 95% interval, so the endpoint cannot adjudicate it in either "
                 "direction at this n."),
    }


def conjunction_adjudicable(*verdicts: str) -> dict:
    """A conjunction with an UNCERTIFIABLE limb is not adjudicable. It is never a PASS."""
    unc = [v for v in verdicts if v == UNCERTIFIABLE]
    return {
        "adjudicable": not unc,
        "why": ("all limbs certified" if not unc else
                f"{len(unc)} limb(s) UNCERTIFIABLE -- the conjunction is NOT adjudicable on "
                f"those limbs and must never be reported as PASS"),
    }


def self_test() -> None:
    """Reproduce GATEFIX-001's published band, or refuse to load."""
    n, t = 47, GATE_V2_THRESHOLD
    band = [k for k in range(n + 1) if three_valued(k, n, t)["verdict"] == UNCERTIFIABLE]
    if (min(band), max(band), len(band)) != (31, 42, 12):
        raise AssertionError(
            f"retention_verdict does not reproduce GATEFIX-001's registered band: got "
            f"{min(band)}-{max(band)} ({len(band)} values), registered 31-42 (12 values). "
            f"Refusing to load a rule module that disagrees with its own registration.")
    if three_valued(38, n, t)["verdict"] != UNCERTIFIABLE:
        raise AssertionError("k=38 must be UNCERTIFIABLE at T=37 per the registered table")
    if three_valued(43, n, t)["verdict"] != CERTIFIED_PASS:
        raise AssertionError("k=43 must be CERTIFIED_PASS at T=37 per the registered table")
    if three_valued(30, n, t)["verdict"] != CERTIFIED_FAIL:
        raise AssertionError("k=30 must be CERTIFIED_FAIL at T=37 per the registered table")


self_test()
