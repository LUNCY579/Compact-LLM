# SCOREFIX-001 — report.py never implemented the registered three-valued retention rule

> **STATUS: REGISTERED AFTER THE ARMS WERE SCORED. This is disclosed first because it
> conditions everything below.** GATEFIX-001 was registered at commit `ffc8079d` before the
> DOSELORA run, but `report.py` was never changed to implement it, so
> `highdose-lora001-resume-20260920/RESULTS.json` (created 2026-09-20T17:49Z) reports the
> **superseded two-valued** verdict. This document does not move a threshold, does not
> re-score an arm, and **does not convert any failure into a pass.**

**Class: CODE-TO-RULE RECONCILIATION.** It changes no threshold, no endpoint, no band and no
prediction.

---

## 1. The defect

`GATEFIX001_PREREG.md`
(`ChatBubble-dmrf/experiments/delta_multi_reasoning_v1/probe/`, commits `ffc8079d` +
`0bce03e3`) makes the `references_retained` conjunct **three-valued** — CERTIFIED_PASS /
UNCERTIFIABLE / CERTIFIED_FAIL — by where the exact 95% Clopper–Pearson interval sits
relative to the threshold. It is marked *"REGISTERED, NOT YET APPLIED... committed before
the DOSELORA-001 launch whose arms would otherwise be the first scored under it."*

`report.py` never implemented it. In both `research/retention_highdose_lora_v1/report.py`
and `..._v2/report.py`:

```
report.py:307  retention_targets = retention["references_retained"] >= 40 and retention["panel_correct"] >= 62
report.py:315  "references_retained": prior.band(retention["references_retained"], (37, 45))
```

Grepping either file for `CERTIF`, `UNCERTIF`, `clopper`, `interval` or `confidence` returns
nothing. **The arms were therefore scored under the rule GATEFIX-001 replaced**, which is the
exact outcome GATEFIX-001 was registered to prevent.

## 2. What the registered rule gives on the observed value

Independently recomputed (exact Clopper–Pearson, pure-Python binomial inversion; reproduces
GATEFIX-001's published table to the decimal at k = 30, 35, 37, 40, 43):

| verdict | k | count |
|---|---|---:|
| CERTIFIED_FAIL | 0–30 | 31 |
| **UNCERTIFIABLE** | **31–42** | **12** |
| CERTIFIED_PASS | 43–47 | 5 |

The band is 12 of 48 possible outcomes = **25.0%**. (The separately quoted "76%" is the share
of the 84 arms *on record* that land there — a different quantity. Do not conflate them.)

**Observed: `references_retained` = 38/47, exact 95% CI = [31.37, 42.70] rows.**

- Against GATE v2's threshold **37**: 37 is inside [31.37, 42.70] → **UNCERTIFIABLE**.
- Against DOSELORA's primary target **40**: 40 is *also* inside → **not adjudicable either**.

`report.py` printed `references_retained >= 40 -> False` and the conjunction as
`FAILED_PREDICTION`. The conjunction did fail on this limb arithmetically, but **the
instrument cannot distinguish 38 from 40**, so reporting it as a failed prediction asserts a
resolution the endpoint does not have — the same defect GATEFIX-001 identifies for 37.

## 3. The deeper finding: the primary target is below its own certifiable minimum

| threshold | smallest k whose CI lower bound clears it |
|---|---:|
| 37 (GATE v2) | **43** |
| 40 (DOSELORA primary) | **45** |

DOSELORA's registered retention band is 37–45 and its primary target is `retained >= 40`.
**Only k = 45 — the single topmost value of the registered band — can certify against 40.**
An arm can satisfy the registered primary target at 40, 41, 42, 43 or 44 while the endpoint
abstains.

> **A target set below the smallest value its own endpoint can certify is not a target; it is
> a coin-flip with a number written on it.** This is the same shape as the recorded result
> that `references_retained` needs ≥ 6 discordant pairs and cannot reach significance even if
> the effect is entirely real, and the same shape as an arm excluded by 2 rows on a 6-row-MDE
> endpoint. It is the MDE-of-the-maximand defect, now hit for the third time.

This is stated as a finding, not a repair. **Changing the target is forbidden** — an arm has
been scored, and after generation an honest correction and moving the goalposts are
indistinguishable. It is recorded so the *next* registration sets a target its instrument can
certify, or declares up front that it is reporting a decomposition rather than a verdict.

## 4. What is registered here

1. `report.py` shall compute, for `references_retained` at the observed n, the exact 95%
   Clopper–Pearson interval and emit `CERTIFIED_PASS` / `UNCERTIFIABLE` / `CERTIFIED_FAIL` per
   GATEFIX-001, **alongside** the existing two-valued field, which is retained unchanged so
   prior outputs stay comparable. No threshold moves. No band moves.
2. `UNCERTIFIABLE` is **not** a pass and shall never be aggregated into one. A conjunction
   containing an UNCERTIFIABLE limb is reported as not adjudicable on that limb, never as
   PASS.
3. The existing `RESULTS.json` is **not** rewritten. It is annotated by reference to this
   document. Re-scoring a completed arm is out of scope.
4. This reconciliation is **outcome-neutral by construction**: it is computable from n = 47
   and the threshold alone and would read identically had no arm ever been trained. It cannot
   promote DOSELORA — 38 is UNCERTIFIABLE, and the primary conjunction remains un-passed
   either way.

## 5. Independent verification requested

Two implementations agreeing on a full row has caught real errors in this programme today.
The `Fine-tuning model audit and improvement` session independently reproduced the
k = 0..47 band (PASS ≥ 43, UNCERTIFIABLE 31–42, FAIL ≤ 30) and the k = 40/42/43 intervals
before this was written. The k = 38 interval [31.37, 42.70] and the two "smallest certifiable
k" figures (43 and 45) are, as of this registration, verified by one implementation only and
should be recomputed before being relied on.
