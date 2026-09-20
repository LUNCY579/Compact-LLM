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

| threshold | certifiable FAIL | UNCERTIFIABLE | certifiable PASS |
|---|---|---|---|
| 37 (GATE v2) | k ≤ 30 | 31–42 (12 values) | k ≥ **43** |
| 40 (DOSELORA primary) | k ≤ 34 | **35–44 (10 values)** | k ≥ **45** |

(Both rows independently reproduced by the `Rineya Delta multi-reasoning experiment` session.)

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
4. **Scope.** GATEFIX-001's threshold is **37** (the GATE v2 conjunct). `report.py:307`'s
   `>= 40` is DOSELORA's own primary target — a *different* rule on the same endpoint, which
   GATEFIX-001 does not govern and which **this document does not redefine**. The
   three-valued verdict is emitted against 37. The separate observation that 40 also falls
   inside the k=38 interval is reported descriptively, as a statement about the primary
   target's resolvability; it is **not** a re-scoring of the primary conjunction.

## 4a. Which direction this could have gone — the disclosure that matters

An earlier draft of this document called the reconciliation *"outcome-neutral by
construction."* **That was wrong and is withdrawn.** It is not outcome-neutral in general,
and it is not even outcome-neutral at k = 38: it changes the reported retention verdict from
`FAILED_PREDICTION` to `UNCERTIFIABLE`, which is a change in the softening direction. Stating
otherwise would have understated exactly what a later reader should be checking.

Full crosswalk, `report.py` two-valued at 40 versus GATEFIX-001 three-valued at 37:

| k | two-valued (≥40) | three-valued (T=37) | effect of the change |
|---:|---|---|---|
| ≤ 30 | FAILED | CERTIFIED_FAIL | agree |
| **31–39** | FAILED | UNCERTIFIABLE | **softens a recorded failure** (k=38 observed) |
| **40–42** | PASS | UNCERTIFIABLE | **withdraws a pass** |
| 43–47 | PASS | CERTIFIED_PASS | agree |

So the change cuts **both** ways: it would have withdrawn a pass at 40, 41 or 42, and it
softens a failure at 31–39. It can never manufacture a pass — no k moves into
CERTIFIED_PASS that was not already ≥ 40 — but "cannot promote" is a weaker and more honest
claim than "outcome-neutral", and it is the only one this document is entitled to.

**The justification is therefore ordering, not neutrality.** GATEFIX-001 was registered at
commit `ffc8079d` **before** the run that produced k = 38; only the *code* lagged the rule.
That single fact is what separates this from repairing a rule after seeing which arm it
excludes. *"Registered after the arms were scored"* and *"registered after seeing which arms
it would exclude"* read identically to a later reader, and only the first is true here — so
the commit ordering must be checked, not taken on trust. Verified:

| event | time (UTC) |
|---|---|
| GATEFIX-001 registered, `ffc8079d` | **2026-09-20T14:11:46Z** |
| GATEFIX-001 amendment 1, `0bce03e3` | 2026-09-20T14:20:28Z |
| renewal `PREREG.md` written (resume root) | 2026-09-20T16:51:53Z |
| one-launch permission transferred | 2026-09-20T16:57:10Z |
| `RESULTS.json` created — k = 38 first exists | **2026-09-20T17:49:46Z** |

The rule predates the registration of the run by 2 h 40 min and the data by 3 h 38 min.
Reproduce with `git -C ChatBubble-dmrf log -1 --format=%cI ffc8079d` against `RESULTS.json`'s
`created_utc`.

## 4b. Implementation scope — two of the four modules must stay red

A reachability test written by the GATEFIX-001 author
(`ChatBubble-dmrf/.../probe/test_registered_rule_is_reachable.py`) finds **four** modules that
*apply* the conjunct rather than merely print it. All four are in
`ChatBubble-dose-param/research/`. Checked before editing anything:

| module | status | may it be wired? |
|---|---|---|
| `retention_highdose_lora_resume_v1/report.py` | `sha256 de752b9a…`, **bound** in the resume root's `PRELAUNCH_BINDING.json` *and* recorded as `RESULTS.json`'s `reporter_sha256` | **NO — sealed evidence** |
| `retention_highdose_lora_v1/report.py` | preserved never-run root; unbound, but byte-intactness is the point of that root | no, by preservation |
| `retention_highdose_lora_v2/report.py` | unbound, lane is DO-NOT-RUN | possible, but inert |
| `retention_dose_parameterization_v1/report_battery.py` | unbound, **live scoring path**, on nobody's list until the test found it | **the only one where wiring changes anything** |

> **A scoring module that is sealed evidence of a completed run cannot be fixed and must not be.
> Editing `resume/report.py` would make the recorded `reporter_sha256` unreproducible from the
> working tree and leave the only DOSELORA arm on record unverifiable. Fixing it is strictly
> worse than leaving the test red.**

That is a third case the test's doctrine (*"fix the scoring path, or withdraw the
registration"*) does not admit, and it must be excluded by **provenance, not convenience** —
exclude a module if and only if its hash is recorded in a completed run's receipt or pinned in
a `PRELAUNCH_BINDING.json`. That criterion is checkable from the artifacts and cannot be
invoked by whoever finds the test inconvenient, so it does not reopen the xfail loophole.

A module that is *both* sealed *and* superseded is a finding, not a backlog item: it records
that a published result was produced under a rule that has since been replaced. It deserves a
distinct status — `SEALED_UNDER_SUPERSEDED_RULE` — reported apart from genuinely-inert
modules, so the red count stays actionable.

**Consequence:** the three-valued rule is currently reachable from exactly **one** live scoring
path, `report_battery.py`, and that module belongs to the dose-parameterization lane rather
than to DOSELORA. Wiring it is therefore a change to another experiment's scoring path and is
**not** taken unilaterally here.

## 5. Independent verification requested

Two implementations agreeing on a full row has caught real errors in this programme today.
The `Fine-tuning model audit and improvement` session independently reproduced the
k = 0..47 band (PASS ≥ 43, UNCERTIFIABLE 31–42, FAIL ≤ 30) and the k = 40/42/43 intervals
before this was written. The k = 38 interval [31.37, 42.70] and the two "smallest certifiable
k" figures (43 and 45) are, as of this registration, verified by one implementation only and
should be recomputed before being relied on.
