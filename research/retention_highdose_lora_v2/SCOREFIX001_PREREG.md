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

### 3a. What the endpoint would need — the constructive number

The complaint "this endpoint cannot certify" is only useful with a bank size attached. Exact
Clopper–Pearson, α = 0.05, k = round(p·n). Verified against the n = 47 values computed
independently above (k=38 → 31.37, k=43 → 37.42, k=45 → 40.17, k=30 → 22.80, all exact):

| arm's true retention rate | n to certify vs T = 37/47 | n to certify vs T = 40/47 |
|---|---:|---:|
| 38/47 = 0.809 (DOSELORA observed) | 1402 † | **never** — rate is below the threshold |
| 42/47 = 0.894 | 58 ‡ | 258 ‡ |
| 44/47 = 0.936 | 47 ‡ | **66** ‡ |
| 45/47 = 0.957 | 47 ‡ | 47 ‡ |

**‡ Cross-confirmed by two independent numerical methods.** The published figures come from
incomplete-beta (Lentz continued-fraction) inversion. An earlier, slower implementation using
direct exact-binomial CDF summation — written first, and run without reference to the beta
implementation's output — independently produced **58 / 258, 47 / 66, 47 / 47**, agreeing on
every digit of all three rows. It then aborted with `OverflowError: int too large to convert to
float` while attempting the 38/47 row, where `math.comb` overflows at n ≈ 1402.

**† The 1402 figure is therefore single-method and should be treated as such** until someone
re-derives it with exact arithmetic (`fractions.Fraction`, or log-space summation). It is also
the least load-bearing number in the table — no programme decision depends on whether an
0.809-rate arm needs 1400 rows or 1600.

The **"never"** cell needs no numerical confirmation at all: 38/47 = 0.809 is *below* the
threshold rate 40/47 = 0.851, so the interval's lower bound cannot exceed the threshold at any
n. That is a logical fact about the point estimate, not a computation.

**The actionable line is the third.** A genuinely strong arm — one truly at 44/47 — is
*already* certifiable against the gate threshold at n = 47, and needs only **66 rows** to
certify against the primary target. Going 47 → 66 is +19 rows, which is a bank-construction
task, not a research programme.

At the current n = 47, only an arm at **≥ 45/47** can certify against the primary target. That
is the whole reason `retained ≥ 40` reads as a target while behaving as a coin-flip: the
instrument admits exactly three passing outcomes out of forty-eight.

This is offered as a specification for the *next* retention bank, not as a change to this one.
No threshold here moves.

> **CAVEAT, and it is load-bearing: more rows buy certification arithmetic, not a second
> dimension.** `references_retained` losses are **nested** (0.3876 against a 0.2383
> size-preserving null, p = 0.0005), so the endpoint behaves as a depth gauge on roughly one
> latent dimension. A table saying "n = 66 makes the target certifiable" is true and will be
> misread as "n = 66 makes the endpoint good." It does not. Rows of the same construction
> sharpen the same measurement; they do not add an axis. Widening the *construction* is a
> separate and unsolved problem — see the third relational class still missing from the floors.

### 3b. What the record actually contains at k ≥ 43

Swept every `references_retained` observation in the records tree — 41 observations, range
20–47. Only three sit at 42 or above, all in `alias001-20260918/COUPLING_LADDER.json`:

| α | references_retained | acquisition | frac acquisition gained | arithmetic (permissive) |
|---:|---:|---:|---:|---:|
| 0.00 | **47** | 14 | 0.000 | 47 |
| 0.25 | **43** | 145 | 0.593 | 59 |
| 0.50 | 42 | 219 | 0.928 | 67 |

The 47 is the **frozen source** — certifiable, and it learned nothing (acquisition 14, zero
fraction gained). But **α = 0.25 reaches exactly 43 with acquisition 145**, i.e. one arm in the
record *is* certifiable against T = 37 while having acquired substantially. The stronger claim
that *no* arm has ever both certified and learned is therefore **too strong**; the accurate one
is that this arm sits at `arithmetic 59 < 62`, so it has never been a *simultaneously* passing
arm. It also lands on exactly the minimum certifiable value — one row lower and it abstains.

**The ladder itself is NOT a new result and must not be presented as one.** The same
non-linearity is already on the record from INT-001, read downward in *alias* terms:
1.00 → 0.75 gives up 1 acquisition row to recover 11 alias (10.98 : 1); 0.75 → 0.50 is
0.93 : 1; 0.50 → 0.25 is 0.26 : 1. Reading it upward in *retention* terms — α 0 → 0.25 buys
+131 acquisition for 4 retention rows, α 0.25 → 0.50 only +74 more for 1 more row while alias
damage triples — is the same knee on a different preservation axis, not a discovery.

**What is new here is only the certifiability overlay:** α = 0.25's retained 43 is *exactly*
the minimum certifiable value at n = 47 against T = 37. That single fact is what refutes the
stronger claim; the ladder underneath it was already written down.

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

| module | sha256 | status |
|---|---|---|
| `retention_dose_parameterization_v1/report_battery.py` | `ae4802a7…` | **SEALED** — `doseint001-20260919/RESULTS.json:reporter_sha256` |
| `retention_highdose_lora_resume_v1/report.py` | `de752b9a…` | **SEALED** — resume `PRELAUNCH_BINDING.json:code_files_sha256[report.py]` *and* `RESULTS.json:reporter_sha256` |
| `retention_highdose_lora_v1/report.py` | `243707b8…` | LIVE (in no receipt); preserved never-run root |
| `retention_highdose_lora_v2/report.py` | `45fc4c86…` | LIVE (in no receipt); lane is DO-NOT-RUN |

> **CORRECTION.** An earlier version of this table called `report_battery.py` *"unbound, the
> only one where wiring changes anything."* **That was wrong and is withdrawn.** It is the
> reporter of record for DOSEINT-001 — the parent experiment whose checkpoint-00193 the entire
> DOSELORA norm control is built from. Wiring it would have broken that result's custody chain.
>
> The error was method, not arithmetic: I checked one root's `PRELAUNCH_BINDING.json`, saw the
> module absent, and generalised "not in *that* receipt" to "in no receipt." **The silence of
> an out-of-scope check is not evidence.** Classification must sweep *every* receipt in the
> records tree — 53 distinct receipt-pinned hashes exist — which is what the artifact-computed
> exclusion in `test_registered_rule_is_reachable.py` (dmrf `21c7602c`) now does.

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

**Consequence, after the correction above: there is currently NO live, unsealed, consequential
scoring path to wire the rule into.** Both sealed modules are reporters of record and must not
be touched; both unsealed modules are a preserved never-run root and a DO-NOT-RUN lane, where
wiring is symbolic.

So **the registered rule presently governs nothing, and cannot be made to govern anything
without either destroying evidence or building a new scoring path.** The next live path that
scores this endpoint will be the first the rule actually governs, and wiring it there is a
build gate for that path — not a retrofit to an existing one. That is the honest state; it is
recorded rather than resolved, because every available way to resolve it today is worse than
leaving it visible.

## 5. Independent verification requested

Two implementations agreeing on a full row has caught real errors in this programme today.
The `Fine-tuning model audit and improvement` session independently reproduced the
k = 0..47 band (PASS ≥ 43, UNCERTIFIABLE 31–42, FAIL ≤ 30) and the k = 40/42/43 intervals
before this was written. The k = 38 interval [31.37, 42.70] and the two "smallest certifiable
k" figures (43 and 45) are, as of this registration, verified by one implementation only and
should be recomputed before being relied on.
