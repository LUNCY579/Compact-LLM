# GATEFIX-001 — GATE v2 abstains where it cannot adjudicate

> **STATUS: REGISTERED, NOT YET APPLIED.** No arm has been scored under this repair. Registered
> on the operator's instruction *"register the gate repair prereg before scoring anything"*, and
> committed **before** the DOSELORA-001 launch whose arms would otherwise be the first scored
> under it.

**Class: SELECTION RULE AMENDMENT.** It changes how a verdict is *reported*, and changes **no
threshold and no endpoint**.

---

## Disclosure, first, because it conditions everything below

**I have already seen the arm this repair would stop excluding.** D2 (768 rows) fails GATE v2
solely on `references_retained 35 < 37` while buying **+37 acquisition** on the dose bank.

Repairing a selection rule after seeing which arm it excludes is the move this programme's
discipline exists to prevent. Two things are done about that, and neither is a promise of good
faith:

1. **No threshold moves.** `62`, `37 of 47` and the family floors are untouched. Moving a
   threshold after seeing what it excludes is the forbidden act; changing the **verdict
   structure** on a power property is not, provided that property is arm-independent.
2. **The justification below is computable from `n = 47` and `T = 37` alone** — it was available
   the day the threshold was set, and would read identically had no arm ever been trained.
3. **The full consequence table over all 84 arms on record is published in §4 at registration
   time**, so the repair cannot be quietly tuned once its effects are inconvenient.

**This repair does not promote D2 and does not admit it.** It reclassifies its verdict from
FAIL to UNCERTIFIABLE, which is not a pass — see §3.

---

## 1. The defect, stated arm-independently

GATE v2 is `panel >= 62 AND references_retained >= 37 of 47 AND every family clears its floor`.

Ask a question that mentions no arm: **for which observed retained counts is the threshold 37
distinguishable from the arm's own exact 95% Clopper–Pearson interval at n = 47?**

| observed retained | exact 95% CI (rows) | verdict available |
|---:|---|---|
| ≤ 30 | e.g. 30 → [22.8, 36.3] | **certifiably FAIL** |
| **31 – 42** | e.g. 35 → [28.0, 40.4]; 37 → [30.2, 42.0]; 40 → [33.7, 44.1] | **NONE — CI contains 37** |
| ≥ 43 | e.g. 43 → [37.4, 45.9] | **certifiably PASS** |

> **The uncertifiable band is 31–42 — twelve of the forty-eight possible outcomes, and it
> STRADDLES the threshold itself. At n = 47 the conjunct cannot distinguish an arm scoring 37
> from the line it is being judged against.**

**Consequences, all arm-independent:**

- A gate that renders PASS/FAIL across 31–42 is reporting a verdict its own instrument cannot
  support. **The binary form asserts precision the endpoint does not have.**
- The threshold `37` is *itself* inside its own uncertifiable band. A line that cannot be
  distinguished from the values it separates is not measuring a difference; it is rounding one.

**This is the third appearance of the same shape** — `references_retained` was already the
endpoint that needed ≥ 6 discordant pairs and could not reach significance if the effect were
entirely real, and the same trap recurred in PRES-007's uniformity and concavity amendments.
Each time it was handled by annotating an individual result. **The defect is in the rule, not in
the results, and annotating results does not reach it.**

## 2. The second defect is already on the record and is NOT repaired here

`GATE_V2_PREREG.md` Finding 1: *"the panel conjunct cannot be satisfied by preserving the
source's behaviour. It requires improving on it by 15 rows."* **The frozen source scores
`panel = 47` and fails its own gate**, as do five function-preserving controls with 47/47
retained.

Decomposition (verified, `kept + NEW == panel` for all seven arms):

| arm | panel | kept /47 | **NEW** |
|---|---:|---:|---:|
| frozen source | 47 | 47 | 0 |
| D1 | 77 | 40 | **37** |
| D2 | 68 | 35 | 33 |
| DOSEINT α=0.25 | 61 | **40** | 21 |

> **`panel >= 62` is a TRANSFER-GAIN requirement, not a preservation requirement.** It is widely
> quoted as *"do not forget arithmetic"*; the source scores 47, so it means **"keep at least half
> of D1's transfer gain."**

**No repair is registered for this.** `62` stays, unmoved, for the same reason it was carried
into GATE v2 verbatim. **The change is to the LABEL and the REPORT**: the conjunct is named
`transfer_gain` and is always reported decomposed as `kept + NEW`, so it can no longer be read as
a retention term. Renaming a conjunct to what it measures is not re-tuning it.

## 3. The repair

```
For the references_retained conjunct ONLY, replace the binary verdict with three values,
using the exact 95% Clopper-Pearson interval on the observed count at n = 47:

    CERTIFIED_PASS   CI lies entirely ABOVE the threshold   (observed >= 43)
    UNCERTIFIABLE    CI CONTAINS the threshold              (observed 31-42)
    CERTIFIED_FAIL   CI lies entirely BELOW the threshold   (observed <= 30)

GATE v2 PASSES only on CERTIFIED_PASS.
UNCERTIFIABLE is NOT a pass. The gate ABSTAINS: it reports the count, the interval and the
other two conjuncts, and the decision escalates to the operator.
```

**UNCERTIFIABLE is deliberately not a pass.** Making it one would let every arm in the 31–42 band
through, which is 76% of the programme, and would be a threshold move wearing a verdict's name.
The gate stops *claiming* to have adjudicated; it does not start approving.

**The thresholds `62`, `37 of 47` and every family floor are unchanged. The bare policy remains
permissive and remains a disclosed pending dependency** (`GATE_V2_PREREG.md` Finding 4: under
strict, `source_correct_total` is 41 and no arm has ever reached 37).

## 4. Published consequences, at registration, over all 84 arms on record

| class | count | share |
|---|---:|---:|
| CERTIFIED_PASS (≥ 43) | **9** | 11% |
| UNCERTIFIABLE (31–42) | **64** | **76%** |
| CERTIFIED_FAIL (≤ 30) | 11 | 13% |

**The finding that makes the repair necessary rather than cosmetic:** the nine arms that
certifiably pass retention are near-null updates — `ced_last1`, `dual28_D56`, `dual28_H`,
`dual28_P` (function-preserving controls, 47/47 retained, **panel 47, failing the panel
conjunct**), `int001_a025` (retained 43, **fails 1 of 3 family floors**), and `int001_lora596`
(47/47 retained at **acquisition 70**, the under-converged LoRA arm).

> **No arm that clears the family floors has ever CERTIFIABLY passed the retention conjunct.**
> Every arm which does is one that barely moved.

**Specific reclassifications**, published now: **D1 40 → UNCERTIFIABLE** (currently PASS) and
**D2 35 → UNCERTIFIABLE** (currently FAIL). D4 at 30 → CERTIFIED_FAIL, unchanged in effect.

**Note that the repair removes a PASS as readily as a FAIL.** D1 is the programme's
gate-passing reference and this repair withdraws its certification too. A rule fitted to admit D2
would not have done that.

## 5. What would falsify the premise

- **If the uncertifiable band were narrow** — say 36–38 — the binary form would be defensible and
  this repair unnecessary. Measured, it is **31–42**.
- **If most arms sat outside the band**, the change would be cosmetic. Measured, **76% are inside
  it**.
- **If any floor-clearing arm certifiably passed retention**, the conjunct would be doing work.
  Measured, **none ever has.**

## 6. Scope, and what this does not do

- **It promotes nothing and admits nothing.** No arm changes from FAIL to PASS under it.
- It applies to `references_retained` at **n = 47** under the **permissive** policy. Under strict
  (`source_correct_total = 41`) the band must be recomputed before use.
- It does not touch acquisition, the family floors, long context or abstention.
- **It does not repair the panel conjunct**, which stays at 62 and is renamed only.
- The three-valued verdict is a **reporting** change. The underlying counts are unchanged and
  every historical artifact remains readable as recorded.

---

# AMENDMENT 1 — "repair" is the wrong word, and the constructive number

The paired session attacked this registration as asked. **The rule survived; the framing did
not.** Both changes below are theirs, accepted.

## 1. The arm-independence claim is now demonstrated rather than asserted

§1 argues the bands follow from `n = 47` and `T = 37` alone. The paired session derived them
**without loading any arm** and got the same boundaries:

    k = 30   CI [0.4852, 0.7733]   excludes T -> FAIL
    k = 31   CI [0.5069, 0.7914]   contains T -> UNCERTIFIABLE
    k = 42   CI [0.7690, 0.9645]   contains T -> UNCERTIFIABLE
    k = 43   CI [0.7962, 0.9763]   excludes T -> PASS

> **That reproduction IS the arm-independence claim.** A band fitted to D2 could not have been
> recomputed by someone who never loaded D2. Re-verified here under exact Clopper–Pearson beta
> quantiles rather than the original bisection: band **31–42**, unchanged.

## 2. Do not call it a repair. It is a disclosure.

The objection, in full because it is right:

> The gate now abstains on **76%** of arms and has never certified a floor-clearing one. That is
> not a fixed gate — it is an honest disclosure that **the retention conjunct never worked.**
> The operator still has to choose an arm, and the three-valued rule gives them nothing for the
> 76%.

**Three sentences that belong beside this registration, so nobody reads "repaired" as "usable":**

1. **The gate now ABSTAINS rather than works.** Silent is better than wrong; it is not the same
   as working.
2. **The certifiable-pass band IS the do-nothing band.** PASS needs retained ≥ 43/47 — **losing
   at most 4 source answers.** Every arm that has ever cleared the family floors loses 5 or more.
   **The certifiable-pass region and the learns-anything region do not intersect**, which is the
   cause of §4's nine-arm finding rather than a restatement of it.
3. **The instrument is roughly 3× too small to adjudicate the arms we actually build.**

## 3. The constructive number — how big it would have to be

Holding each arm's observed **rate** fixed and asking what `n` certifies it. Computed
independently by both sessions, agreeing on every row:

| arm | observed | rate | **n needed** | verdict there |
|---|---:|---:|---:|---|
| a025 | 43/47 | 0.9149 | **47** | PASS — already certifiable |
| Titan-Sparse | 41/47 | 0.8723 | **88** | PASS |
| **D1** | 40/47 | 0.8511 | **151** | PASS |
| **D2** | 35/47 | 0.7447 | **378** | FAIL |
| DOSEINT a050 | 37/47 | 0.7872 | **never** | rate **equals** the threshold exactly |

> **D1 becomes certifiable at 151 rows. D2 needs 378 to be certifiably rejected. The current
> instrument is 47.** So it is ~3× too small to certify what we build and ~8× too small to
> reject what we want to reject — and an arm sitting exactly on the threshold is **never**
> certifiable at any n.

## 4. The caveat that must travel with that number

**More rows of the same kind buy depth, not a second dimension.** Retention losses are
**nested** — observed nesting 0.3876 against a size-preserving random null of 0.2383,
permutation p = 0.0005 — so `references_retained` is a **depth gauge on roughly one latent
difficulty dimension.** Arms lose the same rows in the same order, to different depths.

> **151 rows drawn the same way would fix the certification arithmetic and would still be
> measuring one thing.** *"Make it 3× bigger"* must not be mistaken for *"make it good"*.

A larger instrument answers *"is this arm's retention distinguishable from the line?"* It does
**not** answer *"is this arm's retention different in kind?"* — nothing here measures that, and
a second dimension would have to be designed rather than sampled.

## 5. What this amendment does not change

The rule in §3 is unchanged: three-valued, `UNCERTIFIABLE` is not a pass, no threshold moves.
The objection was to the word, not the mechanism, and the word is corrected here rather than the
rule.

---

# AMENDMENT 2 — I registered a rule and never implemented it. The arms were scored under the old one.

**2026-09-20.** DOSELORA-001 ran at 17:49Z from an isolated clone. **Its arms were scored under
the two-valued rule this document replaced**, and that is my failure, not the executing
session's.

## The defect

This document's own banner reads:

> *"committed **before** the DOSELORA-001 launch whose arms would otherwise be the first scored
> under it."*

**That sentence assumes registration is sufficient. It is not.** `report.py:307` still reads

    retention_targets = retention["references_retained"] >= 40 and retention["panel_correct"] >= 62

with **zero occurrences** of `CERTIF`, `UNCERTIF`, `clopper`, `interval` or `confidence` anywhere
in the file. I changed a rule in prose and changed nothing that executes.

> **A registered rule that is not implemented in the scoring code governs nothing. Registration
> and implementation are separate acts and I conflated them** — and the conflation is invisible
> precisely because the prereg *reads* as though the protection is in place.

This is the same shape as the battery printing `COMPLETE` over a failed leg, and as a
verification that leaves the worktree dirty: **the step that would have caught it is the one
nobody runs after the thing appears to have succeeded.**

## The observed value lands in the band, and the conjunction failed on the limb that cannot resolve

DOSELORA's `references_retained = 38/47`. Reproduced here:

    exact 95% CI -> rows [31.37, 42.70]
      threshold 37 (GATE v2)          INSIDE  -> UNCERTIFIABLE
      threshold 40 (DOSELORA primary) INSIDE  -> UNCERTIFIABLE

`report.py` printed `FAILED_PREDICTION` for the primary conjunction, **failing solely on that
limb** — acquisition (291 and 296 against ≥ 284) and arithmetic (74 against ≥ 62) both cleared.

> **The conjunction failed on the one limb its own endpoint cannot adjudicate.** §1 of this
> document argued that would happen. It has now happened in the wild, to a real arm, and the
> verdict on the record is the superseded one.

## A third appearance of the shape, on a TARGET rather than a gate

Smallest counts that can certify at n = 47, verified independently here:

| threshold | certifiable PASS | certifiable FAIL |
|---:|---:|---:|
| 37 (GATE v2) | k ≥ **43** | k ≤ 30 |
| **40 (DOSELORA primary)** | k ≥ **45** | k ≤ 34 |

**DOSELORA's registered primary target of ≥ 40 sits below its own certifiable minimum of 45**,
and its registered band was 37–45 — of which **only the single value 45 could ever certify
against it.** The defect is not confined to GATE v2; it recurs wherever a threshold on this
endpoint is written without asking what n = 47 can resolve.

## A conflation warned about, and checked

The DOSELORA session flagged that two different quantities are easy to merge. Confirmed distinct
in this document and restated here so they cannot drift:

| quantity | value |
|---|---|
| UNCERTIFIABLE counts 31–42, as a share of **possible outcomes** | 12 of 48 = **25.0%** |
| share of the **84 recorded arms** that land there (§4) | **76%** |

§1 says *"twelve of the forty-eight possible outcomes"* and §4 says 76% of arms; they were not
conflated here, and the warning is recorded because the two are one careless sentence apart.

## Ownership of the fix

The executing session registered the code-to-rule reconciliation as
`ChatBubble-dose-param/research/retention_highdose_lora_v2/SCOREFIX001_PREREG.md` (`9ab7ad5f`)
and offered to hand it to me as GATEFIX-002 since the rule is mine.

**They should keep it.** It moves no threshold, does not rewrite `RESULTS.json`, keeps the
two-valued field for comparability, states that UNCERTIFIABLE is never aggregated into a pass,
and discloses up front that it was registered *after* the arms were scored. It is
outcome-neutral — 38 is UNCERTIFIABLE either way, so it promotes nothing. **It was registered
where the author could actually act, which is the right instinct**, and moving it here would be
territorial rather than useful.

**What is mine is this amendment: the rule was registered and never wired in.**

---

# AMENDMENT 3 — "outcome-neutral" is WITHDRAWN. The defence is ordering, not neutrality.

Amendment 2 endorsed SCOREFIX-001 as *"outcome-neutral — 38 is UNCERTIFIABLE either way, so it
promotes nothing."* **That sentence is withdrawn.** Its author retracted the premise at
`ec4b6602` before Amendment 2 was written, and they are right: it contains a true clause and a
false inference.

**True:** 38 is UNCERTIFIABLE against both thresholds. Reproduced here —

    T=37 : certifiable FAIL k<=30 | UNCERTIFIABLE 31-42 (12 values) | PASS k>=43
    T=40 : certifiable FAIL k<=34 | UNCERTIFIABLE 35-44 (10 values) | PASS k>=45
    boundary: k=34 hi=39.66 < 40 ; k=35 hi=40.45 > 40

**False:** that this makes the change outcome-neutral. **The comparison that matters is the OLD
verdict against the NEW one, not UNCERTIFIABLE@37 against UNCERTIFIABLE@40.** `report.py`
printed `FAILED_PREDICTION` at k = 38; the three-valued rule reports `UNCERTIFIABLE`.
**The reported outcome changes at the observed value, in the softening direction.** Neutrality at
k = 38 is precisely what is not true.

## Full crosswalk — old two-valued (`k >= 40` PASS, as `report.py` applies it) vs new three-valued at T = 37

| k | old | new | effect |
|---|---|---|---|
| 0–30 | FAILED | CERTIFIED_FAIL | agree |
| **31–39** | FAILED | UNCERTIFIABLE | **softens a recorded failure** — *38 observed* |
| **40–42** | PASS | UNCERTIFIABLE | **withdraws a pass** |
| 43–47 | PASS | CERTIFIED_PASS | agree |

*(The two limbs use different thresholds — `report.py` applies DOSELORA's primary target of 40,
GATE v2's conjunct is 37 — so this crosswalk mixes a threshold change with a verdict-structure
change. Both are stated so neither is hidden.)*

> **No k moves into CERTIFIED_PASS that was not already ≥ 40. So the defensible claim is
> "CANNOT PROMOTE", not "outcome-neutral".** The change softens failures and withdraws passes; it
> creates none.

## The defence is ORDERING, and it is stronger than the one it replaces

    GATEFIX-001            ffc8079d   2026-09-20T14:11:46Z
    amendment 1            0bce03e3   2026-09-20T14:20:28Z
    DOSELORA renewal                  2026-09-20T16:51:53Z
    permission transferred            2026-09-20T16:57:10Z
    RESULTS.json (k = 38)             2026-09-20T17:49:46Z

**The rule predates the data by 3h38m.** That, and only that, is what separates this from
repairing a rule after seeing which arm it excludes. **Resting the defence on neutrality was
weaker AND false; resting it on ordering is stronger AND true.**

This matters more for this document than for SCOREFIX-001, because **this is the rule.**

## Refinement to Amendment 2's self-criticism, and it is a real correction

Amendment 2 treated the banner sentence — *"committed before the DOSELORA-001 launch"* — as the
defect. It is not. **Registering before the run was correct and is what makes the result
recoverable at all.**

> **The defect is that nothing in the loop checks whether a registered rule is REACHABLE from the
> scoring path. That is a missing check, not a wrong claim.** A prereg cannot assert its own
> implementation; something has to test it.

## The missing check now exists, and it fails in FOUR places

`probe/test_registered_rule_is_reachable.py`. It finds every module that *applies* the conjunct
(thresholds it, rather than merely printing it) and asserts the three-valued vocabulary is
present. Current state:

    FAILED  retention_dose_parameterization_v1/report_battery.py
    FAILED  retention_highdose_lora_resume_v1/report.py
    FAILED  retention_highdose_lora_v1/report.py
    FAILED  retention_highdose_lora_v2/report.py

**Four, not the three this document had named** — `report_battery.py` was not on anyone's list.

**The test is expected to fail until the rule is wired in, and the suite is correctly red while
it does.** A reachability test that passes on the day it is written tests nothing. It must not be
skipped or xfailed to green the suite: fix the scoring path, or withdraw the registration.
