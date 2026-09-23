# HLOC-001 — is the boundary gain ATTENTION-carried or MLP-carried?

> **STATUS: REGISTERED, NOTHING BUILT, NOTHING SCORED.** Verified at this line: no `HSKILL2_*`
> artifact exists for ANY module-ablated arm anywhere in the records tree — searched against a
> positive control of **46 existing `HSKILL2_*` artifacts**, so the search can find things.

**Class: LOCALISATION. Zero training, zero budget grant, pure tensor arithmetic for the build.**
`quality_claim: false`, `release_permitted: false`. **Promotes nothing.**

---

## 1. Why now — two lanes converged on a question neither can answer alone

**`h025 vs control`, boundary +16, nd=24, p=1.544e-03, FLOORLESS is the ONLY certified positive
in ARMPERP-001's twelve cells.** δ_H buys boundary. Nobody has asked *where in the model*.

Three results make the question sharp:

| result | lane | says |
|---|---|---|
| δ_T, δ_H, δ_MIX allocate identically (cos ≥ 0.9998), MLP ~60% / attn ~40% | mine | **allocation is corpus-invariant** |
| attention 2.00× function-dense for acquisition, MLP 0.66× | DPLOC | density favours attention *there* |
| spreading beats attention-only by **+20 at matched norm** on the transition contract | sibling | **density is SKILL-SPECIFIC, not architectural** |
| the trailing-period convention is **attention-carried** — MLP-only donor at 60.8% of energy, coefficient 1.0, emits **0/192** periods | sibling | **attention carries the OUTPUT CONTRACT** |

> **The hypothesis those four suggest: ATTENTION CARRIES THE CONTRACT, MLP CARRIES THE CONTENT.**
> The boundary endpoint is a *contract* skill — it scores whether the model correctly ABSTAINS
> on an unanswerable item, not whether it computes anything. If the hypothesis holds anywhere in
> my programme, it holds here.

**The naive version is already falsified by my own data** and this prereg says so before
running: if attention carried contract skills, δ_H would be *allocated* more attention than δ_T.
It is not — 39.85% vs 40.16%, cos 0.9998. **So the hypothesis must be about FUNCTION at fixed
allocation, which is exactly what an ablation tests and an allocation measurement cannot.**

## 2. The arms — and the control that the sibling lane proved is mandatory

`P16(δ_H)` measured (canary: total `0.709894355545219` vs the recorded `0.7098943555451119`):

```
mlp    54.5869 %   ||.|| 0.524491        attn  45.3824 %   ||.|| 0.478231
norm    0.0306 %   ||.|| 0.012426        total ||P16(dH)|| 0.709894
```

**The two ablations are NOT norm-matched to each other or to `h025`.** Each therefore gets a
scalar control at *its own* second-component norm:

| arm | second component | ‖2nd‖ |
|---|---|---:|
| `hattn` | `0.25 · P16(δ_H)` restricted to **attention** | 0.119558 |
| `hattn_s` | `0.168418 · P16(δ_H)` **whole** — norm-matched to `hattn` | 0.119558 |
| `hmlp` | `0.25 · P16(δ_H)` restricted to **MLP** | 0.131123 |
| `hmlp_s` | `0.184708 · P16(δ_H)` **whole** — norm-matched to `hmlp` | 0.131123 |
| *(existing)* `h025` | `0.25 · P16(δ_H)` whole | 0.177474 |
| *(existing)* control | none | — |

First component is `P16(δ_T,s17)` at 1.0 in every arm, as in the whole family.

> **WHY THE SCALAR CONTROLS ARE NOT OPTIONAL.** In the sibling lane, `SPARSE_ATTN` scored 68
> against a norm-proportional expectation of ~41 and *would have read as confirmation of the
> module account*. `SCALAR_MATCHED` at the same norm scored 88 and **reversed the conclusion**.
> An ablation without a norm-matched scalar cannot distinguish "this module carries the skill"
> from "this ablation removed less norm".

## 3. Registered predictions

Primary endpoint **boundary** (HSKILL2, n = 112, `sensitivity_abstains_on_boundary`), exact
paired two-sided McNemar, Holm across the four registered comparisons.

**P2 (resolution precondition), registered FIRST because it bound the last lane:**
> A comparison with **discordance < 6 cannot reach p < 0.05** and is **UNRESOLVED**, not
> agreement. An unresolved limb **cannot satisfy any "not significantly different" clause**.
> This is written as a precondition on reading a limb, not a footnote on the verdict — in
> ARMPERP-001 the identical guard existed, fired correctly in the per-row table, and the verdict
> logic negated straight past it.

| outcome | requires | reading |
|---|---|---|
| **H_ATTN** | `hattn` > `hattn_s` significantly, **and** `hmlp` not > `hmlp_s` | The gain is **attention-carried**. Contract-in-attention confirmed on a second skill in a second lane. |
| **H_MLP** | `hmlp` > `hmlp_s` significantly, **and** `hattn` not > `hattn_s` | **MLP-carried** — contradicts the sibling lane's localisation and makes the contract account skill-specific too. |
| **H_DOSE** | neither ablation beats its scalar control | **Location does not matter at fixed norm**; the +16 is a dose effect on this endpoint, and the ablation axis is closed for it. |
| **H_BOTH** | both beat their controls | Distributed. Report both; claim no single carrier. |
| **INDETERMINATE** | any limb needed for the above is UNRESOLVED | Registered as a real outcome so an underpowered result cannot be read as H_DOSE. |

**My expectation, stated so it can be wrong: H_ATTN.** Registered against my own prior finding —
allocation invariance predicts *no* module should matter, so H_DOSE is what my own result
implies and H_ATTN is what the sibling lane's implies. **One of my two beliefs is going to lose.**

## 4. Build gates

1. `cos` and norms measured on the **built** tensors, not assumed; each scalar control's
   second-component norm must match its ablation's to **1e-9**.
2. `‖P16(δ_T,s17)‖` reproduces MAG-001's `0.674029920773733` (tol 1e-6) — the standing joint
   check on source + checkpoint + tensor mapping + SVD.
3. Layer coverage exactly 8–27, 220 tensors, matching the family.
4. **POSITIVE CONTROL, before any correct arm**: a deliberately mis-scaled variant of each
   scalar control (coefficient +1 %) must be **REJECTED** by gate 1.
5. **Ablation is REAL, asserted not assumed**: `hattn` must have **exactly zero** delta on every
   MLP tensor and `hmlp` exactly zero on every attention tensor. An ablation that silently
   ablated nothing is the failure this gate exists for.
6. Output sha256 distinct across all four arms and from every existing comparator.

## 5. What this does NOT license, whatever the numbers

- **Not a promotion.** Every arm here will sit in the UNCERTIFIABLE retention band.
- **Retention is NOT an endpoint** — same reason as ARMPERP-001: across all 28 pairs of this
  family the maximum discordance is 3 against the 6 an exact McNemar needs. Excluded in advance.
- **Not a claim about δ_T**, whose acquisition localisation DPLOC already measured. This is δ_H
  on the boundary endpoint, which has never been ablated.
- **Not transferable to the sibling lane's contract.** Different skill, different instrument,
  different base pairing. A localisation agreeing across the two would be a replication; one
  disagreeing would mean **carrier is skill-specific**, which is the more likely world given
  their P1' already showed density is.
- **`hattn` and `hmlp` differ from `h025` in TOTAL norm as well as location.** The scalar
  controls handle the location-vs-norm confound *within* each pair; they do **not** license
  comparing `hattn` to `h025` directly.

## 6. Envelope

Four arms, CPU tensor arithmetic, **built and scored and deleted one at a time** — peak disk
2.4 GB against 8.0 GB free, and the weights are rebuildable from the committed builder. One
probe leg each (boundary), ~80 s. The lease is taken for **builds as well as scoring**: a
streaming fp64 build over 220 tensors is not lighter than a forward pass on this box, which
this session established the hard way. `hostbudget` gate used unchanged and not tightened.

---

# Amendment 1 — H_DOSE conflates two patterns that read OPPOSITELY. Split, before any score.

**Filed with ZERO HLOC artifacts on disk — verified at this line: `hloc001-20260923` contains
no files, and the battery is still queued behind a peer's lease.** Prompted by that peer's P6
first arm, which landed while my run was queued.

## What was wrong

§3 registered **H_DOSE** as *"neither ablation beats its scalar control"* → *"location does not
matter at fixed norm."* That single cell covers two data patterns whose readings are opposite:

| pattern | correct reading |
|---|---|
| both ablations **indistinguishable** from their scalars | location does not matter — the registered reading |
| one or both ablations **significantly BELOW** their scalars | **restriction itself is harmful**, independent of which family — and location manifestly DOES matter |

**This is the ARMPERP-001 defect repeating**: an outcome cell whose stated reading does not hold
across all the data it would absorb. There I found it after the run; here it is caught before,
and only because a peer's result arrived first.

## The split

| outcome | requires | reading |
|---|---|---|
| **H_DOSE** | **both** ablations RESOLVED and **not** significantly different from their scalar controls | Location does not matter at fixed norm. The ablation axis is closed for this endpoint. |
| **H_RESTRICTION** | at least one ablation significantly **BELOW** its scalar control, and **neither above** | **Restriction to a single module family is harmful whichever family it is.** The carrier question is malformed for this endpoint: the manipulation damages something the matched scalar does not. |

H_ATTN, H_MLP, H_BOTH and INDETERMINATE are unchanged. P2 continues to bind: a limb with
discordance < 6 is UNRESOLVED and cannot satisfy any "not significantly different" clause, so
**H_DOSE now requires two RESOLVED nulls, not two absences of detection.**

## Why H_RESTRICTION is now the live outcome, and my prediction has moved TWICE

The sibling lane, on its own contract and bank, at 192 rows:

```
SPARSE_ATTN     attention only, 40.2% of energy    68
SPARSE_MLP      MLP only,       ~60% of energy     67      <- 50% MORE energy, ONE row apart
SCALAR_MATCHED  all modules, matched to SPARSE_ATTN 88
SPARSE          full delta                        102
```

> **Both restrictions land ~20 below the matched scalar, and EQUAL TO EACH OTHER despite one
> carrying 50 % more energy.** So the cost of restriction is not a function of how much norm
> survives it, and it is not attributable to either family.

**My stated expectation, and its history, recorded so it is visible I am not retrofitting:**

1. At registration I predicted **H_ATTN**, explicitly against my own allocation-invariance
   result (which implies H_DOSE), and said one of my two beliefs would lose.
2. On the peer's contract-obedience data (a matched-norm scalar installs the contract as
   completely as attention-only, 99.5 % either way) I **lowered confidence in H_ATTN**.
3. On their P6 first arm I now expect **H_RESTRICTION**.

All three movements are on *their* data, none on mine, and all three precede any HLOC score.
**H_ATTN is now my least likely outcome and it was my registered prediction** — which is worth
more than quietly updating it, because if H_ATTN fires anyway that is a genuine surprise.

## One additional registered comparison, descriptive

`hattn` vs `hmlp` directly. Their norms differ (0.119558 vs 0.131123, MLP ~10 % larger), so it
is not norm-matched and **no prediction is registered**. But if the sibling lane's one-row
equality replicates here across a ~10 % norm gap, that is the same "restriction cost is not a
function of surviving norm" signature on a second skill and a second instrument. Reported as
descriptive whatever it shows.
