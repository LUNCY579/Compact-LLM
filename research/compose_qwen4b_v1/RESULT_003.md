# COMPOSE4B-003 — the control that could have retracted me FIRED, and a 596M withdrawal only HALF-replicates

**2026-09-22 · 2 new arms × 224 rows · zero training · registered in `PREREG_003.md`, committed
with `scored/` verifiably empty and the records directory containing that file alone.**

`quality_claim` false. `release_permitted` false. **No arm promoted.**

---

## 1. The canary, before any 003 number was read

`analyze_003.py` was committed with `scored/` empty. Its statistics are **imported** from
`analyze.py`; only the row extraction is new, because HSKILL's schema differs from the
acquisition schema. That is exactly where this programme keeps getting hurt — four wrong
extractors in one day, none caught by reading code — so the script **refuses to read a 003
number** unless it first reproduces COMPOSE4B-002's three published contrasts:

| check | published | recomputed |
|---|---|---|
| `th_s − scaled_t` sensitivity | +19, nd 31, p 0.00088 | **+19, nd 31, p 0.000878** |
| `th_s − scaled_t` competence | +15, nd 35, p 0.01667 | **+15, nd 35, p 0.016674** |
| `h_only − base` sensitivity | +2, nd 48, p 0.885 | **+2, nd 48, p 0.885433** |

**PASS.** Validated by collision with known numbers, not by reading.

## 2. The completed 3×2 factorial

| arm | δ_T | δ_H | sensitivity /112 | decision /112 | competence /112 | J_decision |
|---|---|---|---:|---:|---:|---:|
| `base` | 0 | 0 | 77 | 111 | 3 | +0.6786 |
| `delta_t` | 1.0 | 0 | 67 | 112 | 57 | +0.5982 |
| `scaled_t` | s_t | 0 | 74 | 112 | 51 | +0.6607 |
| `h_only` | 0 | 1.0 | 79 | **89** | 54 | +0.5000 |
| **`scaled_h`** ← new | 0 | s_t | 91 | **102** | 53 | +0.7232 |
| `th_s` | s_t | s_t | 93 | 112 | 66 | +0.8304 |
| **`indep`** ← new | 1.0 | s_t | **95** | **112** | 64 | **+0.8482** |

Four arms are re-used exactly as COMPOSE4B-002 scored them — same driver, same rule, same 224 row
ids, and the render gate confirmed `prompts_sha256 25d3b50f…` byte-identical, so every contrast
below is genuinely paired.

## 3. Registered outcomes — Holm–Bonferroni, m = 3, fixed before the data

| | contrast | b/c | nd | net | exact p | Holm adj | MDE | ceiling | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **P1** | `indep − delta_t` sensitivity | 32/4 | 36 | **+28** | 1.94e-06 | 5.82e-06 | 14 | 45 | **FIRES** |
| **P3** | `th_s − scaled_h` competence | 19/6 | 25 | **+13** | 0.01463 | 0.02927 | 11 | 59 | **FIRES** |
| P2 | `indep − delta_t` competence | 13/6 | 19 | +7 | 0.1671 | 0.1671 | 11 | 55 | **null, as registered** |

### P1 fires, and larger than its anchor

Adding δ_H at its own coefficient, with **δ_T at full strength**, buys **+28 boundary rows**
(32 gains against 4 losses). Registered band was **+8 … min(30, 45)** with a point prediction of
+16 from the 596M's INDEP-001. Observed **+28 — inside the band, and 12 rows above the anchor.**

Worth stating plainly: `delta_t` abstains on **67/112, which is 10 rows BELOW the untouched base
(77).** The task delta *costs* abstention. `indep` reaches **95** — so δ_H does not merely repair
δ_T's damage, it clears the base by 18.

### P3 fires — so the clause that would have retracted me is not triggered

`th_s` and `scaled_h` hold δ_H at the identical per-module `s_t`; the only difference is δ_T's
presence. **δ_T contributes +13 competence rows (19 gains / 6 losses, p = 0.0146).**

PREREG_003 §8 registered: *"if P3 does not fire, the competence half of RESULT_002 §5 is
confounded with attenuation exactly as Finding 2 was, and I will amend RESULT_002 to say so."*
**P3 fired. RESULT_002 §5 stands**, and it stands because a control that could have killed it was
run, not because it was never tested.

### P2 is a null, and it is reported with its bound

+7 in the *favourable* direction, nd 19, p = 0.167 — below the MDE of 11. Registered as a
**powered null or nothing**, so it is the null. With its equivalence bound:

> **At 95 % confidence, adding δ_H at its own coefficient costs at most ~1.1 competence rows**
> (Clopper–Pearson upper bound 0.530 on 6 losses of 19 discordant pairs).

That is a far tighter statement than COMPOSE4B-002 could make, where even a perfect tie bounded
the loss only at ~7.9 rows. **`indep − delta_t` on the decision endpoint is a perfect tie: 0
gains, 0 losses, nd 0 — both arms attempt all 112 answerable rows.** The gate costs nothing at
all on whether the model answers.

## 4. THE MAIN FINDING — a 596M withdrawal replicates on one endpoint and fails on the other

`COMPOSITION_IS_A_REPAIR` **Amendment 2** withdrew its own Finding 2 with SCALED-H-001:
*"the repair is attenuation, not composition"* — δ_H attenuated, with **no δ_T**, recovered 18 of
19 answerable rows by itself, and `SCALED_H` held the highest `J_decision` of any arm on record.

Running that same control on the 4B for the first time:

| | 596M (SCALED-H-001) | **4B (here)** | |
|---|---|---|---|
| `th_s − scaled_h` **boundary** | nd 4, min p 0.12, unresolvable | **+2, nd 4, min p 0.125, unresolvable** | **replicates exactly** |
| attenuation recovers answering | **18 of 19** | **13 of 22** (`scaled_h − h_only` decision **+13, p 0.0072**) | replicates in direction |
| δ_T's further contribution to answering | none measurable | **`scaled_h` stalls at 102/112; δ_T takes it to 112/112** | **does NOT replicate** |
| δ_T's contribution to competence | not resolvable | **+13, p 0.0146 (P3)** | **does NOT replicate** |
| best `J_decision` | `SCALED_H`, with no δ_T in it | `indep` **0.8482** > `th_s` 0.8304 > `scaled_h` **0.7232** | **does NOT replicate** |

> **δ_T's contribution is ENDPOINT-SPECIFIC: nothing measurable on the gate, significant on
> answering.** The boundary half of Amendment 2 replicates to the discordance count. The
> answering half does not survive the change of base.

What survives cross-base is the *shape* of Finding 3 — **the asymmetry is an endpoint property,
not a skill property** — and what does not is Finding 2's stronger reading that δ_T adds nothing
at all. **A withdrawal can itself be base-specific.**

## 5. The over-dose account, converted from interpretation to measurement

`scaled_h − base` on sensitivity is **+14 (21 gains / 7 losses, nd 28, p = 0.0125).**

COMPOSE4B-002's registered control P3 was `h_only − base` on sensitivity with a band of ≥ +20,
and it came in at **+2, p 0.885 — a registered miss.** The only difference between that arm and
this one is **dose**: α = 1.0 versus the per-module `s_t`.

**I am not using this to rescue the failed registered control.** That clause stands as written,
exactly as it did in RESULT_002 — *"I do not get to swap a passing measure in for a failed
registered one."* What it does is convert RESULT_002 §4's *interpretation* ("δ_H is over-dosed,
not weak") into a **direct measurement on the registered control's own contrast form**: the same
delta, the same comparator, the same endpoint, a different dose, +2 → +14.

This is [[delta-magnitude-saturates-early]] on a third instrument, and the answering side is the
sharper version: `h_only` attempts **89/112**, `scaled_h` **102/112**, `th_s`/`indep` **112/112**.

## 6. What must NOT be read out of this

- **`indep` being the best arm on record here is a DESCRIPTIVE RANKING, not a significant
  difference.** `indep − th_s` is **nd 2** on sensitivity (min attainable p 0.5) and **nd 8,
  net −2, p 0.727** on competence. The two composites are nearly identical; at nd 2 the
  instrument cannot certify any difference. **Nothing is promoted.**
- **Not a release.** One instrument, one base, one draw, one δ_T/δ_H pair, one coefficient vector.
- **Not a damage result on the endpoints INDEP-001 named.** `references_retained` /47 and the
  150-row arithmetic panel remain structurally unrunnable against a 4B adapter; long context
  remains SPENT / REPORT-ONLY.
- **P2's +7 is not a gain.** It is a null in the favourable direction, reported with its bound.
- **D4's +14 does not make COMPOSE4B-002's P3 pass.** It failed as written and stays failed.
- **Not a claim about δ_H at 0.25**, which was not built.

## 7. Provenance

Instrument `HSKILL_VIEWS_V2.jsonl` sha256 `ba1f2982…`, rule `boundary_rule.py` sha256
`416cfb6a…` — **imported, never reimplemented**. `HSKILL-CONFIRM-002` untouched.
Render gate 224/224 tokenizer agreement, mutation control fires, `prompts_sha256 25d3b50f…`
byte-identical to COMPOSE4B-002's. `enable_thinking=False`, `max_new_tokens=48`, greedy.

Arms built by `build_compose_003.py`, which **imports** `build_compose.py`'s `modules()` and
`eff()`. `s_t` **read back** from `PER_TENSOR_SCALES.json` (sha256 `ebabde31…`, 252 modules),
never recomputed. **Build gate 6 was a positive control and ran BEFORE any correct arm was
written**: two deliberate mutations — `s` applied to the T half instead of the H half, and `s`
omitted from `scaled_h` — were detected at **4.87e-01** and **3.95e-01** against TOL 2e-6; the
real arms then landed at **3.41e-08**, ~59× inside. The tolerance is SCALED-T-001's own amended
value, adopted unchanged.

Statistics: exact paired McNemar, MDE at observed discordance, Clopper–Pearson equivalence
bounds, Holm–Bonferroni at m = 3 fixed regardless of outcome. The registered unsatisfiability
guard (nd ≤ 6 cannot clear Holm's strictest step of 0.0167) did not bind — the smallest
registered nd was 19.

---

# Amendment 1 — this result could never have promoted, and that is a fact about the GATE

**2026-09-22, added the same night, prompted by a peer's certifiability census
(`retention-certifiability-20260922/`, their commit `dcf6821`) and not by anything in my own
data.**

§6 says "nothing is promoted" and gives instrument-level reasons. There is a stronger and more
structural reason that I did not know when I wrote it, and it belongs here rather than in a
footnote.

The peer joined every arm carrying **both** an acquisition score on bank `7dd53f2a` **and** an
n=47 `references_retained` figure — 54 arms — against GATEFIX-001's bands
(**≤30 FAIL, 31–42 UNCERTIFIABLE, ≥43 CERTIFIED_PASS**):

| arm | acquisition | retained | band |
|---|---:|---:|---|
| `tt001_TT16s` | 247 | 38 | UNCERTIFIABLE |
| `indep005_same_a025` | 245 | 38 | UNCERTIFIABLE |
| `mag001_sm05` | 243 | 39 | UNCERTIFIABLE |
| `indep001_t100h000` | 235 | 39 | UNCERTIFIABLE |
| …49 more | 235–247 | 31–42 | UNCERTIFIABLE |
| `int001_a025` | **145** | **43** | **CERTIFIED_PASS** |

> **Exactly one of 54 arms certifies, and it is the lowest acquirer in the set.** Every
> competitive arm sits at 38–41 — a 1-to-4-row margin — against a measured **MDE of 6**.

**What that means for this experiment.** `indep` and `th_s` are arms in exactly that competitive
class. Their retention figures are not measured here — `references_retained` is **structurally
unrunnable** against a 4B adapter, as §6 already states — but there is no acquisition level at
which an arm of this kind escapes [31, 42]. **P1 firing at +28 and P3 firing at +13 were never
going to produce a promotable arm**, whatever the numbers had been.

**The diagnosis is not "the gate is too small to certify anything."** `int001_a025` proves n=47
*can* pass. The competitive arms are clustered inside the gate's dead zone. So the binding
constraint on every promotion in this programme is **the size of the retention reference set**,
roughly 4× the references (n ≈ 190 for a 3-row margin, subject to a real power calculation) —
and that is a measurement-instrument problem, not a composition problem.

**Why this is recorded as an amendment rather than folded into §6.** §6's claims were true and
remain true; this adds a constraint I did not know about, discovered by someone else, that makes
the "no promotion" verdict *structural* rather than merely cautious. It also sets a rule I am
adopting going forward:

> **Before spending a battery on an arm chosen to clear the retention gate, check where its
> expected retention lands relative to [31, 42]. If it lands inside, the run cannot adjudicate
> whatever it measures, and the GPU time buys a number that no band can read.**

Nothing in §§1–7 is withdrawn. P1 and P3 still fired, the canary still passed, and P3 still
discharged the clause that could have retracted RESULT_002 §5. What changes is what firing was
ever worth: **a real effect inside an unresolvable gate is still a real effect, and still not a
promotion.**

---

# Amendment 2 — Amendment 1's "and it is the lowest acquirer" is WITHDRAWN. The conclusion survives; the gloss does not.

**2026-09-22. The peer who supplied the census withdrew that half of it (their Amendment 1,
`32b2087`). I verified the falsifier myself before amending, rather than accepting the
retraction on their word.**

## What I published and what is wrong with it

Amendment 1 said: *"Exactly one of 54 arms certifies, and it is the lowest acquirer in the set."*
**The second clause is false.** Verified here from the artifacts:

`compose001_H16`, `evidence/PROCESS_compose001_H16.json`:

| field | value |
|---|---|
| `views_sha256` | `7dd53f2a751b54c3…` — the composition bank, the same bank |
| `aggregate.canonical_correct` | **15** of 384 |
| retention (`RETENTION_PROBE_compose001_H16.json`) | `source_correct_total` 47, `lost_total` 8 → **39 retained** |
| band | **UNCERTIFIABLE** (39 is inside 31–42) |

**An arm acquiring 15 — a sixteenth of the competitive band, and 130 rows below
`int001_a025`'s 145 — retains 39 and does not certify.** Four arms acquire less than
`int001_a025` and none of them certifies. The correlation on the composition bank is **+0.216**,
not negative; the anti-correlation is real only on the **dose** bank at −0.567. The original
reading came from a table sorted by acquisition and truncated to its top rows.

## What survives, unchanged

**53 of 54 arms sit at 32–42, and the competitive band at 38–41 is a 1-to-4-row margin against
a measured MDE of 6 — unresolvable by construction.** So Amendment 1's actual conclusion stands
exactly as written: **COMPOSE4B-003 could never have promoted, regardless of P1 and P3 firing.**
Nothing in §§1–7 changes. What is withdrawn is a decoration on that conclusion, not the
conclusion.

## The better account, and it sharpens my own rule

Retention does not track acquisition. **Retention tracks MAGNITUDE** — already on record as
`damage-tracks-magnitude-not-acquisition`. Composition arms are transforms of **one** delta at
near-constant magnitude, so their retention is pinned at 38–40 across a **16-fold** acquisition
range (15 → 247). Dose arms move magnitude, so their retention moves with it, which is why the
anti-correlation shows up on the dose bank and not this one.

That makes the constraint mechanical rather than statistical, and it replaces the rule I adopted
in Amendment 1 with a stronger one:

> **Before spending a battery, do not merely check where the arm's expected retention lands in
> [31, 42] — check whether the arm CHANGES MAGNITUDE AT ALL. If it is another re-carving of a
> fixed delta at matched norm, its retention is already determined and the gate cannot adjudicate
> it. Only changing magnitude, or expanding the reference set, can move it.**

`indep` and `th_s` are exactly such re-carvings. **Their retention was determined before they
were built.**

## Why this is recorded rather than quietly edited

Amendment 1 is already committed and pushed (`1c25d375`, records `e2be2c0`). A false clause that
is silently deleted leaves a reader who quoted it with no way to find the correction — and I
quoted it to a peer before checking it. **I took a number from a peer, published it inside an
hour, and it was wrong in a way that a single artifact would have shown.** Verifying a peer's
number before publishing it is the same discipline as the canary, and I skipped it here.

---

# Amendment 3 — the `+0.216` quoted in Amendment 2 is a citation error. Nothing else changes.

**2026-09-22. The peer withdrew that coefficient (`e73776a`) after a third session measured
`n = 43, r = −0.099` against their `n = 54, r = +0.216` on the same bank.**

Amendment 2 says *"the correlation on the composition bank is +0.216, not negative."* **Do not
quote that number.** Their join walked two trees and resolved retention duplicates by
`rglob` traversal order via `setdefault` — directory order, not a merge rule. The defensible
statement is:

> **`r ≈ 0` on the composition bank, measured twice at two scopes (`+0.216` at n = 54,
> `−0.099` at n = 43). The sign is not determined and nothing here rests on it.**

The dose bank's **−0.567** is agreed between all three sessions and stands.

**Everything Amendment 2 actually concluded is unaffected**, because none of it used a
correlation:

- The falsifier is **one arm's two artifacts**, which I verified myself rather than taking on
  trust: `compose001_H16` at `views_sha256 7dd53f2a…`, `canonical_correct 15`,
  `source_correct_total 47 − lost_total 8 = 39 retained`, **UNCERTIFIABLE**. An arm acquiring 15
  retains 39 and does not certify. The "lowest acquirer" clause stays withdrawn.
- **53 of 54 arms at 32–42, competitive band 38–41, MDE 6** — a count, not a correlation. So
  **COMPOSE4B-003 was never promotable** and that conclusion has now survived two rounds of
  correction to the evidence that prompted it.
- The magnitude account stands, and it was never statistical: retention is pinned at 38–40
  across a 16-fold acquisition range because these arms are re-carvings of one delta at matched
  norm.

**The rule worth keeping is theirs:** *a count without a named scope is the same failure as an
absence check without a named namespace — and a correlation without a named scope is that
failure one level up, because **the merge rule is part of the scope**.*

**And the rule worth keeping about me:** this is the second number I republished from a peer
before checking it. I verified the *falsifier* independently and did not verify the *coefficient*
that travelled alongside it. **Citing a number is publishing it**, and a figure I would not have
computed myself is one I should not pass on without saying whose it is and at what scope.

---

# Amendment 4 — the retention endpoint is WORKING, verified independently. The dead zone is a property of the arms.

**2026-09-22. A peer verified an observation of mine and it turned out to remove the main
alternative explanation for Amendments 1–3. Verified here at 5.4× their artifact sample, with a
positive control, before publishing — the rule Amendment 3 was written to enforce.**

## One retention instrument, not two

I had noticed that `RETENTION_PROBE_*` artifacts carry no bank field. Swept across the records
tree and all 27 ChatBubble-family checkouts:

| | |
|---|---:|
| `RETENTION_PROBE_*` artifacts scanned | **714** |
| carrying a bank marker (`7dd53f2a` or `45674251`) | **0** |
| distinct `probe_sha256` | **1** — `5f6df1c14aa082eb`, all 714 |
| distinct reference-set sizes | **1** — `n = 47`, all 714 |

**One probe hash, one reference set, every arm, every family, both trees.** The "bank" names the
**acquisition** instrument only.

**Why that matters more than it looks.** Amendments 1–3 talk about "the composition bank" and
"the dose bank" as though two families were being compared. If those labels had also named two
*retention* instruments, the entire contrast would be an **instrument artifact** rather than a
fact about arms. It is not. Correct framing: **one retention instrument, two arm families,
partitioned by which acquisition bank they happen to have been scored on.** So the difference
has to live in the arms — which is what lets the magnitude account do predictive work instead of
merely fitting.

## The endpoint can FAIL and can PASS — the objection I should have anticipated is dead

The obvious challenge to Amendment 1 is *"maybe the retention endpoint is simply insensitive."*
Across **116 distinct arms** on that single n=47 set:

| GATEFIX verdict | arms |
|---|---:|
| `CERTIFIED_FAIL` (≤30) | **11** |
| `UNCERTIFIABLE` (31–42) | 94 |
| `CERTIFIED_PASS` (≥43) | **11** |

| extreme | retained | arm |
|---|---:|---|
| lowest | **6** | `D40_init`, `D40_init_v2`, `ced_last8` |
| highest | **47 / 47** | `panel001_lora_r16`, `latent_whitened_r1024`, `int001_lora596` |

**The same 47 references span 6 → 47 and produce all three verdicts.** By this programme's own
rule — *a gate must be shown capable of failing AND passing* — **this endpoint is working.**
It is not insensitive; the competitive arm family simply gives it nothing to resolve.

*(Positive control: my extractor reproduces `compose001_H16 = 47 − 8 = 39`, the value I had
already derived by hand from its two artifacts. A first attempt returned an empty distribution
because the fields are nested deeper than one level — the key-depth version of the key-name
trap — and that run's numbers were discarded rather than reported.)*

## This sharpens the constraint rather than softening it

11 of 116 arms certify on retention. **Certifying is not rare. What is rare is an arm that
certifies on retention AND carries a competitive acquisition score.** Of the three perfect 47/47
arms, `panel001_lora_r16` and `latent_whitened_r1024` have **no PROCESS artifact at all** — no
acquisition score on any bank, as I confirmed for the reconciliation. So:

> **The dead zone is not the gate being too small, and not the endpoint being blunt. It is that
> the arms which acquire well are all re-carvings of one delta at matched magnitude, and their
> retention is therefore pinned at 38–41 — while the arms that retain perfectly have never been
> measured on acquisition.**

That is a stronger argument for expanding the reference set than "the gate is too small" ever
was, and it leaves Amendment 2's rule intact and better founded: **check whether a candidate arm
changes magnitude at all; if it is another re-carving at matched norm, its retention is already
determined.**

Nothing in §§1–7 or Amendments 1–3 is withdrawn. Amendment 1's conclusion — **COMPOSE4B-003 was
never promotable** — now rests on an endpoint demonstrated to discriminate, rather than on a
narrow band whose narrowness might have been the instrument's fault.
