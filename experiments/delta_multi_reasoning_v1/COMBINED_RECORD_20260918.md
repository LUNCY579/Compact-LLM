# Combined record — the delta / multi-reasoning programme, both agents, as of 2026-09-18

**Read-only audit of `Agent report (1).pdf` (11 pages) against the committed artifacts, plus
the state of both sessions' work. Written after the operator's laptop restart.**

The PDF is a combined transcript: **pages 1–6 are the peer session's narration**
(fine-tuning audit / RANK-001 / QWEN-FT-001 / COMPOSE-001), **pages 7–11 are this session's**
(DMRF-001). Numbers below are marked VERIFIED when recomputed from a committed artifact here,
CONTRADICTED when the artifact disagrees, and UNREPRODUCIBLE when no artifact implements the
thing the number came from.

---

## 0. First: a correction to my own earlier audit

My Stage 0 audit (`DMRF001_AUDIT.md` §2 and §3, commit `c46b6ba`) stated two things that this
report shows to be wrong:

> "**There is no 35.** Searching every record for an acquisition figure of 35 returns nothing."
> "**'deep-any' — the named mechanism does not exist.**"

The report's page 1 carries this table:

| model | canonical | 1-unwrap | **deep-any (format-free)** | parsed |
|---|---:|---:|---:|---:|
| Supa 1.7 | 14 | 17 | **35** | 353 |
| Titan-Experimental | 234 | 234 | **234** | 384 |
| Qwen3-4B | 0 | 66 | **81** | 262 |

So `deep-any` is a real parser the peer computed, Supa's deep-any score **is 35**, and
**"35 → 234" is a like-for-like comparison within one parser column** — not the condition
splice I called it. The driving prompt's claim was substantially right and my audit was wrong.

What I actually established was narrower than what I wrote: I searched the records directory
and the harness code, where only `strict`, `canonical` and `contract_tolerant` (= the report's
`1-unwrap`) exist. I then reported the absence of a *file* as the absence of a *measurement*.
That is the same error class I spent the audit criticising — treating the artifact I could
reach as the whole evidence base.

**The narrower finding does survive, and it matters more than the correction:**

- **`deep-any` has no implementation in any committed file.** Re-checked today across
  `rineya-dataset-retention-records/` and `ChatBubble-retention-pilot-20260917/`: zero hits for
  `deep_any`, `deep-any`, `format_free`, `format-free`. It exists only in the peer's narration.
- Therefore **the programme's headline premise — "a 0.056% delta took the task from a
  format-neutral 35/384 to 234/384" — cannot be reproduced or audited from the repository.**
- And the driving prompt's actual question about it is still open: is `deep-any` label-blind
  extraction of one answer, or a search for any gold-matching value anywhere in the output? A
  parser that moves Qwen3-4B from **0 → 81** while it clears **0 of 3** family floors is
  behaving permissively enough that the question is live. **I cannot answer it, because there
  is nothing to read.** That is the single highest-value thing for the peer to commit.

One part of my §2 finding does stand: the 234 belongs to **Titan** (the full delta, BF16
export), not to the promoted **k=0.25** core, which is 232 canonical. The peer's own framing is
consistently about Titan, so they did not splice — but anyone quoting "35 → 234" alongside a
discussion of the promoted core would be.

---

## 1. NEW AND DECISIVE — C1's "oracle" is not an oracle

This is the most consequential finding in this audit, and it is load-bearing for the peer's
strategic conclusions.

The report states (page 1, repeated pages 2, 5, 6):

> "A **perfect per-input router** over all seven measured arms buys, over static k=0.25:
> acquisition +3, arithmetic +0, retained +2, long context +0, abstention +1."
> … "k=0.25 already sits essentially at the oracle corner."
> … "Scaling, pruning, or routing a task delta **cannot raise general capability above the base
> model**."

**Test 1 — arithmetic.** Every one of those five figures equals *the best single arm's score on
that endpoint minus k=0.25's score*, exactly:

| endpoint | k=0.25 | best single arm | difference | report's "oracle headroom" |
|---|---:|---:|---:|---:|
| acquisition | 232 | 235 | **+3** | +3 |
| arithmetic | 72 | 72 | **+0** | +0 |
| retained | 41 | 43 | **+2** | +2 |
| long context | 164 | 164 | **+0** | +0 |
| abstention | 12 | 13 | **+1** | +1 |

Five for five. So the quantity computed is the **best-single-arm envelope over aggregate
scores**, not a per-input router. A per-input oracle asks, for each item, *did any arm get this
one right* — a strictly larger set.

**Test 2 — measured.** The arms' own long-context artifacts carry `per_row_correct`, so the
true per-item oracle is computable today. From
`int001-20260917/evidence/LONGCTX_*.json`:

| quantity | value |
|---|---:|
| best single arm (α=0.50) | 164 |
| **true per-item oracle over the arms** | **171** |
| **headroom vs best single arm** | **+7** (report says +0) |
| **headroom vs the frozen base's 161** | **+10** |

Robust to arm selection: the four α arms alone reach 171, and adding the three k arms adds
nothing to the union.

**What this changes — CORRECTED 2026-09-18, after the peer reproduced both tests and caught an
error in my framing.**

The peer reproduced Test 1 (five for five) and Test 2 (171, +7) exactly, then ran the test I had
not: **the acquisition oracle. Union 266 against the best single arm's 235 = +31.** I have since
verified that independently from the `predictions` arrays in the PROCESS artifacts — +31 exactly,
with greedy coverage matching digit for digit:

| arms | acquisition | added |
|---|---:|---|
| 1 | 235 | α=1.00 |
| **2** | **255 (+20)** | α=0.50 |
| 3 | 263 | α=0.25 |
| 5 | **266** | saturates |

**My framing was wrong and the correction matters.** I reported "+10 over the base's 161" as
available long-context headroom. **Both +7 and +10 sit inside REP-001's 13-row band**, so neither
is established — and `analysis/paired.py::noise_verdict` exists precisely to prevent that
sentence. I applied it to the peer's margins and not to my own finding. (This is the second
instance of the same shape in this audit: §0 reported the absence of a *file* as the absence of a
*measurement*. Both are standards applied outward but not inward.)

**The corrected split, which is sharper than what I first wrote:**

- **Acquisition — C1 is FALSIFIED and the frontier is not exhausted.** +31 against a claimed +3,
  a factor of ten, and 4× the noise floor. Two arms give +20, nearly 3× the floor, so a two-way
  selector — a much smaller object than a seven-way router — captures most of it.
- **General capability — C1's conclusion SURVIVES, on noise grounds rather than on its
  arithmetic.** The long-context oracle is +7 inside a 13-row band. The statement that holds is
  *"no static modulation has been shown to raise general capability above the base"* — and it
  holds because the evidence is underpowered, not because the headroom was measured to be zero.

C1's registered falsifier was "an oracle headroom calculation showing more than a few rows
available". It **fired on acquisition and did not fire on long context.** The peer has withdrawn
C1 as stated and C2 in both legs (C2 stood only on C1 after the relative-norm correction), with
the receipt and script committed rather than left in prose.

**A process finding that generalises.** No oracle computation is persisted in any FRONTIER file —
none contains the words oracle, router, headroom or union. Together with `deep-any` and the
routing composite, that is **three load-bearing numbers that existed only in prose.** Not three
coincidences: the programme's failure mode is publishing a conclusion without committing the
computation that produced it.

**Allocation does move the endpoint, from my own suggestion failing.** I proposed per-layer
relative-magnitude flattening as "the one modulation with a mechanistic reason to work". The
peer's `T16flat` scores **184 against T16's 235 — −51 rows at fixed total dose** (g₈ = 0.40 down,
g₂₇ = 1.71 up), 7× the noise floor. My directional reasoning was wrong, and the experiment is
more informative for it: allocation is now measured to move acquisition by 51 rows, which
reopens "apply the delta more cleverly" on allocation grounds independently of the +31 ceiling.
The untested inverse — *steepening* the natural profile — is the obvious next point, with the
honest caveat that the trained profile may be near-stationary and both directions may lose.

**And the composition penalty now partitions.** `TH16s` = 219 recovers +44 of the 60-row
composition penalty (73%), leaving a **16-row residual outside the 7-row floor** — so most of the
penalty is magnitude and a real minority is genuine skill conflict. That is a better answer than
either extreme.

**The per-layer direction then closed on a measurement, which is the outcome worth having.**
The peer parameterised my proposal as `g_b ∝ (target_rel/rel_b)^p` renormalised to
`‖P₁₆(δ_T)‖`, so p=0 is T16 and p=1 is T16flat — and building at p=1.0 reproduces
`arm-T16flat`'s data-section digest exactly, so the sweep provably passes through both
previously measured points. Verified independently here, every figure identical:

| p | allocation | rel. spread | acq /384 | repair | damage | net | paired p | Holm |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| −1.0 | steepen hard | 26.35× | 239 | +15 | −11 | +4 | 0.5572 | no |
| **−0.5** | steepen | 11.63× | **243** | +13 | **−5** | +8 | 0.09625 | no |
| 0.0 | **trained** | 5.13× | 235 | — | — | — | — | — |
| +1.0 | flatten | 1.00× | **184** | +9 | **−60** | **−51** | 2.24e-10 | **yes** |

Holm over the three registered contrasts: flatten rejects at 0.0167; neither steepen point
rejects. **My registered band (225–245) held; the peer's (228–242) failed by one row at 243, and
they recorded it failed rather than rescuing it with the better test.**

**The finding is an asymmetric, non-monotone optimum**: downside catastrophic and unambiguous,
upside undemonstrated, and steepening *harder* is worse than steepening a little, so the curve
turns over. 28 AdamW steps landed on an allocation that cannot be significantly improved by
depth rescaling and degrades sharply if mass moves late. That is **"already exploited", not
"nothing to exploit"** — and C2's conclusion was right for the wrong reason *twice*: wrong
statistic (absolute vs relative norm), then wrong mechanism.

Two details the nets hide, and both matter: **the steepen arm is +13/−5, so it fails the strict
zero-damage criterion regardless of its net** — a fail, not a null; and **flatten is +9/−60**, so
flattening repaired nine items and the −51 is net two-way movement, not uniform destruction.

**A note on the instrument, because the switch was consequential.** The peer moved from
REP-001's draw floor to paired McNemar and declared the switch before using it. That is the
correct decomposition: McNemar asks *is this difference real on these items*, the draw floor asks
*would it survive a different training draw*. These are algebraic variants of **one** δ_T on
identical items, so there is no new draw and the draw floor answers a question the design does
not pose. The one-draw caveat does not vanish — **it relocates** to a generalisation limit: every
allocation finding is conditional on this single δ_T. The switch was also conservative for the
result it affected (turning +8 into a non-result) while strengthening the flatten finding, which
is what makes it a method choice rather than a rescue.

**The convergent per-item finding, with the caveat neither of us stated first.** Two independent
sweeps now say the static quantity is flat where the per-item quantity is large: magnitude arms
**+31** over the best static arm, allocation arms **+15**. But **oracle headroom grows
mechanically with arm count**, so +31 over seven arms and +15 over four are not directly
comparable and some share of both is arm-count rather than structure. What bounds it is the
greedy curve: 235 → 255 → 263 → 265 → 266, **saturating at three arms**. Concentrated headroom is
the real thing; diffuse headroom would have been the artefact. **Everything here remains
label-aware — no feature-based input-only selector has been tried**, and the cautionary data
point is this session's own MR result, where selector utilisation was 0.960 and there was
nothing to select between.

**The arm-count objection is answered, and the right null changes the reading twice.** The peer
ran an item-permutation control (2,000 trials, marginals preserved); it reproduces here
analytically and by simulation. Greedy saturation settles the objection directly: **arms six and
seven add exactly zero items.**

| reference | union | headroom | what it assumes |
|---|---:|---:|---|
| **nesting** (monotone ladder) | 235 | **+0** | more delta ⇒ strictly more capability |
| **observed** | **266** | **+31** | — |
| independence | 382.8 ± 1.1 | +148 | arms uncorrelated at these marginals |

Independence is unreachable by construction for scalar variants of one δ, so "+31 of +148" reads
thin against a ceiling nothing of this kind can hit. **Nesting is the null anyone actually held,
and it is refuted comprehensively: 0 of 21 pairs are nested.** k=0.10 (197) wins **20** items
that α=1.00 (235) loses; α=0.25 (**145**) wins **14**; the two arms that both score 235 differ by
5 items *each way*. **Scaling one delta does not move items monotonically — it reorders them.**
Both references belong in the ledger: independence bounds how redundant the arms are, nesting
bounds how surprising the headroom is.

**Where the headroom sits, and the limit of that finding.** The 31 items are not uniform across
families: ledger_replay 19, prerequisite_plan 8, register_machine 4 — **χ² = 11.68, 2 df,
p = 0.0029**, a 4.7× spread, concentrated on the family every screen calls binding. That refutes
"pure borderline-ness". But **`family` is hidden suite metadata that `router.py` forbids a
selector from using**, so this shows the headroom is *structured*, not that it is
*input-visible*. Family plausibly correlates with request surface, but "a forbidden label
predicts it" and "a permitted feature predicts it" are different claims and only the first is
measured. The mechanism originally hypothesised — that these are items the base already handles
and the delta damages — is **untestable here: no Supa per-item PROCESS artifact exists on this
host.**

**Consequence for any selector experiment.** If the headroom concentrates in one family, the
realistic ceiling is nearer 19 items than 31, less again after surface-feature loss — thin
against a 7-row floor. The experiment must be **powered for that** or its null will be read as
"no signal" when it was "not enough items". Note also the circularity: fitting and evaluating on
the same 384 is invalid, and SELECT-VIEWS-001 is the only selection-admissible acquisition
instrument, so an in-domain split *spends* it. `scout/build_scout.py` can mint a disjoint
transition bank at zero instrument cost as the fitting set, leaving the views as a clean
held-out test — a cross-generator transfer test, harder than an in-domain split and honest about
it. Whether to run it is the operator's call.



---

## 2. Verified from the report, against committed artifacts

| report claim | verdict |
|---|---|
| `‖dW‖/‖W‖ = 0.000558` | **CONTRADICTED, diagnosed.** Tied-weight double count: `tie_word_embeddings` exposes one embedding tensor under two names, so a named-parameter walk adds 364.2551² to the denominator. 1440.8616 → 1486.1911 gives exactly 0.000558361. True: **0.000576** whole-model, **0.000622** trained scope. The peer independently confirmed this. |
| "largest single-tensor relative change 0.00209" | **VERIFIED** — I measure 0.002073. |
| "per-block delta norms near-uniform (0.169–0.199)" | **CONTRADICTED twice.** The true absolute range is **0.1795–0.2027** (the peer confirmed their quoted numbers were wrong), and the *relative* range is **0.000278–0.001641**, a **5.89×** decline with three local reversals. "Nothing to exploit" rests on the wrong normalisation. |
| "16 directions carry 59–88% of the energy" | **VERIFIED on the sampled tensors, understated on the population.** All five named tensors' top-1 and top-16 energies reproduce to three decimals. Over all 140 2D trained matrices the band is **50.8–92.1%** (median 67.4%), and **23 of 140** fall below the quoted 59% floor. |
| "effective rank 16–72 of 1024" | **ESTIMATOR DIFFERS**, not an error. Participation ratio gives 6.1–35.8 (median 16.4). Their energies are the load-bearing quantity and those reproduce. |
| "the same delta costs 8 of 47 source arithmetic answers, 3 of 13 abstentions" | **VERIFIED** — Titan retains 39/47 and scores 10/64 literal UNKNOWN. |
| "damaged long context in 2 of 3 draws (144/152/157 vs 161)" | **VERIFIED** — consistent with REP-001's 13-row spread. |
| "Routing Supa+delta for the contract and Qwen3-4B for general gets 232 + 178 + 26, beating every single model on both axes" | **NOT AN EXECUTED RESULT.** This composites one model's acquisition with another's long-context and abstention. No dispatch was run. The peer does flag that the audit is checking whether any routed system ever beat its best component, but the sentence as written presents an unmeasured composite as an achievement. |
| "Qwen3-4B + LoRA: 270/384 acquisition, 180/256 long context" | **VERIFIED** from `QWENFT001_RESULTS.md`, and the training data is legitimately disjoint from the eval bank — see §3. |
| "the long-context gain is real retrieval, not terseness" | **VERIFIED** — generated-token statistics identical (min/p50/max 1/3/24, 255/256 terminated). Good control. |
| "the 4-bit control earned its place: NF4 alone costs 11 long-context rows and 8 abstentions" | **VERIFIED**, and it is the best-designed control in the corpus. |

---

## 3. The "384 rows" question, settled

The report repeatedly says the 4B was trained on "the same 384 rows". I expected contamination
and measured **none**:

| overlap level | result |
|---|---|
| item (`train.id` ∩ `views.source_id`) | **0** |
| parent (`parent_id`) | **0** |
| identical prompt string | **0** |
| identical user-message content | **0** |

There are **two different 384-row sets**: the TRAIN file
(`rrp015-20260917/control-prepared/microfit-train.jsonl`) and the SELECTION bank
(`select-views-001/SELECT_VIEWS.jsonl`, whose `BUILD.json` declares
`inadmissible_uses: ["TRAIN","CONFIRMATION"]`, `training_admitted: false`). Nobody trained on
the selection bank. **270/384 is same-generator generalisation, not training fit.**

What is still unmeasured: a genuinely *different* held-out structure. Both banks come from one
generator's train split, so nothing speaks to transfer beyond forge_v2.

---

## 4. This session's own results

**N16 — the norm-matched control (measured, `fee32b4`).** Full-rank delta scaled per tensor to
r=16's exact norm profile (0.812249925436939, matching to 15 digits). Built twice,
byte-identical, `502260c9…`.

| endpoint | r=16 | **N16** | draw spread |
|---|---:|---:|---:|
| acquisition /384 | 235 | **236** | 7 |
| arith perm / strict | 68 / 60 | **69 / 62** | — / 13 |
| retained /47 | 39 | **39** | — |
| long context /256 | 159 | **162** | 13 |
| literal UNKNOWN | 10 | **10** | — |

Every difference inside the noise floor. **Direction selection contributes nothing**; rank-16
survives as a *storage* result (7.26M params, 27.7 MB, 43×) and not as a mechanism claim. 8 of
8 registered predictions held. My own arm's premise also came out negative: N16 ≈ the uniform
α=0.75 within 1–2 rows everywhere, so the per-tensor profile carries nothing either.

**DMRF-SCOUT-001 (built and gated, `b259d16`).** 152 items, 24 CAL + 128 CHECK worlds, four
strata, every answer from a programmatic oracle. Verdict **PARTIALLY_USABLE**: `transition`,
`arith_competence`, `longctx_mixed` clear their answer-agnostic floors; nothing below floor;
`multi_step` inconclusive-underpowered at n=6. The gate caught four defects in my own bank
before any contrast was reported, three of them prompt-side.

**MR on system B — frozen Supa, CHECK split, n=128 (measured today).**

| condition | total | transition | multi_step | arith | longctx |
|---|---:|---:|---:|---:|---:|
| O | 36 | 3 | 1 | 0 | 32 |
| T | 36 | 3 | 1 | 0 | 32 |
| REP | 36 | 2 | 2 | 0 | 32 |
| **MR** | **32** | **0** | **0** | 0 | 32 |
| MR (S0 majority) | 35 | 2 | 1 | 0 | 32 |

Paired: **O→MR is 0 repairs, 4 damage, net −4** (p = 0.125, within the 7-row noise band).
REP→MR also −4. Roles **did** diverge (only 20% emitted identical text), so this is not a
name-only failure — and yet **role oracle headroom is 0**: per-role correct R0=36, R1=27,
R2=35, R3=13, and `any_role_correct` = 36 = R0 alone. **No item was solved by R1/R2/R3 that R0
did not already solve.** Selector utilisation 0.889; 35 items needed the frozen tie-break.

Read plainly: on the frozen source, MR **repaired nothing in 128 items** while costing 4, and
the differentiated roles added no coverage for any selector to exploit. The net is within noise;
the zero-repair asymmetry is not a noise statement. Caveat: B's 36 is almost entirely the
long-context stratum, which sits at a **32/32 ceiling** — exactly the ceiling risk I flagged
when the gate passed. System D is running and is the informative arm.

---

## 5. Operational facts that cost real time, and will cost it again

Each of these produced a failure that *looked like a result*.

1. **`enable_thinking=False` is load-bearing for this family.** Without it both cores scored
   **0/24** while reasoning *correctly* inside a think block they opened and never closed
   inside the token allowance. A renderer omitting it produces what looks like total capability
   collapse.
2. **`stage2_generate.py` leaks across rows.** Bisected: 8/16/32/64 rows OK; **128, 256, 896
   OOM** at 7449 MiB *free* VRAM. The eight longest prompts pass alone and rows 64:128 pass as
   their own file, so it is neither prompt length nor a bad item. Fixed by chunking at 64 rows
   per fresh process, generator unmodified.
3. **safetensors `__metadata__` ordering makes whole-file sha256 a coin flip.** Six separate
   processes, identical tensors and metadata: data-section sha identical in all six, file sha
   flipping 5/6 vs 1/6. The peer retracted an Erratum closure over this; my own arm hash is one
   of two valid encodings. **Identity must be the data-section digest.**
4. **`torch.linalg.svd` is thread-count dependent** (and memory-pressure independent, contrary
   to the first diagnosis). Pin `torch.set_num_threads` and record it.
5. **VRAM, not host RAM, is the binding resource on FP32 jobs** — 8.19 GiB card against 15.22
   GiB host. Both sessions' guards watched host RAM and checked `memory.used` once. Both fixed.
6. **Reclaim discipline is per-driver, so a driver that dies never tidies up.** 11.1 GB of stale
   arm weights from finished batteries took the volume to 100%. A crash-surviving fix needs a
   manifest of live arms plus a sweeper.
7. **A shared GPU lease now exists** at `rineya-dataset-retention-records/gpu_lease.sh`, sourced
   by both sessions. Acquire the lease, *then* wait on real VRAM — they arbitrate different
   things.

---

## 6. What is blocked, and by what exactly

| blocker | exact next action |
|---|---|
| **`deep-any` unreproducible** — the headline premise rests on it | Commit the implementation. Then answer the open question: does it extract one answer label-blindly, or search for any gold-matching value? If the latter it is an oracle diagnostic and "35 → 234" must be restated. |
| **C1's oracle figure is wrong** | Recompute per-item from the `per_row_correct` artifacts that already exist, on all five endpoints. Long context is done above: +7, not +0. Then restate or withdraw "k=0.25 sits at the oracle corner" and the ceiling claim that depends on it. |
| **No confirmation instrument** | Everything long-context and abstention is report-only on a `SPENT` suite. Seven screens terminate in "observed, never confirmed". The generator method makes a fresh bank cheap; the binding requirement is an operator commitment never to spend it on selection. |
| **Everything is one training draw** | REP-001 measured spreads of 7 / 13 / 13. Two more draws would put every comparison on a measurable footing. Algebraic arms cannot substitute — they inherit the one delta. |
| **MR/routing on D** | Running now. B is complete and negative. |
| **4B has no retention on the gate's instrument** | Structural: `verify_hf_candidate` requires six sidecars byte-identical to Supa. Do not manufacture a substitute. |
| **Corrupt git object** | `Chat-Bubble/.git/objects/dd/9367ce44…`, dated Sep 11, unreachable but breaks `git gc` in every worktree. Operator's call; `git prune` would clear it if confirmed unreferenced. |

---

## 7. Status of the work

**This session:** branch `research/delta-multi-reasoning-factorial-v1`, 11 commits, tree clean,
**now pushed to origin** (it was unpushed and unbacked-up until today — the single largest risk
in the whole programme, and it survived a laptop restart only by luck).

**Classification, unchanged:** SOFTWARE PASS on four fixture suites; NATIVE EXECUTION PASS on
the N16 battery and system B's matrix; OPEN DEV RESULT on the N16 comparison and B's MR cells;
MECHANISM SCOUT on the magnitude-vs-direction analysis. **CONFIRMATION and RELEASE remain
closed.** `quality_improvement_established` false; `release_permitted` false; nothing promoted.

## 8. The one-paragraph summary

A 0.056%-relative weight update installs a large, genuinely new, narrow capability on a 596M
base, and reliably costs some generality. Every weight-space way of applying that update —
scalar scaling, magnitude masking, rank truncation — is **one lever acting through magnitude
alone**, established by building the norm-matched control nobody had. The subspace framing is
dead from two independent directions: within one delta, directions are inert at matched
magnitude; across two deltas, near-orthogonality does not prevent interference. A 4B base plus
142 seconds of training beats the entire 596M frontier on both axes at once, on legitimately
held-out parents. Multi-reasoning repaired nothing on the frozen source in 128 items. And the
two numbers the programme's strategic conclusions lean on hardest — the format-free 35 and the
oracle headroom of +0 — are respectively **unreproducible** and **measurably wrong**.

---

## 9. Addendum, same day — the programme's first confirmed result, and what it decomposes into

Written after sections 1-8. **`CONFIRMATION` is no longer closed.**

RCI-001 was built, calibrated against a disjoint twin, sealed, registered
(`confirm/RCI001_CONFIRM_PREREG.md`, committed before the seal was opened) and read once.

> **CONFIRMED: the k=0.25 delta damages literal-`UNKNOWN` abstention.** B 40/128, D 25/128,
> **0 gains and 15 losses**, exact McNemar p = 6.10e-05. Registered criterion met, registered
> prediction band hit from the calibration twin. Zero false abstention on either side.

Two things about it belong in the joint record because both cut against earlier sections here.

**The published aggregate floor was too low and the defect was mine** — all four published
predictors were distributional, and a recency rule reading the context scored 0.3711 against
the published 0.2773, because my generator emits every correction *after* what it corrects. The
peer found it; the seal was already spent. The primary endpoint is structurally immune (0/128)
so the confirmation stands, but it stands on an endpoint choice, not on the process. Full
erratum on `confirm/RCI001_CONFIRMATION_001.md`.

**Decomposing the read found a second and larger effect, and killed four hypotheses.** Receipt:
`confirm/decompose_rci.py` → `rci001-20260918/RCI_DECOMPOSITION.json`, write-up in Appendix A of
the confirmation record. Briefly: the confirmed abstention loss is **uniform** (about a third of
the source's correct abstentions, everywhere, Fisher 1.00 across groups), while a **separate and
larger answerable regression** hits tasks requiring a role to be resolved to its occupant — 32.0%
of the source's correct rows against 2.3% elsewhere, Fisher 3.3e-10. Four proposed mechanisms
died, two per session. The last two died on the same mistake: reading a count where the rate over
the baseline's correct rows was the admissible quantity.

**This does not license anything.** `quality_improvement_established` stays false,
`release_permitted` stays false, nothing is promoted. One confirmed claim, and it is a caution.
