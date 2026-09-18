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

**What this changes.** C1 is the measurement that "closed off" per-layer gating, that justified
*"improve the small model cannot mean apply the delta more cleverly"*, and that — after my
correction to C2's normalisation — is the **only remaining leg** C2 stands on. On its own data:

- the headroom figure is wrong by **+7 rows** on long context;
- a per-item selector over the *existing* arms reaches **171, ten rows above the base's 161**,
  so "delta modulation cannot raise general capability above its base" **is not supported by
  C1**;
- "k=0.25 sits at the oracle corner" is false — it sits at the *best-arm* corner.

**What it does not change.** An oracle is label-aware and diagnostic only; nothing here says a
*deployable* router reaches 171, and the +7 may be unreachable without gold. But the difference
between "there is no headroom" and "there are seven rows of headroom no static arm captures"
is the difference between closing a research direction and leaving it open.

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
