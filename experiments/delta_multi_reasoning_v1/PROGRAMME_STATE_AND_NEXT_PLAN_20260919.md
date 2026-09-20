# Rineya programme — state of evidence and next plan

**Compiled 2026-09-19.** Branch `research/delta-multi-reasoning-factorial-v1`. Written to be read
cold, months later, by someone deciding what to run next.

**North star:** improve model quality by acquiring new skills without forgetting existing ones.

**How to read a claim here.** Every number carries a class:

| class | meaning |
|---|---|
| **CONFIRMATION** | pre-registered, scored on a sealed instrument, read budget spent |
| **OPEN DEV RESULT** | measured on real weights, not pre-registered, or read more than once |
| **MECHANISM SCOUT** | exploratory, unsealed bank, openly spendable, promotes nothing |
| **BLOCKED** | prepared and costed, not run |

Only two CONFIRMATIONS exist in the whole programme. Both instruments are now spent.

---

# 0. The decision that reframes everything

**The operator has answered the gating question: 596M is a CHOICE, not a constraint.** The
preference is a small model that runs fast — 596M or smaller. **Qwen3-4B is wanted for insight,
not as a deliverable.**

Three consequences, and they invalidate a framing both working sessions held this morning:

1. **The 4B is an ORACLE, not a candidate.** Its 270/384 is not a competitor to the 596M's best.
   It is a *measurement of how much signal 384 rows contain*.
2. **"The 4B lane dominates" is retired as a direction.** It was never progress on the stated
   problem; it was a change of problem.
3. **Every remaining question is about METHOD, not about the corpus or the base model.**

---

# 1. What is actually established

## 1.1 The two confirmed results

| | instrument | result |
|---|---|---|
| **RCI-001** (spent) | 512 sealed rows | On 128 unanswerable rows the k=0.25 delta loses abstention against frozen Supa: **40 → 25, 0 gains / 15 losses, McNemar p = 6.1e-05** |
| **RCI-002** (spent) | 640 sealed rows | **wh-compliance**: where the interrogative types a person and the context determines none, the delta emits a person more often — 120 incongruent rows, B 85 / D 99, **14 discordant / 0 reversals, p = 1.22e-04**, against a congruent control where each system names a person **1 in 120** |

**wh-compliance is the programme's only confirmed mechanism.** Five competing accounts died
before it. It is the account of the abstention failure.

## 1.2 The acquisition frontier — one delta, exhaustively re-sliced

Every Titan arm is a post-hoc transform of **one delta**: `delta_T_seed17`, 384 rows, 54,172
processed tokens, **28 optimizer updates**. Alphas, masks, ranks, rotations, depth profiles — all
re-slices of one gradient.

`select-views-001`, `views_sha256 7dd53f2a…`, implementation digest `a329eb7bb06f`:

| arm | acquisition /384 | floors | long ctx /256 | literal UNKNOWN /64 |
|---|---:|:--:|---:|---:|
| Supa 1.7 (frozen source) | 14 | 0/3 | 161 | 13 |
| α = 0.50 | 219 | 3/3 | **164** | 12 |
| α = 0.75 | 234 | 3/3 | 163 | 10 |
| **Titan-Sparse k = 0.25** | 232 | 3/3 | 164 | 12 |
| α = 1.00 (full delta, shipped) | **235** | 3/3 | 157 | 9 |
| N16 full-rank, norm-matched | **236** | 3/3 | 162 | 10 |
| Qwen3-4B 4-bit zero-shot | 0 | 0/3 | 167 | 18 |
| **Qwen3-4B 4-bit + LoRA r=16** | **270** | 3/3 | **180** | 9 |

**Established:**
- **Scale alone does not buy the contract** — Qwen3-4B zero-shot is 0/384 and 0/3 floors in *both*
  precisions. A targeted 596M update buys what 3.4B extra parameters do not.
- **Mask / rank / scalar are ONE lever for acquisition.** N16 (full rank, rescaled to r=16's norm
  profile) scores 236 and matches or beats r=16 on every endpoint. The frontier is flat at
  232–236 across every magnitude reparameterisation.
- **An α below 1 dominates the full delta on both lanes** — but the knee is **LANE-DEPENDENT**
  (native 0.75, LWF 0.50) and is **never a rule**. It must be located per lane.
- **The 4B headline is harder than it looks.** 270 is measured **at 4-bit NF4** — the deployable
  precision — and r=16 was **pre-registered** with a band of **100–250** that it **missed upward**.
  That is the structural opposite of a best-arm envelope.

## 1.3 Dose — the axis that was explored and then forgotten

**`PROCESS_DOSE_FRONTIER.md` (RRP-008), four rungs, one seed each, native execution:**

| rung | rows | acquisition | panel | retained | gate |
|---|---:|---:|---:|---:|:--:|
| **D1** | 384 | 239 | 77 | **40** | **PASS** |
| D2 | 768 | 276 | 68 | 35 | FAIL |
| D3 | 1,536 | 289 | 71 | 37 | pass, exactly |
| D4 | 3,072 | **304** | 59 | 30 | FAIL |

> **Acquisition rises monotonically and does not saturate: +37, +13, +15, never reversing.
> The 596M is DATA-limited on acquisition, not capacity-limited.**

**This matters more than any transform result.** The entire INT-001 / SPARSE-001 / RANK-001 /
LWF programme was built on **D1 — the lowest dose rung** — and then spent months masking that one
delta's damage back out.

**Verified here, because the artifact's own limitation line invites the doubt.** The record flags
the bank as *"TRAIN-only derivative views"*, which raises the obvious question of whether the
higher rungs were scored on parents they trained on. **They were not:**

| | |
|---|---|
| D4 training parents (`dose-D4-prepared/microfit-train.jsonl`) | 3,072 |
| dose eval-bank parents (`PROCESS_rrp008_D4.json`) | 384 |
| **intersection** | **0** |

Same structure as the SELECT-VIEWS / microfit relationship. **The dose curve is not
training-fit.** It is parent-disjoint generalisation within one generator split.

**Two caveats that must travel with these numbers:**

1. **Different bank.** `views_sha256 45674251…`, *not* the `7dd53f2a…` used by every arm in §1.2.
   Same implementation digest, so no runner confound — but **304 and 270 are NOT comparable, and
   neither are 239 and 234.** See §4.2.
2. **"Retained" here is a 150-row panel covering 3 of v3's 39 reply classes.** It is not
   `references_retained`/47, not alias, not long context. Weak retention instrument; do not quote
   it as general capability.

## 1.4 Preservation — what the delta breaks

| finding | evidence | class |
|---|---|---|
| **Binding regression is general, not about persons** | PRES-005: person 0/31, object 0/17, place 0/10 damaged — **0 repairs, 58 damages, 3/3 strata** | MECHANISM SCOUT |
| **Answer position is CAUSAL** | delta pays ~5× the source when position is manipulated; masking repairs 44% at k=0.25 | OPEN DEV RESULT |
| **Interpolation repairs binding, purely additively** | PRES-007 person, denominator 33: α=0.416 repairs **20/33**, α=0.624 repairs **12/33**, **zero losses at either** | OPEN DEV RESULT (registered primary) |
| **Rehearsal is HARMFUL** | RRP-015 matched arm: +648 rehearsal rows at double the token budget lost **27 long-context rows**, p = 2.53e-05. Tested four ways (RRP-010/011/012/015), never protective | OPEN DEV RESULT |
| **No training corpus contains the damaged skill** | `CORPUS_BINDING_AUDIT.json`: control **0/384** role phrases, person names and binding sentences; rehearsal 1,032 rows also 0. Detector validated on a positive control (RCI-001 fires **512/512**) | OPEN DEV RESULT |

**The corpus is 384 rows, three JSON state-transition families, three distinct system prompts,
100% JSON user turns, 81 English word types.** Independently re-measured; agrees with the
committed audit exactly.

## 1.5 What has been positively closed

These are **dead ends**. Do not re-open without new evidence — each cost real work to close.

| direction | closed by |
|---|---|
| Further α / k / rank screens on the seed-17 delta | N16 norm-matched control: 236, matching r=16 everywhere. Reparameterisations of one lever |
| α = 0.625 and α = 0.875 | The native span 0.75→1.00 is **one acquisition row** (234→235). Nothing to resolve |
| A per-item selector/router over the α/k arms | Nesting fails **21/21 pairs** — scaling *reorders* items. 5 of 6 feature-partition oracles sit at or below a size-matched random partition |
| Per-layer **flattening** / moving dose to late blocks | T16flat: **−51 acquisition** at p = 2.24e-10, and dominated by α=0.50 on every endpoint |
| Rotation / direction search as a quality lever | A plain scalar at half the norm dominates λ=0.25 on acquisition, alias AND long context |
| Masking as a distinct mechanism at k ≥ 0.50 | s50 vs α=0.75: no violation of the scalar ordering at all |
| Rehearsal / replay as a protective default | Four independent tests, never protective |
| **Salience** as the account of binding damage | The frozen source makes the same error with an identical profile, **Fisher p = 1.000 on both banks** |
| Majority / plurality / unanimity vote over surface forms | Majority 12/40 and 14/40 against mean-single 0.325 and 0.342. When forms disagree the model is usually wrong in all three |
| Aggregation over **role**-diverse candidates | Per-stratum oracle headroom is **0 in all four strata** |
| Extra token budget on the 596M family | 0 repairs / 0 damage over 128 items |
| Scaling the acquisition corpus **to buy quality outright** | The dose ladder — it buys acquisition and costs retention. See §1.3 |
| Long-context rehearsal from data on this host | Full census: no row is long enough |
| A further training draw of the **same** recipe to move acquisition | REP-001: acquisition spans 7 rows across three draws |

---

# 2. The plan

Ranked by **information per unit cost**. Items 1–3 need no GPU and no training.

## Tier 0 — costs nothing, could change a published conclusion

### P1. Re-derive the two "already answered" checks before anything else *(no model calls)*
Two corrections landed today by re-reading records rather than memory (§4). Before committing to
any experiment, grep the *records* for it. **The programme's most common failure is an experiment
that buys a restatement** — and its second most common is a plan built on a withdrawn number.

### P2. STRUCT-002 — does any arm beat an answer-agnostic ceiling? *(no model calls, ~3–4 h)*
`structural_floor_probe.py`'s register predictor defines `correct_kind_no_modulus` for `add`
only. The corrected floor is **100/128** — *above every arm ever measured on `register_machine`*
(best 78/128). **Write a v2 as a NEW file; never overwrite — the records dir has no local undo.**

> If the corrected floor stands, part of the absolute acquisition claim does not. Paired arm
> contrasts are untouched, so the damage is bounded and knowable. **Cheapest thing here that
> could move a headline.**

### P3. FORMSEL-001 — is "multi-reasoning fails twice" about ensembling, or only about voting? *(no model calls, ~2 h)*
The form-diverse generations already exist. Oracle is **23/40 (B) and 22/40 (D)** against
majority's 12 and 14. `form_diverse_mr.py` computes `best_single` per pack **and discards it**.
Nothing has ever been tried between *vote* and *oracle*.
**Commit the selection rule before scoring** — the arms exist, so the window closes at the first
number. Register a random-selection baseline.

## Tier 1 — no training, existing weights

### P4. XCAP-001 — capacity's only clean kill *(no training; adapter already on disk)*
Run the **existing** Qwen3-4B LoRA adapter on PRES-007 and ALIAS-001. Same 384 training rows,
6.7× the parameters, a 0.8% intervention instead of 52.8%.

- **Capacity** predicts essentially no binding damage at 4B.
- **Corpus-induced prior shift** predicts the same purely-subtractive zero-repair signature.

Registered falsifier: **capacity is refuted if pooled damage ≥ 6 rows with ≤ 1 repair.**
The 142 s of training is already paid; this is generation only. **It decides the programme's
direction rather than moving one number.**

### P5. DIST-001 — one mechanism for four regressions *(Stage 0 = no model calls)*
Hypothesis: every preservation regression is the delta raising the prior on emitting a **verbatim
span copied from the context**. It would unify binding, alias, wh-emission and abstention.
Stage 0 classifies ~5,000 **already-stored** predictions as CONTEXT_SPAN or not — zero model
calls. It predicts one buildable corpus edit: rows whose target is *not* copyable from the input.

### P6. PRES-REP-001 — is the damage signature a property of the RECIPE or of seed 17? *(no training)*
**Every preservation result in this programme describes one delta.** Seeds 23 and 29 exist. All
resource guards pass as measured. If either new draw fails to show the signature, the preservation
programme is describing a seed.

## Tier 2 — training, and the one that matters most

### P7. LORA596 — parameterization at fixed dose *(LIVE — running in the paired session)*
The sharpest structural observation of the day:

> **The 596M received a 66× more invasive intervention than the 4B, on the same 384 rows.**
> FP32 `blocks8-27` = 314,618,880 params = **52.8% of the model**, against LoRA r=16's **0.8%**.

Over half of a 596M model rewritten by 384 examples — and the entire transform programme was
post-hoc damage control on that. RANK-001 does **not** cover this: truncating afterwards preserves
but does not repair, whereas **under a rank constraint the optimizer never writes the damaging
directions, so there is nothing to repair.**

**Known confound, with a control:** the control writes a **2.56×** larger update
(‖δ‖ 0.8298 vs 0.3242). Damage tracks magnitude, so a retention win is confounded. LoRA's norm
corresponds to **α ≈ 0.391**; the interpolated bar there is **acquisition ≈ 187, retained ≈ 42.4**.
**LoRA demonstrates that RANK matters only if it beats that bar.**
The decisive cell is **acquisition**, not retention — retention is saturated (6-row MDE, ceiling 47).

**Arm 1 result (2026-09-19), and it is UNDER-CONVERGED:**

| | acquisition | floors | arith | retained | long ctx | UNK |
|---|---:|:--:|---:|---:|---:|---:|
| `lora596` arm 1 | **70** | 1/3 | 68 | **47/47** | 155 | 13 |

Registered P1 (acquisition 150–235) **FAILED**; the matched-norm bar required ≈187 and it
delivered 70. Cause identified: `lr 1e-4` was matched to the control's *number* rather than to its
*intervention strength*, and arm 1's loss ran 0.53–0.93 where arm 2 runs 0.13–0.30 at the same
steps. **Arm 1 is not a test of rank; it is a test of an under-trained adapter.**

The one thing that may survive the confound: **zero source answers lost — 47/47, unique in the
programme.** It is not the trivial small-update result, because **α=0.25's delta norm is 0.2075,
*smaller* than LoRA's 0.3242, and it loses 4.** A 1.56× larger delta forgot less. Hold this as
*suggestive only* — under-convergence explains it equally well, and arm 2 is what separates them.

**Arm 2 result (2026-09-19), scored independently by a scorer committed before any arm-2
endpoint was read.** All three comparability gates pass, including `weight_sha256 eaa40115…`
matching the merge receipt, so the probe read arm 2's merged weights.

| arm | acq /384 | retained /47 | floors | long ctx |
|---|---:|---:|:--:|---:|
| a050 (α = 0.50) | 219 | **42** | 3/3 | 164 |
| a075 (α = 0.75) | 234 | 39 | 3/3 | 163 |
| a100 (control, α = 1.00) | 235 | 40 | 3/3 | 157 |
| **LoRA r=16 lr1e-4** (α-equiv **1.794**) | **223** | **41** | 3/3 | 154 |

Registered bands: **P3' (acq ≥ 235) FAILED** at 223; **P4 (source lost ≤ 4) FAILED** at 6.

**Two things this settles.**

1. **Arm 1's 47/47 was convergence, not rank.** The disconfirmer was: under-convergence predicts
   47/47 *together with* low acquisition; a rank effect predicts 47/47 *surviving as acquisition
   rises*. Acquisition rose 70 → 223 and retention fell 47 → 41.
2. **The cell that would have contradicted damage-tracks-magnitude did not fire.** Retained 41 at
   1.79× the control's magnitude against the control's 40 — one row against a 6-row MDE. Both
   sessions had recorded that prior in writing beforehand.

**POST-HOC, and not part of the registration — the comparison that actually decides it.** Not
against the control, but against the scalar ladder at **matched acquisition**:

> **LoRA minus a050: acquisition +4 (floor 7), retained −1 (floor 6), long context −10 (floor 13).
> Every difference is inside its noise floor.** At this dose LoRA r=16 is indistinguishable from
> α = 0.50 — and α = 0.50 is a plain scalar needing **no training run at all**.

**What this closes and what it does not.** It closes *"a rank-constrained update forgets less at
matched behaviour"* — at matched acquisition it does not. It does **not** close rank at other
ranks or learning rates (one rank, two lrs, and the two arms bracket the control rather than
matching it). And critically it does **not** touch P8: this tests rank at 384 rows, where the
dose ladder says the model is **data-limited**. If anything it sharpens P8 — **at D1 both
parameterizations land in the same place, so D1 cannot discriminate them.**

**Registration lesson for the next arm.** P3' was the right *bar* — the matched-norm comparator
was unexecutable at α-equivalent 1.794, and extrapolating past the ladder's last measured point
is the cue-differential error again. But P3' is a **threshold**: it says the arm failed to beat
the ladder's maximum without saying *where on the ladder it landed*. The adjacent ladder row is
the informative object and it was computable in advance from numbers that already existed.
**Register the matched-acquisition comparison, not only the threshold.**

### P8. Dose × parameterization — the cell nobody has visited
The two axes have been explored **separately and never together**:
- dose at 52.8%-of-model FP32 → acquisition 304, retention collapses
- parameterization at D1 → LORA596, in flight

**A low-invasiveness update at D2/D4 dose is the untested cell, and it is where the north star
lives.** If rank constraint is what prevents damage, then dose can be raised without paying the
retention cost that killed D2 and D4. Cost is small — 77.5 s per 596M LoRA arm, 142 s per 4B arm.

> **P8 CARRIES THE §4.2 TRAP AND MUST BE SEQUENCED AROUND IT.** The dose arms live on bank
> `45674251…`; every LoRA / INT / SPARSE arm lives on `7dd53f2a…`. **A LoRA-at-D2 arm scored on
> `7dd53f2a…` would have no matched comparator**, and comparing it to FP32-D2's 276 would repeat
> exactly the error caught in §4.2 — one step further down the chain, where it is harder to see.

Two ways out, both probe-only. **Only one is available:**

| route | status |
|---|---|
| Re-score an existing RRP-008 dose checkpoint on `7dd53f2a…` — *preferable, it would put the whole dose ladder on the bank where all parameterization work lives* | **UNAVAILABLE.** `DISK_RECLAIM_001.md`: *"each rung now deletes its own checkpoints and materialised model after it has been scored."* Verified — no checkpoint or `.safetensors` survives under any `dose-D*-journal*`. This route needs **retraining**, not re-scoring |
| Score the new LoRA-at-dose arm on bank `45674251…` | **Available**, and the prepared dose corpora (`dose-D4-prepared/microfit-train.jsonl`, 3,072 rows) survive even though the checkpoints do not |

**So: build the cross-bank bridge BEFORE training a LoRA-at-dose arm, not after.** Whichever
route is chosen, it is a build gate — and the retrain route is the reason the dose ladder should
have kept one checkpoint.

### P9. HELDOUT-TRANS-001 — the first genuinely held-out bank *(build = no model calls)*
Every acquisition number in this programme comes from **one split of one generator**.
`metadata/test_iid.jsonl.gz` (200/200/129 parents) and `test_transfer.jsonl.gz` (`transfer = True`
on all 640 rows) have **never been extracted**. Report per family, never pooled.

---

# 3. Operator decisions

| # | decision | recommendation |
|---|---|---|
| 1 | **Host RAM.** Free physical measured **1.86–2.01 GiB** against `MIN_FREE_GIB = 3.0`; `lease.acquire()` refuses. Consumers are desktop apps, not model jobs | Close desktop apps before any generation run. **Never weaken the guard** |
| 2 | **VRAM** — 4,620–4,725 MiB of 8,188 held by Chrome/Edge/ChatGPT/Claude | Same |
| 3 | **A new sealed instrument (RCI-003).** Both existing ones are spent, so *no new CONFIRMATION is possible without one*. Entity-partition space is the scarcity | Build it for the **binding** claim — it is the best-replicated unconfirmed result (1/54 pooled across three banks) |
| 4 | **RTG-1 training grant** — 24,000 rows, P0 PASS, needs 23.03 GiB free disk and a NEW budget grant (10.4× the existing cap) | **Hold.** `rows_reviewed_by_a_person: 0`, and two defects invisible to all 19 automated checks were found today. Read rows first |
| 5 | **Generator repair** — 0 of 2,376 `prerequisite_plan` rows have failing preconditions or overlapping add/delete bits. One third of every training draw and every acquisition score is a bitmask OR that two different wrong rules also satisfy | Repair before the next dose experiment, or a third of the signal is uninformative |

---

# 4. Corrections made today — and what they generalise to

## 4.1 A withdrawn number came back as a premise
I quoted a damage ordering of **0.304 / 0.168 / 0.090** to pressure the capacity account. The
object figure was PRES-005's; realised PRES-007 rates are **0.324 / 0.034 / 0.082**. The ordering
was assembled **across two banks with different object difficulty** — the exact confound that got
the ratio withdrawn, **on the same day the withdrawal was recorded**.

> **A withdrawal has to propagate to everything downstream of it, not just to the sentence that
> asserted it. Grep for the NUMBER, not for the claim.**

## 4.2 The near-miss that the bank hash caught
I was one sentence from writing *"596M at 8× data scores 304, beating the 4B's 270."* Both banks
are 384 views, 3 families × 128, same generator family, **same implementation digest**. Everything
looked comparable except `views_sha256` — `45674251…` against `7dd53f2a…`.

> **Check `views_sha256` before comparing any two acquisition numbers. It is the bank's identity
> and nothing else is.**

## 4.3 A definitional identity reported as a finding
PRES-007 Amendment 2 reported *"the lower α repairs more — monotone down the ray."* Damage is
defined as source-right-and-λ0-wrong, so repair at α=0 is **33/33 by construction**; any monotone
path must show repair rising. Struck. Its replacement — **concavity** — then failed its own power
test: excesses of +3.5 and +3.8 rows against a smallest-detectable **+6.5 and +5.8**.

Also: linear-in-α is a **choice, not a derived null**. Testing against an underived reference is
**a tolerance wearing a null's name.**

## 4.4 A read-order violation, disclosed
Computing PRES-007's damage denominators is arm-independent and was legitimate. Putting the
`a050`/`a075` repair counts in the **same script** read the registered primary before the
amendment existed.

**Cost, stated precisely:** the object exclusion and place's power re-derivation **can no longer
be called pre-registered**. **Not cost:** `person_binding` was the registered primary, declared
adequately powered *before the arms existed*; it stands as registered.

> State which **direction** of an inference broke before retracting anything that used it.

## 4.5 Silence from a check that could not reach the claim
A `find` over local disk concluded a backup was absent. **The vault is a remote.** The search
could never have returned anything else. Same shape as cue differentials computed on cues whose
band swing was exactly 0.000.

> **Before treating a check's silence as evidence, ask what would have to be true for it to return
> something, and confirm the claim lies inside that.**

---

# 5. Standing rules

1. **Never weaken a resource guard.** Escalate the shortfall with its number.
2. **Never overwrite prior evidence.** The records dir now has a git remote and a **load-bearing
   `* -text` `.gitattributes`** — a clone with default `autocrlf` would rewrite line endings and
   invalidate every recorded sha256.
3. **Any check that could force a withdrawal must be a BUILD gate.** After generation, an honest
   withdrawal and moving the goalposts are externally indistinguishable.
4. **Score damage over the rows the BASELINE got right.**
5. **Report a null as "not powered" unless the MDE says otherwise.** This shape has now appeared
   **three times** (`references_retained`, PRES-007 uniformity, PRES-007 concavity).
6. **Break every tie explicitly and report the tied-row count.** An unresolved tie is a silent coin
   flip wearing a rule's name.
7. **Read the rendered text, not just the structure.** Two bank defects survived five structural
   checks; one silently confounded a published contrast.
8. **Never quote a pooled long-context figure** — decompose into answerable-above-gate, below-gate,
   and unanswerable.
9. **Never quote a number produced by a rule you have just impeached.**

---

# 6. What this review did NOT cover

Stated so the next reader knows the edges.

- **The adversarial verification stage was cut short by a usage limit.** 6 audits and 20
  candidates completed; **49 of 71 agents died** and the synthesis never ran. Of 20 candidates
  only 12 verdicts landed. **The ranking in §2 is mine, not a verified consensus.**
- **No candidate in §2 has been costed against a measured GPU-hour budget** beyond the figures
  quoted from existing receipts.
- **The `deep-any` parser remains uncommitted**, and three stated zero-shot capability numbers
  (35, 81) depend on it.
- **Qwen3-8B (16 GB) and Qwen3-0.6B are both on disk and have never been touched.** The 0.6B is
  the more interesting of the two: it is a **same-scale non-Rineya control**, and every *"the 596M
  family cannot do X"* claim is currently confounded with *"this particular 596M family cannot."*
- **`enable_thinking=False` on all 640 Qwen prompts.** Qwen3-4B's actual reasoning mode has never
  been measured, and 122 of 384 outputs contain unevaluated arithmetic — the signature of a model
  reaching for a scratchpad it was denied.
- **Two-model complementarity is unexplored.** On acquisition the 4B and Titan-Sparse are **not
  nested**: 199 both, 71 only-4B, 33 only-Titan-Sparse, 81 neither. Best-of-two envelope **303/384**.

---

# 7. One structural fix that would have prevented three of today's five errors

Three separate corrections in §4 share one shape: **a number was used at a point where its own
correction was not attached.**

- the withdrawn person-vs-object ratio (§4.1) — withdrawn in `BINDING_CONFIRMATION_PROPOSAL`,
  quoted as a premise hours later
- `retained >= 43` justified by *"no floor-clearing arm has ever passed 41"* — corrected in an
  amendment, then repeated verbatim by the other session **after** reading that amendment
- the dose ladder itself — recorded in `PROCESS_DOSE_FRONTIER.md` and absent from both sessions'
  working picture until a systematic audit surfaced it

None of these was carelessness, and treating them as such produces no fix. In every case the
number came from a trusted source, the correction existed, and **the correction lived somewhere
the number did not.**

> **Banner the withdrawal where the CLAIM lives, not only where the withdrawal was decided.**
> A retraction filed in one document does not propagate to the tables, memories and plans that
> already quote the number.

`BINDING_CONFIRMATION_PROPOSAL_20260919.md` identified this and it has never been implemented.
**Implementation is two steps, and the naive version fails.** *"Grep for the number"* is the right
shape, but `41` matches inside `0.4141`, `141`, timestamps and several thousand JSON floats — the
signal-to-noise is bad enough that the practice gets abandoned after one attempt. Instead:

1. **FIND with a word-boundary, prose-first grep**: `grep -rnw '41' --include=*.md`. The claim
   lives in prose. Data files hold measurements that are **still true** and must NOT be annotated
   — a withdrawn interpretation does not retract the number it was computed from.
2. **BANNER the withdrawal in the document where the sentence lives.** The grep finds them once;
   the banner is what the next reader actually hits.

The searchable handle is the number; the durable artifact is the banner. Confusing the two is why
this stayed unimplemented after `BINDING_CONFIRMATION_PROPOSAL` first recommended it.

This is the same lesson as §4.5 — **a check has to reach the thing it is checking** — applied to
corrections instead of to evidence.
