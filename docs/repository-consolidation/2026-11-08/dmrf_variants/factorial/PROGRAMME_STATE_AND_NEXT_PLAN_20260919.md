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

**CORRECTED — my first pass applied the wrong floor here.** I wrote *"every difference is inside
its noise floor"*, quoting long context −10 against a floor of 13. The paired structure says
otherwise, and both sessions computed it independently and agree on every cell:

| contrast | n | a050 | LoRA | gains | losses | exact McNemar |
|---|---:|---:|---:|---:|---:|---:|
| acquisition | 384 | 219 | 223 | 37 | 33 | **p = 0.7202** |
| long context | 256 | **164** | 154 | **3** | **13** | **p = 0.0213** |

> **A floor derived from BETWEEN-DRAW WHOLE-BANK spread does not govern a WITHIN-PAIR
> discordance.** A −10 carried by 16 discordant rows splitting 3/13 is not the same object as a
> −10 carried by 60 rows splitting 25/35. The 13-row floor was computed for the second case and
> I applied it to the first. Same class of error as the damage-denominator rule — a statistic
> borrowed across a change of structure.

**The p is nominal and is NOT load-bearing**: it fails a 3-endpoint Bonferroni (0.0167), and a
verdict that turns on family choice is the weakest part of any finding.

**The conclusion does not need the test, and is firmer without it:**

> **α = 0.50 is not beaten by LoRA r=16 on any endpoint, is nominally better on long context, and
> requires NO TRAINING RUN** — it is a scalar multiply of a delta that already exists.

Acquisition +4 (p = 0.72) and retained −1 are genuine nulls; long context runs the scalar's way.
At this dose the parameterization buys nothing and plausibly costs something. That argument rests
on effect sizes and on the comparator being free, not on a knife-edge p.

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

---

# 8. UPDATE 2026-09-20 — DOSEINT-001 lands, and it falsifies the hopeful reading of §1.3

A third session ran the dose × interpolation cell. **Its design is better than the one proposed
in P8 and supersedes it.**

**Design:** 2,697 rows **disjoint from BOTH acquisition banks** (not just one, as D4 was), one
training draw, four FP32 arms at α = 1.00 / 0.75 / 0.50 / 0.25, all scored on bank `45674251`
under digest `a329eb7bb06f`.

| arm | acq /384 | retained /47 | panel /150 | long ctx | **literal UNKNOWN /64** |
|---|---:|---:|---:|---:|---:|
| Supa (source) | 14 | 47 | — | 161 | **13** |
| **D1** (384 rows, gate PASS) | 239 | **40** | **77** | 160 | 8 |
| a100 | **292** | 30 | 53 | 149 | **0** |
| a075 | **293** | 31 | 58 | 151 | 2 |
| a050 | 279 | 37 | 59 | 150 | 3 |
| a025 | 186 | **40** | 61 | 159 | 12 |

**7 of 8 registered bands HELD. The PRIMARY FAILED** — no interpolant reaches acq ≥ 284 **and**
retained ≥ 40 **and** arith ≥ 62.

## What this kills

> **High dose plus interpolation does NOT break the acquisition/retention trade.** The α ladder
> at 2,697 rows reproduces the same monotone trade as at 384: acquisition 292 → 293 → 279 → 186
> against retention 30 → 31 → 37 → 40. **Interpolation moves along the curve; it does not escape
> it.**

That was the load-bearing hope behind P8's framing, and it is wrong.

## What survives

**a050 against D1: +40 acquisition** — cross-draw, clearing the ≥ 20 cross-draw floor (§8.1) —
**for −3 retention**, inside the 6-row MDE. A measurable gain against an unmeasurable cost, on
those two axes.

**But a050 fails the gate anyway, and not on retention:** the **arithmetic panel** is D1's 77
against every interpolant's 53–61. All four fail the gate's ≥ 62 arm; α = 0.25 misses by one.
**High dose costs the panel ~16–24 rows and interpolation does not buy it back.**

## The sharpest number in the programme: high dose ABOLISHES abstention

Literal `UNKNOWN`, re-derived here from raw predictions rather than the `says_unknown_on_unanswerable`
field (which memory flags as a substring match and therefore an upper bound — **for these arms
the literal count equals the field exactly**, so the caveat does not bite):

| dose | α = 0.25 | 0.50 | 0.75 | 1.00 |
|---|---:|---:|---:|---:|
| 384 rows (INT-001) | 13 | 12 | 10 | 9 |
| **2,697 rows (DOSEINT-001)** | 12 | **3** | **2** | **0** |

> **At α = 1.00 on high dose the model says `UNKNOWN` zero times out of 64.** The 384-row ladder
> only ever went 13 → 9. **High dose does not damage abstention; it abolishes it** — and there is
> a **dose × α interaction**: at 384 rows the full α sweep costs 4 abstentions, at 2,697 rows it
> costs all 12.

This is the confirmed wh-compliance mechanism taken to its limit: **nothing in 2,697 JSON-output
rows ever rewards declining to answer.**

**Caveat that must travel with it:** these rows sit inside the SPENT long-context suite, are
REPORT-ONLY, and **cannot select anything**. But 13 → 0 is not a marginal call, and it makes the
abstention-rows intervention the best-evidenced next step rather than a nice-to-have.

## 8.1 The acquisition band is not resolvable, and D4 cannot be bridged

**Cross-draw floor** — three independent training runs of the *identical* control recipe,
verified independently here:

    rrp015_control 234 vs rep_s23 229   b=36 c=31  p=0.6254   MDE 17
    rrp015_control 234 vs rep_s29 228   b=39 c=33  p=0.5560   MDE 18
    rep_s23 229    vs rep_s29 228       b=43 c=42  p=1.0000   MDE 21

> **Re-running the same recipe moves acquisition by up to 6 views. Any CROSS-DRAW difference
> below ~20 rows is not a measured difference.** The distinction that keeps this from
> over-correcting: **every α/k/r/λ arm is a deterministic transform of ONE training run**, so
> those contrasts are WITHIN-draw and valid at their own discordance.

**And the two banks are not an accident.** D4's 3,072 training parents overlap select-views on
**275 of 384** (`ledger_replay` 76, `register_machine` 71, **`prerequisite_plan` 128/128**)
against **0** on the dose bank. **A D4 score on select-views would be contaminated on 72% of the
bank.** BRIDGE-001 is demoted to a scout observation: the banks are equal difficulty (11 points,
I² = 0, pooled −2.70 rows) but **a bridge does not fix contamination** — an offset corrects for
bank difficulty, not for having trained on the eval parents.

## 8.2 P8 is superseded; the live lane is BLOCKED on an expired envelope

DOSELORA-001 (`research/retention-dose-parameterization-v1`) implements P8 properly, with two
things the P8 sketch lacked:

1. **Dual-bank-disjoint training rows**, removing the contamination gate and the transport step
   together.
2. **β = ‖δ_LoRA‖ / ‖δ_native‖ frozen before inference**, admissible interval [0, 2],
   `BLOCKED_ON_NORM_COMPARABILITY` outside it, no clamping and no search — the 2.56×
   invasiveness confound answered as a **gate** rather than a caveat, with one training run
   yielding both the LoRA merge and its exact norm-matched scalar.

**It is blocked, and the blocker is procedural rather than technical.** Host RAM has since freed
(7.75 GiB) and the GPU sits at 543 MiB, but its registered session deadline
**2026-09-20T09:39:05Z expired 4.0 hours ago**, and its own runbook says: *"preserve this
envelope and register a fresh bounded session prospectively before launching; do not silently
extend its deadline."*

> **OPERATOR DECISION 6: authorise a fresh bounded session for DOSELORA-001, or let the envelope
> stand.** Relaunching under the expired one is exactly the silent extension the runbook forbids.
> No session should take this decision on the operator's behalf.

## 8.3 One infrastructure fix worth more than any single experiment

`RETENTION_PROBE_*.json` records a contemporaneous `weight_sha256` via `hf_candidate_identity`.
**`PROCESS_*.json` records none.** That asymmetry is exactly why LORA596's merge-integrity gate
was checkable and why every BRIDGE-001 point is **identity-unbound** — `model_dir` plus a null
checkpoint is a directory path, not bytes, in a programme that documents an arm-directory
OVERWRITE trap.

**It is a one-field fix in the process probe**, and it would retroactively make every future
acquisition number the difference between a receipt and a path.

## 8.4 CORRECTION — on the dose axis, long context and abstention are ONE endpoint

§8 reports long context and literal `UNKNOWN` as two report-only endpoints. **On the DOSEINT
ladder that double-counts the same 13 rows.** Splitting the 256 rows by answerability —
`answerable = all256_hits − unanswerable_correct`, both fields in the artifact:

| arm | UNKNOWN (= unans /64) | **answerable /192** | total |
|---|---:|---:|---:|
| frozen source | 13 | **148** | 161 |
| 2,697 α = 0.25 | 12 | **147** | 159 |
| 2,697 α = 0.50 | 3 | **147** | 150 |
| 2,697 α = 0.75 | 2 | **149** | 151 |
| 2,697 α = 1.00 | **0** | **149** | 149 |

> **Answerable comprehension is FLAT at 147–149 across the whole high-dose ladder, and α = 1.00
> is the best of the four — above the frozen source's 148. The entire 161 → 149 drop IS the
> collapse of the ability to decline.** Exactly: 161 = 13 + 148, 149 = 0 + 149, net −12 = −13
> abstentions +1 answerable.

On an unanswerable row the only way to score is to emit `UNKNOWN`, so *"unanswerable correct"*
and the literal-`UNKNOWN` count are the same column — which is why the tight-upper-bound check in
§8 and this decomposition agree **by construction** rather than by corroboration.

**What it emits instead is not a format artifact.** On the 64 unanswerable rows the frozen source
gives 13× `UNKNOWN`, 8× `5`, 2× `Mosaic`; α = 1.00 gives 8× `5`, 3× `Wednesday`, 3× a full
sentence, 2× `Tuesday`. **Natural-language fillers, a different weekday each time.** It is not
learning a specific wrong answer, it is learning to *always supply one* — wh-compliance at its
limit, and the dose × α interaction in §8 is that mechanism's dose-response curve.

### The general version is FALSE, and the check that shows it

Across all **62** arms carrying both fields: answerable spans **113–154 (span 41)**, unanswerable
spans **0–14 (span 14)**. Both components move in general. **The clean decomposition is a property
of the DOSEINT ladder — whose answerable span is 2 — not of the instrument.**

The concrete demonstration: **two arms both score exactly 161 and split it differently.**

    supa         161 = 13 unanswerable + 148 answerable
    ownA075_lc   161 =  8 unanswerable + 153 answerable

*"Never quote a pooled long-context net"* is already a standing rule; this is what it looks like
when a pooled total hides its own structure at identical face value.

### Revised injury summary for high dose

> **+acquisition, −arithmetic panel, −abstention, comprehension NEUTRAL.**

The arithmetic-panel reframe in §8 stands untouched and is now the **only separate injury**.

### Practice: carry a canary through every extractor

Three extractor bugs in one day across two sessions — a truncated key list that concluded a bank
had no `answerable` field; a parent parsed from an `id` string instead of the declared
`parent_id`; and a literal-`UNKNOWN` count that compared record dicts to a string and returned
**0 for every arm**. **All three ran cleanly. None was caught by reading the code.** Each was
caught only because its output collided with a number already held — *"1 distinct parent for a
384-row bank"*, *"0 for an arm on record at 13"*, *"0/0"*.

> **Where no such number happens to be held, the bug ships silently.** The only defence that does
> not depend on luck: **compute one quantity whose answer you already know, in the same pass,
> every time.** If the extractor returns 13 for the frozen source it is parsing; if it returns 0
> it is not.

Applied here: the canary is `literal UNKNOWN == unanswerable_correct`, which passes on all four
DOSEINT arms. **It cannot run on the frozen source** — that artifact omits
`unanswerable_correct` entirely, so its 13 rests on the literal count alone with nothing to check
it against. Worth knowing before that 13 is used as an anchor.

## 8.5 CORRECTION to §8.4 — the anchor IS guarded, and two of our checks were never independent

### The anchor is guarded. My "unguarded" claim was drawn at the wrong level.

§8.4 says the frozen source's 13 *"rests on the literal count alone with nothing to check it
against"*, because `LONGCTX_supa.json` omits `unanswerable_correct`. **The summary field is
absent; the evidence is not.** The artifact carries both `per_row_correct` and `predictions`,
which give two genuinely different routes, verified here against the bank's 64 unanswerable ids:

| route | reads | result |
|---|---|---:|
| **A** | the scorer's `per_row_correct` — its **grading decision** | **13** |
| **B** | raw prediction string `== "UNKNOWN"` — the **generated text** | **13** |
| C | same, case-insensitive | 13 |

**A and B are independent**: they diverge exactly when the scorer accepts a variant the literal
match misses (`Unknown`, `unknown`, a trailing period, *"I don't know"*). They agree at 13, so
the anchor the whole −12 decomposition is measured from is checked, by two routes that could
have disagreed.

> **A missing summary field says nothing about whether the evidence is missing. Look below the
> aggregate before declaring something unguarded.**

This is the **fourth** time today a conclusion was drawn at a level that could not see the
thing it was about — a `find` scoped to disk when the vault was remote; `--evidence` roots that
excluded the worktree holding the only informative bridge point; a truncated key list that
concluded a bank had no `answerable` field; and now an absent aggregate read as absent evidence.
**Three of the four were mine.** The common shape is not carelessness about *facts*, it is
failing to ask **what level the question lives at** before answering it.

### Two of our checks were the same measurement, and this needs saying plainly

A later reader will see §8 report literal `UNKNOWN` and §8.4 report `unanswerable_correct`, from
two sessions, two scripts, two write-ups — and count it as confirmed twice.

> **It is one measurement.** On an unanswerable row the only scoring answer is `UNKNOWN`, so
> `unanswerable_correct` and the literal-`UNKNOWN` count are the **same column by construction**.
> They *cannot* disagree. This is the one place in today's cross-checking where agreement carries
> no information — everything else the two sessions duplicated was genuinely independent.

The A-versus-B routes above **are** independent, for precisely the contrast that makes them
worth running: grading against raw text. That is what an anchor check should look like, and it
is why our agreement on 13 is worth less than it appears while the A/B agreement is worth
something.

### Lead with this, not with the dose table

The sharpest statement of *"never quote a pooled long-context net"* is not the ladder. It is
that **two arms score exactly 161 and are injured differently**:

    supa        161 = 13 unanswerable + 148 answerable
    ownA075_lc  161 =  8 unanswerable + 153 answerable

Identical face value. Five rows of abstention traded for five rows of comprehension, invisible
in the total. A rule stated is a rule ignored; a demonstration at identical face value is not.

## 8.6 The arithmetic panel measures TRANSFER, not retention — and D2's exclusion is uncertifiable

Verified independently from `arm_correct_total`, `source_correct_total`, `lost_total` and
`arm_correct_outside_source`. The identity `kept + NEW == panel` holds for **all seven arms**:

| arm | rows | panel | **kept /47** | lost | **NEW** | vs source |
|---|---:|---:|---:|---:|---:|---:|
| frozen source | — | **47** | 47 | 0 | 0 | 0 |
| **D1** | 384 | **77** | **40** | 7 | **37** | **+30** |
| D2 | 768 | 68 | 35 | 12 | 33 | +21 |
| D4 | 3,072 | 59 | 30 | 17 | 29 | +12 |
| DOSEINT α=0.25 | 2,697 | 61 | **40** | 7 | 21 | +14 |
| DOSEINT α=0.50 | 2,697 | 59 | 37 | 10 | 22 | +12 |
| DOSEINT α=0.75 | 2,697 | 58 | 31 | 16 | 27 | +11 |
| DOSEINT α=1.00 | 2,697 | 53 | 30 | 17 | 23 | +6 |

> **Every arm beats the frozen source's 47. Training on JSON transition rows makes the model
> BETTER at arithmetic — and more rows makes it less better.** So the gate's `panel >= 62` is
> not *"do not forget arithmetic"*; source is 47. It means **"keep at least half of D1's
> transfer gain."**

**Interpolation restores the kept component and cannot restore the gain.** At α = 0.25 the
high-dose arm recovers **kept = 40, exactly D1's figure** — while NEW is **21 against D1's 37**.
Both components scale with the delta, so scaling it down removes the damage and the transfer
together. **D1 is strictly better than every DOSEINT arm here: kept 40 AND NEW 37 at once.**

That is why DOSEINT's primary failed: `acq >= 284` needs high dose, `panel >= 62` needs low
dose, and **α does not decouple them.** §8's "α moves along the curve" now has a mechanism.

### D2 is excluded by an endpoint that cannot adjudicate it

D2 clears the arithmetic arm (68) and all three family floors. It fails the gate **solely** on
`retained 35 < 37` — two rows. Paired over the 47 source-correct rows, from the per-class
`lost_ids`:

    D1 keeps & D2 loses :  5
    D2 keeps & D1 loses :  0
    discordant          :  5    net +5
    exact McNemar p     :  0.0625
    smallest detectable |net| at 5 discordant pairs : UNATTAINABLE

> **The deficit is perfectly one-directional — 5 losses, 0 reversals, the strongest pattern
> available — and at 5 discordant pairs the endpoint CANNOT reach p ≤ 0.05 however the rows
> fall.** 2 × 0.5⁵ = 0.0625.

**This is a sharper objection than "it is noise."** The difference does not look like noise; it
looks consistent. The problem is that **the instrument is structurally incapable of certifying
it in either direction at this discordance**, and the gate converts that uncertifiable 5-row
deficit into a hard exclusion — while D2's **+37 acquisition** on the dose bank clears the ~20
cross-draw floor and *is* measurable.

**A measurable gain is being refused by an unmeasurable cost.** That is
`selection-rule-selects-against-acquisition` on the dose axis at **three times the stakes** —
+37 against the knee's +12.

**And it was on the record from the start.** `PROCESS_DOSE_FRONTIER.md` already wrote that D2
sits *inside* D3's seed spread on both panel and retention, is "not a dose effect", and that its
gate failure "is seed noise landing on the wrong side of a hard threshold." Nobody acted on it.
The seed argument and the paired argument are different — one says a re-run might flip it, the
other says this run cannot be adjudicated — and **both point the same way.**

### What is NOT proposed here

**No promotion of D2, and no rule repair.** Repairing a selection rule after seeing which arm it
excludes is precisely the move the registration discipline exists to prevent, and this document
has enforced that against itself twice today. The defect is reported; **the remedy is the
operator's and must be registered before any arm is scored under it.**

### Scope

The panel decomposition is **within-instrument**. Acquisition figures are on the dose bank
`45674251…` and are **not** comparable to select-views numbers. One seed per rung.
`per_class.reference_n` **varies between arms** — it is a union denominator, so do not decompose
per-class off it; the four aggregate fields above are the safe route.

---

# 8.7 CORRECTION — the panel cannot order its own arms, and §8's erosion claim rests on one seed

PANEL-001 measured the arithmetic panel's own noise floor by running **three draws of the
identical recipe** — 384 rows, 28 updates, seeds 17 / 23 / 29. Totals verified here from the
retention artifacts:

| arm | panel /150 |
|---|---:|
| `panel001_d1_s17` (= `rrp008_D1`, the recorded D1) | **77** |
| `panel001_d1_s23` | 71 |
| `panel001_d1_s29` | **67** |
| `rrp008_D2` | **68** |

> **The same recipe spans 77 → 67, and D1 seed 29 scores BELOW D2.** "D1 beats D2 on the panel"
> depends on which draw you compare. The 77 / 68 / 59 column **cannot order its own top two
> rungs.**

The paired session reports the seed difference is *statistically significant* — `s17 → s29`
4 gain / 14 loss, p = 0.0309. **I verified the totals, not the paired tests**: the 150-row panel
correctness vector is not reconstructible from the retention artifacts' `lost_ids`, which cover
only the 47 source-correct rows. Recorded as theirs, single-implementation.

## What this does to §8

§8 states: *"High dose costs the panel ~16–24 rows and interpolation does not buy it back"* and
uses **D1 = 77** as the reference throughout. **That 77 is seed 17 — the most favourable of the
three draws.** Against seed 29's 67, DOSEINT's best interpolant (61) is only **−6**.

The paired session's decomposition, which I have not re-derived: the dose→panel erosion against
α = 0.25 is significant on **one seed of three** (p = 0.0015 / 0.0872 / 0.3075), and **within the
DOSEINT ladder no pair is significant at all** (all p ≥ 0.18) — 61 / 59 / 58 / 53 is one noise
band. What survives is that the *endpoints* differ: D1 → α = 1.00 is significant on all three
seeds.

> **So §8's "the arithmetic panel is the binding constraint" stands as a statement about the
> endpoints and NOT as a ranking of the interpolants.** The panel separates a low-dose arm from a
> full-dose one; it does not separate the interpolants from each other, and it does not reliably
> separate D1 from D2.

## And §8.6's D2 finding is strengthened, not weakened

§8.6 argued D2 was excluded on `retained 35 < 37`, a difference its endpoint cannot certify.
**The panel now turns out to be seed-unstable at the same scale**: D2's panel 68 sits inside the
D1 recipe's own 67–77 range. So **two of the gate's three conjuncts cannot separate D1 from D2**,
and the one that excluded D2 is the one with the uncertifiable band.

This is `acquisition-band-not-resolvable` reappearing on the panel axis: **an instrument that is
go/no-go at the endpoints is being read as a ranking in the middle.**
