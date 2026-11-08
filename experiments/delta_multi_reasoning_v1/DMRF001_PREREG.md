# DMRF-001 — does direction selection do anything the magnitude does not?

**Written and frozen before the N16 arm was built or any N16 token was generated. No training.
Selects nothing, promotes nothing. Long context and abstention are report-only on a SPENT
suite. The Stage 0 audit (`DMRF001_AUDIT.md`) was complete before this document; its findings
are inputs here, not outcomes of it.**

---

## 1. The question, narrowed to what is still open

The Stage 0 audit closed most of what the driving protocol asked to test, either by locating an
existing measurement or by contradicting a claim:

- Rank truncation's behavioural effect is **measured** (RANK-001, completed 23:19 by a
  concurrent session): r ∈ {4,16,64} on acquisition, arithmetic, retention, long context and
  abstention.
- The 4B same-task adaptation is **measured** (QWEN-FT-001), on legitimately disjoint training
  data (audit §4).
- Model-level routing's *premise* is measured: the two components are strongly complementary
  (Qwen3-4B acquisition 0/384 vs the best core's 232; long context 178 vs 164).

One thing the protocol asks for is **not measured anywhere and cannot be inferred**: whether
the rank-16 truncation's behaviour comes from *selecting 16 directions* or merely from
*applying a smaller update*.

Audit §9 shows the second explanation already accounts for the acquisition score. A uniform
α = 0.75 scaling — norm ratio exactly 0.75 against r=16's 0.812 — scores **234/384** where
r=16 scores **235/384**, and the α line is monotone so α ≈ 0.81 is bracketed at 234–235. On
this endpoint, discarding 1008 of 1024 directions costs at most one row of 384.

That bracketing argument has one gap, and closing it is this experiment. **α is a *uniform*
scaling; `P16` shrinks each tensor by its own factor.** Measured on this delta, those per-tensor
factors span **0.712 to 0.960** — a 1.35× range. So r=16's norm profile is *heterogeneous* and
α ≈ 0.81 is only a global match. If the per-tensor profile carries information, a heterogeneous
norm match could differ from both the uniform control and the truncation.

**N16 is that heterogeneous norm match.** It is the tightest available control on r=16.

---

## 2. The arms, pinned

`W0` = Supa 1.7.0-devfreeze, `962abc5b…`. `Delta` = the 220 trained tensors of the RRP-015
control (seed 17, checkpoint-00028) minus `W0`, computed in float64 from `07156fb5…`.

| arm | definition | availability |
|---|---|---|
| **B** | `W0` | exists |
| **D** | `W0 + M_0.25 ⊙ Delta` — the promoted k=0.25 magnitude-mask core, `a28629c9…` | exists, measured |
| **L** | `W0 + P16(Delta)` — rank-16 per 2D tensor, `7ebd0718…` | rebuildable, measured |
| **N16** | `W0 + s_t · Delta_t`, `s_t = ‖P16(Delta_t)‖_F / ‖Delta_t‖_F` per 2D tensor | **THIS EXPERIMENT** |
| *A75* | `W0 + 0.75·Delta`, INT-001 `d5d12da3…` | exists, measured — the uniform control |
| **G0** | Qwen3-4B, snapshot `1cfa9a72…` | exists, measured |
| **G1** | Qwen3-4B + LoRA r=16 on `microfit-train.jsonl` | exists (4-bit), measured |

**Pinning note, required because the authoritative artifact differs from the protocol's
formalism.** The protocol writes `D = W0 + k·Delta` as a scalar multiple. The promoted k=0.25
artifact is a per-tensor **magnitude mask** (`CORE.json`: `theta_supa + M_k ⊙ delta`). The two
operations were measured separately and diverge by 16 acquisition rows at nominal 0.50
(audit §5). `D` is therefore the mask core, and the scalar family is INT-001's α line, used
here as the uniform magnitude control rather than as `D`.

**N16 is one algebraically determined arm, not an alpha sweep.** Every `s_t` is fixed by the
delta itself; there is no free parameter and no value to choose after seeing a score.

**Nonmatrix policy, identical to RANK-001's so that L and N16 are comparable at all.** The 80
one-dimensional norm deltas are kept in full, `s_t = 1`. Measured: 46,080 parameters =
**0.0146% of the delta's parameters and 0.0146% of its norm**, so they cannot carry a result.
140 of the 220 trained tensors are 2D. **Zero-denominator policy:** `‖Delta_t‖ = 0` returns the
source tensor and records `scale: 0.0`; the measured count of such tensors on this delta is
**0**, and the branch exists so the algebra is defined rather than incidentally safe.

**Dtype.** FP32 throughout, never BF16. INT-001 measured a BF16 round erasing 60.61% of this
update's elements; SPARSE-001 measured it zeroing 28.49% of a pruned arm's kept elements. A
BF16 copy would be a different, measurably worse model. Norms and scale factors in float64.
TF32 off.

---

## 3. Instruments and permissions — unchanged, and the bars are unchanged

| instrument | n | role | permission |
|---|---:|---|---|
| SELECT-VIEWS-001 | 384 | acquisition | **SELECTION-ADMISSIBLE**; floors 0.2344 / 0.0781 / 0.0703 published pre-hoc |
| 150-row arithmetic panel + 47 source answers | 150/47 | retention | the gate's own instrument |
| Long-Context Demonstration Suite | 256 | long context | **SPENT**, barred TRAIN/SELECTION/CONFIRMATION — **REPORT-ONLY** |
| the 64 unanswerable rows inside it | 64 | abstention | same bank, same bar — **REPORT-ONLY**, and not independent of the long-context score |

`QUARANTINE.json` is checked, not read at decision time. No quarantine is lifted. The three
`ANY_READ`-barred banks were not opened.

**Decision rule, applying to `canonical_correct` only.** A family clears only if the exact
two-sided binomial 95% lower bound exceeds the stronger of `copy_before` and
`constant_majority`. `contract_tolerant` never clears a floor. Strict and permissive arithmetic
are reported separately and never pooled.

---

## 4. Registered predictions

The primary contrast is **L vs N16 on acquisition**. Everything else is secondary.

1. **N16 acquisition lands in 232–237.** This is the prediction the arm exists to test. It
   brackets r=16's 235, the full delta's 235 and α=0.75's 234. If it holds, direction selection
   contributes nothing detectable on acquisition and RANK-001's Finding 1 is a storage result
   only.
2. **N16 acquisition ≥ 220.** The weaker form. If this fails, direction selection matters
   materially and RANK-001's mechanism reading is vindicated against my audit.
3. **N16 clears all three family floors**, as every arm at norm ≥ 0.6 has.
4. **N16 retained ∈ [39, 43] of 47.** α=0.75 retained 39, α=0.50 retained 42, r=16 retained 39.
5. **N16 long context ∈ [157, 165], and ≥ 159.** α=0.75 scored 163 against r=16's 159; if the
   full-rank update is gentler on long context at matched norm, N16 lands above r=16.
6. **No arm repairs long context beyond the noise band.** N16 does not exceed Supa's 161 by
   more than 13 rows. Stated so that a point estimate above 161 is not read as repair.
7. **Gate v2 PASS** (`arithmetic_permissive ≥ 62 AND retained ≥ 37 of 47 AND all three floors`).
8. **The legacy per-class retention monitor FAILS**, as it has for every arm of every screen in
   this programme including ones losing only 4 of 47.

**Falsification of the audit's §9 reading:** prediction 2 failing, or N16 acquisition falling
more than 15 below r=16's 235.

**Falsification of this arm's premise:** N16 landing within 1 row of α=0.75's 234 on every
endpoint, which would mean the per-tensor profile carries nothing beyond the global norm and
the uniform control was already sufficient.

---

## 5. How a difference may be read

REP-001's three independent draws of the same recipe bound what one draw can show:

| endpoint | single-draw spread |
|---|---:|
| acquisition | 7 of 384 |
| long context | 13 of 256 |
| arithmetic strict | 13 of 150 |

**No difference at or below these may be reported as a direction.** It is `WITHIN_NOISE`.
This is enforced in code (`analysis/paired.py::noise_verdict`) and tested
(`tests/test_paired.py`), so a table in this package cannot quote a within-noise difference
without the verdict beside it.

**Preservation margins, frozen now.** The strict criterion is **zero observed damage**: a
"no observed regressions on these registered cases" statement is permitted only when the paired
damage count is actually 0. A separate noninferiority criterion at margin 3 on
`references_retained` is registered, reported separately, and returns
`INCONCLUSIVE_UNDERPOWERED` when the discordant-pair count cannot detect the margin. **Failure
to find a significant decline is not preservation**, and a large McNemar p-value alone never
licenses a preservation claim.

Every paired comparison reports both-correct / repair / damage / both-wrong. Confidence
intervals resample **worlds**, never items, calls or paraphrases.

---

## 6. Execution, with numeric ceilings

Identical to the qualified lane so the numbers are comparable to the existing table:
interpreter `.venv-rab001-cu126`, transformers **4.51.3**, torch 2.10.0+cu126, implementation
digest `a329eb7bb0`, `max_context` 4096, greedy, TF32 off, device cuda, 0 rows skipped for
overflow required. Sidecars byte-identical to source plus a `MATERIALIZATION.json` per arm,
because the retention probe requires both.

**Ceilings, as numbers.** One model job at a time against 15.22 GiB host RAM. Build refuses
below **4.0 GiB** free (`build_n16.py::require_free`), matching the probes' own convention
rather than a laxer one; the probes' own `require(5.0)` / `require(3.0)` and mid-flight
`guard(1.5)` are **not weakened, not overridden, not retried**. Per-probe timeout **900 s**.
Disk: one arm is 2.38 GB against **33.3 GB** free; the arm's weights are deleted after its
battery with the output hash recorded first — the operational mistake SPARSE-001's Erratum 1
records is not repeated, and unlike RANK-001 the arm is hash-verified on rebuild before any
score is read.

Outputs are written to a new records directory `dmrf001-20260918/`. **Nothing is written to
`rank001-20260918/`, `int001-20260917/` or `sparse001-20260917/`**, which belong to other
screens and, in RANK-001's case, to a concurrently running session. Process inspection
precedes any native run; if another battery is live, this one stands down rather than
contending (audit §13).

On a resource, identity or data violation: stop the affected stage and write a create-only
receipt with `completed: false`. No safeguard is disabled and no partial run is presented as
finished. No outcome-driven budget extension.

---

## 7. What this screen cannot establish

One draw, one update direction, one base model. It cannot show that *two* skills compose —
that is C7, untested, and needs a second skill. Long context and abstention are report-only on
a SPENT suite and cannot be confirmed by anything on this host
(`CONFIRMATION_INSTRUMENT_GAP.md`). The arithmetic panel cannot be run on any Qwen arm at all,
because `verify_hf_candidate` structurally rejects a non-Supa model, so the 4B has no retention
measurement on the gate's own instrument.

`quality_improvement_established` stays false. `release_permitted` stays false. Nothing is
promoted.

---

## 8. Registered but NOT executed in this stage, with the exact blocker

These are designed and, where noted, implemented; they are listed so that "not run" is visible
rather than absent.

| item | status | exact blocker |
|---|---|---|
| O / T / REP / MR inference conditions on B, D, L, N16 | **implemented, NOT RUN** | Needs a frozen CAL/CHECK scout with independently checkable answers. SELECT-VIEWS-001 is selection-admissible and usable, but MRS prior art records a scout whose checkpoints sat at or below the constant-answer baseline, making arm contrasts uninterpretable; a scout must be built and floor-checked before MR is worth a single call. Not started in this stage. |
| the frozen input-only router (R_D0 / R_L0 / R_D1 / R_L1) | **designed, NOT IMPLEMENTED** | The premise is measured (components are complementary), but the two instruments are near-trivially separable by request shape, so a dispatch rule scored on them alone would be close to an upper bound. It needs ambiguous and shortcut-check cases in the scout first. |
| G1 at BF16 | **NOT RUN** | Needs a new training grant and ~1 h. 4-bit cost its own control 11 long-context rows, so a BF16 arm would likely score above 180. |
| G1 retention on the arithmetic panel | **BLOCKED, structural** | `verify_hf_candidate` requires six sidecars byte-identical to Supa and a receipt hashing to Supa's weights. No substitute was manufactured. |
| any confirmation of a long-context or abstention figure | **BLOCKED, structural** | No confirmation instrument exists on this host. |
