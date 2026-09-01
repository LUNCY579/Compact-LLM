# Cross-Scale v3 — Diagnosing the Transport Operator

Protocol: `amendment_v3_transport_diagnosis.md` version `3.0.0-diagnostic`

Parents: v1 `preregistration.md` `1.0.0` and `amendment_v2_exploratory.md`
`2.0.0-exploratory`, both immutable and both unchanged by this stage.

Base commit: `538687fb8ddab999f825e300e7c91d43947c5c4e`

Sealed holdout: **not opened.** DEV-FULL: **not run.** Candidate checkpoints
built by v3: **zero.**

## Abstract

v2 concluded that the problem was the transport operator `T` but never measured
what `T` transmits. v3 measures it.

The P4 operator keeps the *target's* own singular vectors and transplants only
the donor's normalised top-8 singular-value profile. Its induced update has an
exact closed form, `a_i = strength · ||W_target|| · (p_i − q_i)`, verified to
1.5e-14. The donor's entire contribution is 8 scalars per tensor and no
directions at all: 1,568 numbers, or 3.9e-7 of Qwen3-4B. Mean-centred across
tensors those 8 scalars collapse to a participation ratio of 1.40, so the
realised payload is nearer 275 numbers.

Holding the operator fixed and varying the donor retires the study's headline
statistic. The v1 report read the 0.932 cosine between the projected 4B and 8B
deltas as evidence that the donor signal is not random. Under the controls the
amendment actually froze, that agreement sits inside an architecture-only null
band; and iid Gaussian matrices at the true donor tensor shapes reach **0.9988**
through the identical operator, *above* the real donors' 0.9682. Donors sharing
nothing but their shapes reproduce the statistic better than the real donors do.

In the target's own reference frame the transported direction is not merely
weak, it is absent: its cosine against `Delta_Supa = Supa − Qwen3-0.6B` is
+0.0002, against +0.0000 for norm-matched random, −0.0001 for sign-shuffled, and
+0.0001 for layer-shuffled. No depth bucket and no module family departs from
zero. And Supa's own adaptation places 7.2e-06 of its energy in the rank-8
subspace P4 writes into, against a 4.4e-06 chance level.

Functionally, the v1 winner and the v2 TIES candidate are indistinguishable from
ungated Supa: logit cosine 1.0000, symmetric KL 0.0005, top-1 agreement 0.983 on
59 calibration prompts. The two candidates are identical to each other. For
scale, Supa's own adaptation sits at KL 7.99 from Qwen3-0.6B.

The bounded conclusion: **every cross-scale operator this study has ever tested
was incapable of transporting a direction, and the evidence that once motivated
building a wider one has been withdrawn.** That is a much sharper negative than
v1 or v2 recorded, and it is *not* a negative about cross-scale transfer as
such.

## What was run

| ID | Experiment | Status | Cost |
|---|---|---|---:|
| E1 | Transport channel capacity, with donor-varying controls | complete (corrected) | < 1 s |
| E1b | Intended signal versus bfloat16 storage noise | complete | 4 s |
| E1c | Shape-matched Gaussian donor null | complete | 5 min |
| E2 | Target-basis alignment of the shared direction | complete | 48 s |
| E3 | Depth-bucket and module-family localization | complete | (with E2) |
| E4 | Functional alignment atlas, 0.6B basis, last-token | complete | 7.5 min |
| E4b | Same atlas, mean-pooled — robustness check | complete | 7.5 min |
| E5 | Cross-scale functional atlas, donor side | **blocked** | — |
| E6 | Low-rank functional translator | **blocked** | — |

### E1 — what crosses the bridge

`spectral_profile_transfer` reads exactly two things from the donor: its top-8
singular values and its Frobenius norm. The donor's own singular *vectors* are
computed and discarded on the same line. Every direction in the resulting update
is one of the target's own top-8 singular directions, or the target weight
itself.

- 196 of 310 stored tensors are touched (28 layers × 7 matrix families).
  Embeddings, all norms, and `q_norm`/`k_norm` are untouched.
- Channel width: 196 × 8 = **1,568 scalars**, 3.9e-7 of Qwen3-4B.
- Realised payload: participation ratio **1.40**, i.e. about one scalar per
  tensor — how fast that tensor's spectrum decays. Nearer **275** numbers.
- Donor directional information: **zero**.

### E1/E1c — the agreement statistic does not survive fair nulls

| Arm | cos(delta_4B, delta_8B) |
|---|---:|
| **Real Qwen3-4B vs Qwen3-8B** | **0.9682** |
| Norm-matched random donors (3 seeds) | 0.9109 – 0.9146 |
| Layer-shuffled within module role (3 seeds) | 0.7697 – 0.7869 |
| Sign-shuffled (3 seeds) | −0.0676 – −0.0590 |
| Role-mean donor (56 numbers total) | 0.8715 |
| **iid Gaussian at true donor shapes (3 seeds)** | **0.9988** |

The last row is decisive. Two random matrices agree *better* through this
operator than Qwen3-4B and Qwen3-8B do. The mechanism follows from the closed
form: the delta is the gap between two near-parallel normalised spectra (median
cos(p, q) = 0.9974), and the normalised top-8 spectrum of a large matrix is
governed mostly by its aspect ratio. Any two donors of these shapes therefore
produce nearly the same update.

Separately, projecting each real delta onto a "role-mean donor" — one pooled
profile per module family, 56 numbers in total, carrying no tensor-specific
information — explains **84.9%** of the 4B delta energy and 89.6% of the 8B.

This does not mean nothing donor-specific arrives. The orthogonal residual,
15% of the energy, still agrees across the two donors at cosine **0.847**. Real
donor structure does reach the target. It is a minority of an already narrow
channel, and the 0.932 statistic was never measuring it.

### E1b — storage precision is not the explanation

86.1% of the delta actually merged into the v1/v2 candidates is the intended
rank-8 signal, at 1.02× the intended FP32 magnitude. The remaining 13.9% is
bfloat16 quantisation of the target's own weights — shallow, but spread over
every direction, which is what inflated the on-disk delta's entropy-based
effective rank to 448–539 despite a rank-8 construction.

**The v1 and v2 evaluations tested the intervention they meant to test.** This
experiment does not rescue their negative result.

### E2/E3 — the direction points nowhere the target goes

Against `Delta_Supa`, the target model's own known-good adaptation direction:

| Direction | cos vs Delta_Supa |
|---|---:|
| delta_4b / delta_8b / delta_mean | +0.0002 / +0.0001 / +0.0002 |
| delta_p5_4b | +0.0007 |
| control: norm-matched random | +0.0000 |
| control: sign-shuffled | −0.0001 |
| control: layer-shuffled | +0.0001 |

Nothing is distinguishable from the controls. Broken down by depth bucket
(early/middle/late) and by module family (q/k/v/o/gate/up/down), every cell lies
within ±0.0016 of zero. **There is no region where the transported direction is
target-relevant**, so the localization and micro-injection paths are not
justified by evidence.

The mechanism is visible in one number: Supa's own adaptation places
**7.2e-06** of its energy inside the rank-8 subspace P4 writes into, against a
**4.4e-06** chance level — a 1.7× enrichment, i.e. seven parts per million. P4
injects into directions the target's own successful adaptation does not use.

### E4 — the candidates are functionally Supa

Seven checkpoints over 59 frozen calibration prompts, prompts only, one model
resident at a time.

| Pair | logit cosine | symmetric KL | top-1 agreement |
|---|---:|---:|---:|
| Qwen3-0.6B vs Supa | 0.7569 | 7.9879 | 0.3898 |
| Qwen3-0.6B vs P4-4B | 0.9909 | 0.1265 | 0.9492 |
| Qwen3-0.6B vs P4-8B | 0.9943 | 0.0915 | 0.9661 |
| Qwen3-0.6B vs P5-4B | 0.9495 | 1.3732 | 0.7627 |
| **Supa vs CONS_MEAN_A1500** | **1.0000** | **0.0005** | **0.9831** |
| **Supa vs TIES_K50_A1500** | **1.0000** | **0.0006** | **0.9831** |
| CONS_MEAN vs TIES_K50 | 1.0000 | 0.0000 | 1.0000 |

Supa's own adaptation moves the model by KL 7.99 from its base. The entire
cross-scale intervention moves it by KL 0.0005 — four orders of magnitude
smaller — and changes the argmax on 1 of 59 prompts. The v1 winner and the v2
TIES candidate are *bit-for-bit equivalent in behaviour* to each other.

This retroactively explains every prior null. MERGEDEV 48/59 versus 47/59 and
MERGEDEV-fast 13/16 versus 12/16 were single-case coin flips on a model that is
functionally unchanged. Those evaluations were not weak evidence against
cross-scale transfer; they were measurements of a perturbation too small to
register.

**Robustness.** Under last-token pooling the transported donors showed a small
prompt-specific alignment with Supa's representation change in late layers
(P4-4B layer 27: cosine 0.1945 against a 0.0344 prompt-shuffled control). Under
mean pooling over all prompt positions that lift vanishes — every cosine is
matched by its shuffled control to within 0.01. Both views are reported. The
last-token view is decision-relevant; the mean-pooled view bounds how much of
the lift is positional.

## What was skipped, and why

- **E5 (donor-side functional atlas) and E6 (low-rank functional translator):**
  `blocked_no_donor_weights`. Qwen3-4B and Qwen3-8B are no longer resident in
  the Hugging Face cache — only lock directories remain. This is not the v2
  `not_run_compute` situation: the installed Torch build is `2.13.0+cpu` and
  reports CUDA unavailable, so GPU occupancy is irrelevant. Both designs are
  frozen in `protocol/e5_e6_blocked_no_donor_weights.json` rather than dropped.
- **Localization sub-division and micro-injection:** not justified. E3 found no
  depth bucket or module family with any target-relevant signal, so there is no
  coherent region to inject.
- **New candidate checkpoints, coefficient sweeps, DEV-FULL, sealed:** out of
  scope by construction. v3 stops at Gate 1.
- **Fisher weighting, further TIES/DARE variants:** still not justified. v2
  showed they collapse onto mean consensus, and E4 now shows why — every
  candidate in that family is functionally Supa.

## Resource consumption

| Quantity | Value |
|---|---:|
| GPU hours | 0.0 (CUDA unavailable) |
| CPU hours | 0.27 |
| Model loads | 14 |
| Peak concurrent models in memory | **1** |
| Peak process RSS | 2.86 GiB |
| Prompt tokens processed | 126,182 |
| Generated tokens | 0 |
| Intermediate disk | 233 MiB (gitignored) |
| Candidate checkpoints created | **0** |

All parameter arithmetic was FP32 on CPU over memory-mapped safetensors,
streamed one tensor at a time. E1/E1b/E1c/E2/E3 loaded no model at all. One
interruption occurred after four of seven captures; all four were revalidated
for shape, finiteness, and checkpoint digest before resuming, and all four
digests matched the v1 frozen artifacts.

## Strongest result

The transport operator was characterised exactly, in closed form, at machine
precision, from already-committed artifacts and at essentially zero compute
cost. `a_i = strength · ||W_target|| · (p_i − q_i)` is the whole of what two
frontier donors were ever able to say to the target model.

## Strongest negative result

The study's headline evidence has been withdrawn. The 0.932 cross-donor cosine
was read for two protocol versions as showing that "the donor signal does not
appear random." Shape-matched Gaussian donors reach 0.9988 through the same
operator. The statistic is a property of the operator and the donor tensor
shapes, and it carries no information about donor content.

A secondary negative, equally important for planning: the transported direction
is orthogonal to the target's own adaptation direction to four decimal places,
everywhere, with no localized exception.

## Did the cross-scale hypothesis get stronger or weaker?

**Both, and the distinction matters.**

*Weaker as a claim about the tested operator family.* P4 and P5 are now known to
be near-empty channels. P4 transmits ~275 effective numbers and no directions.
P5 does move donor singular vectors, but by raw coordinate slicing —
`u[:out_t, :k]`, `vh[:k, :in_t]` — with no basis alignment, which asserts that
donor neuron *i* corresponds to target neuron *i*. Outcome **D** is well
supported *for these operators*.

*Not weaker as a claim about cross-scale transfer in general.* No operator that
transports donor directions into an aligned target basis has ever been run in
this study. This is not an oversight of framing — the aligned path is literally
present in the codebase. `direction_transfer.ridge_linear_map` and
`direction_transfer.project_weight` build feature maps from paired activations,
and a grep shows they are referenced **only by
`tests/test_direction_transfer.py`**. No script and no checkpoint in this
repository uses them.

The one basis-alignment attempt v1 did make was fitted on the vocabulary
*embedding* matrices and then applied at every depth; its held-out cosines were
0.335 (ridge) and 0.536 / 0.275 (Procrustes). A per-depth residual-stream
correspondence has never been fitted.

So the honest position is: the study has accumulated strong evidence that
spectral-statistic transport does not work, and has not yet tested the
hypothesis it set out to test. Discriminating **B** (functional transfer exists)
from **D** (donor consensus is non-portable) is still open, and is still the
scientifically valuable question.

## The exact next experiment the evidence justifies

**E5 then E6, in that order, and nothing else.**

1. **E5.** Capture final-token residual-stream states for Qwen3-4B and Qwen3-8B
   on the frozen calibration prompts, at matched *relative* depths (fractions
   0.00/0.25/0.50/0.75/1.00 of each donor's 36-layer stack, i.e. layers
   0/9/18/27/35), one donor loaded at a time, CPU float32, streamed and
   released. Build the depth × depth cross-scale similarity surface that neither
   v1 nor v2 ever measured. Decision rule: if donor↔target functional similarity
   at matched depth is no better than the ~0.0002 weight-space alignment already
   measured, outcome **D** is established and the research line closes with a
   clean, well-supported negative.
2. **E6, only if E5 shows structure.** Fit a per-depth low-rank map
   `z_0.6 ≈ A z_donor` by ridge and orthogonal Procrustes at ranks 4/8/16, on
   cached activations, on CPU. Report held-out error on a prompt-disjoint split.
   The frozen design already records the sample-size hazard: 59 prompts give 59
   final-token samples against 1024 target dimensions, so E6 must use all token
   positions or an explicit low-rank constraint.

**This requires a decision that is not mine to make.** Both experiments need the
donor checkpoints, which means re-downloading roughly 25 GB at the two pinned
revisions already recorded in
`01_environment/huggingface_cache_verification.md` (Qwen3-4B
`1cfa9a7208912126459214e8b04321603b3df60c`, Qwen3-8B
`b968826d9c46dd6066d109eabc6255188de91218`). Disk is available: 269 GB free.
Nothing else in v3 is blocked, and no further local measurement would change
this recommendation.

If the download is declined, the correct action is **not** to build more
candidates. It is to close the line at outcome **D-restricted**: "spectral
statistic transport is a near-empty channel, and the evidence once cited for
donor consensus does not survive shape-matched nulls."

## Escalation

- **DEV-FULL: not justified.** v3 produced no candidate. The existing candidates
  are functionally identical to Supa, so re-running them on a larger development
  set would measure evaluation noise, not transfer.
- **Sealed: remains CLOSED.** The preregistered escalation criterion requires a
  frozen finalist packet that beats both controls. No such packet exists, and v3
  did not create one. The sealed JSONL was not read.
- **Recovery LoRA: not justified.** It is reserved for the strongest frozen
  training-free candidate; there is none.

## Corrections to the v1 and v2 record

The frozen v1 and v2 files are unchanged. These are corrections *by this stage*
to claims those stages made, recorded here rather than by rewriting them:

1. v1 `final_report.md` and v2 `v2_addendum.md` both treat the 0.932/0.926/0.915
   cross-donor statistics as evidence of a shared scale direction. E1/E1c
   withdraw that reading. The statistics are correct as measurements; the
   inference from them is not supported.
2. v1 answer 8 ("Are delta4 and delta8 aligned? Yes, strongly") should read:
   aligned, but no more than shape-matched random donors are.
3. v1 answer 9 ("Evidence for a common scale trajectory? Yes for target-basis
   spectral statistics") should read: the observed spectral agreement is
   attributable to tensor shape and module role.

## Errors corrected within v3

Recorded because the amendment forbids silently discarding results.

- E1 attempt 0001 used donor-varying nulls that randomised the donor profile
  norm over a six-fold range — the very quantity the delta consists of — and
  omitted the frozen sign-shuffled arm. It concluded the cross-donor agreement
  was "real and tensor-specific." That conclusion is retracted; the run is
  preserved under `e1_channel_capacity/attempt_0001_unfair_nulls/`.
- E1 attempt 0001 published a `real_4b_vs_null_donor = −0.2462` row that is an
  exact algebraic zero, rescued by a denominator guard.
- E1's closed-form validation checked only the delta's scalar norm, against a
  float32 number written by the same operator. It now checks coordinates against
  the stored delta tensors.
- E1b's interpretation was drafted before its measurement and asserted the
  merged delta was "mostly rounding error." The measurement said 86.1% signal
  and the text was corrected to match.
