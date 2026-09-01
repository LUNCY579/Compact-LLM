# Amendment v3.1 — Donor-Side Functional Atlas and Low-Rank Translator

Protocol version: `3.1.0-diagnostic`

Parents: v1 `preregistration.md` `1.0.0`, `amendment_v2_exploratory.md`
`2.0.0-exploratory`, `amendment_v3_transport_diagnosis.md` `3.0.0-diagnostic`.
All three are immutable and unchanged.

Frozen before any donor weight is read.

## Why this unblocks

`e5_e6_blocked_no_donor_weights.json` recorded E5 and E6 as blocked because the
donor snapshots were absent. The operator has now authorised a pinned
re-download. E5 and E6 proceed under the designs frozen in that file, refined
here for the actual hardware.

## What v3 established, and why E5 is decision-relevant

v3 showed that no operator this study has ever run could transport a *direction*:

- P4 transmits 8 scalars per tensor and zero directions; its realised payload is
  a participation ratio of 1.40, about one scalar per tensor.
- P5 moves donor singular vectors by raw coordinate slicing, asserting that
  donor neuron *i* corresponds to target neuron *i*.
- The cross-donor agreement statistic that motivated the line does not survive a
  shape-matched null (real 0.9682, iid Gaussian at true donor shapes 0.9988).
- In the target's frame the transported direction is indistinguishable from
  random (cosine +0.0002 against `Delta_Supa`).

So the study's central hypothesis is **untested**, not refuted. E5 tests it.

## Hardware envelope, measured at freeze

- Total system RAM 15.2 GiB, free 3.6 GiB at freeze.
- Qwen3-4B is 7.49 GiB and Qwen3-8B is 15.26 GiB on disk in bfloat16.
- Installed Torch is `2.13.0+cpu`; CUDA is unavailable.

**Neither donor can be loaded conventionally.** The 8B model alone exceeds total
RAM. E5 therefore uses a streaming layer-by-layer forward pass over
memory-mapped safetensors: one transformer layer's weights are materialised at a
time, upcast to FP32 for the matmul, and freed before the next layer is read.
Peak resident memory is one weight tensor plus the activation stream, not one
model.

This is a hard requirement, not an optimisation.

## Correctness gate (Gate 0 for E5)

A hand-written streaming forward can be subtly wrong — Qwen3 uses grouped-query
attention, per-head `q_norm`/`k_norm` RMSNorm, and RoPE with `rope_theta` 1e6.
The implementation is therefore validated before any donor is touched:

> Run the streaming forward on **Qwen3-0.6B**, which fits in RAM, and compare
> against `transformers`' own `Qwen3ForCausalLM` on the same prompts. Require
> max relative error below `1e-3` on final-token hidden states at every captured
> depth and on final-token logits.

If that gate fails, E5 does not run and the failure is recorded. No donor
measurement derived from an unvalidated forward pass may be reported.

## E5 — cross-scale functional alignment atlas

**Hypothesis HT6.** Donor and target residual streams at matched *relative*
depth are more similar than the +0.0002 weight-space alignment already measured,
and that similarity is prompt-specific rather than a constant offset.

- Prompts: the frozen `calibration.jsonl` split only
  (SHA `d2f1f415e3cf34a8c7bb7a8cdef4240a8ce993c41d9b25a5d1471f19b8f453a1`),
  **prompts only**. No answer, scorer, or evaluation-bank field is read.
- Models: Qwen3-4B and Qwen3-8B, one at a time, released before the next.
- Depths: relative fractions `{0.00, 0.25, 0.50, 0.75, 1.00}` of each donor's
  36-layer stack, i.e. layers `{0, 9, 18, 27, 35}`, plus the embedding output.
  These match the target's `{0, 7, 14, 21, 27}` of 28 by construction.
- Capture: final-prompt-token residual state at each depth, and the final-token
  logit vector. The tokenizer and vocabulary are identical across all models, so
  donor and target logits are directly comparable.
- Pooling: both `last` and `mean`, as in v3 E4, since v3 showed the two can
  disagree.

**Measurements.** Because donor width (2560 / 4096) differs from target width
(1024), raw cosine is undefined across scales. The atlas therefore uses
representations that are invariant to the basis:

1. **Linear CKA** between the donor's and the target's prompt-by-feature
   matrices at every (donor depth × target depth) pair. This is the depth × depth
   surface v1 and v2 never measured.
2. **Representational similarity analysis**: the prompt × prompt similarity
   matrix computed within each model, then correlated across models (Spearman on
   the upper triangle). Basis-free and width-free.
3. **Logit-space agreement** on the shared vocabulary: cosine, symmetric KL,
   top-1 agreement between each donor and Qwen3-0.6B, and between each donor and
   Supa.
4. **Donor-consensus in function space**: the same statistics between Qwen3-4B
   and Qwen3-8B, which tests whether the 4B/8B agreement that failed in
   parameter space exists in representation space.

**Controls, frozen.** Every similarity is reported beside: prompt-shuffled
pairing; a random matched-norm direction; and a within-model depth permutation.

**Decision rule, frozen before the measurement.**

- If donor↔target CKA at matched relative depth does not exceed the
  prompt-shuffled control by at least 0.10, and donor↔donor functional agreement
  does not exceed its own control, then outcome **D** is established: the donor
  consensus is not portable, and E6 does not run.
- If donor↔donor functional agreement clears its control but donor↔target does
  not, that is outcome **D** with a sharper boundary: the donors share structure
  that has no target-side counterpart.
- E6 runs only if donor↔target similarity clears its control.

## E6 — low-rank functional translator

Runs only on an E5 pass.

- Fit `A` minimising `||z_target - A z_donor||^2` at matched relative depth, by
  ridge and by orthogonal Procrustes, at ranks `{4, 8, 16}` and full.
- **Sample-size hazard, recorded before the fit.** 59 prompts give 59 final-token
  samples against 1024 target dimensions; that fit is rank-deficient. E6 must
  therefore use **all token positions** (approximately 9,013 per model on this
  split), and must report held-out error on a **prompt-disjoint** split, not a
  token-disjoint one, since tokens within a prompt are not independent.
- Report held-out RMSE and cosine against the v1 embedding-fitted baselines
  (ridge 0.335; Procrustes 0.536 for 4B, 0.275 for 8B). A per-depth
  residual-stream map that does not beat those is not progress.
- Fitting is a CPU least-squares problem on cached activations. It is not
  training and no gradient step is taken on any model.

## What E5/E6 may and may not do

They are diagnostics. They may not create a candidate checkpoint, tune a
coefficient, or select a region for injection. If E6 passes, the resulting
candidate design must be frozen in a further amendment before any checkpoint is
built.

DEV-FULL is not run. The sealed holdout remains CLOSED and is not read.

## Provenance

Donors are re-downloaded at the exact revisions already recorded in
`01_environment/huggingface_cache_verification.md`:

- `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`
- `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`

Every downloaded weight shard's SHA-256 is verified against the values recorded
in `02_architecture/qwen4b.json` and `qwen8b.json` before any measurement. A
mismatch aborts E5 and is recorded as a provenance failure.

## Amendment policy

Inherited from v1. Immutable after the first E5 measurement command.
