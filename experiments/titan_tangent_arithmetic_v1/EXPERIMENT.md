# TTA-001 protocol and interpretation

## Paper-derived procedure versus Rineya-specific extensions

| Aspect | Paper | This implementation |
|---|---|---|
| Predictor | Nonlinear, post-hoc linear, separately tangent-trained | Same distinction, same fixed base within each comparison |
| Output/loss | CLIP encoder, normalized embedding and fixed text head | Causal full-vocabulary logits; response+EOS cross-entropy; no CLIP normalization |
| Training | 2,000 iterations, batch 128, LR 1e-5, cosine, 200 warmup steps | Explicit `TrainSpec`; Rineya-matched defaults 28 updates, WSD, LR 1e-5; four matched A/B training runs |
| Parameter scope | Image encoder | Explicit canonical parameter prefixes; a restricted scope is a **partial NTK**, not full-model linearization |
| Weight decay | Paper uses AdamW | Port requires zero decay, matching the historical Titan recipe; delta decay would not equal decay on theta |
| Addition | Common alpha 0:0.05:1, tuned per method on held-out development data | Implemented; maximize macro normalized accuracy; deterministic smaller-alpha tie break |
| Normalization | Each task's own single-task accuracy, Eq. (7) | Implemented separately per method at full single-task strength; zero denominator is undefined |
| Negation | theta0 minus alpha*tau; preserve >=95% of control accuracy | Implemented when a distinct `control` dev/test bank exists; zero baseline control is uninformative |
| Disentanglement | Eq. (6), sum of task-conditioned prediction disagreements | Implemented for response sequences; optional 20x20 [-3,3] diagnostic surface |
| Evaluation | Classification accuracy | First port uses teacher-forced response exact match, NOT autonomous generation |
| Spectral evidence | Finite NTK eigenfunction estimates | **Not implemented**; output-response measurements must not be called eigenfunction localization |

Changing the scorer to free generation is a separately tested adapter extension, not an
interpretation of current teacher-forced results. Normalized scores alone never establish
usefulness: always report absolute scores and per-task acquisition alongside them.

## Stages

**M0, implemented and executed:** analytic JVP/gradient checks, a tied-head causal fixture,
three predictor paths, negative data/source tests, and honest undefined-metric behavior.
No pretrained models, tokenizer calls, official task training, or historical benchmark reads.

**M1, implemented but NOT RUN on a Rineya model:** CPU/eager exact-anchor qualification using
one bound developmental prefix: zero-delta identity, two central finite differences of a
nonzero direction, and nonzero finite reverse-mode gradients through the JVP. This is a
mechanical qualification, not all-input parity. An unsupported operator fails without a
finite-difference fallback pretending to be exact JVP.

**M2, implemented but NOT RUN:** reconstruct the original FP32 Titan delta by subtracting
exact Supa from the **trained partial weights**, not by treating the trained file itself as
a delta. Compare full delta, top-25% head, complementary tail and per-tensor norm-matched
full-rank control at fixed answer-free prefixes. An optional second endpoint enables raw and
norm-matched composition plus the exact, equally attenuated single-component controls.
No optimizer steps. This does not repeat the historical 384/256-row scores.

**M3, engine implemented but native study not registered or executed:** fresh task A/B training
from each chosen base, ordinary and tangent-trained, then addition/negation comparison.
Every method uses the same task-specific corpus, seed, supervision, token budget and schedule.
The post-hoc arm receives the exact ordinary-trained vectors. Train rows never select alpha;
dev selects alpha; test is first scored only after selection. Closed historical confirmation
banks are not accepted as this developmental test split.

Start with Supa as anchor and Titan as historical endpoint. Only after qualification repeat
within Nimbus, Halo or an explicitly bound Titan FP32 anchor, training new A/B vectors at each
anchor. Same architecture does not authorize reusing a Supa delta as another model's own task
vector. Cross-scale transfer remains out of scope.

## Complementary questions the code can distinguish

1. **Small update versus actual local linearity:** measure nonlinear output change and the
   JVP remainder on each task stratum. The relative error is undefined when the actual change
   is zero; output-energy shifts are not automatically damage.
2. **Sparsity versus update magnitude:** compare a top-k update with a full-rank per-tensor
   norm-matched update. Matching k=0.5 against alpha=0.5 is not norm matching.
3. **Composition versus attenuation:** for s_t=||A_t||/||A_t+B_t||, compare s_t(A_t+B_t)
   with s_t A_t, not with unscaled A_t. Also retain s_t B_t and the raw sum. Cancellation is
   an error, never a hidden fallback. These controls can be applied to full or rank-truncated
   deltas; they must use the same actual inputs and coefficients in each paired contrast.
4. **Linear leakage versus nonlinear interaction:** even an exactly linear model can have
   unwanted first-order cross-task effects. Zero mixed interaction is automatic for the
   tangent predictor and does not prove independently composable skills.
5. **Precision as an intervention:** Q_BF16(theta0+delta)-theta0 is a separate effective delta,
   not a scalar interpolation. Native BF16 merging is not allowed in this first port.

## Historic reasoning: supported observations, not inherited conclusions

The inspected history includes original Titan (`7df4329781574ba325fb665370e5437fe84e2eef`),
its precision amendment (`47d419bcf1a3dfe67e63108bd323f1d91b4d8cc7`), sparse promotion
(`5fb460edaa78019df253d2479c5e0def0e9db777`) and REP-001
(`954a0c2637e691b493c5ac3d488653a8c1adf675`). The corrected techniques record at
`07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd` reports a whole-model relative update norm of
0.000576, correcting a tied-embedding denominator double count. None proves linearity.

`experiments/delta_multi_reasoning_v1/analysis/COMPOSITION_PAIRED_20260920.md` at the parent
commit, blob `9c1a84afe27e9f136f880449381fd9b0f8fa2a7f`, motivates the matched-component
control. Its attenuation-only explanation used a ray interpolation, **not the missing native
control**. Do not inherit its strongest mechanism wording as an established finding. Its T16
stand-in also was not byte-identical to COMPOSE-001's arm. Some reconstructed arm weights were
reclaimed locally; local absence says nothing definitive about remote custody.

The RCI-001 record in `Rineya_Published_Cores/FAMILY_PERFORMANCE_AND_RESEARCH.md` at `07e53b0...`
reports Supa 40 versus Titan-Sparse 25 correct abstentions out of 128, with 0 gains/15 losses.
That is preservation counterevidence, not a training or selection bank to reopen here.

The parent commit also preserves sealed historical reporters and a GATEFIX-001 three-valued
retention helper. This package does **not** rewrite those reporters, re-score their outcomes,
or relabel teacher-forced scores as the legacy retention gate. Source inspection is not an
independent rerun of the archived experiments.

## Scientific claim boundary

The current status establishes software mechanics only. Future measurements must include
paired gains/losses, actual scope/norm/precision, per-task absolute scores, separate token and
wall-time costs, and results across independently drawn data/seeds (proposed 17, 23, 29).
Seed ranges are not within-draw significance thresholds. Nonsignificance is not equivalence.
A lossless or superiority claim needs a separately frozen, sufficiently powered instrument and
an explicit minimum effect/noninferiority margin; this package neither creates nor spends one.
