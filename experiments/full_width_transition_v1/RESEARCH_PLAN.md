# Research decisions and staged study plan

## 1. User decision and the question being changed

The previous fixed-parameter-budget study combined compression and architectural
conversion. The new priority is architecture viability at the original backbone
configuration, accepting a moderate parameter increase. Keep the experimental
questions separable: transition first, depth second, compression last.

This is not a plan to maximize a chosen development score indefinitely. "Best
potential" means providing a defensible learning path, informative controls,
adequate data and optimization opportunity, and explicit limits on conclusions.

## 2. Historical evidence, pinned rather than rewritten

Review baseline: ChatBubble commit
`aa9bdcd1ff805066b9566230006542704f8b9321`.
Relevant preserved files relative to `experiments/`:

- `architecture_benchmark_001/budget.py`: original-width C28 and D40 diagnostics
  already exist; C28's parameter-matched arm was never 896-wide.
- `rab001_exploration/ced_depth_sweep.py` and
  `architecture_benchmark_001/evidence/arm-quality/CED_DEPTH_SWEEP.json`:
  original hidden width 1024 and FFN 3072 are retained during the sweep. The
  14+14 initialization records CE about 12.585 versus 5.546 for B28 on 24 open
  calibration rows. Width reduction is therefore not necessary for this damage.
- `architecture_benchmark_001/transfer.py`: hard replacement explicitly is
  non-function-preserving. Widening FFNs with zero down columns is not the same
  intervention as replacing layer-local K/V with one shared bank.
- `architecture_benchmark_001/evidence/trainability/RECOVERY_PILOT_C28_lr1e-05_s40.json`:
  40 steps / 25 rows / 4,968 target tokens improve likelihood, but the tiny task
  pilot remains unsuccessful. These are exploratory observations, not an adequate
  universal test of recovery or a pure-CED impossibility result.
- `rab001_exploration/gated_memory.py`: additive zero-output prototype, with real
  source verification documented as unfinished. Its unsupported implication that
  training can only improve an identity-initialized model is not adopted here.

A later result in the same parent commit is especially important:
`architecture_benchmark_001/evidence/trainability/RECOVERABILITY_FRONTIER.md`
reports a three-seed, 40-step comparison: pooled hits are 73/180 for ordinary
continuation, 69/180 for `ced_last4`, 36/180 for `ced_last8`, and 0/180 for C28.
This supersedes the blanket statement that shared-memory conversions cannot
recover. These are the same 60 task prompts repeated across three training seeds,
not 180 independent test questions. Overlapping marginal confidence intervals
neither prove equivalence nor establish a registered noninferiority margin. We use
the pattern as exploratory motivation for gradual transitions, not as proof of
complete recovery, a universal displacement law, or an exact capability cliff.
The report also explicitly leaves larger-budget C28 recovery unsettled. An
underway local campaign is not a completed result and is not authorized by FWT.

The parent also records exploratory activation-aware per-layer low-rank KV work
in its commit message and `evidence/ARCHITECTURE_PATTERNS.md`. That is a separate
storage/compression direction, not evidence that the present augmentation saves
memory. It remains preserved and is intentionally not mixed into FWT's first
architecture comparison.

Historical scoring and narrative revisions warrant caution. We do not promote
narrow open probes, zero-training sharing sweeps, or 40-step runs into laws about
all architectures. No universal "nats per KV bank" law or impossibility claim is
part of this study. Old receipts grant no permissions here and are not rewritten.

## 3. First experiment: preserve the pretrained path

M28 retains all original blocks and adds a learned shared-memory pathway with
zero-initialized output projections. This differs from immediately replacing the
upper attention modules. At initialization it should compute the source function;
that claim must pass on real source weights, not just by reading the algebra.

The shared memory is formed after source block 13. Each added read follows a
complete original upper block, including its FFN. This placement is fixed for v1;
pre-attention placement, moving the split, and alternative donors are different
implementations and require a new registered version, not tuning on test outcomes.

A passing identity control establishes preservation, not usefulness. At step 0,
Q/K/V paths upstream of the zero output have zero gradients. The output projection
must receive a gradient first; later steps must reach the upstream projections.
The engine tests this explicitly. A pure teacher-matching objective cannot wake an
exact teacher clone, so the bounded smoke includes next-token CE as well as KL.

## 4. Controls and what their contrasts mean

B28 frozen is the provenance, identity and retention reference and the smoke's
teacher. M28 versus P28 is the first extra-capacity-controlled comparison. P28 has
exactly the same tensor shapes, parameter count, initialization, number of added
read modules, trainable policy, examples, order and optimization steps as M28. Its
added branch is tokenwise sigmoid-gated, not an additional cross-token read.

M>P would favor explicit additional cross-token access over this specific
same-token gated alternative under the tested recipe. It would not isolate every
other inductive-bias difference, establish compute-normalized superiority, or show
that parameter count is irrelevant. P itself still receives contextual encoder
states and an intact contextual backbone.

For a later efficacy campaign, include an ordinary continuation control with the
same data exposure and declared backbone trainability policy. Where trained-parameter
budgets differ, report them rather than calling the comparison fully matched.
Control tuning opportunity, not just the nominal learning rate. Keep the frozen
source reference alongside trained controls so improvements and forgetting are
both visible.

Do not call a dormant path a success. Evaluate disabled-read diagnostics and
appropriate intervention controls after training, and require actual held-out task
benefit rather than merely nonzero gates or gradients. Full attribution can require
additional controls; the first M/P contrast is not a universal mechanistic proof.

## 5. What P1 and P2 settle

P1 tests implementation mechanics with the exact locked official source. It checks
native/reference parity before constructing the new arms. Same-path initialization
and reload require bitwise identity; different reduction shapes in full, tail and
chunked paths use the fixed elementwise `atol=rtol=1e-4` rule. Equal argmax alone
cannot substitute. Failures remain failures; no alternate precision or relaxed
threshold is silently selected.

P1 uses temporary nonzero output projections at scale 0.001 to expose the actual
read path, rather than only testing zeros. It checks padding, repeated positions,
packed segment isolation, large positions, future-prefix independence and cache
orders. This is a mechanical perturbation, not a trained result.

P2 is exactly 25 engineering steps for each of M/P, with frozen original backbones,
matched shuffled example order, gradient checkpointing, AdamW at 1e-5, no weight
decay, clipping at 1, and CE plus temperature-2 teacher KL. It is deliberately small
and uses eight new original paragraphs, not the historical calibration/pilot bank.
Its learning rate is an engineering default, not an optimum inferred for the new
parameterization from old full-finetuning results.

P2 reports finite optimization, awakened outputs, upstream gradients for every
added read and shared K/V, unchanged original weights and post-update mechanics.
It does not score broad ability or certify capability retention. The tiny fixtures
are not a recovery dataset. No smoke checkpoint is exported as a released model.

## 6. Prerequisites for a meaningful efficacy campaign (P3 remains closed)

Before any registered recovery/DEV campaign, fix and review all of the following:

1. Family-disjoint training, development and sealed manifests, leakage checks,
   provenance, and a real human-review record rather than a self-declared flag.
2. A task endpoint for closed-context reasoning/retrieval and distinct retention
   endpoints for abilities the frozen source actually demonstrates. Validate
   parsers against raw outputs; do not let punctuation or leading-integer matching
   manufacture capability differences. Loss is secondary, not a substitute.
3. Pre-outcome minimum worthwhile effects, per-family regression margins, a
   selection rule and a cluster-aware power calculation using independent task
   families. Repeated phrasings and training seeds are not independent test items.
   Do not accept a nonsignificant result as evidence of equivalence.
4. At least three genuine training seeds, consumed by data order and any stochastic
   training operations. Match initialization where intended. Record all failed or
   interrupted runs; do not present identical deterministic repeats as replication.
5. Equal exposure and declared tuning/compute budgets for M, P and ordinary
   continuation. Register the permissible LR/regularization search before results.
   Diagnose learning curves and retained capability before committing a larger run.
6. A teacher-distillation and replay policy, including full input-token loss versus
   response-only loss where relevant. Teacher agreement protects teacher behavior,
   not truth or capability beyond the teacher. Replaying broad text is different
   from applying KL only to narrow new-task examples.
7. Retention checks and stop/rollback rules during training. Improvements in CE, KL,
   fluency or termination cannot authorize trading away task capability.
8. A reviewed campaign runner and measured memory/resource preflight for the
   selected runtime. The CPU qualification does not authorize an untested CUDA,
   BF16, quantized, optimizer or checkpointing substitution.

These are unresolved registrations, not secretly selected thresholds. The current
code has no P3 command and no path that converts a P2 pass into campaign permission.
A null from an underpowered or undertrained setting is inconclusive; a bounded
engineering failure calls for diagnosis, not an architecture-wide verdict.

## 7. Progression to depth and then a replacement endpoint

Once the full-width transition is useful, investigate D40 at 1024/3072 and the
original 16/8 heads. Initialization must preserve every retained source layer and
use a verified identity expansion. Identity alone does not show added depth is
useful: train and compare against a continued B28 under declared budgets.

For C40, use the same qualified full-width D40 donor for both ordinary continuation
and the architectural variant, with matched donor-training history. Do not build
a new "full-width C40" from the damaged 896-wide D40 checkpoint or pad pruned
matrices back to 1024: re-inherit from the original source lineage.

A hybrid improvement is not evidence that original upper self-attention can be
removed. Pure-CED transition needs a separately preregistered intervention or
curriculum, a retained-attention control, capability retention during removal,
causal/cache requalification and endpoint evaluation with that dependency actually
gone. Progressive interpolation, a retained local attention path, donor matching,
or other transition mechanisms are candidates, not implemented successes. The
current additive prototype has no hidden pruning switch or pure-CED export.

Analytic full-width budgets under the preserved reference, not measured models:
C28 at 14+14 / 1024 / 3072: 568,786,304 parameters;
D40 ordinary / 1024 / 3072: 784,821,248;
C40 at 20+20 / 1024 / 3072: 744,973,952.
M28/P28 have 656,884,608 because they keep the full original stack and add reads.
The first-phase hybrids retain 29 KV banks versus the source's 28, so no memory or
latency savings may be claimed from the pure-CED budget numbers.

## 8. Compression comes last

Only after an interpretable full-width result, reopen residual-width, FFN-width
and head pruning as separate factors. Consider reducing one axis at a time before
combining them. Test compression together with a fair recovery budget and the
successful uncompressed model as teacher/control. A combined deep-thin success
must show retained capability at the intended resource budget, not just a smaller
checkpoint or lower next-token loss.

This is a dense, causal shared-memory research design inspired by the supplied
idea. It is not an implementation or reproduction claim for a complete named
DeepSeek release. No external release-specific assumption is needed by the code.
