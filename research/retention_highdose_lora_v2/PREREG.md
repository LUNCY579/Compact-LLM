# DOSELORA-001: high-dose low-rank training against an exact norm control

Prospective exploratory experiment, registered before this trajectory or its
evaluation outputs exist. It follows DOSEINT-001's failed joint prediction and
the user's priority3: low-invasiveness training at higher dose. No release,
independent confirmation, or general acquire-without-forgetting claim follows.

## Frozen treatment

Use the exact2697 dual-bank-disjoint training rows from DOSEINT-001, byte-identical
IDs/labels, order, one exposure, seed17,193 groups,380921 processed and37090
supervised tokens. Both384-parent acquisition banks remain excluded. This is the
maximum balanced dual-disjoint dose, not an exact historical3072-row D4 repeat.

Load the same596049920-parameter Supa source in FP32, eager attention, TF32 off,
using the pinned native runtime. No quantization, BF16, CPU model offload, package
installation, or tokenizer change. Inject rank16/alpha32/dropout0/bias-none LoRA
into q/k/v/o/gate/up/down in blocks8–27 only. Train the SAME four norm tensors per
tail block as the native control. There are140 projections,280 adapter tensors,
80 norm tensors,7255040 trainable parameters. The temporary unmerged training
model has603258880 parameters; the dense merged deliverable remains596049920.

Reuse the native shifted response-plus-EOS masking, group supervised-token
denominator, chunked vocabulary loss, CpuAdamW(0.9,0.95), epsilon1e-8, clip1,
zero weight decay, and token-based WSD20/60/20 with its native square-root decay
to0.1 peak. **Peak LR is1e-4 versus the native control's1e-5.** This is a declared
practical LoRA regime, not a parameterization-only causal experiment. No learning-
rate search or second training arm is added after outcomes. Omit the native
in-run memorization generations; they do not update parameters/RNG and are not
held-out endpoints. Qualification and its resource charges remain mandatory.

## Build gates before any optimizer step or endpoint

- Verify source assets, clean pinned pilot checkout, frozen executable hashes,
  exact prepared bytes, token/order/group counts, and both parent exclusions.
- Independently qualify the real unmodified source with the existing native
  zero-update checks, including the2048-token gradient shape.
- At zero B, LoRA's uncached and cached logits must reproduce the original source
  exactly; use the unchanged native tolerance for cache versus full comparisons.
  All360 trainable gradients must be finite; zero A gradients are expected at
  zero B, while B and norms must participate on the short native probe. Frozen
  source matrices/embeddings/out-of-scope norms must have no gradients or changes.
- After all193 updates, save a full, cleared-boundary adapter/norm/optimizer/RNG
  checkpoint. Before accepting success, independently compare the unmerged
  native model to its CPU-FP32 dense merge at source Reply7, the first frozen
  TRAIN row, and a2048-token mechanical fixture. Check the last16 available logit
  positions with atol=rtol=2e-4, the existing native numerical tolerance, and exact
  argmax agreement. Record all220 CPU-merged tensor byte hashes. No widening after
  a failure. A completed checkpoint alone cannot authorize export or evaluation.
- Require the exporter to reproduce every qualified dense tensor byte hash and
  preserve all frozen source tensors and six sidecars. One model worker or large
  materializer at a time under rineya-gpu-0.lock; unchanged resource guards.

## Exact magnitude comparator

Merge LoRA in CPU FP32. Across ALL220 dense tail tensors, including the80 trained
norms, compute FP64 Frobenius sums of the effective FP32 updates from the SAME
source. Let beta=norm(delta_LoRA)/norm(delta_native), where native is the retained
DOSEINT-001 checkpoint-00193. This is not an adapter-factor norm. Freeze beta and
its complete hash receipt before inference. Build source+beta*delta_native in
FP32; direct-copy endpoints when beta is0 or1. Do not clamp beta or search it.
The registered admissible interval is[0,2]; outside it, or with a zero native
norm, record BLOCKED_ON_NORM_COMPARABILITY and score neither new arm. beta>1 is
explicit extrapolation. The actual rounded scalar-update norm must match LoRA
within0.5% relative error, otherwise the same build gate blocks both arms.

Thus ONE training run yields two evaluated artifacts: LoRA full merge and the
norm-matched scalar from the existing native checkpoint. Also report comparisons
to every existing DOSEINT-001 scalar arm on each identical bank. No old unbound
receipt or transported bank offset is used.

## Registered predictions and interpretation

For the LoRA arm, predict acquisition260–325/384 on EACH bank, retained37–45/47,
and arithmetic55–80/150. Primary engineering prediction: acquisition>=284 on EACH
bank, retained>=40, and arithmetic>=62 simultaneously. Also report ratio>=16/50
and termination150/150. Mark failed bands and conjunctions FAILED_PREDICTION.

The stronger beyond-magnitude prediction is: LoRA beats its exact norm control
by >=20 acquisition views on EACH bank with exact paired p<=0.05, while meeting
the primary retention/arithmetic targets and having no lower observed retention
count than the norm control. This is a finite engineering rule, not proof of
retention equivalence/noninferiority. Failure does not establish a capacity limit.
Even a numerical primary pass cannot be called a resolved mechanism advantage
without clearing the registered magnitude comparison.

Every acquisition contrast reports (net,nd,exact two-sided p), the pair's exact
parity-aware significance threshold, and the independent cross-training floor
|net|>=20. All LoRA-versus-native comparisons are CROSS-TRAINING-DRAW comparisons.
Comparisons of the norm scalar to other native alpha arms are within one draw.
Retained-answer contrasts use the same47 frozen permissive-source IDs and their
observed discordance threshold (best-case6, not a blanket threshold at every nd).
Report strict scoring separately. Publish nominal values and Holm-adjusted values
for the two primary acquisition-bank contrasts; descriptive scalar-ladder
comparisons remain a separate declared exploratory family. No convenient family
choice substitutes for the20-view effect requirement.

Long-context256 and literal UNKNOWN on64 unanswerable rows remain SPENT/report-
only. Predict LC145–170 and literal UNKNOWN0–16 for LoRA, without selecting on
them or claiming this teaches missing binding signal. Report both new arms on
all four probes, regardless of favorable/unfavorable endpoint patterns.

Before reporting ANY comparison require matching bank hashes, implementation
digest, probe/wrapper hashes and runtime/numerics, complete paired row IDs, and
contemporaneous before/after weight custody agreeing with each arm receipt.
Failures produce BLOCKED_ON_COMPARABILITY with no endpoint values for that pair.

## New numeric execution envelope

Authority is the user's explicit priority3 and subsequent delegation to perform
the best recommended steps. These are agent-instantiated prospective limits;
human data review remains NOT_PERFORMED. No consumed prior grant is reused.

- One supervised training launch, zero optimizer retries;2400s watchdog,
  <=2500 native worker seconds and2700 owned seconds; <=400000 charged tokens
  INCLUDING native, adapter and final-merge qualification; exactly380921 actual
  optimizer-data tokens. <=128 qualification calls, zero generated tokens in the
  trainer, zero cloud/teacher spending or RL tokens.256MiB checkpoint/log budget.
- Two CPU materializations, <=600s each, one at a time; total new experiment
  logical disk <=7GiB including failures. Preserve the14GiB physical free-space
  floor and require19.5GiB physical free before dispatching training.
- Eight planned probe legs (two arms ×two acquisition banks/retention/LC),900s
  watchdog each; at most ONE total inference retry, fresh label/directory after
  verified failure. <=8100 total probe-worker seconds, <=3000 reserved rows and
  <=270000 reserved generated tokens including a retry. Fixed no-retry totals:
  2348 rows and216096 generation-token reservations.
- Entire new session <=12600 wall seconds including preparation, qualification,
  failed attempts, storage compression, training, exports, probes and analysis.
  Require >=7GiB host RAM before dispatch; retain the unchanged child6GiB training
  gate and standard probe/materialization gates. Never delete another lease or
  manipulate other processes to pass a resource gate.

Storage is reclaimed only by transparent lossless filesystem compression of
completed files owned by DOSEINT-001, with each existing receipt hash verified
before and after. File bytes, paths, checkpoints and the records repo's * -text
attribute are preserved. Compression is not model quantization.
