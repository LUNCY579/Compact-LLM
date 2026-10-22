# DSD-001 prospective native protocol

## Question, lineage, and protected boundaries

Can controlled changes in rollout source, teacher coupling and KL direction
improve acquisition at an acceptable retention cost for the exact 0.6B Rineya?
The authoritative source is arXiv:2609.39494v1, not a social-media summary.

Parent: `post-training-Mimo@03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Existing `MODEL_QUALITY_STATUS.md`, `CONTINUATION.md`, grants, quarantines and
historical thresholds remain unchanged. No historical result is relabeled as a
self-distillation result. This is complementary to CSS/independent coefficients,
sequential acquisition, retention dosing and MiMo; none is silently superseded.
Do not mix GProj or new attention precision changes into this first factorial.

## Independent interventions

Student `p(.|x,y_prefix)` and teacher `q(.|x,c,y_prefix)` share the same response
prefix. Only the teacher receives the reference `c`. Source is student or teacher;
loss is forward `KL(q||p)` or reverse `KL(p||q)`. Teacher and rollout are detached.
The objective averages tokens within each example, then examples. The batch-one
pilot is exactly that reduction for one example. EOS is included when generated;
length-capped responses are retained and marked rather than silently dropped.

EMA uses `teacher <- (1-a)*teacher + a*updated_student`. The initial teacher equals
the initial student. Frozen parameters/buffers are shared and hashed; only selected
FP32 parameters require independent teacher state. Teacher swaps are disallowed
during a live student forward/backward/optimizer scope, and restore on exceptions.
A fixed teacher bank must remain byte-identical at a=0. At a=1 it copies exactly.

The LLM design is `{student,teacher} x {0,.02,.05,.1,.25} x {forward,reverse}` plus
SFT from the same reference information. Full default plan: two separately trained
spatial tasks x three seeds x one LR x 21 methods = 126 cells. This manifest is
not authorization to execute 126 training runs. Rate1 is optional diagnostic-only.
A wider SFT LR grid must be frozen separately before cross-method superiority claims.

## Stages and evidence vocabulary

**S0: controlled equations and CPU mechanics.** Implement and test Appendix E,
not just a trainer stub. Preserve exact prefix weighting, initial-only context
calibration, fixed displacement, actual teacher EMA, and the separate deployment
versus target-path metrics. Report geometry, optimizer, seeds, rates and horizon.
The executed two-seed 144-cell slice is not the paper's 96-seed empirical cohort.
Controlled selection uses opposite seed-parity folds within a fixed axis/geometry
stratum. Never select a coupling rate on the same held-out values used to claim gain.

**S1: native no-update qualification.** Pin all checkpoint and tokenizer files,
reviewed JSONL, code digest, runtime, trainable names, task, seed and resource limits.
Reject partial model loads, remote code, automatic downloads and unknown trainables.
Check complete student/teacher prompt budgets without truncation. On an admitted
training row, exercise both rollout sources and KL directions plus SFT, finite
losses and gradients, teacher initialization, and no model/buffer mutation. Observe
math SDPA through profiler events rather than trusting a config string. This is a
scoped mechanics qualification, not full-sequence/worst-case fit or capability proof.

**S2: explicit matched native cells.** Require S1 identity and --execute-training.
Each cell reloads its exact source. Compare recorded initial parameter hashes.
Fix row order, source examples, initial weights, selected parameter set, optimizer,
LR schedule, precision, scoring and reference information. Newly sampled rollouts
are expected to differ by method and later training state. Raw temperature1/full
vocabulary sampling has no rejection sampling, reward-based filtering or top-k loss.

Use selected FP32 master parameters, AdamW betas(.9,.999), eps1e-8, weight_decay0,
clip1, and a cosine LR schedule. This memory-bounded recipe is a Rineya adaptation,
not the paper's full-parameter training configuration. Record pre-clipping norm,
actual update norm, teacher entropy, response lengths, EOS/length-limit stops,
processed input tokens, forward calls, timing and peak CUDA allocations. Equal
updates do not certify compute matching. Do not retune update size after viewing
retention to present a favorable comparison as prospectively matched.

At every failure, retain the partial step records and failure receipt. Stop on
nonfinite values, missing gradients, source mismatch, memory-floor violations,
unsupported loader state or frozen-parameter/buffer mutation. No automatic fallback,
checkpoint merge, output overwrite or cloud job is permitted.

**S3: behavioral interpretation and fresh confirmation.** Evaluate without
privileged context on separately admitted DEV. Acquisition equals average accuracy
gain at the last three evaluated steps; retention is the final macro-task accuracy
change. Also report original successes retained, damages and repairs; a zero
baseline-success denominator is undefined, not perfect retention.

Two seeds or repeated rows do not create broad evidence. Analyze paired seeds/tasks
without pooling repeated rows as independent trials. Normal intervals are descriptive;
small samples, multiple comparisons and task clustering require caution. Selecting
a model on DEV does not turn DEV into confirmation. A fresh, authorized protected
suite and prospectively defined retention margin are needed for promotion. The
narrow synthetic arithmetic cohort cannot establish general skill preservation.

## Expected outcomes, including negatives

Frozen teachers may preserve a reference but provide insufficient acquisition.
EMA may strengthen useful supervision or amplify drift; no universal rate is
presumed. Teacher rollouts may improve target-path matching without deployment
transfer. KL direction may affect different checkpoints differently. An improvement
in teacher entropy, loss, or a controlled similarity is not an LLM capability gain.

If no cell improves the measured tradeoff, retain the negative result and the
ordinary SFT baseline. If a cell helps acquisition while damaging important skills,
report both rather than redefining retention. Later composition/rehearsal/precision
interactions need separate factorials and cannot retroactively replace this one.
