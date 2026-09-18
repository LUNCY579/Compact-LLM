# RNGF-001 — causal RoLA mechanism × Never Give Up

Runnable CPU mechanics and logical-async training fixtures. **Not pretrained
Rineya results, not an async throughput benchmark, not release qualification.**
This integration extends the original mechanics package in this directory. Historical branches, weights,
gates and spent instruments are not modified.

## Run from the repository root

```bash
python -m pytest -q experiments/rola_ngu_factorial_v1/tests
python -m experiments.rola_ngu_factorial_v1.runner --arm D --draw 101 --tokens 512
python -m experiments.rola_ngu_factorial_v1.smoke
python -m experiments.rola_ngu_factorial_v1.preflight
python -m experiments.rola_ngu_factorial_v1.qualification
```

The last two commands intentionally exit 2 while native prerequisites are absent.
Install the approved CPU packages in `requirements-cpu.txt`; the tests do not need
Transformers. `requirements-native.txt` declares the separate optional stack.
The optional `qualification --native-random` constructs a tiny RANDOM stock Qwen3
only; a missing runtime returns BLOCKED, never a skipped qualification counted PASS.
`qualification --source /absolute/path/to/exact/Supa` only hashes source assets.
No command downloads a checkpoint or invokes pretrained training.

## Current evidence changes the premise

**Tiny delta is not free learning or no forgetting.** Titan used 384 training rows
and 28 optimizer updates. Titan-Sparse construction reuses that update with no new
training, but its source-correct arithmetic retention is 41/47, not 47/47.
REP-001 found paired long-context repairs of +7, +0 and +15 across draws17/23/29.
This motivates testing pruning, not assuming universal benefit.

More importantly, the updated Codex family record at `07e53b0...` records RCI-001
negative confirmation: **40/128 vs 25/128** literal UNKNOWN, **0 gains/15 losses**,
paired p=6.1e-05. Both models had zero false abstention on answerable rows.
Runtime/floor caveats and the correction of conflicting mechanism explanations
are preserved in `SOURCE_REVIEW.md`. RCI-001 is now spent. Historical Demo/UNKNOWN
scores remain report-only. The older blanket statement "no confirmation exists"
is obsolete; no *fresh eligible* confirmation instrument is bound to RNGF-001.

## Primary factorial, unchanged from the agreed design

| Cell | Representation | Sampler | Primary delta keep |
|---|---|---|---|
| A | original dense attention | ordinary async GRPO mechanics | 1.0 |
| B | original dense attention | bounded NGU mechanics | 1.0 |
| C | dense + causal RoLA residual | ordinary async GRPO mechanics | 1.0 |
| D | dense + causal RoLA residual | bounded NGU mechanics | 1.0 |

Every native cell starts from the same exact ungated Supa1.7 in `source_lock.json`,
not Titan, Titan-Sparse, CED, Supa-A1.1 or a biased abstention head. The code parent
is `fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`; updated sibling evidence is cited,
not merged. A concurrent session published the original mechanics at `3a83618...` while this
extension was being validated. This extension descends from that commit without
resetting the branch. See `SOURCE_REVIEW.md` for both remote snapshots.

Use three independent 384-parent TRAIN draws, seeds101/211/307, paired across
A/B/C/D. These are proposed native draws, not already executed replications.
Publish overlap; permutation of one dataset does not create independent draws.
Initialization43, ordering41 and sampling47 have distinct roles. C/D share their
initial added parameters. Keep an untrained source baseline. Native optimizer,
corpus hashes, resource grant, meaningful effect sizes and margins remain unset.

The primary interaction is `(D-C)-(B-A)` on UNPRUNED checkpoints. Main effects
are `((C-A)+(D-B))/2` and `((B-A)+(D-C))/2`. `analysis.py` refuses incomplete
cells, changing row sets/denominators and selectively missing sparse derivatives.
It reports by-draw/family effects and descriptive spread, not pooled row×seed
p-values. `paired_retention` conditions losses on source-correct rows;
`abstention_endpoints` separates literal UNKNOWN from false abstention. No
nonsignificance-to-equivalence conversion or automatic promotion is implemented.

## RoLA adaptation and actor integration

Paper: https://arxiv.org/html/2609.06712v2, equations2–4. The original is a
noncausal sparse/global video-DiT architecture. This language-model adaptation:

```
q_tilde[t] = R_native_truncated(t) SiLU(Pq q_normalized_pre_rope[t])
k_tilde[t] = R_native_truncated(t) SiLU(Pk k_normalized_pre_rope[t])
S[t] = S[t-1] + k_tilde[t] outer v[t]
read[t] = RMSNormalize(q_tilde[t]^T S[t])
out[t] = dense[t] + alpha * RMS(dense[t]) * read[t]
```

Only inclusive causal history is written. RoPE is AFTER the nonlinear projection;
reuse complete native frequency pairs rather than recomputing a rank-based
schedule. GQA expansion is explicit. Native rank16, sites8–27, and scalar alpha=0
are registered engineering choices, not demonstrated optima. Alpha has a first
step gradient; projection gradients become active after alpha moves. This differs
from the paper's tokenwise sigmoid gate. No token-attention sparsity or speedup
claim is made. Adding this branch while retaining dense attention adds work.

`actor.py` is a reference Qwen3-shaped traversal preserving Q/K normalization,
residuals, MLPs, output projections and head dimensions (Q width need not equal
hidden width). Generation and differentiated completion logprobs traverse the
same active branch. It rejects remote-code/PEFT/dispatched/hooked/hybrid models,
scaled RoPE and ambiguous dtypes. The first integration is one unpadded sequence;
batching/padding is qualified at operator level only, not the full actor.

Functional caches bind to actor, sequence, model version and tensor mutation
versions. Cached generation is eval/no_grad-only; stale or cross-sequence reuse
fails. Independent dense fixtures, active-path gradient tests, future-token
perturbations and full/chunk/token decoding tests check more than zero-gate parity.
These fixtures do not substitute for exact-source native qualification.

## NGU semantics, not just "try harder"

Paper: https://arxiv.org/html/2609.13443v1, section4 and appendixE. Both sampler
arms share the event-driven async interface and K=4. Ordinary GRPO drops a
no-signal episode; NGU requeues all-failure episodes with probability.75, at most
8 rounds and 32 pending episodes. Easy episodes are not permanently removed.
All attempts count toward resource cost. No answer or verifier metadata enters
the policy input, and no failed-attempt hints are appended.

Keep whole-episode counts for the reward baseline, including stale failures.
Only samples with learner-version lag<4 contribute loss. Positive advantage is
`1-baseline`; fresh negatives are rescaled to sum to the opposite positive total.
No fresh negatives means a logged no-signal event, not invented contrast.
`objective.py` uses original behavior logprobs, centered advantages, sequence-mean
clipped policy loss and reference KL, identically in both arms. All RNG/state,
versions and actual generated token records remain available.

The **CPU runner is a logical-async simulator**, not a concurrent production
trainer: two immutable sampled groups can be in flight while the learner advances,
so real off-policy probabilities and stale handling are exercised. Execution is
serial; its wall time proves nothing about the paper's asynchronous benefit.
A purely synchronous retry implementation must not be substituted and called NGU.

The fixture has 3 random layers, width 24, 4 query heads, 2 KV heads, head dimension 8, and RoLA sites 1/2
rank4, one-token arithmetic targets and a separate toy AdamW recipe. Its learning
curves are not Rineya capability evidence. `runner` accepts no pretrained inputs.

## Resume and resource accounting

`FixtureRunner.save/load` preserves whole actor/reference weights, optimizer,
RNG, raw rollouts, in-flight groups, pending retries, counters and budgets, bound
to config, trainable inventory and the source manifest. It uses weights-only
loading and a file SHA256 sidecar, never unpickles arbitrary custom objects.
Caches are deliberately not persisted. Resume is exact only on the pinned tested
CPU environment; it is not a native-scale durable-worker guarantee.

Reserve before dispatch, settle actual prompt-plus-generated tokens, charge
unknown failures fully, stop on nonfinite math and retain failure records. Native
compute comparisons must also report prefill/decode/loss tokens, updates, wall
time, memory and FLOPs. A matched token cap alone is not matched compute. The toy
one-token fixture exhausts its declared cap exactly; native residual budget rules
remain to be registered. Save outputs outside the checkout; do not auto-clean a
research worktree to bypass the clean-source gate.

## Titan-style postprocessing stays a separate factor

After all 12 native unpruned checkpoints are frozen, create each paired variant:
`theta_source + M_0.25(theta_trained - theta_source)` in FP32. Use the same 220
pretrained tail-tensor allowlist. The mask keeps floor(numel*.25) largest absolute
deltas per tensor with ascending-flat-index tie breaking. Do not replace pruning
with scaling, rank truncation or BF16 rounding.

New RoLA parameters have no source counterpart: retain their final values
unchanged and identify them in the receipt. This tests shared-backbone delta
pruning CONDITIONAL on each arm, not sparsification of the added module.
Twelve derivatives add zero training runs and do not become 24 independent draws.
`cast_audit` measures paired-source rounding separately; tiny deltas can disappear
under BF16. Sparse delta storage is not sparse dense-model execution.

## Next gates

CPU fixtures first. Then install/verify the separately pinned native runtime;
run `qualification --native-random`. Next, under a dedicated inference-only
qualification contract, bind all seven actual source assets and check unmodified
loader parity, zero/active actor paths, actual-source trainable inventory, cache,
completion logprobs, tokenizer/EOS and resource fit. The exact-source run is not
implemented as an authorized command yet. Production async worker integration,
text/parent/template audits, fresh endpoint instruments, margins/power and an
explicit native training budget must pass before the 12-run experiment opens.

A code/test PASS never authorizes sealed evaluation, efficiency, quality or release.

## Published local verification

`evidence/integration_20260918/VALIDATION.json` binds the executed checks to `manifest.json`;
`evidence/integration_20260918/pytest.txt` is the full scoped test summary. `evidence/integration_20260918/smoke.json`
records twelve random fixture runs and twelve deterministic paired transforms.
Its hashes identify generated rollout/journal records reproducible by `smoke.py`;
no toy accuracy is promoted as an estimate of native efficacy.

The original `evidence/` receipts remain byte-identical historical records for the
97-test package at `3a83618...`; its REMOTE_STATUS describes an earlier failed
attempt. Current integration receipts live only in `integration_20260918/`.
