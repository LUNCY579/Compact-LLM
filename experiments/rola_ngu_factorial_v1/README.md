# RNGF-001 — isolated RoLA-mechanism × Never Give Up experiment

This package implements and tests the causal operator, event-driven sampling
mechanics, shared policy objective, paired analysis, and a separately labeled
post-training delta transform. It is **not a pretrained-model experiment result**
and is **not a native training launcher**. From the repository root:

```bash
python -m pytest -q experiments/rola_ngu_factorial_v1/tests
python -m experiments.rola_ngu_factorial_v1.preflight
```

The preflight intentionally returns exit code 2 with the native blockers.

## Parent decision and provenance

On 2026-09-18 the requested branch already existed, pointing exactly to
`codex/dataset-retention-audit-v1@fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`.
It had no factorial package. Extend that isolated pointer, without resetting it,
merging the single-layer branch, or modifying any other branch. This parent
retains the ordinary dense source/training infrastructure and the latest Codex
measurement corrections. `main@a6ee824584a9e28881826ab92a11db48e124d90c` lacks
those corrections; `single-layer-fine-tuning@954a0c2637e691b493c5ac3d488653a8c1adf675`
is the evidence donor, not the implementation parent. CED, gated Supa-A1.1,
Opallo and Titan are not model initializations for this primary experiment.

Code ancestry does not imply adopting every inherited gate. The Codex tip
records that all three reviewed GATE-v2 designs failed adversarial review;
its panel>=62 term demands improvement, not merely preservation of source=47.
Do not transfer that gate to RNGF-001 or silently alter its historical numbers.
All new files are confined to this package. Existing code and evidence stay intact.

Live evidence inspected:
- `MODEL_QUALITY_STATUS.md` and `experiments/reasoning_posttraining_v1/{sources,native_runtime}.py`
  at the parent: exact ungated source, source inventory, native qualification and
  data limitations; these do not qualify this new implementation.
- `Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/{README.md,
  evidence/REP001_RESULTS.md,evidence/build_sparse.py}` at `954a0c2637e691b493c5ac3d488653a8c1adf675`.
- RoLA: https://arxiv.org/html/2609.06712v1 (6 September 2026), equations 2–4.
- NGU: https://arxiv.org/html/2609.13443v1 (11 September 2026), section 4 and appendix E.

## What REP-001 changes — and does not establish

Each pair shares one FP32 training result; only the mask changes:

| Training draw | Acquisition, full -> k25 | Source retained, full -> k25 | Long context, full -> k25 |
|---|---|---|---|
| 17 | 235 -> 232 /384 | 40 -> 41 /47 | 157 -> 164 /256 |
| 23 | 229 -> 228 /384 | 38 -> 39 /47 | 152 -> 152 /256 |
| 29 | 228 -> 230 /384 | 36 -> 40 /47 | 144 -> 159 /256 |

Thus the two preservation endpoints were non-worse in three observed draws,
not strictly improved in all three. Strict arithmetic decreased by one row at
17 and 23; permissive arithmetic decreased by five at 23. This is not universal
no-harm evidence. The worst draw had the largest repair, but the other two are
not monotonically ordered by damage: damage-dependent repair is a hypothesis,
not an established law from three points. Nonsignificant post-pruning differences
from Supa are not equivalence. The long-context/UNKNOWN suites are spent and
remain report-only. No original model evaluation was rerun for this package.

Consequences: require independent parent draws, report every paired result and
worst draw, preserve strict and permissive endpoints separately, freeze a
positive minimum meaningful effect before superiority testing, and forbid
best-draw selection. No expected +7 or guaranteed retention repair is assumed.
Keep masks deterministic and FP32; quantization is a different intervention.

## Primary factorial

| Cell | Representation | Sampler | Delta keep |
|---|---|---|---|
| A | unchanged dense attention | ordinary asynchronous GRPO | full |
| B | unchanged dense attention | asynchronous NGU | full |
| C | dense + causal RoLA-mechanism residual | ordinary asynchronous GRPO | full |
| D | dense + causal RoLA-mechanism residual | asynchronous NGU | full |

Each cell starts from exact ungated Supa 1.7, weight SHA256
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Use three independently selected 384-parent TRAIN draws, seeds 101/211/307,
paired across all four cells. These are new draws, not REP-001 replications.
Publish actual parent overlap; different order seeds are not new data draws.
Initialization, ordering and sampling seeds are separate. C and D start from
identical branch parameters. Preserve a frozen, no-training source baseline.

The primary interaction is `(D-C)-(B-A)` on the **unpruned** checkpoints.
Report the RoLA main effect `((C-A)+(D-B))/2` and NGU main effect
`((B-A)+(D-C))/2`, by draw and family. The provided analyzer refuses missing
cells, changing source denominators, mismatched evaluation rows and duplicates.
Its mean/SD/range are descriptive, not confidence intervals or confirmation.
Do not turn repeated evaluation rows across seeds into independent observations.

## Exact scope of the RoLA adaptation

The paper is a noncausal video-DiT sparse/global construction. Copying its full
sequence summary into a language model would leak future tokens. This port uses
an inclusive causal prefix sum, with padding-masked writes and reads and an
explicit sequence-tagged cache. It applies per-head low-rank projection, SiLU,
then RoPE, using complete pairs from the *native* schedule. It never reconstructs
a new frequency schedule using the smaller rank. Normalized pre-RoPE Q/K and
unrotated values are mandatory; GQA keys/values are expanded before projections.

For v1 the original dense branch stays unchanged: there is **no token-attention
sparsity**, no routing, no frozen-attention replacement and no speedup claim.
The added branch is normalized and scaled by the dense output RMS. A per-layer
alpha starts at exact zero for source parity; this differs from the paper's
tokenwise sigmoid gate with negative bias. Alpha gets an initial gradient;
projection gradients appear after it activates. Rank=16 and tail sites8–27
are registered engineering choices, not paper-derived optima. This is a test
of a causal RoLA-inspired mechanism, not a faithful reproduction of the full
video architecture or its efficiency result. A later sparse-attention study
needs a sparse-only control and a new registration; it cannot be smuggled in here.

No full-model adapter is implemented in this package. The native integration
must preserve Qwen normalization, rotary layout, dense output projection, tied
embeddings, tokenizer/template, positions, masks and original cache behavior.
The prefix module is CPU/eager FP32/FP64 mechanics, not a fused deployment kernel.
Cache state is functional (no in-place writes), bound to sequence/module/parameter
versions. Do not reuse it after optimizer updates or between independent prompts.

## Exact scope of NGU and the shared loss

Both sampler arms use asynchronous events from frozen policy versions, K=4.
Ordinary GRPO does not retry all-failure episodes. NGU continues an all-failure
episode with p=.75, with eight rounds maximum and 32 pending episodes maximum.
A completed easy episode is dropped, not permanently removed from future epochs.
Generation groups carry original behavior logprobs and immutable sample IDs.
Duplicate events, future versions and changed prompt identities fail closed.

On mixed outcomes, maintain the reward/count baseline from the whole episode,
including stale failures. Use only samples with learner-version lag **<4** for
the update. Retain positive advantage `1-baseline`; assign each retained negative
`-n_positive*(1-baseline)/n_negative`, making the retained sum zero. A positive
with no fresh negative is a logged no-signal event, not a fabricated gradient.
The exclusive age convention is explicit; it is not a claim that a different
inclusive convention in pseudocode has been empirically compared.

`objective.py` applies the same centered, clipped sequence-mean objective and
reference KL in both arms, using stored behavior logprobs. Native policy/version,
optimizer, token-mask and reference-logprob parity still need qualification.
There is no worker scheduler, model generator, durable training resume, or
end-to-end asynchronous trainer here. A synchronous retry pilot must be labeled
as an ablation, not NGU efficacy: the NGU paper reports this distinction matters.
Binary solver-verified rewards only; no executing generated code or partial-credit
coding rewards in this first screen.

## Keep sparsification separate

After every unpruned checkpoint is frozen, produce its paired k=.25 variant:
`theta_sparse = theta_source + mask * (theta_trained - theta_source)` in FP32.
Use the same explicit 220-pretrained-tensor tail allowlist in all cells. Rank
pruning, quantization, delta scaling and sparse optimizers are not included.
New RoLA parameters have no counterpart in the original source: keep their
final values unchanged, identify them in the receipt, and never silently prune
them. This isolates **shared-backbone update pruning conditional on each arm**.
Pruning newly added parameters would need a separately named experiment.

Twelve training runs therefore yield twelve unpruned/pruned comparisons, not
24 independent training runs. Primary factorial statistics never mix the two.
Apply the rule to every arm/draw, not only a selected winner. With all eight
views per draw, sparse-conditional contrasts can be reported as secondary;
pruning itself never counts as another training draw. The tensor helper performs
no model calls, preserves inputs and records masks/norms/nonzero support. It is
not a streaming 596M-checkpoint materializer or a compressed serving format.
FP32 source+delta arithmetic follows the historical builder; no blanket claim
of roundtrip identity for arbitrary extreme FP32 tensors is made.

## Budget and evaluation contracts

Match total actual prompt-plus-generated token exposure across arms, charging
all retries, filtered groups, capped prompts, failures and stale samples. Reserve
worst-case work before dispatch and settle unused reservations. Unknown failed
usage is fully charged. Tokens are not FLOPs: separately report prefill/decode,
loss tokens, updates, wall time, peak memory and estimated compute, including
new RoLA work. Use the same async hardware/concurrency constraints in both arms.
Do not give NGU unlimited attempts or treat equal updates as equal compute.
No numeric training grant or optimizer recipe has been fabricated.

Evaluate each endpoint separately: acquisition, easy/hard strata, strict and
permissive accuracy, source retained/lost/newly gained, long context, literal
UNKNOWN on missing evidence, false abstention on answerable rows, termination,
truncation and length. Freeze task/parent/family partitions and solver rewards;
audit exact text, digit-blinded stems and semantic/template overlap, not just IDs.
Freeze fresh selection and confirmation instruments, positive effect thresholds,
noninferiority margins, sample-size/power assumptions and family/draw-aware
analysis before training. Historical exposed panels cannot authorize superiority.
The new registration intentionally leaves those unresolved fields null and blocks
execution rather than pretending they are approved.

## Gates and first valid next step

M0: run the scoped tests and retain a source-bound no-skip receipt. This is
mechanics only; no Actions dispatch, pretrained download or training is required.
M1: implement a separate native integration on this branch; bind all seven source
assets, actual runtime, complete trainable inventory and a qualified unmodified
loader. Check exact-zero branch full-logit/greedy parity, causal perturbations,
GQA/position/mask/cache correctness, active-branch gradients, full-state resume,
real old-policy probabilities and bounded resource fit at planned shapes.
M2: freeze the new manifests, audits, solver/evaluator, margins, optimizer and
native budget grant. Qualify asynchronous replay/behavior-policy semantics and
preflight receipts; none may be replaced by Boolean claims in a JSON file.
M3: only after explicit authorization run the three-draw factorial. Then apply
paired pruning without retraining. No sealed evaluation or release permission
follows automatically from a mechanics or native-wiring pass.

Outputs belong outside the checkout. Preserve failed attempts and refuse a dirty
source; do not auto-clean or delete unrelated work to satisfy a gate. REP-001's
seed29 dirty-checkout refusal is a reason to separate outputs, not to hide failures.
