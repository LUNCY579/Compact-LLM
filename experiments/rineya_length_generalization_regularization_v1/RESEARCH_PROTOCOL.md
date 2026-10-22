# LGR-001 research protocol — proposal before native evidence

## Motivation and competing explanations

`BROAD_RETENTION.md` at the pinned parent records Supa 161/256, transition-only
150/256 and rehearsal 139/256 on a report-only long-context check. Literal UNKNOWN
on 64 absent-answer rows was 13, 3 and 1. These are historical repository reports,
not rerun results or admissible model-selection data. Formatting, abstention,
replay composition, update dose and length-distribution changes remain competing
explanations. VPAD is a hypothesis about an additional training-side factor.

## Mechanism and exact intervention

The ordinary baseline is `Linear(Dropout(RMSNorm(x)))`, not attention-probability
or adapter-only dropout. The full base-plus-adapter projection sees the perturbed
activation. Q/K/V share the attention-entry draw; gate/up share the MLP-entry draw.
No final output-norm, query-norm, key-norm, attention-probability, per-projection,
RoPE, sliding-window or cache modification is introduced.

For keep probability q, raw feature mean mu, second moment gamma, and normalized
x_hat = x / sqrt(gamma + eps), the upstream numerical policy is:

    variance = max(gamma - mu**2, eps)
    alpha = sqrt(variance / (q * (gamma - q * mu**2) + eps))
    beta = mu / sqrt(gamma + eps) * (1 - q * alpha)
    output = beta + alpha * mask * x_hat

Mask entries are unrescaled Bernoulli(q). The default detaches beta but not alpha.
FP32 moments and epsilon guards match the inspected implementation. The module
bypasses the formula at p=0 and during evaluation by calling its original forward.
Learned RMSNorm weights multiply the result afterward, in the Qwen casting order.

Moment matching concerns the modelled feature/mask moments. It is not preservation
of each feature's conditional expectation, the complete covariance matrix, every
learned projection, or an entire hidden-state distribution. Epsilon guards modify
the ideal equations near degeneracy. Measure actual downstream behavior.

`pre_norm_control` masks only the branch input before the existing norm; it does
not alter the captured residual stream. It is an explicitly labelled control,
NOT a claim of reproducing every detail of the paper's residual implementation.

## Hypotheses and stages

H1: ordinary pre-linear dropout improves the acquisition–length-retention tradeoff.
H2: VPAD adds benefit beyond ordinary pre-linear dropout at the same probability.
H3: optimizer weight decay interacts with length retention independently of H1/H2.
H4: any protection survives realistic downstream tasks, not only isolated needles.

Stage 0 qualifies graph identity, explicit placement, finite backward, checkpoint
RNG semantics, adapter roundtrip and nonzero-adapter merge parity. The native CLI
performs these mechanical checks but does not authenticate a Rineya export or
replace the existing loader, dataset, CHECK/FINAL, memory or operator grants.

Stage 1 uses A_none/B_pre_linear/C_vpad, probabilities 0/0.1/0.1. Default proposal:
rank 8 Q/V LoRA, alpha 16, no adapter dropout, frozen base, AdamW beta=(.9,.999),
LR 1e-4, WD=0, 5% warmup then linear decay, microbatch 1, accumulation 8, 1024 total
training tokens/sequence. Initial cap: 524,288 non-padding tokens AND 512 steps,
whichever is reached first. There is one pass through the selected bank, no hidden
repeat or epoch. Training can stop below the cap when data is exhausted.

A single seed (101) is a screen, not confirmation. All arms receive identical
selected rows/order/labels/token exposure and adapter initialization. Different
noise consumption is intrinsic to the intervention. The incumbent recipe, when
different, is an additional named comparator; never silently redefine it.

Stage 2 fixes the candidate on DEV and uses the generated 2x2 no-regularizer/
candidate x zero/predeclared-reference-WD factorial. WD=.1 is an example sensitivity
value, not an instruction to change an incumbent with WD=0. Learning-rate decay
is distinct from optimizer weight decay. Additional p=.05/.3 and the pre-norm
control are separate registered ablations, not post-hoc selection on FINAL.

Stage 3, only after a useful signal, crosses the intervention with explicitly
length-aware replay. Changing the training-length distribution invalidates any
claim that gains came from short-only training alone.

## Endpoints and evidence design

Use frozen 1K/2K/4K/8K prompt grids first; 16K requires a new resource qualification.
Reserve completion room in addition to prompt length. Keep semantic parents paired
across length and evidence position. Separately vary distractor count, evidence
distance and number of necessary reasoning steps; do not conflate these axes.

The shipped generator provides only synthetic recall, two-value recall, one-step
state update and absent-answer diagnostics. Real RULER multiple-needle/tracing/
aggregation and HELMET downstream subsets require their own pinned task versions,
licenses, scoring adapters, contamination checks and manifests. They are NOT
silently reimplemented or downloaded here. Import their verified tokenized cases
and registered answers into the explicit case schema after qualification.

Acquire eligible, independent acquisition and retention banks as well. Generated
retrieval diagnostics cannot stand in for these endpoints. For abstention report
both correct UNKNOWN on absent answers and unnecessary UNKNOWN on answerable cases.
Strict answer accuracy is primary; formatting or substring-containment diagnostics
must not replace it. Missing rows/truncation are errors, not implicit failures or
removed denominators. Report generation-cap hits and actual token lengths.

Keep training, development and unopened confirmation parents/prompt hashes disjoint.
A synthetic seed namespace alone is insufficient: run the split audit. External
semantic/normalized-text contamination requires the established repository audit;
the token-hash checker does not prove semantic independence or license eligibility.

## Statistics and proposed decisions

Report every seed separately (confirmation proposal: 101, 211, 307). Aggregate
cells by task, prompt length and evidence depth. Resample semantic parents with all
views together and resample training seeds; use the same parent draw across seeds.
Never count repeated views or repeated evaluation cases as independent experiments.
The CLI rejects unmatched rows, duplicate scores, changed prompt hashes and
incomplete seed/view grids. Bound native comparison reports must share a collection
contract (bank, checkpoint, code, runtime, completion budget and context limit).

Proposed review thresholds: long-context macro gain >=5 percentage points with
paired 95% lower bound >0; acquisition, retention and abstention lower bounds no
worse than -2 points. Three seeds, at least two long-context tasks including a
non-synthetic downstream task, and >=256 unique parents per endpoint are minimum
review screens. The sample-size rule is NOT a power calculation. Preregister
power from DEV discordance/clustering and margins before opening confirmation.
Bootstrap intervals can be degenerate with no observed disagreements; human
coverage/power review remains mandatory. No output is an automatic release.

Strong result: similar acquisition with better length robustness, preserved
short-context behavior and abstention. Partial result: ordinary pre-linear works
but VPAD does not add value; prefer the simpler method. Failure/inconclusive:
passkey-only gains, lower acquisition, blanket abstention, missing endpoints,
seed instability, unequal exposure, or improvements lost on the serving path.

## Later integration, not initial confounds

Retention-dose: compare trajectories and matched-acquisition checkpoints, with
selection performed only on DEV. MiMo/distillation: hold targets/data constant.
Composable skills: only later measure whether independently regularized deltas
compose more safely; moment matching does not imply orthogonal parameter updates.
Memory compression/random attention: retain full-cache controls. Recurrence/TAPS,
world models, RL and architectural changes remain separate experiments.
