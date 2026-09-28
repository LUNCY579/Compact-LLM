# RINEYA-DREAM-001 — frozen offline controller protocol, v1

## Scope and scientific claim

Question: can a controller chosen using development-history replay obtain better
valid-outcome attainment per revealed attempt on independent held-out histories?
This experiment concerns experimental allocation, not model-weight improvement.
No native evaluator, trainer, checkpoint loader, cloud client, generated-code
executor, or live deployment adapter is present. DREAM-002 remains unauthorized.
Synthetic fixtures establish software mechanics only. No historical efficacy is
claimed until source-admitted histories have actually been evaluated.

Primary reference: Tong Zheng et al., *Dream-RSI: Recursive Self-Improvement
through Evolving Worlds*, arXiv:2609.14858v1, Section 3 and Appendix B.2,
https://arxiv.org/html/2609.14858v1 (consulted 2026-09-26).
The authors' repository, https://github.com/zhengkid/Dream-RSI, still lists the
full codebase and reproduction scripts as being prepared at inspection.
This is an independently written, Dream-RSI-inspired Rineya adaptation, NOT an
exact reproduction, and NOT the paper's full recursive online system.

The retained abstraction is a fixed discovery history, parent-selection actions,
prefix observations, replay evaluation, and replacement of an exploration policy.
The adaptation uses a bounded parameterized policy family rather than an LLM
rewriting policy source. We make no claim that this search family has the power
of the paper's policy-development agent.

## 1. Historical admission before selection

A replay world is a closed, explicitly non-sealed investigation with a complete
qualified baseline and recorded attempted outcomes. Its topology consists of
ordered root branches followed by single-child chains. Non-root branching,
cycles, missing parents, duplicate IDs/orders, and unexecuted planned arms are
rejected rather than flattened or assigned synthetic outcomes.

Creation order must come from proposal/attempt records, not a result-ranked
report or Git commit order. `context` lists every within-world observation used
to generate the attempt. Only the inherited ancestor prefix is supported in v1.
A proposal that used results from a skipped sibling, another unmodeled branch,
or additional evolving global history cannot be treated as independent of them:
reject it, or reconstruct a richer environment in a separately reviewed protocol.
A Git branch is not automatically a scientific parent-child trajectory.

Historical worlds require an identified curator/review, review of context and
complete failure logging, pinned original receipt locators (repository path,
commit, Git blob and SHA256), and exact JSON-pointer bindings for every baseline
field, checkpoint/protocol/dataset identity, research/task/lineage families,
node identity, parent, creation order, context, and outcome field. Local
receipt bytes and every mapped value are checked before replay. Normalization,
field transformations, legacy Markdown parsing and unit inference are NOT guessed.
An adapter needing these must be independently tested and retain original units.

These checks establish byte identity and consistent extraction, NOT the truth of
records, correct declaration of context, or remote commit membership. The curator
must inspect the pinned origin and original preregistration. Hashes or a checked
box alone are not causal authentication. The included audit admits ZERO real
worlds; its cited summaries are candidate sources, not replay-ready trajectories.

Execution status is distinct from usefulness: a measured but invalid/weak result
is not an implementation exception. Failed and blocked attempts have null quality
metrics, never fabricated zeros. `not_run` is representable in the outcome schema
for audits but forbidden as a replay node. Missing costs stay null. Negative and
failed investigations must not be filtered out of history merely for their result.

## 2. Splits and exposure

The catalog contains metadata and canonical world commitments, not outcomes.
Checkpoint, protocol and dataset SHA256 values plus study/task/lineage families
are mandatory world identities. Model, data, study, task and lineage keys derived
from these identities must be declared; extra contextual keys may be added.
Declare leakage keys for the research program, checkpoint lineage, task family,
dataset/prompt ancestry, and any shared selection history that could couple worlds.
Different IDs or checkpoint hashes alone do not demonstrate independence.
Any shared key across train/validation/test is rejected. Within a split, connected
components over all keys determine statistical clusters. The preregistered minimum
is three training, two validation, and five test components. These minima are an
operational floor, not a power calculation or proof of independent samples.

`fit` opens only train and validation worlds. It generates all policy candidates
using training feedback, then makes one validation selection pass. The test
catalog commitments are known, but test world files are never opened by `fit`.
Freeze binds the engine and this protocol, catalog, complete plan, policy, candidate
count, seed schedule, and the three baseline policies before any heldout loading.
`evaluate` verifies all bindings and writes an exclusive holdout-open receipt
BEFORE opening test outcomes. A failed test evaluation also consumes this receipt.
Reusing that run directory is rejected. A newly exposed holdout must never be
re-labeled unexposed for another revision. Directory guards do not prevent an
owner from deleting or copying files; this requires process discipline as well.

No token/prompt sealed benchmark, active registered run, or original receipt is
modified. Historical source receipts and source branches remain read-only.

## 3. Observation/action contract and unsupported actions

A policy receives only an immutable `View`: revealed observations, opaque
reveal-order handles, currently legal parent handles, remaining public attempt
budget, worker cap, completed rounds, and observed stagnation. It receives no
world name, source path/hash, hidden score, future branch length, or total node
count. Policy inputs and outputs are copied immutable data, not the environment.

The root opens the earliest recorded still-unrevealed root child. A revealed leaf
continues its sole recorded child. A batch must contain distinct legal parents,
fit both worker and remaining-attempt limits, and be legal before any revelation.
It cannot contain an unknown child or a non-leaf ancestor. Outcomes appear only
at the end of the whole synchronous batch; there is no same-batch adaptation.
A nonexistent continuation is an explicit support miss, with no quality outcome.
That endpoint is retired only AFTER its first attempted probe. No future support
availability is leaked to prevent a wasted probe. Termination uses the public
budget, policy stop, or observed support exhaustion, not hidden knowledge that the
best score or all recorded outcomes have already been seen.

Rineya's primary attempt budget charges ALL requested probes, including support
misses. This conservative change avoids rewarding unsupported requests as free
information. The paper-style diagnostic instead uses the number N of revealed
non-root attempts. Report both N and requested probes, never conflate them.
Unsupported requests cannot be extrapolated to new discoveries or native cost.

The implementation uses trusted built-in policies plus a strictly allowlisted
JSON parameter specification. It does not execute LLM-generated code. Frozen
Python value objects are an API boundary, not an OS sandbox for malicious Python
with filesystem access. External policy agents need a separate isolated runner.

## 4. Gates, objectives, and costs

The v1 frontier has two fixed, higher-is-better coordinates in [0,1]: capability
and retention rates under the world's fixed evaluation protocol. The reference
point is (0,0). No endpoint, normalizer, target, or ceiling is fitted to heldout
maxima. Baseline and valid outcomes must pass qualification and protocol gates;
retention must reach the frozen absolute floor. Also require recorded parameter
delta, latency ratio, host RAM and VRAM to meet the registered limits.
Missing required gate measurements fail closed. Failed/blocked/unqualified nodes
are visible diagnostic observations but cannot enter the frontier.

Parameters, latency and memory are hard constraints in v1, NOT additional Pareto
axes. This deliberately narrower endpoint avoids claiming a four-dimensional
trade-off that has not been implemented. Defaults: 12 probes, one worker, at most
24 rounds, retention >= 0.5, parameter delta <= 0, latency ratio <= 1.25, recorded
peak host/VRAM <= 8 GiB. These are example fixture settings, not inherited Rineya
native permissions or empirically optimal thresholds. Freeze suitable thresholds
and consistently measured rates for historical admission BEFORE observing test.

Primary metric: exact 2-D Pareto hypervolume H versus requested probe count,
integrated over the common budget B and divided by B. Treat the curve as a
left-held step function: H before a batch is carried until the entire batch
completes. After stopping, carry the last H through B. This prevents rewarding
partial-batch hindsight. Baseline cost is sunk and excluded equally for all arms.
Report baseline-subtracted AUC as well as raw AUC, endpoint H, best valid capability,
requested/revealed counts, misses, stop reason, and the complete frontier/trace.
At B=1 only endpoint H reflects a discovery at the final instant; this convention
is intentional and separately tested.

Training/validation rank is lexicographic: mean cluster AUC, mean endpoint H,
then fewer mean requests. Initial incumbent wins exact candidate-score ties.
Paper-style diagnostic: best admissible capability - beta_work*N +
beta_parallel*N/max(1,rounds). It is not the Rineya selection metric.

Recorded GPU seconds sum over attempted nodes; wall seconds use the maximum
recorded duration per synchronous batch. Missing duration or any support miss
makes the corresponding total unknown, not zero. W>1 wall time is an idealized
replay proxy and does not model GPU contention. Default W=1 is appropriate for
first offline work; it creates no native workers. Attempt budgets are matched,
GPU budgets are NOT claimed to be matched. No GPU-hour savings may be inferred
from an attempt-only comparison. Curves retain measured costs where available.

## 5. Controllers, evolution, and statistics

Compare four registered arms: fixed prefix schedule (a new explicit baseline,
not a claim to reconstruct Cheng's historical scheduling), greedy valid-anchor
continuation, seeded random legal exploration, and the evolved adaptive policy.
All use identical worlds, gating, budgets, seeds and transitions, and reset state
per rollout. Policy randomness is local and reproducible. Offline configuration
ceilings are 64 workers, 32 seeds and 4096 mutation proposals; these are software
limits, not permission to allocate native compute.

Adaptive parameters govern root exploration, observed quality trend, recovery
priority for actual implementation failures, depth penalty, batch width, minimum
depth and stagnation patience. Invalid measured quality alone is NOT treated as
an implementation failure. Resource/environment failures remain diagnostic.
The family can still be imperfect; only experiments can establish its value.

Defaults: seeds 17/29/43, six generations, six mutations per generation, search
seed 20260926. Mutate bounded values; de-duplicate identical specs. Always retain
the incumbent during training selection. This guarantees non-decreasing replay
rank only on the SAME development worlds/objective/seeds, not on validation,
holdout or a real future experiment. Validation chooses among the frozen candidate
pool once and never generates new candidates. Search logs preserve all candidate
specifications, ranks, scalar per-world feedback and generation ancestry.

Primary heldout comparison: adaptive minus fixed AUC. Average seeds within world,
worlds within declared dependency cluster, then clusters equally. A 2,000-draw
percentile bootstrap resamples whole clusters, not individual prompts, probes, or
seeds. Under five clusters suppress the interval. With only five clusters the
interval remains fragile; it is descriptive, not sufficient evidence of general
superiority. Greedy/random contrasts are secondary descriptive comparisons.
Synthetic intervals have no scientific interpretation about Rineya.

## 6. What this can and cannot establish

Replay is restricted to the support collected by old controllers. It does not
predict unobserved branches, identify causal effects of decisions that changed
proposal context, or establish improvements on new programs. Sparse or selected
histories can be misleading. Cost records may be missing or incomparable.
No oracle policy, hidden optimum, best-known target, or heldout ceiling enters
selection or stopping. Prefix perturbation tests audit this boundary.

Passing component tests and the synthetic demo licenses only further offline
qualification. Next gate: reconstruct curator-reviewed, byte-bound, context-valid
histories and disjoint catalog; freeze historical plan; train/validate/freeze;
perform the heldout run once. A bounded live comparison requires a separate
DREAM-002 protocol, matched resource accounting, explicit operator authorization,
and untouched native model safeguards. Nothing in this package grants that.
