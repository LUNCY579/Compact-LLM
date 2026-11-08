# TRVC-001 research analysis and experimental contract

**2026-10-07 | Independent adaptation | CPU reference executed; native proposal**

## 1. Decision and source boundary

Explore verified computation memory as a data mechanism, and compare objective
choices independently. Do not adopt hard maximum as a default preservation method.
The practical payoff sought is a compact model that can execute useful new
combinations without retrieving its training archive, while retaining previously
accepted skills. A better search system is valuable but is a different achievement.

Primary source P1, *Tropical Reinforcement Learning*, arXiv:2610.02478v1, sections
4.3 and appendices C/D, describes observed-transition memory, verified path joins,
best-path plus coverage training, and a shared-parameter retention limitation.
It does not establish native 0.6B effectiveness. Its policy improvement theorem
assumes separate state parameters. [P1]

P2, *Maximum Likelihood Reinforcement Learning*, arXiv:2602.02710v3, motivates
log-success rather than success probability as an optimization target. Our
finite-bank T=1 objective is related at the population level, but is not its
sample-based on-policy estimator. [P2]

No author implementation was copied. The algorithms, fixtures and study design in
this directory are an independent small-domain implementation. No new paper
benchmark numbers are reproduced here. Public results motivate a test; they do not
supply Rineya measurements.

## 2. Three objects that must remain separate

**A certificate** records an executable path accepted by a fixed verifier. Its
validity depends on the task, state interface, rules, horizon and verifier version.
The certificate must remain useful even when its current policy probability falls.

**A search score** ranks certificates under one frozen policy. Rescoring after a
model update may lower this score. An append-only archive does not imply monotonic
current-policy likelihood. At capacity, this implementation stops rather than
silently discarding certificates; an eviction policy would be a separate study.

**A learning objective** determines which certificates change the actor. It can
use a maximum, a weighted set or explicit replay without changing the validity of
the stored certificates. Maintaining a best-path record does not force every
learning update to use only that path.

These distinctions prevent an archive-size increase from being reported as model
intelligence, and prevent a soft objective from being rejected merely because its
scalar value is not the score of one particular path.

## 3. Mathematical contrasts and the social-media concern

For disjoint, complete accepted paths, write W_i = log P_theta(path_i | problem).
With binary terminal verification, ordinary sampled pass@1 is the SUM of accepted
path probabilities. Partially correct failures have zero terminal reward; two
failures do not outweigh one success under that reward convention.

The best correct-path probability is instead max_i exp(W_i). Consider accepted
probabilities (0.30, 0.30) changing to (0.50, 0.05): the maximum rises but total
success falls from 0.60 to 0.55. This is a counterexample to an implication, not a
claim that a particular published training run made this transition.

For a stronger local example, let three outcome logits equal
log(0.31, 0.29, 0.40) + theta * (1, -2, 0), with the first two correct. A gradient
step increasing the most likely correct outcome can reduce their total mass.
`test_max_improvement_can_reduce_pass1_in_shared_parameter_actor` executes that
example. The shared parameter is crucial: it is not a contradiction of a theorem
whose hypotheses require independent state parameters.

Our primary objectives, averaged equally over eligible problems, are:

| Treatment | Per-problem loss |
|---|---|
| Sequence-uniform | -mean_i W_i |
| Hard maximum | -max_i W_i, deterministic first-index tie rule |
| Soft T | -T log sum_i exp(W_i / T) |
| Coverage-selected | -mean of W_i over best plus coverage-selected paths |
| Secondary per-token imitation | -mean_i W_i / response_length_i on selected paths |

The soft gradient is a weighted sum of path log-likelihood gradients. The weights
are softmax(W/T). T approaching zero concentrates on a winner; T=1 gives negative
log probability mass of the distinct verified candidate set. It equals full
log-success only when the set covers all accepted outputs under the specified
policy/output convention. An incomplete bank optimizes only its known subset.

Low-temperature smoothing does not create missing evidence or guarantee diversity.
A rare correct solution may still receive negligible weight. Conversely, assigning
positive weight to every failed trajectory is not justified by calling it a
near-miss. This study salvages *verified transitions* and trains only on complete
accepted paths. Error correction or process rewards are separate interventions.

Path probabilities must be summed over response tokens, including termination.
Length-normalizing before log-sum-exp changes its meaning. The implementation
keeps per-token imitation separate, rejects impossible candidate mass and uses
numerically stable centering; a temperature that underflows the score dtype fails
explicitly. Neither method implies that nonwinning neural parameters receive zero
change: shared representations and normalization can change many behaviors.

## 4. Exact-state composition, not textual splicing

The initial domain has two bounded integer registers. Actions increment, double
or swap registers. Invalid actions fail without becoming transitions. A state
contains the task hash, register values and depth. Task identity binds initial
conditions, goal, rule version, bounds, horizon and data role. Depth prevents
cycles and prevents a prefix from silently granting a suffix additional budget.

The policy sees only a canonical rendering of the current registers, goal, rules,
bounds and remaining steps. It never sees an oracle continuation, stored future
state, evaluation answer or archive backpointer. Equivalent exact states receive
identical prompts regardless of their originating histories.

This is also a scoring requirement. A suffix probability computed with a different
history is not automatically valid after transplantation. The native encoder must
use the same state-local context during sampling, frozen rescoring and training.
Arbitrary natural-language paragraphs, latent vectors and near-equal embeddings
are not admissible join keys in this version.

An archive stores only transitions actually executed through the exact simulator.
Every new solution is replayed from the original root and must reach an observed
accepting terminal. Edges retain source attempt IDs. Graph and solution capacity
are explicit; capacity errors retain existing memory and account for completed
work rather than silently dropping it.

## 5. First-solution provenance is a scientific control

Three origins are distinct: a complete root sample; a successful frontier suffix
attached to an observed prefix; and a graph recombination not sampled as a full
root trajectory. The first discovery label never changes when the same path is
later sampled from the root. Root-sample membership is recorded separately.

In a deterministic graph with terminal-state success, a union of unsuccessful
root rollouts contains no observed accepting terminal. Recombination alone cannot
manufacture one. This follows because every retained edge came from an observed
transition and a successful terminal would already have ended its root rollout
successfully. The scoped suite checks this invariant across eight random fixtures.

A restart can supply a successful suffix from an intermediate state. The prefix
may come from an earlier failed attempt, but the suffix episode itself succeeded.
The mechanism demo records this as `restart_completion`, not as an unexplained
root-only first solve. A second test demonstrates genuinely novel path orderings
assembled after a root success has supplied an accepting terminal.

This distinction gives us a precise question to investigate when reproducing an
ambiguous first-solve claim: what supplied the accepting ending, and when? It does
not warrant declaring another implementation invalid without its full protocol.

## 6. Two experiments before one combined pipeline

### A. Discovery mechanism, frozen actor

Use the exact same checkpoint and task pool. Compare:

D0: root-only sampling, trainable data limited to complete root successes.
D1: root and frontier sampling, with explicitly labeled prefix-plus-suffix completions.
D2: D1 plus verified DAG recombination.
D3: the same observed data with failed-origin fragments excluded, as a data ablation.

Also retain a no-accepting-terminal negative control and an exact-state diamond
positive control. D0/D1/D2 discovery policies need their own cost ledgers; sharing
a final bank does not retroactively equalize the cost of acquiring it.

Report full attempted problems, problems with any certified solution, novel path
counts, direct-root membership, successful-suffix counts, origin at first solve,
valid failed-fragment contribution, graph capacity events and all resource costs.
Construction witnesses are used only to make synthetic worlds feasible. They are
not actor outputs and never seed the archive.

### B. Objective choice, one frozen candidate bank

Fork the same accepted source or common warmstart into every treatment. Freeze the
problem bank and exact candidate identities before comparing objectives. Recompute
current path probabilities at each update. Compare sequence-uniform, hard-max,
coverage, soft T=0.25 and soft T=1.0. Include root-only-data sequence-uniform as a
data-source control and per-token imitation as a separate normalization control.

The source, trainable parameter inventory, initialization, optimizer, update cap,
batching, checkpoint selection and preservation treatment must match. Count both
candidate-forward work and gradient-receiving exposures: computing all paths before
selecting one has a cost even when only one contributes a direct path gradient.
Do not claim matched total compute from equal attempt or update counts.

**Candidate diversity gate:** before treating hard-versus-soft as informative,
require at least half the enrolled training problems to have two or more distinct
verified paths, and report path-level and operator-level diversity. Two syntactically
different traces may still exercise identical operations. When every problem has
one candidate, hard/uniform/soft sequence gradients coincide; an objective sweep
cannot answer a multiple-solution question. This gate emerged as an important
engineering diagnostic in the executed toy pilot, not as a post-hoc efficacy win.

## 7. Connection to Rineya's cumulative-capability program

The existing integrated `maxrl_composable_skills_v2/README.md` describes a typed
R/U/S/P problem family and explicitly separates acquisition, composition, replay
and guard roles. Its exposed evaluation banks must not be silently relabeled sealed.
Our inspection is pinned to parent 03db218, not a claim that all research branches
or local runs were re-audited. [R1]

This experiment would add a *verified bridge-data mechanism*. CSS, independently
weighted deltas and update-allocation methods would remain alternative parameter
mechanisms. First obtain independently useful skills; then introduce bridge data
as a separately named training intervention. A gain after bridge training is learned
composition, not proof that independent parameter merging achieved zero-shot
composition. No existing dataset or reserved signature is modified by this branch.

The current two-register simulator is independent of R/U/S/P. A correct staged
adapter from those tasks to exact intermediate transitions has NOT been implemented
or assumed. An existing final-answer verifier alone cannot certify invented
intermediate steps. That integration must preserve task semantics and withheld
composition signatures, or define a separate explicitly bridge-trained cohort.

For preservation, use disjoint acquired-skill and legacy training reservoirs, plus
separate accepted-reference evaluation banks. An initial objective can be
L_new + lambda_acquired L_acquired_replay + lambda_legacy L_legacy_replay.
The weights and data quotas are separate experimental choices. Reference KL is an
additional treatment; its teacher is detached and its prefixes and temperature
must be fixed. Freezing a backbone protects its bytes, not the behavior of the
backbone-plus-adapter system.

The code implements replay/KL loss components, not a completed native sequential
curriculum. The shared fixed-bank runner is deliberately the unprotected control.
A real preservation claim requires a qualified replay scheduler, accepted-skill
references, actual native runs and uncertainty-aware confirmation.

## 8. World-model and reverse-model integration: deferred extensions

An exact transition archive can provide labeled state/action/next-state examples.
A forward model could propose continuations; a goal-conditioned reverse model could
propose bridge states. Neither proposal becomes an accepted edge until it is
executed and checked by the independent simulator. A false bridge remains a failed
proposal with an accounted cost, not additional training truth.

Backward dynamic programming through stored edges is not itself a learned reverse
world model. Similarity of latent states does not establish interchangeable future
behavior. Start with the exact executor as the reference; only introduce learned
abstractions after testing their action validity, horizon behavior and out-of-domain
failure rates against that reference. These modules are not implemented here.

## 9. Proposed native stages and resource envelope

E0 must bind the exact 0.6B checkpoint, tokenizer, code, verifier, data and trainable
parameter inventory. On the intended Windows host TheFakeGod, collect actual
post-import memory readings rather than reusing historical headroom. Preserve the
existing conservative 7.4 GiB host admission floor unless a new operating plan is
separately qualified. This package does not launch or modify that host.

Before learning, verify canonical prompt parity, repeated-state scoring parity with
dropout off, untruncated response masks, EOS when EOS equals padding, one causal
shift, complete-path probability accounting, disabled-adapter parity where relevant,
and no oracle fields in policy inputs. Then qualify a disposable backward/update/
resume using the intended precision, shapes and optimizer scope. A source-specific
loss or memory failure is a blocked prerequisite, not evidence against the idea.

E1 is a bounded frozen-policy collector screen. The proposed initial envelope is
32 TRAIN problems, eight root plus eight frontier attempts per problem, horizon at
most four actions, 32 newly generated tokens per action, at most 1,024 retained
edges and 64 solutions per problem. The generated-token upper bound is 65,536;
prompt tokens, rescoring, replay and optimizer work are additional costs. Freeze
actual limits after the tokenizer/resource screen. No final-test problems enter
this collector. Check the diversity gate before launching E2.

E2 is the shared-bank objective comparison with an initially small, source-qualified
adapter or other explicitly fixed parameter scope. Do not silently substitute
full fine-tuning for an 8 GiB host. The first practical dose and optimizer settings
must come from E0, not copied from a larger-model paper or from this tiny actor.

E3 adds separately controlled acquired/legacy replay to the best-supported data
mechanism. Measure an acquisition-retention frontier rather than rewarding an
undertrained model for not changing. A meaningful third-skill sequence follows
only after acquisition and two-skill retention are established.

E4 freezes the recipe and confirms on fresh parents and reserved composition
structures. Keep model-only, archive-assisted and distilled scores separate.
No automatic promotion or base-branch mutation is authorized by a positive pilot.

## 10. Evaluation, uncertainty and decisions

Primary output is unaided, fixed-decoding pass@1 on unseen composition problems.
Secondary measures include pass@k, invalid-action rate, root-only success, path and
operator diversity, earlier acquired-skill performance, legacy behavior, source-
correct examples lost, and resource-normalized acquisition. Retain raw sampled
actions/tokens and verifier outcomes rather than only means.

Freeze evaluation contracts and compare the same task keys, parent IDs, attempt
counts and decoding settings. The implementation reports parent-macro paired
changes with a parent bootstrap for one training draw. Across multiple training
seeds, report seed variation and use a hierarchical analysis rather than treating
all outputs as independent trained models. The included tiny-actor confidence
intervals are descriptive and do not establish native statistical power.

Suggested confirmation targets, to preregister before a native comparison, are a
practically relevant +3 percentage-point pass@1 improvement, a paired confidence
interval excluding zero, and no more than a 1-point decline on each protected
capability under a declared cost ceiling. These are proposed decision margins,
not existing measured effects or automatic code thresholds. Select the actual
confirmation sample size using expected discordance and the protection margin;
a small bank cannot certify a 1-point retention limit merely by showing no losses.

Accept only a model-only improvement that meets its acquisition, preservation,
transfer and cost contract. Classify archive-only gains as system/data successes.
Classify confidence intervals crossing margins as inconclusive. Classify an empty
bank as a collection/competence bottleneck, a singleton bank as an objective-
discrimination bottleneck, and absent native resources as an execution blocker.

## 11. Executed versus pending

Executed: independent CPU implementation, scoped tests, scripted prefix/suffix and
graph controls, a three-seed seven-arm tiny-actor pilot, raw evaluation receipts,
and an intentionally blocked native intake. The pilot is exposed and exploratory.
See `RESULTS.md`; no objective is selected as a Rineya winner.

Pending: exact Rineya checkpoint callbacks and local tokenizer calibration;
source/precision/adapter/backward/resume qualification; frozen native collection;
a candidate-diverse bank; native retention scheduling; R/U/S/P state adapter;
full sequential study; fresh structural generalization and deployed export checks.

## References and inspected integration surface

[P1] Asadulaev et al. *Tropical Reinforcement Learning*. arXiv:2610.02478v1,
2026-10-01. https://arxiv.org/html/2610.02478v1 . Method and limitations inspected
2026-10-07. Inspiration, not independently reproduced benchmark evidence.

[P2] Tajwar et al. *Maximum Likelihood Reinforcement Learning*.
arXiv:2602.02710v3. https://arxiv.org/html/2602.02710v3 .
Official project: https://zanette-labs.github.io/MaxRL/ .

[R1] LUNCY579/ChatBubble, immutable parent
03db218ceebc61a2bc04f2482cf7e094045e01c1,
`experiments/maxrl_composable_skills_v2/README.md`.
https://github.com/LUNCY579/ChatBubble/blob/03db218ceebc61a2bc04f2482cf7e094045e01c1/experiments/maxrl_composable_skills_v2/README.md .
Inspected integration contract; no native result inherited.
