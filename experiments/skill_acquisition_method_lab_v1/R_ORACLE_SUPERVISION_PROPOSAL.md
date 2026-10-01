# R oracle supervision after the failed signal screen

**Status: UNAPPROVED NEW EXPERIMENT.** The registered warm-anchor path ended
STOP_NO_SIGNAL. This proposal authorizes no second warm dose, pilot, enrollment,
S1 run, or other training. It preserves that stop result.

## Measured reason for a new choice

The complete native read used 450 prompts and eight samples per prompt. R had
6/1,200 strict/content successes and only 5/150 mixed groups; the gate required
15 mixed groups. S had 72 mixed groups and P had 50. R had 1,198 answer-object
normalizations, zero counted format-only failures and zero length caps. This
supports investigating content supervision before another sparse-reward run.
It does not prove that a supervised treatment will acquire R.

R resolves requested aliases through links, deduplicates terminal item indices,
and returns the full resolved set in any order. TRAIN has 3,000 R rows, with
1,000 each at difficulties 0/1/2 and set sizes 6/8/10. The completed warm-start
used only 150 R rows, covering difficulties 0/1. Coverage is a hypothesis, not
an identified cause of the native failure.

## Concrete treatment choices

| Choice | Labels on the same 3,000 TRAIN prompts | Purpose |
| --- | --- | --- |
| Canonical oracle SFT (recommended first) | One independently verified ascending-index answer per row | Establish a positive-supervision acquisition baseline. |
| Paired canonical/permutation SFT | A canonical arm and a separately trained fixed-permutation arm | Test whether preferred output ordering changes transfer or interference. |

Every prepared canonical and permutation target passes both the independent
strict checker and corpus reward. The fixed permutation is keyed by domain,
seed 1301, row ID, logical exposure 0 and item index. A retry reproduces its
target. Neither target policy changes the evaluation contract or claims that
the oracle's capped reference list enumerates all legal answers.

Proposed common recipe, to be frozen by the new decision:

- Start each arm independently from exact deployed theta0-prime
  `ff516ff86259c101a8cf55b4bf548e267f74719dc450ad34538f4383ca55d84b`.
- One pass over all 3,000 atomic R TRAIN rows, identical hash order across arms;
  microbatch one, accumulation eight, exactly 375 exposure boundaries.
- Proposed inventory B: blocks 14–27, 220,233,216 parameters. Fresh AdamW per arm:
  lr 1e-5, betas (.9,.999), eps 1e-8, weight decay zero; constant rate, global
  gradient norm clip 1.0. These are proposed settings, not a reused grant.
- Answer plus EOS only; prompt masked; one-shift likelihood; mean labeled-token
  CE per row, then equal row weights within the accumulation boundary. Preserve
  pinned tokenizer/template and FP32 deterministic runtime; no truncation.
- Freeze checkpoint cadence, all attempted-work limits and retry/rollback
  allowances before launching. Qualify the longest TRAIN prefix/target, actual
  update, resume, delta extraction and resource fit. Existing native warm memory
  measurements and tiny CPU checks do not qualify the entire new 3,000-row run.

The pinned-tokenizer census verifies both plans have 2,560,912 prompt tokens,
64,712 target/EOS tokens, and 2,625,624 total sequence positions. Maximum sequence
length is 1,287; all 3,000 rows have identical completion lengths between arms,
exact template alignment and no truncation. Equal exposure counts alone do not
imply equal token compute; this comparison has an explicit matched census. Paired
arms need matched seeds, budget policy and independent optimizer starts; a
single training seed is a feasibility comparison, not a robust efficacy claim.

## Why target permutations are a substantive treatment

For one R prompt let A be its resolved set and V_can the permutations serialized
as compact canonical JSON with fixed EOS 151645. Let p_theta(y) be the sequence
probability including that EOS, P_can=sum_{y in V_can} p_theta(y), and
r_theta(y)=p_theta(y)/P_can. The full strict-verifier event also accepts other
JSON spellings and the second permitted EOS. Thus P_can <= P_strict; neither
the n! count nor the estimator below enumerates that whole event.

Canonical sequence NLL obeys

    -log p_theta(y0) = -log P_can - log r_theta(y0).

The probability of the chosen sequence lower-bounds P_can and P_strict; its NLL
upper-bounds their event NLLs. It also prefers one ordering. For a genuinely
randomized uniform teacher Q over V_can,

    E_Q[-log p_theta(y)] = -log P_can + log |V_can| + KL(Q || r_theta).

Uniform target CE adds an ordering-distribution constraint beyond the task.
It is therefore not equivalent to maximizing total valid-answer probability.
The prepared hash target is one fixed target per row; it is not an exact
uniform-teacher expectation. A comparison must not mislabel it marginalization.

Here |V_can| is 6!, 8!, or 10!: 720, 40,320, or 3,628,800. Exact marginal likelihood
over all Transformer answer prefixes can be expensive; distinct order prefixes
cannot generally be collapsed into one subset state. For independent uniform
permutation draws, (|V_can|/K) sum_k p_theta(y_k) is an unbiased P_can estimate, but
log of that estimate is downward-biased by Jensen, and its self-normalized
gradient is generally biased at finite K. Such a marginalization treatment
would need its own budget and qualification.

These identities use full sequence NLL. The proposed row-token mean divides by
the number of labeled tokens; actual tokenizer lengths must be checked before
asserting equivalence across target permutations.

The exact item-order trie further separates a single teacher path from exhaustive
permutation likelihood. Depth k has n!/(n-k)! distinct ordered prefixes, so its
node count is the sum over k=0..n:

| Resolved set size | Compact orderings | Item-prefix nodes, including root |
| --- | ---: | ---: |
| 6 | 720 | 1,957 |
| 8 | 40,320 | 109,601 |
| 10 | 3,628,800 | 9,864,101 |

Across the measured 1,000/1,000/1,000 TRAIN histogram, this gives 3,669,840,000
prompt–answer order pairs and 9,975,656,000 item-prefix edges, counted separately
for each prompt; one teacher per row
uses 24,000 item transitions. `permutation_complexity.py` audits the closed form
against explicit tries for sizes 1..6. These are combinatorial counts. Actual
token-prefix counts, KV memory and runtime need separate measurements.

Subset-only merging additionally requires a proven state-equivalence property.
For example, prefixes [0,1] and [1,0] visit the same subset, but a general
autoregressive model can assign probabilities .9 and .1 to the required next
item 2. The oracle's subset state does not establish equality of the model's
continuation distributions. The exact audit is retained in
`evidence/R_PERMUTATION_COMPLEXITY_20261001.json`.

## Evaluation and scope before execution

Register exact baseline/donor reads before any new model output:

- Fixed R DEV acquisition comparison against theta0-prime, with no checkpoint
  selection or DEV retry. Proposed criterion: net gain at least 13/250 plus the
  original one-sided exact McNemar threshold .05.
- Named S/P and language/source-retention banks, matched source-correct loss and
  repair denominators, and deterministic enrollment checks. Their results are
  required before claiming that R was acquired without forgetting.
- CALIBRATION has already been inspected during the stopped study. Any new use
  is declared adaptive diagnostics, not an untouched confirmation set. TEST,
  CHALLENGE and guard_test outputs remain sealed during operator/recipe choice.
- A later R/S/P composition panel still needs independent donors, nonpartner
  controls, seed twins, operating-dose calibration and functional retention.

No R-only result completes the broader skill-composition goal. The existing
600-row/Supa warm-start entry point and MaxRL S1 entry point do not implement
this new 3,000-row common-anchor SFT experiment. A separately registered narrow
trainer and approved measured limits remain necessary.

The shared prospective CE boundary is now implemented in
`../skill_acquisition_native_v1/ce_update.py`. The earlier five pure controls and
ten actual CPU model controls passed in the superseding CE-core V2 receipt, including the
independent gradient reference, clipping/Adam order, post-step cleanup failure,
and rejection of an earlier negative second-moment chunk before model work.
Review prompted the chunk-validation fix; the prior receipt is preserved.
This core admits synthetic
CPU fixtures only. It does not implement the native SFT driver or qualify the
existing B2 checkpoint plan for CE8 boundaries, and grants no new training.

A separate synthetic CPU CE8 checkpoint adapter now exists. Its two actual
integration cases passed within a 43-check combined CE/legacy-storage suite:
stochastic fresh-object resume reproduces the uninterrupted second update bit
for bit, and latest-payload corruption fails without mutation or fallback.
The comparison runs within one OS process and uses an in-memory fixture observer.
It does not qualify durable production CE accounting, a fresh-process native
resume, the proposed 3,000-row experiment, acquisition or retention.

The pure CE8 planner now binds both verified 3,000-row target files to 375
identical exposure boundaries and separate label digests. Independent exports
pin the actual completed warm-anchor receipt. Recipe, native runtime and full
trainer-source references remain unresolved; every execution/grant flag remains
false and proposal plans cannot export synthetic checkpoint scope. Full
canonical/policy TRAIN files must be revalidated at future native ingestion.
Receipts are retained under `../skill_acquisition_native_v1/evidence/`.

The shared CE core now binds exact copied prompt/target token contents and a
static recipe digest before admitted model work. The separate synthetic CPU
durable journal retains all admitted reservations across reopening and ancestor
rollback, and requires full update/cleanup and fresh restore-fence proofs.
Its combined 74-case suite passed, including 25 actual CPU cases; collection
imported no frameworks and CUDA remained uninitialized. The four initial test
failures and their fixture-only corrections are preserved with the final receipt.

This is preparation for both treatment choices. Synthetic declared profiles,
bounded bootstrap validation and resume within one OS process do not supply a
native execution grant, measured full-unit budgets, outer setup/hash admission,
a qualified process/resource controller or scientific learning evidence.
