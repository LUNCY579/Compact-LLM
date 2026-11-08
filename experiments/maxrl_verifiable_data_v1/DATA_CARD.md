# Dataset card: Rineya-MaxRL-Verifiable-v1

Status: **SYNTHETIC_DATA_VERIFIED_NATIVE_UNQUALIFIED**.
Experiment: MAXRL-DATA-001. Root seed: 260202710. Version: 1.0.0.
Full manifest SHA-256:
`3a7f1a20d15d6c2ce26eeb02ae3c0a6353f771955230b390c35ac6739c241563`.

## Purpose and counts

Original, bounded, self-contained short-answer problems for online binary-reward
post-training. A successful model sample is any completion accepted by the rules;
reference-string matching is not the reward. There are no fabricated model
rollouts, teacher ratings, pretrained success estimates, or model-training results.

| Split | Problems | Role |
|---|---:|---|
| train | 24,000 | Online MaxRL/GRPO prompts |
| dev | 2,400 | Exposed development |
| test | 2,400 | Exposed held-out-instance testing; not sealed |
| challenge | 2,400 | Larger/harder instances and reserved wording |
| warmstart | 1,600 | Optional disjoint supervised warm start |
| Total | 32,800 | Unique canonical instances, not independent semantic domains |

Each family has 3,000 train, 300 dev, 300 test, 300 challenge and 200 warm-start
instances. TRAIN/DEV/TEST use difficulty 0/1/2 in equal proportions. Challenge
uses level 3. Warm-start uses 120 level-0 and 80 level-1 examples per family.

| Family | Certificate and independent check | Rows with >=2 recovered references, all splits |
|---|---|---:|
| countdown | Arithmetic expression; subset DP versus exact-Fraction AST evaluation | 3,178 |
| subset_sum | Distinct index subset of exact size; combination search versus sum/uniqueness checks | 2,826 |
| linear_constraints | Finite-domain assignment; residual enumeration versus equation evaluation | 2,066 |
| cnf_assignment | Bit assignment; bitmask search versus literal evaluation | 4,100 |
| shortest_path | Minimum-cost path; exhaustive simple paths versus Dijkstra | 296 |
| precedence_order | Any topological order; available-node search versus position constraints | 4,093 |
| finite_program | Affine-operation sequence; algebraic composition versus concrete execution | 1,964 |
| state_tracking | Final register state; symbolic provenance versus concrete replay | 0 |

At least 18,523 problems have multiple recovered accepted completion strings.
These are lower bounds, not exact counts of solutions or independent reasoning
strategies; arithmetic strings can express equivalent computations. Unlisted
valid solutions still receive reward 1. The remaining instances must not be
labelled uniquely solvable merely because bounded search returned one reference.

## Construction and safeguards

Construction uses dedicated seeded streams by family, difficulty and split.
Witnesses establish satisfiable candidate construction and are then discarded.
An independently implemented search recovers references from the public spec.
The actor never receives construction witnesses or oracle sidecars. There are no
external benchmark imports, natural-language teacher generations, network calls,
medical content, personal data, or generated executable programs.

Only exact integer/rational semantics are used. Responses must be one JSON object
with the single key `answer`. No format or length bonus is awarded. Duplicate keys,
nonfinite values, booleans/floats used as integers, extra fields, excessive response
length, invalid syntax, and forbidden arithmetic constructs fail. Corrupt verifier
metadata raises an error instead of turning into a spurious learner failure.

Canonical public-instance hashing assigns each problem to one split independently
of prompt wording or generation seed. Hashing normalizes selected irrelevant
orderings, such as countdown number order and clause order. It does NOT solve graph
isomorphism, variable renaming, equation scaling, or semantic equivalence in general.
No claim of historical-corpus exclusion, unseen algorithmic skills, or general
semantic disjointness is made. The larger CHALLENGE shares family algorithms with
TRAIN; only its increased complexity and prompt intros are reserved.

## Executed quality evidence

The persisted full corpus passed visible-only re-solving, all export alignment
checks, 66,822 positive certificate checks and 360,800 negative checks (10 malformed
responses plus one checked incorrect response per instance). There were zero exact
canonical-instance, prompt-hash, or ID duplicates and zero split violations.
The negative controls test defined failure modes, not absence of every possible
verifier exploit. The local suite passed 177 tests. One initial float equality
assertion was corrected to a numerical tolerance; no behavioral gate was loosened.

An early full construction exhausted the finite easy-linear candidate space in
the warm-start partition and correctly stopped. The unpublished generator's
coefficient support was widened from [-2,3] to [-3,4], then all tests and the full
corpus were rerun. An earlier process was interrupted by the tool time limit.
Neither partial build is included in the release. Published counts/hashes refer
only to the final successful full build and its subsequent complete audit.

Serialized prompt character lengths: min 585, median 774, p95 900, max 1,036.
These are JSON-serialized character lengths, NOT model token counts. Warm-start
answer regex proxy lengths range from 11 to 21, with zero above 24; native token
lengths and the exact historical SFT consumer schema remain unqualified.

## Simple shortcut diagnostics (TRAIN only)

These are deterministic heuristics, not model results or acceptance thresholds:

| Heuristic | Correct / 3,000 |
|---|---:|
| Sum all countdown numbers | 202 |
| Choose first k subset indices | 131 |
| Set all linear variables to zero | 83 |
| Set all Boolean variables to zero | 676 |
| Return direct start-to-goal path | 793 |
| Return ascending precedence indices | 221 |
| Repeat operation zero in the program | 86 |
| Return unchanged initial registers | 2 |

The Boolean and graph baselines show that the corpus includes accessible examples;
this is not an adversarial-hard benchmark. Algorithmic difficulty is not measured
model difficulty. Calibrate actual all-failure/mixed/all-success group rates on
eligible source-model data before selecting a fixed training mixture. Do not
claim that this generator has established rare success under Rineya.

## Remaining limitations and intended use

Human review of stratified prompts, exact model/tokenizer/runtime binding,
response/EOS/logprob parity, external overlap checks, a fresh preservation bank,
and native sampler/trainer qualification remain open. Parquet export could not
run because the engine is absent; the provided converter fails explicitly.
JSONL was generated and audited, but no native training-ready declaration is made.

The initial study is objective-only GRPO versus MaxRL with matched source,
trainable scope, data, tuning allowance, and explicit cost ceilings. Do not mix
NGU, architecture changes, tangent training, sparse derivatives, or adaptive
retry-until-success sampling into this first comparison. See EXPERIMENT.md.
All included evaluation material is exposed; it is not a sealed promotion gate.
