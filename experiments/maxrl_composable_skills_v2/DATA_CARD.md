# Dataset card — Rineya MaxRL Composable Skills v2

**Experiment MAXRL-COMP-002 | Version 2.0.0 | Seed 260922002**

Status: generated, independently audited, CPU mechanics pass; native unqualified.
Source commit: `847cd27e9427c8c5e3acaf81191b2136a628b130`.
Manifest SHA-256: `86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44`.

## Intended contribution

V1 supplied verifier-backed reasoning families. V2 adds explicit skill identity,
dependent multi-skill tasks, held-out combinations, per-stage dependency checks,
separate rehearsal and preservation banks, and actual-response enrollment tools.
It supports a research program, not a claim that data can enforce orthogonal
weights, lossless skill addition or general intelligence. No pretrained outputs,
model-derived labels, synthetic rollout histories, or model success rates occur
in the delivered data. Oracle responses are solver-produced audits or disjoint
optional supervised targets, never fabricated experience for online RL.

## Skills and combinations

R: resolve requested aliases through a finite reference graph to candidate items.
U: replay add/set/copy/swap events on current item values, then apply a cutoff.
S: select exactly k eligible items whose final values sum to a target.
P: order selected items according to a globally acyclic precedence relation.

With RUSP, the selected and ordered answer depends on all four stages. Missing
stages have explicit defaults; every public prompt includes all information needed.
Two data renderings (JSON or labelled fields) are invertible and tested against
verifier metadata. All answers use a single shared JSON envelope: one distinct-
index list or `{"answer":"UNKNOWN"}` only if no valid list exists.

| Combination class | TRAIN/CALIBRATION | DEV/TEST/CHALLENGE |
|---|---|---|
| Atomic R, U, S, P | Present | Present |
| Bridges RU, US, SP | Present in separate banks | Present |
| Pairs RS, RP, UP | Absent | Present |
| Triples RUS, RUP, RSP, USP | Absent | Present |
| Four-skill RUSP | Absent | Present |

WARMSTART contains only atomic tasks. Unseen combinations are unseen SIGNATURES,
not unseen primitive algorithms. DEV exposure/tuning prevents claiming a sealed
zero-shot test; no combined training does not mean no development exposure.
Use atomic-only donor manifests for the cleanest independent-composition study.
Using bridge/boundary training changes the experiment and must be declared.

## Counts and statistical units

41,500 primary rows across 10,450 parent clusters. Raw roles: train 22,000;
calibration 1,250; dev 4,150; test 4,150; challenge 4,150; warmstart 600;
replay_train 1,600; guard_dev 1,200; guard_test 1,600; guard_challenge 800.

The 22,000 TRAIN rows comprise 12,000 atomic, 9,000 bridge and 1,000 answerability
rows. There are 33,900 core rows, 2,400 answerability rows and 5,200 legacy-domain
preservation/replay rows across all roles. The 56,100 counterfactual stage checks
are additional auxiliary diagnostics, NOT counted as independent primary training
examples. Different views of one generated world are correlated and share a
parent and split. Curriculum ID manifests do not create extra examples or seeds.

## Verifier-first construction

Finite worlds are constructed with positive solution opportunities, then an
independent implementation recovers certificates from the public specification.
For R, reference closure is separate from reward-time per-request traversal;
for U, symbolic state provenance is separate from concrete replay; for S,
combination enumeration is separate from cardinality-sum dynamic programming;
for P, topological enumeration is separate from positional validity checks.
Neither reward nor reference search uses a hidden construction witness.

Rewards are exactly 0 or 1. Valid solutions need not match the reference list.
No style, brevity, partial-credit or format bonus is mixed into correctness.
Bounded strict parsers reject duplicate keys, nonfinite numbers, float/Boolean
integer substitutes, invalid indices, duplicate selections, extra fields and
oversized responses. Corrupt specifications are host errors rather than false
learner failures. There is no eval, execution of generated programs, remote-code
model loading or network access in the generator or verifiers.

## Certificate diversity, with the right interpretation

37,980 rows have at least two recovered accepted completion STRINGS, including
33,819 core, 1,188 boundary and 2,973 legacy rows. Many differ only in a permitted
ordering. Among the new composition domains, 8,997 core and 740 boundary rows have
recovered references with different selected ITEM SETS; another 24,822 core and
448 boundary rows have different orders but only one selected set in the bounded
reference list. These are lower bounds from at most three references per row,
not exact solution counts, independent reasoning strategies or a uniqueness proof.
The verifier also accepts valid solutions not recovered in that bounded list.

## Genuine dependence and answerability controls

Every active stage of every core row has a positive-control counterfactual. Only
that stage's specification changes; the original response fails and a separately
recovered response succeeds. Total 56,100 checks: R 12,450; U 15,600; S 15,600;
P 12,450. Some controls change an entire precedence relation rather than one edge;
we claim stage-local dependence, not minimal intervention or absence of shortcuts.

There are 1,200 answerable/impossible twin pairs. Only the selection target changes
by +1; the impossible target lies inside the minimum/maximum achievable sum range.
Independent enumeration and a separate DP checker agree on feasibility. This avoids
an out-of-range-label shortcut but not every possible parity or arithmetic shortcut.
UNKNOWN recall, false abstention and both-twins-correct are separate endpoints.
Core tasks are all feasible; impossible examples live in an explicitly separate bank.

## Preservation, not a no-forgetting promise

Eight legacy families are included: countdown, subset sums, linear constraints,
Boolean CNF, minimum-cost paths, precedence orders, affine programs and register
state tracking. They use unchanged v1 source. All 5,200 new instances are excluded
by exact normalized public-instance hashes against the provided 32,800-row v1
corpus; 5,604 colliding candidates were rejected during construction. Prompt hashes
are also checked. The original v1 ZIP's canonical file hashes were verified.

The 1,600 replay rows never enter guard banks. The 3,600 guard rows are evaluation
candidates, not proof of baseline mastery. `behaviour.py` enrolls ACTUAL source-
correct membership, binds the bank and scoring conditions, then reports lost
source-correct rows independently of gains. A zero source-correct denominator is
undefined, not perfect retention. The same mechanism can enroll the A-trained
model before B to measure retention of newly acquired A, not only old source skills.

To rehearse A during A->B, use A's TRAIN bank, never its evaluation children. Legacy
replay remains a separate budget. Mean-centered MaxRL gives zero signal on all-
success replay groups; explicit anchor-KL or supervised replay is a separate
controlled regularizer and is NOT implied by having replay data. This corpus has
no real-world long-context, medical, conversational or safety-retention guarantee.

## Split integrity and limits

A hash of each full latent world fixes its split; all children and controls inherit
it. Each primary problem, prompt and ID is unique. Auxiliary-control versus primary
and other-control hashes are also checked across splits. The complete audit found
zero cross-split exact collisions and zero exact collisions with provided v1 rows.

No claim is made about graph isomorphism, renamed aliases/variables, arithmetic
semantic equivalence, pretraining contamination, or every other historical corpus.
All supplied evaluation material is EXPOSED, not sealed. Challenge shares primitive
algorithms with TRAIN but uses longer histories/chains and reserved introductions.
Generator-defined difficulty is not measured Rineya difficulty or a validated
monotonic hardness scale. Some levels share item counts.

## Construction biases and shortcut diagnostics

Worlds deliberately contain groups of three equal-valued items to support multiple
valid certificates and stage-replacement controls. Items/aliases and event histories
are randomized, but repeated-value structure is a dataset bias. Precedence relations
are dense layered DAGs, not a representative graph distribution. No adversarial-hard
or natural-language transfer claim is made. The prompt interface is specification-
heavy; generative-skill validity under this interface requires actual model tests.

On the 3,000 TRAIN core rows per signature, the simple heuristic 'return the first k
numeric indices, or all indices without selection' succeeds on S:424, US:428,
SP:202, and zero on R/U/P/RU. Always UNKNOWN succeeds on zero core rows, by design.
These are fixed heuristic diagnostics, not model scores or universal difficulty
thresholds. They show why baseline calibration is needed, not why a model will learn.

Composition reference lengths range from 16 to 47 CHARACTERS. Serialized prompt
character lengths are min 586, median 1,740, p95 3,967, max 5,543. None are native
token counts. Prompt lengths, response budgets, EOS and chat templates remain to be
bound to the exact Rineya tokenizer. Silent truncation is never allowed.

## Executed validation and reproducibility

Combined suite: 462 passed (177 v1 + 285 v2). Full persisted corpus independently
re-solved: 41,500 rows, 116,528 positive certificates accepted, 581,000 defined
negative checks rejected, all 56,100 stage controls and 1,200 twin pairs passed.
Native pretrained runs, tokenizer loads and model-derived enrollment: zero.
Remote GitHub Actions execution is not claimed. All ten new Python Git blobs were
compared to the exact locally tested bytes; unchanged v1 source remains inherited.

An early full build was interrupted by a tool timeout and is omitted. The release
is a separate completed build with subsequent full audit. An early test incorrectly
required exactly three references for every precedence case; it was corrected to
the documented bounded one-to-three contract. The generator/verifier acceptance
criteria were not weakened. Synthetic fixture source-response tests are not model
results. Full audit source hashes include tests, so editing Python requires a new
source-bound receipt/version rather than reusing this release's claim.

## Formats, open requirements and provenance

Complete canonical, policy-only and verl-shaped JSONL, oracle sidecars, diagnostics,
ID bank manifests, optional disjoint messages-only SFT, source and receipts are in
the ZIP. No Parquet files were generated: PyArrow is missing and its installation
was network-blocked. The optional exporter checks roundtrip identity when available.
Native trainer compatibility and the successful Parquet path remain unqualified.

Remaining: exact model/runtime/tokenizer, actual mixed-success rates and skill-
positive-controls, preservation enrollment, token/compute budgets, historical
exclusion, broader human review, and prospectively fixed statistical margins.
Independent per-delta scaling is motivated by inspected Rineya evidence; vector
orthogonality is not established as a protection mechanism. Three-plus-skill
composition and retention are testable hypotheses, not dataset achievements.

All rows are generated locally from original bounded specifications. No external
benchmark examples, personal information, protected data or teacher prose were
imported. v1 code is the user's existing source and is included unchanged as a
dependency. No new blanket redistribution license is asserted for repository code.
