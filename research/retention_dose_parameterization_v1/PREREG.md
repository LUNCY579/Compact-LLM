# DOSEINT-001: D4 dose plus FP32 interpolation

Prospective exploratory development experiment. Written before this experiment's
training, materialization, or probe outputs exist. No release, promotion, or
confirmation claim. Authority: the current user's goal-objective.md explicitly
requests one D4 training run and alpha = 0.25/0.50/0.75.

## Fixed design and build gates

Reuse dose-D4-prepared and dose-D4-PLAN.json byte-for-byte: 3,072 unique training
parents, 433,813 processed tokens, one exposure, seed 17, FP32 blocks 8–27,
219 optimizer groups. Execute the existing qualified native lane in the clean
pilot checkout with a fresh grant and journal, including its automatic fresh
zero-update qualification. Research code lives on
research/retention-dose-parameterization-v1, based on the audit lineage.

Keep the native checkpoint. Build four HF arms with identical source sidecars:
alpha = 1.00, 0.25, 0.50, 0.75, in that execution order. Use FP32 throughout;
copy alpha=1 directly. No BF16 export, adaptive grid, retraining, replay, or
parameterization change. The four arms share ONE training draw.

Before launch verify all prepared-file hashes, exact plan inventory, source
assets, unique IDs/parents, and disjointness against the acquisition bank.
The original dose bank is process-transition-probe-001/CANDIDATE_VIEWS.jsonl
(384 parents, sha256 starts 45674251). It remains REPORT-ONLY / NEVER FOR
SELECTION. This experiment characterizes a curve and does not select a model.

The pre-training audit found 275/384 SELECT-VIEWS-001 parents in D4 training.
Consequently SELECT-VIEWS-001 MUST NOT be used for a held-out D4 bridge. Do not
score it in this experiment. A later bridge needs a newly frozen disjoint
instrument and source baseline; do not infer a bank offset or transport totals.

## Predictions and fixed bands

Historical totals in the user's objective only motivate the following frozen
targets. They are not newly validated pairwise comparisons to historical arms.
The old probes lack contemporaneous model-byte custody, and the historical
retention probes omit most per-row predictions. Their direct contrasts are
BLOCKED_ON_COMPARABILITY, not silently reconstructed.

| alpha | acquisition /384 band | retained /47 band |
|---|---|---|
| 1.00 | 284–324 | 24–36 |
| 0.25 | 220–285 | 40–47 |
| 0.50 | 260–310 | 37–44 |
| 0.75 | 280–320 | 31–40 |

Primary prediction: at least one of the three interpolants simultaneously has
acquisition >=284, retained >=40, and arithmetic panel >=62/150. These are fixed
engineering targets, not an equivalence/noninferiority proof. Also report whether
that same arm restores ratio_proportion to >=16/50 and termination to 150/150.
Mark every individual band and the primary conjunction PASS or FAILED_PREDICTION.
No band is changed after results.

Secondary predictions: acquisition is nondecreasing across 0.25→0.50→0.75→1.00;
source-answer retention is nonincreasing. Long-context total and literal UNKNOWN
are report-only: publish all four arms and paired contrasts, do not select on
these spent instruments. No prediction that the confirmed abstention mechanism
is repaired by interpolation.

## Comparability and analysis

Each probe has a contemporaneous custody receipt: SHA256 of weights, six source
sidecars and materialization receipt before and after execution; raw result hash;
scorer script hash; supervisor and terminal receipt hashes; bank hash and scorer
implementation digest. Both execution legs and result validation must succeed.
Use identical frozen scripts and banks across all arms. A missing/mismatched
binding produces BLOCKED_ON_COMPARABILITY with no endpoint numbers for that
comparison. This adds provenance; it does not alter generation or scoring.

Compare each interpolant to the new alpha=1 arm by row ID. Every acquisition
contrast reports (net, nd, exact two-sided conditional binomial p), wins/losses,
and its discordance-specific minimum detectable net at p<=0.05. nd=0 gives p=1.
The absolute minimum significant net is 6 when all six discordances favor one
arm; larger discordance raises it. The >=20 cross-training-draw floor is NOT
applied to these deterministic transforms. No new cross-draw contrasts are
authorized without valid bindings and |net|>=20.

For references_retained, compare correctness on the SAME 47 frozen source IDs,
not the arm/source union reference_size or its misleading retained aggregate.
Record the 150 full predictions, strict and permissive policies separately,
panel correctness, lost IDs, per-class ratio performance and termination.
Use the same paired statistics. Retention's best-case detectable net is 6/47;
observed nd determines the actual threshold. A nonsignificant difference does
not establish equivalence. Long-context and literal-UNKNOWN maps receive the
same exact paired report and endpoint-specific detectable net.

The primary result is the fixed numerical conjunction. Nominal p-values are
descriptive; report Holm-adjusted p-values over the three alpha-vs-full contrasts
separately for each endpoint, without making a finding depend on choosing a
convenient multiplicity family. No population or general intelligence claim.

## Numeric session envelope and stopping

Agent-instantiated limits under the user's explicit one-run/three-transform
instruction, not fabricated separate human review:

- One native training launch, zero training retries, 433,813 processed tokens;
  native automatic qualification and <=240 evaluation calls /30,720 generated
  tokens; 2,000s child timeout; <=2,400s native worker budget, <=3,600s owned work.
- Four materializations, <=16 GiB total new disk including both native checkpoints
  and all HF arms; keep >=14 GiB free; do not delete evidence or checkpoints.
- Twelve probe legs (4 arms × acquisition/retention/long context), sequential;
  at most one retry per failed inference-only leg in a NEW attempt directory,
  900s timeout each, <=10,800 worker seconds including failures/retries,
  <=6,320 evaluated rows and <=850,000 generation tokens including retries.
- <=14,400s total session wall time; zero paid/cloud/teacher spending, zero
  adaptive model-selection rounds. The native journal's required one round is
  bookkeeping for the single fixed training launch, not adaptive search.

Honor rineya-gpu-0.lock. Keep the unchanged 6 GiB training and 4.6 GiB probe host
RAM guards and pinned environments. Never load competing model workers. Stop
on a failed build/comparability gate, budget exhaustion, unverified checkpoint,
or unexplained model changes. Preserve every failure. No wrapper prints COMPLETE
unless every required leg's exit code and success receipt are verified.
