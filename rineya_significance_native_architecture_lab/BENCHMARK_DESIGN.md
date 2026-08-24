# Benchmark design

The active model scope is the published Qwen3-0.6B family, with Halo as the
primary benchmark target. The design separates base-model capability, context
preservation, reasoning, and decode.
Rineya/R2 and Aegis models are excluded from current evaluation and conclusions.

The broad Halo confirmatory protocol is `DRAFT_NOT_REGISTERED`. A 32-cluster
calibration set, an outcome-blind frozen retention endpoint, a 1,176-case
R1-R8 design ladder, corrected R7 v2, matched control studies, and Experiments
B-F now exist. The strict all-token diagnostic saturated at zero;
`layer-fact-retention-v1` replaces it. A first prospective dataset stopped at
B0, while a difficulty-calibrated second dataset passed B0 and completed one
12-cluster local registration. That narrow validation does not freeze the broad
independent-cluster plan.

## Ladder

| Stage | Capability isolated | Independent verifier |
|---|---|---|
| `R0` | full-cache Halo capability gate | task-family scorer |
| `R1` | multi-key retrieval and composition | arithmetic/key solver |
| `R2` | relational chains, 2–8 hops | graph solver |
| `R3` | entity/state updates and late correction | state simulator |
| `R4` | conflicts, negation, authority, supersession | ordered evidence resolver |
| `R5` | query-similar distractors | annotated evidence matcher |
| `R6` | multi-hop numerical reasoning | deterministic calculator |
| `R7` | rules, exceptions, and overrides | rule engine |
| `R8` | stack/queue/register/automaton tasks | state-machine simulator |
| `R9` | selected public long-context tasks | official task scorers |

`R2` in this table is a benchmark-stage label, not a Rineya/R2 model. Only
cases solved by full-cache Halo inform compression-conditioned answer claims;
unconditional outcomes are still reported.

The generated [`halo_r1_r8_ladder_v1`](results/benchmarks/halo_r1_r8_ladder_v1)
instantiates R1-R8 as 1,176 exact-aligned cases in 84 independent clusters:
R1 has 490 cases over required-fact counts 2/3/4/6/8; R2-R8 each have 98.
It crosses seven clustered evidence depths with seven fixed layouts plus the
clustered layout. This is Tier-A construction evidence only, not 1,176 model
outcomes or independent replications.

The bounded R5-R8 B0 screen used eight clusters (two/family) from this ladder.
Its raw scores were R5 `2/2`, R6 `1/2`, R7 `2/2`, and R8 `0/2`, but R7 v1 is
invalid/uninformative: the seven generated seeds are 6 NO / 1 YES, both screen
seeds are NO, an exact lock sentence permits a solver shortcut, and all seven
facts were annotated required. Its 2/2 cannot count as capability; corrected
R7 v2 is described below. The currently valid R5/R6/R8 subset is descriptively 3/6 and R8
stops at full-cache capability. No compression policy ran.

Corrected `halo_r7_v2` contains 80 balanced cases over eight semantic clusters,
crossing three evidence depths and eight layouts. Its independent rendered
solver reads only question/segment text, and necessity plus counterfactual
audits prove the annotated active proof. The frozen screen was `4 YES / 4 NO`.
Halo emitted `YES` for every case, scoring `4/8` and `0/4` on the NO side. The
benchmark repair is valid, but the prospective two-sided B0 gate failed, so no
R7 v2 compression cell ran.

## Evidence contract: strict diagnostic rejected, endpoint v1 frozen

`halo_calibration_v3` aligns each required fact to exact rendered-prompt token
positions. The first candidate endpoint required, for every one of 28 evaluated
layers, every required fact token to be resident in at least one KV head. In the
11-cluster pilot it was false for B2, B3, and B4 on every cluster (0/11 each).
This all-zero saturation rejects it as a useful primary endpoint for the
current implementation/budget; it does not show that retained partial evidence
was uniformly absent.

Outcome-blind calibration then froze `layer-fact-retention-v1`:

```text
layer_fact_exact_completion_rate
  = mean over (layer, required fact) of
    1[every exact token of that fact is resident in some KV head of the layer]
```

The calibration read only evidence mappings from all 33 pilot rows; condition,
correctness, predictions, accepted answers, failure labels, and task family
were forbidden. The endpoint had 14 distinct values, range `[0, 0.482143]`,
floor fraction `0.4242`, and no ceiling cases, so it passed the predeclared
non-saturation screen. The strict Boolean remains a stronger diagnostic.

Its application back to the same 11 endpoint-design clusters is post hoc
(`B3-B2=+0.061688`, 95%
cluster-bootstrap interval `[0.030844, 0.094156]`, exact sign-flip
`p=0.0078125`). Prospective v1 kept the endpoint frozen but stopped at its B0
gate (`9/24`). Difficulty-calibrated v2 passed B0 (`18/24`) and locally
registered 12 clusters before compressed execution. Its primary B3-B2 endpoint
contrast was `+0.0401786`, interval `[0.0133929, 0.0758929]`, `p=0.01953125`.
Answer exact-match moved from B2 6/12 to B3 5/12 (`p=1.0`) and remains distinct
from retention. The local result validates the endpoint narrowly; a larger,
diverse broad protocol remains required.

## Cluster-aware generation

A latent world/template instance may generate multiple evidence depths,
placements, policy runs, and compression schedules. All such derivatives share
a stable `cluster_id` and are treated as one independent unit.

Required rules before freeze:

- split train/calibration/evaluation by `cluster_id`, never rendered prompt;
- keep every B0/B2/B3/control version of a case in the same cluster;
- keep schedule and depth variants of the same latent world in the same cluster;
- resample or randomize whole clusters;
- report independent cluster count separately from prompt/run count;
- freeze the minimum eligible cluster rule before model outcomes.

The old Tier-A run's three repeated-exposure traces do not automatically count
as three independent clusters. Its methodology is under review.

The calibration implementation uses `cluster_id = family + latent seed` and
contains 32 independent clusters, eight per R1-R4 family. The ladder expands
this to 84 independent clusters while retaining all depth/layout derivatives
inside their latent cluster. Both are valid for development accounting, but the
confirmatory split and sample matrix remain unregistered.

## Position and schedule factors

Candidate evidence depths are:

```text
0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
```

Layouts may include all-early, all-middle, all-late, early+late,
early+middle+late, clustered, uniform, and seeded random placement. The
compression interaction changes active-chunk size while holding tokenized
prompt, question, Halo checkpoint, answer, retained history, transient query,
bytes, evidence, and seed fixed.

The ladder instantiates these depth/layout factors for construction testing.
The held-constant schedule control v2 additionally ran five schedules per case
with fixed 64-token terminal suffixes, 128 retained-history entries, and B2/B3
query access held equal. Across 11 B0-selected clusters it produced 110 cells
and 55 matched pairs. Realized round counts varied by prompt; the cluster-level
analysis, not the 110 rows, is the inferential unit. These settings remain
exploratory rather than registered.

## Case-generation contract

Every generated case has separate views:

1. **Inference:** prompt/messages, question, public metadata, and no answer or
   evidence annotation.
2. **Evaluation:** answer, accepted answers, exact evidence units/positions,
   distractors, stale facts, solver record, and cluster ID.

The views join only by case ID after execution. Generation validates unique IDs
and hashes, exact post-render token counts, independent solutions, annotation
presence, absence of oracle data from policy input, cluster-level split
integrity, and disjoint confirmatory/exploratory clusters.

## Existing ChatBubble assets

`Rineya_Long_Context_Demo_v1` has prompt-only and evaluation views and is useful
for compatibility smoke tests. The five completed family smokes each used the
same prompt-only row truncated to 192 tokens, finished at B48+C64, and decoded
no answer. That asset alone cannot support repeated-compression or long-context
quality claims.

The corrected bounded exploratory `halo_mini_eval_20260823_v2` decoded three
fixed early/middle/late cases from that asset. B0, B2, and B3 each scored 1/3;
only one case passed B0 and both compressed conditions were correct on it.
These preselected cases are development evidence, not independent confirmatory
clusters, and they do not resolve the pending exact evidence metric.

The generated [`halo_calibration_v3`](results/benchmarks/halo_calibration_v3)
artifact contains 32 separated inference/evaluation pairs with exact evidence
token positions. Full-cache Halo scored 18/32 (`R1=7/8`, `R2=0/8`, `R3=3/8`,
`R4=8/8`). The advance gate required at least two correct in each family and
therefore failed because R2 was 0/8. R2 is a benchmark-family label here, not a
Rineya/R2 model.

The follow-on `halo_retention_pilot_v2` used only 11 B0-eligible R1/R3/R4
clusters, capped at four per family. B2/B3/B4 answer counts were 5/6/1. The
strict diagnostic was 0/11 in every condition. The outcome-blind endpoint v1
was subsequently frozen, but its reuse on these same rows remains post hoc.
These calibration clusters cannot later be relabeled as independent
confirmatory data.

Prospective endpoint v1 used 24 entirely new clusters but stopped before
compression after B0 scored `9/24`. V2 changed difficulty before compressed
outcomes, scored B0 `18/24`, then froze four B0-correct clusters per R1/R3/R4
family in a timestamped local registration. Its 36 B2/B3/B4 cells are valid for
that registered endpoint contrast, but the difficulty-tuning history and modest
sample are part of the immutable claim limit. The v2 clusters also become
development data for later B-D reuse; those later architecture conditions were
not in the endpoint registration.

The 88-cell count study applied eight count transforms to the same 11 selected
clusters. Age-derived and head-averaged conditions were exact trajectory
equivalents of true realized C for every cluster under the fixed 128-token
schedule. Constant and shuffled endpoint deltas were nominally nonzero but not
Holm-significant; shuffled and wrong-token transforms are additionally
eligible-pool confounded because their global permutations cross protected
sink/recent slots (224/224 audited layer-head cells changed eligible-pool C
multisets). Delayed count is descriptive only. Its 264-row boundary
artifact scores only facts fully observed by each boundary.

Eligible-pool count v2 adds 33 cells on the same 11 development clusters. Its
two interventions preserve protected denominators plus eligible C and semantic-
ID multisets separately in all 22,176 layer/head applications. Both endpoint
comparisons are Holm-null. Same-process identity passes, but separately launched
v1/v2 CPU-bf16/SDPA trajectories do not reproduce bitwise; conclusions are
within-process only.

The 110-cell held-constant schedule study found a positive B3-B2 frozen-endpoint
interaction slope of `+0.00936074` per round (95% interval
`[0.00310399, 0.01759850]`, `p=0.00390625`) but no detected answer interaction
(`p=0.4375`). It is B0-selected, calibration-reusing, and exploratory.

The Experiment-B adaptive-budget screen replays three recorded boundaries for
11 clusters under uniform, entropy-proportional, inverse-concentration, and
shuffled-statistic allocations. It has 132 boundary-local rows but only 11
independent clusters. Every allocation conserves the exact 28,672-entry global
layer/head budget, and uniform replay matched runtime in 33/33 checks. The
primary-boundary specificity gate failed because shuffled-statistic allocation
(`+0.115260`, Holm `p=0.043945`) exceeded entropy-proportional (`+0.055195`,
Holm `p=0.214844`). This is a negative open-loop mechanism screen; it contains
no answer outcome and cannot substitute for closed-loop adaptive allocation.

The closed-loop Experiment-B quality screen reuses the 12 locally registered
v2 case IDs and frozen endpoint, but its three allocation conditions were not
registered. Its 36 cells therefore remain exploratory. Experiment C similarly
adds five attention-prior conditions for 60 cells, and Experiment D adds six
memory conditions for 72 cells. All preserve whole-cluster pairing and defer
answer/evaluation access until unscored artifacts are durable. Frozen case and
endpoint reuse improves comparability; it does not retroactively register new
conditions.

Experiments E and F reuse 11 earlier B0-selected clusters. E's open-loop risk
curve is resubstitution-only. F commits a no-retry SHA-256 split of whole
clusters into 5 train, 3 calibration, and 3 evaluation units before labels are
opened. Both classes appear in each split, but three held-out clusters yield
only two positive-negative AUC pairs. This is valid leakage-safe machinery and
an underpowered screen, not a sufficient learned-controller benchmark.

Training/development assets cannot become confirmatory test data. Generator
patterns may inform new cases only with new templates, entities, latent worlds,
clusters, and leakage checks.

## Public benchmarks and family expansion

Public subsets require license, scorer, contamination, and full-cache Halo
checks. They are labeled separately from controlled generated tasks. Secondary
published family cores may receive separately identified compatibility or
exploratory runs, but quality claims are not silently pooled into the Halo
primary result.

## Planned metrics

- exact/structured answer correctness and correctness conditional on B0;
- the frozen `layer-fact-retention-v1` endpoint plus its strict and partial-token
  sensitivity diagnostics;
- distractor and stale-evidence outcomes;
- score/count/age/evidence rank correlations;
- retained-set churn and head/layer agreement;
- retained history, active chunk, transient query, peak live entries, KV bytes,
  and total persistent bytes;
- prefill, decode, scoring, controller, recomputation, RSS, and GPU VRAM.

The current [literature audit](LITERATURE_NOVELTY_AUDIT.md) is a scoped prior-art
review and retains `NO_NOVELTY_CLAIM`. The 50-file Halo visualization program
comprises [28 controls](results/figures/halo_controls_v1) and
[22 deep-architecture panels](results/figures/halo_deep_architecture_v1), each
under a source-backed per-figure claim limit.

The runtime is a local recorder implementing audited semantics. Four audited
source primitives have exact parity; end-to-end source benchmark reproduction
is `NOT_RUN`. The locally registered endpoint-v2 run used Python 3.11.15,
Torch 2.13.0+cpu, and Transformers 5.15.0; later B-D screens and the current
lock use Torch 2.12.1+cu126 and Transformers 5.14.1. Every comparison is
interpreted inside its paired process, and endpoint v2 is not claimed as a
cross-runtime reproduction.
