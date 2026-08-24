# Confirmatory protocol

## Registration status

| Field | Value |
|---|---|
| Protocol version | `draft-5` |
| Broad protocol status | `DRAFT_NOT_REGISTERED` |
| Registered date | none |
| Primary target | published `Rineya-Halo-1.2.0-devfreeze` |
| Resolved-config template | `configs/confirmatory.yaml` |
| Confirmatory results | `NOT_RUN` |
| Narrow endpoint validation | `halo-endpoint-validation-v2-registered`: local prospective execution complete |

This document is a design draft, not a preregistration. The strict exact-token
Boolean was rejected as primary because it saturated at zero. Its outcome-blind
replacement, `layer-fact-retention-v1`, has completed one narrow local
prospective registration after a failed B0-only attempt and explicit difficulty
calibration. That local record does not register this broad protocol. The final
cluster/sampling matrix, task-family coverage, runtime replication, and minimum-
information rule remain unresolved. Running the template now would produce
exploratory evidence only.

Rineya/R2 and Aegis are excluded as active model targets. RineyaBeta supplies
read-only mechanism provenance only.

One-case compatibility smokes have passed on Halo, Halo Uncensored, Lyra,
Nimbus, and Pulse. Those five engineering gates are not confirmatory outcomes.
This protocol's primary model remains Halo; secondary-core smokes or later
evaluations are neither pooled nor substituted into the Halo primary analysis.

The completed `halo_mini_eval_20260823_v2` run is bounded exploratory evidence:
B0, B2, and B3 each scored 1/3, with B2 and B3 both 1/1 on the sole B0-eligible
case. Because two cases failed the capability gate, the sample is preselected,
and no exact evidence metric or inferential procedure was frozen, this result
cannot register or validate the confirmatory endpoint. Its cases must not be
silently reused as independent confirmatory clusters.

The newer calibration sequence is also development evidence. The 32-cluster
`halo_calibration_v3` dataset has exact rendered-prompt evidence positions, but
full-cache Halo scored 18/32 with `R1=7/8`, `R2=0/8`, `R3=3/8`, and `R4=8/8`.
The advance gate failed because it required at least two correct in every
benchmark family. R2 here is a task-family label, not a Rineya/R2 model.

`halo_retention_pilot_v2` selected 11 B0-correct R1/R3/R4 clusters. B2/B3/B4
answer counts were 5/6/1. The strict diagnostic was 0/11 in every condition.
Endpoint v1 was chosen using evidence fields only and passed its non-saturation
screen, but reapplying it to these same rows is post hoc: B3-B2 `+0.061688`,
95% interval `[0.030844, 0.094156]`, `p=0.0078125`. The one B3 answer win with
no loss had `p=1.0`. None of those pilot clusters may be silently reused as
independent confirmation data.

The first new prospective dataset kept endpoint v1 fixed but failed the B0
advance rule at `9/24`; no B2/B3/B4 run followed. Difficulty-calibrated v2 then
passed B0 at `18/24`. Before any compressed model cell, a timestamped local
registration froze 12 B0-correct clusters (four per R1/R3/R4), the endpoint,
the B3-B2 primary contrast, and cluster-level procedures. B3-B2 was
`+0.0401786`, interval `[0.0133929, 0.0758929]`, `p=0.01953125`. Answers did
not improve: B3 `5/12`, B2 `6/12`, one loss/no wins, `p=1.0`. This validates a
narrow retained-state contrast, not this broad protocol, answer mediation, or
the untuned task population.

The 1,176-case R1-R8 ladder is a build artifact with 84 independent latent
clusters, not a model result. A bounded eight-row R5-R8 B0 screen produced raw
R5 `2/2`, R6 `1/2`, R7 `2/2`, and R8 `0/2`. R7 v1 is invalid/uninformative due
to label skew, selected all-NO seeds, a rendered lock-sentence shortcut, and
overbroad required-fact annotation; corrected R7 v2 is described below. The currently valid
R5/R6/R8 subset is 3/6 and R8 stops at capability. No compression policy was
tested. The 88-cell count study and 110-cell schedule
study also reuse the selected R1/R3/R4 development clusters. Age-derived and
head-averaged counts exactly reproduced true-C trajectories in the fixed
schedule; constant/shuffled nominal deltas were Holm-adjusted nonsignificant
and delayed is descriptive. Shuffled/wrong-token are eligible-pool confounded
because their permutations cross protected sink/recent positions. The schedule frozen-endpoint interaction was
`+0.00936074`/round (`[0.00310399, 0.01759850]`, `p=0.00390625`) while the
answer interaction was not detected (`p=0.4375`). These are protocol-design
evidence only.

Corrected R7 v2 is balanced and solver-audited, but Halo returned all `YES` on
its frozen 4-YES/4-NO screen. The 4/8 total with 0/4 NO fails the prospective
two-sided gate, so no R7 compression condition ran. Valid benchmark repair is
not capability success.

Eligible-pool count v2 repairs the earlier shuffled/wrong-token confound. Its
shuffled and circular interventions preserve protected denominators and
eligible C/identity multisets in all 22,176 audited layer/head applications;
both endpoint contrasts are Holm-null. Same-process identity passed, but
separate v1/v2 CPU-bf16/SDPA trajectories were not bitwise reproducible, so the
results remain process-local development evidence.

The 132-row adaptive-budget screen is also development-only. It conserves the
global entry budget and passes uniform parity, but shuffled-statistic allocation
exceeds entropy-proportional allocation at the primary boundary; the decision
is `NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE`. It is open loop, loads no model,
generates no answers, and does not measure sparse-storage overhead.

Closed-loop Experiments B-D reuse the 12 local endpoint-v2 case IDs but add
conditions that were not in that registration. B's entropy arm scored 8/12
versus uniform 5/12 (`p=0.25`), while shuffled retained more and padded physical
storage exceeded logical payload. C's five prior arms all scored 5/12 and
normalized endpoint equaled raw/shuffled. D's six matched-capacity arms all
scored 5/12 and normalized memory was below M0/random on exact-token retention.
All three fail their architecture-specific promotion rules.

Experiment E verified accept/full-recompute actions, but its fixed risk signal
had AUC 0.50. Experiment F's frozen 5/3/3 split yielded perfect logistic metrics
on only three held-out clusters/two positive-negative pairs; it is classified
`UNDERPOWERED_OVERFIT_SCREEN_ONLY` with high overfit risk, not Finding D. None of B-F is
incorporated into this draft primary contrast.

## Research question

On the published Halo core, does the locally observed B3-over-B2 retention
contrast replicate under a materially larger, diverse, frozen-difficulty and
byte-matched protocol, and is any retention change accompanied by
capability-conditioned answer change?

The draft broad contrast direction remains `B3 - B2`. A narrow local
registration has already evaluated it, but this protocol is not registered.
Its future primary hypothesis test must use clusters outside endpoint design,
local validation, and later B-D reuse, with the task and runtime contract frozen
before outcomes.

## Target identity

Primary checkpoint:

```text
Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze
rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4
```

Expected architecture: `Qwen3ForCausalLM`, 28 layers, hidden size 1024,
intermediate size 3072, 16 query heads, 8 KV heads, head dimension 128,
vocabulary 151,936, native maximum positions 40,960.

Expected SHA-256 values:

- weights: `38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544`;
- tokenizer: `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`;
- config: `2284cc1687a4f4f6e3a64a13dfcf7562e592b951afa2f91244cc7b212c9cfb85`.

No silent checkpoint substitution is allowed. Secondary-core quality
evaluations need separately frozen identities and analyses; they cannot be
pooled into the Halo primary result after outcomes are known.

## Runtime provenance boundary

Runtime status: local recorder implementing audited semantics. Exact parity has
passed for four audited primitives (bottom-right causal mask, post-RoPE GQA
score, count-lag gather, and tie-free sink/heavy/recent selection), but
end-to-end source benchmark reproduction is `NOT_RUN`. Identity/rollback smokes
passed separately on all five published family cores. Neither result promotes
the local runtime to a complete source-faithful reproduction. A `B1`
source-runtime condition is inadmissible unless a separate end-to-end parity
protocol runs.

The local endpoint-v2 execution was internally paired under Python 3.11.15,
Torch 2.13.0+cpu, and Transformers 5.15.0. The current lock and later B-D
screens use Torch 2.12.1+cu126 and Transformers 5.14.1. No cross-version
reproduction has been established; a broad registration must freeze one
runtime and rerun its identity gates.

## Conditions

| ID | Policy | Role |
|---|---|---|
| `B0` | full normal cache | Halo capability reference; not memory matched |
| `B2` | raw cumulative score, `e=0` | primary control |
| `B3` | realized-count normalization, `e=1` | primary treatment |
| `B4` | sink plus recent positional retention | matched content-free control |
| `B5` | deterministic age-derived exposure | tests whether realized `C` adds information |
| `B6` | seeded shuffled significance | intended identity break; must preserve the eligible heavy pool |
| `B7` | seeded random retention | matched lower control |
| `B8` | oracle evidence retention | synthetic-only upper bound; never runtime deployable |

All compressed conditions use identical prompt rendering, query access,
compression boundaries, decoding, and random seeds. Score pooling, attention
prior, memory slots, and learned controller are disabled.

## Budget contract

Never use a single ambiguous “cache budget.” Report separately:

```text
retained history + active chunk + transient query
```

The draft planning values are 1,024 retained-history entries, a 512-token
active chunk, and at most 32 transient query tokens. Therefore the draft maxima
are 1,536 stored entries after a full active chunk and 1,568 entries during a
query probe. These values remain provisional until protocol freeze.

`B2` carries the same count allocation as `B3` even though ranking ignores it.
Content-free controls reserve equivalent auxiliary bytes or reduce retained
history so their total persistent state does not exceed the treatment. KV-only
and total-persistent-state bytes are both reported. Transient workspaces are
reported separately from persistent state.

## Cluster-aware benchmark design

The controlled task ladder now instantiates generated stages `R1`–`R8`; here
`R2` is a benchmark stage label, not a Rineya/R2 model. Cases derived from the same latent world,
template instance, or generator seed are not independent.

Before registration, every case must receive a stable `cluster_id`. All policy,
evidence-depth, and compression-schedule variants derived from one latent case
remain in the same cluster. Train/calibration/evaluation splits occur by
cluster, and uncertainty is computed by resampling or randomizing whole
clusters. The number of independent clusters—not rendered prompts—is the
effective sample size.

The calibration implementation defines `cluster_id = family + latent seed`;
every depth/schedule derivative stays in that cluster. It produced 32 clusters,
eight per R1-R4 family. The draft primary procedure is a two-sided cluster
sign-flip test on cluster-mean paired differences with a 10,000-resample cluster
bootstrap interval. Calibration required at least 12 B0-eligible clusters
overall and at least two per family. Although 18/32 were B0-correct overall,
the gate failed because R2 was 0/8. The task-family repair/replacement rule,
final case matrix, and pilot-derived variance use remain pending before freeze;
R2 cannot simply be dropped after observing this result. The larger ladder has
1,176 rows but only 84 independent latent clusters. Its eight-cluster R5-R8 B0
screen also found R8 `0/2`, while R7 v1 was invalid; neither can be advanced
post hoc. Corrected R7 v2 is valid but failed its own balanced gate after Halo
answered YES on all eight (`4/8`, `0/4` NO), so it also cannot advance.

## Primary estimand pending registration

The direction of the intended paired estimand is:

```text
exact evidence outcome under B3 - exact evidence outcome under B2
```

The first exact-token outcome is
`REJECTED_AS_PRIMARY_SATURATED`: requiring all required fact tokens in every
evaluated layer yielded 0/11 under B2, B3, and B4. The replacement outcome is
frozen as `layer-fact-retention-v1`, the macro-mean of complete fact-retention
indicators across layer/fact cells. Selection was outcome-blind, but it reused
the pilot evidence rows. A separate local registration has now validated the
endpoint on 12 difficulty-calibrated clusters. Task-answer exact correctness,
reported unconditionally and conditional on `B0` success, remains a secondary
outcome and did not improve there. The broad protocol still needs its own
registered test, multiplicity family, and minimum sample rule.

The local p-value and interval are valid only for their local registration. No
statistic becomes evidence for this broad protocol until its endpoint, cluster
plan, runtime, and analysis are registered together.

## Capability gate and exclusions

If `B0` fails, the case is labeled `FULL_CACHE_CAPABILITY_FAILURE` and cannot
support a compression-conditioned answer claim. It remains in unconditional
summaries.

The observed R2 and R8 capability failures are design results, not permission
for post-hoc family exclusions. R7 v1 is a separate task-design invalidity;
R7 v2 repairs it but reveals a balanced B0 capability failure. A future
protocol must prospectively repair or replace incapable task-family slices,
rerun B0 on new clusters, and freeze any family eligibility rule before
compressed-condition outcomes.

A case may be excluded only for a frozen pre-answer failure: generator/solver
failure, hash mismatch, token-contract violation, invariant misalignment,
non-finite values, or an unrecoverable interrupted run. Wrong answers, evidence
eviction, decode failure, and `UNKNOWN` outcomes are not exclusions.

## Registration checklist

Before changing the status from `DRAFT_NOT_REGISTERED`:

1. define a materially larger replication split excluded from metric design,
   both endpoint-validation attempts, and all later architecture screens;
2. freeze stable cluster IDs, split rules, resampling/test procedure, and
   minimum independent-cluster requirements;
3. freeze the case matrix and hashes without inspecting condition outcomes;
4. freeze retained-history, active-chunk, transient-query, and byte contracts;
5. record the Halo hashes and runtime environment;
6. pass identity, rollback, alignment, leakage, and matched-byte gates;
7. timestamp and immutably preserve the registered protocol/config.

The 11 retention-pilot clusters, all 32 calibration clusters, both endpoint-
validation datasets, their local registered cells, 88+33 count cells, 110
schedule cells, 132 adaptive-allocation rows, B-D quality cells, E/F controller
rows, and all ladder derivatives sharing those latent worlds are development
data for the broad protocol. They are excluded from its independent split.

Any later change is an amendment or exploratory run, never a silent replacement.
