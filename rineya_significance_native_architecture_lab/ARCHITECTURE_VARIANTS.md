# Architecture variants

The active architecture target is the published Qwen3-0.6B Rineya family,
starting with Halo. Rineya/R2 and Aegis models are excluded. Compatibility
smokes have separately passed on Halo, Halo Uncensored, Lyra, Nimbus, and Pulse;
no quality or cross-core generalization follows from them.

Runtime status: local recorder implementing audited semantics; 4/4 audited
source primitives match exactly and end-to-end source-runtime parity is
`NOT_RUN`. “Source-faithful” is therefore not an allowed description of the
current runtime.

The local endpoint-v2 result was paired under Torch 2.13.0/Transformers 5.15.0;
the current lock and Experiments B-D use 2.12.1/5.14.1. Architecture decisions
use within-run controls only; the endpoint result is not cross-runtime
reproduction evidence.

## Baselines and controls

| ID | Description | Role | Halo quality status |
|---|---|---|---|
| `B0` | full normal cache | capability reference | calibration 18/32 gate failed; endpoint v1 9/24 stop, difficulty-calibrated v2 18/24 pass; R7 v2 valid but all-YES 4/8 stop; R8 0/2 stop |
| `B1` | source-runtime Significance Matrix | parity-gated reference | inadmissible; only 4/4 primitive parity, end-to-end parity `NOT_RUN` |
| `B2` | raw accumulation, `e=0` | primary control | locally registered v2: 6/12 answers, endpoint 0.172619 |
| `B3` | observation normalization, `e=1` | primary treatment | locally registered v2: 5/12 answers, endpoint 0.212798; B3-B2 `+0.0401786`, `p=0.01953125` |
| `B4` | positional sink + recent | matched content-free control | locally registered secondary: 1/12 answers, endpoint 0.000000 |
| `B5` | deterministic age-derived exposure | matched causal control | exactly matched true-C trajectory on 11/11 fixed-schedule clusters |
| `B6` | shuffled significance/count identity | identity-breaking control | v1 global count version confounded; v2 eligible-pool shuffled/circular clean and Holm-null |
| `B7` | random selection | matched lower control | `NOT_RUN` |
| `B8` | oracle evidence retention | synthetic-only upper bound | `NOT_RUN` |

`B0` is not a storage-matched competitor. `B8` can never be described as
deployable. Model-free Tier-A control artifacts are under methodology review
and do not populate the Halo status column.

The three-case Halo mini evaluation found no measurable B3 gain over B2. The
larger development controls now show retained-state separation but still no
detected answer interaction. None changes the confirmatory statuses.

## Count-state and schedule findings

`layer-fact-retention-v1` was frozen outcome-blind. Applying it to the endpoint-
design pilot yielded B3-B2 `+0.061688` (`p=0.0078125`), which is post hoc and
B0-selected. A first new prospective set stopped at B0 (`9/24`). After explicit
difficulty calibration, a second set passed B0 (`18/24`) and a 12-cluster local
registration prospectively found B3-B2 `+0.0401786`, interval
`[0.0133929, 0.0758929]`, `p=0.01953125`. Answers moved the other way—B3 5/12,
B2 6/12, `p=1.0`. This is retained-state support under a local registration,
not broad confirmation or mediation.

The 88-cell count control establishes exact trajectory equivalence among true
C, deterministic age-derived C, and head-averaged C for the tested regular
128-token schedule. This does **not** mean these implementations are generally
equivalent; irregular observation opportunities are the condition under which
explicit realized exposure could add information. Constant and shuffled count
deltas were nominal but Holm-adjusted nonsignificant; delayed is descriptive.
Shuffled and wrong-token C are not clean controls because their permutations
cross protected sink/recent positions (224/224 audited layer-head eligible-pool
multisets changed).

V2 repairs that confound with eligible-only shuffled and circular permutations.
All 22,176 audited layer/head interventions preserved protected denominators
and eligible C/identity multisets with zero semantic fixed points. Both endpoint
contrasts were Holm-null (shuffled `p=0.4921875`; circular `p=0.25`). A same-
process true-C identity passed, while separately launched CPU-bf16/SDPA v1/v2
runs were not bitwise trajectory-reproducible, so the clean nulls are process-
local exploratory evidence.

The 110-cell schedule control holds the final pre-decode prompt cache and query access fixed
within 55 B3-B2 pairs. The frozen-endpoint interaction is `+0.00936074` per
additional compression round (95% interval `[0.00310399, 0.01759850]`,
`p=0.00390625`), while the answer interaction is not detected (`p=0.4375`).
These are exploratory development findings, not promotion criteria by
themselves.

## Capacity contract

Every bounded execution reports three distinct quantities:

```text
retained history + active chunk + transient query
```

The first is the compressed history carried between chunks. The second exists
during the current context forward. The third exists only for query-conditioned
scoring and must be rolled back. Persistent and peak-live byte accounting must
not collapse these into an ambiguous “slot budget.”

Each completed family smoke used **B48+C64**—48 retained-history entries and a
64-token active chunk—plus a ten-token transient query (32 configured maximum).
Each ended with 112 stored entries and recorded two scoring events per layer.
These are engineering gates, not policy comparisons.

## A — significance-native KV retention

Carry `S` and `C` with every surviving KV entry while preserving Qwen3 GQA
mapping, RoPE positions, causal masks, true sequence length, query rollback,
sink/recent protection, stable entry identity, and aligned gather. One case per
published family core verified this compatibility path. Matched retention was
subsequently evaluated on Halo only. The locally registered v2 endpoint
separates B3 from B2, but answers do not improve; broader confirmation,
cross-core quality, and retention-to-answer mediation remain `NOT_RUN`. Halo
remains the primary target.

## B — adaptive layer/head retained-history allocation

Allocate a fixed global retained-history byte allowance using entropy, Gini,
top-k mass, stability, and churn. Compare uniform, entropy-proportional,
inverse-concentration, inverse-stability, shuffled-statistic, and eventually a
tiny learned controller.

Allocation respects Halo's 16-query-head to 8-KV-head grouping and conserves
the exact global logical allowance. The original 132-row open-loop screen
failed specificity because shuffled-statistic allocation exceeded entropy.

The closed-loop adapter now carries unequal valid lengths per KV head inside
padded layer tensors. Its uniform path is bitwise identical, and a Halo smoke
passed masks, rollback, append, and exact logical-byte gates. The 36-cell
quality screen scored uniform/entropy/shuffled at `5/12`, `8/12`, and `5/12`;
entropy minus uniform answer `p=0.25`. Shuffled had the highest retention,
failing specificity again. Physical padding was about 9.7%–12.5% above logical
payload in adaptive arms, so this is not a ragged-memory saving. Status:
`CLOSED_LOOP_RUNTIME_PASS / QUALITY_SPECIFICITY_FAILED`.

## C — soft significance-aware attention

Add a bounded historical-significance prior:

```text
L'_ij = L_ij + lambda_layer * bounded_normalize(R_j)
```

Required controls include zero bias, normalized versus raw state, shuffled
state, and age-only bias. `lambda=0` must reproduce the ordinary Halo path
within the frozen numerical contract. The runtime smoke passed exact zero-bias
identity and bounded history-only bias. The 60-cell screen found all five arms
`5/12`; normalized, raw, and shuffled shared mean endpoint `0.21577` and the
specificity rule failed. Status:
`RUNTIME_PASS / NO_MEASURABLE_ANSWER_GAIN / SPECIFICITY_NOT_SUPPORTED`.

## D — significance memory slots

Compress evicted entries into a tiny fixed number of persistent slots. Compare
zero slots, mean pooling, recency weighting, raw-score weighting,
normalized-score weighting, random weighting, and a frozen-backbone tiny
projection. Slot bytes count toward total persistent state. `M=0` is bitwise
identical to hard eviction; `M=4` adds exactly 491,904 bytes. The 72-cell total-
capacity screen used M0 B128 versus M4 B123 plus slots. All six arms scored
`5/12`; normalized endpoint `0.183036` was below M0 `0.211310` and random
`0.191964`. Status: `RUNTIME_PASS / NEGATIVE_OR_NEUTRAL_FOR_TESTED_M4`.

## E — significance-controlled recovery

Use pre-frozen uncertainty diagnostics to choose between the compressed result
and a bounded recovery action such as a larger temporary history allowance,
CPU-side block restore, recomputation, or full-cache verification. Evaluate
risk/coverage and average compute. The fixed answer-blind six-feature score had
error ROC AUC `0.50` on 11 selected clusters. A live smoke nevertheless proved
literal accept identity and safe one-shot full-cache recomputation. Status:
`MECHANISM_PASS / RISK_SIGNAL_NO_MEASURABLE_GAIN`.

## F — tiny learned controller

Candidate inputs include significance entropy/concentration, top margins, age
distribution, head/layer agreement, churn, observation counts, and output
uncertainty. Logistic regression, isotonic rules, shallow trees, or a tiny MLP
are eligible; backbone weights remain frozen. Splits must be by latent-case
cluster. The first run froze a no-retry SHA-256 5/3/3 split. A seven-parameter
logistic model achieved held-out AUC/BA `1.0` and recovered both errors, but
evaluation had only three clusters and two positive-negative pairs; the tree
control collapsed from perfect training to held-out AUC/BA `0.5`. Status:
`UNDERPOWERED_OVERFIT_SCREEN_ONLY`; high overfit risk, Finding D unestablished.

## Feature gates

Every component must remain independently disableable. Disabled-path identity,
zero-bias identity, zero-slot identity, exact rollback, and persistent/peak-live
byte conservation are release-blocking tests.
