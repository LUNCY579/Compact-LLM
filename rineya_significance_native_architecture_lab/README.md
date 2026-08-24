# Rineya Cornerstone-1 Significance-Native Architecture Lab

This standalone, CPU-first lab applies the Cornerstone-1 Significance Matrix
mechanism to the **published Qwen3-0.6B Rineya core family** in this repository.
The primary empirical target is **Rineya Halo**
(`Rineya-Halo-1.2.0-devfreeze`). One-case compatibility smokes have also run on
Halo Uncensored, Lyra, Nimbus, and Pulse. Each core keeps a separate identity;
secondary smoke results are not pooled into the Halo evidence conclusion.

Rineya/R2 checkpoints and Aegis are outside the active experiment scope. The
separate `RineyaBeta` repository is read-only mechanism provenance; its R2-era
implementation and results are not evidence about Halo or this local runtime.

## Current status

| Workstream | Status | Evidentiary scope |
|---|---|---|
| Audited source-primitive parity | `PASS` on 4/4 checks | exact primitive tensors/indices only; not end-to-end reproduction |
| Published Qwen3-0.6B family CPU smokes | `PASS` on 5/5 cores | one case per core; runtime compatibility only |
| Halo calibration dataset v3 | `CALIBRATION_DATASET_BUILT` | 32 independent exact-evidence clusters; construction/alignment only |
| Halo full-cache capability calibration v2 | `PASS` execution; advance gate `FAILED` | 18/32 B0-correct; benchmark-family R2 was 0/8 |
| Halo endpoint prospective v1 | `STOPPED_AT_B0_GATE` | 9/24 B0-correct; no compressed condition ran |
| Halo endpoint prospective v2 | `LOCALLY_REGISTERED_VALIDATION_COMPLETE` | difficulty-calibrated B0 18/24; frozen 12-cluster B2/B3/B4 result; not a public preregistration |
| Halo R1-R8 controlled ladder v1 | `TIER_A_LADDER_BUILT` | 1,176 cases in 84 independent clusters; construction/alignment only |
| Halo R5-R8 / corrected R7 capability | R7 v1 invalid; R7 v2 valid but `STOPPED_AT_B0_GATE`; R8 stopped | R7 v2 balanced 4/8 with all-YES outputs; no R7 compression policy |
| Halo retention pilot v2 | `EXPLORATORY_POSTHOC` | 11 B0-selected clusters; frozen-endpoint reuse, no confirmatory or population claim |
| Real-Halo count callback identity | `PASS` | one case; standard B3 and true-count callback exact on tokens/residents/accounting only |
| Halo count causal controls v1 | `COMPLETE_EXPLORATORY` | 88 matched cells; age-derived and head-averaged counts exactly reproduced true-C trajectories for this fixed schedule |
| Halo eligible-pool count controls v2 | `COMPLETE_EXPLORATORY_CLEAN_CONTROLS` | 33 cells; shuffled/circular interventions preserve eligible C multisets; both endpoint contrasts null after Holm |
| Halo held-constant schedule control v2 | `COMPLETE_EXPLORATORY` | 110 cells / 55 matched B3-B2 pairs; positive frozen-endpoint interaction, no answer interaction |
| Halo adaptive-budget screen v1 | `NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE` | 132 boundary-local open-loop rows; shuffled-statistic control exceeded entropy; no model answers |
| Experiment B heterogeneous budgets | `RUNTIME_PASS`; `QUALITY_SPECIFICITY_FAILED` | 36 closed-loop cells; entropy 8/12 vs uniform 5/12 (`p=0.25`), but shuffled retained more and padded storage was larger |
| Experiment C soft attention prior | `RUNTIME_PASS`; `NO_MEASURABLE_ANSWER_GAIN` | 60 cells; all conditions 5/12; normalized endpoint matched raw/shuffled and failed specificity |
| Experiment D significance memory | `RUNTIME_PASS`; `NEGATIVE_OR_NEUTRAL` | 72 byte-matched cells; all conditions 5/12; normalized memory retained less exact evidence than M0/random |
| Experiment E controlled recovery | mechanism `PASS`; risk signal `NO_MEASURABLE_GAIN` | fixed answer-blind risk AUC 0.50; live accept and full-recompute branches verified |
| Experiment F tiny learned controller | `UNDERPOWERED_OVERFIT_SCREEN_ONLY` | high overfit risk; frozen 5/3/3 split; held-out n=3 logistic AUC/BA 1.0 but only two positive-negative pairs; no advancement |
| Halo figure sets | `PASS` | 50 manifest-verified figures (28 controls + 22 deep architecture) with per-panel claim limits |
| Literature/novelty audit | `NO_NOVELTY_CLAIM` | scoped audit completed; experimental distinction only |
| Broad Halo confirmatory protocol | `DRAFT_NOT_REGISTERED`; `NOT_RUN` | local endpoint-validation registration is narrower and does not register this protocol |
| Halo bounded mini evaluation | `PASS` execution; `NO_MEASURABLE_GAIN` descriptively | 3 fixed exploratory cases; no inferential statistics |
| Halo complete showcase-64 evaluation | `PASS` execution; B3-over-B2 `NULL_AT_CLUSTER_LEVEL` | all 64 showcase cases; B0 30/64, B3 19/64, B2 17/64; 16-pack contrast `p=1.0`; unregistered |
| Tier-A synthetic/open-loop run | `INVALID_OR_CONFOUNDED` | config/execution mismatch; machinery audit only |
| GPU confirmation | `NOT_RUN` | no evidence |

Runtime status: local recorder implementing audited semantics; audited
source-primitive parity `PASS`, end-to-end source benchmark reproduction
`NOT_RUN`. [`results/source_parity.json`](results/source_parity.json) records
exact equality for the bottom-right causal mask, post-RoPE GQA score,
count-lag gather, and tie-free sink/heavy/recent selection primitives. This
narrow result does not mean the whole source runtime was reproduced.

## Completed published-family smokes

The five completed bf16 CPU runs are:

- [`halo_cpu_smoke_20260823_v2`](results/runs/halo_cpu_smoke_20260823_v2) — primary
  Halo core;
- [`halo_uncensored_cpu_smoke_20260823`](results/runs/halo_uncensored_cpu_smoke_20260823);
- [`lyra_cpu_smoke_20260823`](results/runs/lyra_cpu_smoke_20260823);
- [`nimbus_cpu_smoke_20260823`](results/runs/nimbus_cpu_smoke_20260823);
- [`pulse_cpu_smoke_20260823`](results/runs/pulse_cpu_smoke_20260823).

Each loaded its expected published-core hashes and executed the same truncated,
prompt-only 192-token compatibility case.

- Model, tokenizer, and config hashes were verified for every core. Halo's
  values were `38fb96fe...36544`, `be756060...5506`, and
  `2284cc16...cfb85`.
- The disabled chunked path and the single-shot reference produced exactly
  identical recorded logits on every core's case: maximum and mean absolute
  delta `0.0`, argmax agreement `1.0`.
- The bounded path represented 192 true positions with 112 stored states after
  the final chunk: 48 retained-history entries plus the active 64-token chunk.
- A transient ten-token query probe was rolled back exactly and is not part of
  the 112 stored states.
- Every one of the 28 layers recorded two query-conditioned scoring events in
  every run.
- No answer was decoded by any smoke. The runs make no accuracy, reasoning-quality,
  matched-policy, deployment-latency, or GPU claim.

Budget reports use three separate terms:

```text
retained history + active chunk + transient query
```

For every family smoke these were **B48+C64**: 48 retained-history entries plus
the 64-token active chunk, with up to 32 transient query entries configured
(ten used). “48-slot cache” is therefore not an accurate description of live
execution.

## Completed bounded Halo mini evaluation

[`results/runs/halo_mini_eval_20260823_v2`](results/runs/halo_mini_eval_20260823_v2)
ran bf16 manual-greedy decoding on three fixed Halo cases under B0 full cache,
B2 raw accumulation, and B3 observation normalization.

| Condition | Exact correctness | B0-conditioned correctness |
|---|---:|---:|
| B0 | 1/3 | not applicable |
| B2 | 1/3 | 1/1 |
| B3 | 1/3 | 1/1 |

The full-cache capability gate failed on 2/3 cases. B3 showed no measurable
gain over B2 in this bounded slice; both were correct on the sole B0-eligible
case. With three preselected cases, one eligible case, no inferential
statistics, and no frozen exact-evidence metric, this is descriptive exploratory
evidence only—not a positive, equivalence, family-wide, or confirmatory claim.
The answer key was opened only after all inference and the unscored prediction
artifact had been written.

## Completed Halo showcase-64 evaluation

[`halo_showcase64_eval_v1`](results/runs/halo_showcase64_eval_v1) is the first
Cornerstone-1 run to cover the **complete** 64-case showcase split (16 scenario
packs x four variants), rather than the three base-variant cases of the mini
evaluation. It reuses `run_published_core_eval.execute_condition` unchanged, so
both runs share one audited execution path. `run_showcase64_eval.py` is a
sibling script; the three-case evaluator and its recorded artifact are untouched.

| Condition | Strict exact | unanswerable |
|---|---:|---:|
| B0 full cache | 30/64 = `0.469` | 2/16 |
| B3 `e=1` | 19/64 = `0.297` | 0/16 |
| B2 `e=0` | 17/64 = `0.266` | 1/16 |

B3's two-answer edge over B2 is null at the correct unit: 4 case-level wins and
2 losses (`p=0.6875`) collapse to 2 packs better, 2 worse, and 12 tied
(`p=1.0`), because the four wins fall in only two scenario packs. Compression
cost 17.2 accuracy points while holding a median 32.0% of B0's entries and
33.7% of its KV bytes. Halo's full-cache `0.469` compares with `0.219` for the
recorded `qwen3_06b_showcase` baseline on identical cases, key, and scorer.
The run is exploratory and unregistered: no frozen endpoint, no advance gate,
no inferential claim, and no published-family result.
See [SHOWCASE64_EVAL_REPORT.md](SHOWCASE64_EVAL_REPORT.md).

## Halo calibration and retention pilot

[`halo_calibration_v3`](results/benchmarks/halo_calibration_v3) contains 32
independent `R1`-`R4` calibration clusters with inference/evaluation separation
and exact rendered-prompt evidence-token positions. It is a dataset artifact,
not a model outcome. Full-cache Halo then scored 18/32 in
[`halo_capability_calibration_v2`](results/runs/halo_capability_calibration_v2):
`R1=7/8`, `R2=0/8`, `R3=3/8`, and `R4=8/8`. The predeclared advance rule
required at least 12 correct overall and at least two in every benchmark family;
it failed because R2 had no eligible cluster. Here R2 is a benchmark-family
label, not a Rineya/R2 checkpoint.

[`halo_retention_pilot_v2`](results/runs/halo_retention_pilot_v2) consequently
used 11 B0-eligible development clusters from R1, R3, and R4, capped at four per
family. Exact-answer counts were B2 `5/11`, B3 `6/11`, and B4 `1/11`. The one
B3-over-B2 answer win with no loss had a two-sided exact cluster sign-flip
`p=1.0` and supports no answer-quality claim.

The candidate strict all-token endpoint was `0/11` under each of B2, B3, and
B4: it requires every required fact token to be available in every evaluated
layer. This all-zero saturation rejects that strict Boolean as a primary
endpoint for the current implementation and budget.

An outcome-blind calibration subsequently froze
[`layer-fact-retention-v1`](EVIDENCE_ENDPOINT_CALIBRATION.md). Its primary value
is the macro-mean of complete fact-retention indicators over layer/fact cells.
Endpoint selection pooled all 33 pilot evidence records without reading
condition labels, predictions, correctness, accepted answers, failure labels,
or task family. Applying the new endpoint back to the same endpoint-design
pilot was necessarily post hoc: B3 minus B2 was `+0.061688` (95%
cluster-bootstrap interval `[0.030844, 0.094156]`, exact cluster sign-flip
`p=0.0078125`).

Two prospective attempts then kept the endpoint definition fixed. The first
24-cluster dataset stopped at its predeclared B0 gate after Halo solved only
`9/24`; no compressed condition ran. Difficulty-calibrated v2 changed the task
templates before outcomes, not the endpoint: Halo solved `18/24`, allowing 12
B0-correct clusters (four each R1/R3/R4) to enter a timestamped local
registration. In the registered B2/B3/B4 execution, mean endpoint values were
`0.172619`, `0.212798`, and `0.000000`. The sole primary contrast was:

```text
B3 - B2 = +0.0401786
95% cluster-bootstrap interval = [0.0133929, 0.0758929]
exact two-sided cluster sign-flip p = 0.01953125
```

This prospectively supports a narrow retained-state separation on the locally
registered 12-cluster sample. It does not support answer improvement: B3 was
`5/12`, B2 `6/12`, with one B3 loss, no wins, and `p=1.0`. The templates were
difficulty-tuned after the v1 capability failure, the registration is local
rather than public, and the broad confirmatory protocol remains unregistered.
The run used Python 3.11.15, Torch 2.13.0+cpu, and Transformers 5.15.0; the
current lock and B-D screens use Torch 2.12.1+cu126 and Transformers 5.14.1.
It is internally paired but not reproduced under the locked runtime.
See [ENDPOINT_VALIDATION_V2_RESULT.md](ENDPOINT_VALIDATION_V2_RESULT.md).

## Controlled ladder and full-cache screen

[`halo_r1_r8_ladder_v1`](results/benchmarks/halo_r1_r8_ladder_v1) contains
1,176 separated inference/evaluation cases in 84 independent latent clusters.
It covers R1-R8, seven evidence depths, and fixed/clustered layouts; its exact
evidence-token alignment is a dataset result only. R1 contributes 490 cases
across five required-fact counts, and each of R2-R8 contributes 98 cases. Here
and throughout the lab, `R2` is a benchmark-family label, not a Rineya/R2 model.

The bounded full-cache screen
[`halo_r5_r8_capability_screen_v1`](results/runs/halo_r5_r8_capability_screen_v1)
opened answers only after inference. Its raw artifact reports R5 `2/2`, R6
`1/2`, R7 `2/2`, and R8 `0/2`, but the R7 v1 task is invalid/uninformative:
its seven ladder seeds are label-skewed 6 NO / 1 YES, both screened seeds are
NO, the rendered prompt exposes an exact lock-sentence shortcut, and all seven
facts were marked required. R7's raw 2/2 is therefore not Halo capability and
must be replaced by a corrected task. Among the six currently valid R5/R6/R8
screen rows, the descriptive raw count is 3/6; R8 remains 0/2 and stops at the
capability gate. No compression policy was evaluated.

[`halo_r7_v2`](results/benchmarks/halo_r7_v2) repairs the R7 design with 80
balanced cases across eight independent clusters and counterfactual proof
checks. Its frozen eight-cluster screen was exactly `4 YES / 4 NO`. Halo
answered `YES` on all eight, scoring `4/8`: `4/4` on YES and `0/4` on NO. The
prospective gate required at least one correct answer on each side, so R7 v2
stopped at B0. This validates the benchmark repair but supplies no R7
Significance Matrix result. See [R7_V2_REPORT.md](R7_V2_REPORT.md).

## Completed count and schedule controls

[`halo_count_causal_controls_v1`](results/runs/halo_count_causal_controls_v1)
completed all 88 planned cells: 11 B0-selected R1/R3/R4 clusters under true,
constant, age-derived, shuffled, wrong-token, delayed, noisy, and head-averaged
count transforms. Query access and the 128-entry retained-history budget were
matched. Age-derived and head-averaged counts reproduced the true-C effective
counts, scores, selections, resident positions, predictions, frozen endpoint,
and correctness exactly in all 11 clusters. This exact equivalence is confined
to the tested fixed schedule; it shows that explicit granular C added no
information beyond deterministic age there, not that C is generally redundant.

Before interpreting those controls, the one-case
[`halo_count_control_identity.json`](results/halo_count_control_identity.json)
engineering check compared the standard Halo B3 compressor with the true-count
callback in the same process. Generated token IDs, every resident-position
array, accounting, policy, and exponent matched exactly. This validates that
callback path on one real-Halo case; it is not a quality or population result.

Constant and shuffled counts reduced the frozen endpoint by `-0.055195` and
`-0.058442` versus true C. Their nominal `p` values were `0.015625` and
`0.046875`, but Holm-adjusted values were `0.109375` and `0.28125`; neither is
multiplicity-adjusted evidence. Moreover, shuffled and wrong-token transforms
permuted C globally across the protected four-sink/32-recent partition. On the
audited case, the eligible-heavy-pool C multiset differed in all 224 layer-head
cells, so these are
`GLOBALLY_DISTRIBUTION_PRESERVING_BUT_ELIGIBLE_POOL_CONFOUNDED`, not clean
identity controls. Delayed count was descriptively higher
(`+0.063312`, nominal `p=0.1171875`, Holm `0.5859375`) and supports no directional
claim. Boundary evidence contains 264 rows and never counts future or
boundary-crossing facts as evicted.

[`halo_count_causal_controls_v2`](results/runs/halo_count_causal_controls_v2)
repairs the v1 identity-control confound. Its eligible-shuffled and eligible-
circular interventions leave the four sink and 32 recent denominators fixed,
preserve the eligible heavy-pool C and semantic-ID multisets separately in
every layer/head, and have zero semantic fixed points. The 33-cell run passed
all 22,176 layer/head invariant checks. Neither clean endpoint contrast versus
true C survived the two-control Holm family: shuffled `-0.021104`, nominal and
Holm `p=0.4921875`; circular `-0.014610`, nominal `p=0.125`, Holm `p=0.25`.
Answer contrasts were also not detected. A same-process true-C identity replay
was exact, but v1/v2 reference traces produced by separate CPU-bf16/SDPA
processes were not bitwise reproducible (10/11 predictions, 9/11 endpoint
values, and 0/11 resident trajectories exact). The paired v2 conclusions are
therefore process-local and exploratory; no cause for the cross-process drift
is asserted.

[`halo_schedule_control_v2`](results/runs/halo_schedule_control_v2) completed
110 cells: 11 clusters x five schedules x B2/B3, yielding 55 matched method
pairs. It held final pre-decode prompt-cache size and within-pair query access
constant while varying realized compression rounds. For the frozen endpoint, the cluster-level
B3-minus-B2 interaction slope was `+0.00936074` per additional round (95%
interval `[0.00310399, 0.01759850]`, `p=0.00390625`). The answer interaction was
not detected: slope `+0.02248157`, interval `[-0.03599509, 0.07641278]`,
`p=0.4375`. Both analyses reuse B0-selected development clusters and remain
exploratory/non-confirmatory.

[`halo_adaptive_budget_screen_v1`](results/runs/halo_adaptive_budget_screen_v1)
is a Tier-A boundary-local replay of Experiment B, not closed-loop execution.
It evaluates 11 clusters x three recorded boundaries x four allocation
strategies (132 rows) under an exact 28,672-entry global layer/head budget.
Uniform allocation reproduced the recorded runtime selection in 33/33 boundary
checks. At the primary boundary, entropy-proportional and inverse-concentration
deltas versus uniform were `+0.055195` (`p=0.107422`, Holm `0.214844`) and
`+0.025974` (`p=0.345703`, Holm `0.345703`). Shuffled-statistic allocation was
`+0.115260` (`p=0.014648`, Holm `0.043945`) and exceeded entropy, so the
specificity gate failed: `NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE`. No model was
loaded, no answer was generated, earlier allocation changes were not propagated
through hidden states, and sparse physical-storage overhead was not measured.

## Architecture Experiments B-F

Experiment B progressed from that negative open-loop replay to a real closed-
loop padded runtime. The uniform padded path is bitwise identical to the
rectangular path, and a Halo smoke verified exact logical-budget conservation,
query rollback, and unequal head/layer allocations. The 36-cell quality screen
then compared uniform, live entropy, and shuffled-statistic allocations on 12
frozen clusters. Answers were `5/12`, `8/12`, and `5/12`; entropy minus uniform
was `+0.25` with `p=0.25`. The retention-specificity gate failed because
shuffled had the largest mean endpoint (`0.23810`, versus entropy `0.22321` and
uniform `0.21131`). Logical payload was matched, but padding made adaptive
physical storage about 9.7%–12.5% larger than logical payload. This is not a
ragged-memory saving or a Cornerstone-specific quality result. See
[HETEROGENEOUS_BUDGET_RUNTIME.md](HETEROGENEOUS_BUDGET_RUNTIME.md) and
[HETEROGENEOUS_BUDGET_QUALITY_SCREEN.md](HETEROGENEOUS_BUDGET_QUALITY_SCREEN.md).

Experiment C's bounded historical attention prior passed zero-bias identity
and real-Halo runtime gates. Its 60-cell screen tested no prior, normalized
`S/C`, raw `S`, shuffled `S/C`, and age-only. Every condition scored `5/12`;
normalized, raw, and shuffled each had mean endpoint `0.21577`, and normalized
failed the frozen specificity rule. The intervention changed every compared
logit trajectory and cache state, so the null answer result is not an inactive-
feature artifact. It supports no feedback-safety or answer-gain claim. See
[ATTENTION_PRIOR_QUALITY_REPORT.md](ATTENTION_PRIOR_QUALITY_REPORT.md).

Experiment D's `M=0` path is exactly identical to hard eviction and its `M=4`
runtime is finite with an exact 491,904-byte slot ledger. The 72-cell quality
screen matched total persistent capacity as closely as integer ordinary-entry
units allow: M0 used B128; every M4 arm used B123 plus slots, 108,416 bytes
below M0. All six conditions scored `5/12`. Normalized `S/C` memory had endpoint
`0.183036`, below M0 `0.211310` and random `0.191964`, and beat no simpler M4
control. Exact-token endpoint losses partly reflect replacing exact positions
with blended slots; answer accuracy supplies no recovery gain. See
[SIGNIFICANCE_MEMORY_QUALITY_SCREEN_REPORT.md](SIGNIFICANCE_MEMORY_QUALITY_SCREEN_REPORT.md).

Experiment E implements accept-versus-full-cache recovery without oracle
inputs. Its first fixed six-feature risk score had compressed-error ROC AUC
`0.50` on 11 B0-selected clusters, so no controller-quality signal is
supported. A two-case live smoke still verified both actions: accept performs
zero recomputation and preserves the compressed output exactly; recover starts
from an empty cache at position zero, restores every source token, recomputes
once, and matches the prior full-cache prediction. See
[RECOVERY_CONTROLLER_REPORT.md](RECOVERY_CONTROLLER_REPORT.md).

Experiment F froze a SHA-256 whole-cluster split of the same 11 feature rows
into 5 train, 3 calibration, and 3 evaluation clusters, with both classes in
every split and no salt retry. A seven-parameter L2 logistic controller scored
held-out AUC and balanced accuracy `1.0`, recovered both errors, accepted the
one success, and had a recorded counterfactual wall multiplier `1.172x`.
However, the evaluation contains only two error-success pairs, one training
feature is constant, and the depth-two tree visibly overfit (held-out AUC and
balanced accuracy `0.5`). The classification is
`UNDERPOWERED_OVERFIT_SCREEN_ONLY` with high overfit risk; Finding D and advancement remain
unsupported. See [TINY_CONTROLLER_REPORT.md](TINY_CONTROLLER_REPORT.md).

The manifest-verified [controls](results/figures/halo_controls_v1) and
[deep-architecture](results/figures/halo_deep_architecture_v1) sets contain 50
figures (28+22) backed by exported data and source hashes. Each panel's own
claim limit governs its use; none converts the development screens into a
broad confirmatory result. The scoped [literature audit](LITERATURE_NOVELTY_AUDIT.md)
finds extensive prior art and retains a `NO_NOVELTY_CLAIM` verdict.

## Tier-A evidence boundary

The existing [`tier_a_20260823`](results/runs/tier_a_20260823) run loaded no
model and remains useful as a deterministic machinery artifact. Its benchmark,
trace, pairing, and statistical methodology were invalidated for scientific
inference by a recorded-config/execution mismatch and non-exchangeable round
settings. Its prior toy-policy contrasts are not current scientific evidence
and must not be generalized to Halo or the published family.

## Scientific object

The defining persistent state is:

```text
S_i <- S_i + s_i
C_i <- C_i + exposure_i
R_i(e) = S_i / max(C_i, 1)^e
```

The invariant under growth, query probing, rollback, selection, gather, and
eviction is:

```text
(K, V, S, C, absolute_position, entry_id) remain aligned.
```

This lab isolates Cornerstone 1. Span pooling or contiguous-evidence methods
are separate mechanisms and cannot be credited as Cornerstone-1 gains.

## Repository boundaries

- Working repository: `https://github.com/LUNCY579/ChatBubble.git`
- Active model root: `Rineya_Published_Cores/`
- Primary model: `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze`
- Read-only mechanism provenance: `https://github.com/LUNCY579/RineyaBeta.git`
- Published checkpoints are inputs and are never overwritten.

The source remote has a non-obvious branch layout. Exact refs, commits, and
blobs are recorded in [REPO_PROVENANCE.md](REPO_PROVENANCE.md); symbolic `HEAD`
must not be used as an implementation identifier.

## Safe first checks

The CLI requires this source checkout because it loads `scripts/` and `schemas/`
from the lab tree; those are not standalone-wheel resources. There is no
supported console entry point. From the ChatBubble repository root:

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  .\rineya_significance_native_architecture_lab\tests -q `
  -m "not slow and not model and not gpu"

Push-Location .\rineya_significance_native_architecture_lab
$env:PYTHONPATH = (Resolve-Path .\src).Path
$modelPath = (Resolve-Path `
  "..\Rineya_Published_Cores\Rineya-Halo-1.2.0-devfreeze" `
).Path

& ..\.venv\Scripts\python.exe -m rineya_significance_lab.cli run `
  --config configs\halo_cpu_smoke.yaml `
  --model-path $modelPath `
  --output results\runs\halo_cpu_smoke_manual
Pop-Location
```

Use a new output directory; never overwrite the completed canonical smoke.
The draft confirmatory config is a planning artifact and must not be presented
or launched as a registered confirmatory run. Endpoint v1 has one completed
local prospective registration on 12 difficulty-calibrated clusters, but that
record is not a public preregistration or registration of the broader
confirmatory protocol. The ladder, capability screens, controls, and
architecture screens remain development data and cannot be relabeled as broad
confirmation.

## Documentation map

- [CONFIRMATORY_PROTOCOL.md](CONFIRMATORY_PROTOCOL.md)
- [CPU_EXECUTION_GUIDE.md](CPU_EXECUTION_GUIDE.md)
- [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md)
- [ARCHITECTURE_VARIANTS.md](ARCHITECTURE_VARIANTS.md)
- [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md)
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)
- [EFFICIENCY_REPORT.md](EFFICIENCY_REPORT.md)
- [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)
- [SIGNIFICANCE_CAUSAL_ANALYSIS.md](SIGNIFICANCE_CAUSAL_ANALYSIS.md)
- [EVIDENCE_ENDPOINT_CALIBRATION.md](EVIDENCE_ENDPOINT_CALIBRATION.md)
- [ENDPOINT_VALIDATION_V2_RESULT.md](ENDPOINT_VALIDATION_V2_RESULT.md)
- [SHOWCASE64_EVAL_REPORT.md](SHOWCASE64_EVAL_REPORT.md)
- [R7_V2_REPORT.md](R7_V2_REPORT.md)
- [HETEROGENEOUS_BUDGET_QUALITY_SCREEN.md](HETEROGENEOUS_BUDGET_QUALITY_SCREEN.md)
- [ATTENTION_PRIOR_QUALITY_REPORT.md](ATTENTION_PRIOR_QUALITY_REPORT.md)
- [SIGNIFICANCE_MEMORY_QUALITY_SCREEN_REPORT.md](SIGNIFICANCE_MEMORY_QUALITY_SCREEN_REPORT.md)
- [RECOVERY_CONTROLLER_REPORT.md](RECOVERY_CONTROLLER_REPORT.md)
- [TINY_CONTROLLER_REPORT.md](TINY_CONTROLLER_REPORT.md)
- [LITERATURE_NOVELTY_AUDIT.md](LITERATURE_NOVELTY_AUDIT.md)
- [results/figures/README.md](results/figures/README.md)
