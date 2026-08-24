# Results directory

Current reviewed run inventory:

| Run | Tier/mode | Status | Supported scope |
|---|---|---|---|
| [`source_parity.json`](source_parity.json) | deterministic primitive comparison | `PASS` | 4/4 exact source-primitive checks; not end-to-end reproduction |
| [`halo_calibration_v3`](benchmarks/halo_calibration_v3) | generated calibration dataset | `CALIBRATION_DATASET_BUILT` | 32 exact-evidence clusters; no model outcome |
| [`halo_capability_calibration_v2`](runs/halo_capability_calibration_v2) | B / full-cache CPU, bf16 | execution `PASS`; gate `FAILED` | 18/32 overall; R2 family 0/8 |
| [`halo_evidence_endpoint_calibration_v1.json`](statistics/halo_evidence_endpoint_calibration_v1.json) | outcome-blind endpoint calibration | `PASS`; frozen | non-saturated definition; later local validation is indexed separately |
| [`halo_endpoint_validation_capability_v1`](runs/halo_endpoint_validation_capability_v1) | B / prospective B0 gate | `STOPPED_AT_B0_GATE` | 9/24; no compressed condition |
| [`halo_endpoint_validation_v2_capability`](runs/halo_endpoint_validation_v2_capability) | B / difficulty-calibrated B0 gate | `PASS` | 18/24; 12 clusters advanced to local registration |
| [`halo_endpoint_validation_v2_registered`](runs/halo_endpoint_validation_v2_registered) | B / locally registered prospective CPU | `PASS_NARROW_RETENTION_ONLY` | B3-B2 endpoint +0.0401786, p=.01953125; answers 5 vs 6, p=1 |
| [`halo_r1_r8_ladder_v1`](benchmarks/halo_r1_r8_ladder_v1) | generated controlled ladder | `TIER_A_LADDER_BUILT` | 1,176 cases / 84 clusters; no model outcome |
| [`halo_r5_r8_capability_screen_v1`](runs/halo_r5_r8_capability_screen_v1) | B / full-cache CPU, bf16 | execution complete; R7 invalid, R8 stop | raw 5/8; valid R5/R6/R8 subset 3/6; no compression policy |
| [`halo_r7_v2`](benchmarks/halo_r7_v2) + [screen](runs/halo_r7_v2_capability_screen) | corrected dataset + B0 CPU | design `PASS`; gate `FAILED` | balanced benchmark; Halo all-YES 4/8, 0/4 NO; no compression |
| [`halo_retention_pilot_v2`](runs/halo_retention_pilot_v2) | B / selected exploratory CPU, bf16 | `EXPLORATORY_POSTHOC` | 11 B0-selected R1/R3/R4 clusters; endpoint-design reuse, not independent validation |
| [`halo_count_control_identity.json`](halo_count_control_identity.json) | real-Halo implementation identity | `PASS` | one-case standard-B3 vs true-count callback exact identity; no quality result |
| [`halo_count_causal_controls_v1`](runs/halo_count_causal_controls_v1) | B / closed-loop exploratory CPU, bf16 | `COMPLETE` | 88 cells; exact age/head equivalence in fixed schedule; adjusted count contrasts null |
| [`halo_count_causal_controls_v2`](runs/halo_count_causal_controls_v2) | B / clean eligible-pool controls | `COMPLETE` | 33 cells; 22,176 invariant checks; both Holm-null; paired process-local evidence |
| [`halo_schedule_control_v2`](runs/halo_schedule_control_v2) | B / held-constant exploratory CPU, bf16 | `COMPLETE` | 110 cells / 55 matched pairs; retention interaction, answer interaction null |
| [`halo_adaptive_budget_screen_v1`](runs/halo_adaptive_budget_screen_v1) | A / boundary-local open-loop replay | `NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE` | 132 rows; shuffled-statistic strongest; no model answers |
| [`halo_heterogeneous_budget_smoke_v2`](runs/halo_heterogeneous_budget_smoke_v2) | B / Experiment-B runtime | `PASS` | uniform identity, unequal allocation, exact logical budget; padded storage |
| [`halo_heterogeneous_budget_quality_screen_v1`](runs/halo_heterogeneous_budget_quality_screen_v1) | B / Experiment-B quality | `SPECIFICITY_FAILED` | 36 cells; entropy 8/12 vs uniform 5/12, p=.25; shuffled retention higher |
| [`halo_attention_prior_smoke_v1`](runs/halo_attention_prior_smoke_v1) | B / Experiment-C runtime | `PASS` | zero-bias identity and bounded active prior |
| [`halo_attention_prior_quality_screen_v1`](runs/halo_attention_prior_quality_screen_v1) | B / Experiment-C quality | `NO_MEASURABLE_ANSWER_GAIN` | 60 cells; all 5/12; normalized equals raw/shuffled endpoint |
| [`halo_significance_memory_smoke_v1`](runs/halo_significance_memory_smoke_v1) | B / Experiment-D runtime | `PASS` | exact M0 identity; finite M4 and 491,904-byte ledger |
| [`halo_significance_memory_quality_screen_v1`](runs/halo_significance_memory_quality_screen_v1) | B / Experiment-D quality | `NEGATIVE_OR_NEUTRAL` | 72 byte-matched cells; all 5/12; normalized below M0/random endpoint |
| [`halo_recovery_controller_screen_v1`](runs/halo_recovery_controller_screen_v1) + [smoke](runs/halo_recovery_controller_smoke_v1) | B / Experiment E | signal `NO_MEASURABLE_GAIN`; mechanism `PASS` | risk AUC .50; accept/full-recompute branches verified |
| [`halo_tiny_controller_v1`](experiment_f/halo_tiny_controller_v1) | C / Experiment F | `UNDERPOWERED_OVERFIT_SCREEN_ONLY` | high overfit risk; frozen 5/3/3; held-out n=3 logistic AUC/BA 1.0; no advancement |
| [`halo_controls_v1`](figures/halo_controls_v1) | verified visualization set | `PASS` | 28 figures with source hashes and per-figure claim limits |
| [`halo_deep_architecture_v1`](figures/halo_deep_architecture_v1) | verified deep-architecture set | `PASS` | 22 figures covering local endpoint validation, count v2, R7 v2, and B-F; per-panel limits |
| [`halo_cpu_smoke_20260823_v2`](runs/halo_cpu_smoke_20260823_v2) | B / closed-loop CPU, bf16 | `PASS` | primary Halo one-case compatibility smoke |
| [`halo_uncensored_cpu_smoke_20260823`](runs/halo_uncensored_cpu_smoke_20260823) | B / closed-loop CPU, bf16 | `PASS` | one-case compatibility smoke |
| [`lyra_cpu_smoke_20260823`](runs/lyra_cpu_smoke_20260823) | B / closed-loop CPU, bf16 | `PASS` | one-case compatibility smoke |
| [`nimbus_cpu_smoke_20260823`](runs/nimbus_cpu_smoke_20260823) | B / closed-loop CPU, bf16 | `PASS` | one-case compatibility smoke |
| [`pulse_cpu_smoke_20260823`](runs/pulse_cpu_smoke_20260823) | B / closed-loop CPU, bf16 | `PASS` | one-case compatibility smoke |
| [`halo_mini_eval_20260823_v2`](runs/halo_mini_eval_20260823_v2) | B / bounded exploratory CPU, bf16 | `PASS` | B0=B2=B3=1/3; no measurable gain, no inferential statistics |
| [`tier_a_20260823`](runs/tier_a_20260823) | A / open-loop synthetic | `INVALID_OR_CONFOUNDED` | preserved machinery artifacts only |

All five family runs verified their expected weights, tokenizer, and config
hashes. Each represented 192 true tokens with 112 stored states after the last
chunk (**B48+C64**). The ten-token query probe was transient, every layer
recorded two events, and disabled-path recorded logits were exactly identical
on every case. No smoke generated an answer or measured quality. Halo remains
the primary experimental target; these results are not pooled quality evidence.

Runtime status: local recorder implementing audited semantics; audited
source-primitive parity `PASS` on 4/4 exact checks, end-to-end source benchmark
reproduction `NOT_RUN`. RineyaBeta is provenance only. Rineya/R2 and Aegis are
outside the current evidence scope.

The Tier-A run loaded no checkpoint. Its benchmark, pairing, clustering, and
statistical interpretation are invalid for scientific inference; existing files
are retained for auditability and must not be presented as current Halo evidence.

The bounded Halo mini evaluation had two B0 capability failures, leaving one
eligible case; B2 and B3 both scored 1/1 on that case. The unsuffixed
`halo_mini_eval_20260823` directory is superseded and ignored because its
manifest timestamp ordering was invalid. Only the corrected `_v2` run is
canonical.

The v3 calibration dataset has eight independent clusters per R1-R4 benchmark
family and keeps inference inputs separate from answers/evidence annotations.
Full-cache Halo scored R1 `7/8`, R2 `0/8`, R3 `3/8`, and R4 `8/8` (18/32). Its
advance gate required at least two correct per family and therefore failed.
R2 in this result is a benchmark-family label, not a Rineya/R2 checkpoint.

The retention pilot selected 11 B0-correct clusters from R1/R3/R4. B2, B3, and
B4 scored 5/11, 6/11, and 1/11. Outcome-blind evidence-only calibration froze
`layer-fact-retention-v1`; its B3-B2 reuse on those design rows was post hoc
(`+0.061688`, `p=0.0078125`). Prospective v1 then stopped at B0 `9/24`. After
explicit difficulty calibration, v2 passed B0 `18/24` and locally registered 12
clusters before compression. The primary B3-B2 endpoint contrast was
`+0.0401786`, interval `[0.0133929, 0.0758929]`, `p=0.01953125`. Answers were
B3 5/12 versus B2 6/12 (`p=1.0`). This is narrow retained-state support, not
answer gain, public preregistration, or broad confirmation.

That registered run used Python 3.11.15, Torch 2.13.0+cpu, and Transformers
5.15.0. The current lock and Experiments B-D use Torch 2.12.1+cu126 and
Transformers 5.14.1. The endpoint result is internally paired but has not been
reproduced under the locked runtime.

The R1-R8 ladder has 1,176 rows but 84 independent clusters. R7 v1 remains
invalid/uninformative. Corrected R7 v2 is balanced and solver-audited, but Halo
answered YES on all eight frozen cases: 4/8 overall, 0/4 on NO. It failed the
two-sided B0 gate and ran no compression. R8 likewise remains stopped at 0/2.

Count v1 completed 88 cells and found age-derived/head-averaged exact true-C
trajectory equality only under the fixed schedule; its global shuffled/wrong-
token controls are eligible-pool confounded. Count v2 repairs that design with
33 clean eligible-shuffled/circular cells and 22,176 passing layer/head
invariants. Both endpoint contrasts are Holm-null. Same-process identity is
exact, but separate v1/v2 processes were not bitwise trajectory-reproducible,
so v2 is process-local exploratory evidence. The schedule run's 110 cells yield
a positive retention interaction (`+0.00936074`/round, `p=0.00390625`) but no
answer interaction (`p=0.4375`).

The open-loop adaptive-budget replay failed specificity. Its closed-loop
Experiment-B successor also failed promotion: entropy scored 8/12 versus
uniform 5/12 (`p=0.25`), shuffled retained more, and padded physical storage
exceeded matched logical payload. Experiment C's five attention-prior arms all
scored 5/12 and normalized endpoint equaled raw/shuffled. Experiment D's six
matched-capacity memory arms all scored 5/12 and normalized retention was below
M0/random. Experiment E's fixed risk score had AUC 0.50, although live accept
and full-recompute mechanics passed.

Experiment F froze a no-retry 5/3/3 whole-cluster split. Its seven-parameter
logistic model had held-out AUC/balanced accuracy 1.0 and recovered both errors,
but evaluation had only three clusters and two positive-negative pairs; a
depth-two tree visibly overfit. It remains
`UNDERPOWERED_OVERFIT_SCREEN_ONLY`, with no Finding D or advancement.

`halo_count_control_identity.json` separately verifies one real-Halo callback
identity case. It is an engineering check only.

Run-specific artifacts under `runs/`, versioned benchmark manifests under
`benchmarks/`, and `source_parity.json` are authoritative. Root-level
convenience indexes must not override them. New runs write to unique directories
and never overwrite a completed run.

Expected generated layout:

```text
runs/<run_id>/
traces/<run_id>/
statistics/<run_id>/
figures/<run_id>/
benchmarks/<suite_version>/
```

The old Tier-A figures are review artifacts only. The two current Halo
collections contain 50 manifest-verified figures (28+22); each panel inherits
its source artifact's local, exploratory, or engineering limits and adds no
independent evidence.
