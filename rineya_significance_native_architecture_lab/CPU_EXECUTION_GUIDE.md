# CPU execution guide

The active model scope is the published Qwen3-0.6B family, with Halo primary.
Rineya/R2 and Aegis model commands are intentionally absent because those
models are outside the current experiment scope.

## Source-checkout contract

Run the lab from this source checkout. The CLI dynamically loads runners from
`scripts/` and result validation reads schemas from `schemas/`; those resources
are not promised as standalone wheel contents. There is no supported console
entry point. Invoke the module directly.

From the ChatBubble repository root:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e `
  ".\rineya_significance_native_architecture_lab[model,analysis,dev]"

Push-Location .\rineya_significance_native_architecture_lab
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

The editable install provides dependencies and import metadata; execution still
depends on the checked-out `scripts/` and `schemas/` directories. The declared
dependency pins Transformers `5.14.1`. The newer count, schedule, and R5-R8
v1 manifests and the locally registered endpoint-v2 run record Transformers
`5.15.0` and Torch `2.13.0+cpu`; their same-process comparisons remain
internally matched. Eligible-count v2 and Experiments B-D record the locked
Transformers `5.14.1` / Torch `2.12.1+cu126` environment. Endpoint v2 has not
been reproduced under that lock, and separate-process count traces were not
bitwise stable. These differences must not be hidden or pooled into a cross-
runtime claim. Any new environment requires identity and cache-invariant gates.

## Fast validation

```powershell
& ..\.venv\Scripts\python.exe -m pytest tests -q `
  -m "not slow and not model and not gpu"

Get-ChildItem configs\*.yaml | ForEach-Object {
  & ..\.venv\Scripts\python.exe -m rineya_significance_lab.cli `
    validate-config --config $_.FullName
}
```

Restore location after work:

```powershell
Pop-Location
```

## Halo model path and hashes

Resolve the published core without committing a user-specific absolute path:

```powershell
$modelPath = (Resolve-Path `
  "..\Rineya_Published_Cores\Rineya-Halo-1.2.0-devfreeze" `
).Path
```

The expected SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| `model.safetensors` | `38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544` |
| `tokenizer.json` | `be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506` |
| `config.json` | `2284cc1687a4f4f6e3a64a13dfcf7562e592b951afa2f91244cc7b212c9cfb85` |

Verify before every new model run:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
  (Join-Path $modelPath "model.safetensors")
Get-FileHash -Algorithm SHA256 -LiteralPath `
  (Join-Path $modelPath "tokenizer.json")
Get-FileHash -Algorithm SHA256 -LiteralPath `
  (Join-Path $modelPath "config.json")
```

Do not silently substitute another family core. Halo is the primary target;
each secondary core requires its own immutable identity, config, and result.

## Tier A

Tier A loads no checkpoint. The preserved `tier_a_20260823` artifacts are under
methodology review. New Tier-A executions are diagnostic until the experimental
unit, cluster definition, controls, and analysis are re-approved.

```powershell
& ..\.venv\Scripts\python.exe -m rineya_significance_lab.cli run `
  --config configs\tier_a.yaml `
  --output results\runs\tier_a_review_manual
```

Do not turn config validation alone into a completed-run manifest.

## Published-family CPU smokes

Use a fresh output directory:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
& ..\.venv\Scripts\python.exe -m rineya_significance_lab.cli run `
  --config configs\halo_cpu_smoke.yaml `
  --model-path $modelPath `
  --output results\runs\halo_cpu_smoke_manual
```

The canonical Halo command above has sibling source-checkout configs for Halo
Uncensored, Lyra, Nimbus, and Pulse. The five canonical 20260823 runs all used
CPU bf16 on one identical 192-token prompt-only case per core. Each finished at
**B48+C64** (112 stored), rolled back the transient ten-token query exactly,
recorded two scoring events per layer, and produced exact disabled-path logits
(maximum and mean delta 0.0; argmax agreement 1.0). Expected model/tokenizer/
config hashes were verified for every core. No smoke decoded an answer.

Runtime status: local recorder implementing audited semantics; 4/4 audited
primitives match exactly, while end-to-end source-runtime parity is `NOT_RUN`.
This smoke is not an answer-quality,
long-context, throughput, peak-memory, or deployment-latency benchmark.

## Budget terminology

Always report:

1. retained history after compression;
2. active chunk during the current forward;
3. transient query probe before rollback;
4. stored state after the final chunk;
5. KV-only and total persistent bytes.

The config loader currently retains `total_budget_slots` as a compatibility key
for the retained-history limit. It must not be described as total live residency.

## Draft confirmatory work

`configs/confirmatory.yaml` is `DRAFT_NOT_REGISTERED`. Validate it, but do not
launch or label it confirmatory. Endpoint v1 now has one narrow local
prospective registration, but the broad cluster/sample/runtime plan is not
frozen and no broad registration record exists:

```powershell
& ..\.venv\Scripts\python.exe -m rineya_significance_lab.cli `
  validate-config --config configs\confirmatory.yaml
```

The generic CLI intentionally has no implemented runner for that draft.

`configs/halo_mini_eval.yaml` produced the completed bounded-exploratory run
`results/runs/halo_mini_eval_20260823_v2`. B0, B2, and B3 each scored 1/3; B2 and
B3 each scored 1/1 on the sole B0-eligible case. Two cases failed the full-cache
capability gate. The run has no inferential statistics or exact-evidence result
and must not be described as confirmatory, equivalent-performance evidence, or
a population-level quality finding.

## Completed Halo development artifacts

The following canonical outputs are complete and must not be overwritten:

| Artifact | Scope |
|---|---|
| `results/statistics/halo_evidence_endpoint_calibration_v1.json` | outcome-blind endpoint freeze for future evaluations |
| `results/runs/halo_endpoint_validation_capability_v1/` | prospective B0 9/24 stop; no compressed run |
| `results/runs/halo_endpoint_validation_v2_capability/` | difficulty-calibrated B0 18/24 pass |
| `results/runs/halo_endpoint_validation_v2_registered/` | locally registered 12-cluster endpoint contrast; no answer gain |
| `results/benchmarks/halo_r1_r8_ladder_v1/` | 1,176-case / 84-cluster construction artifact |
| `results/runs/halo_r5_r8_capability_screen_v1/` | eight raw rows; R7 v1 invalid/uninformative, valid R5/R6/R8 subset 3/6, R8 0/2 stop |
| `results/benchmarks/halo_r7_v2/` + `results/runs/halo_r7_v2_capability_screen/` | valid balanced R7 repair; Halo all-YES 4/8 gate stop |
| `results/runs/halo_count_causal_controls_v1/` | 88 completed count-control cells and 264 boundary rows |
| `results/runs/halo_count_causal_controls_v2/` | 33 eligible-pool clean-control cells; within-process paired nulls |
| `results/runs/halo_schedule_control_v2/` | 110 completed schedule cells / 55 matched pairs |
| `results/runs/halo_adaptive_budget_screen_v1/` | 132-row boundary-local open-loop screen; specificity gate failed |
| `results/runs/halo_heterogeneous_budget_smoke_v2/` | padded heterogeneous runtime and exact uniform identity |
| `results/runs/halo_heterogeneous_budget_quality_screen_v1/` | 36 closed-loop cells; specificity failed; padded overhead measured |
| `results/runs/halo_attention_prior_smoke_v1/` + quality screen | Experiment C runtime pass; 60-cell all-5/12 specificity failure |
| `results/runs/halo_significance_memory_smoke_v1/` + quality screen | Experiment D M0 identity; 72-cell negative/neutral result |
| `results/runs/halo_recovery_controller_screen_v1/` + smoke | Experiment E AUC 0.50; accept/recompute mechanism pass |
| `results/experiment_f/halo_tiny_controller_v1/` | Experiment F frozen 5/3/3 underpowered overfit-risk screen |
| `results/figures/halo_controls_v1/` + `halo_deep_architecture_v1/` | 50 manifest-verified exploratory/engineering figures (28+22) |

Endpoint v1 has one completed local registration; the broader protocol remains
unregistered. The count and schedule runs reuse 11 B0-selected calibration
clusters. Age-derived and
head-averaged counts are exact true-C trajectory equivalents only for that
fixed schedule. The schedule retention interaction is exploratory, and its
answer interaction is not detected (`p=0.4375`). Shuffled and wrong-token count
transforms are eligible-pool confounded and must not be reused as clean identity
controls without preserving the protected sink/recent partition.
The adaptive-budget artifact is replay-only: it loaded no model, generated no
answers, and did not measure sparse physical-storage overhead. Its closed-loop
successor uses padded storage and also fails specificity. Experiments C and D
likewise pass disabled/runtime gates but not their quality-specificity gates. E
supports recovery mechanics only, and F's perfect n=3 held-out logistic result
is explicitly underpowered/high-overfit-risk.

Safe read-only checks:

```powershell
Get-Content results\statistics\halo_evidence_endpoint_calibration_v1.json |
  ConvertFrom-Json | Select-Object status, classification, decision

Get-Content results\runs\halo_count_causal_controls_v1\run_state.json |
  ConvertFrom-Json | Select-Object status, completed_executions, planned_executions

Get-Content results\runs\halo_schedule_control_v2\run_state.json |
  ConvertFrom-Json | Select-Object status, completed_cells, planned_cells

Get-Content results\runs\halo_adaptive_budget_screen_v1\summary.json |
  ConvertFrom-Json | Select-Object status, decision, row_count, global_entry_budget

Get-Content results\runs\halo_endpoint_validation_v2_registered\summary.json |
  ConvertFrom-Json | Select-Object status, classification, claim_limit

Get-Content results\runs\halo_count_causal_controls_v2\statistics.json |
  ConvertFrom-Json | Select-Object status, classification,
    cross_run_reference_reproducibility

Get-Content results\experiment_f\halo_tiny_controller_v1\summary.json |
  ConvertFrom-Json | Select-Object status, classification, decision

Get-Content results\figures\halo_controls_v1\figure_manifest.json |
  ConvertFrom-Json | Select-Object status, figure_count, classification
```

The count runner exposes a non-mutating plan check:

```powershell
& ..\.venv\Scripts\python.exe scripts\run_halo_count_controls.py `
  --config configs\halo_count_causal_controls.yaml --dry-run
```

Do not rerun the ladder, capability screens, endpoint validation, count or
schedule controls, Experiments B-F, or figure generator into their canonical
directories. Copy the config and use
a fresh output path/version before new execution. Resume is permitted only for
an actually interrupted run with matching immutable fingerprints; it is not a
way to append to a completed canonical result.

## Windows and CPU cautions

- Use `Path`/`Resolve-Path`, not hard-coded user homes.
- Keep `num_workers=0` until Windows spawn behavior is tested.
- Record on-disk dtype/bytes separately from bf16 runtime dtype.
- Do not request and retain full per-layer `output_attentions` tensors merely
  to collect reduced significance scores.
- Respect Halo's native `max_position_embeddings=40960` unless RoPE scaling is
  a separately declared experiment.
- CPU latency cannot stand in for GPU deployment latency.

A run may resume only when config, repositories, model, tokenizer, benchmark,
and precision identities match. Never rewrite an interrupted run as `COMPLETE`
without a full validation pass.
