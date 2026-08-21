# Rineya Layer-wise Soup Lab v1.0

A standalone, resumable research pipeline for combining two compatible Rineya checkpoints:

- **Anchor** — the preservation-oriented 12-layer student.
- **Greedy endpoint** — the improved greedy-soup checkpoint derived from the same model lineage.

The lab is deliberately isolated from the official training phases. It never edits either source checkpoint. Every merged candidate, evaluation result, coefficient recipe, selection decision, and sealed-test access is written under a new run directory.

## What this repository implements

The pipeline evaluates three nested experiments under one frozen evaluation contract:

1. **Uniform soup**

   \[
   \theta_{0.5}=0.5\theta_A+0.5\theta_G
   \]

2. **Global score-guided interpolation**

   \[
   \theta(\alpha)=(1-\alpha)\theta_A+\alpha\theta_G
   \]

   A coarse sweep is followed by an optional local refinement. Every candidate is scored with the same Quality Evaluator **v4.5** development split. Metric transformations and weights are declared before the sweep; paired bootstrap resampling estimates uncertainty without changing scores after seeing results.

3. **Layer-adaptive behavioral soup**

   \[
   \theta^{(l)}=(1-\alpha_l)\theta_A^{(l)}+\alpha_l\theta_G^{(l)}
   \]

   For each Transformer layer, the lab performs:

   - Anchor with one Greedy layer inserted;
   - optionally, Greedy with one Anchor layer inserted;
   - a centered single-layer coefficient grid around the globally selected model;
   - per-response utility deltas and paired bootstrap intervals;
   - reliability shrinkage toward the global coefficient;
   - soft adjacent-layer regularization;
   - bounded coordinate descent, with an optional pairwise refinement budget.

The result is not merely a set of averaged checkpoints. It is a complete evidence trail showing how each coefficient was chosen.

## Core safeguards

- Source model directories are treated as **read-only**.
- Quantized GPTQ/AWQ/GGUF-style sources are rejected by default.
- Tensor keys, shapes, layer indices, critical architecture fields, tokenizer fingerprints, and non-floating buffers are checked before merging.
- Floating-point tensors are accumulated in **FP32** and only then converted to the configured output dtype.
- Development and test IDs/prompts are audited for overlap.
- The sealed test cannot run until `selection/selection_freeze.json` exists.
- Only the frozen candidate is evaluated on the sealed split.
- Interrupted evaluations are resumable by stable case ID.
- Candidate weights can be deleted after evaluation while evidence remains durable; selected candidates are reconstructed from the immutable endpoints when needed.
- Every checkpoint, configuration, evaluator policy, and result summary receives a provenance hash.
- Every materialized merge records an output-checkpoint digest; a damaged or altered run-owned cache is rebuilt rather than silently reused.

## Repository layout

```text
Rineya_Layerwise_Soup_Lab_v1.0/
├── configs/
│   ├── experiment.example.yaml
│   └── official_v45/
├── docs/
├── schemas/
├── scripts/
├── src/rineya_soup_lab/
├── tests/
├── artifacts/                 # intentionally empty; run outputs go elsewhere
├── pyproject.toml
└── README.md
```

## Halo pair vs the 12-layer student study

`configs/experiment.local.yaml` is the original **12-layer** Rineya student study (frozen_anchor × greedy soup). Do not use that file to compare against Halo.

`configs/experiment.qwen3_halo_uncensored.yaml` is a **same-shape 28-layer** Qwen3 experiment:

| Endpoint | Path | Uniform α on this pair |
|---|---|---|
| Anchor | official `Qwen3-0.6B/` | α=0 |
| Greedy | vault Halo Uncensored (do not overwrite) | α=1 |
| Halo control | implied blend on kept blocks | α=0.5 |

Dropped Qwen3 blocks, embeddings, norm, and `lm_head` are identical on both endpoints, so they stay official Qwen3 at every α. Search is restricted to the 19 Phase 2B kept layer ids. Coefficients are chosen on chat DEV v4.5 (510), not on the frozen 64-example showcase. After freeze, `scripts/score_showcase_finalists.py` scores `uniform` / `global_best` / `layerwise_seed` / `layerwise_best` once on that showcase (descriptive only). Sealed TEST stays off. `models.allow_hf_serialization_drift` is on only for this pair: newer transformers rewrote Uncensored `config.json` and omitted tied `lm_head.weight`.

On an 8GB GPU the Halo-pair run waits for other Python compute jobs, loads with `device_map=auto` and a VRAM cap, and deletes ephemeral candidate weights after each eval.

```powershell
rineya-soup run --config configs\experiment.qwen3_halo_uncensored.yaml --through preflight
rineya-soup run --config configs\experiment.qwen3_halo_uncensored.yaml --run-dir runs\qwen3_halo_uncensored --through freeze
python scripts\score_showcase_finalists.py --config configs\experiment.qwen3_halo_uncensored.yaml --run-dir runs\qwen3_halo_uncensored
```

A completed run contains:

```text
runs/<run-id>/
├── config.input.yaml
├── config.resolved.yaml
├── manifests/
│   ├── compatibility.json
│   ├── dataset_split_audit.json
│   ├── environment.json
│   └── preflight.json
├── candidate_cache/
├── evaluations/
│   ├── dev/<protocol>/<candidate-id>/
│   └── test/<protocol>/<candidate-id>/
├── analysis/
│   ├── global/
│   ├── layers/
│   ├── probability/
│   └── protocols/
├── selection/selection_freeze.json
├── sealed/access_ledger.json
├── reports/
│   ├── RESEARCH_REPORT.md
│   └── HUMAN_REVIEW_PACKET.md
└── state/run_state.json
```

## System requirements

Designed for:

- Windows 11 or Linux;
- Python 3.11 or newer;
- a local Hugging Face-compatible causal-LM checkpoint;
- enough disk space for at least one materialized model plus temporary shards;
- CUDA strongly recommended for real evaluation.

The merge process is CPU/RAM/disk oriented and streams tensors one at a time. Model evaluation uses `transformers` and can require CUDA according to the selected protocol.

## Windows installation

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build appropriate for your driver from the official PyTorch selector, then install this project:

```powershell
python -m pip install -e ".[dev]"
```

Verify the environment:

```powershell
python scripts\validate_install.py
pytest -q
```

The included source-level test suite uses synthetic checkpoints and does not require `transformers` or a GPU. Real model evaluation does.

## Configure the experiment

Copy the annotated template:

```powershell
rineya-soup init-config --output configs\experiment.local.yaml
```

Edit at minimum:

```yaml
models:
  anchor_path: C:/path/to/rineya_anchor
  greedy_path: C:/path/to/rineya_greedy_soup
  tokenizer_path: C:/path/to/rineya_anchor
  expected_layers: 12
  lineage_acknowledged: true

datasets:
  dev: C:/path/to/official_phase_4_5_dev.jsonl
  test: C:/path/to/official_phase_4_5_sealed_test.jsonl
```

The two model directories must have the same architecture, parameter names, tensor shapes, vocabulary, and tokenizer. They should descend from the same model initialization/lineage. Shape compatibility alone cannot prove that lineage, which is why the acknowledgement is explicit. Full Hugging Face checkpoints are used directly; PEFT endpoints are first materialized against `models.base_model_path` in a run-owned directory so LoRA factors are never averaged independently.

## Run everything

```powershell
rineya-soup run --config configs\experiment.local.yaml
```

This intentionally executes every enabled stage, including the sealed test. For research use, the safer default is to stop at `--through freeze`, inspect the development evidence, and then invoke the sealed stage once.

The command prints the run directory. All stages are resumable:

```powershell
rineya-soup run `
  --config configs\experiment.local.yaml `
  --run-dir runs\<run-id>
```

Run only through development selection, leaving the sealed test untouched:

```powershell
rineya-soup run `
  --config configs\experiment.local.yaml `
  --through freeze
```

Generate the development-only report, then review `reports/RESEARCH_REPORT.md`, `reports/HUMAN_REVIEW_PACKET.md`, and the raw replies:

```powershell
rineya-soup report `
  --config configs\experiment.local.yaml `
  --run-dir runs\<run-id>
```

Run the sealed test only after accepting the frozen selection:

```powershell
rineya-soup stage sealed `
  --config configs\experiment.local.yaml `
  --run-dir runs\<run-id>
```

Convenience wrappers are also available:

```powershell
scripts\run_all.ps1 -Config configs\experiment.local.yaml
scripts\run_stage.ps1 -Config configs\experiment.local.yaml -RunDir runs\<run-id> -Stage report
```


## Recommended execution strategy

A full 12-layer study can require many development evaluations: two endpoints, the uniform control, a global sweep, up to 24 bidirectional layer swaps, centered per-layer grids, and bounded coordinate refinement. Start with a smoke run using fewer layers and a small development subset, then create a new full run with the final predeclared configuration. Never tune on the sealed split.

Typical sequence:

```powershell
# 1. Create a new run and verify paths, compatibility, tokenizer, and split integrity.
rineya-soup run --config configs\experiment.local.yaml --through preflight
# Record the run directory printed by the command.

# 2. Resume that exact run through development selection only.
rineya-soup run --config configs\experiment.local.yaml --run-dir runs\<printed-run-id> --through freeze

# 3. Generate the development-only report and review raw outputs.
rineya-soup report --config configs\experiment.local.yaml --run-dir runs\<printed-run-id>

# 4. Access the sealed split once.
rineya-soup stage sealed --config configs\experiment.local.yaml --run-dir runs\<printed-run-id>

# 5. Regenerate the final report.
rineya-soup stage report --config configs\experiment.local.yaml --run-dir runs\<printed-run-id>
```

Use `configs/experiment.smoke.example.yaml` for a low-cost pipeline check before the full configuration. Smoke results are engineering evidence only, not final research evidence.

## Evaluation backends

### `bundled_v45` — recommended standalone mode

Uses the vendored Quality Evaluator v4.5 policy and a local Hugging Face generator. This backend supports the primary no-thinking protocol and an optional separately labeled thinking protocol.

### `official_phase_v45`

Launches the evaluator from your official short-response repository:

```yaml
evaluation:
  backend: official_phase_v45
  official_repo_root: C:/path/to/official/repository
```

This is useful when you want byte-for-byte continuity with the official phase runner. The supplied official evaluator forces Qwen thinking off, so use `bundled_v45` for an independent thinking-mode experiment.

### `external`

Runs a user-specified command template and validates that the resulting `summary.json` and `results.jsonl` identify Quality Evaluator v4.5.

### `mock`

For unit tests only. It is rejected unless `mock_allowed: true`. Mock results must never support a research claim.

## Thinking mode

The default coefficient-selection protocol is `official_no_think`, preserving comparability with the official phase evaluation.

A separate Qwen thinking protocol is included but disabled:

```yaml
evaluation:
  selection_protocol: official_no_think
  protocols:
    think_then_short:
      enabled: true
      enable_thinking: true
      extract_final_after_thinking: true
```

Setting `selection_protocol: think_then_short` makes that protocol the coefficient-selection basis. Do not mix scores from thinking and no-thinking protocols in one interpolation curve; they are distinct experiments.

## Optional token-distribution profile

Enable:

```yaml
experiments:
  probability_profile:
    enabled: true
    max_cases: 150
    top_k: 128
```

The profiler teacher-forces both endpoints over the same target trajectory and reports target-token probabilities, EOS probabilities, entropy, top-k overlap, and a **coarse-grained Jensen–Shannon lower bound**. It retains only token IDs appearing in both endpoint top-k sets and collapses all remaining vocabulary mass into one shared residual bucket. This avoids double-counting unknown mass and is explicitly not presented as full-vocabulary JS divergence. The profile is diagnostic and is not used to choose coefficients.

## Dataset contract

Each JSONL row should contain:

```json
{
  "id": "stable_case_id",
  "category": "simple_fact",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "reference": "Paris.",
  "expected": {
    "min_tokens": 1,
    "max_tokens": 12
  }
}
```

An SFT-style row ending in an assistant message is also accepted; the final assistant message is removed from the generation prompt and retained only as the reference.

The v4.5 evaluator treats category labels and reference wording as diagnostics unless an explicit response contract requires specific behavior.

## How layer influence is interpreted

The lab deliberately avoids saying that one layer “caused 40% of an answer.” Instead, for case `i` and layer `l`, it measures a behavioral intervention:

\[
\Delta_{i,l}^{A\rightarrow G}
=U(A\text{ with }G_l, x_i)-U(A,x_i)
\]

and, when bidirectional analysis is enabled:

\[
\Delta_{i,l}^{G\rightarrow A}
=U(G,x_i)-U(G\text{ with }A_l,x_i).
\]

The two directions are averaged into a Greedy-layer advantage for that response. This is a task-distribution-dependent intervention effect, not a unique mechanistic decomposition.

## Research outputs to inspect first

1. `analysis/global/interpolation_curve.csv`
2. `analysis/global/global_selection.json`
3. `analysis/layers/per_response_layer_influence.jsonl`
4. `analysis/layers/layer_influence_summary.csv`
5. `analysis/layers/centered_layer_grid.csv`
6. `analysis/layers/coordinate_search_trace.jsonl`
7. `selection/selection_freeze.json`
8. `sealed/sealed_test_result.json`
9. `analysis/layers/delta_influence_correlation.json`
10. `analysis/protocols/secondary_protocols.json`
11. `reports/HUMAN_REVIEW_PACKET.md`
12. `reports/RESEARCH_REPORT.md`

## Manual recipe materialization

A coefficient recipe can be materialized independently:

```json
{
  "schema_version": "1.0",
  "kind": "merge",
  "global_alpha": 0.60,
  "layer_alphas": {
    "4": 0.80,
    "5": 0.90,
    "6": 0.75
  },
  "group_alphas": {},
  "label": "manual_candidate"
}
```

```powershell
rineya-soup merge `
  --config configs\experiment.local.yaml `
  --recipe recipe.json `
  --output C:\path\to\manual_candidate
```

The manual command supports either full checkpoints or PEFT adapters configured with the same `models.base_model_path` contract as the main pipeline. It refuses output paths that overlap either source checkpoint, even when `--overwrite` is supplied.

## Scientific boundaries

- A better development score is not automatically a publishable discovery.
- A layer-wise coefficient pattern should be replicated across seeds, development subsets, and preferably a second task distribution.
- The sealed test should remain sealed; rerunning it to choose coefficients converts it into development data.
- Layer-wise merging can exploit interactions and compensation. It does not prove layers act independently.
- Automated v4.5 scoring should be paired with raw-output review and human preference evaluation before deployment.

See [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) for the full methodology.
