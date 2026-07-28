# Phase 6 - Activation Profiling and Sparse Simulation

**Blueprint reference:** pages 18-19. This implementation profiles the frozen dense reference, creates per-layer router datasets, and measures the quality frontier of oracle MLP block masking. Oracle masking still executes dense projections, so Phase 6 never claims speed.

## Prerequisite

`reports/PHASE_5_STOP_GATE.md` must contain an explicit human `PASS`. Any change to the dense reference weights, tokenizer, chat template, or quantization invalidates previously collected traces.

## 1. Build a calibration corpus

Use development/calibration records only. Include casual chat, facts, long-context memory, code/math controls, multilingual or unusual wording, safety-sensitive prompts, and adversarial ambiguity.

Rows may contain:

- `messages` ending in an assistant target;
- `messages` ending in a user request;
- a plain `prompt` or `text` field.

For a decode-oriented router dataset, use `--positions last`. For teacher-forced assistant-token coverage, use `--positions assistant`.

## 2. Collect router examples

```powershell
python .\scripts\phase_6_sparse_simulation\profile_activations.py `
  --model <Phase-4B-selected-model> `
  --calibration data\calibration\calibration.jsonl `
  --block-size 128 `
  --retain-contribution 0.95 `
  --positions last `
  --out artifacts\phase_6_sparse\router_dataset.pt
```

For each layer, the artifact contains:

- `hidden`: MLP input vectors visible to a future router;
- `block_scores`: oracle contribution proxy per contiguous block;
- `oracle_mask`: smallest block set retaining the requested contribution;
- record and token-position metadata.

## 3. Compute the oracle frontier

```powershell
python .\scripts\phase_6_sparse_simulation\simulate_sparsity.py `
  --traces artifacts\phase_6_sparse\router_dataset.pt `
  --ratios 1.0 0.8 0.65 0.5 0.35 `
  --out artifacts\phase_6_sparse\frontier.json
```

This reports retained oracle contribution only. It does not establish generation quality.

## 4. Run real masked generation comparisons

```powershell
python .\scripts\phase_6_sparse_simulation\simulate_sparsity.py `
  --traces artifacts\phase_6_sparse\router_dataset.pt `
  --model <Phase-4B-selected-model> `
  --benchmark data\benchmark\shortchatbench_pilot.jsonl `
  --ratios 1.0 0.8 0.65 0.5 0.35 `
  --out artifacts\phase_6_sparse\frontier_with_generations.json
```

The script replaces each gated MLP with an **oracle-masked dense-compute wrapper**, then saves dense and masked replies for review. Use the official evaluator or human review for correctness, naturalness, context, and safety.

## Verification checklist

- [ ] Dense reference hash and tokenizer hash recorded.
- [ ] Calibration corpus is separate from sealed evaluation.
- [ ] Every layer has sufficient router samples.
- [ ] Block size divides or safely pads the intermediate width.
- [ ] Mean active ratio and contribution retention are inspected per layer.
- [ ] Raw dense/masked generations are reviewed.
- [ ] No speed, VRAM, or active-parameter claim is made from dense masking.
- [ ] Stop Gate 5 passes only when meaningful blocks can be omitted inside the quality limit.

## Outputs

- `router_dataset.pt`: tensor artifact for Phase 7.
- `router_dataset.json`: human-readable per-layer summary.
- `frontier.json`: active-ratio/contribution frontier.
- optional `*_generations.jsonl`: raw dense versus masked replies.
