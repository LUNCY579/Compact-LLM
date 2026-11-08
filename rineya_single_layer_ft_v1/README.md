# Single-Layer Fine-Tune Lab v1

Exploratory smoke: **LoRA on Supa-A1.1 layer 8 only**, Puzzle-51K easy `think_then_short`, dual eval (puzzle val exact + layer-8 identity-skip ablation).

## Hypothesis

Layer 8 is the strongest **null** under leave-one-out identity residual skip (cos ≈ 0.978). Training only that block may (a) raise puzzle exact on a held-out easy val subsample and (b) make skipping L8 more catastrophic. That would be evidence the layer can be made load-bearing — not proof that “redundant ⇒ free capacity.”

## Locked choices

| Knob | Value |
|---|---|
| Base | `Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze` (FP16, trust_remote_code) |
| Adapter | LoRA r=16 α=32 on layer **8** attn+MLP only |
| Not used | QLoRA, DoRA, multi-layer, layers 0–2 |
| Data | Puzzle easy only; never test/challenge |
| Vault | Published cores never overwritten; adapters under `artifacts/` |

## Dual metrics (“activated”)

1. Puzzle val exact: no material regression vs intact Supa on the same subsample.
2. Ablation: L8 skip cosine **drops** or catastrophe score **rises** vs pre-FT.

Both required for `activated_claim_eligible`. An all-layer matched-budget control is deferred.

## Run smoke

From repo root:

```powershell
$env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"
$lab = "rineya_single_layer_ft_v1\scripts"

& $py $lab\build_puzzle_view.py
& $py $lab\train_layer_lora.py
& $py $lab\eval_puzzle_val.py --condition baseline
& $py $lab\eval_puzzle_val.py --condition layer8_lora
& $py $lab\rerun_layer8_ablation.py --condition baseline
& $py $lab\rerun_layer8_ablation.py --condition layer8_lora
& $py $lab\write_smoke_summary.py
```

Or: `.\rineya_single_layer_ft_v1\scripts\run_smoke.ps1`

Summary lands at `rineya_single_layer_ft_v1/reports/layer8_smoke_summary.json` (and under the run output dir).

## Layout

| Path | Role |
|---|---|
| `configs/layer8_lora_puzzle_smoke.yaml` | Locked smoke config |
| `data/supa_layer_ablation_smoke.jsonl` | 8-prompt bank (same as protocol 15) |
| `scripts/train_layer_lora.py` | Selective-layer LoRA trainer |
| `scripts/eval_puzzle_val.py` | Intact vs adapter puzzle exact |
| `scripts/rerun_layer8_ablation.py` | Identity-skip L8 only |
| `artifacts/rineya_single_layer_ft_v1/...` | Views, adapter, eval (gitignored) |
