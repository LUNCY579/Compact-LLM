# Rineya Teacher Soup Mini v1

Curiosity-only lab: uniform **FP32 reconstructed-delta** soups of teacher dual-SFT adapters on the 19-layer Pulse base, scored on the frozen 64-row long-context **showcase**. This is not a recorded core and must not overwrite Pulse or Aegis.

64 rows is a noisy selection signal (Aegis is 4/64). Do not treat a winning soup here as a deploy decision.

## What is averaged

Phase 4 merge: each PEFT adapter is applied to a fresh Pulse copy, the full-weight **delta** is taken in FP32, deltas are averaged with equal `1/n` weights, and the mean delta is added back to Pulse. That is the same method as Pulse’s `seed_37` + `seed_49` soup. LoRA A/B factors are not averaged directly.

Round A uses the seven finished teacher dual-SFT **v1** LR-sweep adapters (mix 16,410). Soups are **one-by-one with the Aegis ingredient** `v1_lr5e4`: equal 0.5/0.5 of 5e-4 and each other finished adapter. Round B is `v1_lr5e4` with `v2_lr1e4` (mix 44,410). Unfinished v2 5e-4 hot-screen checkpoints are excluded.

## Combinations

See [`configs/combinations.json`](configs/combinations.json).

- **Singles (PEFT, no 1.8GB dump):** seven v1 LRs + `v2_lr1e4`
- **Pairwise soups:** `v1_5e4_1e5`, `v1_5e4_2e5`, `v1_5e4_5e5`, `v1_5e4_1e4`, `v1_5e4_2e4`, `v1_5e4_3e4`
- **Round B:** `v1_5e4_v2` (5e-4 with v2@1e-4)

## Run

From the repo root, with the teacher GPU job left alone:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe Rineya_Teacher_Soup_Mini_v1\scripts\run_lab.py
```

Resume is default: completed 64-row prediction files are skipped. Soup `model.safetensors` files are deleted after eval; `soup_manifest.json` stays under `artifacts/`.

## Eval protocol

Same as prior Pulse / Aegis / Qwen3 showcase runs: greedy decode, 24 new tokens, 4096 context, seed 17. No Qwen3 thinking flag (Rineya chat template).
