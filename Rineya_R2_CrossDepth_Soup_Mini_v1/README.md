# Rineya R2 Cross-Depth Soup Mini v1

Apply the published Qwen3 CrossDepth stitch formulas to **R2** (`rineya-core-en32k-28L`): same 28-layer Qwen3 decoder geometry, **vocab 32,768**.

The read-only source is `C:\Users\clclc\RineyaExperiment\artifacts\rineya-core-en32k-28L`. This lab holds a **local copy** under `scaffolds/`. Do not write back to RineyaExperiment.

## Formula

On the 19 Phase-2B kept decoder blocks:

`(1 − α) · R2_block + α · 19L_donor_block`

Dropped R2 layers plus `embed_tokens`, final norm, and tied `lm_head` stay the R2 scaffold (32K tokenizer). Donors keep the Qwen3 151936 vocab; those embeddings are never blended.

## Hybrids

See [`configs/hybrids_r2.json`](configs/hybrids_r2.json).

| Name | α | Donor | Qwen3 analogue |
|---|---|---|---|
| `r2_x_aegis` | uniform 0.5 | vault Aegis 1.1 | Nimbus 1.3 |
| `r2_x_v1_5e4_1e5` | uniform 0.5 | v1@5e-4 + v1@1e-5 on Pulse | Lyra |
| `r2_x_v1_5e4_v2` | uniform 0.5 | v1@5e-4 + v2@1e-4 on Pulse | Halo |
| `r2_x_v1_5e4_v2_a1` | uniform 1.0 | same Halo donor | Halo Uncensored |
| `r2_late_rineya` | late_rineya ramp, mean 0.5 | same Halo donor | Pulse 1.6 |

CPU only (`CUDA_VISIBLE_DEVICES` empty). Hybrid weights are kept for eval.

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe Rineya_R2_CrossDepth_Soup_Mini_v1\scripts\preflight_shapes.py
.\.venv\Scripts\python.exe Rineya_R2_CrossDepth_Soup_Mini_v1\scripts\run_lab.py
.\.venv\Scripts\python.exe Rineya_R2_CrossDepth_Soup_Mini_v1\scripts\score_cpu.py
```
