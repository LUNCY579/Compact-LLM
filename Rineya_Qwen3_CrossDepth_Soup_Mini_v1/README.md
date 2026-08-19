# Rineya Qwen3 Cross-Depth Soup Mini v1

Curiosity-only lab: start from official **28-layer Qwen3-0.6B**, 50/50-average the **19 decoder blocks** that Pulse/Aegis still have, and leave the nine dropped Qwen3 layers (plus embeddings, final norm, and tied lm_head) unchanged.

The 50/50 hybrids are recorded as **Nimbus** (`qwen3_x_aegis`), **Lyra** (`qwen3_x_v1_5e4_1e5`), and **Halo** (`qwen3_x_v1_5e4_v2`). The Halo-donor **α=1.0** replace is **Halo Uncensored**. Do not overwrite Pulse, Aegis, Halo, Halo Uncensored, Nimbus, Lyra, or the local Qwen3 snapshot.

64 rows is a noisy selection signal. Do not treat a hybrid that beats 14/64 here as a deploy decision. Dropped Qwen3 layers still assume their original neighbors.

## Layer map

Phase 2B v4 stage 09 kept original Qwen3 indices

`[0, 1, 2, 3, 4, 5, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 25, 26, 27]`

and dropped

`[6, 7, 8, 9, 10, 14, 18, 23, 24]`.

See [`configs/layer_map.json`](configs/layer_map.json).

## Hybrids

See [`configs/hybrids.json`](configs/hybrids.json).

- `qwen3_x_aegis` — vault Aegis donor, 50/50 (**24/64**; recorded as Nimbus)
- `qwen3_x_v1_5e4_1e5` — rematerialized 5e-4 + 1e-5 teacher soup, 50/50 (**28/64**; recorded as Lyra)
- `qwen3_x_v1_5e4_v2` — rematerialized 5e-4 + v2@1e-4 teacher soup, 50/50 (**30/64**; recorded as Halo)
- `qwen3_x_v2_lr5e4` — v2@5e-4 alone, 50/50 (26/64)
- `qwen3_x_v2_5e4_v1_5e4` — v2@5e-4 + v1@5e-4, 50/50 (26/64)
- `qwen3_x_v2_1e4_5e4` — v2@1e-4 + v2@5e-4, 50/50 (26/64)
- `qwen3_x_v1_1e4_v2_5e4` — v1@1e-4 + v2@5e-4, 50/50 (24/64)
- `qwen3_replace_v1_5e4_v2` — Halo donor **α=1.0** (**31/64**; recorded as **Halo Uncensored**; pair with Halo)
- `qwen3_replace_*` — other donors with **α=1.0**: kept Qwen3 blocks fully replaced; lab-kept, not vaulted

## Run

From the repo root:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe Rineya_Qwen3_CrossDepth_Soup_Mini_v1\scripts\run_lab.py
```

Resume is default. 50/50 hybrid and rematerialized 19L `model.safetensors` files are deleted after showcase eval; manifests stay. **Replace-kept** hybrids (`qwen3_replace_*`) keep their dense weights for later pair experiments.

Phase 1 chat quality + Phase 9 RAG DEV for the original three hybrids:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe Rineya_Qwen3_CrossDepth_Soup_Mini_v1\scripts\run_phase_evals.py
```

That rematerializes lab copies as needed, uses the Halo vault for `qwen3_x_v1_5e4_v2`, and writes `outputs/phase_evals/scoreboard.md`.

## Eval protocol

Same frozen 64-row showcase: greedy, 24 new tokens, 4096 context, seed 17, **`--disable-thinking`** (official Qwen3 chat template). Compare to Qwen3-0.6B 14/64 and Aegis 4/64.
