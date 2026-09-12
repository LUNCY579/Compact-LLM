# Every published Rineya core can halve its KV cache exactly, for free

This is shape arithmetic, confirmed empirically. It is not a discovery, and it is not a
measurement. It is recorded because it **resets the bar** that every measured KV-reduction
result in this programme has to clear — and most of them do not clear it.

Exploratory. Outside the hashed root — no receipt, no gate. No weights were modified, no
sealed or registered DEV data was touched, no efficacy claim is made.

---

## The argument

A layer caches `2 · n_kv · head_dim` values per token: `n_kv · head_dim` for K and the same
for V. Those values are produced by

    W_KV = [W_K ; W_V]        shape (2 · n_kv · head_dim,  hidden)

whose rank is at most `min(2 · n_kv · head_dim, hidden)`. So when `2 · n_kv · head_dim >
hidden` the matrix is **rank-deficient by shape**, and a latent of rank `hidden` reproduces K
and V exactly at a strictly smaller cache:

    free_exact_reduction = 2 · n_kv · head_dim / min(2 · n_kv · head_dim, hidden)

## Measured across the published family

`kv_shape_scan.py` reads `config.json` only. Ten cores read, **zero skipped**:

| core | hidden | kv heads | head_dim | L | KV/layer | max rank | free |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rineya-Aegis-1.1.0-devfreeze | 1024 | 8 | 128 | 19 | 2048 | 1024 | **2.00×** |
| Rineya-Baya-1.7.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Halo-1.2.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Halo-Uncensored-1.5.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Lyra-1.4.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Nimbus-1.6.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Pulse-1.6.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Supa-1.7.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Supa-A1-1.0.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |
| Rineya-Supa-A1.1-1.1.0-devfreeze | 1024 | 8 | 128 | 28 | 2048 | 1024 | **2.00×** |

Every core in the family: **2.00× exact, zero training, zero quality loss.**

Independently confirmed on real weights, not just on paper — the `r = 1024` control in
`latent_sweep.py` reproduces the source logits to **7.8e-05** in FP32 (roundoff through two
matmuls), for both the plain-SVD and activation-whitened fits.

## What this is not

**Not novel.** At `rank = hidden` the down-projection is an invertible square matrix, so the
latent is just an invertible map of the layer input. This is the standard observation that a
GQA model whose per-layer KV exceeds its width may cache the layer input instead of K and V.
It was always true of these weights.

**Not a measurement.** No weights are loaded by the scan. Exactness follows from the rank
bound. The empirical check above is a confirmation, not the basis of the claim.

**Not general.** It holds exactly when `2 · n_kv · head_dim > hidden`, which is characteristic
of *small* models that shrink `hidden` while keeping `head_dim 128` and a GQA head count.
Verified here for ten Rineya cores and for no other model. A wide model with few KV heads gets
nothing from this.

## Why it is recorded anyway — it reframes the programme

The measured sharing frontier, from `evidence/arm-quality/ARCHITECTURE_COMPARISON.json`. All
rows are the **same 24-row calibration split**, so they are directly comparable; the
uniform-distribution reference on that split is `ln(151936) = 11.9312`.

| approach | KV | CE | usable? |
|---|---:|---:|---|
| source (reference) | 1.000× | 5.5458 | — |
| `ced last 4` (best usable sharing point) | 0.893× | 6.4065 | marginal |
| `ced last 14` (C28) | 0.536× | 12.5854 | no — worse than uniform |
| `local group 2` | 0.500× | 12.9136 | no |
| **latent `r = 1024`** | **0.500×** | **5.5458, exact by construction** | **yes, free** |

The latent row is not an independent measurement: at `r = 1024` the factorisation reproduces K
and V, so its CE *is* the source's CE on whatever split is used. That is the point.

At **the same 0.500× budget** where every sharing topology produces output worse than the
uniform distribution, shape arithmetic is exact. **Every KV saving the CED family can offer is
already beaten without an experiment.**

Two consequences, both binding:

1. No KV-reduction result in this programme below 2× is worth reporting as an achievement.
2. The only interesting region for per-layer latent compression is **rank < 1024**, which is
   what `latent_sweep.py` measures, at budget-matched points (0.500×, 0.250×, 0.143×, 0.071×)
   that were **pre-declared** against the sharing frontier's measured ratios rather than chosen
   after seeing which ranks did well.

## Reproduction

```bash
C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe experiments/rab001_exploration/kv_shape_scan.py
```

Reads `config.json` from each published core. Loads no weights. Artifact:
`evidence/latent/KV_SHAPE_SCAN.json`.
