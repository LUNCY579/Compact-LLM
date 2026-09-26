# v3 E2 — Target-Basis Alignment of the Transported Direction

Reference frame: `Delta_Supa = Supa-A1.1 - Qwen3-0.6B`, the target model's own
adaptation direction, measured over tensors where it is non-zero.

## Does any transported direction resemble the target's own adaptation?

| Direction | cos vs Delta_Supa | sign agreement | ||delta|| / ||Delta_Supa|| |
|---|---:|---:|---:|
| control_norm_matched_random | +0.0000 | 0.4915 | 0.4051 |
| control_sign_shuffled | -0.0001 | 0.3498 | 0.4051 |
| delta_4b | +0.0002 | 0.3150 | 0.3816 |
| delta_8b | +0.0001 | 0.3420 | 0.4427 |
| delta_mean | +0.0002 | 0.3499 | 0.4051 |
| delta_p5_4b | +0.0007 | 0.3635 | 7.3036 |
| control_layer_shuffled | +0.0001 | 0.3457 | 0.4051 |

## Does Supa's own adaptation use the subspace P4 writes into?

- Tensors measured: 196, subspace rank 8.
- Mean fraction of `Delta_Supa` energy inside P4's rank-8 target subspace: **7.2443e-06**.
- Chance level for an isotropic direction: 4.3597e-06.
- Enrichment over chance: **1.7x**.
- Mean fraction parallel to the weight itself: 1.7714e-06.
- Mean `||Delta_Supa|| / ||W||`: 1.5563e-02.

## Effective rank

| Tensor | Delta_Supa | donor mean | full rank |
|---|---:|---:|---:|
| model.layers.0.mlp.down_proj.weight | 353.6 | 517.7 | 1024 |
| model.layers.7.mlp.down_proj.weight | 50.3 | 512.8 | 1024 |
| model.layers.14.mlp.down_proj.weight | 51.6 | 447.9 | 1024 |
| model.layers.21.mlp.down_proj.weight | 352.2 | 468.4 | 1024 |
| model.layers.27.mlp.down_proj.weight | 386.6 | 538.8 | 1024 |
