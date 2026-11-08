# Implementation audit

Variant: **FALCON-2 prefix-aligned**, auxiliary pathway, rank 16, KV-head state.

## What is FALCON vs what is ours

- Recurrence, RAW read-after-write, residual NLMS write, per-column `η`: paper (A).
- Auxiliary merge `h + α P(m)`, low rank 16, layer-14 only, frozen Supa: engineering (B).
- Significance gate: Rineya-only (C).

## Parameter counts (one instrumented layer, hidden=1024, 8 KV heads, rank=16)

| Arm | Trainable FWA tensors | Recurrent state bytes (fp32, B=1) |
|---|---:|---:|
| A | 0 | 0 |
| B | 524,593 | 0 (no carry) |
| C | 524,593 | 8,192 |
| D | 525,618 | 8,192 |

B and C match exactly. D adds `Linear(1024 → 1)` (1,025 parameters).

State formula: `n_kv_heads * rank * rank * 4 = 8 * 16 * 16 * 4 = 8192`.

## FLOPs proxy per token (one layer)

Rank-16 FALCON-2 scan: `O(H * R^2)` state update plus three `hidden→H*R` projections
and one `H*R→hidden` projection. Dominated by the frozen 0.6B backbone, not by FWA.

## Loader

`resolve_supa_path()` reads the immutable core. Writes are refused under both this
worktree's `Rineya_Published_Cores/` and the sibling Chat-Bubble vault.
