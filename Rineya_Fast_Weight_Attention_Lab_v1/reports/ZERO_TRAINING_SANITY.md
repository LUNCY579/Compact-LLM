# Zero-training sanity

Date: 2026-09-01. Interpreter: Chat-Bubble `.venv` Python, torch CUDA on RTX 4070 Laptop (8.0 GB).
No scientific claims. Implementation-only.

## Unit suite (CPU)

`pytest tests --ignore tests/test_model_alpha_zero.py` → **21 passed**.

| # | Invariant | Test | Result |
|---|---|---|---|
| 1 | Causal state update | `test_causal_state_update_future_tokens_do_not_change_past` | pass |
| 2 | No future-token leakage | `test_no_future_token_leakage_in_gradients` | pass |
| 3 | Deterministic recurrence | `test_deterministic_recurrence` | pass |
| 4 | State reset | `test_state_reset_between_sequences` | pass |
| 5 | Batch isolation | `test_batch_isolation` | pass |
| 6 | Sequence-boundary sentinel | `test_sequence_boundary_sentinel_x0_is_zero_write` | pass |
| 7 | Padding mask | `test_padding_mask_does_not_update_state` | pass |
| 8 | Dtype (fp16 in, fp32 state) | `test_dtype_state_is_fp32` | pass |
| 9 | Save/load | `test_save_load_block` | pass |
| 10 | Long-sequence stability (T=256) | `test_long_sequence_numerical_stability` | pass |
| 11 | State-size accounting | `test_state_size_accounting` | pass |
| 12 | α=0 reproduces inner module | `test_alpha_zero_reproduces_inner_attention`, `test_eval_alpha_zero_matches_unwrapped_logits` | pass (exact) |

T=2048 was not run in the unit job; T=256 is the committed CPU budget. The scan is the same loop.

## Real Supa-A1.1 (GPU)

`tests/test_model_alpha_zero.py` → **passed** in 62 s.

Checkpoint resolved read-only from the sibling vault
`Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze` because this
worktree's copy of the core has gitignored weights. No bytes were written there.

Arm C installed on layer 14 with `init_alpha=0`. Short-prompt logits matched the
unmodified model within `rtol=1e-3, atol=1e-2` (exact skip in eval).

## Stop-gate

α=0 recovers ordinary Supa behaviour. Training is allowed to proceed.

---

## Correction (appended after the P2 post-mortem)

Two rows of the table above overstated what was checked.

**Row 5, "Batch isolation → pass", was vacuous.** `falcon2_scan` computed

```python
outputs[:, t] = mem * gate.squeeze(-1)
```

`gate` is `(B,1,1)` and `mem` is `(B,H,R)`. The squeeze makes the gate `(B,1)`,
so broadcasting right-aligns it and matches the gate's **batch** axis against the
memory's **head** axis:

- `B != H` → `RuntimeError`;
- `B == H` → no error, but row *j*'s validity is applied to head *j*.

`test_batch_isolation` used `batch=2, heads=2`, i.e. `B == H`, with all tokens
valid — so the gate was all ones and the test passed while the bug was present.
Row 7 (padding mask) only ever ran at `batch=1`, where the bug is invisible.

No recorded result is affected: every P1/P2/P2b run uses batch 1.

Fixed by dropping the squeeze. Two new regressions cover both halves:

| # | Invariant | Test |
|---|---|---|
| 5a | batch > 1 with `B != H` runs, rows do not interact | `test_batch_isolation_and_batch_gt_one` (B=3, H=4) |
| 7a | with `B == H`, padding masks the padded **row**, not the same-index head | `test_padding_gate_is_applied_per_row_not_per_head` (B=H=4, one row padded) |
| 7b | padded positions emit zero and cannot influence real positions | `test_padding_is_masked_out` |

**Row 12, "α=0 reproduces inner module", is correct but load-bearing in a way
worth stating.** `P.weight` is zero-initialised, so at exactly `α = 0` both
`dL/dα` and `dL/dP` are identically zero and the block is a permanent no-op.
`α = 0` is therefore valid as the *equivalence* check and invalid as a *training*
initialisation. `train.py` hardcoded `init_alpha = 0.05` while both configs
declared `0.0`; from P2b the value is passed by the caller and recorded in the
training manifest.

**Row 10, "long-sequence numerical stability (T=256)", does not cover the
evaluated range.** P2b evaluates to 4096 tokens. The scan is the same loop, but
the retention arithmetic is not scale-free: at the P1/P2 initialisation the
ridge decay is 0.99211 per token, so a 4096-token prefix is attenuated by
`1e-14`. That is numerically stable and semantically empty — see
`reports/MECHANISM_DIAGNOSTIC.md` §2.
