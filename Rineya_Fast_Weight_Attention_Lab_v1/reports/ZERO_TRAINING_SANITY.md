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
