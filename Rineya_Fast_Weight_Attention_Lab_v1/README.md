# Rineya Fast Weight Attention Lab v1

Exploratory, low-resource test of a FALCON-2-inspired **auxiliary** fast-weight
memory pathway on frozen **Supa-A1.1**. Negative results are valid and must be
kept.

This lab does **not** assume FALCON is beneficial.

## Isolation record

| Field | Value |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\RineyaBeta-supa-fwa-v1` |
| Branch | `research/supa-a1.1-fast-weight-attention-v1` |
| Base branch | `research/positional-sequential-attention` |
| Base commit | `ee13448f11efaa7f14206dab95b277816b912ce6` |
| Worktree created (UTC) | `2026-09-01T22:04:34Z` |
| Source checkpoint | `Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze` |
| Core ID | `rineya.core.supa-a1.1.v1.1.0-devfreeze.28L.qwen3-nimbus-lcr-step200-scale-0p8125-fp16-abstain-b5p15625` |
| Weights SHA-256 | `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026` |
| Store dtype | float16 (mandatory) |
| Architecture | 28-layer Qwen3-family, hidden 1024, 16 Q / 8 KV heads, head_dim 128 |
| Abstention | token 51378, `+5.15625` logit offset in `Qwen3AbstainForCausalLM.forward` |

The checkpoint directory is **immutable**. No file in this lab writes under
`Rineya_Published_Cores/`.

## Scientific questions

- **Q1.** Can a FALCON-style fast-weight memory improve frozen Supa-A1.1?
- **Q2.** If so, is it the recurrent dynamics, or merely extra parameters?
- **Q3.** Does it change long-context positional robustness (middle vs edge)?
- **Q4.** Does significance-gated writing beat ordinary FWA writing?
- **Q5.** What are the memory, latency, throughput, and VRAM tradeoffs?

## Arms

| Arm | Meaning |
|---|---|
| A | Unmodified Supa-A1.1 |
| B | Parameter-matched **static** control (no recurrent state carry) |
| C | Frozen Supa + prefix-aligned FALCON-2 pathway |
| D | Arm C + Rineya significance-gated plasticity/forgetting |
| E | Deferred until A–D are interpretable |

Primary contrasts: **C−A**, **C−B**, **D−C**.

## Layout

```
Rineya_Fast_Weight_Attention_Lab_v1/
  README.md
  PREREGISTRATION.md
  research/PAPER_NOTES.md
  configs/
  src/rineya_fwa/
  scripts/
  tests/
  reports/
artifacts/supa_a1_1_fast_weight_attention_v1/   # gitignored run outputs
```

## Sealed-data ban

Do not open, train on, or iterate against:

- CC0 `sealed_eval/`
- Invisible-Limit TEST / TEST2
- Demo v1 showcase64 / stability192 as a tuning set
- PSA `data/sealed/`

## Status

P2 complete for seeds 11 and 22. Decision: **NO-GO**. See
`reports/FINAL_INTERPRETATION.md`.
