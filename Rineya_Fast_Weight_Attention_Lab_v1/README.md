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

**The P2 NO-GO is withdrawn as unsupported.** It was the null result of an
experiment that could not have produced anything else:

| What P2 concluded | What measurement shows |
|---|---|
| "C = B, so recurrence adds nothing" | The arms differed by 0.12 % / 1.41 % / 3.22 % of the attention output (B/C/D) and up to 3.4 logits. The eval could not see it. |
| "no position benefit" | Retention half-life was **87.5 tokens** against ~990-token contexts, with full overwrite on key collision. No fact could reach its query. |
| "14/14 on the position suite" | Filler was one sentence repeated; answer space was 10 colours; one item per cell, so a paired test needed 6/14 items to flip for p<0.05. |
| lengths 512-4096 per config | `run_experiment.py` hardcoded `[256, 512]`. |
| `init_alpha: 0.0` per config | `train.py` hardcoded `0.05`. |
| corpus = unsealed CC0 | runs used `data/synthetic/` (10 answers); the committed CC0 shard was never read. |

See `reports/MECHANISM_DIAGNOSTIC.md`.

**P2b** re-runs the matched arms with each defect fixed (`PREREGISTRATION.md`
Amendment 1, `configs/arms_abcd_p2b.yaml`): token-exact 1K/2K/4K contexts, real
CC0 filler, two-hop `vault -> unit -> code` binding with 24 confusable decoys, a
552-code single-token answer space scored by candidate argmax in one forward
pass, 168 position items, 56 interference items, paired exact McNemar with
Wilson intervals, and a retention horizon matched across arms that can actually
span the context — plus an arm C-short ablation holding the original horizon.

Calibrated so the frozen backbone is **off ceiling**: arm A scores 0.778 at 2048
tokens, and 0.333 at relative position 0.05.

Hardware envelope, measured: 1K/2K/4K native (3.64 GiB at 4K); 8K only via
host-memory spill at ~6x slowdown; **16K OOMs and is not evaluated**.
