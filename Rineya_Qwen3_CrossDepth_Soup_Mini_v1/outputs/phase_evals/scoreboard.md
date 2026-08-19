# 28L hybrid chat + RAG eval

- Generated: `2026-08-19T20:41:51Z`
- Protocol: Phase 1 chat quality (`dev.jsonl`, 32 tokens, eos_only) and Phase 9 RAG
  (`dev.reviewed.jsonl`, oracle_rag + real_rag, query_aware_v21). Thinking disabled.
- Same dual-skill DEV protocol as teacher dual-SFT. Not sealed TEST.

| Model | Dual | Chat Q | Task adequacy | Oracle supp | Real supp | Oracle abstain | Real proc |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Aegis 19L (teacher dual-SFT v1 5e-4, PEFT on Pulse)` | 0.8382 | 0.8594 | 0.4978 | 1.0000 | 0.7358 | 1.0000 | 0.9533 |
| `qwen3_x_aegis` | 0.7814 | 0.8602 | 0.5032 | 0.8906 | 0.6736 | 0.8273 | 0.8600 |
| `qwen3_x_v1_5e4_1e5` | 0.7445 | 0.8667 | 0.5289 | 0.8415 | 0.6283 | 0.7266 | 0.7000 |
| `qwen3_x_v1_5e4_v2` | 0.7381 | 0.8619 | 0.5117 | 0.8472 | 0.6283 | 0.6906 | 0.7000 |
| `qwen3_replace_aegis` | 0.7129 | 0.8471 | 0.4540 | 0.8849 | 0.6679 | 0.3022 | 0.8333 |
| `qwen3_replace_v1_5e4_1e5` | 0.7025 | 0.8636 | 0.5129 | 0.8528 | 0.6340 | 0.2878 | 0.7200 |
| `qwen3_replace_v1_5e4_v2` | 0.6823 | 0.8603 | 0.5035 | 0.8698 | 0.6491 | 0.0000 | 0.7733 |

50/50 cores: Nimbus / Lyra / Halo. Halo-donor α=1.0 replace is **Halo Uncensored** (oracle abstain 0.00; pair with Halo). Other replace-kept rows stay lab-only. Aegis row is the prior 19L PEFT dual-SFT v1 5e-4 DEV freeze, not a 28L hybrid.
