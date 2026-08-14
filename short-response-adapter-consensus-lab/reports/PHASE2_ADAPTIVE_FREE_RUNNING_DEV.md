# Phase 2 Adaptive Free-Running DEV Analysis

Role: **DEV EFFICIENCY CONFIRMATION / SAME-SPLIT**. The K3 core, K7 escalation set, and threshold were frozen before these outputs were inspected.

| Condition | Correct | Exact acc. | Mean forwards | Forwards/token | Mean latency (s) |
|---|---:|---:|---:|---:|---:|
| hard_plurality | 1025/4050 | 25.31% | 34.15 | 7.00 | 1.919 |
| adaptive_k3_to_k7 | 1017/4050 | 25.11% | 14.83 | 3.28 | 0.928 |

Adaptive minus hard accuracy: **-0.20 pp** (paired bootstrap 95% CI [-0.77, +0.37] pp; adaptive-only/hard-only 63/71; exact McNemar p=5.455e-01).

The adaptive policy reduced measured expert-token forwards by **56.58%**. Consultation was triggered on 7.12% of generated token decisions and in 29.75% of examples.

Latency is reported as an environment-specific observation; forward-pass reduction is the primary efficiency result.
