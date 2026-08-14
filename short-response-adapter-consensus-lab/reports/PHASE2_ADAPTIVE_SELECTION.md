# Phase 2 Adaptive Consultation Selection

Role: **DEV MODEL-SELECTION / CALIBRATION**. The predeclared core is Track A + Track B + General. The policy escalates to all seven experts when K3 agreement is strictly below the threshold.

| Threshold | Token acc. | All-token example acc. | Escalation | Mean forwards/token | Forward reduction | False non-escalation |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 78.81% | 23.26% | 0.00% | 3.00 | 57.14% | 1.22% |
| 0.34 **SELECTED** | 79.12% | 25.11% | 5.90% | 3.24 | 53.77% | 0.46% |
| 0.67 | 78.49% | 25.38% | 19.34% | 3.77 | 46.09% | 0.00% |

Full K7 anchor: 78.33% token, 25.26% all-token example, 7.00 forwards/token.
Core K3 anchor: 78.81% token, 23.26% all-token example, 3.00 forwards/token.

Selected threshold: **0.34** under the frozen rule: minimum mean forwards among policies no worse than full hard by 0.5 token pp and 1.0 all-token-example pp; otherwise minimum-forward lexicographic fallback.

This cached study chooses a DEV policy; it does not establish free-running accuracy retention or measured latency. Those require a separately frozen reference-decoder run.
