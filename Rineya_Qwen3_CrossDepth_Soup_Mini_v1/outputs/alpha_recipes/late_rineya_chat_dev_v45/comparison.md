# late_rineya vs Halo family — chat DEV v4.5

Same Quality Evaluator v4.5 / `official_no_think` contract as the Halo-pair soup lab.
`late_rineya` was scored on **CPU**; Halo rows are from the soup-lab GPU run.

| Name | Mean quality | Worst decile | Complete | Obs pass | Median tok | Dup rate | Objective | Showcase |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen3_06b (α=0) | 0.8334 | 0.6987 | 0.9843 | 0.9941 | 15.0 | 0.2706 | 0.8108 | 14/64 |
| Halo uniform α=0.5 | 0.8619 | 0.7064 | 1.0000 | 0.9961 | 11.0 | 0.3627 | 0.8397 | 30/64 |
| global_best α=0.40 | 0.8643 | 0.7128 | 0.9922 | 0.9980 | 12.0 | 0.3471 | 0.8423 | n/a yet |
| Halo Uncensored α=1 | 0.8601 | 0.7133 | 1.0000 | 0.9980 | 9.0 | 0.4333 | 0.8377 | 31/64 |
| late_rineya | 0.8616 | 0.7065 | 1.0000 | 0.9961 | 9.0 | 0.3333 | 0.8404 | 31/64 |

Notes:
- Showcase: `late_rineya` 31/64 is +1 vs Halo 30/64 (noise band); not a vault core.
- Chat DEV objective is the soup-lab weighted score, not showcase hits.
