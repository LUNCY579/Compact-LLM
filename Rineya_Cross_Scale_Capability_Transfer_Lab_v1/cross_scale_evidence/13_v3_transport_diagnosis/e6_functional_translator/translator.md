# v3 E6 — Low-Rank Cross-Scale Functional Translator

9013 token positions, 40 training prompts and
19 held-out prompts. The holdout is prompt-disjoint, not
token-disjoint, because tokens inside a prompt are not independent.

## Can a linear map carry donor representations into the target basis?

| donor -> target | depth | ridge R2 | ridge cos | rank-8 R2 | rank-16 R2 | Procrustes cos | token-shuffled R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen4b -> qwen06 | 0 | +0.9497 | +0.9797 | +0.0284 | +0.0721 | +0.9314 | -0.1864 |
| qwen4b -> qwen06 | 1 | +0.8669 | +0.9256 | +0.1100 | +0.1531 | +0.8462 | -0.4949 |
| qwen4b -> qwen06 | 2 | +0.7857 | +0.8794 | +0.0545 | +0.0995 | +0.7766 | -0.4771 |
| qwen4b -> qwen06 | 3 | +0.8472 | +0.9291 | +0.0706 | +0.1325 | +0.8320 | -0.5920 |
| qwen4b -> qwen06 | 4 | +0.8878 | +0.9473 | +0.1889 | +0.2904 | +0.8828 | -0.4963 |
| qwen4b -> supa | 0 | +0.9493 | +0.9796 | +0.0286 | +0.0716 | +0.9307 | -0.2137 |
| qwen4b -> supa | 1 | +0.8849 | +0.9357 | +0.0849 | +0.1337 | +0.8513 | -0.4868 |
| qwen4b -> supa | 2 | +0.8342 | +0.9028 | +0.0610 | +0.1165 | +0.7817 | -0.4613 |
| qwen4b -> supa | 3 | +0.8477 | +0.9306 | +0.0643 | +0.1120 | +0.8272 | -0.5750 |
| qwen4b -> supa | 4 | +0.8826 | +0.9485 | +0.1777 | +0.2718 | +0.8788 | -0.4563 |
| qwen8b -> qwen06 | 0 | +0.9470 | +0.9818 | +0.1710 | +0.2303 | +0.9249 | -0.3477 |
| qwen8b -> qwen06 | 1 | +0.8695 | +0.9256 | +0.1439 | +0.1715 | +0.8459 | -0.7205 |
| qwen8b -> qwen06 | 2 | +0.7736 | +0.8734 | +0.0504 | +0.0944 | +0.7795 | -0.8058 |
| qwen8b -> qwen06 | 3 | +0.8432 | +0.9273 | +0.0795 | +0.1222 | +0.8341 | -0.9811 |
| qwen8b -> qwen06 | 4 | +0.8914 | +0.9492 | +0.2108 | +0.3337 | +0.8878 | -0.7659 |
| qwen8b -> supa | 0 | +0.9466 | +0.9817 | +0.1615 | +0.2277 | +0.9243 | -0.3298 |
| qwen8b -> supa | 1 | +0.8873 | +0.9363 | +0.1139 | +0.1442 | +0.8515 | -0.7226 |
| qwen8b -> supa | 2 | +0.8271 | +0.8992 | +0.0326 | +0.1150 | +0.7821 | -0.8385 |
| qwen8b -> supa | 3 | +0.8419 | +0.9282 | +0.0584 | +0.1011 | +0.8280 | -0.9791 |
| qwen8b -> supa | 4 | +0.8874 | +0.9503 | +0.2113 | +0.2953 | +0.8850 | -0.7412 |

Best held-out mean cosine: **0.9818**. v1's embedding-fitted maps reached
0.335 (ridge) and 0.536 / 0.275 (Procrustes).

## Does the donor explain anything the target's own base does not?

Predicting Supa's representation from Qwen3-0.6B alone, versus from Qwen3-0.6B plus the
donor. The random column adds the same number of features as the donor, so any gain from
sheer dimensionality is subtracted.

| donor | depth | base only R2 | + donor | + random | donor gain | gain over control |
|---|---:|---:|---:|---:|---:|---:|
| qwen4b | 0 | +0.9852 | +0.9773 | +0.9819 | -0.0079 | -0.0046 |
| qwen4b | 1 | +0.9760 | +0.9691 | +0.9617 | -0.0069 | +0.0074 |
| qwen4b | 2 | +0.9462 | +0.9406 | +0.9142 | -0.0056 | +0.0264 |
| qwen4b | 3 | +0.9594 | +0.9514 | +0.9351 | -0.0080 | +0.0163 |
| qwen4b | 4 | +0.9547 | +0.9504 | +0.9351 | -0.0044 | +0.0153 |
| qwen8b | 0 | +0.9852 | +0.9736 | +0.9756 | -0.0116 | -0.0020 |
| qwen8b | 1 | +0.9760 | +0.9640 | +0.9330 | -0.0120 | +0.0309 |
| qwen8b | 2 | +0.9462 | +0.9339 | +0.8562 | -0.0123 | +0.0778 |
| qwen8b | 3 | +0.9594 | +0.9433 | +0.8905 | -0.0161 | +0.0528 |
| qwen8b | 4 | +0.9547 | +0.9477 | +0.8961 | -0.0070 | +0.0516 |

Best donor increment over the width-matched control: **+0.0778**.
