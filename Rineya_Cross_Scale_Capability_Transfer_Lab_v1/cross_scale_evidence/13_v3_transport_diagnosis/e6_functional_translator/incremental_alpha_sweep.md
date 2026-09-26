# v3 E6b — Incremental Test, Ridge Penalty Sweep

6115 training tokens against 1024 + donor-width columns, so the
joint fit sits near the rank-deficient regime. If E6's 'the donor adds nothing' result
were a variance artifact, a stronger penalty would reveal it.

| donor | depth | alpha | base only | + donor | + random | donor gain |
|---|---:|---:|---:|---:|---:|---:|
| qwen4b | 2 | 0.001 | +0.9424 | +0.9096 | +0.9061 | -0.0328 |
| qwen4b | 2 | 0.01 | +0.9462 | +0.9406 | +0.9140 | -0.0056 |
| qwen4b | 2 | 0.1 | +0.9333 | +0.9382 | +0.8968 | +0.0048 |
| qwen4b | 2 | 1 | +0.8565 | +0.8854 | +0.8314 | +0.0289 |
| qwen4b | 2 | 10 | +0.6506 | +0.7328 | +0.6448 | +0.0822 |
| qwen4b | 2 | 100 | +0.3047 | +0.4526 | +0.3041 | +0.1479 |
| qwen4b | 3 | 0.001 | +0.9562 | +0.9234 | +0.9310 | -0.0328 |
| qwen4b | 3 | 0.01 | +0.9594 | +0.9514 | +0.9353 | -0.0080 |
| qwen4b | 3 | 0.1 | +0.9428 | +0.9455 | +0.9112 | +0.0027 |
| qwen4b | 3 | 1 | +0.8668 | +0.8980 | +0.8430 | +0.0312 |
| qwen4b | 3 | 10 | +0.6475 | +0.7473 | +0.6408 | +0.0998 |
| qwen4b | 3 | 100 | +0.2538 | +0.4223 | +0.2530 | +0.1685 |
| qwen4b | 4 | 0.001 | +0.9507 | +0.9309 | +0.9320 | -0.0199 |
| qwen4b | 4 | 0.01 | +0.9547 | +0.9504 | +0.9349 | -0.0044 |
| qwen4b | 4 | 0.1 | +0.9371 | +0.9415 | +0.9074 | +0.0044 |
| qwen4b | 4 | 1 | +0.8685 | +0.8977 | +0.8482 | +0.0291 |
| qwen4b | 4 | 10 | +0.7113 | +0.7897 | +0.7066 | +0.0785 |
| qwen4b | 4 | 100 | +0.3574 | +0.5490 | +0.3566 | +0.1916 |
| qwen8b | 2 | 0.001 | +0.9424 | +0.8860 | +0.8177 | -0.0565 |
| qwen8b | 2 | 0.01 | +0.9462 | +0.9339 | +0.8558 | -0.0123 |
| qwen8b | 2 | 0.1 | +0.9333 | +0.9362 | +0.8613 | +0.0028 |
| qwen8b | 2 | 1 | +0.8565 | +0.8909 | +0.8170 | +0.0344 |
| qwen8b | 2 | 10 | +0.6506 | +0.7513 | +0.6413 | +0.1007 |
| qwen8b | 2 | 100 | +0.3047 | +0.4847 | +0.3036 | +0.1800 |
| qwen8b | 3 | 0.001 | +0.9562 | +0.9002 | +0.8705 | -0.0560 |
| qwen8b | 3 | 0.01 | +0.9594 | +0.9433 | +0.8904 | -0.0161 |
| qwen8b | 3 | 0.1 | +0.9428 | +0.9430 | +0.8795 | +0.0001 |
| qwen8b | 3 | 1 | +0.8668 | +0.9032 | +0.8291 | +0.0364 |
| qwen8b | 3 | 10 | +0.6475 | +0.7695 | +0.6370 | +0.1220 |
| qwen8b | 3 | 100 | +0.2538 | +0.4630 | +0.2526 | +0.2092 |
| qwen8b | 4 | 0.001 | +0.9507 | +0.9286 | +0.8876 | -0.0222 |
| qwen8b | 4 | 0.01 | +0.9547 | +0.9477 | +0.8962 | -0.0070 |
| qwen8b | 4 | 0.1 | +0.9371 | +0.9418 | +0.8772 | +0.0047 |
| qwen8b | 4 | 1 | +0.8685 | +0.9037 | +0.8364 | +0.0352 |
| qwen8b | 4 | 10 | +0.7113 | +0.8028 | +0.7037 | +0.0915 |
| qwen8b | 4 | 100 | +0.3574 | +0.5932 | +0.3561 | +0.2358 |

## The correct test: each model at its own optimal penalty

Ridge shrinkage degrades both fits. Where the base is over-shrunk, correlated donor
columns partially compensate, which looks like a gain without being new information.
So the question is whether the donor raises the *best achievable* held-out fit.

| donor | depth | best base only | best base + donor | delta |
|---|---:|---:|---:|---:|
| qwen4b | 2 | +0.9462 | +0.9406 | -0.0056 |
| qwen4b | 3 | +0.9594 | +0.9514 | -0.0080 |
| qwen4b | 4 | +0.9547 | +0.9504 | -0.0044 |
| qwen8b | 2 | +0.9462 | +0.9362 | -0.0101 |
| qwen8b | 3 | +0.9594 | +0.9433 | -0.0161 |
| qwen8b | 4 | +0.9547 | +0.9477 | -0.0070 |

Largest naive per-penalty increment: +0.2358 at qwen8b depth 4 alpha 100 — but there base-only R2 is only +0.3574 against +0.9594 at its own optimum.

**E6's conclusion is robust: the donor never raises the best achievable fit.**
