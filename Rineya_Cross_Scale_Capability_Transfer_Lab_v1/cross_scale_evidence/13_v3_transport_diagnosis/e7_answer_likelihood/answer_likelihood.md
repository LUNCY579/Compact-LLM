# v3 E7 — Is There a Capability Gap to Transfer?

Teacher-forced log-probability of the reference answer, 55 of 59
calibration cases, mean per answer token. Higher is better. One forward pass per
model, no generation.

| model | mean log-prob / answer token | median |
|---|---:|---:|
| qwen4b | -7.1641 | -4.8616 |
| qwen8b | -4.2065 | -0.0000 |
| qwen06 | -4.1673 | -2.9574 |
| supa | -1.6764 | -0.1052 |

## Against supa

| comparison | mean delta | median delta | cases higher | win rate |
|---|---:|---:|---:|---:|
| qwen4b__vs__supa | -5.4877 | -3.0931 | 23/55 | 0.418 |
| qwen8b__vs__supa | -2.5301 | +0.0025 | 37/55 | 0.673 |
| qwen06__vs__supa | -2.4910 | -1.5508 | 11/55 | 0.200 |
