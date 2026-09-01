# v3 E4 — Cross-Model Functional Alignment Atlas

59 frozen calibration prompts, prompts only, CPU float32.
Token pooling: **mean** (mean over all prompt positions, a robustness check on the final-token view).
Depth slots: embedding, layer_0, layer_7, layer_14, layer_21, layer_27.

Donor-side rows (Qwen3-4B, Qwen3-8B) are absent: those checkpoints are no longer
resident on this host. See `protocol/atlas_models.json`.

## Representation similarity vs the reference base (mean cosine, mean pooling)

| pair | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| qwen06__vs__supa | 1.0000 | 0.9975 | 0.9997 | 0.9994 | 0.9977 | 0.9558 |
| qwen06__vs__p4_4b_s010 | 1.0000 | 0.9999 | 0.9999 | 0.9999 | 0.9998 | 0.9969 |
| qwen06__vs__p4_8b_s010 | 1.0000 | 0.9998 | 1.0000 | 1.0000 | 0.9999 | 0.9986 |
| qwen06__vs__p5_4b_s010 | 0.9972 | 0.9974 | 0.9994 | 0.9994 | 0.9987 | 0.9844 |
| qwen06__vs__cons_mean_a1500 | 1.0000 | 0.9975 | 0.9997 | 0.9994 | 0.9977 | 0.9559 |
| qwen06__vs__ties_k50_a1500 | 1.0000 | 0.9975 | 0.9997 | 0.9994 | 0.9977 | 0.9558 |

## Linear CKA vs the reference base

| pair | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| qwen06__vs__supa | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 0.9996 | 0.9873 |
| qwen06__vs__p4_4b_s010 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9994 |
| qwen06__vs__p4_8b_s010 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 |
| qwen06__vs__p5_4b_s010 | 1.0000 | 0.9995 | 1.0000 | 1.0000 | 0.9996 | 0.9964 |
| qwen06__vs__cons_mean_a1500 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 0.9995 | 0.9874 |
| qwen06__vs__ties_k50_a1500 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 0.9995 | 0.9874 |

## Logit space

| pair | logit cosine | symmetric KL | top-1 agreement |
|---|---|---|---|
| qwen06__vs__supa | 0.7569 | 7.9879 | 0.3898 |
| qwen06__vs__p4_4b_s010 | 0.9909 | 0.1265 | 0.9492 |
| qwen06__vs__p4_8b_s010 | 0.9943 | 0.0915 | 0.9661 |
| qwen06__vs__p5_4b_s010 | 0.9495 | 1.3732 | 0.7627 |
| qwen06__vs__cons_mean_a1500 | 0.7567 | 7.9908 | 0.4068 |
| qwen06__vs__ties_k50_a1500 | 0.7567 | 7.9948 | 0.4068 |
| supa__vs__p4_4b_s010 | 0.7584 | 7.9307 | 0.3898 |
| supa__vs__p4_8b_s010 | 0.7641 | 7.5567 | 0.4237 |
| supa__vs__p5_4b_s010 | 0.7637 | 5.7941 | 0.5424 |
| supa__vs__cons_mean_a1500 | 1.0000 | 0.0005 | 0.9831 |
| supa__vs__ties_k50_a1500 | 1.0000 | 0.0006 | 0.9831 |
| p4_4b_s010__vs__p4_8b_s010 | 0.9974 | 0.0281 | 0.9492 |
| p4_4b_s010__vs__p5_4b_s010 | 0.9519 | 1.3669 | 0.7458 |
| p4_4b_s010__vs__cons_mean_a1500 | 0.7587 | 7.9316 | 0.4068 |
| p4_4b_s010__vs__ties_k50_a1500 | 0.7588 | 7.9355 | 0.4068 |
| p4_8b_s010__vs__p5_4b_s010 | 0.9533 | 1.1682 | 0.7627 |
| p4_8b_s010__vs__cons_mean_a1500 | 0.7642 | 7.5581 | 0.4407 |
| p4_8b_s010__vs__ties_k50_a1500 | 0.7643 | 7.5618 | 0.4407 |
| p5_4b_s010__vs__cons_mean_a1500 | 0.7637 | 5.7949 | 0.5593 |
| p5_4b_s010__vs__ties_k50_a1500 | 0.7637 | 5.7977 | 0.5593 |
| cons_mean_a1500__vs__ties_k50_a1500 | 1.0000 | 0.0000 | 1.0000 |

## Representation-change alignment (the decisive table)

For each model X: does `h_X - h_qwen06` point where `h_supa - h_qwen06` points?

## p4_4b_s010

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.0000 | +0.0000 | +0.0066 | 0.0000e+00 | 5.4149e-09 |
| layer_0 | -0.2028 | -0.2043 | -0.0074 | 2.1846e-02 | 7.2059e-02 |
| layer_7 | -0.4507 | -0.4425 | +0.0054 | 3.5507e-02 | 1.9357e-02 |
| layer_14 | -0.4353 | -0.4256 | -0.0010 | 3.4176e-02 | 2.7178e-02 |
| layer_21 | -0.0859 | -0.0699 | -0.0030 | 3.3991e-02 | 6.6221e-02 |
| layer_27 | -0.0307 | -0.0388 | -0.0005 | 7.8759e-02 | 2.9548e-01 |

## p4_8b_s010

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.0000 | +0.0000 | +0.0049 | 0.0000e+00 | 5.4149e-09 |
| layer_0 | -0.2050 | -0.2035 | +0.0035 | 2.5446e-02 | 7.2059e-02 |
| layer_7 | -0.5146 | -0.5067 | +0.0041 | 4.0780e-02 | 1.9357e-02 |
| layer_14 | -0.4913 | -0.4815 | -0.0061 | 3.8388e-02 | 2.7178e-02 |
| layer_21 | -0.1205 | -0.1015 | +0.0030 | 3.3973e-02 | 6.6221e-02 |
| layer_27 | -0.0620 | -0.0478 | +0.0037 | 5.5081e-02 | 2.9548e-01 |

## p5_4b_s010

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | -0.0152 | -0.0150 | +0.0031 | 1.2035e-01 | 5.4149e-09 |
| layer_0 | +0.2802 | +0.2838 | +0.0058 | 2.1442e-01 | 7.2059e-02 |
| layer_7 | -0.5291 | -0.5316 | +0.0033 | 2.6161e-01 | 1.9357e-02 |
| layer_14 | -0.4937 | -0.4969 | -0.0039 | 2.5122e-01 | 2.7178e-02 |
| layer_21 | -0.0718 | -0.0742 | -0.0000 | 2.3276e-01 | 6.6221e-02 |
| layer_27 | +0.0734 | +0.0187 | +0.0014 | 1.8639e-01 | 2.9548e-01 |

## cons_mean_a1500

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.9322 | +0.2129 | -0.0054 | 5.4149e-09 | 5.4149e-09 |
| layer_0 | +0.9988 | +0.8710 | +0.0006 | 7.1360e-02 | 7.2059e-02 |
| layer_7 | +0.9139 | +0.7196 | +0.0012 | 1.7747e-02 | 1.9357e-02 |
| layer_14 | +0.9675 | +0.7057 | -0.0039 | 2.5399e-02 | 2.7178e-02 |
| layer_21 | +0.9972 | +0.6211 | +0.0056 | 6.5724e-02 | 6.6221e-02 |
| layer_27 | +0.9995 | +0.8081 | +0.0045 | 2.9506e-01 | 2.9548e-01 |

## ties_k50_a1500

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.9322 | +0.2129 | -0.0003 | 5.4149e-09 | 5.4149e-09 |
| layer_0 | +0.9988 | +0.8710 | +0.0021 | 7.1349e-02 | 7.2059e-02 |
| layer_7 | +0.9129 | +0.7187 | +0.0002 | 1.7759e-02 | 1.9357e-02 |
| layer_14 | +0.9667 | +0.7047 | +0.0043 | 2.5389e-02 | 2.7178e-02 |
| layer_21 | +0.9971 | +0.6209 | -0.0039 | 6.5744e-02 | 6.6221e-02 |
| layer_27 | +0.9995 | +0.8080 | -0.0085 | 2.9525e-01 | 2.9548e-01 |
