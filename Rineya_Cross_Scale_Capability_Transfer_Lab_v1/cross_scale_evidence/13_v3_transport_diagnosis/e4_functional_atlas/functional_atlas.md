# v3 E4 — Cross-Model Functional Alignment Atlas

59 frozen calibration prompts, prompts only, final prompt token, CPU float32.
Depth slots: embedding, layer_0, layer_7, layer_14, layer_21, layer_27.

Donor-side rows (Qwen3-4B, Qwen3-8B) are absent: those checkpoints are no longer
resident on this host. See `protocol/atlas_models.json`.

## Representation similarity vs the reference base (mean cosine of final-token hidden state)

| pair | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| qwen06__vs__supa | 1.0000 | 0.9963 | 0.9292 | 0.8896 | 0.7547 | 0.8109 |
| qwen06__vs__p4_4b_s010 | 1.0000 | 0.9998 | 0.9955 | 0.9958 | 0.9945 | 0.9935 |
| qwen06__vs__p4_8b_s010 | 1.0000 | 0.9997 | 0.9978 | 0.9979 | 0.9967 | 0.9959 |
| qwen06__vs__p5_4b_s010 | 0.9989 | 0.9967 | 0.9820 | 0.9771 | 0.9699 | 0.9615 |
| qwen06__vs__cons_mean_a1500 | 1.0000 | 0.9964 | 0.9293 | 0.8895 | 0.7549 | 0.8109 |
| qwen06__vs__ties_k50_a1500 | 1.0000 | 0.9964 | 0.9293 | 0.8896 | 0.7549 | 0.8109 |

## Linear CKA vs the reference base

| pair | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| qwen06__vs__supa | 0.0000 | 0.9996 | 0.9521 | 0.8602 | 0.8053 | 0.8759 |
| qwen06__vs__p4_4b_s010 | 0.0000 | 0.9999 | 0.9911 | 0.9892 | 0.9923 | 0.9935 |
| qwen06__vs__p4_8b_s010 | 0.0000 | 0.9999 | 0.9955 | 0.9947 | 0.9961 | 0.9959 |
| qwen06__vs__p5_4b_s010 | 0.0000 | 0.9987 | 0.9603 | 0.9432 | 0.9729 | 0.9776 |
| qwen06__vs__cons_mean_a1500 | 0.0000 | 0.9996 | 0.9518 | 0.8582 | 0.8062 | 0.8760 |
| qwen06__vs__ties_k50_a1500 | 0.0000 | 0.9996 | 0.9517 | 0.8581 | 0.8063 | 0.8760 |

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
| embedding | +0.0000 | +0.0000 | +0.0000 | 0.0000e+00 | 0.0000e+00 |
| layer_0 | -0.1611 | -0.1587 | +0.0003 | 2.4337e-02 | 8.5473e-02 |
| layer_7 | -0.0941 | -0.0902 | +0.0030 | 9.7057e-02 | 3.9438e-01 |
| layer_14 | +0.1142 | +0.0686 | +0.0005 | 9.2055e-02 | 4.7768e-01 |
| layer_21 | +0.1361 | +0.0423 | +0.0074 | 9.8477e-02 | 7.2554e-01 |
| layer_27 | +0.1945 | +0.0344 | +0.0034 | 1.0934e-01 | 5.7086e-01 |

## p4_8b_s010

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.0000 | +0.0000 | +0.0000 | 0.0000e+00 | 0.0000e+00 |
| layer_0 | -0.1541 | -0.1522 | +0.0010 | 2.6800e-02 | 8.5473e-02 |
| layer_7 | -0.0663 | -0.0578 | -0.0001 | 6.6283e-02 | 3.9438e-01 |
| layer_14 | +0.2353 | +0.1814 | -0.0074 | 6.4167e-02 | 4.7768e-01 |
| layer_21 | +0.2409 | +0.0991 | -0.0001 | 7.4175e-02 | 7.2554e-01 |
| layer_27 | +0.2160 | +0.0598 | +0.0066 | 8.5162e-02 | 5.7086e-01 |

## p5_4b_s010

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.0000 | +0.0000 | +0.0000 | 1.0916e-01 | 0.0000e+00 |
| layer_0 | +0.1410 | +0.1437 | +0.0044 | 2.1103e-01 | 8.5473e-02 |
| layer_7 | +0.0690 | +0.0398 | -0.0080 | 2.7414e-01 | 3.9438e-01 |
| layer_14 | +0.3701 | +0.2269 | -0.0022 | 2.7703e-01 | 4.7768e-01 |
| layer_21 | +0.2960 | +0.1472 | -0.0005 | 2.9787e-01 | 7.2554e-01 |
| layer_27 | +0.2801 | +0.0242 | +0.0019 | 2.7408e-01 | 5.7086e-01 |

## cons_mean_a1500

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.0000 | +0.0000 | +0.0000 | 0.0000e+00 | 0.0000e+00 |
| layer_0 | +0.9990 | +0.9809 | +0.0061 | 8.4976e-02 | 8.5473e-02 |
| layer_7 | +0.9996 | +0.8950 | +0.0006 | 3.9365e-01 | 3.9438e-01 |
| layer_14 | +0.9999 | +0.6870 | -0.0041 | 4.7756e-01 | 4.7768e-01 |
| layer_21 | +0.9999 | +0.2663 | +0.0024 | 7.2441e-01 | 7.2554e-01 |
| layer_27 | +0.9999 | +0.3224 | +0.0013 | 5.7107e-01 | 5.7086e-01 |

## ties_k50_a1500

| depth | cos vs Supa change | prompt-shuffled control | random control | ||change|| / ||h|| | Supa ||change|| / ||h|| |
|---|---|---|---|---|---|
| embedding | +0.0000 | +0.0000 | +0.0000 | 0.0000e+00 | 0.0000e+00 |
| layer_0 | +0.9990 | +0.9808 | -0.0027 | 8.5009e-02 | 8.5473e-02 |
| layer_7 | +0.9996 | +0.8950 | +0.0018 | 3.9367e-01 | 3.9438e-01 |
| layer_14 | +0.9998 | +0.6869 | -0.0019 | 4.7748e-01 | 4.7768e-01 |
| layer_21 | +0.9999 | +0.2662 | -0.0031 | 7.2435e-01 | 7.2554e-01 |
| layer_27 | +0.9999 | +0.3223 | +0.0029 | 5.7106e-01 | 5.7086e-01 |
