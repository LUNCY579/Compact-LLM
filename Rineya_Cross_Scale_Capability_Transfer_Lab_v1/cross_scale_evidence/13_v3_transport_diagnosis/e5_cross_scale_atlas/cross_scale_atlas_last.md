# v3 E5 — Cross-Scale Functional Alignment Atlas (last pooling)

59 frozen calibration prompts, prompts only. Donors run through the
validated streaming forward (Gate 0 PASS, worst relative error
2.81e-06). Widths:
  qwen4b 2560, qwen8b 4096, qwen06 1024, supa 1024.

Cross-width cosine is undefined, so the atlas uses linear CKA and RSA, both
basis-free, plus logit agreement over the shared vocabulary.

## qwen4b__vs__qwen06 — depth x depth linear CKA

| donor depth \ target depth | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| embedding | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layer_0 | 0.0000 | 0.9614 | 0.4588 | 0.3315 | 0.4202 | 0.4733 |
| layer_9 | 0.0000 | 0.4434 | 0.8690 | 0.8428 | 0.8366 | 0.7903 |
| layer_18 | 0.0000 | 0.5096 | 0.7735 | 0.8450 | 0.8233 | 0.8066 |
| layer_26 | 0.0000 | 0.3751 | 0.7393 | 0.7621 | 0.9114 | 0.9063 |
| layer_35 | 0.0000 | 0.3764 | 0.6860 | 0.6790 | 0.8700 | 0.9034 |

Matched relative depth, against the prompt-shuffled control:

| depth | CKA | shuffled | lift | RSA | RSA shuffled |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | +0.0000 | +1.0000 | +1.0000 |
| 1 | 0.9614 | 0.0274 | +0.9340 | +0.9194 | +0.0015 |
| 2 | 0.8690 | 0.0717 | +0.7973 | +0.8163 | -0.0196 |
| 3 | 0.8450 | 0.0701 | +0.7748 | +0.7836 | -0.0091 |
| 4 | 0.9114 | 0.1773 | +0.7341 | +0.7255 | -0.0084 |
| 5 | 0.9034 | 0.2153 | +0.6880 | +0.5896 | -0.0014 |

## qwen8b__vs__qwen06 — depth x depth linear CKA

| donor depth \ target depth | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| embedding | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layer_0 | 0.0000 | 0.9007 | 0.4903 | 0.3849 | 0.4901 | 0.5369 |
| layer_9 | 0.0000 | 0.3874 | 0.8488 | 0.8409 | 0.7731 | 0.7548 |
| layer_18 | 0.0000 | 0.5784 | 0.7568 | 0.8159 | 0.8258 | 0.8065 |
| layer_26 | 0.0000 | 0.3834 | 0.7198 | 0.7473 | 0.9255 | 0.9058 |
| layer_35 | 0.0000 | 0.3131 | 0.6442 | 0.6244 | 0.8074 | 0.8380 |

Matched relative depth, against the prompt-shuffled control:

| depth | CKA | shuffled | lift | RSA | RSA shuffled |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | +0.0000 | +1.0000 | +1.0000 |
| 1 | 0.9007 | 0.0439 | +0.8567 | +0.8590 | -0.0072 |
| 2 | 0.8488 | 0.0704 | +0.7784 | +0.8736 | -0.0071 |
| 3 | 0.8159 | 0.0734 | +0.7424 | +0.7684 | +0.0047 |
| 4 | 0.9255 | 0.1797 | +0.7458 | +0.6906 | +0.0029 |
| 5 | 0.8380 | 0.1836 | +0.6544 | +0.4276 | -0.0107 |

## qwen4b__vs__supa — depth x depth linear CKA

| donor depth \ target depth | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| embedding | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layer_0 | 0.0000 | 0.9592 | 0.4224 | 0.4378 | 0.3707 | 0.3861 |
| layer_9 | 0.0000 | 0.4444 | 0.8969 | 0.8478 | 0.7863 | 0.6805 |
| layer_18 | 0.0000 | 0.5113 | 0.8111 | 0.8184 | 0.7610 | 0.6740 |
| layer_26 | 0.0000 | 0.3751 | 0.7777 | 0.8608 | 0.8580 | 0.8407 |
| layer_35 | 0.0000 | 0.3752 | 0.7178 | 0.7960 | 0.8314 | 0.8864 |

Matched relative depth, against the prompt-shuffled control:

| depth | CKA | shuffled | lift | RSA | RSA shuffled |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | +0.0000 | +1.0000 | +1.0000 |
| 1 | 0.9592 | 0.0271 | +0.9321 | +0.9188 | +0.0016 |
| 2 | 0.8969 | 0.0805 | +0.8163 | +0.8082 | -0.0152 |
| 3 | 0.8184 | 0.0901 | +0.7283 | +0.6866 | -0.0139 |
| 4 | 0.8580 | 0.1873 | +0.6707 | +0.8208 | -0.0294 |
| 5 | 0.8864 | 0.2124 | +0.6740 | +0.7200 | -0.0122 |

## qwen8b__vs__supa — depth x depth linear CKA

| donor depth \ target depth | embedding | layer_0 | layer_7 | layer_14 | layer_21 | layer_27 |
|---|---|---|---|---|---|---|
| embedding | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layer_0 | 0.0000 | 0.9025 | 0.4536 | 0.4995 | 0.4090 | 0.4492 |
| layer_9 | 0.0000 | 0.3893 | 0.8997 | 0.7798 | 0.7729 | 0.6368 |
| layer_18 | 0.0000 | 0.5824 | 0.7940 | 0.8542 | 0.7032 | 0.7084 |
| layer_26 | 0.0000 | 0.3835 | 0.7554 | 0.8747 | 0.8606 | 0.8857 |
| layer_35 | 0.0000 | 0.3136 | 0.6701 | 0.7676 | 0.7316 | 0.8495 |

Matched relative depth, against the prompt-shuffled control:

| depth | CKA | shuffled | lift | RSA | RSA shuffled |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | +0.0000 | +1.0000 | +1.0000 |
| 1 | 0.9025 | 0.0439 | +0.8586 | +0.8635 | -0.0075 |
| 2 | 0.8997 | 0.0803 | +0.8194 | +0.8793 | +0.0025 |
| 3 | 0.8542 | 0.0941 | +0.7601 | +0.7275 | -0.0098 |
| 4 | 0.8606 | 0.1882 | +0.6724 | +0.8373 | -0.0150 |
| 5 | 0.8495 | 0.1750 | +0.6745 | +0.6408 | -0.0333 |

## qwen4b__vs__qwen8b — depth x depth linear CKA

| donor depth \ target depth | embedding | layer_0 | layer_9 | layer_18 | layer_26 | layer_35 |
|---|---|---|---|---|---|---|
| embedding | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layer_0 | 0.0000 | 0.9241 | 0.4477 | 0.6275 | 0.4741 | 0.3862 |
| layer_9 | 0.0000 | 0.5593 | 0.9489 | 0.8992 | 0.8268 | 0.6989 |
| layer_18 | 0.0000 | 0.5965 | 0.9091 | 0.9258 | 0.8204 | 0.6943 |
| layer_26 | 0.0000 | 0.5278 | 0.7845 | 0.8275 | 0.9526 | 0.8330 |
| layer_35 | 0.0000 | 0.5331 | 0.7176 | 0.7803 | 0.9232 | 0.9142 |

Matched relative depth, against the prompt-shuffled control:

| depth | CKA | shuffled | lift | RSA | RSA shuffled |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | +0.0000 | +1.0000 | +1.0000 |
| 1 | 0.9241 | 0.0491 | +0.8750 | +0.8984 | +0.0032 |
| 2 | 0.9489 | 0.0926 | +0.8563 | +0.9142 | -0.0109 |
| 3 | 0.9258 | 0.0986 | +0.8272 | +0.8469 | -0.0117 |
| 4 | 0.9526 | 0.2073 | +0.7454 | +0.8385 | -0.0108 |
| 5 | 0.9142 | 0.2044 | +0.7098 | +0.7077 | +0.0109 |

## Logit space (shared vocabulary)

| pair | logit cosine | symmetric KL | top-1 agreement |
|---|---:|---:|---:|
| qwen4b__vs__qwen8b | 0.5521 | 8.8887 | 0.6271 |
| qwen4b__vs__qwen06 | 0.3913 | 9.2227 | 0.4746 |
| qwen4b__vs__supa | 0.3403 | 10.1845 | 0.5763 |
| qwen8b__vs__qwen06 | 0.4348 | 12.1550 | 0.3898 |
| qwen8b__vs__supa | 0.3998 | 4.7509 | 0.7288 |
| qwen06__vs__supa | 0.7569 | 7.9879 | 0.3898 |

## Frozen decision

- Best donor->target matched-depth CKA lift over control: **+0.9340**
- Best donor->donor matched-depth CKA lift over control: **+0.8750**
- Threshold from amendment v3.1: +0.10

**donor-target structure present; E6 authorised**
