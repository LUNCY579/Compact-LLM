# Pilot report (P1)

## Setup

- Arm C only, seed 11, layer `[14]`, rank 16.
- Frozen Supa-A1.1, FWA tensors in **float32**, backbone float16.
- 24 optimizer steps, microbatch 1, accumulation 8, max length 512.
- Training mix: lab unsealed synthetic colour-retrieval (`data/synthetic/`, sha256
  `0d830fe7f55cde3ab848351c33ceb6aae72e53cbdbfda289c87a0a81f5bdc211`).
- Official CC0 2k subset was also built (`data/unsealed/`, train sha256
  `14bcddb3bbb275539e03b45e9b45a4024cb9aa352f82af7376bc61bce39d8c2c`) but **not
  used for P1**: untruncated 2k rows plus fp16 FWA adapters produced non-finite
  loss. Training uses short synthetic rows that fit the 512-token budget so the
  needle is not truncated away.

## Outcome

| Field | Value |
|---|---|
| Optimizer steps | 24 / 24 |
| Aborted | no |
| Examples seen | 192 |
| Tokens seen | 72,699 |
| Wall | 336 s |
| GPU-hours | 0.093 |
| Peak VRAM | 2,512,948,224 bytes (~2.34 GB) |
| Trainable params | 524,593 |
| First logged loss | 0.00548 |
| Last logged loss | 0.0483 |

Loss stayed finite. It did **not** show a clean downward trend. Frozen Supa
already assigns very low NLL to these short colour-retrieval targets, so P1
only establishes that the FWA path can be optimized without crashing.

## Layer placement

P1 used middle layer 14 only. No widen: no loss signal to justify extra layers.
P2 keeps `[14]`.
