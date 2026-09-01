# Resource report

Measured on NVIDIA GeForce RTX 4070 Laptop, 8188 MiB. One model load at a time.
Supa-A1.1 loaded float16 from the sibling vault. FWA adapters trained in float32.

## Parameters (layer 14, rank 16)

| Arm | Trainable | Recurrent state (B=1) |
|---|---:|---:|
| A | 0 | 0 |
| B | 524,593 | 0 |
| C | 524,593 | 8,192 bytes |
| D | 525,618 | 8,192 bytes |

KV-cache bytes **if used** at T=1024: 28 × 2 × 8 × 1024 × 128 × 2 ≈ 14.7 MB.
This experiment kept `use_cache=False` for the recurrent path, so FWA did **not**
replace the KV cache. Bounded state is 8 KB beside a still-present attention stack.

## Peak VRAM (forward profile)

| Arm | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|
| A | 1.28 GB | 1.36 GB | 1.51 GB |
| B seed 11 | 1.28 GB | 1.36 GB | 1.52 GB |
| C seed 11 | 1.29 GB | 1.37 GB | 1.52 GB |
| D seed 11 | 1.29 GB | 1.37 GB | 1.52 GB |

Training peak (B/C/D, T=512, backward): **~3.02 GB**.
P1 training peak: **2.34 GB**.

No VRAM saving versus Arm A at these lengths. FWA did not shrink memory growth
with length; KV attention is still computed in full.

## Throughput (forward tokens/sec, measured)

Arm A T=256: ~3627 tok/s (no FWA scan).
Arms B/C/D T=256–1024: ~800–1350 tok/s (Python FALCON-2 scan + `use_cache=False`).

Inference is slower with FWA. Do not claim an efficiency improvement.

## Training cost

| Item | Value |
|---|---|
| Completed optimizer steps | 24 × 7 runs (P1 C + P2 B/C/D × 2 seeds) = 168 |
| Training GPU-hours | ≈ 0.64 |
| Seed 33 | not run (OOM while starting the seventh sequential training load) |
| Adapter size | ~1.05 MB (`fwa_adapter.pt`) |
