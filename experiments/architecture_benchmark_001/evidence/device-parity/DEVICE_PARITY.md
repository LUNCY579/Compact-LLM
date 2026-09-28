# Can this programme use the GPU? Measured, by workload.

Short answer: **inference yes, training no, and the cuDNN default would have silently corrupted
the numbers.**

Exploratory. Outside the hashed root — no receipt, no gate. Open splits only, no efficacy claim.

## Memory — training does not fit, and every workaround changes the experiment

596,049,920 parameters. RTX 4070 Laptop, 8,188 MiB total, **6,923 MiB (6.76 GiB) free**.

| workload | VRAM | fits? |
|---|---:|---|
| full-model FP32 AdamW (w + grad + `exp_avg` + `exp_avg_sq`) | 8.88 GiB | **no** |
| bf16 weights+grads, fp32 Adam | 6.66 GiB | no, once activations count |
| pure bf16 | 4.44 GiB | yes — but changes numerics *and* optimizer-state precision |
| freeze tied embeddings (26% of params) | 7.14 GiB | no — *and* trips the harness's `no frozen parameters` assert |
| **inference** | **2.31 GiB measured** | **yes** |
| **latent-only training, donor frozen** (r=146→1024) | **2.41–3.53 GiB** | **yes** |

The open question is *budget*, so confounding it with a precision change would destroy the
comparison — and the C28 500k arm is already complete on CPU FP32, so its control must match the
device. But **the most promising next experiment fits**: fine-tuning the low-rank latents (12.6M
to 88.1M trainable params with the donor frozen) is how one would try to recover the collapsed
0.143× and 0.071× compression points.

## The harness blocked the obvious shortcut, correctly

All three attempts under the pre-existing CUDA environment died identically:

    run.py:102  GateClosed: install the pinned native Transformers version in requirements-native.txt

`Chat-Bubble/.venv` carries transformers **5.14.1** against the pinned **4.51.3** (also torch
2.12.1, safetensors 0.8.0, numpy 2.4.6). The pinning discipline caught a cross-runtime
comparison before it could produce a number. That environment is unusable here.

`requirements.txt` pins `torch==2.10.0` and states the pin "does not prescribe a CUDA build", so
`torch==2.10.0+cu126` is *permitted*, not a workaround. Built at
`C:\Users\clclc\BusinessProject\.venv-rab001-cu126`.

## Parity — 4 dev rows, 458 target tokens, identical inputs

| configuration | torch | TF32 | dev CE | tok/s | peak VRAM |
|---|---|---|---:|---:|---:|
| `cpu_ref` | 2.10.0+cpu | — | 5.490954 | 12.0 | — |
| `cpu_cu126venv` | 2.10.0+cu126 | — | 5.490954 | 57.2 | — |
| `cuda_fp32` | 2.10.0+cu126 | off | 5.490954 | 76.7 | 2.31 GiB |
| `cuda_tf32` | 2.10.0+cu126 | **on** | 5.491146 | 55.0 | 2.31 GiB |

Elementwise over every logit — the same standard the HOTFIX-001/002 parity work uses:

| comparison | max \|Δlogit\| | mean \|Δ\| | top-1 | within 1e-4 |
|---|---:|---:|---:|---|
| `cpu_ref` vs `cpu_cu126venv` | **0.000e+00** | 0.000e+00 | 1.0000 | **bitwise identical** |
| `cpu_ref` vs `cuda_fp32` | **9.966e-05** | 4.477e-06 | 1.0000 | yes, *barely* |
| `cpu_ref` vs `cuda_tf32` | **5.416e-02** | 2.497e-03 | 0.9978 | **no — 540× outside** |

### Three findings

**1. The CUDA wheel is bitwise identical to the CPU wheel, on CPU.** Zero delta. Swapping
`torch==2.10.0+cpu` for `torch==2.10.0+cu126` changes nothing for CPU work, so the new
environment is a safe drop-in and does not fork the runtime.

**2. CUDA FP32 sits at 9.966e-05 against a 1e-4 tolerance — 99.7% of the budget, on 4 rows.**
Top-1 agreement is perfect and dev CE matches to six decimals. But a max over 458 tokens already
consumes almost the whole tolerance, so **it will cross on a larger sample**. The correct reading
is *"GPU FP32 is very close to CPU FP32 but not interchangeable with it"*, not "GPU passes
parity". GPU numbers should be compared against other GPU numbers, and the HOTFIX-001/002 parity
work — which is *about* FP32 elementwise behaviour at this tolerance — can never move.

**3. TF32 is on by default in cuDNN on this card, and it is catastrophic here.** 5.416e-02 max
error, 540× outside tolerance, and it flips the top-1 token on 0.22% of positions. Any script
touching CUDA in this programme sets both `torch.backends.cuda.matmul.allow_tf32` and
`torch.backends.cudnn.allow_tf32` to `False` explicitly, so the setting is recorded rather than
inherited.

## Speed — the GPU is not the win here

**76.7 tok/s on GPU against 57.2 tok/s on CPU under comparable conditions: 1.34×.** Not an order
of magnitude. At 596M parameters with ~115-token rows, kernel-launch overhead and the reference
implementation's O(T²) materialised attention masks dominate, so the device barely matters.

(The `cpu_ref` 12.0 tok/s figure is not a runtime difference — it was measured while a training
job was saturating the CPU. That gap *is* the point below.)

The real value is **parallelism**: GPU work does not contend with CPU training, so inference
sweeps can run concurrently with a multi-hour training run instead of queueing behind it. That
roughly doubles throughput on this host without making any single job faster.

## Reproduction

```bash
CPU=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
GPU=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
$CPU experiments/rab001_exploration/device_parity.py cpu_ref       cpu  4
$GPU experiments/rab001_exploration/device_parity.py cuda_fp32     cuda 4
$GPU experiments/rab001_exploration/device_parity.py cuda_tf32     cuda 4 --tf32
$CPU experiments/rab001_exploration/device_parity_compare.py cpu_ref cuda_fp32 cuda_tf32
```

Artifact: `evidence/device-parity/DEVICE_PARITY.json`.
