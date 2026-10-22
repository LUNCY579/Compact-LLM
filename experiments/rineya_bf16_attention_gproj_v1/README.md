# BAG-001 — Rineya BF16 attention-gradient audit and GProj research

**Branch:** `research/rineya-bf16-attention-gproj-v1`  
**Parent:** `post-training-Mimo` at `03db218ceebc61a2bc04f2482cf7e094045e01c1`  
**Status:** CPU numerical/mechanical tests passed. Exact Rineya checkpoint, CUDA,
Transformers binding and native efficacy remain **UNQUALIFIED / NOT RUN**.

This is an isolated correctness experiment for the exact 0.6B-class Rineya
checkpoint. It does not change existing loaders, optimizers, production defaults,
research evidence, model weights or other projects. It does not solve forgetting.
Read [PROTOCOL.md](PROTOCOL.md), [SOURCE_NOTES.md](SOURCE_NOTES.md) and
[VERIFICATION.md](VERIFICATION.md) before interpreting any result.

## Implemented paths

`core.py` implements FP64 attention/VJP references, BF16 cast-error models, actual
probability-mass normalization, separate query/key corrections, GQA reductions,
masking and zero-reference-aware metrics. `backends.py` explicitly selects PyTorch
SDPA implementations and provides an opt-in dense custom backward. `capture.py`
records represented Q/K/V, saved O and incoming dO without cutting context, and
restores interception state on exit. Capture tensors stay local.

`native.py` binds only a locally supplied safetensors Qwen3-derived Rineya in the
0.6B parameter envelope. It rejects downloads, remote code, quantization, unknown
trainable names, missing SDPA calls and unexecuted capture sites. The loader uses
FP32 master tensors for the explicitly selected trainable parameters and BF16
projection operands under autocast. Native qualification changes no parameters.

`runner.py` implements a four-arm, matched-seed micro-pilot after qualification:

| Arm | Intervention | Interpretation |
|---|---|---|
| `native` | BF16 operands, explicitly forced configured SDPA backend | Measured control; not an assertion about historical auto dispatch |
| `fp32` | FP32 math attention, original represented output boundary | Higher-precision reference control |
| `cast_only` | Dense repaired forward, BF16-cast backward operands | Negative control sharing GProj's forward |
| `gproj` | Same dense forward, corrected dQ and dK contractions | Research implementation, **not a fused kernel** |

All arms start from identical checkpoint bytes, train the same named parameters
with the same AdamW recipe and row schedule, and evaluate with fixed FP32 math
attention. Reports include pre-clipping gradient norms, actual update norms,
per-row development losses, frozen-weight checks, timings and memory counters.
**Acquisition/retention loss cohorts are proxies, not behavioral success rates.**
No result automatically promotes a model or changes a production default.

## CPU verification (no model or internet required)

Use an environment with PyTorch installed. Validation here used PyTorch
`2.10.0+cpu`, Python `3.13.5` and pytest. No repository-wide dependency files are
changed. Run from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  -c experiments/rineya_bf16_attention_gproj_v1/pytest.ini \
  --confcutdir=experiments/rineya_bf16_attention_gproj_v1 \
  experiments/rineya_bf16_attention_gproj_v1/tests -q

python -m experiments.rineya_bf16_attention_gproj_v1 synthetic \
  --output /tmp/bag001-synthetic-new.json
```

Output paths must be new. The synthetic report is explicitly incapable of
qualifying native training. The four-key witness removes the erroneous `-24`
coordinate, not every BF16 rounding error in the gradient.

## Native sequence — opt in, never auto-launch

The optional native dependency is Transformers with Qwen3 and SDPA support
(reference API: Transformers 4.57.1). No Transformers version was executed in this
build environment. Use a separate research environment and qualify its exact
version; do not upgrade an existing successful training environment in place.

1. Hash the exact locally materialized checkpoint; symlinks must be materialized.
2. Copy `configs/native.example.json` outside the source checkpoint and fill real
   checkpoint/data hashes, local paths, exact parameter names and SDPA call sites.
3. Supply prospectively admitted, pre-tokenized TRAIN and DEV rows; run qualification.
4. Only a matching passing qualification plus `--execute-training` can start the pilot.

```bash
python -m experiments.rineya_bf16_attention_gproj_v1 manifest \
  --checkpoint /absolute/path/to/exact-rineya-0.6b \
  --output /tmp/bag001-checkpoint-manifest-new.json

# SHA256 of the JSONL bytes (copy into the frozen config):
python -c "import hashlib; print(hashlib.sha256(open('/absolute/path/to/admitted.jsonl','rb').read()).hexdigest())"

python -m experiments.rineya_bf16_attention_gproj_v1 qualify \
  --config /absolute/path/to/frozen-native.json \
  --output /absolute/path/to/new-qualification-directory

python -m experiments.rineya_bf16_attention_gproj_v1 train \
  --config /absolute/path/to/frozen-native.json \
  --qualification /absolute/path/to/new-qualification-directory/qualification.json \
  --execute-training --output /absolute/path/to/new-pilot-directory
```

Qualification defaults in the example are proposed research thresholds, not paper
results: relative L2 <= 0.05 **or** absolute L2 <= 1e-6 for reference comparisons,
plus full-model selected-parameter comparisons. Freeze or prospectively amend
them before reading native results. Failures are retained; do not relax a gate
post hoc and label the old run qualified.

### Tokenized input row schema

```json
{"id":"fresh-train-001","group_id":"family-a","split":"train","purpose":"acquisition","source":"reviewed provenance identifier","admitted_for":"train","input_ids":[1,2,3,4],"labels":[-100,-100,3,4]}
```

This is a schema illustration, not training data or valid model tokenization.
Supply both DEV purposes (`acquisition`, `retention`) and at least one TRAIN row.
The loader rejects cross-split group/exact-token overlap, duplicate IDs, missing
admission and absent supervision. It cannot certify that provenance declarations
are truthful or detect semantic near-duplicates: upstream dataset review remains
required. Existing sealed, spent, historical confirmation and selection-barred
banks are not made admissible by relabeling them.

### Replay a capture

```bash
python -m experiments.rineya_bf16_attention_gproj_v1 audit \
  --capture /absolute/path/to/capture_000.pt \
  --backend math --device cpu --output /tmp/bag001-replay-new.json
```

Replaying on CPU is not GPU qualification. Changing device, software, config,
checkpoint, data or source code invalidates the native training qualification.

## Deliberate limits

Dense attention is quadratic and is not a replacement for FlashAttention at long
context. The pilot caps sequences at 256, score elements at 2,000,000, trainable
parameters at 20 million, rows at 128, and steps at 32 per arm. These bounds are
not a memory-fit certificate. Dropout, KV-cache decoding, concurrent sessions,
compiled/captured graphs, higher derivatives and external FA2/FA3 entry points are
unsupported. A loader that bypasses the intercepted SDPA function fails closed.

Native SDPA internals do not expose their saved LSE or accumulator tensors here.
The dense GProj reference uses recomputed probabilities and FP32 contractions;
it does not reproduce fused accumulation order or every Appendix-B rounding
operation. No optimized GProj kernel, 4.7% overhead replication, long-context
certificate, skill gain or forgetting reduction is claimed.
