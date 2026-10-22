# RHF-001: Hebbian fact memory for Rineya

**Implemented research package; pretrained Rineya efficacy is NOT established.**
Dedicated branch: `research/rineya-hebbian-fact-memory-v1`, based on
`post-training-Mimo` at `03db218ceebc61a2bc04f2482cf7e094045e01c1`.
This folder is additive. Existing model weights, training pipelines, frozen
benchmarks, game projects, and other branches are not changed.

The experiment asks whether structured fact storage can separate acquisition
from interference in a compact model. A constructed memory can retrieve a
supplied vector association without an optimizer. That does **not** establish
that the association is available in a pretrained model's residual stream,
that a replaced MLP preserves its other functions, or that reasoning improves.

## What runs

| Component | Implemented behavior |
| --- | --- |
| `memory.py` | Bilinear identity-gated MLP; raw, diagonal, full-whitened, and two-solve data-dependent construction; matrix-free checked CG; immutable full-bank readout editing |
| `metrics.py` | Strict margins, geometry/crosstalk decomposition, controlled query noise, paired repairs/damages, fact-level bootstrap utility |
| `bench.py` | Capacity/anisotropy grid, precision-rounding diagnostics, sequential acquisition, edit locality, parameter-accounted controls, fact-adaptive attention experiment |
| `native.py` | Explicitly named native MLP activation capture, reversible blend, exact zero-alpha bypass, parameter/buffer integrity, held-out next-token drift and one-step cache parity |
| `artifacts.py` | Create-only JSON and safetensors, bounded reads, inventory/hash/shape checks; no pickle |
| `cli.py` | Smoke, sweep, SSFR, fit, edit, opt-in local-only native screen |
| `tests/` | Numerical oracles, boundary cases, mocked integration, failure and persistence tests |

`PAPER_NOTES.md` records equation choices and differences from the released code.
`PROTOCOL.md` separates development evidence, confirmation, and native promotion.
`evidence/VALIDATION.md` reports actual execution, including numerical failures.

## Quick start

Use a separately prepared environment; do not upgrade an existing pinned Rineya
environment. `requirements-tested.txt` records the exact CPU validation versions.
No install, download, remote code execution, CI dispatch, or model loading happens
on import or in a synthetic command.

Run from the **repository root**:

```sh
python -m pytest experiments/rineya_hebbian_fact_memory_v1/tests -q
python -m experiments.rineya_hebbian_fact_memory_v1.cli smoke --out rhf_smoke_001
python -m experiments.rineya_hebbian_fact_memory_v1.cli sweep --config experiments/rineya_hebbian_fact_memory_v1/configs/development.json --out rhf_dev_001
python -m experiments.rineya_hebbian_fact_memory_v1.cli ssfr --seed 17 --steps 400 --out rhf_ssfr_001
```

Every output directory must be new. Existing and partially completed runs are
never overwritten. Numerical nonconvergence is recorded as `BLOCKED_NUMERICS`,
not silently accepted. CLI input/resource errors return 2; a completed native
screen that misses its gates returns 3. A synthetic run finishing is not a
claim that every construction or nested sequential subexperiment succeeded.

The development grid includes clean and correlated inputs, narrow and wider
memories, and seeds 17/29. Confirmation seeds 43/71 are reserved in a separate
config and were **not executed** during delivery. These configs are reproducible
plans, not external preregistration or secret/sealed datasets.

## Construct and edit a vector memory

Prepare a `.safetensors` file containing exactly `keys` (N x D) and `targets`
(N x output_D), both finite floating-point tensors. Targets are desired vectors,
not token IDs, reward labels, or a substitute for specifying native edit targets.
`configs/native_reconstruction.json` is also a usable full-whitening build config.

```sh
python -m experiments.rineya_hebbian_fact_memory_v1.cli fit --config experiments/rineya_hebbian_fact_memory_v1/configs/native_reconstruction.json --data facts.safetensors --out rhf_memory_001
python -m experiments.rineya_hebbian_fact_memory_v1.cli edit --config experiments/rineya_hebbian_fact_memory_v1/configs/native_reconstruction.json --memory rhf_memory_001 --data all_retained_and_new_facts.safetensors --out rhf_memory_002
```

An edit keeps the original memory object/artifact unchanged, preserves A/G, and
rebuilds B using the **entire retained and updated bank**. Supply the same
readout variant/convention as the original when testing same-algorithm edits.
The edit is not a constant-time local update or a no-rehearsal claim. Missing old
facts are not magically retained. Data-dependent A/G are not refit during edits.

Inference stores `2*M*D + M*output_D` scalars, with no key table. The full bank is
needed for construction/edits and counts toward working memory. Frozen buffers
are not trainable parameters but still count toward memory/parameter budgets.
Precision diagnostics round weights to FP32/BF16 then compute in FP64: they do
not benchmark low-precision kernels or compression-file overhead.

## Native bridge: reconstruction first, not invented factual efficacy

The optional loader requires an isolated **Transformers 4.51.3** environment,
stock-Qwen3-compatible compact checkpoints, and a full local file-hash manifest.
It never downloads weights, executes custom checkpoint code, loads pickle
weights, quantizes, silently falls back to another device, or writes a checkpoint.
It rejects missing/unexpected model tensors. This loader was **not run here**.

Prepare a JSON object mapping every checkpoint-relative filename to its SHA-256.
Keep that manifest **outside** the checkpoint directory. Review its provenance;
a hash manifest establishes content identity, not trust in a checkpoint's origin.
Provide a JSON array of tokenized prompt-only records:

```json
[
  {"id":"train-1","split":"train","input_ids":[100,101]},
  {"id":"train-2","split":"train","input_ids":[100,102]},
  {"id":"check-1","split":"check","input_ids":[103,104]},
  {"id":"check-2","split":"check","input_ids":[103,105]}
]
```

The numbers above illustrate the schema, not a real calibration corpus. Supply
properly tokenized prompts from the **exact model tokenizer/chat template**;
there is no implicit tokenization, truncation, or cross-model token-ID transfer.
Use 4..128 records with at least two per split, at most 256 input tokens each,
unique IDs, and no exact prompt duplicates across or within splits. This is only
an exact-token overlap check; semantic/template contamination needs human audit.

```sh
python -m experiments.rineya_hebbian_fact_memory_v1.cli native-screen --execute-native --config experiments/rineya_hebbian_fact_memory_v1/configs/native_reconstruction.json --model-dir /EXACT/LOCAL/CHECKPOINT --manifest checkpoint_hashes.json --rows reviewed_tokenized_rows.json --module model.layers.14.mlp --alpha 0.01 --device cpu --dtype float32 --out rhf_native_screen_001
```

Layer 14 is an explicit example, **not a qualified best layer for Rineya**. For a
separately qualified CUDA environment use `--device cuda:0 --dtype bfloat16`.
The command checks 3 GiB available host RAM and, for CUDA, 3 GiB free VRAM before
loading. These are point-in-time preflights, not continuously enforced memory or
wall-time containment. Run one model on an otherwise quiescent host. Synthetic
construction is CPU FP64 with major-tensor element bounds, not a hard RSS limit.

The screen captures final-prompt MLP input/output pairs, constructs a replacement
from train rows only, and checks held-out rows. Its blend is
`base + alpha*(memory-base)`; alpha=0 is a literal bypass and alpha=1 replacement.
Its diagnostics include held-out activation MSE, full-vocabulary next-token KL,
top-1 agreement, exact bypass/restoration, and one continuation-step cache parity.
It does not generate or evaluate multi-token answers. A screen pass never sets
`native_qualified=true`; broad retention and new-fact acquisition remain separate.

## Evidence interpretation

Read `evidence/VALIDATION.md` before selecting an arm. Exact fit can coexist with
severe query-noise sensitivity. Numerical failures consume an experimental
attempt and remain visible. Two synthetic seeds and repeated contexts are not
independent evidence of broad capability gains. No published paper speedup,
asymptotic capacity result, frontier-intelligence gain, or native Rineya score
is claimed by this branch.
