# EFE-FUNC-001 native source review

Reviewed September 29, 2026. Integration base: `c934589026f7456352d62a83bb102c453e15519a`.
This review ran no model, GPU work, package install, or protected-bank read. No ancestor or
EFE-specific `AGENTS.md` exists in the inspected path; the isolated MQ package's continuation
rules were read when reviewing its implementation, without changing that package.

## Bound sources

| Source | Commit / inspected identity | Git blob or SHA-256 |
|---|---|---|
| DMRF `inference/gen_chunked.py` | `b8e40d377c9cd461cd30c215a2c713515904197b` | `ad0d9d9393ff3d6f6a14412b8964e3ae0f673923` |
| DMRF `analysis/build_scaled_arm.py` | same | `f95b96881c151f0f6b96606393ddc6f3fc7ea56a` |
| HLOC `research/composition_endpoint_audit_v1/run_hloc.py` | `9abd010e675d2853fe689404c188baecb21af719` | `b5ce1180631e22fc0593a230cf9cc73b96527deb` |
| MQ `experiments/model_quality_gpu_v1/native.py` | integration base | `79f0d467c2c4bd9ae8004c82afb0d4c84a637bf6` |
| MQ `resources.py` | integration base | `deffbe9ece3cb106c99e27d8a526452cd68d09e7` |
| CLARE `clare/native.py` | integration base | `16644b58154dd283b900d493712911ca2147cea9` |
| fresh-dual `gpu_lease.py` | integration base | `c2b772d3179c6129f54ca5e22eca429b54df2846` |
| RVS `driver.py` | `ac9b79ac07ad650d8815c76eb4194a180f23e8d4` | `44ec9c615bb2d4f7b3b7cceb949f4f9c47f60eda` |
| external `qwen001-20260918/stage2_generate.py` | records tree, not Git-qualified here | SHA-256 `5e9cc3201e75d30bb054bc657571c983190ad23568ecfc38755d38c807a91973` |
| external `indep001-20260921/build_indep.py` | records tree, not Git-qualified here | SHA-256 `2c642dc721c1ccc5fd3c3495d8263411681e90c3875977b708cb3e14729beb65` |

External paths above are under `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records`.
Their script contents were inspected read-only; they must not become runtime imports.

## Candidate serialization and arithmetic

The exact anchor is BF16 `model.safetensors`, SHA-256
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
MQ's source lock identifies the same 596,049,920-parameter tied Qwen3 anchor, not a different
Supa draw. The registered T endpoint is the partial FP32 checkpoint
`int001-20260917/trained_fp32.safetensors`, SHA-256
`07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d`.
The H endpoint is `compose001-20260918/trained_H_fp32.safetensors`, SHA-256
`a1103060591a91ee7d9ceb2ad7450621be352b087cd19188f2c383a0dbbc7b90`.
These are references from saved receipts, not evidence that endpoint files currently exist.

Original INDEP constructs each changed tensor as follows, with 16 CPU threads:

1. Subtract endpoint FP32 minus losslessly upcast BF16 source **in FP32**.
2. Cast the difference to FP64. For matrices, use full CPU `torch.linalg.svd` with
   `full_matrices=False`, then `(U[:, :16] * S[:16]) @ Vh[:16]`; keep vectors whole.
3. Form `alpha_h * pH + alpha_t * pT` in FP64, then `(source.double() + update).float()`.
4. Copy absent/untouched source tensors in their original BF16 storage. Thus original INDEP
   checkpoints have mixed untouched-BF16/changed-FP32 storage, despite receipt shorthand
   `stored_dtype: float32`. The FP32 loaded function uses exact upcasts for untouched weights.

The h025 saved materialization binds output SHA-256
`b5a70d700cce689ddab1e80bc449f755b2dc24b165ecfb32aa2e27162c16efec`.
Historical HLOC computes subtraction in FP64 and adds `source + pT + 0.25*pH` sequentially;
that is a different last-bit arithmetic path. It must not silently replace original INDEP.
SVD output depends on runtime/thread count. A newly reconstructed candidate must compare
parameter data against the historical materialization or a separately bound reconstruction
receipt; norm agreement alone does not prove identity. An all-FP32 streamed native load may
preserve the loaded function while changing checkpoint file bytes and storage inventory.

`build_scaled_arm.py` is a **single-ray**, full-output-dictionary builder, not the independent
two-direction candidate builder. It upcasts source BF16 to FP32, requires FP32 parent, copies
alpha=1 endpoints, permits explicit partial parents, and writes all FP32. It has no RAM,
lease, or bounded-execution guard, and its CLI rejects alpha=0 despite documentation claiming
both endpoint policies. Do not directly reuse it for the fresh protocol.

## Loader and generation choices

MQ loads local-only `Qwen3ForCausalLM`, FP32 eager attention, verifies dense geometry, all
assets, finite/materialized parameters, 596,049,920 parameters, and tied vocabulary pointers;
runtime pins are torch 2.10.0, transformers 4.57.1, safetensors 0.7.0. Its model is loaded on
CPU then moved to the requested device. That is a useful minimal design to port with EFE's own
binding and qualification; its old qualification cannot certify this candidate or generation.

CLARE loads the BF16 anchor directly onto the device and patches FP32 computation with per
module upcasts, FP32 RMSNorm/RoPE, tied embedding and chunked vocabulary head. It pins Python
3.11.15, torch 2.6.0+cu124, transformers 5.14.1 and a 0.78 allocation fraction. Casting trained
FP32 endpoints to that loader's BF16 storage erases update coordinates; its compatibility and
qualification do not transfer to .venv-mq001 or this composition. Do not use it here.

DMRF's `gen_chunked` only invokes an external HF `stage2_generate.py`: greedy batch one,
FP32, TF32 off, fixed EOS list, context-overflow skips, no truncation. It uses at most 64 rows
per fresh subprocess after historical 128-row OOM, checks row count/order/unique IDs, and
aggregates receipts. But it deletes previous output and chunk records, has no worker timeout,
lease or resource guard, and copies metadata from the last chunk without asserting all chunks
agree. Stage2 itself permits remote resolution, offload/quantization/adapter overrides, and
does not pin eager attention. Its receipt explicitly labels the serving path unqualified.
Neither wrapper nor historical path-control counts qualify this fresh run.

## Existing floors and shared coordination

- MQ qualification preload: 4.6 GiB free host and 3.0 GiB free GPU; ongoing 1.5 host / 0.5 GPU.
  Its default training profile remains 6/6 and is irrelevant evidence for full-shape inference.
- DMRF run_matrix: at least 6000 MiB free GPU before each FP32 pass. `build_n16.py` defaults to
  4.0 GiB host and 3.0 GiB disk; full output state is approximately 2.38 GB, plus tensor/SVD.
- HLOC: 3.0 GiB free host for its own sequential build/score path; shared lease covers builds
  because CPU SVD work previously starved peers. HSKILL's DenseReference probe has 3.0 host.
- CLARE probe: 3.0 host, 3.5 GPU, 4.0 disk after artifacts; training host 3.6 and ongoing 1.5.
  These are that loader's guards, not permission to loosen MQ's chosen path.

RVS's common lock is `C:/Users/clclc/BusinessProject/Chat-Bubble/.git/rineya-central-gpu0.lock`
and holds a raw nonce. CLARE/fresh-dual uses
`C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock` with a JSON ownership token. Check and
acquire both atomically/create-only before one model worker; release only owned tokens. Do
not delete foreign or malformed locks, including suspected stale records. DMRF's older
`GPU_LEASE.json` was a third time-based convention and is unsafe to inherit unreviewed.
Locks are intent signals; also inspect actual competing model processes and resources.

## Qualification still required

Native status remains **NOT_RUN**, regardless of CPU helper tests. Before fresh outcomes:
bind exact asset hashes/config/tokenizer and full key/shape/precision inventory; validate
all 220 last-20-layer endpoint tensors with no unexplained missing keys; prove tied weights,
untouched-source identity, finite tensors, reconstruction norms and parameter-data parity;
bind source code, software, threads, precision, recipe, data/scorer/protocol and session.

Then earn target-host cached versus full-prefix logit/argmax parity, manual versus HF greedy
token/EOS/termination parity, repeated row/process order behavior, prompt rendering parity,
and the registered maximum context/generation/cache envelope. Use one model at a time and
in-place candidate arithmetic, never simultaneous full-model copies. Record peak RAM/RSS,
allocated/reserved VRAM, stage wall time and generated tokens; include failure and overflow
rows. A short probe cannot qualify the maximum envelope. A guarded refusal is not execution.
No launch may proceed with missing assets, unresolved admissibility/statistical sensitivity,
or a current protocol whose `native_launch_enabled` is false.

## Prepared adapter and local validation

`native_adapter.py` ports the local FP32 Qwen3 load design into this package; it imports no
historical runtime module. Its standalone schema-1 `asset_binding` requires the seven exact
source hashes, both exact partial endpoints, registered runtime pins, fixed rank/threads and
arithmetic. The binding is an integrity record, never qualification or launch permission.
`load_candidate` currently refuses every native request, including a caller-flipped launch
flag; the private CPU loading mechanics cannot be reached through that public entry.
CUDA generation is likewise closed pending a same-run qualifier and supervised orchestration.

The prospective frozen arms are source, T, H (full standalone skill), H025 (the deployed
quarter-strength H component), and independent (1, .25). H025 supplies the matched component
comparison; adding the full H arm is an explicitly fixed diagnostic, not coefficient fitting.
Raw sum and shared scale have zero execution budget.

Local command under the existing `.venv-mq001`:

```text
python.exe -B -m unittest experiments.rineya_evidence_function_equilibrium_v1.fresh_validation.test_native_adapter -v
Ran 11 tests in 8.394s — OK
```

All eleven tests use stdlib fixtures, tiny synthetic CPU tensors, or a scripted fake decoder;
no native/random Qwen model was constructed and no CUDA API called. They cover lazy import,
closed launch, missing/wrong bindings including rehashed mutations, fixed endpoint inventory,
rank/vector behavior, historical subtraction/rounding order, invalid dtype/nonfinite input,
full-KV call chaining, EOS and max-token counts, overflow and resource refusal. They do not
qualify actual source loading, tied weights, reconstruction identity, native generation or
target-host memory fit. Those remain **NOT_RUN**.
