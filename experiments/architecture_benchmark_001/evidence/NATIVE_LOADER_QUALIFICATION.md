# Native source-loader qualification

Scope: the `run.py::_load(..., native=True)` construction path only. This is a
**mechanical** result on tiny random fixtures. No official Supa weights were loaded,
no G0-G2 receipt was produced, and G3-G5 remain closed. It is not a model-quality,
recovery, latency, memory, or deployment claim.

- Task branch: `research/ced-native-loader-qualification-v1`
- Base commit: `f680b5dc62fac36f4b4347793f034215c3580a87` (freshly re-resolved from
  `origin/research/benchmark-001-architecture-ced40-v1`; unchanged from the reference
  SHA in issue #4, no history rewrite)
- Issue: <https://github.com/LUNCY579/ChatBubble/issues/4>

## 1. The defect reproduces on the real loader

Issue #4 raised this as a source-identified risk, not an observed failure. It reproduces.

`Qwen3RotaryEmbedding.inv_freq` is registered `persistent=False`
(`transformers/models/qwen3/modeling_qwen3.py:330` in the pinned 4.51.3), so **no
checkpoint can supply it**. `_load` builds the model under `torch.device('meta')`, and
`load_state_dict(native_state, strict=True, assign=True)` only replaces keys the state
dictionary actually contains — it still returns `<All keys matched successfully>` while
`inv_freq` stays a meta tensor. The next line then fails:

```
NotImplementedError: Cannot copy out of meta tensor; no data! Please use
torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module
from meta to a different device.
```

`NotImplementedError` subclasses `RuntimeError`, so `run.main` catches it and reports
`BLOCKED_OR_FAILED` rather than crashing. Either way **`qualify_source` (G0) could never
pass**: it calls `_load(source, device, native=True)` before any gate is evaluated.

`Qwen3RotaryEmbedding.original_inv_freq` is a plain attribute, not a registered buffer,
so `Module.to()` never touches it and it is left on meta as well. For the registered
official config (`rope_scaling: null` → `rope_type == 'default'`) that one is
**latent-only**: the two readers in `modeling_rope_utils.py` sit in the `dynamic` and
`longrope` branches, which `dynamic_rope_update` never dispatches to. It is repaired
anyway because the native constructor sets it, and `source_config` is what keeps the
dynamic branches unreachable.

Enumeration is closed: `inv_freq` is the **only** `persistent=False` buffer in the pinned
qwen3 module, and the only tensor in `Qwen3ForCausalLM` that the checkpoint cannot supply.

## 2. The fix

`run.py`, +41 lines, no behaviour change on any passing path:

- `_materialize_native_buffers(model, device)` — after `tie_weights()` and before `.to()`,
  walk `named_modules()`; for a module still owning meta buffers, re-run **that module's
  own initializer** (`module.rope_init_fn(module.config, torch.device(device))`) and make
  the same three assignments `Qwen3RotaryEmbedding.__init__` makes. Anything else raises
  `GateClosed`.
- `_require_materialized(model, device)` — after `.to()`, every parameter **and buffer**
  must be a real tensor on the intended device, or `GateClosed`.

No permissive/non-strict loading, no `ignore_mismatched_sizes`, no zero fill, no
`to_empty()`, no dependency upgrade, no tolerance change, no tokenizer/identity/topology
change, and no fixture fallback in the CLI. An unrecognized unmaterialized buffer fails
closed instead of being guessed.

### Values are natively correct, not merely present

Against the same model built the ordinary (non-meta) way from the same tensors:

| Property | Result |
|---|---|
| `inv_freq` vs native | bitwise equal (`torch.equal`), float32, max abs diff `0.0` |
| `original_inv_freq`, `attention_scaling`, `rope_type`, `max_seq_len_cached`, `original_max_seq_len` | all equal |
| every state-dict tensor | bitwise equal |
| `inv_freq` in `state_dict()` | absent — still nonpersistent |
| embedding tying | `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` |
| source files after load | SHA-256 unchanged |
| `.to(device, float32)` after the rebuild | bitwise no-op; same tensor object |

Full logits differ from an ordinarily-constructed model by `5.96e-08` (one float32 ULP).
That is **not** attributable to the meta path or the fix. Isolated by construction:

```
normal (fresh alloc)     vs A (fresh alloc, copy-load)  : bitwise equal
loaded (meta+assign+fix) vs B (fresh alloc, ASSIGN-load) : bitwise equal, max|d| = 0.0
A (copy-load)            vs B (assign-load)             : max|d| = 5.96e-08
data_ptr() % 64          : normal 0, A 0  |  loaded 56, B 56
```

The delta tracks `assign=True` keeping safetensors-backed (56-byte-offset) storage instead
of freshly allocated 64-byte-aligned storage, which changes CPU BLAS blocking. The meta
path and the fix contribute exactly zero.

## 3. Before and after

| | Before (base commit) | After |
|---|---|---|
| `run._load(..., native=True)` | raises `NotImplementedError` | returns a fully materialized model |
| Scoped suite, pinned native env | 109 would be 94 passed, 0 skipped | **109 passed, 0 skipped** |
| Tests exercising `run._load` | **0** | 15 |
| Tests asserting anything about buffers | **0** | 8 |
| `python -m ... matrix` / `status` | exit 0 | exit 0 |

The 94-passed baseline was captured in this same pinned environment before editing
(`transformers` installed, so `test_native.py` executed rather than skipping). It is not
the checked-in `92 passed, 1 skipped` figure, which was produced on Linux with
`transformers: not_installed` and is preserved unchanged in `LOCAL_MECHANICAL.json`.

**The suite passed identically with and without the fix before this change** — that is
precisely why the defect survived: nothing called `_load`.

Regression coverage was verified by disabling only the two new call sites: 6 of the
then-12 loader tests failed with the exact meta-tensor error; the helper unit tests, which
do not go through `_load`, still passed.

## 4. What the new tests cover

`tests/test_native_loader.py`, 15 cases. Native construction, tensor assignment and device
transfer are the real ones. A **real** tiny tokenizer is built and saved, so no tokenizer
substitution is needed. The single substitution is `run.source_config`, because the
registered official shape is 596,049,920 parameters and cannot be a fixture;
`source.source_config` and `audit_source` are untouched and still gate every real run.

- buffer *and* parameter materialization, device, and the `original_inv_freq` attribute
- rebuilt rotary state bitwise vs native; nonpersistence preserved
- weights and logits vs ordinary native construction
- native vs `DenseReference`: unpadded, left-padded, **and in the exact G0 call shape**
  (no `attention_mask`, no `position_ids`)
- cached continuation vs full re-forward, compared **on logits**
- BF16 checkpoint upcast, matching the registered `tensor_dtype: BF16`
- embedding tying and source immutability
- `native=False` needs no reconstruction (`DenseReference` registers zero buffers)
- fail-closed on an unrecognized meta buffer; no-op when nothing is on meta
- `_require_materialized` rejects meta tensors and foreign devices
- reference/native RoPE agreement across the registered 512-token context

### The cached-continuation oracle was strengthened after adversarial review

The first version compared only argmax token ids. On a tiny random fixture the model emits
one constant token, so that oracle was vacuous: injected `position_ids=0`, frozen
`position_ids`, and dropped-cache faults **all passed it**. Comparing step logits at the
G0 tolerance catches all three:

| Injected fault | argmax-id oracle | logit oracle | max abs logit delta |
|---|---|---|---|
| none | pass | pass | 1.19e-07 |
| `position_ids` zeroed | pass (blind) | **catches** | 7.81e-02 |
| `position_ids` frozen | pass (blind) | **catches** | 7.81e-02 |
| cache dropped each step | pass (blind) | **catches** | 2.76e-01 |

## 5. Characterized limitation, not fixed here

`model.py:35` builds RoPE inverse frequencies as `theta**(-i/h)`; pinned Transformers
builds `1/(theta**(i/h))`. Equal in real arithmetic; in float32 they differ in 24 of 64
entries, max relative `1.178e-07`, and the resulting angle error scales with absolute
position:

| position | max abs cos/sin difference | inside G0 `atol/rtol = 1e-4`? |
|---:|---|---|
| 128 | 6.6e-06 | yes |
| 511 | 3.0e-05 | yes |
| 4096 | 2.2e-04 | **no** |
| 40959 | 3.1e-03 | **no** |

The registered decoding context is 512 and `_prompt`/`_text` refuse to exceed it, so G0
parity is inside tolerance today. This is **not** changed here: altering `model.py`'s RoPE
would move the reference numerics that every existing test and the CED design bind to, and
would require requalifying the whole reference — outside this bounded deliverable.
`test_reference_and_native_rope_agree_across_the_registered_context` pins the guarantee so
that raising `decoding.max_context` cannot silently cross the gate.

## 6. Implementation digest and a platform caveat

`gates.implementation_digest()` keys its manifest on `str(p.relative_to(ROOT))`, which
yields `tests\test_native.py` on Windows and `tests/test_native.py` on POSIX. The same
bytes therefore hash differently per platform. This is **pre-existing**, not caused by this
change, and no test, gate, or workflow consumes the digest programmatically — but it means
digests must be compared like for like.

| Tree | POSIX-keyed | Windows-keyed |
|---|---|---|
| base `f680b5d` | `52e7285ae1ef0bf5ef1cc8a5da0e499df2445d9db7ebc1b3c75c78af9c1710c7` | `58e9ff70e868a78cbd2d45dc99e693b0544a70f0d8fe7f1b4d25f5da761619a3` |
| this branch | `d745d310df09c13fb5703bcdf8057e90599925a59e1058411e6f02bd5b1d8fa1` | `c5b952d85318a58a008855b304d6ebd483329d436d39ced7f902bc7861e649c6` |

The POSIX value at the base commit reproduces the `52e7285a…` recorded in
`LOCAL_MECHANICAL.json`, `LOCAL_STATUS.json` and the base commit message exactly, which
confirms those receipts were produced on Linux and are internally consistent.
`NATIVE_LOADER_MECHANICAL.json` in this directory was produced on Windows and records the
Windows-keyed value; that is the honest value for that run, not a drift.

## 7. Exact tested identity

- Runtime: python 3.11.15, Windows-10-10.0.26200-SP0, torch 2.10.0+cpu,
  transformers 4.51.3 (the pinned `native_transformers_version`), safetensors 0.7.0,
  numpy 2.3.5, pytest 9.0.2, in an isolated venv outside the repository.
- Tested source: **tiny random Qwen3 fixtures written to a pytest temporary directory**,
  vocab 37, hidden 24, 4 layers, head_dim 8, in float32 and bfloat16 variants.
- Official Supa `962abc5b…` was **not** present and **not** loaded. `source_audit`
  remains `not_requested`/`BLOCKED`. `official_weights_loaded: false`.
- Tiny random fixtures are mechanical evidence only. They are not model-quality evidence,
  and they do not open G0.
- Local execution only. No GitHub Actions run was dispatched and no workflow was modified;
  the RAB-001 workflow triggers only on `research/benchmark-001-architecture-ced40-v1`,
  so pushing this task branch runs no CI and creates no tag.

## 8. Remaining blockers

1. **G0 still requires the exact official Supa 1.7.0-devfreeze directory.** The loader is
   now qualified mechanically; it has never run on official weights.
2. The `implementation_digest()` path-separator sensitivity above.
3. `registered_context_executed` and `unmodified_native_readout` are literal `True` in the
   G0 checks dict; they are asserted, not computed. Untouched here.
4. The decode loop stops only on `tokenizer.eos_token_id`, while the source's
   `generation_config.json` declares two stop ids. Untouched here.
5. RoPE construction parity beyond 512 tokens (section 5).

Items 2-5 are recorded, not repaired: each is outside a bounded loader/correctness fix.
