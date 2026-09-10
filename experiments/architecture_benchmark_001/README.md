# RAB-001 | First architecture benchmark: deeper/thinner + causal shared memory

**Foundation v1, not a demonstrated model improvement.** This is the first numbered
benchmark in this CED architecture program, not a claim that ChatBubble had no earlier
architecture research. All additions are isolated from published models and older labs.

- Branch: `research/benchmark-001-architecture-ced40-v1`
- Immutable foundation tag: `benchmark-001-architecture-ced40-foundation-v1`
- Base: `ChatBubble@a6ee824584a9e28881826ab92a11db48e124d90c`
- Package: `experiments/architecture_benchmark_001`

The tag is created by the scoped CPU workflow only after its mechanical suite and
random-native-Qwen checks pass. It labels an engineering foundation, never a trained
checkpoint or a successful reasoning benchmark. Existing tags are never moved.

## 1. Basis and experimental question

The supplied September 10 CED40 report and reference bundle are the design basis.
`materials.json` preserves their original hashes and identifies the three adapted
reference files. The PDF and ZIP remain original user attachments; no official weights
or full paper PDFs are copied into this package. The original 29-test receipt is
historical, not evidence for this implementation.

The report's core hypotheses remain separate:

1. Can more, narrower ordinary blocks use a comparable parameter budget better after
   defensible transfer and matched recovery? Compare D40 with B28 and a B28 continuation.
2. Does replacing upper-layer independent KV sources with one causal encoder-final
   memory help the quality/cost tradeoff? Compare C28 and C40 with their ordinary
   controls, including a same-shape diagnostic and matched continuation training.

**Outside evidence, not a guarantee:** Muralidharan et al., *Compact Language Models
via Pruning and Knowledge Distillation* (Minitron, arXiv:2407.14679), report width
pruning outperforming depth pruning after retraining in their studied settings.
Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device
Use Cases* (ICML 2024, arXiv:2402.14905), support deep/thin designs for their small-model
training settings. Neither paper establishes that this particular pretrained Rineya
conversion retains quality, fits a particular GPU, or runs faster. Our activation
estimator is explicitly **Minitron-inspired**, not a reproduction of every aggregation
or head-collapse detail. No DeepSeek release-dependent claim is needed to execute RAB-001.

Sources: https://arxiv.org/abs/2407.14679 and https://arxiv.org/abs/2402.14905.

## 2. Registered four-arm matrix

| Arm | Ordinary layers or encoder + decoder | Residual width | FFN width | Q / KV heads | Parameters |
|---|---:|---:|---:|---:|---:|
| B28 | 28 ordinary | 1,024 | 3,072 | 16 / 8 | 596,049,920 |
| D40 | 40 ordinary | 896 | 2,496 | 14 / 7 | 597,263,232 |
| C28 | 14 + 14 | 1,024 | 3,392 | 16 / 8 | 596,311,424 |
| C40 | 20 + 20 | 896 | 2,752 | 14 / 7 | 594,279,808 |

All retain **head dimension 128**, vocabulary 151,936, RoPE theta 1,000,000, Q/K
head RMSNorm, SwiGLU, dense GQA, and tied input/output embeddings. In particular,
B28's query projection has width 2,048, not 1,024. Reduced residual width also reduces
the embedding matrix width; vocabulary IDs and tying survive, not unchanged embeddings.

The four arms are within 0.5% of B28's parameter count. They are not FLOP-, latency-,
or FFN-shape-matched. `matrix` includes the original candidate presets plus a W28
width-only intermediate. Identity D40 at width 1,024, W28, and the width-896/FFN-2,688
ordinary/CED pair are explicitly unmatched diagnostics, not competitive equal-budget arms.

At BF16 and 8,192 cached tokens, the analytic persistent KV counts are B28 896 MiB,
D40 1,120 MiB, C28 480 MiB, C40 588 MiB. C40's count is 34.375% below B28, but this
excludes temporary attention tensors, weights, activations, allocator overhead, and
metadata. It is **not measured total-memory or speed evidence**. Dense attention and
40 sequential layers can still be expensive; 14/7-head optimized kernels are unqualified.

## 3. Actual dataflow and implemented modules

The ordinary control uses independent per-layer causal KV. The CED encoder remains
causal. A single learned normalized K/V projection turns its final token states into
shared memory. Each decoder layer has independent Q/O projections, Q normalization,
and an FFN, but no private K/V projection or decoder-local self-attention. Decoder
reads are masked by physical order, absolute positions, validity, and document segment.
Causal encoder states from future positions remain forbidden to earlier queries.

For pure CED inference, all prompt tokens run through the encoder and only the final
valid prompt token needs to run through the decoder. Every generated token still runs
through both stacks and extends encoder caches and shared memory. Training computes
all supervised positions. This is neither a recurrent variable-depth thinking loop nor
an implementation of a full sparse/MoE/vision architecture.

| Module | Implemented contract |
|---|---|
| `budget.py` | Validated dimensions, exact parameter/cache formulas, registered and diagnostic presets |
| `model.py` | Ordinary and pure-CED eager forward; tied embedding readout; causal/padding/segment masks; request-local cache, chunk/token parity and tail-only CED prefill |
| `source.py`, `source_lock.json` | Frozen official Supa asset hashes, byte count, exact native tensor inventory/shapes/BF16 headers; no downloads or fallback |
| `transfer.py` | Strict native Qwen tensor import; zero-output identity insertions; masked activation calibration; global residual and grouped-head/FFN selection; Cartesian tensor slicing; declared-donor CED conversion |
| `losses.py` | Shifted-token CE plus exact forward teacher-to-student KL with T-squared, teacher stop-gradient, padding and packed-boundary exclusion |
| `run.py`, `gates.py` | Offline source/readout qualification, source-bound identity/width qualification, exactly 25 optimizer smoke steps; content-bound append-only receipts; nonpersistent native buffer reconstruction with a materialized-on-device post-condition |
| `measure.py` | Actual eager-reference prefill and cached-decode timing, persistent cache-byte inspection, CUDA synchronization when available; paired group-level bootstrap utility |
| `receipt.py` | JUnit-derived mechanical evidence, explicitly incapable of licensing official-model stages |

Cache ownership, parameter versions, dtype/device, bank topology and metadata are
validated. Cache tensors are request-owned and must not be mutated by callers. This
is a correctness reference with explicit attention and temporary repeated KV heads,
not a production serving kernel. It has no eviction, HF `generate()` wrapper, GGUF
export, decoder-local attention, sparse selection, MoE, quantization, or model-registry
promotion. The custom serialized `model_type` prevents mistaking converted checkpoints
for ordinary Qwen checkpoints.

## 4. Transfer and initialization decisions

First verify the exact ungated **Supa Core 1.7.0** checkpoint from the frozen registry.
Its weight SHA-256 is
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
All tokenizer/configuration assets are pinned as well. The A1 readout variant is not
admitted merely because it shares weight bytes. Source directories are read-only;
unknown tensors, custom Python modeling, adapters and mismatched assets fail closed.

Then qualify 28-to-40 same-width identity expansion: preserve every original block
in order, insert twelve deterministically spaced donor copies, and zero **only** their
attention output and FFN down projections. Same-width output preservation must be
measured on the actual checkpoint before narrowing is considered. This 784M diagnostic
is not an equal-budget candidate.

Calibrate trained B28 blocks on a distinct **calibration** split, masking padding.
The estimator aggregates mean-absolute normalized residual channels across blocks,
mean-absolute FFN activations per block, and mean-L2 query-head outputs per block.
Stable ranking retains one global residual basis, complete KV/query groups, and
per-layer FFN neurons. Full head/RoPE coordinates survive. Every index is in lineage.

The executable D40 route is `calibrated B28 -> W28 -> identity-expanded D40`, after the
same-width identity gate has already passed. This is a deliberate implementation
ordering: score the original trained layers, not the zero-output inserted blocks.
The copied identity donors are sliced consistently; narrowing is not claimed to be
function-preserving. Original source tensors and files remain unchanged.

`to_ced` is an implemented, mechanically tested primitive, **not an enabled official
C40 campaign**. Shared K/V and memory normalization come from one declared upper-layer
donor, not an unexplained average. Budget-matching FFN growth copies cyclic gate/up
rows and adds zero down-projection columns. This preserves the FFN function initially,
not the changed memory topology. Full output inventories and donor indices are recorded.
C40 recovery must wait for interpretable ordinary recovery and a matched D40 continuation
control; the strict common-teacher architecture comparison is a distinct controlled study.

## 5. Install and run the foundation

Use a fresh Python 3.13 environment. From the **repository root**:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/architecture_benchmark_001/requirements-native.txt
python -m pytest -q experiments/architecture_benchmark_001/tests
python -m experiments.architecture_benchmark_001 matrix
python -m experiments.architecture_benchmark_001 status
```

For a CUDA machine, install its appropriate official PyTorch build instead of the CPU
wheel; changing hardware/runtime requires fresh source-bound qualification. The pinned
native bridge uses Transformers **4.51.3**; this is a new qualification runtime, not a
claim to repeat the published core's historical runtime or showcase scores. Local-only
loading and offline CI prevent surprise model downloads or paid training launches.

`requirements.txt` suffices for non-native mechanical tests. Missing Transformers
produces an explicit skip of both native modules. CI installs native dependencies and
separately requires the random-native parity cases to pass; a skipped native suite cannot
publish the foundation tag. Random native Qwen is never substituted for official Supa.

`tests/test_native_loader.py` drives the actual `run._load` construction path on tiny
random fixtures, because the metadata-only build assigns only the tensors a checkpoint
contains and cannot supply `Qwen3RotaryEmbedding.inv_freq`, which pinned Transformers
registers `persistent=False`. Its only substitution is the official B28 shape validator;
the tokenizer is real, and `source.source_config`/`audit_source` still gate every real
run. See `evidence/NATIVE_LOADER_QUALIFICATION.md`.

## 6. Executable next real milestone and gate sequence

The environment must already contain the exact official model directory and assets.
The first command intentionally rejects a metadata-only GitHub directory:

```bash
python -m experiments.architecture_benchmark_001 audit --source /path/to/official/Supa
python -m experiments.architecture_benchmark_001 qualify-source \
  --source /path/to/official/Supa --device cpu --out /path/to/new/rab001-g0
python -m experiments.architecture_benchmark_001 qualify-identity \
  --source /path/to/official/Supa --device cpu \
  --g0 /path/to/new/rab001-g0/G0.json --out /path/to/new/rab001-g1-identity
python -m experiments.architecture_benchmark_001 calibrate-d40 \
  --source /path/to/official/Supa --device cpu --calibration /path/to/calibration.jsonl \
  --g1 /path/to/new/rab001-g1-identity/G1_IDENTITY.json --out /path/to/new/rab001-g1-width
python -m experiments.architecture_benchmark_001 smoke \
  --source /path/to/official/Supa --device cpu --calibration /path/to/calibration.jsonl \
  --train /path/to/train-smoke.jsonl --g1 /path/to/new/rab001-g1-width/G1_WIDTH.json \
  --out /path/to/new/rab001-g2
```

Use the same `--pilot` path across stages when overriding the supplied one. New output
directories are mandatory; earlier outputs and published source directories cannot be
overwritten. Failure receipts never mean permission to proceed. These reference paths
can consume substantial RAM/VRAM, especially while two FP32 models and full logits are
resident. There is **no 8-GB fit promise**, no automatic device fallback, and no reliable
training-time estimate before real qualification. A CPU run is allowed, not promised fast.

| Gate | Actual boundary in foundation v1 |
|---|---|
| G0 | Exact source audit, unmodified native/reference logits, 24-row open prompt/parser/termination pilot and actual short-context resource execution |
| G1_IDENTITY | Source-bound same-width 40-layer exact-output check, chunked-cache parity, unchanged source and lineage |
| G1_WIDTH | Calibrated registered D40 tensors, complete lineage, finite forward and exact save/reload; **not quality recovery** |
| G2 | Exactly 25 finite SGD engineering steps with connected/nonzero intended gradients over the run, teacher stop-gradient, save/reload and cost counts |
| G3 | **Closed:** recovery campaign not implemented or licensed |
| G4 | **Closed:** three-seed registered DEV campaign not implemented or licensed |
| G5 | **Closed:** sealed evaluation campaign not implemented or licensed |

G0 requires at least 80% strictly parseable integer answers, 80% EOS termination, and
at least one correct and one incorrect open-pilot response. A floor/ceiling or formatting
failure blocks conversion. Redesign an uninformative pilot only through a versioned
pre-student amendment, never by weakening thresholds after seeing student outcomes.
The supplied 24 arithmetic-composition cases are an **open engineering pilot**, not a
sealed reasoning benchmark or a training dataset. Their row count is not a power analysis.

G2 uses SGD without momentum at 1e-5, equal CE/KL weights, temperature 1, and gradient
clipping. This limits optimizer-state allocation for an engineering check. It is not a
recommended recovery optimizer or evidence that 25 steps restore pretrained capability.
The loss chunks supervised positions but still receives full model logits; it does not
solve full-vocabulary LM-head memory costs.

## 7. Data, receipts and controls before scientific training

Calibration and smoke-training JSONL rows must have this explicit structure:

```json
{"id":"cal-0001","group_id":"cal-document-family-01","split":"calibration","text":"A complete training-permitted text..."}
```

Training rows use `split: "train"`. Do not label evaluation examples as calibration.
IDs, declared template/document groups, and normalized content must be disjoint across
pilot/calibration/train. This catches obvious leakage, not all semantic contamination;
curation, licenses, split provenance and later sealed-data access remain human-reviewed
requirements. The runner rejects truncation beyond its registered 512-token pilot/smoke
context. Long-context campaigns need a separate preregistration and hardware qualification.

Receipts bind implementation, protocol, source audit/weight hash, data hash, device,
precision, backend, runtime, actual output artifacts, and parent receipt digest. Artifact
changes, code/data drift, and synthetic receipt classes block downstream stages. Hashes
are drift detection, **not cryptographic attestations against a malicious actor**. No
arbitrary JSON flag can enable the absent G3-G5 runner.

Before implementing G3, fill the intentionally null protocol fields: group-disjoint
recovery/regression/DEV/TEST manifests, training-token and search budgets, a meaningful
minimum quality effect, noninferiority and regression margins, and measured resource
envelopes. Keep the frozen teacher and answer budgets constant. Charge teacher forward
passes, calibration, failed tuning runs and extra recovery to each recipe. Compare B28
continuation, D40 continuation, and same-shape topology controls rather than just the
untouched source. Three registered seeds are 17, 29, and 43; never report only the winner.

`paired_group_interval` is a tested analysis utility, not an evaluator that opens TEST.
It requires aligned IDs, matching group membership and boolean outcomes, then resamples
independent group-mean effects. Report every seed and its uncertainty; aggregate seed
and task dependence explicitly before drawing a scientific conclusion. The profiler
measures this eager reference only, separates prefill/first choice from cached decoding,
and reports sample timings and actual persistent KV bytes without claiming deployment speed.

## 8. Evidence and limitations

`evidence/` contains freshly executed local receipts and logs; `local-junit.xml.gz`
is the losslessly compressed original JUnit report (the receipt hashes its XML bytes). Their runtime, code hash,
passed/skipped counts and explicit lack of official model evidence are authoritative.
The original `LOCAL_*` receipts record `92 passed, 1 skipped` on Linux with Transformers
absent, and are preserved unchanged. `NATIVE_LOADER_*` and `native-loader-*` record a
separate, later run of the loader qualification in the pinned native environment:
`109 passed, 0 skipped`, still with no official weights. Note that
`implementation_digest()` keys its manifest on the native path separator, so its value is
platform dependent and must be compared like for like; both conventions are tabulated in
`NATIVE_LOADER_QUALIFICATION.md`.
The scoped GitHub workflow uploads its own JUnit/receipt/matrix/status artifacts bound
to the pushed SHA. A green CI checks mechanics and random native compatibility, not
that G0-G5 have run. Git metadata and remote workflow status must be inspected before
claiming a remote tag or CI success.

The next decisive milestone is a genuine unmodified official Supa G0 receipt, followed
by real-weight identity parity. Until then, no Rineya quality improvement, recovery,
GPU fit, reasoning gain, latency improvement, or deployable replacement is established.
