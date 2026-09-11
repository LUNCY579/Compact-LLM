# Official-source gate ladder, executed on the exact registered Supa checkpoint

**All four registered arms can now be built from official weights**, with a chained receipt
DAG. This is still **not** a model-quality, recovery, efficacy, latency, memory or deployment
claim, and **no arm has been compared with any other**. G3-G5 remain closed and were verified
closed after the run.

- Task branch: `research/ced-native-loader-qualification-v1`
- Base commit: `f680b5dc62fac36f4b4347793f034215c3580a87`
- Enabled by the nonpersistent native buffer fix in `NATIVE_LOADER_QUALIFICATION.md`. Before
  that fix `_load(..., native=True)` raised `Cannot copy out of meta tensor`, so **G0 could
  not run at all**.

## Exact tested identity

| | |
|---|---|
| Source directory | `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze` (local, read-only) |
| `core_id` | `rineya.core.supa.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125` |
| `model.safetensors` | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`, 1,192,135,096 bytes, 310 BF16 tensors |
| `audit_digest` | `97cd76af0fb2d6064c5cb954906ac493d21e40429cebe55f0bed0000b23d9ef8` |
| Parameters | 596,049,920 |
| Runtime | python 3.11.15, Windows-10-10.0.26200-SP0, torch 2.10.0+cpu, transformers 4.51.3, safetensors 0.7.0 |
| Device / precision / backend | cpu / float32 / eager |
| `implementation_sha256` | `dc212c9fc032258a3d5e844d99b7f27a41b850327e0c06b432b3b7fa852cbd0c` (Windows-keyed; see the platform caveat in `NATIVE_LOADER_QUALIFICATION.md` §6) |

All seven registered asset SHA-256 values matched `source_lock.json`; the tensor inventory,
per-tensor shapes and BF16 dtype all matched. The source was re-audited after every stage and
was byte-identical each time.

**Requalification note.** An earlier ladder passed at `implementation_sha256 c5b952d8…`
(commit `2f65f9e`). Adding the `convert-ced` stage changed the implementation digest, so
every receipt was regenerated from scratch, exactly as `AGENTS.md` requires ("Requalify after
code/data/source/runtime changes"). The superseded receipts remain in this branch's git
history at that commit; they are not relabelled, and nothing from them is reused here.

## Results

| Stage | Status | Receipt | Parent | Wall |
|---|---|---|---|---|
| G0 | **PASS** | `907ae301ce14e1de…` | – | 46.9 s |
| G1_IDENTITY | **PASS** | `24c169bacac5588d…` | G0 | ~35 s |
| G1_WIDTH | **PASS** | `8af4f26ac0b145ff…` | G1_IDENTITY | ~100 s |
| G2 | **PASS** | `29067288a7de0816…` | G1_WIDTH | 198.0 s |
| G1_CED_C28 | **PASS** | `8c8929ddb1ccb441…` | G0 | ~61 s |
| G1_CED_C40 | **PASS** | `5878e9fd8f657d99…` | G1_WIDTH | ~61 s |
| G3 / G4 / G5 | **BLOCKED** (verified after the run) | – | – | – |

The chain is a DAG, not a line: G0 feeds both G1_IDENTITY and G1_CED_C28, and G1_WIDTH feeds
both G2 and G1_CED_C40. Every `parent_receipt_sha256` was validated by `validate_receipt`
against a freshly recomputed context, so a changed source, dataset, runtime or code digest
would have blocked the child stage.

### G0 — source and readout qualification

All seven checks true. `hits 6 / 24`, `parse_rate 1.0`, `termination_rate 1.0`,
`max_abs_native_reference_error 1.7929e-04`.

The pilot is graded, not flat:

| group | correct |
|---|---|
| `pilot-depth-1` | **6 / 6** |
| `pilot-depth-2` | 0 / 6 |
| `pilot-depth-4` | 0 / 6 |
| `pilot-depth-6` | 0 / 6 |

Every one of the 24 responses was a strictly parseable integer terminating on EOS in 2-3
tokens. This satisfies `diagnostic_not_floor_or_ceiling`: depth-1 is a saturation control and
depths 2-6 have full headroom, so the pilot can discriminate in both directions. **6/24 is not
a benchmark score** — it is a 24-row open engineering pilot on arithmetic composition, and its
row count is not a power analysis.

**On the parity number.** `max_abs_native_reference_error` is the raw maximum absolute logit
difference. The gate is `torch.testing.assert_close(atol=1e-4, rtol=1e-4)`, which passes when
`|a-b| <= atol + rtol*|b|`; `native_reference_logits` is `true`, so every element satisfied
that criterion. Contributions are BF16→FP32 upcast, 28-layer accumulation order, and the
reference/native RoPE construction difference characterized in
`NATIVE_LOADER_QUALIFICATION.md` §5.

### G1_IDENTITY — same-width 28 to 40 expansion

All five checks true. **`max_abs_logit_error: 0.0`** — bitwise function-preserving on the real
pretrained checkpoint, not merely close. Chunked-cache parity held. 784,821,248 parameters.
Twelve zero-output identity blocks were inserted at `[3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36,
39]`, with all 28 original layers retained in order. This 784M model is an explicitly
**unmatched diagnostic**, not a competitive arm.

### G1_WIDTH — calibrated conversion to the registered D40

All six checks true. **597,263,232 parameters**, matching `D40_896_near_baseline_budget`
exactly. Route: calibrated B28 → W28(896) → identity-expanded D40. Selection retained 896 of
1024 residual channels, 7 of 8 complete KV groups (14 of 16 query heads, full head coordinates
and GQA groups preserved), and 2496 of 3072 FFN neurons per layer, with a genuinely per-layer
result: layer 0 dropped KV group 5 while layer 27 dropped group 1.

`lineage.json` records every retained index, plus `function_preserving: false` and
`recovery_required: true`. **This D40 is pruned and unrecovered. Its accuracy was deliberately
not measured**, because an unrecovered pruned model's score has no control to compare against
and would invite exactly the misreading the protocol forbids.

### G2 — 25-step optimizer engineering smoke

All five checks true: 25 steps completed, **every intended parameter received a finite,
nonzero gradient over the run**, exact save/reload, source and training data unchanged. SGD
without momentum at lr 1e-5, CE/KL 0.5/0.5, temperature 1, gradient clipping, frozen teacher.
3,180 student and 3,180 teacher forward tokens.

Losses stayed finite in `[2.911, 4.030]`, mean 3.448, and **show no downward trend** —
first-five mean 3.484, last-five 3.520. That is the correct outcome: 25 steps at lr 1e-5 over a
different row each step is a plumbing check. **It is not recovery, not learning, and not
evidence that the pruned model can be restored.**

### G1_CED_C28 and G1_CED_C40 — declared-donor CED conversion

Both arms pass all eleven checks: registered dimensions (verified on the *reloaded* config, so
this also covers the serialization round-trip), exact parameter count, registered donor layer,
shared encoder-final memory with **no private decoder K/V**, absent decoder-local attention,
finite logits, exact save/reload, exact decoded choice under both tail-only prefill and
chunked cache, unmodified donor, unchanged source.

| | C28 | C40 |
|---|---|---|
| donor | `B28_1024_official_shape` (the official source) | `D40_896_near_baseline_budget` (from G1_WIDTH) |
| parent receipt | G0 | G1_WIDTH |
| donor layer | 14 | 20 |
| parameters | 596,311,424 | 594,279,808 |
| ratio to B28 | 1.000439 | 0.997030 |
| analytic KV bytes/token | 61,440 | 75,264 |
| analytic KV ratio to B28 | 0.5357 | 0.65625 |

Both are inside the registered 0.5% parameter tolerance. C40's 0.65625 reproduces the
README's analytic 34.375% reduction exactly. **These are analytic cache counts, not measured
memory**, and neither conversion is function preserving.

## Measured cache-order numerics, with the ordinary control

The CED decoder replaces every upper-layer KV source with one incrementally-built shared bank,
so it is worth asking whether chunked or tail-only decoding still reproduces full decoding.
Measured on the real converted arms, with the ordinary D40 as the control, on the same
36-token pilot prompt:

| model | property | max abs Δ | Δ / logit scale | elements outside `atol+rtol*\|expected\|` | top-1 |
|---|---|---|---|---|---|
| D40 (ordinary control) | tail-only | 2.34e-05 | 1.1e-06 | 0 / 151,936 | same |
| D40 (ordinary control) | chunked | 5.67e-05 | 1.8e-06 | 0 / 2,734,848 | same |
| C28 (CED) | tail-only | 2.10e-04 | 1.1e-05 | 19 / 151,936 | same |
| C28 (CED) | chunked | 3.95e-04 | 1.8e-05 | 4,114 / 2,734,848 (0.15%) | same |
| C40 (CED) | tail-only | 8.92e-05 | 6.9e-06 | 0 / 151,936 | same |
| C40 (CED) | chunked | 3.03e-04 | 1.5e-05 | 1,251 / 2,734,848 (0.05%) | same |

**CED cache-order divergence is about an order of magnitude larger than the ordinary
control's**, consistently across both arms and both properties. That is a real, measurable
architectural consequence, not a bug: a single shared bank funnels every decoder read through
one incrementally-built K/V, so chunk-boundary accumulation differences propagate to every
decoder layer instead of averaging across independent per-layer banks.

It is benign for decoding at this scale — **the argmax is exact in all six cases** — and the
elements that exceed the strict per-element reading are concentrated at near-zero logits,
where an absolute tolerance of 1e-4 is below what fp32 can carry through 28-40 layers.

**How this is gated, and why.** The stage gates on (a) exact decoded choice, which is not a
tunable quantity, and (b) `max|Δ| <= atol + rtol*max|expected|`, a scale-appropriate bound for
tensors spanning ±30. It records `elementwise_g0_violations` as a metric so the stricter
reading is visible rather than hidden. This is deliberately disclosed: an earlier draft of the
stage gated *only* on the scaled bound and reported no violation counts, which would have made
three of these four results look unconditionally clean. The ordinary-control row is what
establishes that the CED numbers are an architectural property and not a conversion defect.

## Data provenance

The registered pilot (`data/pilot.jsonl`, 24 rows, sha256 `69bdc5ac…7da6b`) was already in the
package. Calibration and smoke-training splits did not exist and were built for this work:

| split | file | rows | families | sha256 |
|---|---|---|---|---|
| calibration | `data/calibration.jsonl` | 156 | 156 | `8b76a625…c98655` |
| train | `data/train-smoke.jsonl` | 25 | 25 | `c32868c8…51da38` |

Both are drawn from `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`, which
its own data card declares **synthetic, procedurally generated, train-only**. Per-class file
hashes are in `data/CALIBRATION_PROVENANCE.json`.

Declared curation decisions a reviewer may disagree with:

1. **`arithmetic_chain` excluded** from both splits, so calibration cannot mirror the pilot's
   add/multiply-chain format. Other quantitative families (`percentage_reasoning`,
   `ratio_proportion`, `unit_conversion`, `counting_sets`, …) were **kept**, because removing
   all numeric reasoning would bias the pruning against a capability the model is used for.
2. **One row per template family**, so all rows are independent under the protocol's declared
   `template_or_document_family` resampling unit.
3. Rendered with the **official Supa chat template**, `enable_thinking=False`, so calibration
   activations match the distribution G0 decodes.
4. Seeds 17 (calibration) and 29 (train), both registered protocol seeds.
5. Disjointness from the pilot and from each other asserted at build time on id, group and
   normalized content, and again by the runner's `ensure_disjoint`.

**Limitation, stated plainly:** the corpus records `human_review_status: pending` for every
row. This is an **engineering** calibration adequate for a mechanical gate whose own receipt
records `quality_recovery_demonstrated: false`. It is not a reviewed scientific corpus, and a
G3 recovery campaign must not reuse it without that review.

## What is deliberately not in Git

Four 2.3-2.4 GB `model.safetensors` outputs (G1_WIDTH, G2, C28, C40) are excluded by
`.gitignore`; weights never enter this repository. Their SHA-256 values are bound inside the
receipts, so `validate_receipt` on those copied receipts needs the local run directory to
re-verify that one artifact. The G0 and G1_IDENTITY receipts bind only small artifacts and
re-validate in place from a fresh clone.

Receipt and artifact files are copied **byte-exact**; `evidence/.gitattributes` marks the
directory `-text` so no clone's autocrlf setting can rewrite bytes a SHA-256 binds.
Normalizing line endings was tried once and correctly failed the artifact check.

## Reproduction

```bash
S=Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
D=experiments/architecture_benchmark_001/data
python -m experiments.architecture_benchmark_001 audit --source $S
python -m experiments.architecture_benchmark_001 qualify-source   --source $S --device cpu --out OUT/g0
python -m experiments.architecture_benchmark_001 qualify-identity --source $S --device cpu \
    --g0 OUT/g0/G0.json --out OUT/g1-identity
python -m experiments.architecture_benchmark_001 calibrate-d40    --source $S --device cpu \
    --calibration $D/calibration.jsonl --g1 OUT/g1-identity/G1_IDENTITY.json --out OUT/g1-width
python -m experiments.architecture_benchmark_001 smoke            --source $S --device cpu \
    --calibration $D/calibration.jsonl --train $D/train-smoke.jsonl \
    --g1 OUT/g1-width/G1_WIDTH.json --out OUT/g2
python -m experiments.architecture_benchmark_001 convert-ced      --source $S --device cpu \
    --arm C28 --g0 OUT/g0/G0.json --out OUT/ced-c28
python -m experiments.architecture_benchmark_001 convert-ced      --source $S --device cpu \
    --arm C40 --calibration $D/calibration.jsonl \
    --g1 OUT/g1-width/G1_WIDTH.json --out OUT/ced-c40
```

Whole ladder: about 9 minutes of CPU wall time. Peak resident memory stayed under roughly 8 GB
on a 15.2 GB machine; G1_IDENTITY and G2 page against the commit charge. There is still **no
8-GB fit promise** and no throughput claim — these are FP32 eager CPU runs on 24-64 token
prompts.

## Remaining blockers and the next milestone

1. **G3 recovery is closed and must stay closed** until the intentionally null `protocol.json`
   fields are registered by a human: group-disjoint recovery/DEV/TEST manifests, recovery-token
   and search budgets, a meaningful minimum effect, noninferiority and regression margins, and
   measured resource envelopes. Choosing those after seeing G0 would be a results-based choice.
2. C40 needs a **D40 control receiving the matched continuation budget**; matching parameter
   counts does not make a controlled comparison.
3. The calibration corpus needs human review before supporting any scientific claim.
4. Every arm produced here is unrecovered. Nothing about relative quality is known.
5. Unchanged pre-existing items from `NATIVE_LOADER_QUALIFICATION.md` §8: the path-separator
   digest sensitivity, the two hardcoded-`True` G0 checks (`registered_context_executed`,
   `unmodified_native_readout`), the single-EOS decode stop, and RoPE parity beyond 512 tokens.
6. Remote CI on the CED branch still fails before any step executes at the exact head
   `f680b5dc` (runs `34508018827` and the later `34508605035`). Infrastructure-level, not a
   test failure; unchanged by this work.
