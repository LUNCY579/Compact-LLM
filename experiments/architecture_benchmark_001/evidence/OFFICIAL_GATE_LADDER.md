# Official-source gate ladder G0 - G2, executed on the exact registered Supa checkpoint

**This is the first official-model execution evidence for RAB-001.** Every previously
recorded result in this package was mechanical or random-native. It is still **not** a
model-quality, recovery, efficacy, latency, memory or deployment claim, and it compares
no architecture arm against any other. G3-G5 remain closed and were verified closed after
the run.

- Task branch: `research/ced-native-loader-qualification-v1`
- Base commit: `f680b5dc62fac36f4b4347793f034215c3580a87`
- Enabled by the nonpersistent native buffer fix recorded in
  `NATIVE_LOADER_QUALIFICATION.md`. Before that fix `_load(..., native=True)` raised
  `Cannot copy out of meta tensor`, so **G0 could not run at all**.

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
| `implementation_sha256` | `c5b952d85318a58a008855b304d6ebd483329d436d39ced7f902bc7861e649c6` (Windows-keyed; see the platform caveat in `NATIVE_LOADER_QUALIFICATION.md` §6) |

All seven registered asset SHA-256 values matched `source_lock.json`; the tensor
inventory, per-tensor shapes and BF16 dtype all matched. The source directory was
re-audited after every stage and was byte-identical each time.

## Results

| Stage | Status | Receipt | Parent | Wall |
|---|---|---|---|---|
| G0 | **PASS** | `1a97aa893c899b54194b43d2f8ad1301f128135a5aa8aafbfbaa5eda3a3a0d61` | – | 35.8 s |
| G1_IDENTITY | **PASS** | `ba6e165f4135c4b7e9e285edb440d519a5e9c789d25faaf5260b6bba5a344dde` | G0 | 42.0 s |
| G1_WIDTH | **PASS** | `95bd7814bd833afbe3f4b56eaa77837591006a8b2d0ceb73176cffb97c1428b5` | G1_IDENTITY | 81.9 s |
| G2 | **PASS** | `396210983aa3eba97b1d1363d9a2a365f5bbfbc21058218900e519e299dad4f6` | G1_WIDTH | 167.3 s |
| G3 / G4 / G5 | **BLOCKED** (verified after the run) | – | – | – |

Each receipt's `parent_receipt_sha256` chains to the previous stage, so the ladder is
content-bound end to end.

### G0 - source and readout qualification

All seven checks true. `hits 6 / 24`, `parse_rate 1.0`, `termination_rate 1.0`,
`max_abs_native_reference_error 1.7929e-04`.

The pilot is graded, not flat:

| group | correct |
|---|---|
| `pilot-depth-1` | **6 / 6** |
| `pilot-depth-2` | 0 / 6 |
| `pilot-depth-4` | 0 / 6 |
| `pilot-depth-6` | 0 / 6 |

Every one of the 24 responses was a strictly parseable integer that terminated on EOS in
2-3 tokens. This satisfies `diagnostic_not_floor_or_ceiling` and means the pilot can
discriminate in both directions: depth-1 is a saturation control, depths 2-6 have full
headroom. **6/24 is not a benchmark score.** It is a 24-row open engineering pilot on
arithmetic composition, not a sealed reasoning evaluation, and its row count is not a
power analysis.

**On the parity number.** `max_abs_native_reference_error` is the raw maximum absolute
logit difference. The gate is `torch.testing.assert_close(atol=1e-4, rtol=1e-4)`, which
passes when `|a-b| <= atol + rtol*|b|`. `native_reference_logits` is `true`, so every
element satisfied that combined criterion; the raw 1.79e-04 maximum is a different and
stricter quantity and does not contradict it. Contributions are BF16→FP32 upcast,
28-layer accumulation order, and the reference/native RoPE construction difference
characterized in `NATIVE_LOADER_QUALIFICATION.md` §5.

### G1_IDENTITY - same-width 28 to 40 expansion

All five checks true. **`max_abs_logit_error: 0.0`** - the expansion is bitwise
function-preserving on the real pretrained checkpoint, not merely close. Chunked-cache
parity held. 784,821,248 parameters. Twelve zero-output identity blocks were inserted at
deterministically spaced positions `[3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]`, with
all 28 original layers retained in their original order.

This 784M model is an explicitly **unmatched diagnostic**, not a competitive arm.

### G1_WIDTH - calibrated conversion to the registered D40

All six checks true. **597,263,232 parameters**, matching the registered
`D40_896_near_baseline_budget` exactly. Route: calibrated B28 → W28(896) → identity-expanded D40.

Selection retained 896 of 1024 residual channels, 7 of 8 complete KV groups (14 of 16
query heads, full head coordinates and GQA groups preserved), and 2496 of 3072 FFN
neurons per layer. The selection is genuinely per-layer, not a constant: layer 0 dropped
KV group 5 while layer 27 dropped group 1.

`lineage.json` records every retained index. It also records
`function_preserving: false` and `recovery_required: true`. **This D40 is a pruned,
unrecovered checkpoint. Nothing here claims it retains the source model's quality**, and
its accuracy was deliberately not measured, because an unrecovered pruned model's score
has no control to be compared against and would invite exactly the misreading the
protocol forbids.

### G2 - 25-step optimizer engineering smoke

All five checks true: 25 steps completed, **every intended parameter received a finite,
nonzero gradient over the run**, exact save/reload, source and training data unchanged.
SGD without momentum at lr 1e-5, CE/KL 0.5/0.5, temperature 1, gradient clipping, frozen
teacher. 3,180 student and 3,180 teacher forward tokens.

Losses stayed finite and bounded in `[2.911, 4.030]`, mean 3.448. **They show no downward
trend** - first-five mean 3.484, last-five mean 3.520. That is the expected and correct
outcome: 25 steps at lr 1e-5 over a different row each step is a plumbing check that
gradients flow and the optimizer does not explode. **It is not recovery, not learning,
and not evidence that the pruned model can be restored.**

## Data provenance

The registered pilot (`data/pilot.jsonl`, 24 rows, sha256
`69bdc5ac3350814ad8592e79001815dde58484f16ff6030519a7d09f57e7da6b`) was already in the
package. Calibration and smoke-training splits did not exist and were built for this run:

| split | file | rows | families | sha256 |
|---|---|---|---|---|
| calibration | `data/calibration.jsonl` | 156 | 156 | `8b76a625…c98655` |
| train | `data/train-smoke.jsonl` | 25 | 25 | `c32868c8…51da38` |

Both are drawn from `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`,
which its own data card declares **synthetic, procedurally generated, train-only**. Full
per-class file hashes are in `data/CALIBRATION_PROVENANCE.json`.

Declared curation decisions, which a reviewer may disagree with:

1. **`arithmetic_chain` was excluded** from both splits, so the calibration cannot mirror
   the pilot's add/multiply-chain format. Other quantitative families (`percentage_reasoning`,
   `ratio_proportion`, `unit_conversion`, `counting_sets`, …) were **kept**, because
   removing all numeric reasoning would bias the pruning against a capability the model
   is actually used for.
2. **One row per template family**, so all 156 calibration and 25 training rows are
   independent under the protocol's declared `template_or_document_family` resampling unit.
3. Rows are rendered with the **official Supa chat template**, `enable_thinking=False`,
   so calibration activations match the distribution G0 actually decodes.
4. Seeds 17 (calibration) and 29 (train), both registered protocol seeds.
5. Disjointness from the pilot and from each other is asserted at build time on id,
   group and normalized content, and again by the runner's own `ensure_disjoint`.

**Limitation, stated plainly:** the corpus records `human_review_status: pending` for
every row. This calibration set is therefore an **engineering** calibration adequate for
a mechanical gate whose own receipt records `quality_recovery_demonstrated: false`. It is
not a human-reviewed, registered scientific calibration corpus, and a G3 recovery campaign
must not reuse it without that review.

## What is deliberately not in Git

The two 2,389,097,392-byte `model.safetensors` outputs of G1_WIDTH and G2 are excluded by
`.gitignore`; weights never enter this repository. Their SHA-256 values are bound inside
the receipts (`6cfdddfd…` and `fe309263…`). Consequently `validate_receipt` on the copied
G1_WIDTH and G2 receipts cannot re-verify those two artifacts from a fresh clone - it
needs the local run directory. The G0 and G1_IDENTITY receipts bind only small artifacts
and **do** re-validate in place from the repository; that was checked for G0 after copying.

Receipt and artifact files are copied **byte-exact**. An `.gitattributes` with `* -text`
was added to `evidence/` so no clone's autocrlf setting can rewrite bytes that a SHA-256
binds. Normalizing line endings was tried first and correctly failed the artifact check.

## Reproduction

```bash
S=Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
python -m experiments.architecture_benchmark_001 audit --source $S
python -m experiments.architecture_benchmark_001 qualify-source   --source $S --device cpu --out OUT/g0
python -m experiments.architecture_benchmark_001 qualify-identity --source $S --device cpu \
    --g0 OUT/g0/G0.json --out OUT/g1-identity
python -m experiments.architecture_benchmark_001 calibrate-d40    --source $S --device cpu \
    --calibration experiments/architecture_benchmark_001/data/calibration.jsonl \
    --g1 OUT/g1-identity/G1_IDENTITY.json --out OUT/g1-width
python -m experiments.architecture_benchmark_001 smoke            --source $S --device cpu \
    --calibration experiments/architecture_benchmark_001/data/calibration.jsonl \
    --train experiments/architecture_benchmark_001/data/train-smoke.jsonl \
    --g1 OUT/g1-width/G1_WIDTH.json --out OUT/g2
```

Peak resident memory stayed under roughly 8 GB on a 15.2 GB machine; G1_IDENTITY and G2
page against the commit charge. There is still **no 8-GB fit promise** and no throughput
claim: these are FP32 eager CPU runs on 24-64 token prompts.

## Remaining blockers and the next milestone

1. **There is no CLI stage that converts to CED at all.** `to_ced` is an implemented and
   mechanically tested primitive, but `run.py` exposes only the ordinary D40 route
   (`qualify-source`, `qualify-identity`, `calibrate-d40`, `smoke`). C28/C40 - the actual
   subject of this benchmark - cannot yet be produced from official weights by any
   registered command. **This is the next single milestone.**
2. C40 additionally requires a recovered ordinary donor and a matched D40 continuation
   control, both of which live behind G3.
3. Before G3, the intentionally null `protocol.json` fields must be frozen: group-disjoint
   recovery/DEV/TEST manifests, token and search budgets, minimum effect, noninferiority
   and regression margins, and measured resource envelopes.
4. The calibration corpus needs human review before it can support a scientific claim.
5. The pre-existing items in `NATIVE_LOADER_QUALIFICATION.md` §8 are unchanged: the
   path-separator digest sensitivity, the two hardcoded-`True` G0 checks
   (`registered_context_executed`, `unmodified_native_readout`), the single-EOS decode
   stop, and RoPE parity beyond 512 tokens.
6. Remote CI on the CED branch still fails before any step executes, at the exact head
   `f680b5dc` (runs `34508018827` and the later `34508605035`). Infrastructure-level, not
   a test failure; unchanged by this work.
