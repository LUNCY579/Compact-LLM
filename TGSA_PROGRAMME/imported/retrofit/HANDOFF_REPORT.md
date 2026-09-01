# TGSA program handoff — Phase-1 v3 execution and real-model retrofit

Reporting standard per PART 11. Observations are separated from interpretation,
and every claim is tagged `OBSERVED`, `SUPPORTED`, or `HYPOTHESIS`. Negative and
inconvenient results are included.

---

## 1. OBSERVED FACTS

### 1.1 Phase-1 v3 confirmatory matrix — INCOMPLETE

`OBSERVED` Launch identity, pinned and unchanged:

| item | value |
|---|---|
| branch | `research/tgsa-phase1-identifiability-v3-powered-confirmatory` |
| design-freeze commit | `04d3c02b6f9c7398fede8416deea81ce170ea7c2` |
| design-freeze tag | `tgsa-phase1-v3-launch-2026-08-29` |
| execution commit | `72395b69beb96dc71dc80bc8c73ef75205ea51a1` |
| CI | run 33271024242, success, confirmatory job skipped |
| config SHA-256 | `9dadb157b04fefad6a4d1ed8cfee1d5ed23916e4c12e13f2dce61b097a7b8163` |
| preregistration SHA-256 | `6cfa608a510672181a847241b4b6e91fabe8d12032071c346230efb456361085` |
| dataset manifest SHA-256 | `cad5851eeba887584c9c6aa38cef11f00d752c71fb2c71c6877cdc44a3e52d54` |

`OBSERVED` Stage status:

| stage | state |
|---|---|
| PREPARE | complete |
| TRAIN | **7 / 60 cells complete, 8 in flight** |
| BARRIER | not run |
| CONFIRMATORY EVALUATE | not run |
| EVIDENCE VERIFY | not run on confirmatory artifacts |
| ANALYZE | not run |

`OBSERVED` Contamination check, run directly against the filesystem:

* confirmatory test prediction files on disk: **0**
* `matrix_manifest.json`: absent
* `analysis.json`: absent
* development prediction files: 7 (produced by the TRAIN phase, which is
  registered to produce them)
* corrupt checkpoints: 0
* preserved failure records: 0 (no cell has failed)

`SUPPORTED` The study is uncontaminated. The TRAIN phase cannot emit a test
prediction; test predictions exist only after the EVALUATE phase, which has not
run and cannot run until the barrier passes.

`OBSERVED` Nothing in the registered design was modified: seeds, thresholds,
estimator, task generation, exclusions, difficulty filtering, intervention
scheme, training steps, and architecture are all as registered at `04d3c02`.

`OBSERVED` Measured cost: ~218 ms/step (cascade), ~288 (history_control), ~258
(learned_delta) on an idle machine. All 32 cores saturate at 8-way concurrency,
so additional parallelism does not increase throughput. 150,000 steps total.

`OBSERVED` **Operational failure mode, twice.** Background training drivers were
reaped between agent turns, silently halting the run. Recovered without loss
because cells checkpoint every 250 steps and the driver is idempotent. The
current run is a genuinely detached Windows process (`Start-Process`), which has
survived. This is an environment fact, not a study defect, but it is why the
matrix is not finished.

### 1.2 Phase-1 results

`OBSERVED` **None exist.** No H1a/H1b/H1c effect, interval, per-seed table,
guardrail value, realised seed dispersion, difficulty-response effect,
intervention coverage, or NLL can be reported, because the confirmatory
evaluation has not been run. Reporting any number here would be fabrication.

### 1.3 Real Rineya-Supa-A1 checkpoint

`OBSERVED` Obtained the actual published checkpoint. Source manifest built and
verified:

* 12 artifacts hashed; **all match the publisher's `FILES.json`** (`passed: true`)
* 310 individual tensors hashed
* `model.safetensors` 1,192,135,096 bytes, sha256 `962abc5b…`
* architecture: Qwen3, 28 layers, hidden 1024, 16 Q heads, 8 KV heads,
  head_dim 128, rope_theta 1,000,000, bfloat16, vocab 151,936

`OBSERVED` Provenance from `CORE.json`: a dense LoRA merge with baked
abstention — base `Rineya-Nimbus-1.6.0-devfreeze`, adapter
`halo_long_context_sft_v1/stage_a/checkpoint-200`, adapter scale 0.8125, merged
in float32 and saved in bfloat16.

### 1.4 Real-checkpoint conversion damage

`OBSERVED` Full model, 5 converted layers (0, 7, 14, 21, 27), float32/eager,
8 development prompts:

| condition | KL max | top-1 agreement | max abs logit diff | bitwise identical |
|---|---:|---:|---:|---|
| **neutral conversion** | **0.000e+00** | **1.0000** | **0.000e+00** | **True** |
| positive control std=1e-3 | 2.826e-02 | 0.9737 | 1.402e+00 | False |
| positive control std=1e-1 | 4.747e+00 | 0.7237 | 1.521e+01 | False |

`OBSERVED` Per-layer real-weight exactness, layers 0/14/27, all 6 pretrained
tensors reused:

| dtype | vs eager | vs sdpa (relative max) |
|---|---|---:|
| float64 | bitwise identical | 1.2e-07 |
| float32 | bitwise identical | 2.5e-07 |
| bfloat16 | bitwise identical | 6.5e-03 |

`OBSERVED` Trace on the real model (layer 14, bfloat16): the converted model
answers "The capital of France is" with " Paris". Head 0 velocity 0.0000,
control 0.0000; head 1 velocity 0.2275, control 0.2275; conditioning 0.0000 on
every head.

`OBSERVED` `control_rms == velocity_rms` exactly on real weights, which is the
residual-around-identity encoder being the identity at step 0.

---

## 2. REGISTERED DECISION

`OBSERVED` **The analyzer has not run. There is no registered decision.**

The frozen analyzer emits `hypothesis_hierarchy.interpretation` directly into
`analysis.json` so the reading cannot be renegotiated after the fact. That field
does not yet exist.

---

## 3. SUPPORTED INTERPRETATION

`SUPPORTED` The **conversion path** is validated on the real checkpoint:

1. Conversion is exactly function-preserving against transformers' own eager
   attention, bitwise, at float64, float32, and bfloat16, on real pretrained
   weights, at the full-model level and the per-layer level.
2. Every pretrained tensor is reused — `q_proj`, `k_proj`, `v_proj`, `o_proj`,
   `q_norm`, `k_norm` — and the loader refuses any native tensor the retrofit
   lacks.
3. The trajectory pathway is live at step 0 (non-zero carrier and control) while
   being provably unable to affect the output (conditioning identically zero).
4. The harness that measures this is demonstrably sensitive: positive controls
   at std=1e-3 move KL to 2.8e-2 and top-1 to 0.9737.

`SUPPORTED` Against the fused SDPA kernel there is small non-zero drift
(6.5e-03 relative in bfloat16). This is reduction order, not lost function: the
retrofit computes attention per head because the carrier must evolve between
heads, while SDPA computes all heads in one fused call.

---

## 4. UNSUPPORTED CLAIMS

`OBSERVED` None of the following can be inferred from anything established so far:

* that finite-difference trajectory guidance improves capability (H1a);
* that it beats the sequential cascade (H1b);
* that example-aligned trajectory content is causal (H1c);
* that the direct-lookup guardrail holds;
* any difficulty-response signature;
* that the mechanism transfers to a pretrained model;
* that a bake would improve Rineya-family capability;
* any quality/compute frontier claim.

`OBSERVED` Specifically, **exact conversion is not evidence for the mechanism.**
It shows the retrofit starts as the pretrained model, nothing more. A model that
is bitwise identical to its parent has, by construction, learned nothing.

---

## 5. FALSIFIERS AND ANOMALIES

`OBSERVED` **Two real defects found, both by going to the real model.**

**A. `q_norm` / `k_norm` were missing.** Real Qwen3 applies per-head RMSNorm to Q
and K before RoPE. The hand-written retrofit in `attention.py` does not implement
them, and its exactness test compared against a reference I wrote myself which
shares the omission — so the test was **green while two pretrained tensors per
layer would have been silently dropped**. This is the concrete reason
architecture-reference exactness must not be used to infer real-checkpoint
exactness. Fixed by subclassing `Qwen3Attention` so transformers itself creates
every native tensor.

**B. `rope_theta` was read from the wrong place.** transformers 5.x nests it
under `rope_parameters`, which this model uses. The original reader would have
silently used 10000 instead of 1,000,000, corrupting every positional frequency
— and the exactness test would have stayed green, because the reference is built
from the same config. Now reads both layouts and fails closed on neither.

`OBSERVED` **Head 0 is structurally inert.** It has no previous carrier, so its
velocity and control are identically zero and its conditioning weights receive no
gradient. The trajectory pathway acts on `H − 1` heads (15 of 16 on this model).
Inherited from the Phase-1 operator, unchanged for compatibility, now explicit
via `effective_conditioned_heads` and asserted by tests.

`OBSERVED` **bfloat16 + SDPA drift of 6.5e-03 relative** is the practical
deployment path and is not negligible. It should be measured at the model level
before any bake, not assumed to match the eager result.

`OBSERVED` No training failures, no corrupt checkpoints, no non-finite values, no
collapsed gates observed in the 7 completed cells. Nothing to report as a
telemetry anomaly, because no confirmatory telemetry has been analysed.

---

## 6. LIMITATIONS

* The Phase-1 confirmatory result does not exist. Everything downstream of the
  mechanism gate is therefore unauthorized.
* The R0 training funnel has **not been run on real weights**. Free RAM is
  ~2.1 GB while the confirmatory matrix trains; R0 needs roughly 3× that for
  gradients. The code is written, compiles, and fails closed, but "operational"
  means executed, and it has not been.
* Layer-placement and warm-start studies are implemented but not executed, for
  the same reason.
* Conversion damage was measured on 8 short development prompts at sequence
  length 19, and per-layer at length 32. Long-context damage is not yet measured.
* The environment reaps background processes between turns; long unattended runs
  need the detached-process launch path.

---

## 7. CLAIM CEILING

The strongest defensible statement today:

> A trajectory-guided sequential attention retrofit for the real
> Rineya-Supa-A1 checkpoint has been implemented and shown to be **exactly
> function-preserving at conversion** — bitwise identical to native attention at
> float64, float32, and bfloat16 against transformers' eager path, on real
> pretrained weights, while reusing every pretrained tensor and leaving the
> trajectory pathway live but unable to affect the output.

Nothing about whether trajectory guidance **helps** is established. That is the
Phase-1 question and it is still running.

---

## 8. NEXT PERMITTED EXPERIMENT

1. Let the detached TRAIN driver finish all 60 cells.
2. Run BARRIER. It must verify 60 cells, 2,500 updates each, one clean source
   head, and matching checkpoint/train-log hashes.
3. Run the single registered EVALUATE sweep. Once only.
4. Run VERIFY, then ANALYZE.
5. Read `analysis.json` → `hypothesis_hierarchy.interpretation` and follow the
   matching result-conditional case. Do not force a TGSA-positive reading.

Only after step 5 returns Case A does the real-model pilot bake become
authorized, starting at R0.

---

## 9. RESULT-CONDITIONAL DECISION

`OBSERVED` **The mechanism gate has not returned, so none of the four registered
decisions applies.**

`MECHANISM_VALIDATED_FOR_REAL_MODEL_PILOT`,
`SEQUENTIAL_STATE_CONDITIONING_SUPPORTED_DELTA_NOT_IDENTIFIED`,
`DELTA_PARAMETERIZATION_HELPFUL_ALIGNMENT_CAUSALITY_UNESTABLISHED`, and
`MECHANISM_NOT_SUPPORTED_IN_PHASE1` are each statements *about a result*. There
is no result. Selecting one now would misrepresent an unfinished experiment.

Current state: **`PHASE1_CONFIRMATORY_IN_PROGRESS — MECHANISM_GATE_PENDING`**,
with the conversion path independently validated and the pilot bake correctly
blocked.

`READY_TO_BAKE` is explicitly **not** claimed: the real checkpoint has passed
conversion validation, but the first training funnel has not been run.
