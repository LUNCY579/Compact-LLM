# PCSM results and evidence ledger

Evidence snapshot: **2026-08-27 05:23 UTC**.

## 1. Outcome summary

| Stage | Status | Permitted conclusion |
|---|---|---|
| Software qualification | PASS | The qualified implementation satisfies the tested engineering invariants |
| Synthetic mathematical feasibility | PASS | The pathway can learn and exploit peer-dependent structure when the synthetic hidden optimum is constructed from that structure |
| Real resource pilot | VALID_RESOURCE_PILOT | One-seed resource/correctness evidence only; it cannot advance |
| Full transformer one-step Phase C | RUNNING / PENDING | Pre-outcome schema-v2 hardening/restart is active; no confirmatory effectiveness conclusion yet |
| Real-peer advancement gate | PENDING | No PASS or STOP evidence has been computed yet |
| Full matched SFT Phase D | STOP | Prohibited unless the complete Phase-C gate passes |

Nothing in the current positive evidence proves transformer-layer cooperation,
real long-context improvement, inference-time improvement, causal layer
specialization, speed improvement, or superiority to PEFT methods.

## 2. Software qualification

### 2.1 Current run

The qualification suite was run from the PCSM lab root with its pinned,
dev-synced local environment. The original repository environment is reserved
for the CUDA model runtime described in Section 4.

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m ruff check src tests scripts
& '.\.venv\Scripts\python.exe' -m mypy --strict src\rineya_pcsm
```

Observed current result:

```text
142/142 tests passed
All checks passed!
Success: no issues found in 28 source files
```

An earlier 98-test milestone had already passed. Direct gap tests, schema-v2
runtime tests, and adversarial gate tests then landed; 142/142 is the current
evidence. Test count alone is not treated as proof: the direct tests cover every
one of the 25 required qualification checks plus global support-union reporting,
root-of-trust authentication, exact cell grids, forged evidence, and Phase-D
reauthentication.

Development environment captured during the run:

- Python 3.12.11;
- PyTorch 2.13.0+cpu;
- Transformers 5.16.1;
- NumPy 2.3.5 (pinned below 2.4);
- safetensors 0.8.0;
- PyYAML 6.0.3;
- pytest 9.1.1;
- mypy 2.3.1 with `types-PyYAML` installed;
- Ruff 0.16.4.

The separate original training environment used by Phase C is Python 3.11.15,
PyTorch 2.12.1+cu126, Transformers 5.14.1, and CUDA 12.6 on an NVIDIA GeForce
RTX 4070 Laptop GPU. Keeping development qualification and model-runtime
environments explicit avoids pretending the CPU dev lock is the CUDA runtime.

### 2.2 Direct invariant coverage

| Required check | Direct evidence area |
|---:|---|
| 1 | Descriptor dimension, FP32 dtype, finiteness, determinism, scale normalization |
| 2-3 | Property-map shapes/nonnegativity and equal/different-shape projection |
| 4-7 | Exact integer TopK, deterministic fractional K, binary values, bitwise exact STE forward |
| 8 | Genuinely nonzero generator meta-gradient and STE gradient path |
| 9-11 | No self advice, same-role filtering, incompatible/empty peer fail-closed behavior |
| 12 | Positive peer scales cannot reverse selected task-gradient direction |
| 13 | Coordinate shuffling changes structural features and generated scores |
| 14-15 | Plan round trip preserves masks exactly and validates mask/content hashes |
| 16 | Missing calibration gradient fails closed |
| 17-18 | Only planned tensors require gradients; unplanned tensors/coordinates remain unchanged |
| 19-20 | Momentum and AdamW decay cannot move frozen coordinates |
| 21 | BF16/FP16 native parameters keep FP32 moments |
| 22 | Nonfinite/non-positive/out-of-range scales are rejected |
| 23 | `A=0.25` versus `A=1.0` gives the required 1:4 realized first Adam step |
| 24 | Resume verifies plan/support identity and keeps state sparse/restricted |
| 25 | Package imports in normal repository execution |
| Additional | Whole-model bitwise delta audit, `support_union_fraction`, data grouping/sealed policy, actual one-step controls, paired statistics |

Passing tests qualify the implementation; they do not substitute for a real
held-out experiment.

## 3. Synthetic mathematical feasibility

### 3.1 Scope and protocol

**SYNTHETIC MATHEMATICAL FEASIBILITY ONLY; NOT EVIDENCE OF REAL-MODEL OR
LONG-CONTEXT IMPROVEMENT.**

The controlled problem uses a `64 x 96` matrix, latent rank 6, six noisy peers,
noise standard deviation 0.18, 18 independent generator meta-training steps,
and one update per non-frozen arm. It was repeated over seeds
`11,17,23,29,31,37,41,43` and fractions `0.01,0.05,0.10,0.50,1.00`.

The latent structured matrix has low-rank, row, and column effects. Same-role
peers are independent noisy observations. The hidden sparse optimum is defined
from held-out peer structural maps; the observed task gradient is corrupted by
substantial independent coordinate noise. Meta-training uses independent
synthetic tasks rather than the held-out evaluation matrix.

The 400 raw records comprise ten arms x five budgets x eight seeds. The required
A-H set contributes 320 records (40 per arm); magnitude support and real PCSM
without the stabilizer are additional ablations.

Arm legend:

- **A** `no_update`
- **B** `random_support`
- **C** `target_only_support`
- **D** `gradient_topk`
- **E** `coordinate_shuffled_peers`
- **F** `real_pcsm_peers`
- **G** `dense_noisy_update` — unmatched dense reference
- **H** `oracle_support_update` — unmatched privileged oracle

Required sparse arms B-F have identical K and identical one-step parameter-space
update L2 for every paired seed/budget. G and H are explicitly unmatched and
are never presented as fair sparse controls.

### 3.2 Exact A-H mean final loss

Every cell has `n=8`. Lower is better. Fractions are synthetic support fractions
(1%, 5%, 10%, 50%, and 100%).

| Fraction | A | B | C | D | E | F | G | H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 5.521642862274236E-05 | 6.355143719143108E-05 | 6.318239752283352E-05 | 6.254361887749371E-05 | 6.300607427059191E-05 | 5.0834182028269216E-05 | 0.0008428603823069106 | 0 |
| 0.05 | 0.0002619446004582419 | 0.0002969914750862707 | 0.00029794547922738475 | 0.00029226533193776243 | 0.00029885958932772237 | 0.00023978576702486277 | 0.0009992369622806613 | 0 |
| 0.10 | 0.0005183523045394805 | 0.0005886993856398774 | 0.0005816243983080434 | 0.0005626523474994058 | 0.0005869464981109225 | 0.00047288853028412583 | 0.0011920721565059594 | 0 |
| 0.50 | 0.002540232007029258 | 0.0026190234242187355 | 0.0025907510104939786 | 0.002429962656455696 | 0.002628216196374019 | 0.002314792195175255 | 0.002703590310939476 | 0 |
| 1.00 | 0.005202711017058843 | 0.004694421627584369 | 0.004694421627584369 | 0.004694421627584369 | 0.004694421627584369 | 0.004694421627584369 | 0.004694421627584369 | 0 |

At full support, B-G select every coordinate and therefore tie by construction;
this tie is retained rather than discarded.

### 3.3 Exact A-H mean improvement

Improvement is `initial_loss - final_loss`; positive is beneficial.

| Fraction | A | B | C | D | E | F | G | H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | -8.335008568688721E-06 | -7.965968900091149E-06 | -7.327190254751346E-06 | -7.789645647849545E-06 | 4.382246594473145E-06 | -0.0007876439536841683 | 5.521642862274236E-05 |
| 0.05 | 0 | -3.504687462802885E-05 | -3.600087876914289E-05 | -3.0320731479520545E-05 | -3.6914988869480495E-05 | 2.215883343337911E-05 | -0.0007372923618224194 | 0.0002619446004582419 |
| 0.10 | 0 | -7.034708110039701E-05 | -6.3272093768563E-05 | -4.4300042959925527E-05 | -6.859419357144201E-05 | 4.546377425535459E-05 | -0.0006737198519664791 | 0.0005183523045394805 |
| 0.50 | 0 | -7.879141718947783E-05 | -5.051900346472076E-05 | 0.00011026935057356207 | -8.798418934476089E-05 | 0.00022543981185400326 | -0.0001633583039102177 | 0.002540232007029258 |
| 1.00 | 0 | 0.000508289389474474 | 0.000508289389474474 | 0.000508289389474474 | 0.000508289389474474 | 0.000508289389474474 | 0.000508289389474474 | 0.005202711017058843 |

### 3.4 Exact A-H mean oracle-support recall

| Fraction | A | B | C | D | E | F | G | H |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | 0.010080645161290322 | 0.026209677419354836 | 0.01814516129032258 | 0.010080645161290322 | 0.9717741935483871 | 1 | 1 |
| 0.05 | 0 | 0.06209415584415584 | 0.10186688311688312 | 0.0689935064935065 | 0.04667207792207792 | 0.9675324675324675 | 1 | 1 |
| 0.10 | 0 | 0.10040650406504065 | 0.16300813008130083 | 0.12784552845528455 | 0.09898373983739837 | 0.9737804878048781 | 1 | 1 |
| 0.50 | 0 | 0.5023600260416667 | 0.5498860677083333 | 0.5162353515625 | 0.501953125 | 0.9726155598958334 | 1 | 1 |
| 1.00 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

### 3.5 Pre-specified decisive checks

The summary aggregates paired means over all declared seeds and budgets,
including full-support ties:

| Check | Real | Control | Result |
|---|---:|---:|---|
| final loss: F < B random | 0.0015545444604193764 | 0.001652537469944137 | PASS |
| final loss: F < D gradient TopK | 0.0015545444604193764 | 0.0016083691164709452 | PASS |
| final loss: F < E shuffled peers | 0.0015545444604193764 | 0.0016542899971335252 | PASS |
| oracle recall: F > B random | 0.977140541756313 | 0.3349882662224307 | PASS |
| meta-training loss decreases | 0.21972321718931198 initial | 0.02366479649208486 final | PASS in all 8 seeds |

All declared support cardinalities were exact, all support unions equaled their
fixed support, and every unsupported coordinate remained bitwise exact.

The paired E-minus-F effect by budget is:

| Fraction | Mean paired loss difference | SD | 95% paired bootstrap CI | Wins | Exact sign-flip p |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 1.2171892242322692E-05 | 4.713217179001265E-06 | [9.242899148346155E-06, 1.5367503215647232E-05] | 8/8 | 0.0078125 |
| 0.05 | 5.90738223028596E-05 | 1.1873561565256468E-05 | [5.1368149768724624E-05, 6.663029562947876E-05] | 8/8 | 0.0078125 |
| 0.10 | 0.0001140579678267966 | 1.6252264096733353E-05 | [0.00010313603022075419, 0.00012380967901624856] | 8/8 | 0.0078125 |
| 0.50 | 0.00031342400119876415 | 3.462142828807556E-05 | [0.00029014034667996586, 0.000335232734556876] | 8/8 | 0.0078125 |
| 1.00 | 0 | 0 | [0, 0] | 0/8 (8 ties) | 1.0 |

### 3.6 Stabilizer ablation

Real PCSM without `A` recovered exactly the same mean oracle support as F at
each budget. Its final loss was slightly higher than F at the four sparse
budgets, but every paired 95% interval for
`no_stabilizer_loss - real_loss` included zero:

| Fraction | Mean difference | 95% CI |
|---:|---:|---:|
| 0.01 | 3.719306915061719E-07 | [-5.360404816786464E-08, 8.573701011020246E-07] |
| 0.05 | 9.807115826989407E-07 | [-8.697698820938111E-08, 1.9234596963396913E-06] |
| 0.10 | 9.964859711755617E-07 | [-4.402280269668181E-07, 2.250342827686616E-06] |
| 0.50 | 1.9796363633272566E-06 | [-6.933456543106362E-07, 4.814766489252195E-06] |
| 1.00 | 0 | [0, 0] |

The synthetic evidence therefore supports peer-dependent support recovery. It
does not independently establish a stabilizer-head benefit.

### 3.7 Raw evidence identities

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `results/synthetic/raw_per_seed.json` | 956,263 | `a4da3f4fcb91d94e809293ebe9df41f42e0a9eb84c0126b0a0113b7d2190e242` |
| `results/synthetic/raw_per_seed.csv` | 405,234 | `d27a32b6935a7b06862d28b8fdb3452efff7801f738d837ab18544c648073a9b` |
| `results/synthetic/summary.json` | 134,570 | `bd51dfa3eae77dd0b8924370e831740b8f0f5460c5a27b002e371200fc1e748c` |

The raw JSON asserts `all_required_arms_present: true` and labels itself
`SYNTHETIC_MATHEMATICAL_FEASIBILITY_ONLY`.

## 4. Real transformer Phase C

### 4.1 Frozen setup completed

Completed, machine-readable setup evidence includes:

- leakage-resistant grouped split manifest;
- outcome-blind sample manifest with content and token hashes;
- quality-blind resource preflight;
- authoritative `budget_manifest_v4.json`;
- fixed model/config hashes;
- five seeds, five budgets, eight core arms, and four held-out dev rows;
- explicit `sealed_test_read: false` in the frozen budget.

Resource preflight selected CUDA before model-quality execution:

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU;
- total GPU memory: 7.996 GiB;
- free GPU memory at preflight: 6.533 GiB;
- system memory: 15.218 GiB total, 1.997 GiB available;
- target: 3,145,728 coordinates;
- individual target buffers: 6 MiB BF16 weight, 12 MiB FP32 score,
  12 MiB FP32 stabilizer, 3 MiB boolean mask;
- explicit artifact note: dense backward GEMMs remain dense; no sparse-compute
  claim.

### 4.2 Calibration-gradient cache complete

Both FP32 calibration gradients completed under the authoritative v4 budget.
The cache summary is bound to budget file SHA-256
`bc31ae08fcf8a1235d83887afb25aaaead409d56329b92d690a4ee8e8d2b6dc7`,
records `native_weight_delta: bitwise_zero`, and records
`sealed_test_read: false`.

| Section | Device history | Tokens / supervised | Calibration loss | Gradient L2 | Elapsed seconds | Tensor SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `META_TRAIN` | resumed verified CUDA artifact | 2,743 / 6 | 0.463367223739624 | 13.649961471557617 | 25.33717850000005 | `9758a7f131713b2cbb5cd080eb46f6c2c7c4a962780aff4fdda12eb8befa0e06` |
| `MASK_CAL` | CPU | 2,686 / 4 | 2.6310513019561768 | 25.903955459594727 | 19.622757099999944 | `7f6e502f5d20fa641c98976e5e479ad0bd60813a76942c8483eedb6eaf4dd686` |

The completed summary file SHA-256 is
`6db534112a84c5bc9310bfda3cd7f9020494ed555707256d025a3d0eab3298e5`.
The cache stage took no optimizer step; these are selection inputs, not held-out
effectiveness results.

### 4.3 Resource pilot complete and valid

The pre-specified resource pilot used seed 17, fraction 0.001, exact `K=3,146`,
and one 2,755-token validation row with 9 supervised tokens. Every arm loaded a
fresh base and measured actual one-step held-out NLL on CPU.

| Arm | Exact K | Selected LR | Mean NLL | NLL minus frozen | Outside-support changes | Unplanned hash unchanged |
|---|---:|---:|---:|---:|---:|---|
| frozen / no update | 0 | 0 | 1.900450348854065 | 0 | n/a | yes |
| random sparse native | 3,146 | 0.0002 | 1.9086384773254395 | 0.0081881284713745 | 0 | yes |
| gradient TopK sparse native | 3,146 | 0.0002 | 1.9037142992019653 | 0.0032639503479004 | 0 | yes |
| coordinate-shuffled peer PCSM | 3,146 | 0.0002 | 1.8798811435699463 | -0.0205692052841187 | 0 | yes |
| real-peer PCSM calibrated | 3,146 | 0.0002 | 1.8544496297836304 | -0.0460007190704346 | 0 | yes |

All sparse plans had matched K, all delta audits were valid, and no unplanned
parameter hash changed. The decision artifact says:

```json
{
  "status": "VALID_RESOURCE_PILOT",
  "pilot_can_advance": false,
  "advancement_gate_evaluated": false,
  "advancement_gate_status": "not_applicable_to_pilot",
  "matched_support_counts": true,
  "all_delta_audits_valid": true,
  "sealed_test_read": false
}
```

Artifact file identities:

| Artifact | SHA-256 |
|---|---|
| `pilot_plans.json` | `b4e2ae817ee6c836af02d8690190a1568827eb26e91b737ab14143d20e2225e4` |
| `pilot_run.json` | `ed15ba571a19ab367d088224b2df0e7caaba33774cdaf11963de48fe52997eb7` |
| `PILOT_DECISION.json` | `81bf94f31415551fb70865b536688bfdd6cac7103bc71a5f8884ec575ae96459` |

#### Critical pilot mechanism audit

The lower real-peer pilot NLL is **not coworker evidence**:

- real and coordinate-shuffled supports intersect at 3,134 of 3,146 selected
  coordinates (99.618563% of real support); their union is 3,158 and Jaccard is
  0.9924002533248891;
- real and ordinary gradient-TopK supports intersect at 3,133 coordinates;
  their union is 3,159 and Jaccard is 0.9917695473251029;
- real and random supports intersect at only 3 coordinates;
- the real generator feature weight is 0.9999291896820068 on task-gradient
  magnitude and 0.00007084853859851137 on gradient-peer interaction; the other
  six feature weights are collectively negligible;
- the coordinate-shuffled generator is likewise 0.9969630837440491 task-gradient
  magnitude and 0.003036943031474948 gradient-peer interaction;
- every selected `A` value is exactly 1.0 in both real and shuffled plans.

Thus this one-seed, one-budget, one-dev-row pilot mostly recovered gradient
support, and the scale head was saturated identically. Its 0.0254315137863159
real-versus-shuffled NLL gap depends on a very small support difference and has
no uncertainty estimate. The pilot establishes resource feasibility and exact
audit behavior only.

### 4.4 Legacy partial plans are preserved but non-gating

The pre-hardening planner was stopped before any qualification outcome and its
partial file was renamed, not overwritten:

- path: `results/phase_c/qualification_plans_legacy_incomplete.json`;
- schema: 1.0;
- recorded status: `running` (incomplete historical state);
- records: 54 of the required 175 non-frozen plans;
- file bytes: 440,293;
- file SHA-256:
  `26a279e2c613baacf8e1ebdf7d51b87ece2c34f28f8ca9a6fc4743e334577f95`;
- seeds represented: 35 records for seed 11 and 19 for seed 17;
- budgets represented: 14 at 0.0001, 14 at 0.0005, 12 at 0.001,
  7 at 0.005, and 7 at 0.01;
- no held-out run records or qualification NLL outcomes.

Its arm counts are random 8, gradient TopK 8, target-only 8,
coordinate-shuffled 8, layer-shuffled 8, real bootstrap 7, and real calibrated
7. The frozen arm correctly has no plan.

The legacy file has no committed preregistration binding, schema-v2 provenance,
source-tree signature, access log, or result self-seal. It cannot be resumed by
the schema-v2 planner and cannot be supplied to the gate. Its individual plans
remain preserved solely as historical engineering evidence.

### 4.5 Schema-v2 qualification is running; confirmatory result is pending

The restart freezes a committed `results/phase_c/PREREGISTRATION.json` before
new planning. That root binds the exact model/manifests/caches/source tree,
five deterministic nonidentity layer-depth mappings, 175 ordered plan cells,
and 200 ordered run cells. New planner/runner artifacts must be schema 2.0 and
must carry the root hash, access logs, and generator/optimizer/data signatures.

A real-model preregistration smoke on the stable producer tree succeeded with
200 run cells, 175 plan cells, one pilot encoding, four qualification encodings,
four tokenizer artifacts, a sealed validation-only access log, and a valid
self-seal. Smoke artifact file/self hashes are intentionally not reported as
official identities. The two clearly marked post-commit hash placeholders live
in [EXPERIMENT_PROTOCOL.md](../EXPERIMENT_PROTOCOL.md) and will be filled from
the committed official artifact, not from a smoke file.

Already byte-determined identities are source-tree
`e899999d2f33dfc327b493f25bb24bcd594757bfc01c3eb6f46a5c4db5c94b9c`,
preregistration access log
`4578ca7b1e3792a6d446a655d1c220c6e2fd267c879fde1a88ff9f433bf0ced5`,
pilot encoded data
`aebb2d04037926dedf21f318dbb9ae9746f09758754f71858eb9583c8692449c`,
and qualification encoded data
`edaf37f53e648c2abbda6448ed5b679371a249ded51ce88eb9acdbc28d5d448a`.
The complete 33-file source and input/cache hash tables are in
[EXPERIMENT_PROTOCOL.md](../EXPERIMENT_PROTOCOL.md).

There is still no completed:

- complete schema-v2 175-cell qualification plan set or 200-cell one-step run;
- five-seed real-versus-shuffled paired effect;
- layer-identity permutation effect;
- full qualification delta-audit matrix;
- Phase-C advancement-gate artifact;
- confirmatory real-model long-context quality claim.

Accordingly the correct real-model conclusion remains **PENDING**, not positive,
negative, or inconclusive. No confirmatory metric is guessed from the pilot or
either partial plan generation.

## 5. Phase D decision

**STOP.** Full matched SFT has not run and must not run now.

`configs/phase_d.yaml` is disabled and requires a passed
`results/phase_c/ADVANCEMENT_GATE.json`. `scripts/train_phase_d.py` fails closed
without `verdict: PASS_ADVANCE_TO_PHASE_D` and `passed: true`; even after such a
pass it requires a separate reviewed implementation milestone.

Therefore there is no real SFT score, no LoRA/dense comparison, and no training
speed result to report.

## 6. Interpretation and next experiment

The strongest supported statement is:

> The implementation can learn and exploit peer-dependent structural
> information when such information genuinely determines a synthetic hidden
> sparse optimum.

The exact next experiment is the frozen real Phase-C protocol:

1. generate, review, and commit the schema-v2 `PREREGISTRATION.json` root of
   trust before new plans;
2. build exactly 175 schema-v2 plan cells and verify every record/artifact seal;
3. run exactly 200 logical cells; evaluate the deterministic frozen baseline
   once and clone it into its 25 pre-registered cells while loading a fresh base
   for every sparse cell;
4. verify every access log, signature, exact-support audit, and canonical
   unplanned-parameter fingerprint;
5. compute the pre-specified real-versus-coordinate-shuffle and layer-
   permutation paired analyses;
6. complete the registered causal-ablation evidence without cherry-picking;
7. preserve a `PASS_ADVANCE_TO_PHASE_D` or `STOP_NO_PHASE_D` gate artifact;
8. keep Phase D stopped unless every gate condition passes.

If real peers do not beat shuffled peers, preserve that negative result and
interpret any PCSM baseline gain as a generic structural prior or generator
capacity effect rather than coworker-specific information.
