# Peer-Conditioned Stabilizer Matrix (PCSM)

PCSM is an experimental, training-time native-weight method for asking one
specific question:

> Can compatible transformer layers provide useful information about **where**
> a target layer should remain plastic and **how strongly** selected coordinates
> should learn, while the task gradient remains responsible for update direction?

The “coworker” language is an intuition, not a biological or causal claim.
Peers advise the plasticity plan; they never copy weights into the target and
never synthesize an independent signed update.

## Evidence status

- Experimental branch: `codex/exp-peer-stabilizer-matrix`
- Clean base commit: `b97cd8c636d1b1eebb9a450f369d789532f0dd4e`
- Model scoped for real Phase C: Rineya Nimbus, a 28-layer Qwen3 causal LM
  (`hidden_size=1024`, `intermediate_size=3072`, BF16, 40,960-token maximum
  context)
- Initial real target: `model.layers.14.mlp.down_proj.weight`, shape
  `1024 x 3072` (3,145,728 native coordinates)
- Software qualification: a 98-test qualification milestone passed; after
  direct gap and schema-v2 adversarial gate tests landed, the current suite
  passes **142/142 tests** in the pinned development environment
- Synthetic feasibility: complete over 8 seeds, 5 budgets, and 8 required A-H
  arms; every declared decisive synthetic check passed
- Real one-step Phase C: **RUNNING/PENDING**. The one-seed resource pilot is
  complete and valid, but cannot advance; a 54-plan schema-v1 partial artifact
  is preserved as legacy/non-gating while schema-v2 qualification is hardened,
  and no confirmatory result or gate verdict exists
- Full SFT Phase D: **STOP** unless and until the pre-specified Phase-C gate
  produces `PASS_ADVANCE_TO_PHASE_D`

See [results/RESULTS.md](results/RESULTS.md) for exact evidence and claim
boundaries. Synthetic success is not evidence of real-model or long-context
improvement. Although the resource pilot's real-peer NLL was lower than its
shuffled control, their supports overlapped 99.6185%, the real score head was
99.9929% gradient magnitude, and selected scales were all exactly 1.0. The
pilot is therefore not interpreted as coworker evidence.

## Design at a glance

For a target matrix \(W_t\), PCSM freezes two tensors before training:

\[
M_t \in \{0,1\}^{d_{out}\times d_{in}}, \qquad
A_t \in [a_{min},a_{max}]^{d_{out}\times d_{in}}.
\]

`M` is an exact TopK support: where learning is allowed. `A` is a bounded,
positive post-Adam scale: how strongly each selected coordinate learns. In the
custom optimizer, the selected-coordinate update is

\[
W_i \leftarrow W_i - \eta A_i\left(
\frac{\hat m_i}{\sqrt{\hat v_i}+\epsilon}+\lambda W_i
\right), \qquad i\in M.
\]

The gradient, moments, normalized direction, weight decay, and final write are
all restricted to the fixed support. `A` scales the realized AdamW direction
*after* moment normalization, including selected-coordinate decay. Unsupported
coordinates are never assigned by the optimizer.

The planning stack is:

1. discover architecture roles from the module tree;
2. admit only other matrices with the same semantic role;
3. summarize each matrix with a cheap 16-32 value FP32 descriptor (24 by
   default);
4. compute auditable cosine-similarity attention plus a conservatively
   initialized learned residual;
5. aggregate element, row, column, and joint structural property maps;
6. combine eight explicit target/gradient/peer coordinate features;
7. freeze one exact, stable TopK mask and one bounded scale matrix;
8. persist the plan, reload the byte-identified base, apply one native update,
   and bitwise-audit every coordinate outside support.

No cross-layer weight averaging occurs. The self-attention coefficient is
exactly zero because the target is removed from its peer set before softmax.
Missing same-role peers fail closed.

## Repository layout

| Path | Responsibility |
|---|---|
| `src/rineya_pcsm/architecture.py` | Architecture-derived roles and same-role peer discovery |
| `src/rineya_pcsm/descriptors.py` | Scale-normalized FP32 descriptors |
| `src/rineya_pcsm/attention.py` | Transparent peer attention and persisted diagnostics |
| `src/rineya_pcsm/property_maps.py` | Element/row/column/joint structural maps and projection |
| `src/rineya_pcsm/generator.py` | Eight-feature score head and bounded stabilizer head |
| `src/rineya_pcsm/topk.py` | Stable exact TopK and custom exact-forward STE |
| `src/rineya_pcsm/plan.py`, `persistence.py` | Immutable, content-addressed JSON + safetensors plans |
| `src/rineya_pcsm/application.py` | Parameter binding, freeze flags, and coordinate gradient guards |
| `src/rineya_pcsm/optimizer.py` | Exact-support AdamW with sparse FP32 moments |
| `src/rineya_pcsm/delta_audit.py` | Bitwise whole-parameter before/after audit |
| `src/rineya_pcsm/data.py` | Grouped split construction, sealed-data policy, fail-closed tokenization |
| `src/rineya_pcsm/real_plans.py`, `phase_c.py` | Real-model plan generation and one-step primitives |
| `src/rineya_pcsm/phase_c_runtime.py` | Schema-stable resume, baseline reuse, and canonical unplanned-state evidence |
| `scripts/freeze_phase_c_preregistration.py` | Immutable schema-v2 root-of-trust generator |
| `scripts/` | Split, sample, budget, cache, preregistration, plan, one-step, gate, and Phase-D guard entry points |
| `tests/` | Qualification and regression tests |
| `results/synthetic/` | Raw synthetic JSON/CSV and statistical summary |
| `results/phase_c/` | Frozen real-model setup artifacts and live status |

The full equations and invariants are in [THEORY.md](THEORY.md). The frozen
experimental decisions are in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Selection and training are separate

The required lifecycle is deliberately transactional:

1. Load and hash the base checkpoint.
2. Run `META_TRAIN` and `MASK_CAL` forward/backward passes without an optimizer
   step.
3. Generate, review, and commit `results/phase_c/PREREGISTRATION.json` before
   qualification planning. This schema-v2 root binds the model, manifests,
   caches, all tokenizer artifacts, encoded dev token/label hashes, source tree,
   200 run cells, 175 plan cells, and five nonidentity layer-depth mappings.
4. Generate the peer-conditioned support and scale plans from that exact root.
5. Save plans atomically; existing destinations are verified rather than
   overwritten.
6. Discard calibration model state.
7. Reload the byte-identical base and verify model, config, tokenizer, sample,
   split, gradient summary/cache, preregistration, plan, support, source, and
   tensor hashes.
8. Bind only planned parameters, perform the declared update, and evaluate
   held-out `ONE_STEP_DEV` examples.
9. Reject the run if any undeclared native coordinate changes by even one bit.

For the first real experiment, support is generated once and frozen. Therefore
the reported support union equals the declared K. Dynamic masks are outside the
current protocol.

## Windows / PowerShell quick start

The lab-local virtual environment is the pinned development/qualification
environment. Real CUDA calibration and one-step execution deliberately use the
original repository's shared training environment, whose CUDA stack is already
qualified for the model. The reproduction driver selects the appropriate one.

```powershell
$OriginalRepo = 'C:\Users\clclc\BusinessProject\Chat-Bubble'
$Lab = 'C:\Users\clclc\BusinessProject\Chat-Bubble-pcsm\rineya_pcsm_native_weight_lab_v1'
$DevPython = Join-Path $Lab '.venv\Scripts\python.exe'
$RuntimePython = Join-Path $OriginalRepo '.venv\Scripts\python.exe'

$env:PCSM_BASE_MODEL = Join-Path $OriginalRepo 'Rineya_Published_Cores\Rineya-Nimbus-1.6.0-devfreeze'
$env:PCSM_CORPUS_ROOT = Join-Path $OriginalRepo 'rineya_long_context_cc0_v1_main'
$env:PYTHONPATH = Join-Path $Lab 'src'
Set-Location $Lab
```

Adjust `PCSM_CORPUS_ROOT` if the generated corpus lives elsewhere. It must
contain `reports/main_lineage/{train,validation}.jsonl` and
`data/sft/{train,validation}/*.jsonl`. The Phase-C split builder intentionally
does not open test data.

Run qualification:

```powershell
& $DevPython -m pytest -q
& $DevPython -m ruff check src tests scripts
& $DevPython -m mypy --strict src\rineya_pcsm
```

Reproduce synthetic feasibility into a separate directory (do not overwrite
the checked-in evidence):

```powershell
& $DevPython scripts\run_synthetic.py `
  --config configs\synthetic.yaml `
  --output-dir results\reproduction\synthetic `
  --bootstrap-resamples 10000 `
  --statistics-seed 20260826
```

For the official Phase-C evidence tree, freeze the root of trust only after the
gradient summary is complete and before any schema-v2 plan:

```powershell
& $RuntimePython scripts\freeze_phase_c_preregistration.py `
  --budget results\phase_c\budget_manifest_v4.json `
  --sample-manifest results\phase_c\sample_manifest.json `
  --split-manifest results\phase_c\split_manifest.json `
  --gradient-summary results\phase_c\gradient_cache_summary.json `
  --gradient-cache cache\phase_c\gradients `
  --model $env:PCSM_BASE_MODEL `
  --corpus-root $env:PCSM_CORPUS_ROOT `
  --maximum-tokens 40960 `
  --output results\phase_c\PREREGISTRATION.json
```

Review the generated hash tables and then commit that exact file. The gate
finds the last commit that changed the preregistration, proves that commit is an
ancestor of the current `HEAD`, and proves the working bytes equal that committed
blob. Later code/docs commits therefore do not invalidate an unchanged root;
merely generating an uncommitted file is still insufficient.

The PowerShell driver exposes each fail-closed stage and keeps outputs under an
explicit run root:

```powershell
$RunRoot = Join-Path $Lab 'results\reproduction\phase-c-001'

& .\scripts\reproduce\Invoke-PCSMReproduction.ps1 `
  -Stage PhaseCPrepare -RunRoot $RunRoot

& .\scripts\reproduce\Invoke-PCSMReproduction.ps1 `
  -Stage PhaseCCache -RunRoot $RunRoot -Device cuda

& .\scripts\reproduce\Invoke-PCSMReproduction.ps1 `
  -Stage PhaseCPreregister -RunRoot $RunRoot
```

For an official qualification, inspect and commit the generated
`PREREGISTRATION.json` before continuing. The planner now requires
`--preregistration`; the runner requires `--preregistration`,
`--split-manifest`, and `--gradient-summary` in addition to its prior inputs.
The driver supplies all of them explicitly.

```powershell
& .\scripts\reproduce\Invoke-PCSMReproduction.ps1 `
  -Stage PhaseCPilot -RunRoot $RunRoot -Device cuda
```

Only a valid resource pilot permits the expensive qualification matrix:

```powershell
& .\scripts\reproduce\Invoke-PCSMReproduction.ps1 `
  -Stage PhaseCQualification -RunRoot $RunRoot -Device cuda

& .\scripts\reproduce\Invoke-PCSMReproduction.ps1 `
  -Stage PhaseCGate -RunRoot $RunRoot
```

`PhaseDGuard` is intentionally non-runnable without a passed gate and still
requires a separate reviewed Phase-D implementation milestone after a pass. It
accepts the full run/plan/preregistration/manifest/cache chain and re-audits it;
an isolated gate JSON is never sufficient authorization.

## Real Phase-C scope

The confirmatory one-step matrix uses:

- seeds `11, 17, 23, 29, 31`;
- support fractions `0.0001, 0.0005, 0.001, 0.005, 0.01`
  (0.01%, 0.05%, 0.10%, 0.50%, 1.00%);
- a fixed `mlp.down_proj` target and all other `mlp.down_proj` layers as eligible
  peers;
- exact-K random, gradient TopK, target-only, coordinate-shuffled,
  layer-shuffled, real-peer bootstrap, and real-peer calibrated controls, plus
  the frozen no-update reference;
- the same checkpoint, examples, K, optimizer, LR grid, tokenization, and
  evaluation budget for each paired cell;
- actual held-out NLL after one native update, not score correlation alone.

The frozen Cartesian grid has exactly `5 seeds x 5 budgets x 8 arms = 200`
logical run cells. The frozen arm has no plan, so the exact plan grid is
`5 x 5 x 7 = 175` cells. The runner evaluates one deterministic no-update
baseline once and clones that self-sealed evidence into the 25 logical baseline
cells; every sparse arm still reloads its own byte-verified base. Reuse changes
compute cost only, never the pre-registered output schema or statistical grid.

The decisive effect is
`coordinate_shuffled_loss - real_peer_loss`; positive values favor real peers.
Advancement requires real peers to win at least 4/5 seeds, a paired 95%
interval strictly above zero, survival of the layer-identity permutation test,
matched support/capacity/compute, valid bitwise delta audits, and proven sealed
non-access. A pilot can never pass this gate.

## Boundaries and limitations

- Current positive evidence is synthetic mathematical feasibility only.
- The real experiment studies one target matrix family and initially one target
  layer; it cannot establish a universal transformer property.
- Structural projection is a deterministic coarse map, not a claim of neuron
  identity across layers.
- Attention weights are audit data, not causal explanations; causal ablations
  are required.
- A one-step gain may disappear during full SFT, and adaptation may harm
  retention.
- Exact sparse optimizer state and compact deltas can save state/storage, but
  the model still executes dense forward/backward kernels. **No speed claim is
  made.** Wall-clock, throughput, and VRAM must be measured.
- The full structural-map implementation currently materializes dense FP32
  feature maps during planning. It is resource-bounded, not sparse-kernel
  acceleration.
- No novelty claim is made without a dedicated literature review.
