# IDR-001 — Quality-Constrained Skill Routing with ID Balancing

**Status: implemented and CPU-tested; native Rineya/Qwen3/CUDA efficacy is NOT qualified.**

Dedicated research branch: `research/rineya-id-balancing-skill-routing-v1` in
`LUNCY579/ChatBubble`. Parent: freshly inspected `main`, commit
`b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2`. All additions are confined to this directory.
No registered world-model record, baseline, game, or other branch is changed.

The experiment tests whether useful, qualified alternatives are neglected by a
router. It does not assume uniform usage is desirable or that load control teaches
missing rules. Keep the current world-model pipeline and competence work intact.

## Implemented paths

| Path | Implementation | Evidence boundary |
|---|---|---|
| Paper controller | Magnitude-integral + worsening-gated derivative, centered bias; sign, integral-only, none, exact-quantile reference | Equation/unit tests and synthetic score streams; not the paper's large-scale reproduction |
| Frozen specialist routing | One operation per pool, qualification/applicability/quality masks, bounded correction, nonuniform targets, preserved strength, whole-call adapter execution | Offline replay and mocked execution tests; native backend requires local artifacts |
| Trainable adapters | Shared frozen FP32 projection, four LoRA experts, Top-2, sparse token dispatch, unmodified mixture scores, auxiliary-loss baseline | Actual CPU optimizer tests and three-seed tiny regression matrix; not LLM training |
| Data and diagnosis | Counterfactual capture, train-only quality head, label-independent predictions, fresh ledger challenge generator, self-rollout helper | Generator labels and mocks are explicitly not a learned world model |
| Evaluation | Parent-clustered paired intervals, family/stratum protection, calibrated-state freeze, overlap checks, write-once confirmation claims | No automatic qualification or model promotion |

## Quick start

From the repository root:

```bash
cd experiments/rineya_id_balancing_skill_routing_v1
python -m pip install -r requirements.txt
python -m pytest -q
python -m idr demo --out runs/cpu-demo-001
python -m idr verify --freeze validation/source_freeze.json
```

The demo executes eight frozen-router arms (four controllers × two target policies),
five controller-reference traces, and fifteen tiny optimizer runs (five methods ×
three seeds). The tiny task is constructed regression, not a pretrained transformer.
`runs/` is ignored; use a new output name for each run.

The default gains are pilot starting values in the code's score units, **not**
transferred large-model hyperparameters. Compare methods under equal tuning budgets.
See [protocol](docs/PROTOCOL.md), [native runbook](docs/RUNBOOK.md),
[contracts](docs/CONTRACTS.md), and [source notes](docs/SOURCES.md).

## Native execution

There are local-only commands for `preflight`, `capture-native`, `execute-native`,
and `train-native`. Stage A loads registered ordinary LoRA adapters without merging.
Stage B trains **fresh** adapter banks on explicitly named dense `down_proj` modules.
The optional native loader accepts a small dense Qwen3 safetensors export in FP32;
it does not pretend to support every custom Rineya checkpoint layout.

```bash
python -m idr preflight --config configs/native.template.json --out runs/preflight-001.json
```

The template intentionally stops. It has no model paths, hashes, qualified experts,
or native authorization. Supply real local custody information and inspect parity,
retention, memory, and OOD behavior before running the native protocol.

`moe.inject(model, paths, ...)` also accepts an already-loaded in-memory dense FP32
model from an existing qualified loader; use that seam instead of rewriting a loader.
No module is imported from an unavailable sibling research branch.

## Important distinctions

Bias changes which expert is selected; it does not change registered skill strength.
Forward-model routing is based on input and predicted reliability, never desirable
consequences. A proposer and a forward predictor must not compete in one pool.

A zero-target or unqualified specialist is not activated to meet a quota. If only
one safe alternative remains, the target is one and balancing has no useful effect.
The quality guard limits **predicted** quality loss, not actual error from a bad
quality estimator. Oracle outcomes are restricted to training-label capture and
analysis; they are not deployed as a router.

Replay prices cannot establish end-to-end speedups. All-expert data collection is
charged separately. Registered scalar objectives, per-stratum protection, input
hashes, and independent training seeds are required before claiming an improvement.

## Validation records

`validation/STATUS.json` and `validation/cpu_matrix.json` record what actually ran.
`validation/native_preflight.json` records why native execution did not run here.
All native qualification fields remain false. The committed source freeze excludes
validation outputs to avoid circular self-hashing.
