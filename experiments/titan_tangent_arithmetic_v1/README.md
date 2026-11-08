# TTA-001: Titan tangent-space task arithmetic

**Method port implemented; CPU mechanics tested; pretrained execution NOT RUN.**
New, isolated code. No inherited model, reporter, checkpoint, registration or result is modified.
Parent: `21c7602ccad59ec7b69e1ae296d06612fcffa90b` on ChatBubble's delta research line.

## The experiment

Compare three genuinely different predictors at the **same frozen anchor**:

1. **Ordinary:** train independent task deltas A/B through `f(theta0 + delta)`.
2. **Post-hoc:** reuse those exact trained deltas through `f(theta0) + J(theta0) delta`.
3. **Tangent-trained:** train new independent A/B deltas directly through the latter predictor.

`torch.func.jvp` computes the tangent response without materializing a Jacobian. Gradients
through that response train the delta; the anchor never moves. This is not small-learning-rate
ordinary FT, orthogonalization, low-rank fitting, or merging a tangent-trained delta back into
the ordinary nonlinear model. Those are different experiments.

The implementation follows Eq. (3), Eq. (6), Eq. (7), Section 5 and Appendix B of
Ortiz-Jimenez, Favero and Frossard, *Task Arithmetic in the Tangent Space: Improved Editing of
Pre-Trained Models*, NeurIPS 2023, arXiv:2305.12827v3.
Official reference implementation: `gortizji/tangent_task_arithmetic/src/linearize.py`, observed
blob `74e4db02114a5665db455fb10c68e935139ab166`. This package is independently written.

**This is a causal-language-model method port, NOT a replication of the paper's CLIP benchmark
numbers.** The first scorer is teacher-forced response-sequence exact match, not free generation,
not canonical JSON semantic scoring, and not the historical SELECT-VIEWS-001 instrument.
The loss is response-plus-EOS causal cross-entropy. See `EXPERIMENT.md` for explicit deviations.

## What runs now

From the repository root, with the tested CPU PyTorch environment:

```bash
python -m pytest -q experiments/titan_tangent_arithmetic_v1/tests
python -m experiments.titan_tangent_arithmetic_v1.smoke --out /tmp/tta-smoke-001.json
```

The output path must be new. The smoke uses a tiny synthetic causal module, **not pretrained
weights**, and executes four 12-update fixture training arms. Its numbers must never be
reported as Titan results. It deliberately reports zero single-task denominators as undefined,
not as zero normalized accuracy or success. The frozen-anchor finite-difference and backward
checks passing do not establish task competence.

## Source layout

| File | Purpose |
|---|---|
| `tangent.py` | Frozen, unique-tied-parameter anchor; direct deltas; exact JVP; logical-call budget |
| `study.py` | Matched training, independently tuned 21-point alpha grid, addition, negation, Eq. (6) surface |
| `controls.py` | Deterministic top-k, head/tail, norm-matched dense, SVD, effective BF16 export, exact attenuation controls |
| `diagnostics.py` | Zero identity, finite differences, reverse-through-JVP check, fixed-prefix historical-update diagnostics |
| `data.py` | Explicit token IDs/response labels; no hidden rendering/truncation; parent/content split checks |
| `metrics.py` | Absolute/normalized scores, sum of task-conditioned disagreement, paired gains/losses, output remainders |
| `binding.py`, `host.py` | Local-only exact-source/file/runtime binding and a one-session CPU subprocess supervisor |
| `tests/` | Positive and adversarial CPU mechanics tests; no pretrained assets |

## Native integration

The host adapter accepts local **dense Qwen3-format** Supa 1.7, explicit FP32 Titan anchors,
Nimbus 1.6 and Halo 1.2 bindings. It does not imply those models have passed qualification.
Supa-A1's custom logit bias, routed systems, NF4 and BF16 arithmetic are excluded. The first
adapter is CPU/eager/no-cache; CUDA is not requested and no other study's GPU lease is touched.

Read `HOST_HANDOFF.md`. First run a separately bound diagnostic probe through `qualify`, then
use its successful, hashed receipt for `diagnostic` or `study`. The user has requested this
experiment; the remaining requirements are technical source/data/runtime/resource bindings,
not another blanket request for approval. No automatic retries or fallback precision changes.

Existing historical deltas support **post-hoc diagnostics**, not a substitute for independently
training the tangent arm. All new native outputs and saved delta safetensors live in fresh
external session directories. Published model files remain unchanged.
