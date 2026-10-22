# RDD-001 — Rineya distillation dynamics

**Implemented and CPU-tested experiment software. Native Rineya compatibility,
teacher qualification, training, and efficacy are NOT established by this delivery.**

2026-10-09: [Reviewed publication handoff](REVIEW_HANDOFF_20261009.md) preserves
the necessary historical records, resolves causal-input and parity-asset drift
checks, and records 179 guarded offline passes (11 update tests excluded). GPU
and qualified measurement inputs are deferred; no native action or model-quality
gain is established. Historical inventories and permission flags stay unchanged.

2026-10-05: offline preparation adds a selected A/B/D/SFT design at one candidate
LR/seed, two steps and per-step checkpoints; a qualification-only supervisor mode;
complete consumed-prompt provenance; and prospective original standard-PEFT saves.
The historical full/forward matrices remain available but are outside this scope.
The current exact-asset and host audit is retained under
`C:/Users/clclc/BusinessProject/rdd001-records/preparation-20261005/`.
No model loads, optimizer updates or confirmation reads were performed during
this preparation. Native E0, teacher competence and HF/PEFT parity remain blocked.

Dedicated branch: `research/rineya-distillation-dynamics-v1`.
Parent: `post-training-Mimo` at `03db218ceebc61a2bc04f2482cf7e094045e01c1`.
All additions live in this directory. No existing code, branch, weights, bank,
production inference, scheduler, or release policy is changed.

## Question

For one exact **0.6B-class Rineya** checkpoint, which combination of rollout
source, token-level KL direction, and learning rate gives the best
acquisition–retention trade-off? Does it generalize to harder tasks and improve
useful sampling coverage? A smaller parameter update alone is not a success.

Inspired by Piskorz, Berthon and van der Schaar, *On-Policy or Off-Policy Learning?
A Systematic Study of Distillation Dynamics*, arXiv:2609.35259v1, 2026-09-28.
This is an original adapter-based Rineya adaptation, not a reproduction of their
full-parameter Llama/Qwen2.5 experiments. See `SOURCES.md` and `PROTOCOL.md`.

## Delivered

| Component | What actually runs |
|---|---|
| Factorial | A: teacher/FKL; B: student/FKL; C: teacher/RKL; D: student/RKL; each at two learning rates and paired seeds. Verified-solution SFT at both rates is separate. |
| Objective | Exact full-vocabulary forward and reverse KL, response-only causal positions, detached teacher and sampled trajectories, distinct teacher/student widths and frozen output heads. First-order chunked backward recomputes probabilities. |
| Rollouts | Immutable per-seed teacher bank shared between teacher arms; fresh current-student rollouts per optimizer step. No success filtering; capped rollouts remain in training and are counted. |
| Trainer | Actual PyTorch AdamW updates on fixed-scope q/v LoRA; fixed independent retention replay; identical task schedules, source initialization and optimizer settings except registered treatment factors. |
| Evidence | Exclusive output creation, per-step trace provenance, frozen model hashes, adapter checkpoints, cumulative token accounting, source-specific paired retention and DEV checkpoint selection across all registered seeds. |
| Confirmation | Separate catalog digest committed before training; DEV selection locked first; confirmation access is single-use even when interrupted. No automatic promotion. |
| Native candidate | Explicit Windows-supervised local HF Qwen3 export path, existing E0 receipts plus reviewed export parity, exact parameter/tokenizer checks, disposable runtime KL/backward/save-reload qualification, teacher competence screen, and shared GPU lock. **Not run on native assets.** |
| Integration | Standard-PEFT format export with provenance for later GD-001 ingestion. Actual PEFT/native parity remains a separate qualification; no automatic GD import or branch merge. |

The default three-seed full campaign has **24 distillation runs + 6 SFT controls**.
The forward-only screen has **12 distillation runs + 6 SFT controls**. Unchanged
source predictions are recorded before every arm. The example learning rates are
engineering placeholders requiring pilot review, not recommendations imported
from the paper.

## Safe commands now

From the repository root, using a CPU test environment with PyTorch, safetensors
and pytest (see `requirements-cpu.txt`):

```bash
python -m pytest experiments/rineya_distillation_dynamics_v1/tests -q
python -m experiments.rineya_distillation_dynamics_v1 plan \
  --plan experiments/rineya_distillation_dynamics_v1/configs/pilot.json --stage screen
python -m experiments.rineya_distillation_dynamics_v1 demo --output /tmp/rdd001-demo
```

Use a new demo output path on every invocation. The demo uses tiny, random,
non-pretrained models (including an unqualified random teacher) and a byte
tokenizer. It exercises real tensor losses/optimizer updates, not Rineya learning.
Its scores must never be cited as native evidence. Its zero-gain result correctly
leaves no eligible configuration rather than manufacturing a winner.

## Native boundary

Do not run native commands while the laptop is unavailable or being serviced.
Code/push authorization is not a GPU, paid-API, or training authorization. Native
execution requires explicit `--allow-model-load`, `--allow-training`, and
`--ack-host-available`, as well as all existing qualification evidence.

The new loader supports **only reviewed standard local Qwen3ForCausalLM
safetensors exports** with matching tokenizer/vocabulary/template. It does not
convert original custom Rineya artifacts, load pickled weights, quantize models,
substitute stock Qwen, download models, call teacher APIs, or lower existing
resource floors. The 344M target is excluded.

The external E0 and export-review receipts are prerequisites, not created by this
package. Attestations and file hashes do not prove provenance against a malicious
operator. Native execution also performs its own disposable checks and restores
the common source before any treatment.

A separate larger teacher may exceed the inherited 4 GiB process-tree RSS ceiling.
CPU offload reduces simultaneous device residency, **not total host-memory
requirements**, and can be very slow. Such a run must remain blocked; do not relax
limits or quietly replace the teacher. Remote/quantized teacher providers would
be a separately authorized extension.

See `RUNBOOK.md` for enrollment and commands, `PROTOCOL.md` for all estimands and
limits, `RECONNAISSANCE.md` for prior-work boundaries, and `VALIDATION.json` for
measured delivery evidence. Interrupted runs are not resumed automatically.
