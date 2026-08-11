# v2.3.1 — Grokking Dynamics Observatory

## Purpose

v2.3.1 is a **training-dynamics research release**, not a new production-candidate release. It is built directly on the revised v2.3 repository and the completed v2.3 residual-capability experiment.

The motivating observation is that the v2.3 reasoning residual showed a threshold-like capability-dose curve: low/intermediate LoRA doses transferred almost no answerable reasoning, while the full residual dose produced measurable DEV and structural-OOD accuracy. That pattern is compatible with several very different mechanisms:

1. **Readout/logit threshold:** internal improvement is smooth, but exact accuracy changes only after the correct token crosses a competing logit.
2. **Ordinary feature learning:** train and OOD improve together over time.
3. **Memorization without abstraction:** train performance improves but structural OOD remains flat.
4. **Grokking-like delayed generalization:** train fitting occurs first; OOD stays low through a prolonged plateau; continued optimization on the same information later produces a sharp OOD rise.
5. **Latent-representation progress + delayed access:** representation geometry improves before discrete answer accuracy exposes it.

v2.3.1 is designed to distinguish these explanations rather than presuppose that grokking exists.

## Important terminology

A **capability-dose threshold is not grokking**. The repository only uses the label `GROKKING_LIKE_TRANSITION_DETECTED` when temporal evidence meets the preregistered delayed-generalization criteria. `DELAYED_GENERALIZATION_REPRODUCED` additionally requires the configured number of independent seed controls.

These are research diagnostics, not release gates.

## Immutable anchors and data policy

- Optimization anchor: v2.2 `greedy_soup`.
- Source residual/preservation data: the completed v2.3 run.
- Consumed v2.2 sealed data: **forbidden**.
- Fresh v2.3 sealed data: **forbidden**.
- v2.3.1 uses only v2.3 residual/preservation pools plus packaged DEV/OOD-DEV splits.

The validator and individual evaluators fail if a sealed path is passed into the observatory.

## Main experiment

The core run keeps the exact v2.3 reasoning training mix, LoRA rank 16, LR `6e-6`, KL anchor `0.30`, and the same greedy anchor, but trains for a much longer horizon with frequent lightweight adapter snapshots.

Three profiles are available:

| Profile | Core steps | Adapter snapshot cadence | Factor-screen steps | Intended use |
|---|---:|---:|---:|---|
| `pilot` | 800 | 50 | 300 | plumbing / quick dynamics check |
| `standard` | 2400 | 100 | 600 | default research run |
| `extended` | 4800 | 100 | 1200 | long-horizon follow-up |

Trainer optimizer checkpoints retain only the most recent two copies for resumability. Lightweight PEFT snapshots are retained at every save event in `dynamics_snapshots/`, so the temporal experiment does not require dozens of full optimizer states.

## What is measured at every retained checkpoint

### Discrete behavior

- residual TRAIN answerable correctness;
- DEV answerable correctness;
- structural-OOD answerable correctness;
- hard answerable correctness;
- abstention correctness;
- answer coverage;
- false-abstention rate.

### Continuous readout metrics

- mean target-token NLL;
- mean gold-token probability;
- gold-token rank;
- gold-vs-best-wrong logit margin;
- first-target NLL;
- first-target rank;
- first-target probability;
- first-target entropy.

A sharp exact-accuracy transition accompanied by smooth logit-margin/NLL motion is formalized as `smooth_logits_with_sharp_accuracy` on the time curve and contributes to readout evidence. It is not by itself sufficient for a grokking label.

## Step × capability-dose surface

After the temporal curve is measured, v2.3.1 selects milestone checkpoints plus checkpoints around the largest observed OOD changes. Each is evaluated at:

`0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.2`

This produces a two-dimensional surface:

`training step × residual dose → TRAIN / DEV / OOD behavior + continuous logits`

The surface is reduced to a `critical_dose(t)` trajectory: at each retained step, the lowest dose whose OOD answerable accuracy reaches a preregistered fraction of that step’s max-dose OOD accuracy (subject to an absolute floor). A leftward-moving critical dose is qualitatively different from a fixed high interpolation threshold; both statistics are first-class detector outputs and appear in `DOSE_TIME_SURFACE.md` and the scientific report.

## Optimizer / gradient dynamics

`train_student.py` now has opt-in dynamics telemetry. Normal Phase 9 runs remain unchanged unless the new arguments are supplied.

At the configured optimizer cadence, the logger records:

- trainable-parameter norm;
- LoRA-A norm;
- LoRA-B norm;
- gradient norm;
- gradient cosine with the previous sampled gradient;
- gradient cosine with an EMA gradient;
- EMA gradient norm;
- fast residual gradient norm;
- slow/total gradient-norm ratio.

Only trainable tensors are copied to CPU for the direction history, keeping the official path practical for LoRA.

## Adapter-space dynamics

For every saved adapter snapshot, v2.3.1 measures:

- LoRA-A Frobenius norm;
- LoRA-B Frobenius norm;
- effective dense `B @ A` delta norm;
- delta effective rank;
- largest non-zero singular value;
- per-layer delta norm;
- optional delta norm relative to the greedy anchor's matched target weights.

The non-zero singular spectrum is calculated from an `r × r` QR core, avoiding full dense delta materialization for the rank analysis.

## Representation dynamics

Selected temporal checkpoints are probed at depth-normalized layer positions around 0%, 25%, 50%, 75%, and 100% of the transformer stack.

On answerable TRAIN-format and structural-OOD-format prompts, the observatory records:

- cross-format nearest-centroid classification of the underlying reasoning family;
- within-family minus between-family cosine-separation gap;
- linear CKA to the frozen greedy anchor;
- activation effective rank.

The representation probe is intentionally format-crossing. If email/table/etc. examples and JSON/log/etc. examples become organized by the same underlying reasoning operation before exact OOD answers improve, that is evidence of latent abstraction progress.

## One-factor phase diagram

v2.3.1 does **not** launch an uninterpretable Cartesian hyperparameter grid. Instead, every run changes one factor from the same anchored baseline.

The standard screen covers:

- training/data quantity;
- unique residual cases versus repeated exposure;
- residual/preservation ratio;
- weight decay;
- KL-anchor strength;
- learning rate;
- LoRA rank;
- random seed;
- attention-only vs MLP-only vs both target modules;
- all layers vs middle 20% vs middle 40% LoRA placement.

The extended profile additionally screens:

- gradient accumulation / effective batch behavior;
- warmup ratio;
- anchor temperature.

### Exposure-controlled dataset variants

The variant builder distinguishes **unique examples** from **number of exposures**. For the exposure-control axis, residual exposure count is held approximately fixed while the number of unique residual rows changes. This is designed to test whether a transition is governed more by structural coverage or repeated optimization on the same information.

## Procedure residual control

When the completed v2.3 procedural training view, retrieval DB, and frozen calibration are present, the standard workflow also trains a shorter long-horizon procedural residual control and measures its Real-RAG procedural DEV/OOD trajectory.

It is a contrast/control branch and does not determine the reasoning grokking label.

## Transition detector

Primary temporal diagnostics use configurable, versioned values from `configs/phase_9_grokking_dynamics_v231.json`:

- train progress fraction: `0.80` of observed train gain;
- OOD progress fraction: `0.50` of observed OOD gain;
- minimum plateau intervals: `3`;
- minimum observed OOD gain for a grokking-like claim: `0.03`;
- minimum OOD/train delay ratio: `2.0`;
- memorization minimum train gain: `0.10`;
- final-step readout destination dose minimum: `0.75`;
- critical-dose progress fraction: `0.50` of that step’s max-dose OOD accuracy, with absolute floor `0.01`;
- critical-dose leftward-shift minimum: `0.15`;
- temporal readout: largest OOD accuracy jump must be at least `0.50` of total OOD gain, with margin/NLL linear \(R^2 \ge 0.70\).

Because these choices are diagnostic conventions rather than universal scientific constants, the report also runs a sensitivity grid over train-progress fraction, OOD-progress fraction, and delay ratio. A fragile conclusion is visible rather than hidden behind one threshold.

### Formalized surface and readout statistics

The detector reduces the measured step × dose surface to a first-class `critical_dose(t)` series:

- lowest dose whose OOD answerable accuracy reaches the configured fraction of that step’s maximum OOD accuracy;
- start/end critical dose, leftward shift, monotonic-leftward flag;
- `critical_dose_migrates_left` versus `fixed_high_critical_dose`.

Separately, the time curve is tested for **temporal readout**: a sharp discrete OOD accuracy jump accompanied by smooth gold-token margin and/or NLL motion (`smooth_logits_with_sharp_accuracy`).

Primary labels still follow a conservative temporal-first cascade. Competing active patterns are always emitted in `competing_evidence` so a cascade winner cannot silently suppress readout, leftward-dose, representation-precede, or memorization signatures.

Possible labels:

- `READOUT_THRESHOLD_LIKELY`
- `ORDINARY_FEATURE_LEARNING`
- `MEMORIZATION_WITHOUT_ABSTRACTION`
- `GROKKING_LIKE_TRANSITION_DETECTED`
- `DELAYED_GENERALIZATION_REPRODUCED`
- `INSUFFICIENT_EVIDENCE`

### Scientific report contents

`GROKKING_DYNAMICS_REPORT.md` is compiled as a paper-style methods/evidence document, not a results dump. Required sections include:

- abstract / diagnostic verdict;
- experimental question and mechanism taxonomy;
- preregistered methods and numeric thresholds;
- observation pattern catalog with confirming techniques and falsifiers;
- results with technique↔evidence mapping (temporal, critical-dose, readout, LoRA/representation/optimizer, factors, procedure control);
- sensitivity analysis and fragility;
- competing explanations matrix;
- limitations and artifact index.

## Automatic command

Run after a completed v2.3 DEV/OOD experiment:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_grokking_dynamics_v231.ps1 `
  -V23RunRoot ".\artifacts\phase_9_residual_capability_v23" `
  -V22RunRoot ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200" `
  -Profile standard `
  -RequireCuda
```

This single command runs the full standard observatory unattended:

1. prerequisites and sealed-data audit;
2. training-variant/probe construction;
3. core long-horizon training;
4. every-checkpoint temporal evaluation;
5. LoRA norm/rank dynamics;
6. adaptive step × dose surface;
7. representation dynamics;
8. one-factor phase screen;
9. procedural long-horizon contrast control when available;
10. transition detection and sensitivity analysis;
11. final `GROKKING_DYNAMICS_REPORT.md`.

Interrupted training can be relaunched with the same `-OutputRoot -ReuseExisting`; the runner uses the newest retained resumable Trainer checkpoint when the final training manifest does not yet exist.

## Runtime controls

For an initial plumbing run:

```powershell
-Profile pilot
```

For the fullest long-horizon experiment:

```powershell
-Profile extended
```

The expensive branches can be disabled independently with:

```text
-SkipDoseSurface
-SkipRepresentation
-SkipFactorSweep
-SkipProcedureControl
```

`-FactorMaxRuns N` is also available for debugging the factor orchestrator. It should not be used to draw final factor conclusions unless the omitted runs are explicitly documented.

## Primary output directory

A complete standard run produces, among other artifacts:

```text
GROKKING_DYNAMICS_REPORT.md
grokking_dynamics_report.json
TRANSITION_DETECTION.md
transition_detection.json
core_time_curve/training_time_curve.json
core_time_curve/training_time_curve.csv
dose_time_surface/dose_time_surface.json
dose_time_surface/dose_time_surface.csv
adapter_dynamics.json
adapter_dynamics.csv
representation_dynamics.json
factor_phase_diagram.json
factor_phase_diagram.csv
FACTOR_PHASE_DIAGRAM.md
core_reasoning_long_horizon/optimizer_dynamics.jsonl
core_reasoning_long_horizon/dynamics_snapshots/
```

## Release interpretation

v2.3.1 does not automatically replace the v2.2 greedy production/research anchor. Its purpose is to characterize *how* residual capability appears. Any subsequent model-training decision should be based on the measured regime:

- readout threshold → calibrate/readout/delta geometry;
- ordinary learning → improve data/teacher coverage;
- memorization → increase structural diversity or constrain capacity differently;
- reproducible delayed generalization → study the transition boundary and only then consider interventions intended to accelerate it.
