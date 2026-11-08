# Operator runbook

Run commands from `experiments/rineya_id_balancing_skill_routing_v1`. Python 3.10+
is required. Tests use Python/PyTorch/NumPy versions recorded in validation. Native
optional dependency ranges express expected APIs, not executed compatibility tests.
Use a separate environment; choose the appropriate PyTorch CUDA build for the host.

## 1. Reproduce the offline implementation

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m idr demo --out runs/cpu-demo-001
python -m idr compare --candidate runs/cpu-demo-001/prior_id_report.json --baseline runs/cpu-demo-001/prior_integral_report.json --objective quality --out runs/cpu-demo-001/comparison.json
python -m idr verify --freeze validation/source_freeze.json
```

The comparison remains `NO_PROMOTION`, regardless of synthetic fixture scores.
Repeat the demo into a different directory to compare deterministic JSON outputs.
The score-stream traces include quantile; the trainable matrix has none/sign/I/ID/aux.

## 2. Bind real local artifacts without altering them

Copy `configs/native.template.json` and `configs/pool.template.json` to `local/`.
Keep `local/` out of version control. Produce manifests **outside** each model folder:

```bash
python -m idr manifest --root "C:/Models/Rineya-FP32" --out local/base-manifest.json
python -m idr manifest --root "C:/Models/ledger-adapter" --out local/ledger-manifest.json
```

Put each complete manifest under its entry's `manifest` key in `local/native.json`.
Set the actual paths. In `local/pool.json`, set expert tree hashes, operation,
qualified eligibility, priors and fixed coefficients. `core` is reserved for the
adapter-off base; native adapters are keys under the native config's `adapters` map.
Do not mark an adapter qualified merely to make a command run.

After ensuring exclusive host access and explicit local execution authorization,
set `allow_native: true`. Keep `dtype: float32`. Inspect the preflight report:

```bash
python -m idr preflight --config local/native.json --out runs/native-preflight-001.json
```

Exit 2 means blocked. Do not bypass missing hashes, model types, dtypes, dependencies,
or VRAM. A ready preflight does not establish parity with the registered loader or
predict activation/optimizer peak memory. Run a very small native parity test first.

## 3. Build fresh development inputs

The following are example budgets, not a scheduled or authorized full GPU sweep:

```bash
python -m idr generate-ledger --split train --parents 120 --seed 31 --out local/ledger-train.json
python -m idr generate-ledger --split calibration --parents 80 --seed 31 --out local/ledger-calibration.json
python -m idr generate-ledger --split dev --parents 80 --seed 31 --out local/ledger-dev.json
```

Before admission, call `artifacts.validate_rows(train + calibration + dev)` and
check against all historical fingerprints. The CLI checks each file and retains
calibration identities for cross-file evaluation protection; it cannot discover
external historical payloads automatically. Keep confirmation inaccessible until
selection and preregistration. Increase independent parents based on the power
analysis, not by counting the six variants as six independent cases.

## 4. Stage A — train-only success estimation and frozen routing

Acquire counterfactual labels for TRAIN using every qualified candidate. This is
paid acquisition work, not deployable oracle routing:

```bash
python -m idr capture-native --config local/native.json --spec local/pool.json --rows local/ledger-train.json --out local/train-capture.json
python -m idr fit-quality --spec local/pool.json --rows local/train-capture.json --out local/quality.pt
```

Read the SHA256 from `local/quality.pt.json`. Substitute that real digest below:

```bash
python -m idr predict-quality --spec local/pool.json --rows local/ledger-calibration.json --head local/quality.pt --head-sha256 "ACTUAL_SHA256" --out local/calibration-predicted.json
python -m idr calibrate --spec local/pool.json --rows local/calibration-predicted.json --out local/controller.json
```

The quality head is a trained success predictor, not automatically calibrated.
Check held-out reliability/Brier curves before letting a five-point predicted-quality
margin carry decision weight. It uses only the eight input features in the supplied
ledger generator. Other domains must provide a frozen, independently qualified,
goal-blind feature contract. No hidden expected state or desired goal may enter it.

Default cost priors are equal and `cost_weight=0`. Do not rank experts by the actual
counterfactual elapsed cost after observing a test answer. Calibrate cost predictions
on development data before enabling cost-aware routing.

For actual held-out execution:

```bash
python -m idr predict-quality --spec local/pool.json --rows local/ledger-dev.json --head local/quality.pt --head-sha256 "ACTUAL_SHA256" --out local/dev-predicted.json
python -m idr execute-native --config local/native.json --spec local/pool.json --rows local/dev-predicted.json --state local/controller.json --split dev --out runs/native-dev-001.json
```

Adapter choices occur before complete model calls. Outputs use the exact-state
scorer. For a routing-headroom diagnostic, `capture-native` may capture DEVELOPMENT
counterfactual outcomes; then `predict-quality` and `replay --evidence native_replay`
can compare controller choices. That replay is not an actual end-to-end speed test.
Do not counterfactually capture confirmation to search for another winner.

Prepare none/sign/integral/ID configurations independently, with the same qualified
bank and predictor. Calibrate their states separately and use equal development
budgets. Use no test feedback to select gains, priors, coefficients or exclusions.

## 5. Stage B — fresh shared-backbone LoRA experts

Make a separate native config with **no registered adapters**. Register exact
projection paths after inspecting the existing native model. For a compatible
28-layer dense Qwen3 export, an example path is `model.layers.24.mlp.down_proj`;
this is an example to verify, not an instruction to modify an unsupported core.
The wrapper does not accept existing PEFT, sparse, tangent, or quantized layers.

```bash
python -m idr train-native --config local/native-bank.json --rows local/ledger-train.json --arm id --steps 20 --seed 0 --out runs/native-bank-id-seed0
```

This executes bounded completion-only SFT, with the base frozen, B factors starting
at zero, and one controller update after each successful optimizer step. It saves
adapter/router/controller tensors, never a rewritten base. Use matching
none/sign/integral/auxiliary arms and seeds only after the first parity/memory screen.
Every generated report remains a native **pilot**, not qualified model improvement.

For evaluation restore through the Python interface:

```python
from idr.artifacts import load_json
from idr.native import load_base, load_bank

cfg = load_json("local/native-bank.json")
model, tokenizer = load_base(cfg)
report = load_json("runs/native-bank-id-seed0/report.json")
bank = load_bank(model, cfg, "runs/native-bank-id-seed0/adapter_bank.pt",
                 report["checkpoint_sha256"])
# model is now eval-only; controller biases remain frozen during generation.
# Pass it to the existing qualified Rineya evaluation harness, with fresh inputs.
```

The standalone native runner is single-process, FP32, unpadded one-example SFT.
Full optimizer resume, DDP, gradient checkpointing, arbitrary nonuniform token
marginals, and full backbone MoE conversion are deliberately unsupported. The
per-projection API supports explicit padding masks for future qualified integrations.

## 6. Confirm once, analyze once

Select the configuration before generating/reading confirmation outcomes. Obtain
predictions from the fixed TRAIN-fitted head without outcome feedback. Bind every
input into a per-arm freeze:

```bash
python -m idr freeze --inputs local/pool.json local/confirm-predicted.json local/controller.json local/native.json --out local/id-confirm-freeze.json
python -m idr execute-native --config local/native.json --spec local/pool.json --rows local/confirm-predicted.json --state local/controller.json --split confirm --freeze local/id-confirm-freeze.json --out runs/id-confirm.json
```

Use a separate freeze for the preselected baseline's fixed state. The first command
is only a mechanical hash freeze; record the complete protocol, selected objective,
sample sizes, seed policy and analyses in the existing research decisions log before
native calls. Any consumed freeze remains consumed after failure.

```bash
python -m idr compare --candidate runs/id-confirm.json --baseline runs/baseline-confirm.json --objective quality --out runs/confirmation-comparison.json
```

The output can identify a statistical candidate, never automatically approve one.
Add reference-loader parity, held-out general-language retention, actual calibrated
feature cost, repeated hardware timing, and external plan validity as required by
the protocol. Do not report exact-JSON state scoring as Blocksworld plan validation.

## 7. Where to integrate without duplication

Insert `CallExecutor` before a complete existing RCWM/BWM transition prediction.
Use the current qualified model/adapter backend rather than importing a different
branch's loader. Keep the proposer, world model and language response pools separate.
The existing goal readout, planner, resource lock, fingerprint registry and external
verifier remain authoritative; this experiment does not replace them.

For fresh LoRA mixtures, `moe.inject` is the seam for a preloaded compatible model.
Do not merge this branch into the integration branch or copy registered artifacts
until the operator explicitly requests it. Native progress, including negative
results, should be recorded with actual hashes and evidence level.
