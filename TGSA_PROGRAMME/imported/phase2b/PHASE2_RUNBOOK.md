# TGSA + MTP Phase-2 Runbook

## Branch and environment

```powershell
git fetch origin
git switch research/tgsa-mtp-rineya-complementarity-v1
cd Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 1. Unit and contract tests

```powershell
python -m pytest tests/test_mtp.py `
  tests/test_phase2_data.py `
  tests/test_phase2_model.py `
  tests/test_phase2_decode.py `
  tests/test_phase2_analysis.py -q
```

## 2. Non-evidentiary smoke matrix

Use a fresh artifact root. Reusing a populated output directory is refused.

```powershell
$Root = "artifacts/phase2_smoke"
Remove-Item -Recurse -Force $Root -ErrorAction SilentlyContinue

python scripts/run_phase2_mtp_study.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir "$Root/dataset" `
  --output-dir "$Root/study"

python scripts/verify_phase2_mtp_evidence.py `
  --config configs/phase2_tgsa_mtp_smoke.json `
  --dataset-dir "$Root/dataset" `
  --output-dir "$Root/study"
```

Expected scope: `gate.json` reports `SMOKE_ONLY`. A synthetic competence
`NO_GO` does not fail the workflow; it proves the gate refuses to reinterpret a
short smoke run as mechanism evidence.

Preserve at minimum:

- config and dataset manifest hashes;
- every initialization audit;
- run manifests, fixed final checkpoints, and training traces;
- per-example generation predictions;
- `analysis.json`, `competence_report.json`, `gate_inputs.json`;
- `evidence_verification.json` and `gate.json`.

## 3. Development competence run

```powershell
$Root = "artifacts/phase2_development_v1"
python scripts/run_phase2_mtp_study.py `
  --config configs/phase2_tgsa_mtp_development.json `
  --dataset-dir "$Root/dataset" `
  --output-dir "$Root/study"

python scripts/verify_phase2_mtp_evidence.py `
  --config configs/phase2_tgsa_mtp_development.json `
  --dataset-dir "$Root/dataset" `
  --output-dir "$Root/study"
```

Do not select a regime because `tgsa_mtp` wins on development data. Selection is
arm-blind and uses the weakest primary cell's absolute competence, NLL,
per-family behavior, generalization gap, and per-seed reliability.

A confirmatory config must set `confirmatory: true`, point to a frozen
`competence_report` whose current schema says `GO`, use new seeds, and write to a
new output root.

## 4. Real-checkpoint conversion smoke

Install both packages and the Hugging Face extra:

```powershell
cd ..\Rineya_TGSA_Retrofit_v1
python -m pip install -e ".[dev,hf]"

python scripts/run_tgsa_mtp_smoke.py `
  --model-dir "D:\Models\Supa-1.7.0" `
  --layers "2,7,12,17" `
  --dtype bfloat16 `
  --device cuda `
  --max-offset 4 `
  --output "artifacts\supa_tgsa_mtp_smoke.json"
```

Interpretation:

- exact greedy equivalence must be true;
- record both bitwise equality and numerical drift;
- bfloat16/SDPA drift is measured, not assumed zero;
- no quality claim follows from conversion neutrality alone.

Start with a few distributed layers. Do not convert every layer before a
placement sweep and quality-preservation test.

## 5. Adapter-only warm start on Rineya SFT JSONL

The trainer consumes the existing Rineya contract: the final assistant message
is the target and prompt loss is masked.

```powershell
python scripts/train_tgsa_mtp_adapter.py `
  --model-dir "D:\Models\Supa-1.7.0" `
  --train-jsonl "D:\Data\rineya_train.jsonl" `
  --output-dir "artifacts\supa_tgsa_mtp_adapter_v1" `
  --layers "2,7,12,17" `
  --stage adapters `
  --steps 500 `
  --batch-size 1 `
  --gradient-accumulation 8 `
  --max-length 2048 `
  --dtype bfloat16 `
  --device cuda
```

The first stage trains only new TGSA and MTP tensors. Release native attention
only after the adapter stage preserves the sealed baseline. The output
checkpoint intentionally excludes native pretrained tensors.

## 6. Real-model decision matrix

The minimum real evidence package contains the four paired cells:

1. native/parallel AR;
2. native/parallel + MTP;
3. TGSA AR;
4. TGSA + MTP.

Use identical data order, update budget, prompt bank, decoding settings, and
seeds. Add duplicate-next auxiliary controls before attributing an effect
specifically to future-horizon supervision.

Report separately:

- next-token and task quality;
- exact greedy equivalence;
- MTP acceptance by offset;
- tokens per target-model forward;
- per-forward latency and end-to-end wall-clock latency;
- peak memory and cache size;
- TGSA layer placement and converted fraction.

A serial-round gain with slower wall-clock is not a speed success. A wall-clock
gain with quality regression outside the preregistered margin is not a combined
success.
