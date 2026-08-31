# TGSA Phase 2 × MTP runbook

This runbook executes the implemented development workflow. It deliberately does
not bypass the Phase-1 v4 competence gate or evaluate the Phase-2 test split.

## 1. Branch and environment

Expected lineage:

```text
research/tgsa-phase1-v4-competence-first
  8b5737e260b5478427b8bb033cffa394a5f3fc5d
    └── research/tgsa-phase2a-ci-and-competence
```

From the repository root on PowerShell:

```powershell
$Lab = "Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1"
python -m pip install --upgrade pip
python -m pip install -e ".\$Lab[dev]"
python -m pytest ".\$Lab\tests\test_mtp.py" -q --tb=short
```

Linux/macOS equivalent:

```bash
LAB=Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
python -m pip install --upgrade pip
python -m pip install -e "./${LAB}[dev]"
python -m pytest "./${LAB}/tests/test_mtp.py" -q --tb=short
```

The complete historical TGSA suite should also remain green:

```powershell
python -m pytest ".\$Lab\tests" -q --tb=short
```

## 2. Run the fixed eight-cell rehearsal

PowerShell:

```powershell
$Root = "evidence\phase2_mtp_smoke_local"
Remove-Item $Root -Recurse -Force -ErrorAction SilentlyContinue

python ".\$Lab\scripts\phase2_tgsa_mtp_rehearsal.py" `
  --config ".\$Lab\configs\phase2_tgsa_mtp_smoke.json" `
  --dataset-dir "$Root\dataset" `
  --output-dir "$Root\study" `
  --repo-root "."
```

Linux/macOS:

```bash
ROOT=evidence/phase2_mtp_smoke_local
rm -rf "${ROOT}"
python "${LAB}/scripts/phase2_tgsa_mtp_rehearsal.py" \
  --config "${LAB}/configs/phase2_tgsa_mtp_smoke.json" \
  --dataset-dir "${ROOT}/dataset" \
  --output-dir "${ROOT}/study" \
  --repo-root .
```

The command is intentionally a **rehearsal**. The bundled config sets
`confirmatory=false`, uses CPU, one seed, three update steps, and evaluates only
the development split.

## 3. Expected artifact tree

```text
phase2_mtp_smoke_local/
├── dataset/
│   ├── dataset_manifest.json
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── study/
    ├── phase2_report.json
    └── runs/
        └── <tgsa-arm>__mtp_<mode>/
            └── seed_<n>/
                ├── final_model.pt
                ├── training.jsonl
                ├── predictions_dev.jsonl
                └── run_manifest.json
```

The test file is materialized and hash-sealed to pin the split, but this runner
never opens it for evaluation. There must be no `predictions_test*.jsonl` under
the Phase-2 evidence root.

## 4. Mechanical acceptance checks

A valid rehearsal must show all of the following:

1. every requested cell finishes;
2. all cells in a seed have one identical MTP-head initialization hash;
3. `dormant`, `backbone`, and `trajectory` cells within a TGSA arm have identical
   backbone initialization hashes;
4. ordinary next-token logits agree across MTP modes at initialization;
5. the auxiliary-only audit gives a nonzero MTP-head gradient;
6. every trajectory-conditioned cell gives a nonzero gradient on TGSA's existing
   control tensor;
7. dormant MTP parameters do not enter the training objective;
8. fixed-step training completes with finite losses;
9. reports validate against `phase2_mtp_run.schema.json` and
   `phase2_mtp_report.schema.json`;
10. the top-level decision remains `smoke_only` and efficiency fields remain
    explicitly non-gating.

The dedicated GitHub Actions workflow performs these checks automatically and
uploads the rehearsal bundle.

## 5. Read the development report correctly

The report answers four separate questions:

1. `answer_accuracy` / `answer_nll`: did task behavior change?
2. `mtp_horizons`: can the auxiliary head predict held-out future tokens at each
   trained horizon?
3. `answer_horizons`: can the answer be predicted earlier, rather than merely
   learning prompt formatting?
4. `teacher_forced_future_prefix`: how many consecutive future tokens are
   correct when the preceding future token is teacher-forced?

The fourth is a quality/feasibility surrogate only. It excludes target
verification, KV-cache mutation, rollback, sampling, and draft overhead, so it
must never be reported as speculative speed.

The two most informative interaction keys are:

```text
mtp_x_trajectory_mechanism
trajectory_readout_specificity
```

A positive rehearsal number is not a result. It is only evidence that the
estimand is wired and available for later power planning.

## 6. Confirmatory fail-closed check

Copying the smoke config and changing only `confirmatory` to `true` must fail. A
valid confirmatory config additionally requires:

```json
"registration": {
  "status": "frozen",
  "phase1_competence_report": "<path-to-current-GO-report>"
}
```

The report must pass the current Phase-1 competence loader. Missing, malformed,
obsolete-schema, or `NO_GO` evidence stops execution before either the output
root or dataset is created.

Do not manufacture a `GO`, substitute the v4 pilot mean, or relax the gate for
Phase 2. The v4 pilot explicitly found a mixture of converged and truncated
runs.

## 7. First real development run after Phase-1 competence

After the registered v4 competence run returns `GO`:

1. retain the same competence-certified task/model regime;
2. calibrate a small, outcome-blind MTP grid, for example:

```text
max_horizon:      2, 3, 4
loss_weight:      0.05, 0.10, 0.20
transition_rank:  8, 16, 32
```

3. select by minimum absolute competence across all required cells—not by a
   favorable learned-delta contrast;
4. verify that every required cell remains competent and no primary family is
   saturated;
5. estimate realized paired variance and set the seed count;
6. freeze cells, thresholds, estimator, test-bank hash, and one-evaluation
   barrier;
7. only then run the sealed Phase-2 test evaluation.

The advanced `raw_delta`, `qkv_conditioned`, `predictor_corrector`, and `full`
arms should be run under the selected MTP setting as a secondary full-ladder
screen, or separately powered if they acquire primary status.

## 8. Rineya-family checkpoint retrofit

The TinyLM workflow is a mechanism gate. A family-model implementation should
start from the selected Supa/Rineya checkpoint and preserve its RoPE, GQA, cache,
tokenizer, quantization, and tied-head contracts.

Recommended order:

1. attach parameter-matched MTP modules without TGSA conditioning and prove
   target alignment, loss, and gradients;
2. expose the production TGSA control coordinate without enabling full
   quadratic telemetry;
3. add a compact readout at selected layers and verify no ordinary-logit change
   at installation;
4. train `dormant`, `backbone`, and `trajectory` cells from byte-identical model
   and optimizer states;
5. evaluate quality and horizon acceptance potential before writing custom
   decoding kernels;
6. implement recursive drafting plus an exact fixed-shape target verifier;
7. test cache commit, rejection rollback, EOS, sampling, batch, and context
   boundary behavior;
8. benchmark accepted tokens per verifier call, draft cost, target cost, TTFT,
   TPOT, decode tokens/s, VRAM, GPU utilization, and output parity.

A quality gain without a timed verifier is a quality result. A faster draft head
without end-to-end target speed is a microbenchmark. Neither alone establishes
that MTP compensates for TGSA's sequential inference cost.
