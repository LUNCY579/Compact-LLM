# TGSA Phase 1 implementation summary

## What changed

This branch converts the Phase 0 reference operator into a trained, falsifiable mechanism-identification study without consuming the later sealed long-context evaluation.

### Model-backed experiment

- tiny decoder-only causal language model built from the existing TGSA reference block;
- required arms: sequential cascade, information-matched history control, learned finite difference;
- deterministic pointer-chasing, modular state-update, and direct-lookup tasks;
- longer-chain held-out development and test strata;
- answer-token training with fixed update count and no dev checkpoint selection.

### Identifiability controls

- history and finite-difference arms have the same state-dictionary structure and parameter count;
- history and finite-difference arms start from byte-identical weights for every paired seed;
- cascade and finite-difference arms copy every compatible shared tensor from one canonical supernet initialization;
- all arms receive the same materialized examples, order, optimizer, update count, and paired seed;
- per-example predictions preserve exact pairing for analysis.

### Falsifiers and guardrails

- batch-rolled trajectory controls break example alignment without moving tokens across sequence positions;
- zero and sign-negated controls remain diagnostic interventions;
- direct lookup is a short-horizon non-inferiority guardrail;
- a cascade-only win cannot support a finite-difference claim unless the history control is also beaten.

### Evidence machinery

- SHA-256 dataset, instance-ID, source, checkpoint, log, and prediction records;
- fixed final checkpoints and resumable intermediate checkpoints;
- pre/post trajectory telemetry;
- gradient-group norms, batch signatures, and example IDs at training steps;
- failure bundles with traceback and last reproducible batch;
- stale completed-run rejection after config, data, or source drift;
- hierarchical paired bootstrap over seeds and within-stratum examples;
- separate confirmatory and CI-smoke configurations;
- JSON schemas for dataset, run, and analysis artifacts.

### Developer and agent support

- one-command doctor for lineage, initialization, finite forward/backward, interventions, and hook cleanup;
- focused unit tests for data, model, intervention, runner, analysis, and evidence-reuse contracts;
- GitHub Actions end-to-end smoke workflow;
- optional manual full matrix on a Linux self-hosted runner labeled `tgsa`;
- PowerShell commands and a failure-triage playbook.

## Important files

| Area | Files |
|---|---|
| Data | `tgsa/phase1_data.py` |
| Model and interventions | `tgsa/phase1_model.py` |
| Training/evidence runner | `tgsa/phase1_runner.py` |
| Frozen statistics | `tgsa/phase1_analysis.py` |
| Registered configs | `configs/phase1_model_identifiability*.json` |
| CLIs | `scripts/prepare_phase1_data.py`, `phase1_doctor.py`, `run_phase1_model_study.py`, `analyze_phase1_model_study.py` |
| Scientific contract | `PHASE1_PREREGISTRATION.md`, `PHASE1_EVIDENCE_LEDGER.md` |
| Operator handoff | `PHASE1_MODEL_IDENTIFIABILITY.md`, `AGENT_DEBUGGING_GUIDE.md` |
| CI | `.github/workflows/tgsa-phase1.yml` |

## What has not been claimed

The implementation and smoke workflow do not establish that trajectory guidance improves capability. The registered full 3-arm × 5-seed experiment must complete and pass every frozen gate before the narrow Phase 1 mechanism statement is supportable.

Even a positive Phase 1 result does not establish Qwen/GQA transfer, natural-language long-context gains, efficiency, scaling, novelty, or an arXiv-primary claim. Those remain later, separately registered stages.
