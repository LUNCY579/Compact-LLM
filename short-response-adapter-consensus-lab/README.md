# Short-Response Adapter Consensus Lab

This isolated sibling lab studies exact multi-adapter next-token consensus over the official short-response model. The official repository is read-only; the two v22 mini-labs are excluded. The completed Phase 2 package independently verifies the 4,050-example cornerstone, analyzes paired TEST rescue/harm and token traces, evaluates calibration and every nonempty DEV committee, completes selected free-running, adaptive, calibrated-confirmation, and cross-corpus panels, and generates provenance-linked tables and figures.

## Environment

Validated on Windows, Python 3.11.15, torch 2.12.1+cu126, transformers 5.14.1, PEFT 0.19.1, and an NVIDIA RTX 4070 Laptop GPU (8 GB). The shared workspace virtual environment is `..\.venv`.

## Exact PowerShell reproduction

From `short-response-adapter-consensus-lab`:

```powershell
$Python = "..\.venv\Scripts\python.exe"

& $Python -m pytest -q
& $Python scripts\discover_candidates.py
& $Python scripts\snapshot_official_sources.py --label before
& $Python scripts\reproduce_baselines.py --limit 8
& $Python scripts\probe_peft_capabilities.py
& $Python scripts\verify_cache_parity.py --limit 4 --max-new-tokens 16
```

`verify_cache_parity.py` is expected to exit nonzero in this environment; the recorded failure is why every research config uses `cache_mode: reference`.

The complete Phase 2 command order, including immutable manifests, full DEV runs, cross-corpus confirmation, analysis, and final artifact generation, is maintained in [`paper/EXPERIMENTAL_PROTOCOL.md`](paper/EXPERIMENTAL_PROTOCOL.md). Do not refit or refreeze selections after reading TEST outcomes.

For a documentation-only entry point, start at [`documentation/phase2_paper/README.md`](documentation/phase2_paper/README.md).

Run bounded DEV experiments:

```powershell
& $Python scripts\run_smoke.py --config configs\hard_vote.yaml --dataset v22_dev --limit 16
& $Python scripts\calibrate_temperatures.py --limit 16
& $Python scripts\run_smoke.py --config configs\fusion_sweep.yaml --dataset v22_dev --committee-size 7 --limit 8
& $Python scripts\run_smoke.py --config configs\negative_controls.yaml --dataset v22_dev --limit 8
& $Python scripts\run_smoke.py --config configs\eos_strict_majority.yaml --dataset v22_dev --limit 8
& $Python scripts\run_smoke.py --config configs\eos_confidence_threshold.yaml --dataset v22_dev --limit 8
& $Python scripts\run_smoke.py --config configs\eos_minimum_tokens.yaml --dataset v22_dev --limit 8
& $Python scripts\run_smoke.py --config configs\adaptive_consultation.yaml --dataset v22_dev --limit 16
& $Python scripts\run_rag_smoke.py --limit 16
& $Python scripts\analyze_pilots.py
```

Run a larger DEV continuation (omit `--limit` for full DEV; it is expensive with the exact decoder):

```powershell
& $Python scripts\run_dev.py --config configs\hard_vote.yaml --dataset v22_dev --limit 128
```

The frozen artifact already precedes the sealed result. To reproduce from a clean lab copy only:

```powershell
& $Python scripts\freeze_finalists.py --sealed-limit 32
& $Python scripts\run_sealed.py --max-vram-gb 7.5
& $Python scripts\analyze_sealed.py
& $Python scripts\snapshot_official_sources.py --label after
& $Python scripts\audit_completion.py
```

Do not refreeze after seeing sealed results. The scripts refuse to refreeze when sealed JSONL exists.

## Outputs

- Raw result/traces: `results/runs/*.jsonl`, `results/sealed/*.jsonl`
- Summaries/statistics: `results/summaries/`
- Frozen config: `artifacts/frozen_configs/frozen_final_config.json`
- Source integrity: `artifacts/manifests/official_sources_{before,after}.json`
- Quality–cost plots: `reports/plots/`
- Phase 2 synthesis: `reports/PHASE2_MECHANISM_REPORT.md`
- Primary-source literature review: `paper/RELATED_WORK.md`, `paper/REFERENCES.md`, and `paper/references.bib`
- Generated paper tables and figures: `paper/TABLES.md`, `paper/tables/`, and `paper/figures/`
- Final written-synthesis verification: `reports/phase2_final_synthesis_verification.json`
- Phase 2 completion certification: `reports/phase2_completion_audit.json`
- Machine-readable completion audit: `reports/completion_audit.json`

Runs are append-durable and resumable. An exclusive PID lock prevents concurrent writers. `scripts/repair_results.py` exists for audited deterministic deduplication and archives the unrepaired source before modification.
