# Phase 2C File Responsibilities

| File | Responsibility | Must not do |
|---|---|---|
| `run_phase_2c.ps1` | Windows orchestration, complete parameter forwarding, checked execution, resume selection. | Mark the human gate PASS or silently change a recorded run. |
| `build_phase_2c_dataset.py` | Merge approved train/dev data with deterministic concise-reasoning records; validate structure, grammar, uniqueness, leakage, ordering, and hashes. | Read sealed data or invent unverifiable factual targets. |
| `rank_surviving_layers.py` | Measure Taylor, temporary-bypass NLL, original-layer lineage, and optional Phase 2B structural signals. | Treat training importance as proof of removable inference depth. |
| `phase2c_common.py` | Pure ranking, candidate-matrix, category-quality, gate, efficiency, and profile-selection helpers. | Hide hard failures inside one composite score. |
| `targeted_recovery.py` | Fingerprint inputs, train/evaluate controls, create same-seed controls, repeat finalists, build frontier/catalog/report/gate. | Mix adapters from different reduced architectures or reuse changed inputs under `--resume`. |
| `resolve_phase_2c_profile.py` | Hand an explicitly approved, automated-eligible, seed-stable model/adapter pair to Phase 3. | Bypass either automated evidence or the human decision. |
| `check_phase_2c_gate.py` | Verify catalog profiles point to eligible candidates with the exact repeated-seed evidence. | Replace semantic/output/human review. |
| `audit_phase_2c_artifacts.py` | Read-only completeness, profile, seed, adapter, summary, and ranking audit. | Delete or repair artifacts automatically. |
| `README.md` | Explain experiment boundaries, controls, evidence, and outputs. | Claim measured quality or efficiency before execution. |
| `EXECUTION_MANUAL.md` | Exact run, isolated-run, resume, restart, audit, approval, and handoff instructions. | Encourage overwriting Phase 2B evidence. |
