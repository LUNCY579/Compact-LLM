# File responsibilities — Phase 8 release candidate

The PowerShell runner is the normal entry point. Python executables may be run individually for debugging or resumable work.

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_8.ps1` | Runs sealed evaluation, collects available ablation evidence, builds the release audit, packages the selected model, and writes the final gate. | Authorization to ship. |
| `collect_ablation_evidence.py` | Maps existing artifacts to required A–I controls and marks missing evidence honestly. | A benchmark generator or evidence fabricator. |
| `build_release_audit.py` | Checks prior gates, sealed floors, runtime decision, model/reference hashes, and release blockers. | A substitute for product QA, licensing, or security review. |
| `package_release_candidate.py` | Creates a manifest-only package or copies the selected model plus audit/reference metadata and the response-control policy/contract. | Model quantization or installer construction. |
| `create_phase_8_gate.py` | Creates the final pending human release decision file. | Human approval. |
| `README.md` | Explains release inputs, package modes, blockers, and rollback expectations. | An execution log. |
| `EXECUTION_MANUAL.md` | Provides sealed-run/package commands and final review procedure. | Evidence that rollback or product integration passed. |
| `FILE_RESPONSIBILITIES.md` | Documents ownership and boundaries for every file in this folder. | Runtime code. |

Phase 8 intentionally fails or remains pending when required evidence is missing. That is a production safeguard, not an orchestration failure.
