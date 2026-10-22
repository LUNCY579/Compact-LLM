# Wave 3 — CED official conversion v1 (conflicting historical candidate)

Source branch: `research/ced-official-conversion-v1`, pinned to `8056e41cc2483407a90ac066099e2dc58c74e878`.
Source execution baseline referenced by this historical branch: `2f65f9e96f8649a86c5d2f4369181df5b8b21aa5`.

## Preservation decision
The modern MiMo central tree already contains `experiments/architecture_benchmark_001/tests/test_ced_conversion.py` at Git blob `3635b4959d9561ac804a6927420ab4a6987b5245`, different from the source branch's test blob `3aae5cd307035628ba0910a61df687263138d0b2`. The central runner also has registered CED routes and stricter stage context. This PR does **not** overwrite that existing suite or inject an older command implementation into a newer evidence-bound runtime.

Instead, preserve all six source-specific file blobs under `docs/repository-consolidation/2026-10-22/archive_sources/ced_official_conversion_8056e41/`. The two Python files use `.source.txt` suffixes so archive tests do not auto-discover or run, while their blob content remains byte-for-byte original. The source branch commit is the second parent of the integration commit, preserving the complete original tree, paths, and historical ancestry.

This is **historical consolidation, not production porting**. The source reported 87 scoped mechanical tests passing, while the official checkpoint was not loaded, official CED not constructed, and official receipts not issued. No fresh tests were run for this archive integration. A live production port requires a separate compatibility review, source audit and tests against the current central runner.

Owner-managed branch retirement only after merge and ancestry checks and branch-name consumer handoff. Never delete evidence tags. `main`, `post-training-Mimo`, and CRS remain unchanged.
