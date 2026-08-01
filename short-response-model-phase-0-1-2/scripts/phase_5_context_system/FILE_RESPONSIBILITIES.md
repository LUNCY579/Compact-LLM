# File responsibilities — Phase 5 context system

The PowerShell runner is the normal entry point. Python executables may be run individually for debugging or resumable work.

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_5.ps1` | Resolves the selected model, evaluates retrieval/context budgets, benchmarks exact prefix reuse, and writes report/gate evidence. | A retrieval-only mock runner. |
| `context_system.py` | Implements deterministic relevance/salience/recency scoring, stale/contradicted/duplicate filtering, tier budgets, and prefix keys. | A vector database or persistent production storage layer. |
| `evaluate_context_system.py` | Produces retrieval traces and metrics and optionally compares actual compact-versus-full model generations. | A hidden semantic judge. Its rules are explicit and auditable. |
| `benchmark_prefix_cache.py` | Measures full prefill versus exact stable-prefix KV reuse and checks next-token logits consistency. | Cross-model or stale-prefix cache reuse. |
| `check_phase_5_gate.py` | Converts retrieval quality and context-token/KV savings into a proportional opportunity comparison and writes the pending gate. | Human approval. |
| `check_human_gate.py` | Backward-compatible Phase 5 gate wrapper. | The canonical multi-phase gate checker. |
| `README.md` | Explains architecture, required case schema, artifacts, and production boundaries. | An execution log. |
| `EXECUTION_MANUAL.md` | Provides normal/resume commands and trace-review procedure. | Evidence that retrieval is correct. |
| `FILE_RESPONSIBILITIES.md` | Documents ownership and boundaries for every file in this folder. | Runtime code. |

The memory store and retrieval cases must be product-specific before release; the included fixture proves the pipeline shape, not the user's final production memory policy.
