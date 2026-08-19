# Source Grounding and Evidence Boundaries

The architecture was informed by the project’s supplied source materials, especially:

- *Short-Response Model — Research and Engineering Blueprint*: context-heavy/output-light product contract, bounded context assembly, external memory, stale-context avoidance, and measured efficiency.
- *MemHarness: Memory Is Reconstructed, Not Replayed*: critique/reconstruction between retrieval and action and the risk of verbatim memory replay.
- *AI Agent Systems From First Principles*: explicit state, deterministic verification, small tool surfaces, bounded loops, graph/state separation, permissions, tracing, and evals.
- The project’s recursive-state-forgetting examples and prior memory-governor work: global invariants, replaceable current state, HOT/WARM/COLD/PURGED tiers, reversible archive, and deletion-regret evaluation.

Evidence boundary:

- The supplied works support reconstructive memory, structured state, context management, and controlled harness design.
- They do **not** establish that this exact v0.1 architecture improves Rineya.
- The explicit `delete_memory` transaction, replacement-before-delete rule, and RSF evaluation suite are project-specific engineering hypotheses that require controlled experiments.
