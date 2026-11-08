# Independent original CPU Track B matrix auditor

`audit_alignment_matrix_v2.py` is ready; the full real audit has not run. It is a standard-library-only audit tool stored under evidence, outside the frozen scientific source map. Twelve synthetic tests passed. All 139 currently registered Track B source hashes still match after tool preparation.

Run **only after all 16 pairs and their successful aggregate exist**:

```powershell
& 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe' 'C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/cross_scale/evidence/07_independent_audit/tools/audit_alignment_matrix_v2.py' --run-id alignment-cpu-v2-002 --output-dir 'C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/cross_scale/evidence/07_independent_audit/alignment-cpu-v2-002-audit-001'
```

The output directory must be new and strictly below `cross_scale/evidence/07_independent_audit`, outside `tools`. A later retry needs a different audit output suffix. Existing audit output is never replaced. Science evidence and local artifacts are read-only.

The tool checks the frozen original CPU config, registration contract digest and full embedded-registration digest, source/config/data hashes, started-to-terminal contract preservation, one complete attempt per pair, and a complete aggregate. It inventories failed and abandoned attempts, their terminal/failure receipts and preserved partial files. Unfinished attempts or missing matrix/aggregate coverage reject before selection. It never computes selection from a partial matrix.

For complete evidence, it streams SHA-256 over the shared activation cache and all 16 bridge artifacts without deserializing them, checks all 96 objective cells, verifies 96 optimizer steps only for B2/B3 with consistent complete training-prompt permutations, validates capture/data/checkpoint metadata, donor replay/frozen-state claims and required token-stratum counts/finite bounds. All-token counts must equal the saved held-out capture lengths; final-token counts must equal the 32 held-out prompts. It reconstructs all 48 B1/B2/B3 candidate gate records with the exact six preregistered inclusive inequalities and rank order, then checks both `selection.json` and its aggregate-manifest copy. Every failed/abandoned pair must also appear in the aggregate's source inventory. Input hashes are rechecked at the end.

Outputs are `attempt_inventory.json`, `input_hashes.json` and `audit_receipt.json`. Exit code 0 means audit PASS, 1 means evidence rejected (receipt preserved), and 2 means output/path setup was refused. The audit's own PASS status is separate from the reconstructed MECHANISM_GO/MECHANISM_NO_GO scientific gate.

This is metadata/scalar/ranking integrity only. Raw hidden/bridge tensor arithmetic, optimizer numerics, exact donor replay and live gradients are not independently recomputed. Canonical/donor checkpoint claims are compared with registered metadata, without opening checkpoint weight files. Ordinary/special counts must agree across the entire matrix, but cannot be independently derived from token IDs because those are inside the opaque activation artifact. No GPU, torch, model execution, scoring, source/protocol edit or Git command occurs. Git commitment status and a live resource environment are not certified by this auditor.
