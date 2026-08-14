# Provenance and reproduction

## Integrity anchors

- Sealed v2.2 cornerstone SHA-256:
  `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`.
- Selected DEV free-running raw SHA-256:
  `59776b7782174d76251716984357829e9e2b00f064726d3a78735c8462f3cc09`.
- Selected committee raw SHA-256:
  `e690613c8d3c60082b77f38c68839d55dce1c87f4939f71f34181ec3b33f6f03`.
- Final synthesis verifier status: `FINAL_SYNTHESIS_VERIFIED`.
- Completion audit status: `PASS`.
- Full test suite: 67 passed.

## Runtime boundary

CUDA allocation failures occur before durable append. A failed attempt is
discarded; only a successful exact recomputation becomes a JSONL row. The final
committee worker appended 61 valid rows and exited before closing its runtime
manifest. Reconciliation left the raw file unchanged, marked the invocation as
an unclean exit, independently revalidated all 48,600 unique identity-bound
rows, and recorded a post-hoc verified completion basis.

## Git artifact policy

Raw inference JSONL files and the 464 MiB teacher-forced top-k cache remain local
because they are generated artifacts and exceed ordinary Git-hosting limits.
Git tracks their hashes, frozen configs, runtime manifests, strict derived
analyses, reports, tables, figures, and audit packets. This preserves scientific
traceability without placing multi-hundred-megabyte model outputs in normal Git
history.

## Authoritative checks

- [Cornerstone verification](../../reports/PHASE2_CORNERSTONE_VERIFICATION.md)
- [Final synthesis verification](../../reports/phase2_final_synthesis_verification.json)
- [Completion audit](../../reports/phase2_completion_audit.json)
- [Rank implementation parity](../../reports/PHASE2_RANK_TOPK_PARITY.md)
- [Runtime boundaries](../../reports/PHASE2_SELECTED_RUNTIME_BOUNDARIES.md)
- [Experimental protocol](../../paper/EXPERIMENTAL_PROTOCOL.md)
- [Evidence ledger](../../paper/EVIDENCE_LEDGER.md)
