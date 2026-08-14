# Phase 2 paper documentation

This folder is the human-facing navigation layer for the completed Phase 2
study. It sits at the lab root beside `paper/`, `reports/`, and `results/`, and
contains documentation only. Canonical numerical evidence remains in the
machine-generated analysis files linked below; this folder does not replace
those sources.

## Start here

1. [Central claim](CENTRAL_CLAIM.md) — the one-sentence paper claim, supporting
   argument, and explicit non-claims.
2. [Technique breakdown](TECHNIQUE_BREAKDOWN.md) — decoder architecture,
   comparators, experiments, and statistical design.
3. [Key results](KEY_RESULTS.md) — compact result tables and decisive paired
   effects.
4. [Mechanistic observations](MECHANISTIC_OBSERVATIONS.md) — what the evidence
   says about calibration, policy composition, trajectories, committee size,
   and specialist contribution.
5. [Provenance and reproduction](PROVENANCE_AND_REPRODUCTION.md) — integrity
   hashes, completion checks, runtime boundaries, and raw-artifact policy.

## Canonical paper package

- [Paper analysis and abstract framework](../../paper/PAPER_ANALYSIS.md)
- [Methods and mathematics](../../paper/METHODS_AND_MATH.md)
- [Experimental protocol](../../paper/EXPERIMENTAL_PROTOCOL.md)
- [Results analysis](../../paper/RESULTS_ANALYSIS.md)
- [Claims and support ledger](../../paper/CLAIMS_AND_SUPPORT.md)
- [Evidence ledger](../../paper/EVIDENCE_LEDGER.md)
- [Limitations and ethics](../../paper/LIMITATIONS_AND_ETHICS.md)
- [Generated tables](../../paper/TABLES.md)
- [Final mechanism report](../../reports/PHASE2_MECHANISM_REPORT.md)
- [Final completion audit](../../reports/phase2_completion_audit.json)

## Source-of-truth order

When two documents differ in precision, use this order:

1. raw result hash and runtime manifest;
2. machine analysis JSON;
3. generated table/figure metadata;
4. final mechanism report and claims ledger;
5. navigation summaries in this folder.
