# Rineya intelligence quality foundry: research evidence

This publication preserves the dataset pipeline, code, records, failures and discoveries produced by the isolated Rineya dataset foundry. It is an evidence archive in the private ChatBubble repository, not a new training release or model recipe.

The original foundry lives outside this Git checkout. Included files retain their exact bytes. Repeated immutable snapshots share one SHA-256-addressed compressed object, while the path catalog preserves every included historical occurrence. No source file, historical receipt, role, target or line ending is rewritten.

## Start here

- [Research review and limits](REVIEW_SUMMARY.md)
- [Completed publication and backup results](PUBLICATION_REPORT.md)
- [Original foundry README](archive/source/README.md)
- [End-to-end plan](archive/source/docs/END_TO_END_PLAN.md)
- [Every-row supervision contract](archive/source/docs/REVIEW_WORKFLOW.md)
- [Long-context curriculum](archive/source/docs/LONG_CONTEXT_CURRICULUM.md)
- [RPQ2 prospective design disposition](archive/source/program/20261010_resource_precedence_cohort_design_v1/ROOT_DISPOSITION.json)

Small core documents, source code and selected reports are directly readable under `archive/source/`. The complete included evidence tree is represented by `archive/paths.jsonl.gz`, `archive/objects.jsonl.gz` and compressed content-addressed objects. `archive/omissions.jsonl.gz` records deliberate exclusions and unsuccessful captures. The publication report records the actual inventory and validation results.

## Verify and restore

Use the standard-library publication tool; it imports no model framework:

```text
python publication_tools/snapshot.py verify archive
python publication_tools/snapshot.py restore archive /a/new/empty/foundry
```

Restoration preserves included relative paths and exact bytes. It refuses traversal, symlinks and overwriting existing files. Historical absolute Windows paths inside records remain historical identities; unavailable original inputs or runtimes are not made available by restoration. Do not execute archived experiment wrappers merely because they appear here.

## Scope and omissions

The archive includes owned core/cycle history and the RPQ2 design, together with its three explicitly referenced cross-program evidence files and the newly supplied continuation checkpoint 061. That checkpoint records other-owner research; it does not authorize copying their entire active program. The four earlier admission-v10 files accidentally written in the experiment checkout are retained separately under `_supplement/accidental_checkout_v10/`; their distinct versions are not silently replaced by later foundry copies.

Protected CHECK/FINAL/evaluation/spent/private-bank/sealed payload paths, recognized credential filenames, weights, caches, derived databases and unreferenced other-owner active programs are excluded. Detected credential signatures cause the whole occurrence to be omitted; the heuristic scan cannot guarantee absence of every possible secret. Excluded originals remain local. Metadata references to unavailable artifacts are retained; they do not imply those artifacts were opened or published. The catalog and exclusion report define the exact publication boundary.

## Scientific status

The pilot is a `REVIEWED_RESEARCH_PILOT`, with additional admission gates. Historical finite-language checks, source review, metadata/context checks and exact matches retain their original scope. Related views and repeated revisions are not independent skills or examples. RPQ2 remains a proposed design on HOLD, excluded from training and admission qualification; its evidence is archived here. No new rows, reservations, source permissions, native qualification or learner comparison were created by this publication. Intelligence, acquisition and retention gains remain UNMEASURED; training readiness and authorization remain false.

## Dataset byte backup

The separate [TRAIN-declaration backup package](hf_dataset_backup001/README.md) preserves 102 existing files with TRAIN-related inventory labels and 38 small source metadata files. Payloads are stored in the private Hugging Face dataset `Luncy1978/rineya-crs-001-research`; their exact content identity, member hashes, capture/readback receipts and remote verification accompany this Git publication. Its existing CRS snapshots cover another project and were not overwritten. TRAIN-related labels are preserved declarations, with false eligibility; they do not establish canonical selection, source ownership, permitted model-development use or admission. Archive payloads and download caches are excluded from Git; evidence and source remain here.
