# Publication contents review, 2026-10-09

Reviewed the campaign tree only, with no model calls, Git mutations, deletions, or network publication. Inventory and exact recommended path dispositions are in publication_review_20261009_contents.json. The inventory is a review-time snapshot; root publication work may add new files afterward.

The reviewed tree contained **4,073 files / 243,143,605 bytes**. Proposed evidence inclusion is **2,838 files / 229,050,291 bytes**; proposed recreated material omission is **1,235 files / 14,093,314 bytes**. The largest file is the 24,914,900-byte native runtime-debug operations journal, which is substantive raw evidence and should remain.

## Preserve

- Full native outcomes, trajectories, operation/forward journals, resource and terminal receipts, including failed runs and their original analyses.
- Admissions, prospective registrations, grants and approvals, source/asset bindings, independent review records, CPU test stdout/JUnit results, tests and source.
- Historical corrected and superseded reports, immutable source/input versions and archives, and provenance indexes. Duplicate source and inputs remain important at their frozen paths. The 495 nonempty duplicate groups account for 55,987,317 redundant bytes; Git can compress identical content, so do not remove evidence snapshots merely to deduplicate.
- All 11 native qualification .npz measurements. The original 13-file inventory also included 2 generated CPU fixture .npz files already listed for omission. No trained checkpoint or model-weight extension was identified.
- **The tiny run-specific HF module cache trees**, containing the actually loaded custom head source. Current analysis readers explicitly require those paths to exist and hash-match. Excluding every hf_cache directory would invalidate reader reproduction. Fifty files total only 20,710 bytes.
- Tiny canonical_metadata copies already captured in admission-bound versions.

## Omission candidates

- The duplicate 11,422,654-byte tokenizer cache asset at runs/causal-replay-tokenizer-cache-001/tokenizer.json; preserve its model/tokenizer hashes, token bindings and exact canonical asset prerequisite.
- Plotting/font caches: 3 files / 434,373 bytes.
- Generated CPU fixture and temporary trees: 1,231 files / 2,236,287 bytes. Keep test/review receipts and source snapshots outside those trees. The omission list must be published so historical references to temporary fixture paths are explained. No deletion is recommended or performed.

## Security and provenance limits

A nine-pattern screen read all 4,053 supported text files and found zero credential-shaped token, literal credential, URL credential or private-key hits. Filename screening found no sealed/protected/secret/credential key paths, and filesystem inspection found no symlinks. These screens are evidence of the checks performed, not a proof against every possible sensitive value.

Local usernames, canonical paths and host/runtime metadata occur in identity and qualification records. Preserve the underlying receipt bytes rather than silently rewriting them. No independent legal audit of copied inherited code is claimed. No source notices were found in the first 3,000 characters of campaign Python files; repository/source ownership and any applicable notices must remain intact. The copied experiment sources and tiny custom-head source should retain their existing provenance. Cached external model weights are absent.

The root owns final inclusion, source-bound replay verification and Git publication. This review does not establish model efficacy or complete the active improvement goal.
