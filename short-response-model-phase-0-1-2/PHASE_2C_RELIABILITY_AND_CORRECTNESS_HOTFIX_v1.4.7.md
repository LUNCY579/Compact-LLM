# Phase 2C Reliability and Correctness Hotfix v1.4.7

## Scope

This release was produced from the attached v1.4.6 repository after a repository-wide review focused on interrupted Phase 2C execution, resume correctness, artifact integrity, and candidate-selection validity.

It addresses every high-confidence defect found during that review. It does not claim that no future defect can exist, and it does not replace human review of model quality.

## Critical correctness defect

The v1.4.6 Phase 2C deterministic reasoning rows stored correct assistant targets but did not define explicit response contracts. Evaluator v4.5 deliberately treats categories and reference wording as non-normative for open conversation. Consequently, a fluent wrong response could satisfy generic conversational checks and receive an unjustifiably high quality score on an objective reasoning case.

Examples observed during the run included wrong arithmetic, ordering, comparison, implication, contradiction, and set-membership outputs receiving scores near the normal conversational partial-credit level or higher.

v1.4.7 adds explicit prompt-grounded contracts to every generated objective Phase 2C reasoning row. Alternate correct wording remains accepted. Missing required answer content, forbidden wrong content, and missing required clarification receive proportional issue costs under the existing Evaluator v4.5 policy.

## Safe migration of existing work

The generated IDs, visible messages, and assistant training targets are unchanged. Only evaluation metadata was added. The dataset manifest records the exact authorized v1.4.6 predecessor hashes and an unchanged semantic digest.

On an identical `-Resume` run, v1.4.7:

- updates the run configuration only when the change exactly matches the approved contract-only migration;
- preserves completed adapters;
- preserves durable raw model replies;
- rescans and rescales every preserved reply under the new contracts;
- rebuilds generation summaries and output deviations;
- reuses teacher-forced NLL evidence because tokenized inputs/targets are unchanged;
- continues partial generation from the first missing stable ID;
- records migration audit JSON files.

Any unrelated provenance change remains blocked.

## Reliability fixes

### Windows atomic replacement

Atomic writes now retry transient Windows access/sharing/lock violations with bounded backoff. Persistent failures remain visible. Reconstructible progress sidecars are best-effort so they cannot discard a completed generated example.

### End-to-end resume

Phase 2C now forwards `--resume` to both generation and teacher-forcing evaluators. Teacher forcing persists one completed stable-ID row at a time and can resume. Generation manifests normalize equivalent local path spellings such as `artifacts\...` and `.\artifacts\...` while continuing to detect real provenance differences.

### Sentinel and comparison integrity

JSON sentinels are parsed and checked for required fields before reuse. Invalid sentinels are quarantined rather than trusted. Output comparison requires exact stable-ID set equality, rejects duplicates, and fingerprints both result files. A comparison is rebuilt when rescoring changes either input.

### Adapter and disk safety

Phase 2C disables intermediate Trainer checkpoints by default, validates free disk space before training, stages the final adapter, verifies adapter configuration and non-empty weights, and restores the previous valid adapter if final publication fails.

### Reporting and audits

Optional numerical fields render as `N/A` rather than crashing on `None`. Category reference-alignment aggregation tolerates non-applicable values. The completed-artifact audit and Phase 2C automated gate now verify adapter completeness, candidate identity, case coverage, unique IDs, repeated-seed evidence, same-seed controls, profile eligibility, profile validation, and catalog/registry consistency.

## Installation

Back up the repository source. Preserve the entire `artifacts` directory.

Extract the hotfix ZIP directly over the existing repository root:

```powershell
Expand-Archive -Path ..\Short_Response_Model_Hotfix_v1.4.7_Phase2C_Reliability_and_Correctness.zip -DestinationPath . -Force
```

Verify:

```powershell
python -m compileall -q .
pytest -q
python .\scripts\common\verify_quality_evaluation_v4_5.py
python .\scripts\common\audit_relative_opportunity_policy.py
```

Then rerun the exact original Phase 2C command with `-Resume`.

## Expected v1.4.6 continuation behavior

The first resume should report the dataset upgrade and a contract-only run-config migration. It may reopen completed generation summaries to perform a fast rescore of saved replies. It should not retrain valid completed adapters. Partially evaluated candidates should continue after their last durable stable ID.

## Evidence boundary

The automated suite validates repository logic without performing GPU training in this packaging environment. The user’s existing adapters and generated outputs remain the actual hardware/model evidence and must finish the v1.4.7 rescore and Phase 2C audit before profile approval.
