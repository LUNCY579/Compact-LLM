# Independent review of owner-exception proposal 001

Date: 2026-10-02. Reviewer `/root/data_audit`; proposal/reader builder
`/root/runtime_audit`. **REVIEWED_BOUNDED_ENGINEERING_PROPOSAL_ONLY.** The two
minimal compatibility changes are concrete and supported by the bounded CPU
regression. This is not owner consent, integrated-source acceptance, live
admission, another repair round, or probe/reader/native/model qualification.
Both candidates remain quarantined. No frozen source was edited or patched;
no probe, model, framework, tokenizer, GPU API or canonical host database ran.
Shared model/workspace ancestry is disclosed; no self-approval by the builder.

## Exact reviewed proposal

All final files in FREEZE were read and independently hashed, including README,
TEST_PLAN, PROPOSAL, both diffs, manifest, regression and its prior result.
FREEZE SHA256:
`7757d0376510df914c55219434f1faa8f8925a004ef2da6a033cf3ccb82c3dfc`.

| Artifact | SHA256 |
| --- | --- |
| probe_disk_usage.patch | `73caca4835843b0bee2412fca40a3a62f8f9b675fbe93aa23e3f94aabd1dc48e` |
| reader_supervisor_bytes.patch | `6f815fa33fd9717358aa3b20d4f2a94bcab947739bba546e9bbc4e2a814513e2` |
| PATCH_MANIFEST.json | `bafd9f5606b5b9f498351d76b151ee55f2aecdfac2a010763330523b041cf535` |
| PROPOSAL.json | `643d8663c60a0d6ac80d678b169625a4444d8412b2ff14d3d7aa9608a2f70ec7` |
| TEST_PLAN.md | `762233f4e2ee43898ff60b2318304cdde1985ed03e9bb3e669bb24a16254825e` |
| regression_test.py | `ec16f841d522884a0b56cadb096b80f3bad5d2f72a0e85d8ad193d3eaef76851` |

Both `git apply --check` invocations passed independently without applying a
patch. The regression applies each exact diff in memory, checks all original
contexts and proposed source hashes. Frozen probe431f, readerb420/execution0dcd,
native845/d79 and operationsba27 remain byte-identical. Proposed source hashes
are0e85d5cf… probe,9a6d2806… reader, andc66a1b64… execution, as recorded in the
manifest; those are prospective source bytes, not installed/qualified programs.

## Actual bounded regression

Two tests independently passed in **0.07218309999734629 seconds** on the
registered interpreter. The create-only reviewer result is
`outputs/CPU_REGRESSION_INDEPENDENT_003.json`, SHA256
`6ed426fd8f79805eb6afbfca51aa5eba2ef5cbe1e059f846ecc768f3a37c1271`.

The actual pinned Windows psutil7.2.2 API rejects a WindowsPath and accepts str
on a proposal-owned temporary directory. Its returned disk data are used only
to validate this argument contract, not as current host admission. Both failed
call sites are covered by the in-memory probe diff. The diff also changes fixed
KEY001 to the explicit new KEY002; old terminal001 is not reset or retried.

The unchanged actual native.write_new produces the supervisor newline bytes;
the frozen validator rejects its correct file hash. The proposed validator AST,
executed only in memory against the clearly synthetic CPU schema, accepts the
actual same-run file reference. Same parsed-object/changed-byte input rejects;
updating that reference without the immutable terminal evidence also rejects.
No native outcome or serving qualification was published. Temporary data and
the new regression report remain in the proposal's ignored output scope.

## Source and caller assessment

The probe patch changes exactly two Path-to-str arguments and the fixed new
registration key. It changes no source schedule, deadline, resource floor,
registry, profile boundary, cleanup, ownership, numerical setting or claimed
operation. A new002 immutable profile/output/source binding and approvals are
still mandatory; an approval for oldf9e5/code431f cannot license changed bytes.

The reader patch adds a mandatory immutable artifact-reference argument, checks
the exact registered SUPERVISOR_RESULT path under this run's results directory,
checks actual file SHA and strict parsed-object equality, then binds terminal
evidence to that actual file SHA. It neither rewrites old evidence nor guesses
serialization. The two production consumers—serving verification and retained
evidence recovery—both pass their already checked reference mappings. The
post-run publisher reaches that recovery path; no unupdated production call was
found. Standalone test callers intentionally require adaptation.

Retaining digest(outcome) is correct for the unchanged OperationsStore writer:
write_receipt writes canonical(body) with no newline, and verify_receipt requires
those exact canonical bytes plus path/hash/kind/key/filename identity. It is not
the same byte protocol as native.write_new. The proposal preserves that
distinction and adds no canonicalization fallback for mismatching files.

## Required future conditions; approval is still absent

The owner request explicitly sets the default/maximum three review-and-repair
rounds per milestone (retained attachment lines635/673). The precisely scoped
exception offered for owner decision is one additional targeted round for each
of these two candidates, necessary actual-producer/API and reference-fixture
updates, new immutable source/profile pins, and **one newly registered CPU-only
002 observation at most60 seconds after unchanged live gates**. It grants no
global iteration loop, retry of001, old-native exception, model/reader launch,
training, numerical/resource relaxation or qualification from a CPU pass.

Before any integrated candidate acceptance, TEST_PLAN must be completed: use
the actual writer in existing standalone artifact fixtures and real temporary
OperationsStore reconciliation/read-only recovery, preserve all prior negatives,
verify no dispatch/reserve/acquire/reconcile/release during reader recovery,
freeze reviewed code/tests/contracts and obtain fresh independent acceptance,
clean commit and exact-byte attestations. The two proposal tests are sufficient
compatibility evidence for considering the exception; they are not that full
integration test or a full existing-suite execution.

Reviewer assignment must stay genuinely independent of whoever authored/applies
the changed source. Runtime authored these diffs and cannot self-approve their
integration. The current probe code hardcodes science=runtime_audit and
operations=data_audit and bans original root/scientific_audit builder aliases;
the planned002 profile/review policy therefore needs an explicit, consistent
non-builder assignment or separately versioned reviewer policy during the
authorized registration work. A role string cannot manufacture independence.
This report does not silently alter that policy or authorize broader code work.
Root accepts this condition: prospective probe SCI review must use a fresh
neutral agent who built neither probe nor patch, with operations review by
`/root/data_audit` and fresh exact-source attestations. The corresponding policy
binding update belongs only to the owner-authorized registration scope; it must
not widen authority or numerical/resource gates. Reader Runtime remains unable
to approve its own source; Root/data review discloses the shared design context.

Until the owner explicitly grants the stated exception, neither diff may be
integrated and no new profile/worker may be dispatched. Afterwards, a refusal or
completed002 observation exhausts its one-attempt key; any unresolved issue at
the extra round remains quarantined. All original failures, costs, sources,
approvals, histories and the native/reader/training quarantine remain recorded.
The existing47.421 charge is not reset; a new60-second reservation is additive
and unknown creation/exit remains held. The proposal/this review did not query
the live canonical database or current RAM/GPU/process readiness.

The engineering direction is suitable for a concrete owner decision under
these conditions. No exception, integration, launch or qualification is granted
by this report. Independent 32-parent TRAIN preparation may proceed separately
under its existing CPU-only authorization while the owner decision is pending.

```json
{"review_id":"CRS-OWNER-EXCEPTION-PROPOSAL-REVIEW-001","reviewer":"/root/data_audit","builder":"/root/runtime_audit","status":"REVIEWED_BOUNDED_ENGINEERING_PROPOSAL_ONLY","freeze_sha256":"7757d0376510df914c55219434f1faa8f8925a004ef2da6a033cf3ccb82c3dfc","independent_regression_tests":2,"independent_regression_passed":2,"independent_regression_seconds":0.07218309999734629,"regression_sha256":"6ed426fd8f79805eb6afbfca51aa5eba2ef5cbe1e059f846ecc768f3a37c1271","patches_applied_to_frozen_source":false,"owner_exception_granted":false,"new_probe_key_proposed":"CRS-OWNED-IDENTITY-002","new_probe_runs_max":1,"new_probe_seconds_max":60,"model_or_framework_calls":0,"canonical_DB_accessed":false,"native_or_reader_qualification":false,"future_source_test_profile_role_commit_and_live_gates_required":true}
```
