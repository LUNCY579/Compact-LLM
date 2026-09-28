# The merged CED packages run here — except 13 tests that cannot build their fixtures on Windows

Two remote branches were fetched and merged into
`research/ced-rab001-hotfix001-qualification-v1`. Both are purely additive and neither touches
`architecture_benchmark_001` or `rab001_exploration`; the implementation digest is unchanged at
`a329eb7b` and the 306 retention-gate tests still pass.

| merged package | what it is | upstream state |
|---|---|---|
| `ced_native_loader_issue4_v1` | 32-case native loader suite for issue 4 | `IMPLEMENTED_FRESH_NATIVE_EXECUTION_BLOCKED`, `executed_cases: 0` |
| `full_width_transition_v1` | FWT-001, capability-preserving full-width memory transition | implementation + mechanical tests, no official-weights run |
| `dual28_ced_v1` | D28-001, full E28 + full D28 with zero-init memory reads | implementation + mechanical tests, no official-weights run |
| `cross_scale_remote_receipt_d82_v1` | E28/D36 cross-scale receipt | remote runner attempts preserved as failures |

All three CED packages state the same blocker: the environment that wrote them lacked the
official weights and Transformers. **This worktree has both**, so merging moves them somewhere
they can execute.

## Test result in this environment

    dual28_ced_v1                    89 passed,  5 failed
    full_width_transition_v1         70 passed,  5 failed
    ced_native_loader_issue4_v1      30 passed,  3 failed  (+5 subtests)
    cross_scale_remote_receipt_d82   23 passed,  0 failed
    ------------------------------------------------------
    total                           212 passed, 13 failed

**All 13 failures are Windows environment limits, not logic defects.** Two causes, both in
*negative* tests — tests that construct a corrupt or hostile input to prove a guard rejects it:

**1. safetensors mmap, `os error 1224` (9 tests).**
`The requested operation cannot be performed on a file with a user-mapped section open.`
The test loads a checkpoint — which memory-maps it — then rewrites the same path with a corrupted
tensor to check the loader fails closed. POSIX permits rewriting a mapped file; Windows does not.
The guard is never reached.

    dual28      test_bad_checkpoint_rejected[missing|nan|shape|dtype]
    fwt         test_checkpoint_corruption_rejected[extra|nan|shape|dtype]
    issue4      test_corrupt_checkpoint_fails_closed[missing|extra|shape]

**2. Symlink creation privilege, `WinError 1314` (2 tests).**
`A required privilege is not held by the client.` Creating a symlink on Windows needs
Administrator or Developer Mode. The tests create one specifically to prove the scope guard
rejects symlinked paths, so again the fixture — not the guard — is what fails.

    dual28      test_output_scope_and_no_overwrite
    fwt         test_source_hash_and_symlink_checks

## What this does and does not license

**It does not say the guards work.** Thirteen safety behaviours — corrupt-tensor rejection,
NaN rejection, shape/dtype mismatch rejection, symlink scope enforcement — are **unverified on
this platform**, because their fixtures could not be built. That is a weaker statement than "they
pass" and a different statement from "they fail". Do not report these packages as green here.

**It does say the packages are otherwise portable**: 212 of 225 pass unmodified on Windows with
the pinned CPU runtime, including every positive construction, parity, inventory and receipt test.

## Suggested fixes, for whoever owns those packages

Both are small and local to the tests:

- **mmap:** release the loaded tensors and force collection before rewriting the path
  (`del state; gc.collect()`), or write the corrupted copy to a *different* file and point the
  loader at that. The second is more robust and platform-independent.
- **symlink:** skip with a clear reason when `os.symlink` raises `OSError` with `winerror 1314`,
  rather than failing — the guard itself is still worth exercising wherever symlinks can be
  created. A `pytest.skip` with the privilege reason keeps the signal honest.

These are not changed here. They belong to the packages' own branches and their own review, and
silently patching another experiment's negative tests from this branch is exactly the kind of
cross-package edit this programme's agent contract forbids.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY -m pytest experiments/dual28_ced_v1/tests experiments/full_width_transition_v1/tests \
              experiments/ced_native_loader_issue4_v1 experiments/cross_scale_remote_receipt_d82_v1 -q
```
