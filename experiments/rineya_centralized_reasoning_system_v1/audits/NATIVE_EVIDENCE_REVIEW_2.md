# CRS-001 native attempt 003 evidence review

Recorded 2026-10-02. Reviewer `/root/scientific_audit`; runtime builder
`/root/runtime_audit`; coordinator `/root`. Disposition:
**FAILED_ENVIRONMENTAL_ABORT_NO_NATIVE_NUMERICAL_RESULT.** The retained abort,
owned-tree exit and charged cost are supported by the raw records. This review
does not qualify the model, reader, bulk capture or training and is not another
native source repair round.

Only small JSON/JSONL/log/source receipts were read with stdlib. No framework,
model, weights, tokenizer, native worker, new data or protected labels were used
by the reviewer. Same-model lineage, shared workspace/evidence and OS privileges
limit review independence. I examined raw receipts before deriving the conclusion
and checked all 11 raw-file hashes declared by the coordinator's summary.

## Provenance and admission

Evidence directory: `evidence/native_d2_attempt_003/`. Its original run directory
is the isolated worktree's `results/d2_1fd862b8_epoch001_rev3`; the copied bundle
does not imply a run from the primary checkout's newer commit.

- Executed source identity from retained clean preflight:
  `1fd862b863f30ac772f54b19e7ab337d68a4215b`.
- Registered native SHA256:
  `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3`.
- Registered contract SHA256:
  `d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a`.
- Job `CRS-D2-003-1fd862b8-d79f3b6a`, owner
  `CRS-D2-c5429989f4194432956911a778284c20`, fence3.
- Launch command is the native `--worker` namespace and exact reviewed contract,
  not the reader namespace or a changed numerical contract.

`LAUNCH_INTENT.json` contains READY_DIAGNOSTIC_ONLY preflight, clean source,
available host RAM 6.3563690185546875 GiB and GPU free 6.9599609375 GiB, with no
competing/unresolved/unclassified GPU PID. Its second observation reports
6.413074493408203 GiB host and 6.9599609375 GiB GPU with the same empty blocking
lists. These are the actual retained launch observations; a separately reported
earlier 6.289 GiB observation is not substituted for them. Admission at those
times does not guarantee an unchanged host after launch.

## Observed sequence and counters

`LAUNCH_ATTEMPT.json` records dispatch intent at 1790922720.1410499 and command
hash `917a1aa6abf3be6744818e0e3885ad05fab08ff57e1a3aae522e036667daa267`.
The owned venv launcher is PID694020, creation time1790922720.1594937. The exact
CPython child is PID693968, creation time1790922720.4209812. The suspended launcher
was the only member before resume; both were members of the exact supervisor
Job before GO. The transferred duplicate was query-only access4, noninheritable,
bound to the child PID/time. `CONTAINMENT_RECEIPT.json` reports explicit Job flags
8192/0x2000, immediate-NULL flags12288/0x3000, exact membership, no breakaway,
kill-on-close, and query-handle closure before framework import.

The **entire** `EVENTS.jsonl` contains one event:
`framework_import_attempt` at1790922722.1867964. There are no recorded construction,
checkpoint-load, tensor-copy, forward attempt/return or measurement events, no
DIAGNOSTIC_RESULT.json, and no runtime/model receipt. Both stdout/stderr are empty.
Thus zero model loads and zero forwards were *observed in the durable journal*;
the four planned forward cases remain unattempted. This is not a claim of zero
elapsed native-worker cost or proof that framework import consumed no GPU/CPU
resources. There are no measurements to classify numerically.

All49 `WATCHDOG.jsonl` entries were parsed. The final entry at elapsed
7.437000000005355 seconds records `GPU_PROCESS_CONFLICT`: GPU sample
1790922727.3219452 contains unclassified GPU PID683892, named
`C:\Program Files\Unity\Hub\Editor\6000.6.2f1\Editor\Unity.exe`. It is absent
from the two admission GPU inventories and present in the final GPU inventory.
The data establish a newly observed GPU participant, not necessarily that the
underlying Unity process was newly created. At abort, GPU free was6.94140625 GiB,
host reserve4.595073699951172 GiB, and peak observed owned-tree RSS618078208 bytes.
No numerical gate failure, RAM-floor failure or CUDA fit conclusion is supported.

## Outcome and retained cost

`SUPERVISOR_RESULT.json` is FAILED/GPU_PROCESS_CONFLICT, exit2, launch attempted
and worker dispatched, creation outcome known, reader/training flags false.
The exact owned tree was observed empty at1790922727.5713382, with no remaining
owned PIDs. The independent outcome-observation receipt matches this owner/fence,
launcher's PID/time/command hash and exit proof, and charges
7.469000000004598 seconds. No receipt authorizes killing the unrelated Unity PID.

The summary's charged total43.140000000003056 seconds agrees, at floating-point
display precision, with retained previous35.671 seconds plus this7.469-second
failed attempt. Epoch7200 and rolling24h14400 limits are unchanged, with unknown
reservations conservatively charged in full. Summary reports no active native
jobs; that is its recorded registry snapshot, not a new live host audit performed
by this reviewer. No budget reset or automatic retry is justified by this abort.

## Byte bindings and scope

Summary SHA256: `70a169331c56065c6b18b428e59fe08f515553603f778d2d56ae51bbc7bb4776`.
All11 entries in its `files` map matched freshly computed hashes of the copied
raw bytes. Key bindings:

| Evidence | SHA256 |
| --- | --- |
| LAUNCH_INTENT.json | 5e0c83c086923e57e9dc51a084cd96c19c56dcf347d0cb3c7c0e1e5c13e215b5 |
| EVENTS.jsonl | b809f7693b149e3607267143b361a805af9d3fa44a74297a396a1d0a541c74bd |
| WATCHDOG.jsonl | c8fc179a3099191a3e439e2a08ef7434f8b2ae901f37e0178919a11dc5cba41d |
| SUPERVISOR_RESULT.json | f63f39e06f3a3941de2a7f919cfff0b073cecf3ee0831fef10918f57054e2667 |
| CONTAINMENT_RECEIPT.json | 9fbda1aeacb4e70d96a8a42e16024cb6d4c30ee9689e9740d691657128bde3c9 |
| TREE_CONTAINMENT.json | e50ae7a8f324e06c85488834fcee3b514440f2c1b6349d5450fd3b135f80d96d |
| outcome-observation receipt | 129e0a4a43246fe1d087ca24bb3f360d7a82f29113c2608986dc4044ae162ffc |

This abort does not invalidate or repair earlier scoped fixed131 worker evidence;
that earlier whole path remains FAILED and its cost remains retained. Attempt003
adds no bitwise-causality, prefix/bulk equivalence, GEMM mechanism, reader576,
language accuracy, acquisition, retention or training observation. Preserve the
historical bulk failure and .03125/.01 limits. The environmental gate worked by
refusing an unregistered GPU participant; any future attempt requires new live
admission, unchanged reviewed source/contract and retained aggregate budgets.
