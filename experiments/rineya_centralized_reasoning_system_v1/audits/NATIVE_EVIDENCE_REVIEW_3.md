# CRS-001 native attempt 004 and qualification quarantine

Recorded 2026-10-02. Reviewer `/root/scientific_audit`; builder
`/root/runtime_audit`; coordinator `/root`. Disposition:
**FAILED_IDENTITY_REFUSAL_CAUSE_UNRESOLVED; WHOLE_PATH_QUALIFICATION_QUARANTINED
AT THREE REVIEW/REPAIR ROUNDS.** The approved native source remains preserved.
No fourth repair round, model retry, executable whitelist expansion or numerical
qualification is authorized by this review.

This independent review used small retained JSON/JSONL/log/source files and
stdlib only. No framework/model/tokenizer imports, weights, native launches,
protected labels or new research cases were used by the reviewer. Reviewers and
builders share model lineage, workspace privileges and evidence; this is a fresh
artifact review, not independent physical observation or OS isolation.

## Raw provenance and admission

Read all files at `evidence/native_d2_attempt_004/` and checked all11 raw-file
hashes in SUMMARY.json. They match. The original isolated-worktree run is
`results/d2_1fd862b8_epoch001_rev3_statechange2`, as identified by the retained
launch command; the copied primary evidence does not change its source identity.
The source preflight is clean commit
`1fd862b863f30ac772f54b19e7ab337d68a4215b`, native SHA256
`8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3`,
contract SHA256
`d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a`.
Job `CRS-D2-004-1fd862b8-d79f3b6a`, owner
`CRS-D2-4423e70e39f04efd9b3904bdc00bba88`, fence4.

LAUNCH_INTENT.json reports READY_DIAGNOSTIC_ONLY. Preflight host/GPU free were
5.826648712158203/6.5849609375 GiB; second observation
5.908515930175781/6.5849609375 GiB. Both blocking GPU PID lists are empty. Unity
PID683892 from attempt003 is absent from these retained GPU inventories. This
absence corroborates changed admission conditions; it is not a fresh reviewer
observation of that process exiting or proof of subsequent host stability.

## What was and was not observed

The suspended venv launcher PID701704, creation1790923016.4936807, and CPython
child PID704348, creation1790923017.341483, were exact owned-Job members before
GO. Query-only access4/noninheritable transfer targeted the child's exact PID/time.
The child's containment receipt records explicit Job flags8192/0x2000, immediate
NULL flags12288/0x3000, no breakaway and query-handle closure before framework.

The complete EVENTS.jsonl contains only `framework_import_attempt` at
1790923020.583944. No recorded model construction, checkpoint load, tensor copy,
forward attempt/return, numerical measurement, diagnostic result or runtime/model
receipt exists. Stdout and stderr are empty. All four planned model forward cases
remain unattempted in the durable evidence. This states recorded counters, not
zero worker cost or zero resource use during import.

All8 successful WATCHDOG.jsonl entries have reason=null and show only the two
registered Python members. Peak logged owned-tree RSS35131392 bytes matches the
supervisor's peak. The supervisor fails with
`NativeBlocked: unregistered owned runtime executable`, exit2, after
4.219000000004598 seconds. The exact failing PID, executable path and actual hash
are **not recorded**. No failing member inventory was retained before that
exception.

The frozen source's `job_member_witness` at `native.py:456` checks
`expected and sha(executable) == expected` at `:477` using that one generic error.
The message therefore cannot distinguish an absent registered path from a hash
mismatch; it does not identify a transient helper, process creator or framework
cause. The most recent successful Python observations do not identify the next
failed sample. Candidate explanations such as a transient helper, path/identity
race or binary drift remain hypotheses only. Do not add a guessed executable to
the registry or describe this as a diagnosed cause.

The guard raises before the supervisor writes that iteration's watchdog record.
Thus successful earlier inventories cannot supply the missing offending identity.
An operation refuses safely while its cause remains unmeasured; neither the
framework-import event nor the exception supplies a model numerical result.

## Exit, cost and retained evidence bindings

SUPERVISOR_RESULT.json reports FAILED, dispatched=true, known creation outcome,
reader/training=false, and exact owned tree empty at1790923020.7004237. The
outcome-observation receipt matches owner/fence, launcher PID/time/command hash
`2a441447dcc9ae7c428b8887645f089984a0a050f466f8d816ff9355d1acf825`,
and charges4.219000000004598 seconds with process/owned-tree exit evidence.
Previous43.14 plus4.219 equals retained47.359 seconds at display precision.
Summary epoch7200 and rolling14400 limits remain unchanged, and it reports no
active native jobs. That is the retained registry snapshot, not a reviewer live
host/process inspection or budget reset.

Summary SHA256 `851f275f0666eb771b07b6653f6f4b1d2ff430687c736d7930c1d2e4f1b16500`.
All11 referenced raw bytes matched. Selected bindings:

| Evidence | SHA256 |
| --- | --- |
| LAUNCH_INTENT.json | a229086b7d1b1e4d9e3d240ca128bb13bde8de7a946abb4d11a54f47f7878d2b |
| EVENTS.jsonl | 65482fb6aabb17398ba27965b34e86f63cca00af2ebfc1580afcad62731e7876 |
| WATCHDOG.jsonl | 021fda0be2a43de51ebdf5d832a9d1ba8408aed7c0a8c4c650b69a8822f1a399 |
| SUPERVISOR_RESULT.json | 469b8ba8d075e6b7d0de113bd29198233aa0969223bc3c7f55dfc4f68e931230 |
| CONTAINMENT_RECEIPT.json | 84dfaa45c81359f1351447fc8c860bc275f40c58e54b0ee0e946c5b52b33877a |
| TREE_CONTAINMENT.json | a5c0ca7bbd32fe8bc135b3e54ef3e8de49a62f1f0b64117d361d8985f9f4ebab |
| outcome-observation receipt | c4a3d59d3341390e7d806b3ccb4cc78cab9d8f678c38dc04521c5ccee722d527 |

## Bounded resolving experiment, not executed

Prospective contract: `contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_DRAFT_V1.json`.
It is a DRAFT requiring root and independent operations review of implementation,
source/contract/receipt identities and containment before any probe. It cannot
launch native D2, a model or framework. Keep the approved native.py byte-for-byte
unchanged and its registry unexpanded.

Use one exact-Job contained stdlib/psutil worker, the same pinned Python launcher
and CPython runtime, and reviewed query-handle/membership/exit primitives. A
separate observer must record every complete exact-Job PID inventory and all
resolvable member creation times, executable paths, actual executable hashes,
expected registry entries, and independent explicit Job membership **before**
calling the unchanged strict member/RSS guard. Never filter unknown members out
of the observational inventory. Retain query errors/vanished members and the
phase/guard result; observations grant no executable permission.

Invoke the existing `committed_snapshot`, `verify_sources` and `runtime_metadata`
source-check paths inside the contained child, with a fixed phase schedule and
begin/end receipts. The source-check part of `guard` is thereby exercised without
calling its Torch/CUDA-dependent part. Do not call native preflight/worker or
verify_assets, which would hash model binaries or enter model execution. Source
checks can spawn subprocesses; identify their actual observations rather than
guessing them. A20ms observational sampler and exact Job creation/exit notifications
are proposed alongside the unchanged100ms strict guard. Missing short-lived
identities remain explicit coverage gaps; sampling cannot be claimed exhaustive.

One run at most60 seconds,8 owned live processes,16MiB logs, unchanged native
RAM/reserve/admission/disk floors, one host-wide worker lease and conservative
cost accounting. Only the probe's exact owned tree may be drained/closed; unrelated
desktop jobs remain untouched. No retry, loop until pass, tolerance/resource
relaxation or whitelist learning is part of the draft.

Outcomes are descriptive: either an offending identity/refusal is observed,
all specified no-model paths complete under sampled known identities, or coverage
is unavailable/inconclusive. A successful source-check probe cannot prove a
framework-import process inventory or native whole-path qualification. If it
does not reproduce the refusal, the original cause remains unresolved. Any later
changed operational contract needs a new explicitly bounded resolving milestone
and fresh review; preserve the original three-round history and all failed costs.

Historical bulk FAILED and .03125/.01 limits remain. Prior fixed131 worker-only
raw equality observations remain scoped evidence whose whole path failed. Reader
576, bulk capture, training/RL, efficacy, acquisition/retention and publication
gates remain closed. The track and open questions are preserved; quarantine
applies to the unsupported qualification claim, not deletion of CRS-001.
