# CPU Windows process-containment qualification

Result: **11 passed, 0 failed, 0 skipped** in9.55 seconds. Evidence is in the exclusive
runs/supervision-probe-003/ directory. This is Windows supervisor apparatus
qualification only: zero model loads, zero v1 native entrypoint executions, zero
native worker time. All actual workers and descendants were short-lived sleeping
Python fixtures. No foreign process was terminated and no environment was changed.

Interpreter: C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe.
Recorded runtime: Python3.11.15 MSCv.1944 AMD64, Windows10.0.26200, psutil7.2.2,
pytest9.1.1. The runner used -B and disabled pytest's cache provider.

## What passed

1. 64-bit Windows ABI sizes and offsets: BasicLimits64, IOCounters48,
   ExtendedLimits144 and ThreadEntry28 bytes.
2. The campaign Job handle is non-inheritable and close is idempotent.
3. The venv launcher is created with CREATE_SUSPENDED before assignment. After a
   brief pre-assignment wait it has no descendants and the fixture has not run.
   Only then does WindowsJob.assign succeed and resume_suspended resume its sole
   initial thread. IsProcessInJob verifies the launcher, actual base interpreter,
   and actual child interpreter belong to the exact campaign Job.
4. Both kill-on-close and TerminateJobObject stop the actual worker and its actual
   descendant, including the venv redirector chain. An unrelated launcher and its
   actual interpreter remain alive.
5. Forced death of the actual interpreter owning the Job handle, without a
   cooperative cleanup call, stops the worker and descendant. An unrelated real
   sentinel remains alive.
6. resource_sample includes the worker's actual descendants, excludes the
   unrelated sentinel, and computes the RSS total from its recorded members.
7. native_entry.main rejects false containment and times out on an absent release
   before calling runpy.run_module. A positive CPU-fixture release preserves the
   exact v1 native-dev argument vector; runpy is mocked, so v1 never executes.
8. The real native_supervisor.run path receives a deliberately fabricated
   CPU-only admission/review fixture and a suspended no-model child. Assignment
   refusal kills the exact unassigned Popen handle, emits no release, removes its
   own lock and records the fixture elapsed time in a terminal ledger.
9. The real monitoring loop releases a successfully assigned CPU fixture tree,
   reaches a2-second hard deadline, kills all actual members, retains the unrelated
   sentinel, records RSS samples and measured elapsed time, writes a terminal
   SAFETY_STOP receipt, and removes its own lock. The child command substitution
   asserts that the real dispatcher requested CREATE_SUSPENDED.

The fabricated statuses used by the unit tests are explicitly marked
CPU_TEST_FIXTURE_NOT_A_NATIVE_PERMISSION / FABRICATED_FIXTURE_NOT_INDEPENDENT_REVIEW.
They live only beneath exclusive pytest fixture directories, use no source/model
assets and cannot constitute a scientific or independent-review receipt.

## Preserved failed attempt and correction

supervision-probe-001 preserves6 passes/4 failures. Three assertions assumed
Popen.pid was the actual Python interpreter; Windows venv python.exe redirects
to another PID. Those failures ended before verifying descendant termination.
The fourth assertion demanded a positive elapsed duration on an immediate
assignment refusal shorter than the host's monotonic clock resolution.

The redirector finding exposed a real containment risk in launching an ordinary
Popen before Job assignment: an already-spawned descendant may precede assignment.
The coordinator changed the dispatcher to create the launcher suspended, assign
it, resume its initial thread through Windows APIs, then publish the release.
The test now tracks real interpreter PIDs, verifies exact Job membership and kills
the actual owning parent for the death test. The measured refusal fixture uses a
bounded40ms delay so accounting is resolvable by the host clock.

supervision-probe-002 preserves10 passes before adding the actual dispatcher
hard-deadline path. supervision-probe-003 is the final11-test run. Earlier outputs
were not replaced. A post-run process snapshot found no remaining Python processes
matching the sleeping test fixtures.

## Source and evidence bindings

Hashes below were observed immediately after the final successful run.

| File | SHA256 |
|---|---|
| supervision.py |894e7ffec1d7863d686cb82d410309a3a100d4fdb392892a4b534b989d4a4fcc|
| native_entry.py |ab5b60f0ad72f719367490737d1c71173e5384db3a31246ca2ed751557c6725c|
| native_supervisor.py |482e0876c5393a31620e02b9cc6b573d18c350977dd00faf82ee49061c5b6f16|
| tests/test_supervision.py |146acab920cfd3e9eaa0033a4c12a4f8bbe64efe22e55e530496430bccde3199|
| runs/supervision-probe-003/junit.xml |358ab24c0c7689ad24d45e768bc8845c692135cb0f75d79ef2f114be0828c6b4|
| runs/supervision-probe-003/pytest.log |7c39f6b72b2562a088361fbe6989f49c10db7fc27cc82c1cafdb8ce95f3860be|

Reproduction from the workspace root, using a new output directory:

    & 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider experiments/rineya_taps_adaptive_recurrence_campaign_v2/tests/test_supervision.py --basetemp '<exclusive-campaign-run>\tmp' --junitxml '<exclusive-campaign-run>\junit.xml'

## Boundaries and next gate

These tests do not execute the reviewed native admission, native v1 preflight,
model loader, full registered matrix, model-memory maximum shape or a real hung
model forward. The watchdog test proves termination of the tested owned CPU tree
through the actual dispatcher loop; it is not a kernel/driver guarantee.

The test author does not issue independent-review PASS. The reviewer must inspect
the resulting source and tests, bind the actual dispatch admission and source
hashes, and resolve outstanding issues before the coordinator admits native work.
Any substantive source change invalidates the corresponding hashes and requires
affected tests to be rerun in another exclusive directory.

