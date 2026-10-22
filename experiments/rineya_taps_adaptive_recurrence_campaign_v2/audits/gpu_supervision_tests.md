# GPU successor parent supervision — CPU-only tests

Result: **19 passed, 0 failed, 0 skipped** in3.94 seconds. Exclusive evidence:
runs/gpu-supervision-probe-001/junit.xml and pytest.log. This qualifies the tested
GPU-parent admission and containment apparatus only. It does not qualify a model,
GPU runtime, numerical lane, maximum shape or scientific matrix.

No gpu_native import/execution, model load, actual CUDA model operation or real
gpu_inventory/nvidia-smi query occurred. Every worker was an owned CPU Python
sleeping fixture. Native model-worker charge:0 seconds. No environment changed,
and no unrelated process was terminated.

The earlier11 CPU-supervisor tests did not automatically qualify this new parent.
The new suite executes actual gpu_supervisor.run branches with fake resource
inventory and explicit fabricated CPU test admission/review records.

## Checks that passed

Twelve parameterized no-launch cases verify that the real GPU dispatcher refuses
before Popen, output creation, ledger mutation or lock replacement:

- Wrong review status and review/admission digest mismatch.
- Bound dispatch-source mutation after the fixture review.
- Host RAM at7GiB minus1 byte.
- GPU free memory at6GiB minus1 byte.
- GPU UUID mismatch and driver-version mismatch.
- Source-path mismatch and interpreter mismatch.
- Admission status not ready.
- Existing unknown lock; its bytes remain unchanged.
- Output outside the fixture campaign.

The GPU resource boundary is new:7GiB parent RAM includes the registered6GiB
worker floor plus1GiB import reserve. A successful CPU fixture test uses exactly
7GiB parent RAM and6GiB fake GPU free memory; the low-boundary cases refuse.
Fake inventory values do not assert actual GPU identity or capacity.

One separate regression invokes the actual CPU native_supervisor.run admission
path with8.4GiB minus1 byte. It refuses before launch/lock creation, retaining
the historical7.4GiB floor plus1GiB parent import reserve. The original CPU worker
guard was not lowered.

Two GPU-bootstrap negative cases prove false containment and an absent release
refuse before runpy.run_module. Two positive cases preserve the exact explicit
GPU successor module/argument vector for qualification and dev stages, with
runpy mocked. They do not execute the successor or permit a CPU fallback.

The assignment-refusal probe uses a real suspended Windows launcher and an
actual gpu_supervisor.run branch. The deliberately refusing Job assignment
never resumes the launcher. Cleanup kills the exact unassigned child handle,
publishes no release, removes its own lock and writes the measured elapsed
fixture cost in a terminal ledger.

The final probe uses the real GPU parent monitoring loop and shared WindowsJob.
It asserts the real dispatcher requests CREATE_SUSPENDED, then substitutes a
CPU bootstrap waiting on the release file. After assignment/resume/release it
spawns an actual interpreter and descendant through the Windows venv launcher.
The parent's2-second hard deadline kills the actual worker and descendant;
an unrelated real sentinel remains alive. Terminal status, stop reason,
elapsed worker cost, RSS samples, source scope and ledger start/completion
events are checked. The fixture's exact-boundary fake resources satisfy admission.

The GPU tests reuse CPU-only process helpers from tests/test_supervision.py.
The shared WindowsJob implementation is bound below; exact Job membership,
kill-on-close and actual-parent-death coverage remain in its prior11-test
qualification. The new test does not pretend to independently retest all
Windows APIs, but proves this GPU dispatcher uses the suspended containment
path and terminates its own tested actual child tree.

## Evidence and source bindings

Interpreter used:
C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe with -B,
pytest cache disabled, exclusive campaign-local basetemp. Source/artifact hashes
were observed immediately after the successful run.

| File | SHA256 |
|---|---|
| gpu_supervisor.py |90e2f1eb4c8c8f29fd85a27a239a084ee6968fd8e41d40dcecd39a9a254df5fa|
| gpu_entry.py |9114dc53822656509097a9d5a2db585f02e27791ef06e4fafae104b421b581e2|
| native_supervisor.py |860495d4fa7a8c69c82c392a1ce14d28374fc8fe01f2a84078d780efc4787575|
| supervision.py |894e7ffec1d7863d686cb82d410309a3a100d4fdb392892a4b534b989d4a4fcc|
| tests/test_supervision.py |146acab920cfd3e9eaa0033a4c12a4f8bbe64efe22e55e530496430bccde3199|
| tests/test_gpu_supervision.py |b9cf838a25bed3e8d8f872fd291cdf0368a21067a7049ba471265e1b783f3043|
| gpu-supervision-probe-001/junit.xml |aa019ec551390749b030488d16e5c4aabf76fe544ce3cfb3d7a19e03a3b833e9|
| gpu-supervision-probe-001/pytest.log |47b91d926ef43b68338508fed9f0cbe5bfbbd1ad5535f1ae9d3a6e736067b579|

A post-run process snapshot found no remaining Python sleeping fixtures.

Reproduction from the workspace root, with a new exclusive output directory:

    & 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider experiments/rineya_taps_adaptive_recurrence_campaign_v2/tests/test_gpu_supervision.py --basetemp '<exclusive-campaign-run>\tmp' --junitxml '<exclusive-campaign-run>\junit.xml'

## Claim limits and required next gate

Resource inventory and admission/review records in these tests are explicitly
CPU_TEST_FIXTURE_NOT_NATIVE_PERMISSION / FABRICATED_FIXTURE_NOT_INDEPENDENT_REVIEW.
They are confined to the pytest fixture tree. Fields named native_worker_seconds
inside those fixture ledgers record CPU fixture elapsed time, and do not consume
or establish a campaign native model run. The campaign's real ledgers/review
records were never edited by this suite.

These tests do not demonstrate actual GPU worker preload success, CUDA numerical
safety, maximum-shape fit, model weight preservation, GPU throughput or model
efficacy. They also do not certify every possible runtime resource race or
kernel/driver hang.

The independent third agent must inspect the new GPU parent/bootstrap/tests,
bind actual registered admission/source hashes and resolve review findings
before native dispatch. The test author does not issue independent-review PASS.
Source changes require affected checks to be rerun in another exclusive directory.

