# Resource-held validation queue

This queue preserves full scope. It does not claim any deferred test has passed,
authorize a model job, refill allocation slots or replace immutable directories.
RAM has repeatedly remained below the384MiB small-validation admission threshold,
and several earlier observations were below the256MiB host reserve. The latest
checkpoint observation is310,988,800bytes at2026-10-01T09:37:46.7360954Z; this is
an observation, not future resource admission. No listed worker is live; no
observation timeout is a reason to restart a worker. See
`RESOURCE_HOLD_CHECKPOINT_001.md` for the reviewed source versions and resume order.

| Work | Current state | Complete next operation |
|---|---|---|
| Admission v4 complete metadata fixture | Final held source pin and independent static review complete; full fixture/tests/guard faults NOT_RUN | In the separately authorized Hermes metadata runtime, run `check_metadata_contract_v4.py --emit-fixture`, then the complete `check_metadata_contract_v4.py` suite under its owned resource guard. Preserve all schema/node/edge/negative coverage. Reverify the exact final held source hashes before execution. |
| Blind-interface v2 precision repairs | Independent static review supports the two precision repairs;20 tests NOT_RUN | Run all retained9 tests,5 reviewer probes and6 repair groups using `run_repair_tests.py`. It creates write-once test output. No actor/model/row votes or interface enablement follows from test success. |
| Feature staging controller v3 | Final independent static review supports manifest/count repairs;11 mock tests and independent compile NOT_RUN | Run `python -B -m unittest -v test_metadata_only.py` in the v3 folder under the small-process resource gate. Process doubles must never launch a real feature operation. Complete runtime review remains required before a controller PASS; the static review cannot authorize staging. |
| Existing-control shortcut extraction | Exact implementation/controller reviews passed; `shortcut_receipts_001` contains a real resource-hold receipt with0workers | Use the reviewed controller v3 with exact review hashes and a new output folder such as `shortcut_receipts_002`. Retain001 unchanged. Recheck code/input identity and fresh resources; never treat the old resource observation as admission. |
| Prospective feature-control staging | Feature v3 reviewed; root staging-only approval prepared; no control worlds/bank created | After exact staging-controller review and fresh resource admission, stage one complete new bank. Independently verify every intent/world/source/role/seal before any control query. A complete bank is not execution, row admission or native qualification. |

Small metadata tests require one process at a time, a128MiB owned-process budget
and256MiB host reserve, with a fresh observation of at least384MiB. Admission v4
uses its explicit guarded128MiB ceiling after memory-efficient immutable payload
sharing and streaming; its owned watchdog must fail closed on resource/telemetry/
receipt faults and preserve unique per-attempt evidence. If that budget cannot
cover the complete work, retain the failure and review a new implementation.
No test subset is allowed to replace full validation.

Actual staging and extraction keep their independently specified4.375GiB start
gate,1GiB owned-worker cap,1.25GiB running host reserve and4GiB disk reserve.
Their controllers govern only their own processes. Do not kill applications,
modify the shared branch or launch competing native/model workers to make room.

On any resource hold, preserve the exact observation and known0starts. On a fatal
started phase, preserve exact errors and partial files and leave unavailable
counts UNKNOWN. Never equate a failed kill with exit or a nominal seal with a
successfully completed phase. On actual test failures, repair a new source version
and review the affected scope before retrying; do not silently rerun into old
directories.

Source-use/current-row joins, eligible dual reviews, exact release assembly,
native compatibility and behavioral acquisition/retention remain separate
requirements. Completing this queue cannot establish model intelligence gains.
