# OSIDE-001 attempt log

Every launch of `run_o_side.py` is recorded here: start time, how it was launched, exit status, which score
files it completed, and every move to `scores_failed/` or partial-build removal the driver logged. The only
allowed rerun is after a harness failure (crash, non-finite NLL, canary FAIL or provenance failure; PREREG.md §0
as amended by AMENDMENT_1.md). A rerun resumes after the last COMPLETE score file (it parses and passes the
provenance check) and never rescores a complete file still in scores/; the only files rescored are ones the
driver moved to scores_failed/ (an incomplete file, or every file of a stage whose canary failed). A
final-analysis HARNESS_FAILURE ends the attempt; any move after it is an operator decision recorded here.

| # | started (UTC) | launcher | exit | completed score files | note |
|---|---|---|---|---|---|
| 1 | 2026-09-25 23:33:20 | WMI (wmi_runner.py; log RUN_O_SIDE_LOG.txt, rc RUN_O_SIDE_RC.json) | 1 (after 23 s) | none | Preflight refused before the lease: `NOT STARTED: RAM 3.71 GiB / disk 3.0 GiB` (registered minimum 6 GiB disk). After the ~19:40Z host reboot, commit ~22.8 GB grew the system-managed pagefile to 19.25 GiB. No lease, no score, no build, no PREFLIGHT.json; nothing read. |
| 2 | 2026-09-26 00:21:13 | WMI (log RUN_O_SIDE_LOG_2.txt, rc RUN_O_SIDE_RC_2.json) | none: killed, no rc file | BASE, OPALLO, OPALLO-BF16, SPARSE | Preflight PASS; stage-1 canary PASS (BASE and OPALLO max abs diff vs O_REPLY_NLL.json 0.0). Killed inside ADD-100 (row ~2000) by an operator-initiated host restart at 00:53:53Z (System log: User32 1074 from the Start menu; nvlddmkm 153 one second later during shutdown). A crash-class harness failure. The lease was left stale; its owner (this lane) released it at 05:07Z after verifying the holder pid 228212 and runner pid 220040 dead and no live process carrying the label. All four score files pass the completeness/provenance check; no partial files. |
| 3 | 2026-09-26 05:07:05 | WMI (log RUN_O_SIDE_LOG_3.txt, rc RUN_O_SIDE_RC_3.json) | 0 (2814 s) | ADD-100, ADD-O50, H1, H1c, SEED23, SEED29, SEED41 (+ the 4 complete files kept from attempt 2) | Preflight PASS (PREFLIGHT_rerun_20260926T000734.json). Stage-1 canary re-run PASS (0.0); the 4 complete files skipped, none rescored; stage-2 canary PASS (BASE/OPALLO/SPARSE/ADD-100 max abs diff 0.0; provenance PASS on 5). The rebuilt arms matched their registered hashes; each built arm was checked before its weights were deleted. Lease released; analysis status VERDICT. |
