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
