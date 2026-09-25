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
