# Vendored code

`gpu_lease.py` is a VERBATIM copy of
`rineya-dataset-retention-records/hplat-rehearsal-20260921/gpu_lease.py`
(sha256 `049742734eea9bddd7aaf149d2da824c3d9eec1e2157fab6c6c586d033fda00a`, identical after copy).
It is the one lease implementation in the programme that (a) takes the lock with O_EXCL,
(b) treats an unreadable card as BUSY, (c) re-checks the card 3 s after writing, and
(d) never deletes a lock it does not own. Do not edit it here; if it needs a change, change the
source and re-vendor, so the sha keeps meaning something.
