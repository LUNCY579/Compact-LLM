# Import009: actual GPU admission denial

The corrected parent entry and metadata checks reached original resource
admission. The first pinned NVML observation found1,862,463,488 bytes free
(about1.73GiB), below the unchanged4GiB floor. The supervisor returned
`ADMISSION_DENIED` with no worker attempted, launched or resumed. RAM and disk
checks passed. No Torch import, tensor, model or training result exists.

[Preserved actual evidence](evidence/import009_gpu_admission_denied_20261009/ROOT_PRESERVATION.json)
includes the supervisor/anchor receipts, original lease, dispatch logs/exit2,
approval and exact packets, refreshed byte evidence, independent review,
qualifier replay and196 source snapshots. The postrun CLI reports
`IMPORT_OBSERVATION_NOT_RESUMED_FAILURE_RETAINED` with no semantic terminal audit.
Sampler/telemetry cleanup and frozen partial output are recorded.

A later read-only `nvidia-smi` observation still reported1779MiB free. Do not
repeat expensive unchanged admission or weaken the floor. Continue when GPU
availability changes, with fresh live admission and a new registered output;
do not stop unrelated jobs. Standing authorization remains active. The complete
M0/M1/M2 transfer/retention comparison is still pending.
