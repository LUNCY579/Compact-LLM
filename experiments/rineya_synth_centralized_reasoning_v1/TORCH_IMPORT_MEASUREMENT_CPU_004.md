# Import-only effect measurement guard and raw replay

A separate measurement guard now derives direct and nested DLL additions from
raw observations, leaving expected effects explicitly unset. It retains original
source-byte, selected path/handle, scope, directory, event sequence, snapshot and
cleanup helpers. Unknown loads, unloads, duplicate handles, event loss and loads
outside active measurement intervals refuse. Required extension create/exec pairs,
bounded nesting and closed cleanup remain mandatory.

The independent replayer derives effects from the raw boundary snapshots and
events rather than accepting producer summaries. Results are
MEASUREMENT_RECORDED_UNQUALIFIED or preserved failure, with native qualification
and authority false. The original exact-effect validator remains unchanged and
rejects the separate raw kind. Future exact registration can use measured effects
only after a separately permitted run and independent terminal audit.

Six authored and Root controls pass. Four [independent Root probes](evidence/torch_import_measurement_reviewed_cpu_20261007/ROOT_INDEPENDENT_MEASUREMENT_REVIEW001.json)
pass in 0.292 seconds with selected source pins unchanged. They execute the real
guard and raw replay with inert provider/files, independently check direct/nested
conservation, and reject map/source/effect/authority rebindings and outside-interval
loads. Tests cover failed required pairs and partial cleanup; the first incorrect
test expectation for a poisoned enclosing boundary remains preserved.

[Preservation](evidence/torch_import_measurement_reviewed_cpu_20261007/ROOT_PRESERVATION_004.json)
binds twelve files and 79,552 bytes. Guard is
`55bcdff206a8574b6246e5f2c0c047ded4cac3735e1898dc1ec85c32fd01d8d1`;
replayer `ce8ea36cd113310d35bd09ee223f279bdbf5703b8c7d93ae143325bd9f95124f`.
Integration must externally select the measurement source descriptor before
execution; its self-read at import checks current-byte conservation, not initial
source authentication. Incomplete records without required raw clock/schema may
be semantically refused while the original raw bytes remain preserved.

The dedicated import-only worker, scope-specific gate and exact runtime/config/
registration are still being assembled. Native approval is absent. Torch import
may reach CUDA driver interfaces; this stage must stop before explicit user CUDA
initialization, tensor allocation, model/forward/backward/optimizer/decoder/scoring.
No native API, DLL, Torch, CUDA, model or training ran in these CPU controls.
