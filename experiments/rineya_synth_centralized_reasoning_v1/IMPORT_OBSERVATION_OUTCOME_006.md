# Import006: supervisor entry failure before worker launch

The owner approved proposal005 and automatic continuation. Separate approved
dispatch006 was committed at `ffd583f95b3d2b4dc9186a904d4b01bca0d7fe41` with the
unchanged implementation8dcc, source/runtime maps, scope and resource limits.
Root and independent reviewers checked the exact packets, human receipt and
dispatch grant before invoking the registered postrun CLI.

The [actual preserved result](evidence/import006_no_resume_failure_20261009/ROOT_PRESERVATION.json)
is `ADMISSION_DENIED`, with `ModuleNotFoundError: No module named 'psutil'`.
Supervisor preparation encountered the failure while checking historical lease
births, before a new worker attempt or resource admission sample. No Torch
import, user CUDA context, tensor, model, optimizer or scoring result exists.

The original supervisor receipt records zero admission observations, no attempted,
launched or resumed worker, terminal sampler/GPU cleanup, and frozen partial
output. The postrun CLI returns `IMPORT_OBSERVATION_NOT_RESUMED_FAILURE_RETAINED`
with no semantic terminal audit because no worker ran. Exit code2 is retained,
together with dispatcher logs, exact grant/approval/packet bytes and196 source
snapshots. This is an actual failed entry attempt, not native readiness or a
resource-admission failure.

The next fix qualifies the parent's psutil dependency before original lease
checks under the no-site invocation, preserving child bootstrap isolation,
original resource checks and cleanup. Preserve this attempt and use new output
directories/registration for its reviewed successor. Standing authorization
allows that necessary correction and continuation without another question.
The full M0/M1/M2 transfer and prior-correct preservation comparison is still open.
