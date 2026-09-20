# High-dose LoRA: blocked on host RAM

DOSEINT-001 completed with FAILED_PREDICTION. Its trained checkpoint, four scalar
arms, two-bank measurements and exact paired analysis remain authoritative. This
new phase tests one declared FP32 tail20 rank16 regime at peak LR1e-4 against a
native scalar at the same actual dense-update norm. Read PREREG.md and
AMENDMENT_STORAGE.md before any model work.

Records root:
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/highdose-lora001-20260920`.
The code branch is `research/retention-dose-parameterization-v1` in
`ChatBubble-dose-param`. The pinned native pilot checkout is5b4c22d. Do not change
the pilot, source assets, pinned venvs, shared GPU lease or resource guards.

Current state: input/code/source gates and CPU mechanics tests passed. No native
LoRA model has run. Both the initial900s prelaunch wait and wait-002 ended
WAIT_EXPIRED_NO_WORKER. Wait-002 explicitly records controller_launched=false
and worker_launched=false; no journal or dispatch reservation exists and the
grant is unspent. The goal is blocked pending a host-memory change. A supervised runtime-only observation
confirmed the actual native train guard refuses roughly5.55GiB available against
its6GiB requirement after imports. It loaded no source model or tensors.

Once the required host RAM is available, a fresh bounded wait can use:

```powershell
& 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe' `
 'C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v1/launch.py' `
 --wait-id wait-003
```

Every wait ID is create-only. Inspect existing handles/receipts first. The helper
refuses any new dispatch if a native journal or NATIVE_DISPATCH_RESERVATION.json
already exists. The create-only reservation is shared by all wait IDs and is
preserved after a launch error; never delete it to retry automatically. Inspect
its owner and authoritative native receipts. The launcher checks the full2700s
native owned-work reserve before readiness and again immediately before dispatch.
It distinguishes starting the controller from proof of a model-worker launch.
Each subsequent export, compression and probe has its own deadline admission;
the complete battery is not guaranteed to fit, and partial outputs are preserved.
The existing session deadline is2026-09-20T09:39:05UTC. If it can no longer admit
the native reserve, preserve this envelope and register a fresh bounded session
prospectively before launching; do not silently extend its deadline.
An expired observation
is not a stopped worker; this first wait is authoritatively terminal and explicitly
records that no worker was launched. New waits retain the existing session deadline
and still require7GiB before dispatch; the child independently enforces6GiB after
runtime imports. Never lower these gates to manufacture a launch.

After a genuinely successful qualified training run, use:

```powershell
& 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe' `
 'C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_highdose_lora_v1/posttrain.py'
```

This checks the successful supervisor/result, builds LoRA under a600s CPU watchdog,
verifies its canonical merge and norm record, transparently compresses the immutable
weight file, and checks the second-export space gate. It then builds the exact norm
control, verifies actual0.5% norm tolerance, and runs eight sequential probes. LoRA
uses BUILD_GATE_LORA_FINAL; the scalar uses BUILD_GATE_NATIVE_NORM. Wrong gates fail
because these arms have different training plans/checkpoint identities.

The scalar coefficient is frozen before inference. Beta outside[0,2] is blocked,
never clipped/searched. One total inference retry is prospectively allowed after a
verified failure; it needs a fresh attempt and a recovery completion receipt. This
runner does not automatically retry or delete any failed output.

After eight valid legs, run report.py --root <records-root>. The report requires
all provenance and actual supervision/budget custody before exposing endpoints.
It applies the cross-training20-view floor, fixed bands and two declared Holm
families. A numerical engineering pass is not equivalence or release permission.

Storage audit:664 prior completed model/checkpoint files were losslessly compressed
and rehashed, with every logical byte preserved. Compressed safetensors mmap/readback
also passed. Source weights were untouched. There is about18.2GiB physical free
space. A sequential, hash-verified export/compression plan retains the14GiB floor;
if the second-export gate fails, preserve the first arm and stop before scoring.

The independent binding-signal generator/compiled prototype under
`binding-signal-prep-20260920/candidate-v1` is preparation for user priority4, not
part of this LoRA experiment. It contains no reused sealed cases, but its external
exclusion audit and training/evaluation authorization remain pending.
