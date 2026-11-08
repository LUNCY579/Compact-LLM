# Amendment 008: attempted-work accounting for E0

29 September 2026. The resumed E0 attempt v3 used the unchanged registered model,
config, source and containment. It stopped at the preload host-RAM gate, before
checkpoint hashing, tokenizer construction, model creation or model work. Its
original receipts remain immutable. This is the first blocked observation after
resumption; the prior seven-turn count is not reused as a resumed blocked audit.

Read-only review identified a separate future-failure accounting gap. Existing
load/update counters describe completed stages, and completed context profiles were
only attached at the final pass. A later failure could therefore leave partial
work underreported or omit earlier profiles. The current v3 no-model-work claim
is independently supported by its traceback, event order and bound source, not
merely by zero counters.

Register accounting version2 prospectively for future E0 attempts. Journal each
model-construction, checkpoint-load, initial base-checkpoint tensor-copy, forward,
backward, optimizer-step, checkpoint-save and reload attempt before invocation
and record Python returns or exceptions afterward. The tensor-copy operation
covers the 310 explicit serialized-checkpoint-to-base copies. Adapter, optimizer
and RNG restoration belong to the composite checkpoint-reload span; separate
restore_initial housekeeping is not individually counted. This journal records
selected operation boundaries, not every tensor transfer or CUDA operation.
Record submitted forward token positions, including
padding where present, and training-forward positions separately. These are
submission/return counters, not FLOP measurements or declarations of numerical
success. A hard kill may leave a pending call; missing completion cannot mean zero
work. Retain journal and partial completed context profiles on failure.

Count optimizer calls that returned before any later finite-state validation can
fail, consistently across ordinary and resume paths. Retain separate attempted
training input counts and legacy completed-update token counts with explicit
semantics. Enforce the existing maximum of five optimizer calls before dispatch.
The generated-token count already advances per emitted sample and is preserved.

The pass checker must validate version2 journal ordering, closure, counters and
cost/result consistency, alongside every existing numerical, model, source,
resource and containment gate. Failed historical receipts remain interpretable
without rewriting them into the new schema. A new accounting pass alone grants
no native qualification, E1 authorization or behavior claim.

Prepared E1 plans that bind the prior gate-checker source remain historical plans,
not current launch manifests. Before any native E1 execution, rebind the plan to
the qualified current source and runtime. This amendment changes no planned data
order, dose or optimizer schedule and does not reselect a candidate.

No model, data, context, sampler, RNG operation, input, mask, dtype, optimizer,
tolerance, resource floor or compute ceiling changes. Added telemetry overhead is
part of the subsequently measured runtime. Preserve all prior evidence; qualify
the changed source in a new run only after a real host-resource change permits
admission. No native retry is earned by this bookkeeping fix. Tests are bounded
CPU/synthetic failure controls, not pretrained results.
