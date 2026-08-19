# Recovery Runbook

## Revision conflict

1. Reload the run and full active state.
2. Do not replay a stale patch with a new revision blindly.
3. Reconstruct the intended transition against current state.
4. Submit a new idempotency key.

## Model structured-output failure

1. Preserve raw response in a bounded trace.
2. Attempt at most one schema-focused repair in a future configured policy.
3. Fall back to abstention or deterministic direct mode.
4. Add the failure to the structured-output regression set.

## Search/fetch failure

1. Record the failed URL and typed error.
2. Continue with independent sources if possible.
3. Reformulate the unresolved information gap.
4. Abstain when the search budget ends.

## Incorrect deletion

1. Restore the evicted record or pre-transition checkpoint.
2. Record deletion regret.
3. Add the trajectory to RSF regression data.
4. Strengthen dependency/future-utility rules before retraining.
