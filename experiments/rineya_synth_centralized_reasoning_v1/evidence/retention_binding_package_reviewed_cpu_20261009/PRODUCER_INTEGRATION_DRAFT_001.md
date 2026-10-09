# Checkpoint retention and GD binding preparation

This external preparation addresses an accepted producer requirement. It grants
no execution, model qualification, GD admission or capability claim. The SYNTH
owner retains sole responsibility for source integration and native work.

The existing engine saves and publishes only at updates 9, 18 and 27. The native
backend also limits resumable snapshots to those boundaries. Preserve that
scientific resume protocol while adding a separate retained archive after every
completed update 1–27 for each trained arm. No retained archive may become a
resumable checkpoint merely because its update is a multiple of nine.

The owner is staging the following separate interface:

- `snapshot_retained_update(binding, *, producer_policy_sha256)`
- `save_retained_update(path, binding, *, producer_policy_sha256)`
- Archive kind `SYNTH_STUDY_RETAINED_UPDATE_V1`, with `resumable: false`.
- Record kind `SYNTH_STUDY_RETAINED_UPDATE_RECORD_V1`, joining the completed
  binding, exact backend descriptor, producer policy and posted save costs.
- Entry references for the record publication and any GD parity request.

These names are a draft until the owner freezes the implementation, policy,
configuration, approval and review hashes. Register the separate save, path,
verification and publication costs prospectively. On a failure, preserve its
attempt and earlier archives, refuse the next update, and do not roll back costs
to a previously published scientific checkpoint. Completion requires all 27
archives and their publication references; resume checks must inspect the
retained chain and bytes rather than relying on asserted counts.

The external validator prepares structural joins only. Its frozen schedule must
contain exact source/runtime/recipe/data/base/tokenizer identities, producer
kind/schema, all 27 cumulative occurrence/token/order records and the exact LR
sequence. Whole-run token totals and three chunk hashes cannot supply those
per-update records. Missing bindings remain pending and must fail validation.
Import-observation authority or runtime qualification cannot qualify a STUDY run.

GD receives original standard-PEFT exports at steps 9 and 27, after each complete
export has returned. Preserve the distinct scientific checkpoint references,
base/tokenizer inventories, export payloads, post-save receipts, full consumed
prompt occurrences and source/filter/unused/replay lineage. Bind separate strict
load and producer/GD logit/greedy parity evidence under the reviewed GD contracts.
An export request, descriptor or boolean cannot establish native parity.

Before treatment execution, the owner must integrate and qualify the producer
hook, snapshot restoration, failure handling, completed trajectory consumer and
actual standard-PEFT export/parity path. This preparation does not fill those
native evidence slots or relax any resource, numerical or protected-bank guard.
