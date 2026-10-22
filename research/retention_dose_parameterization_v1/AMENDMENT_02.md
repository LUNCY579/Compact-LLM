# DOSEINT-001 Amendment 2: one preserved pre-load refusal

Before any model load, qualification call, generated token or optimizer step,
journal-dual899/train-001 exited with the unchanged host RAM guard: 5.95 GiB
available against the 6.0 GiB training minimum. The supervisor observed return
code 2; the journal charged one launch and seven seconds, zero training/evaluation.
This is a resource refusal, not a scientific result. Preserve its full journal.

The intended one actual training run has not happened. Authorize exactly one
additional worker attempt under the original user's one-run authority, with a
fresh grant/journal explicitly linked to the refusal. This is an agent-declared
prospective execution amendment, not fabricated human approval. Across both
attempts the aggregate caps are two worker launches, one pre-load retry, and
380,921 processed training tokens; no retraining after any optimizer step.
Other session and native numeric ceilings remain unchanged (deduct the first
seven seconds and 4 MiB reservation from the second grant's wall/disk ceilings).

Wait at most 900 seconds for >=7.0 GiB host RAM BEFORE importing the training
runtime, a buffer above the real 6.0 GiB child guard. Do not kill other tasks,
flush their memory, remove another lease, change the environment, or lower a
guard. The unchanged child guard is still authoritative. If this second attempt
is refused or fails, preserve it and stop model execution under this envelope.

The registered data, alpha values, evaluation bands and analysis do not change.
