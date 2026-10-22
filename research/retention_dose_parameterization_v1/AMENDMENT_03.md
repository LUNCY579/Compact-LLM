# DOSEINT-001 Amendment 3: correct omitted qualification-token accounting

The previous envelope is CLOSED_FAILED. Its second worker stopped at update 191
of 193, before a final checkpoint existed. No dose-bank, SELECT, retention or
long-context endpoint has been generated for this experiment. Both failed workers
and the early checkpoint remain preserved. This amendment does not retroactively
make either worker complete or expand either consumed grant.

Cause: the agent set processed_training_tokens_all_attempts to the data plan's
380,921, overlooking native_runtime.qualify's charge of 5*31+2048 = 2,203 under
the same counter. The journal recorded 377,281 optimizer-data tokens + 2,203
qualification tokens = 379,484, leaving 1,437; the next group's reservation was
correctly refused. The guard worked. The agent's allowance was wrong.

The early checkpoint is at update 2, and no later checkpoint survives; there is
no honest way to recover the lost update-191 model. Resume also requires the
exact old grant binding, so do not rewrite a manifest or reuse that spent grant.

Under the user's subsequent instruction to analyze thoroughly and perform the
best recommended step while they sleep, open ONE fresh fixed recovery attempt
from the frozen source with the identical PLAN_DUAL899 and prepared bytes. It is
a replacement training draw after an engineering failure, not an extra alpha
candidate or a comparison to the incomplete run. Register before execution. No
endpoint-informed change to data, optimizer, seeds, alpha values or bands.

Fresh grant/journal: GRANT_DUAL899_RECOVERY.json and journal-dual899-recovery.
One launch, zero retries inside this new envelope; processed-token cap 400,000,
covering exactly 380,921 planned optimizer-data tokens and the independently
verified 2,203 qualification-token charge (expected journal total 383,124).
The unused buffer licenses no extra data or optimizer steps; the frozen plan
still stops at update 193. Retain the lane's native 2,400 worker /3,600 owned
seconds, 240 calls, 30,720 generation tokens and 8 GiB disk ceilings.

Global accounting across all three workers is explicit: three launches, one
pre-load refusal, one failed optimization attempt, one recovery attempt; three
journal bookkeeping rounds and zero adaptive selection. Before recovery, native
cost is 1,088 seconds, 379,484 charged tokens, 64 model calls, 6,144 reserved
generation tokens and 3,784,863,744 reserved disk bytes. Global ceilings are
779,484 charged tokens, 304 model calls, 36,864 generation tokens, 3,488 native
worker seconds and 4,688 native owned seconds. These include failures. The actual
optimizer-data token total, if recovery completes, is 758,202.

Increase the overall new-records disk ceiling from 17 to 21 GiB to retain the
failed early checkpoint AND both recovery checkpoints AND all four FP32 arms.
The unchanged free-space floor remains 14 GiB. A clean, task-created code
worktree was made sparse to research/ and experiments/ using git sparse-checkout,
recovering approximately 2.2 GiB of unrelated tracked working copies. Those files
remain in Git and other worktrees; no model, evidence, source, or checkpoint was
deleted. Measured C: free space after this reversible change: 31.888 GiB.

Wait at most 900 seconds for >=7.0 GiB host RAM and a free GPU lease before the
fresh attempt; retain the child's 6.0 GiB guard. Stop and preserve any failure.
The 18,000-second total session wall ceiling and all probe limits stay unchanged.
