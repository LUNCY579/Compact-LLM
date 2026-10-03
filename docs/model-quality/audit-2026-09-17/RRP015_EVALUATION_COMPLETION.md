# RRP-015 prospective evaluation-completion amendment

The original Supa baseline attempt failed before any model load because psutil was
missing. Its reserved retry then exceeded the agent-chosen 600-second watchdog.
Both attempts remain FAILED; neither is relabeled complete or given a score.
The latter logged 336/384 rows at 515 seconds and was terminated at 600.156 seconds.
The legacy scorer writes full predictions only at completion, so no partial
prediction artifact survived. This exposes a real restartability limitation.

The timeout was an engineering planning error: the already preserved identical
instrument's full Supa measurement took 642.17 seconds. The agent had not used that
known runtime when choosing the original 600-second limit. The timeout is not
evidence about model quality, and no new candidate held-out score has been read.

Before any further evaluation outputs, authorize one separately identified
completion attempt under the user's broad local-compute instruction:
`eval-rrp015_supa-process-completion001`, maximum **900 seconds**. The wrapper's
hard maximum becomes 900; all seven other planned evaluations keep their original
600-second limits. No new training or adaptive recipe choice is introduced.
No further retry belongs to this amended pilot.

Aggregate limits: **12 total launches**, **8 completed evaluation outputs**, at most
**3,072 evaluated/attempted rows**, and the unchanged **8,400 owned model seconds**
and **245,760 generated-token ceiling**. The time reservation fits: completed
training consumed about 739 seconds, the failed baseline 601 seconds including
pre-model failure, and the remaining eight successful outputs reserve at most
900 + 7*600 seconds, totaling less than 6,500 seconds. Storage limits stay as in
RRP015_STORAGE_AMENDMENT.md. Record measured usage rather than erasing failed work.

This changes operational allocation prospectively; the original execution envelope
was not completed. Report the final work as the **amended exploratory pilot**.
Training plans, source hashes, prepared rows, masks, optimizer, precision,
generation limits, shortcut floors, quality thresholds and all host/GPU/disk
minima are unchanged. This licenses no confirmation or release claim.
