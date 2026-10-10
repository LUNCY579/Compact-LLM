# Centralized Titan 0.6B program

Primary: **Rineya-Titan-Experimental-1.0.0-devfreeze**, 596,049,920 parameters.
The full verified acquisition and retention objective remains open.

Publication review on 9 October 2026 recovered two completed native runs that
had not been indexed in the public summaries. The owned 40-row/two-update
adapter qualifier passed zero-adapter parity, reference gradients, nontrivial
state/RNG restoration and copied-gradient Adam continuation, with **1.672089 GiB**
peak process-tree RAM. The fixed-shape four-forward causal diagnostic passed
all six protected hidden/logit comparisons, with **1.350418 GiB** peak RAM.
These are bounded mechanics and diagnostic results, not capability gains.

The earlier CE512 trial passed eight checks and four disposable updates at
1.748646 GiB peak RAM. All 5120 TRAIN rows passed source/array binding; maximum
length 398 fits the 512-token envelope without truncation. The model loads
through a fixed 2 MiB CPU buffer and stays on GPU with FP32 rank-8 q/v adapters.

The original prefix-versus-bulk likelihood comparison still failed. Passing
fixed-shape causal controls does not prove shape-invariant numerics or qualify
capture. No DEV baseline, preservation result or production E1 epoch is claimed.
Current best remains unchanged Titan; CHECK and FINAL remain closed.

Use the [next experiment](NEXT_EXPERIMENT.md), [current laptop plan](manifests/LAPTOP_EXECUTION_PLAN_V3.json),
[runbook](LAPTOP_RUNBOOK.md), [resource policy](RESOURCE_POLICY.md), and
[experiment report](CENTRALIZED_EXPERIMENT_REPORT.md). The September 30 blocked audit remains historical; the app has since resumed
the full goal. This review itself launches no experiment and does not treat
an old RAM reading as current.

Evidence:

- [Owned native mechanics](receipts/NATIVE_BACKEND_QUAL_V1_NATIVE_RESULT_20261001.json)
- [Fixed-shape causal diagnostic](receipts/CAPTURE_CAUSAL_WAIT_V2_NATIVE_RESULT_20260930.json)
- [Earlier CE512 mechanics](receipts/LAPTOP_NATIVE_STAGE_005_RESULTS.json)
- [TRAIN joins](receipts/train_scope_v1/TRAIN_SCOPE_RECEIPT.json)
- [Original likelihood failure](receipts/CAPTURE_SHORT_V1_NATIVE_FAILURE.json)
- [Historical blocked audit](receipts/GOAL_BLOCKED_AUDIT_20260930_V1.json)

Language labels/calibrated grading, native long-context preservation, sustained
training and current E1 launch binding remain prerequisites. No paid grading or
cloud rental is authorized. Local checkpoint binaries remain outside Git;
byte-exact metadata/log copies and their hashes are published.
