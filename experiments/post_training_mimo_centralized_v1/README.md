# Centralized Titan 0.6B program

Primary: **Rineya-Titan-Experimental-1.0.0-devfreeze**, 596,049,920 parameters.
The original verified capability and retention objective remains open.

The user's laptop RAM revision now has measured support. Exact BF16 loading uses
a fixed 2 MiB CPU buffer and keeps the model on GPU. Complete unpadded CE512
mechanics passed eight checks, four disposable adapter updates, bitwise resume
and unchanged-base verification, with 1.748646 GiB peak process-tree RAM.
All 5120 prepared TRAIN rows passed actual source/array joins; maximum length
398 fits this envelope with no target truncation.

The next capture pilot passed memory checks but failed the fixed raw-model
likelihood comparison. A bounded diagnostic reproduced the failure and localized
the difference upstream of the output head; changing that head to FP32 alone
still fails. This is a numerical blocker, not a finding that Titan needs 7.4 GiB
free RAM. No DEV collection, capability gain, retention result or E1 production
training has been claimed. Current best remains unchanged Titan.

Use the [laptop plan V3](manifests/LAPTOP_EXECUTION_PLAN_V3.json),
[next experiment](NEXT_EXPERIMENT.md), [runbook](LAPTOP_RUNBOOK.md) and
[resource policy](RESOURCE_POLICY.md). The plan prioritizes one GPU worker,
streamed inputs, microbatch 1 and sequential evaluation. It distinguishes the
measured short CE mechanics from sustained training, which still needs its own
registered input/accumulation/RSS measurement.

Evidence: [CE512 native result](receipts/LAPTOP_NATIVE_STAGE_005_RESULTS.json),
[TRAIN joins](receipts/train_scope_v1/TRAIN_SCOPE_RECEIPT.json),
[capture failure](receipts/CAPTURE_SHORT_V1_NATIVE_FAILURE.json), and
[geometry diagnostic](receipts/CAPTURE_GEOMETRY_D1_NATIVE_RESULT.json).
The [research ledger](RESEARCH_LEDGER.md) and CSV ledgers retain earlier attempts
and append actual work. Original 7.4/6 GiB configurations, full-padding failure,
1024 VRAM failure and prior blocked audits remain historical evidence. Named
laptop successors change the operational route without rewriting those records.

Language grading, native long-context preservation and the current E1 executor
remain prerequisites. Existing language packets are unassigned diagnostics;
independent labels, calibrated grader identity, blinding and call/cost budgets are
required. CHECK/FINAL remain closed. No paid grading or cloud rental is authorized.
