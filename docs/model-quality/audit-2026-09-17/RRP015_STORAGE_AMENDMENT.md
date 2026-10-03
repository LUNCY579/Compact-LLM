# RRP-015 storage accounting amendment

Recorded during control training, before any held-out candidate evaluation or
rehearsal-arm launch. No scientific recipe, rows, endpoint, threshold, inference
setting, time budget, training grant or resource guard changes.

The initial 18 GiB disk ceiling did not distinguish retained high-water space from
cumulative writes of redundant evaluation exports. Two complete checkpoint pairs
use about 14.06 GiB. Each candidate's two legacy evaluation layouts add about
3.33 GiB (BF16 HF 1.11 GiB plus FP32 research format 2.22 GiB). Keeping one arm's
exports at a time fits within 18 GiB retained space, but creating both arms'
layouts writes about 20.72 GiB cumulatively.

The bounded arrangement is now explicit: **18 GiB retained high-water new space;
22 GiB cumulative newly written artifacts**. This is an agent-instantiated local
storage allowance under the user's broad local-compute authority. No paid service
or source asset changes. The original per-arm 8 GiB training budgets and the
controller's 14 GiB free-space floor remain unchanged.

Run both trainings before exports. Materialize and score the first candidate,
retain its JSON hash/source/checkpoint custody receipts and every raw prediction,
then remove only this experiment's redundant exported weight file(s) after
verifying the absolute path and the original COMPLETE checkpoint still exists.
Repeat for the second candidate. Never delete checkpoint tensors, optimizer state,
failed attempts, source files, model registries or raw measurements. Native source
weights and all full training-state checkpoints remain available for reproduction.

Do not create an export if it would leave less than 14 GiB free. A single layout
may be generated, scored and reclaimed before generating the other layout if
needed. This changes storage scheduling only; models/scorers stay identical.
