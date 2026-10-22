# CQD-001: public-prompt shortcuts and verifier stress

This CPU-only study generates 256 TRAIN_SELECTION and 256 DEV semantic parents
(64 per family in each role) with the inherited role allocator and independent
verifier. TRAIN, CHECK and FINAL quotas are zero. It fits only a modal-answer
baseline on TRAIN_SELECTION, freezes that baseline, then scores DEV once.

The twelve predictors receive only public prompt text. Private family labels and
DEV answers are evaluation-only. Constant, number-copying and unchanged-state
answers establish shortcut floors; they do not establish reasoning or model gains.
Separately labelled gold-derived mutations test verifier acceptance/rejection;
they are not policy predictions. Generator and verifier correctness do not prove
that native Rineya relies on shortcuts.

Run from this dedicated worktree root through the campaign CPU supervisor:

    python -B -m experiments.cheap_quality_shortcuts_v1.study --out experiments/cheap_quality_shortcuts_v1/runs/study-001

Inputs/source/config/interpreter are frozen before generation. Outputs are
create-only JSON, public/private data exports, CSV and HTML. No model/runtime
import, GPU, training, download or paid service occurs. Limits: 512 MiB worker
tree RSS, 900 seconds, 50 MiB output, one worker. Native guards are unchanged.
