# Exact-prefix fixed-trace V1: bounded attempt stopped before native work

The registered, implemented and independently reviewed six-forward experiment
was launched once. It stopped at the specific supervisor Windows Job membership
predicate before importing Torch. No model likelihood was measured and capture
remains unqualified. This is a containment blocker, not a numerical result.

## Source and deliverables

- Isolated worktree: `C:\Users\clclc\Documents\Codex\2026-10-01\task\ChatBubble-mimo-exact-prefix`.
- Branch: `codex/mimo-exact-prefix-capture-20261001`.
- Verified central base: `19551ddebc87c93161c303865ce6d0f130de0093`.
- Prospective registration: `0963de398b0becbf2ad34abb6c3b652a22c199b7`.
- Reviewed clean source used by the attempt: `fa36c38d467f7c48711dbee3823c1bac1cb9c93d`.
- Added protocol/config: [AMENDMENT_022.md](AMENDMENT_022.md) and
  [CAPTURE_EXACT_PREFIX_FIXED_TRACE_V1.json](configs/CAPTURE_EXACT_PREFIX_FIXED_TRACE_V1.json).
- Added implementation: [exact_prefix_capture.py](scripts/exact_prefix_capture.py),
  [qualify_exact_prefix.py](scripts/qualify_exact_prefix.py), focused
  [tests](tests/test_exact_prefix_capture.py).
- The subsequent local commit only preserves receipts, reviews and this report.
  No source changes or second native attempt followed the blocked attempt.

## Question and choice

Git metadata inspection covered 45 matching ChatBubble, SupaExperiment and
RineyaArchitecture worktrees, including this isolated checkout. No newer local
registered exact-prefix capture successor was found. Amendment 016 allowed a
later registration but did not itself register one. D2-WAIT-V2 had passed six
same-shape causal hidden/logit comparisons bitwise. Backend40 had already passed
its two-update mechanics scope and was not repeated. COMP-RS remained
STOP_NO_SIGNAL; neither the same recipe nor S1 was launched.

The proposed test compares raw greedy online likelihoods with independently
rebuilt exact-prefix teacher-forced scoring on the already exposed 128-token
mechanical prompt and three historical greedy tokens `[16, 22, 151645]`.
Offline scoring visits lengths 130, 129, 128 in reverse target order, excluding
the current target and future tokens. Raw lowest-ID greedy online choices must
reach natural EOS and reproduce the historical trace; they are never forced.
Both paths use eager BF16, cache off and one retained logit row. Maximum/mean
absolute likelihood tolerances stay 0.03125/0.01. This scope does not establish
prefix/full bulk-shape equivalence or capability.

## Frozen hardware budget and actual attempt

Laptop: Ryzen 9 7940HX, approximately 15.22 GiB usable host RAM, RTX 4070 Laptop
with 8188 MiB VRAM. One constructor/load, 310 initial streamed tensor copies,
six forwards, 774 input positions, three generated tokens, 318 logical calls;
zero training/backward/optimizer/teacher/checkpoint operations. Maximum worker
time 600 seconds with the existing 100 ms external watchdog and kill-on-close Job.
Limits: worker-tree RSS 2 GiB, host reserve 1.25 GiB, GPU free 4 GiB at admission
and 0.5 GiB during work, disk free 4 GiB. Preworker requires two observations
at least 3.3125 GiB RAM two seconds apart; post-import uses the existing 60-second
dynamic growth-budget wait and strict fresh preload admission. The latter stages
were not reached.

Fresh launch observations: host free 5.8133 and 5.8174 GiB, GPU free 7.2080 GiB,
disk free approximately 70.7004 GiB. Matching research-worker overlap was empty.
Both shared GPU leases were acquired exclusively and released after the stop.

Actual supervisor elapsed time: **1.219 seconds**. Peak observed worker-tree RSS:
**51,752,960 bytes (49.3555 MiB)**. All constructors, pretrained loads, tensor
copies, forwards, input positions, generated tokens, backward, optimizer and
teacher calls: **zero**. No Torch/GPU allocation or native model resource peak
was measured. The worker exited 1; supervisor recovery preserved matching
failure/result/cost accounting. The sole registered attempt is consumed.

## Validation and diagnostic limit

All **58 stdlib CPU tests** passed (29 focused successor tests, 15 inherited
greedy/EOS tests and 14 post-import admission tests). Final suite wall time was
0.532 seconds. The tests establish fixture orchestration, identity, EOS, bounds,
pending/partial/corrupt failure accounting and recovery controls. Independent
pre-native review passed after three identified ownership/accounting/source
issues were fixed. The reviewer independently confirmed the post-attempt block
and supported zero-operation accounting from the traceback and source order.

The failing line combines `IsProcessInJob` API success and membership. API return,
Windows last-error and actual worker PID/creation time were not retained at that
stage. A venv launcher spawning its interpreter before Job assignment is plausible
but unproven. Do not describe this receipt as proving that race or a successful
membership query returning false. Do not bypass the guard or rerun V1 unchanged.

## Evidence and preservation

- [Attempt summary](receipts/EXACT_PREFIX_V1_ATTEMPT_SUMMARY.json) binds exact
  counts, admission readings, asset hashes and all 13 original/preserved artifacts.
- [Worker result](results/exact_prefix_fixed_trace_v1_20261001/worker/CAPTURE_RESULT.json),
  [cost ledger](results/exact_prefix_fixed_trace_v1_20261001/worker/COST_LEDGER.json),
  [supervisor result](results/exact_prefix_fixed_trace_v1_20261001/supervisor/SUPERVISOR.json)
  and [traceback](results/exact_prefix_fixed_trace_v1_20261001/supervisor/stderr.log).
- [CPU receipt](receipts/EXACT_PREFIX_V1_CPU_VALIDATION_003.json),
  [source audit](receipts/EXACT_PREFIX_V1_SOURCE_AUDIT.json),
  [pre-native review](receipts/EXACT_PREFIX_V1_INDEPENDENT_REVIEW.json),
  [post-attempt review](receipts/EXACT_PREFIX_V1_POST_ATTEMPT_REVIEW.json).
- [Initial inspection](receipts/EXACT_PREFIX_V1_LOCAL_INSPECTION.json) and
  [final preservation audit](receipts/EXACT_PREFIX_V1_FINAL_PRESERVATION_AUDIT.json).
- [Final cleanup observation](receipts/EXACT_PREFIX_V1_FINAL_CLEANUP.json):
  at 17:25:16 UTC no matching research workers, both leases absent, host free
  5.5673 GiB and GPU free 7380 MiB.

Raw run artifacts remain under `artifacts/mimo_central/exact_prefix_fixed_trace_v1_20261001`
and its `_supervisor` sibling. Thirteen copies in `results/` are byte-identical.
Original model assets, D2 and backend40 receipts remain unchanged. Central base
remains clean; all five unpublished patch worktrees and commits remain unchanged
and were not cherry-picked. No existing tracked producer/validator was edited.
Final metadata comparison observed 1 changed external worktrees;
the audit records any difference without resetting or touching concurrent work.

The old bulk likelihood failure remains max 0.04143241047859192 and mean
0.02629935139945398 against frozen 0.03125/0.01. General capture, capability,
DEV collection, CHECK/FINAL, E1 and independent language grading grants remain
false. Long-context qualification, native evidence validators and sustained
resource fit remain open. Current best remains unchanged.

Requested assistant settings were GPT-6.1 Sol/Max/Standard with no fast mode.
The active thread's settings could not be changed or conclusively verified by
workspace tools. Existing global configuration gpt-6.1-sol/ultra/priority was
left untouched; independent review explicitly requested gpt-6.1-sol/max, whose
tool provides no service-tier override. Standard mode is not claimed as verified.

## Next bounded action

Prospectively register a short **stdlib-only Windows Job containment diagnostic**
with zero Torch/GPU/model work. Record worker/launcher lineage, PID creation times,
specific-Job API return, membership and Windows error separately. Select and
qualify the ownership correction from that evidence, then register one corrected
six-forward capture successor with the same numerical tolerances and hardware
limits. Preserve this consumed V1 and the bulk failure. No expanded native budget,
paid compute, training, E1, S1, merge, deployment or publication is authorized here.
