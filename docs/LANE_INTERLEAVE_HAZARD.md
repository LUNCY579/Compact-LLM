# You cannot interleave RRP training and RAB scoring in one script

A train → score → train pipeline over this repository **poisons itself on the
second launch**, and the error names neither the cause nor the file.

## The mechanism

- `experiments/rab001_exploration/process_probe.py` and `retention_probe.py`
  write their results into `experiments/architecture_benchmark_001/evidence/…`,
  which is **inside** the worktree.
- `experiments/reasoning_posttraining_v1/controller.py:118` refuses any native
  launch when `git status --porcelain --untracked-files=all` is non-empty:
  `native execution requires a clean checkout`.

So the moment rung N is scored, rung N+1 cannot train. Observed three times in one
session on the RRP-008 dose ladder; D2 and D4 both died this way with zero model
calls and zero training tokens consumed.

**It also costs a retry.** `budget.start()` charges `retries: 1` on the second
launch of a journal, and the default ceiling allows exactly one, so two pre-flight
refusals exhaust a journal and force a fresh one.

## What to do instead

**Sequence by phase, not by rung.** Train every rung first, then score every rung:

```
for rung in ...; do train "$rung"; done      # worktree stays clean throughout
git add -A && git commit                      # or leave scoring to the end
for rung in ...; do score "$rung"; done      # dirties the tree, but nothing launches after
```

If rungs must be interleaved, commit (or move aside) the evidence files between
them — but note that committing moves HEAD, which changes the recorded
`binding.checkout` and, if the changed files live under
`experiments/reasoning_posttraining_v1/`, also `code_sha256`. For a ladder or a
replication whose whole point is "only the data varies", that is not free.

`package_identity()` hashes only `reasoning_posttraining_v1` and skips
`__pycache__`, `.pytest_cache`, `outputs` and `evidence`, so committing RAB
evidence JSON changes `checkout` but leaves `code_sha256` and `mq_code_sha256`
untouched. That is the cheap case; a source edit is not.

## Why this is not just "remember to be careful"

The two lanes were designed under different contracts. The RAB lane treats the
repository as the evidence store and writes into it by design. The RRP lane treats
a clean worktree as a precondition for reproducibility and refuses otherwise. Both
are reasonable; together they are a trap, and the trap is silent because the error
message points at "the checkout" rather than at the JSON file a scorer wrote ninety
seconds earlier.
