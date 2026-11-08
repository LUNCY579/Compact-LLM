# RRP-005 A2: durable source evaluation

## Actual preceding attempt

`preservation-baseline-journal/runs/baseline-001` ran real Supa inference at commit
`00c577b`. The supervised child exited with code 2 after **422.469 seconds**. Its journal
charged **554 calls** (6 qualification forwards + 548 case starts) and **35,072 reserved
generation tokens**. The failure was the unchanged 1.5 GiB host floor observing 1.49 GiB.
There were zero optimizer steps and no active reservation after settlement.

The evaluator's completed predictions were held in RAM until final return. Consequently
**no partial prediction bank survived**. Do not score or reconstruct those lost outputs.
The byte-preserved receipts/logs are under `attempts/baseline-001/`; their hashes and the
exact budget settlement are in `AMENDMENT_A2.json`. Qualification completed as shown by
subsequent evaluation calls, but its detailed receipt was also delayed until final output.

## Corrections, not a silent rerun

- Every completed case is now fsynced in a create-only hash-chained record. Records bind
  exact source state, bank/token rows, plan, grant, runtime, code and checkout. Per-row CE,
  generated IDs, EOS/cap status, text/score consistency and resource snapshots are checked.
- A fresh attempt may continue a validated prefix under the **identical binding**. It
  generates only the missing suffix and accounts copied evidence separately from new
  model calls. Tampering, gaps, changed identities and a failed source-state integrity
  check are refused. Failed artifacts remain untouched.
- The zero-update inference qualification is saved immediately. Source parameter identity
  is still checked before/after the full evaluation by the controller; a stream COMPLETE
  marker alone explicitly does not establish that final check.
- The A2 policy releases only unused CUDA allocator blocks after completed independent
  cases. Live weights and per-case KV computation are unchanged. This may reduce cache
  pressure; it does **not** establish a reduction in host RAM. See
  [PyTorch 2.10 documentation](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html).
- RAM/VRAM limits are unchanged. No process was terminated, no source weights changed,
  and no new runtime dependency was installed.

Tests: **397 passed, 1 skipped** in `tests-streamed-evaluation.xml`. These include an actual
native-evaluator fixture interrupted by its guard, exact prefix-continuation checks,
changed-binding/tamper/gap refusals and all prior accounting/data/optimizer checks.

## Bounded budget amendment

Old remaining wall/GPU, disk, training and selection allowances transfer exactly to A2.
Only 1,000 logical evaluation/qualification calls, 65,536 reserved generated tokens and
one launch are added, because the prior evaluator lost the completed work. This is
explicitly agent-instantiated under the user's local-compute authorization. No old grant
is rewritten. A2 permits no training, RL, model selection, teacher/cloud spend or release.

Fresh launch, from a clean committed checkout and sufficient live resources:

```powershell
& ./research/reasoning_posttraining_v1/rrp005/run_baseline_a2.ps1 -RunId baseline-001
```

Only after a diagnosed interruption and verified process exit, and without changing the
binding/checkout, a new attempt can use:

```powershell
& ./research/reasoning_posttraining_v1/rrp005/run_baseline_a2.ps1 -RunId baseline-002 `
  -ResumeEvaluation C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/preservation-baseline-a2-journal/runs/baseline-001/prediction-stream
```

No automatic retry loop is present. The original nonstreamed baseline lane is retired.
Do not commit code/docs between an interrupted A2 attempt and its possible continuation;
new binding means new qualification, not permission to splice incompatible predictions.

## Concurrent CPU-only data finding

While the original model run was active, an isolated tokenizer-only process inspected
**448 TRAIN worlds**, selected prospectively as 32 IDs per family/status. No model or GPU
was loaded by that process. A public-scope filter discarded only records belonging to
another explicitly named scope; reparsing proved the canonical task world unchanged.

| View | Fits full 2048-token training window | Mean processed tokens | Mean supervised tokens |
|---|---:|---:|---:|
| Full, final-only |139/448|4,966|11.9|
| Scope-only, final-only |339/448|1,555|11.9|
| Full, released reasoning trace |113/448|5,392|411.2|
| Scope-only, released reasoning trace |312/448|1,981|411.2|

This is an explicitly easier training input and a **token-efficiency finding**, not model
improvement, actual speedup or demonstrated long-context transfer. Long-context evaluation
must retain full inputs. Reasoning views need separately qualified serialization/training
and generation budgets. The longest complete target is **7,442 tokens**. Scope filtering
does not solve every family's coverage: evidence-chain reasoning remains 0 fitting cases,
and ledger-replay reasoning also remains 0. Compact verified intermediate targets still
need investigation. The complete selection/measurements and reproducible script are
`SCOPE_TOKEN_SURVEY.json` and `scope_efficiency_probe.py`.
