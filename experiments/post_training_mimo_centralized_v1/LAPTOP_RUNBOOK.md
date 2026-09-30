# Laptop execution route

30 September 2026, America/Chicago. The earlier 7.4 GiB free-RAM requirement was
an inherited policy for a different CPU-reference workflow. It was not measured
Titan demand. Exact Titan loading and a 256-token GPU forward have now succeeded:
observed process-tree RSS peak 1.928 GiB; PyTorch CUDA allocation peak 1.400 GiB.
All 32 zero-adapter comparisons also matched exactly in a separate full attempt.

The smallest registered route uses a fixed 2 MiB checkpoint read buffer, a 1.75 GiB
worker RSS ceiling and a 1.25 GiB system reserve. It budgets 3 GiB before worker
launch, with live admission accounting for already occupied worker RSS. The launcher
adds 64 MiB for its supervisor and waits for two observations >=3.0625 GiB. The
new loader passed actual CPU BF16 byte-identity/RNG controls; its first native
attempt stopped before weight access because desktop/background RAM fell further.
Its complete GPU path is still unqualified. No new model numerical tolerance was
relaxed, and no user app is terminated by this route.

## Run outside the desktop app

Double-click `scripts/run_laptop_adapter_trial.cmd` in File Explorer, or invoke
`scripts/run_laptop_trial.ps1` from a regular PowerShell window outside Codex/ChatGPT.
Closing the desktop app can reduce its observed roughly 2 GiB working set, but
working-set totals are not guaranteed reclaimable RAM; the controller measures
actual available memory. The trial does not need the chat UI to remain open.

The default is the separately scoped backward_resume stage. It waits up to ten
minutes for headroom, then launches one process-contained trial with a 30-minute
worker ceiling. It stops on memory, GPU, time, source, numerical or resume failure.
There is no automatic model retry, guard increase or E1 launch. Waiting time and
native time are recorded separately. Keep the experiment checkout committed and
unchanged while it waits or runs. Close the regular PowerShell window to stop its
controller; the native supervisor's Windows Job Object contains its model worker.

Outputs are write-once under `artifacts/mimo_central/headless_*`, including a
separate controller dispatch receipt and native worker/supervisor receipts. Failed
or timed-out trials remain useful evidence. Adapter checkpoint binaries remain
local and disposable; the published Titan checkpoint is never overwritten.

## What each phase establishes

| Stage | Bound | Scope |
|---|---|---|
| load_forward | 600 s; zero updates/generation | Exact GPU load and one finite 256-token forward |
| full | 1800 s; at most five disposable updates | Every original E0 numerical/resource check; currently stops on padded likelihood |
| backward_resume | 1800 s; at most five disposable updates | 256/512/1024 backward fit, confined gradients, save/resume and frozen-base identity; padding/sampler unresolved |

Native adapter training fit and resume remain unmeasured until the last phase runs.
Its resource-only success cannot qualify full E0 or demonstrate capability gain.
E1 still requires numerical qualification, measured runtime/budgets, source capture,
qualified language grading and preservation checks. CHECK/FINAL remain closed.

After qualification, prioritize one resident GPU worker, pretokenized streamed
microbatches, rank-8 q/v adapters and adapter-only checkpoints. Perform evaluation
and token decoding sequentially rather than keeping a CPU reference model or
extra tokenizer/evaluation process resident during training. Use separate registered
successor plans; preserve all historical resource-policy receipts.
