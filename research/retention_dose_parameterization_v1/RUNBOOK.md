# DOSEINT-001 execution and custody

Start with the records repository's PREREG.md and AMENDMENT_01.md/02.md. The exact
historical D4 launch was refused before dispatch. The current design uses 2,697
parents disjoint from both acquisition banks, not the historical 3,072-row draw.

Code worktree: `ChatBubble-dose-param`, branch
`research/retention-dose-parameterization-v1`. The qualified executable lane is
the separate clean pilot clone `ChatBubble-retention-pilot-20260917`, at a7d1bef.
Do not switch or clean the original user's working tree, alter pinned venvs,
remove the GPU lease, or add this branch to the controller allowlist.

Records root:
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/doseint001-20260919`.
Keep the records repository `.gitattributes` rule `* -text`.

The one approved training run uses `journal-dual899-preloadretry`. The earlier
`journal-dual899` is a preserved 5.95/6.0 GiB RAM refusal, with zero model calls
and optimizer steps. Do not issue another train command while a launch runs or
consume another grant. `wait_train.py --execute` is create-only and was already
dispatched; it is not a resume command.

After the training supervisor has returned and its result, budget, qualification,
data/source/code binding and final checkpoint are verified, run:

```powershell
& 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe' `
 'C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/retention_dose_parameterization_v1/run_battery.py' `
 --native-run 'C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/doseint001-20260919/journal-dual899-preloadretry/runs/train-001' `
 --execute
```

Omit `--execute` for a read-only plan. The battery verifies native completion,
builds four FP32 arms sequentially, then runs each arm on both acquisition banks,
retention, and long context. All child exits and success receipts are checked.
The final report rechecks byte custody before extracting endpoints. Never use a
partially completed battery as a successful model-quality result.

If any leg fails, inspect its FAILURE.json, wrapper worker receipts, and preserved
launcher log. Do not rerun this master command into the same directories. One
new inference-only attempt per failed leg is allowed by the envelope, but must
have a new output and legacy label and be explicitly selected in the report.
There is no permission for another training draw after any optimizer step.

Metadata/statistics checks require no model imports:

```powershell
& 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe' -B -m unittest `
 research.retention_dose_parameterization_v1.test_paired -q
```

Run the tiny CPU tensor builder fixture only while the model lane is idle:

```powershell
& 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe' -B -m unittest `
 research.retention_dose_parameterization_v1.test_build_arm -q
```

These checks establish mechanics only. Every scientific contrast still requires
matching views, scorer, code, contemporaneous weight bindings and complete row IDs.
Use `(net, nd, exact p)` and the exact attainable threshold at that discordance.
Do not pool banks, infer historical offsets, or call a nonsignificant difference
equivalence. Retain failed predictions and abstention damage in the report.
