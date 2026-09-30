# Laptop execution route

30 September 2026. The exact Titan BF16 model can load and perform short adapter
mechanics on this 16 GB laptop using GPU-resident state and a fixed 2 MiB CPU
loading buffer. Complete CE512 mechanics observed 1.748646 GiB peak process RAM;
it passed save/resume and unchanged-base checks. The inherited 7.4 GiB free-RAM
floor was a policy for a different CPU-reference workflow.

Follow [plan V3](manifests/LAPTOP_EXECUTION_PLAN_V3.json). Run one native GPU
worker at a time. Use one unpadded sequence per microbatch; all prepared TRAIN
rows have at most 398 tokens. Accumulation, sustained input consumption and full
training still need separate registered measurement. The short worker's 1.75 GiB
cap has only 1.387 MiB observed margin; do not treat it as a sustained training fit.

## Execution without the desktop app

The retained `scripts/run_laptop_adapter_trial.cmd` and
`scripts/run_laptop_trial.ps1` work from a regular Windows shell. The default is
the scoped unpadded_ce512 mechanics qualifier, not E1 training. That qualification
has already passed; an unchanged repeat is unnecessary. The controller waits for
two actual available-RAM observations at least 3.0625 GiB, including its supervisor
allowance, before launching the bounded contained worker. After imports the worker
reserves only its remaining permitted RSS growth plus 1.25 GiB of system headroom.

`scripts/run_capture_trial.ps1` uses the separate 2 GiB RSS profile and 3.3125 GiB
before-worker budget, with a 600-second native ceiling. Its first pilot stopped
on the fixed likelihood comparison, with memory headroom. The completed geometry
diagnostic reproduced that failure and identified an upstream hidden-state
difference; FP32 output-head projection alone did not resolve it. Do not rerun
the unchanged failed qualifier or treat its memory pass as a capture pass.
AMENDMENT_016 now has a reviewed four-forward causal diagnostic and fourteen
passing CPU controls. Its admission window ended with no native invocation as
available RAM fell below the 3.3125 GiB budget. The controller is terminal;
run only after real headroom changes. No native causal result exists yet.

The shell controller does not need the chat UI to remain open. Closing a desktop
app is optional and does not guarantee any particular amount of reclaimable RAM;
actual available memory decides admission. No user app or unrelated process is
terminated. A finite admission window ends with receipts if headroom is insufficient;
it does not restart automatically.

## What is measured and what remains

| Work | Result | Next condition |
|---|---|---|
| Exact streamed GPU load | Passed with exact 310 tensor copies | Same immutable checkpoint/runtime |
| Single-sequence CE through 512 | Eight checks and four disposable updates passed | Separate sustained/accumulation/input overhead measurement |
| Actual TRAIN contract | All 5120 source/array joins passed; reader peak 39.004 MiB | Stream that dose into a registered native executor |
| Short greedy capture | Memory checks passed; raw-model likelihood failed | Qualified distinct numerical capture scope |
| Held-hidden diagnostic | Exact failure reproduced; head layout isolated | Register smallest upstream/scoring experiment |
| Deterministic baseline | 512 input rows frozen; no responses collected | Capture qualification and measured production budgets |
| Language preservation | Candidate packets prepared, grading unqualified | Independent labels, identified calibrated grader and blinding/cost evidence |
| Long-context preservation | Inputs prepared | Separate native 2048/4096 qualification |
| E1 acquisition and retention | Unmeasured | Current executor and all scientific prerequisites |

Outputs are write-once under `artifacts/mimo_central`, with separate worker,
supervisor and controller receipts. Keep the checkout committed and unchanged
during a run. Windows Job containment applies only to its owned worker tree.
Source, memory, GPU, time or numerical failures stop the attempt and retain its
partial accounting. Adapter binaries remain local; the published base is not
overwritten. Decode/score panels sequentially rather than holding another model
or tokenizer/evaluation process during training. CHECK/FINAL remain closed.


AMENDMENT_018 adds a source-reviewed bounded resource wait after framework imports
for the causal diagnostic only. It retains2 GiB workerRSS,1.25 GiB systemreserve,
GPU4/0.5 GiB,disk4 GiB and600seconds; no model call precedes two current full-budget
observations and a strictfresh preload check. The maximum wait is60seconds with
2-second polls. Timeout or source/runtime-floor failure stops without model retry.
The prior admitted worker failed before any model call; the successor's first
preworker window remained below3.3125 GiB. It is not native-qualified or E1-authorized.
