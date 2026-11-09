# Rineya composable-skills continuation

The active experiment tests whether independent old-skill examples protect existing
behavior during low-cost acquisition. It is a controlled precursor to stacking skills,
not yet a demonstration of sequential or joint composition.

Read [PROTOCOL.md](PROTOCOL.md) for the hypothesis, fixed arms, limits and next decisions.
Completed first-pilot outcome: [RESULTS_PILOT001.md](RESULTS_PILOT001.md). Both arms
failed useful acquisition; old-input KL restored two binding answers but did not remove
all measured forgetting. The research goal remains unachieved.
The executable is [pilot.py](pilot.py); pure data/scoring is [core.py](core.py).
The supervisor's independent audit and cross-scale plan are in
`C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/research/continual_skill_audit_v1/`.

## Current frozen run

`runs/pilot004/PLAN.json`, SHA256:
`d044e060f7f8ae8e68e3cc6f18c715047e156a007df300d3b60a21b4ab1550b7`.

Two FP32 tail20 rank16 LoRA arms from exact Supa1.7: ordinary new-task CE versus
CE plus independent old-input KL. Each gets the same384 new examples and28 optimizer
updates. Old anchors and development each have168 examples in disjoint worlds and
templates. Acquisition is all384 SELECT-VIEWS-001 examples. The scripts never change
historical banks, artifacts or weights and never use protected confirmation data.

Stages, sequentially from the repository root with the existing Chat-Bubble Python:

```powershell
$py = 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe'
$run = 'research\composable_skills_v1\runs\pilot004'
& $py -m research.composable_skills_v1.pilot baseline --run $run
& $py -m research.composable_skills_v1.pilot train-control --run $run
& $py -m research.composable_skills_v1.pilot train-anchor --run $run
& $py -m research.composable_skills_v1.pilot eval-control --run $run
& $py -m research.composable_skills_v1.pilot eval-anchor --run $run
& $py -m research.composable_skills_v1.pilot summarize --run $run
```

These are create-only commands. Do not rerun completed or failed stages. Check both
`COMPLETE.json` and `supervisor.json` before continuing. Any scientific change needs a
separate prospective plan; existing results stay attributed to their frozen source.

## Verified setup

-11 passing software tests, including native tiny-Qwen selected-logit API mechanics.
-384 historical training prompts reproduce byte-for-byte as token IDs in the installed
  tokenizer. Actual FP32 base/adapter dtype and frozen-head constraints are checked.
-Source anchors:144/168 correct; binding110/112, arithmetic25/28, clock9/28.
  Binding has54/56 correct answerable and56/56 correct missing-evidence responses.
-Selected-token teacher-cache reconstruction has exactly0 maximum absolute logit error.
-Original/old-correct losses are reported separately from gains and net accuracy.
-The pilot uses its own matched HF FP32 path; it does not inherit DenseReference or
  native deployment qualification.

## Preserved engineering failures

`pilot001`: Windows Python redirector broke an immediate-parent ownership check.
`pilot002`: Transformers5 changed the chat-template default to return a dictionary.
`pilot003`: the cache check compared different LM-head GEMM row counts.
Each has an immutable log, receipt, source snapshot and failure note. The fixes changed
execution mechanics, not data, hypotheses or hyperparameters after candidate outcomes.
No training updates occurred in those attempts. Their62.234seconds of supervised wall
time and168 source-anchor generations are separately recorded in
`runs/ENGINEERING_OVERHEAD.json`.

## What would still remain after a successful pilot

Useful acquisition plus fewer observed losses would justify independent-draw replication
and acquisition/norm-matched controls. Then test acquisition of a second independently
qualified skill, both skill orders, retention of every earlier skill, and tasks requiring
both skills together. Expand preservation beyond these synthetic families and test an
untouched confirmation bank on the actual exported precision. Transfer the procedure,
not raw deltas, to a smaller independently qualified base. None of those claims is
established by this first pilot alone.
