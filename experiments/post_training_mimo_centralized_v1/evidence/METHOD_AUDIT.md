# Central method inventory audit

Audit date: 2026-09-28. Inspected centralized source: `03db218ceebc61a2bc04f2482cf7e094045e01c1`.

This audit inspects source code and existing receipts. It does not rerun historical tests, load a model, execute training, or establish native qualification on the current host. No held-out dataset or FINAL bank was opened. The primary subject is the exact Titan-series Rineya 0.6B checkpoint to be bound by the central program; no 344M training gate applies.

## Preserved initial drafts

Before editing either CSV, both original files were copied byte-for-byte with create-new semantics into `evidence/superseded_initial_draft/`. Existing backups are never overwritten.

| Preserved file | Bytes | SHA256 |
| --- | ---: | --- |
| `METHOD_INVENTORY.csv` | 35262 | `4ca391ccc5775b0b65e52400cf11cbb0b5fbfee4e9ac56c249634e9022f58521` |
| `METHOD_STATUS_MATRIX.csv` | 29252 | `aad7f67dfba5e02b5cd6925af31f6fbda19ec16841919ad5ccc04709ae8df6db` |

The existing CSV headers and evidence-file SHA256 convention are retained. Both tables now contain the same 68 method IDs, including four additions. `central_native_requalified` remains `false` for every row. A `VERIFIED SOURCE` row asserts inspected implementation, not newly reproduced execution or efficacy. Historical `NATIVE-QUALIFIED` rows retain their original source/runtime scope and do not authorize the central experiment.

## Corrections

- **LORA:** Replaced the stale 344M next gate with exact Titan-series 0.6B PRIMARY_STUDENT checkpoint/tokenizer and native forward/backward/resume qualification.
- **REPLAY:** `experiments/progressive_rl_curriculum_v1/backends.py` implements independent replay CE in its RL update. Its generic SFT entry point accepts supervised rows. `experiments/reasoning_posttraining_v1/preserve.py` is instead a torch-free, previously exposed DEV evaluation bank and scorer. It is removed from replay source paths and inventoried separately as `PRES_EVAL`. Reference KL on RL completions does not qualify a retention-reference-KL implementation for E2B.
- **GFLOW:** `experiments/gflowrl_reasoning_v1/QUALIFICATION_RECEIPT.json` explicitly reports a blocked native experiment. `CANONICAL_R0_BINDING.json` requires a qualified, bound reward-maximizing RL control; the existing SFT objective is not that control. `PAPER_FIDELITY_MATRIX.json` also leaves native ratio scaling, filtering, overlong shaping and aggregation unresolved. The gate now preserves these requirements.
- **SOFT:** The recorded `Qwen/Qwen3-0.6B` contract is compact, but it is not the exact Rineya Titan PRIMARY_STUDENT. The corrected wording preserves this identity distinction and the synthetic-only status.
- **MAXRL_DATA:** The row names verifiable task data explicitly; standalone estimator mechanics now have their own row.

## Independently inventoried mechanisms

| Added ID | Classification | Direct evidence | Boundary |
| --- | --- | --- | --- |
| `GRPO` | MECHANICS-ONLY | `experiments/progressive_rl_curriculum_v1/objectives.py` | Clipped sequence-mean policy loss and the update path in `backends.py`; no central native qualification |
| `MAXRL_ESTIMATOR` | MECHANICS-ONLY | `experiments/maxrl_verifiable_data_v1/maxrl.py` | Binary-reward estimators and prompt/policy/group/cost checks; no native trainer |
| `NGU` | MECHANICS-ONLY | `experiments/rola_ngu_factorial_v1/sampling.py` | Bounded retries, staleness and token reservations; fixture execution does not establish native throughput |
| `PRES_EVAL` | EVALUATION-ONLY | `experiments/reasoning_posttraining_v1/preserve.py` | Exposed DEV preservation/abstention scoring; no replay optimization or fresh confirmation evidence |

The historical combined `RNGF` factorial remains recorded. The separate NGU row allows the central program to establish an objective first and then compare sampling with RoLA disabled. No native source binding from the historical Supa factorial transfers automatically to Titan.

## Objective conventions requiring an explicit E3 choice

`GRPO` is not a complete mathematical specification. The inspected implementations differ:

- PRLC `objectives.py::advantages` uses population variance (`ddof=0`) without an additive denominator epsilon; its loss defaults to symmetric clipping `0.2` and reference-KL coefficient `0.02`.
- MaxRL `maxrl.py::advantages(method="grpo")` uses sample standard deviation (`ddof=1`) plus epsilon `1e-6`; its MaxRL estimator divides centered rewards by mean reward plus epsilon.
- RoLA/NGU `objective.py::policy_loss` defaults to asymmetric clipping `0.2/0.28` and reference-KL coefficient `0.01`. Its sampler supplies its own count/baseline and staleness-adjusted advantages.

E3 must freeze advantage construction, normalization, clipping, KL, reduction, masks, grouping, behavior-policy handling and total generation cost. Reusing the same name does not establish matched controls.

## Verification and remaining scope

All original evidence paths existed and their recorded SHA256 values matched the inspected files. Shared fields in the original two tables agreed. Post-edit verification passed: 68 unique and equal method IDs, unchanged headers, identical shared fields, valid classifications, 68/68 matching evidence hashes, 2/2 matching original backup hashes and `central_native_requalified=false` throughout. Parsed-field comparison against the preserved originals confirmed exactly five corrected existing rows, 59 unchanged existing rows and four additions.

The inspected PRLC, harness, Titan soup, CSS, CLARE, JumpLoRA, cross-scale/SCP, MaxRL data, RoLA/NGU, GFlowRL, SoftToken, F-WANDA, Dream-RSI, factorized memory, NSA, LoopSpec and Opallo records otherwise support their stated evidence boundaries. Historical native qualification is not native requalification, and neither mechanics tests nor collection establish a capability gain. This was a targeted source/receipt audit, not a new experiment or exhaustive paper reproduction.
