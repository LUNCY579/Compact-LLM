# CLARE-RINEYA audit (Phase 0) — what exists, what is reused, and why

Audited 2026-09-27 before any CLARE code was written, from the parent head
`origin/research/titan-model-soup-mini-v1` @ `816a1e8f4d1ab1c9d1ab847697d962a141521bac` (fetched the
same day; the parent branch is not modified). Seven read-only readers covered: the soup
infrastructure, the parent-branch mask and composition experiments, geometry and functional scoring,
the MAXRL-COMP-002 dataset, the native training lanes, the retention instruments, and the CLARE paper.
Facts that this study depends on were then re-verified first-hand (**V** in the tables below); the
rest are recorded as reported by the audit (**R**).

## 1. The parent branch: what it is

`experiments/titan_model_soup_mini_v1` (SOUP-MINI-001) is an offline FP32 checkpoint-interpolation
package: `soup.py` (pinned preflight plus a streaming 2-way lerp writer), `score.py` (paired report),
`screen.py` (frozen-receipt CAL/CHECK selector) and `collect_native.py` (fail-closed collector,
`native_authorized: false`). The branch then grew ten further experiments on delta composition:
subspace geometry, a fresh transition instrument, module-disjoint composition, functional additivity
(FA-001), CONFIRM-001/002, convention localisation, the interference mechanism, and OSIDE-001.

**None of it trains anything.** Every delta on the branch was trained elsewhere (the RRP/MQ native
lane) and composed afterwards. CLARE-RINEYA-001 is the first experiment on this line that controls
where a delta is learned.

## 2. Reused as-is (imported, never copied or edited)

| component | where | why | |
|---|---|---|---|
| shared GPU lease | `experiments/fresh_dual_skill_instrument_v1/gpu_lease.py` | lock semantics must match every peer session (same lock file as the MQ lane) | V |
| MAXRL-COMP-002 verifier/contracts | `Rineya-Dataset/experiments/maxrl_composable_skills_v2/` | the dataset's own scorer (any valid certificate, strict JSON); the package fingerprints its sources, so it is loaded in place under a private name with bytecode writing OFF | V |
| MAXRL-COMP-002 data | `Rineya-Dataset/data` (manifest `86ac8f0d…`) | not duplicated; every file hash-checked against the manifest | V |
| n = 47 retention panel | `ChatBubble-ced-native-loader-qualification/experiments/rab001_exploration/retention_probe.py` (rab001 venv) | the programme's only standing general-retention endpoint; its decode loop must not be re-implemented; called by subprocess (pending an operator decision, since it writes into that worktree) | R |
| detached launcher | `experiments/interference_mechanism_v1/wmi_runner.py` | long jobs survive session teardown only via WMI | R |

## 3. Vendored (byte-identical or verbatim, with provenance)

| component | source | in this package |
|---|---|---|
| `hostbudget.py` | `git show 42eb390b:experiments/rab001_exploration/hostbudget.py` (not an ancestor of this branch) | `vendor/hostbudget.py`, sha256 `9f9dcf35…` checked identical to the live copy |
| `rate_at`, `group_updates`, `compile_row` | `ChatBubble-retention-pilot-20260917` @ `5b4c22da` (files last changed `2cd46c16`) | `clare/recipe.py`; the only adaptation normalises transformers-5's `BatchEncoding` return type |
| `exact_mcnemar` | `titan_model_soup_mini_v1/score.py` @ `816a1e8f` | `clare/metrics.py`; a test compares the two functions' ASTs |
| FA-001 κ estimator and outcome rule | `functional_additivity_v1/amendment_1/amend_analysis.py` @ `6903e583` | `clare/functional.py` (fresh RNG per call, unclipped inputs) |

## 4. Deliberately NOT reused, and why

| component | reason |
|---|---|
| `soup.py` `stream_pair`/`build`/`ARMS` | only a 2-way lerp between full checkpoints, closed arm table, hash-pinned by `screen.py`; editing it would break SOUP-MINI-001's chain |
| `verify_independent.ST`, soup `local-runs/geometry.py` | gitignored; exist only in another session's live worktree. Four committed scripts on the branch import them and **fail with ImportError in a fresh checkout** (recorded here as an existing defect of the parent branch, not fixed) |
| the numpy mask builders (RAND_MASK, DARE, DISJOINT-DARE, module masks) | post-hoc masks over already-trained deltas, re-derived from seeds and **never saved**; CLARE needs training-time masks saved explicitly |
| the native-lane engine (`engine.microfit`) | pinned to torch 2.10/transformers 4.57.1, hardcodes the tail, peak LR 1e-5 and 220 tensors, and has no mask hook; running there also needs a registered id and an operator grant |
| `score_arms.py` generation | single EOS id, no TF32/memory-fraction guards; a different decode contract |
| `behaviour.enroll/evaluate` (dataset) | requires every row of a role (4,150 DEV rows) and refuses subsets; `metrics.damage` implements the same rule on the registered subset |

## 5. Built new (because nothing equivalent existed)

Everything in `clare/`: an exact-budget radix selection with seeded tie-breaking; packed,
explicitly saved masks with content checksums; a strict allocation state with save/resume; dense and
sparse delta containers, streaming writers and slot-at-a-time disk readers (host RAM is the binding
resource); `DeltaHost`, which trains masked deltas on any `nn.Linear` footprint; the CLARE two-stage
orchestrator with its A–F controls; BF16-storage/FP32-compute Qwen3 patches; the 2×2 interaction
scorer; and the analysis metrics.

## 6. Facts this study depends on

| fact | | evidence |
|---|---|---|
| Supa = Qwen3ForCausalLM 28L/1024/3072/151936, tied, 310 tensors, 596,049,920 params, BF16 | V | header read; `configs` copy; tiny-model parity tests |
| blocks 8–27 MLP = 60 tensors, 188,743,680 coordinates | V | `footprint.qwen3_mlp_footprint`, `cli.py plan` |
| TRAIN has R,U,S,P,RU,US,SP only; RS,RP,UP,all triples,RUSP held out | V | `contracts.py` `SEEN_PAIRS`/`TRAIN_SIGNATURES`; `data.py` gates |
| the four atomic banks share the same 3,000 parent worlds | R | dataset audit (set intersection = 3,000) |
| no SFT file for the train banks; targets = oracle `references` (3 per row) | V | `oracles/train.jsonl`; every sampled `references[0]` verifies |
| no model has ever been scored on MAXRL-COMP-002 | R | records and experiments grep |
| COMP-RS-001 (live) registered TEST/guard_test as one read per arm | R | its Amendment 1 |
| CLARE: two-stage, reset, `k_t = round((1−ρ)|A_t|)`, λ = 1e-4 SGD, random mask −0.51 (10 tasks) / −5.65 (100 tasks) | V | paper text (arXiv:2609.17026) |
| parent-line separation results (random mask = expectation, DARE = addition, module = scalar, orthogonality = nothing) | R | CONFIRM-001, INTERFERENCE-MECH, OSIDE-001, ARMPERP records |
| Titan recipe: FP32, TF32 off, AdamW (0.9, 0.95), clip 1, WSD 20/60/20, ≤ 2048-token groups, answer+EOS loss, peak LR 1e-5, 5.4 s/update | R | Titan CORE.json and steps logs |

## 7. Hazards carried into the design

1. **Host RAM is the binding resource** (15.2 GB shared by several live sessions): models load with
   `device_map` straight to the GPU; dense deltas are streamed to disk slot by slot and read back one
   slot at a time; `hostbudget.require/guard` gate every native command.
2. **Windows WDDM spills CUDA into host RAM**: `set_per_process_memory_fraction(0.78)`.
3. **TF32 defaults on** for cuDNN on this card: forced off and asserted.
4. **Driver resets (TDR)**: generation scoring is resumable per row; training resumes per skill.
5. **`enable_thinking`**: the prompt must end in the closed think stub (asserted).
6. **BF16 rounding erases deltas** (60.61 % of stored Titan's update): every CLARE delta is FP32.
7. **Line endings**: every artifact is written in binary mode with LF.
8. **Sibling answer leakage and high shortcut floors** in DEV: floors are reported per signature.
9. **The GPU can disappear.** During this session the NVIDIA adapter vanished from Windows (System
   log, 16:00:44–16:01:08Z: `nvlddmkm`, NVIDIA Display Container and NVPCF set to *disabled*; an
   `IOMap64.sys` kernel service installed), and returned later. Preflight now records the display
   adapters, so "no NVIDIA device" is distinguished from "nvidia-smi failed".
10. A dataset-audit reader accidentally wrote three `.pyc` files into
    `Rineya-Dataset/experiments/maxrl_composable_skills_v2/__pycache__/` (10:57:53 local). They are
    gitignored and outside the manifest; they were left for the operator rather than deleted.
