# RAB-001 continuation brief

Verified continuation started 2026-09-13. Work only in this worktree, on
`research/ced-rab001-hotfix001-qualification-v1`:
`C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification`.
Do not change to the main Chat-Bubble checkout. Treat this brief as leads; artifacts win.

## 1. Verify before acting

Use PowerShell from the worktree root:

```powershell
git branch --show-current
git status --short --untracked-files=normal
git diff --stat HEAD -- experiments/architecture_benchmark_001/
& 'C:\Users\clclc\BusinessProject\.venv-rab001\Scripts\python.exe' -c "from experiments.architecture_benchmark_001.gates import implementation_digest;print(implementation_digest())"
Get-ChildItem experiments/architecture_benchmark_001/evidence/retention-design-v1/ARM_DOSE_*.json
Get-ChildItem experiments/architecture_benchmark_001/outputs/arm-checkpoints
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {$_.CommandLine -match 'rab001|architecture_benchmark'} | Select-Object ProcessId,WorkingSetSize,CommandLine
& 'C:\Users\clclc\BusinessProject\.venv-rab001\Scripts\python.exe' -c "import sys;sys.path.insert(0,'experiments/rab001_exploration');import hostbudget;print(hostbudget.describe())"
```

Expected full implementation digest:
`a329eb7bb06fc003a9220ea41713e721fba575c478008240749c425774b2b58e`.
A changed digest requires investigation and requalification before frozen comparisons.
Evidence markdown/JSON is not hashed implementation; every Python file under
`experiments/architecture_benchmark_001/` is. Exploratory code belongs outside that root.

The seven `Supa/` assets were verified against every source_lock SHA-256 on 2026-09-13;
all are ordinary files with two NTFS hardlinks, not symlinks. Their other links are in the
published source directory. NEVER modify them. `Supa/` is untracked; NEVER `git add -A`.

## 2. Execution discipline

- CPU runtime: `C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe`.
- GPU runtime: `C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe`.
- Hardware: 8 GiB VRAM, 15.2 GiB host RAM. ONE model job at a time. Inspect process command
  lines and hostbudget before every launch. Free cores are not free RAM.
- Train on GPU, probe on CPU. Force TF32 off. At initialization compare all 150 output texts
  AND termination flags with the frozen CPU source. Stop on a mismatch.
- Run/checkpoint directories are create-only. Preserve failures, partial results and locks;
  never remove an ACTIVE.lock while its process is alive. Never edit a running harness.
- Do not edit dual28_ced_v1, full_width_transition_v1, ced_native_loader_issue4_v1, or
  cross_scale_remote_receipt_d82_v1 in this task. Source files and published cores are immutable.
- Preregister comparisons before new scores. The fixed continuation families and recipes are
  in `experiments/architecture_benchmark_001/evidence/retention-design-v1/DISTINGUISHABILITY_PREREG.md`,
  continuation addition committed as `2effac5` before new jobs from this session.
- Stage explicit paths, commit and push. End commit messages with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## 3. Verified inherited endpoints

These six ARM_DOSE artifacts are completed, have source-matching initialization outputs,
CUDA training / CPU probes, and TF32 disabled. They consumed 5233 targets / 41 steps / seed17.
B28 full below is from the older RETENTION_DOSE harness and has aggregates only.

| Arm | Recipe | Correct/150 | Kept/47 | Gains: wrong/format/runaway | Lost |
|---|---|---:|---:|---|---:|
| B28 full (historical) | cosine .03, beta2 .999 | 73 | 41 | unknown split; 32 total | 6 |
| B28last8 | cosine .03, beta2 .999 | 47 | 46 | 1 / 0 / 0 | 1 |
| B28last12 | cosine .03, beta2 .999 | 52 | 44 | 8 / 0 / 0 | 3 |
| B28last16 (inherited run, now complete) | cosine .03, beta2 .999 | 55 | 40 | 9 / 6 / 0 | 7 |
| P | WSD .2/.2, beta2 .95 | 52 | 32 | 12 / 8 / 0 | 15 |
| P | cosine .03, beta2 .999 | 48 | 34 | 7 / 7 / 0 | 13 |
| H | WSD .2/.2, beta2 .95 | 37 | 23 | 9 / 5 / 0 | 24 |
| H | cosine .03, beta2 .999 | 16 | 11 | 4 / 1 / 0 | 36 |
| Hlast4 | WSD .2/.2, beta2 .95 | 49 | 47 | 1 / 1 / 0 | 0 |

The headline scorer is the registered permissive policy. Hlast4 passes that finite-panel
retention rule, but its strict sensitivity result FAILs: 40/41 retained, one ratio loss.
Its +2 versus source has exact p=.5 and does not establish an acquisition benefit.

Checkpoints existed for B28last8, H+WSD, P+WSD and Hlast4+WSD at takeover. Historical full B28
weights did not exist. Check the current directories rather than assuming that remains true.
The inherited B28last16 checkpoint is also now saved (251,695,104 trainable parameters;
SHA-256 `52f2ff047b310021ad195f3e288cffc664f56bff2423f45d5f5a117c5f8f8074`).
Its old manifest incorrectly says complete_model=true; it contains only the last16 blocks.
Preserve that historical manifest and reconstruct from source plus named trainable tensors.
The updated harness correctly labels new tails as partial checkpoints.

## 4. Corrections to the inherited narrative

1. Hlast4 versus B28last8: 146/150 correctness decisions agree, but only 70/150 output texts
   match. The models are not established to be behaviorally identical. Exact McNemar is
   3 versus 1 discordant, p=.625.
2. Hlast4's observed zero loss does not prove that injection-site count causes forgetting,
   that branch strength is irrelevant, or that the arm is undertrained. Site position,
   trainable count and learned activations also changed. Norm/sqrt(number of sites) is an
   RMS weight norm, not measured residual injection magnitude. Larger-dose Hlast4 is a
   possible later hypothesis test, not a conclusion from zero losses.
3. The six-pair WSD/Hlast4/B28last8 comparison family used uncorrected tests despite the
   preregistration requiring correction for added comparisons. P+WSD versus H+WSD raw
   p=.01067384053 becomes Holm p=.06404304318. None of those six pairs survives correction.
   This correction is retrospective; do not call those added comparisons preregistered.
4. WSD P/H and cosine B28last8 share dose/seed and approximately trainable budget, but not
   schedule or beta2. Do not describe that as an identical-recipe architecture comparison.
5. Historical B28 has higher observed retention and acquisition than P+WSD. It cannot be
   paired without per-row outputs. These results do not establish universal CED inferiority.
   Inference parameter/cache counts are analytic costs, not measured speedups.
6. The historical cosine H versus B28last8 5-36 discordance has p=7.841572369e-7; the previous
   brief's 5.9e-7 was incorrect. Raw p-value ratios do not quantify optimizer causation.
7. Scaling all P trainable deltas is NOT the same as gating its trained branch: memory K/V,
   query and normalization tensors also train. An exact per-site post-hoc gate holds all
   trained tensors fixed and scales only reads.*.o_proj.weight. B28 source interpolation
   and P output gating must carry distinct labels.
8. Repeating B28's aggregate 73/150 cannot establish determinism or recover identifiable
   historical weights. It can establish aggregate replication and bank a new checkpoint.
9. B28last28 (blocks only) would still freeze embeddings/final norm; full B28 trains them.
   The full-model endpoint is not a pure depth-only intervention.
10. Source headroom is 84 wrong_answer / 17 format_only / 2 runaway; grammar-limited ceiling
    is 64/150, not 71. Split losses as well as gains. Source-stratum conversion is evidence
    on these rows, not a general reasoning certificate.
11. TRAIN/DEV use 36 off-panel classes, with zero rows from the three panel classes. The
    historical improvement is transfer. Dev CE cannot choose an endpoint. The longer-dose
    result is nonmonotonic and motivates a 10k-60k bracket, not indefinite training.
12. There are 6492 spare on-panel corpus rows, but after calibration/pilot exclusion there
    are 6416 across 965 families. The default builder samples ONE row per family: 845 TRAIN
    and 120 DEV. It does not train on all 6492 rows. Actual monitor_probe and task_probe are
    identical decoded rows, IDs and families; the builder's exclusion replay was verified.
13. OffloadedAdam currently has no weight-decay argument or update. A WD experiment needs
    an implemented and validated optimizer change; merely exposing a CLI argument would
    not apply decay. No such experiment is part of the current fixed schedule.

## 5. Work underway and remaining order

At takeover an external B28last16 run was active and arm_dose.py already had an uncommitted
B28last<N> extension. The user explicitly authorized taking over subsequent runs after it
finishes. Preserve that run and inspect its final artifact before launching another. The
continuation registration records that training preceded this session's registration.

Current session update: B28last16 completed in 885.4 seconds. B28last12 completed in about
355 seconds through its final probe, with 52 correct /44 kept /8 gains /3 losses; all gains
and losses are wrong_answer. Both checkpoints' on-disk SHA-256 values were independently
verified. The new frontier reporter validated both artifacts and preserved the incomplete
three-comparison snapshot in outputs/depth-through-last12-20260913.json.

B28last20 was launched next with endpoint-only probes and final save; inspect
`experiments/rab001_exploration/outputs/B28last20_20260913.log` and processes for its current
status. Its prelaunch host budget passed. Do not change arm_dose.py while it is active:
that script hashes its own bytes and refuses evidence after an edit.

P alpha memory: idle free physical RAM was about 5.5 GiB, below the P runtime+model envelope.
The user has been asked to free about 1.5 GiB by closing unused applications, or leave P
pending while B28 work proceeds. Do not lower hostbudget thresholds or start a second job.

R6 data preparation is complete: outputs/splits_onpanel contains 845 TRAIN rows /111,795
targets and 120 DEV rows /502 reference-answer tokens. No models have trained on this split.
The exact first60 selection rows were frozen before candidate scoring in
`experiments/rab001_exploration/outputs/SOUP_DEV_FROZEN_20260913.json`, selected-row canonical
SHA-256 `3e1ec82e4de837526264633921e520551dfb643e94db89cf966f53ed593f3579`.
Verify this hash against model_soup's selection plan before interpreting a soup.

New tooling: alpha_sweep.py (one fixed checkpoint/grid), frontier_report.py (ten depth
comparisons), alpha_report.py (twelve alpha comparisons), and the repaired model_soup.py.
All are under experiments/rab001_exploration. Their tests are mechanical checks only, not
model evidence. The alpha harness uses --kind b28-wise or p-output-gate and a create-only
--out under experiments/rab001_exploration/outputs. Soup requires --run-id and explicit tags.
headroom_audit.py now exposes a model-free failure_stratum helper; the original source
strata reproduce exactly under both policies. Frontier/alpha reports split losses fully
using that same helper and record its hash. Do not regenerate/overwrite historical audit
JSON merely to use the helper. Mechanical validation passed 51 tests: tail boundaries (3),
alpha metadata/tensor endpoint interpolation (11), soup guards/streamed roundtrip (11),
frontier+alpha reporting and source/loss strata (26). These are not model-quality evidence.

1. Finish/audit the inherited B28last16, run B28last12 and B28last20 at fixed recipe.
2. Rerun full B28 with .25/.5/.75/1 probes and final checkpoint; record stratum split and
   compare aggregates with historical 73/41/32/6.
3. Fixed alpha grid 0/.25/.5/.75/1 on new B28 and saved P+WSD; endpoint controls first.
   Never choose alpha on the panel. See registration for twelve-test Holm family.
4. Build and freeze in-domain DEV and its first60 selection rows. Keep it family/ID-disjoint.
5. Bank full B28 soup candidates: seed23 and41 at lr1e-5/b5200, seed17 at lr3e-6/b5200,
   seed17 at lr1e-5/b25000. Include the new seed17/lr1e-5/b5200 baseline. All cosine .03,
   beta2 .999. Treat seed/dose variants as ingredients, not matched-order controls.
6. Greedy uniform soup selected by terminated normalized exact match on the fixed60 DEV;
   strict improvement only, lexical-tag ties. Final soup versus DEV-best single is one
   registered paired panel test. Save weights, all candidates, rejections and per-row outputs.

No population-level efficacy, official gate receipt, registered G3 training or release is
licensed by these exploratory runs. Report negative results and integrity failures plainly.
