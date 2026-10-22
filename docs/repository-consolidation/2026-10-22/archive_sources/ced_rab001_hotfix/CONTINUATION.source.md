# RAB-001 continuation brief

**Verify every claim here before acting on it.** This programme has a long record of
confident claims that later measurement refuted -- roughly a dozen in one session, all
corrected in git. Treat this as leads with verification commands attached. Where this file
and the repository disagree, the repository wins, and say so.

---

## 1. Environment

| | |
|---|---|
| Worktree (work ONLY here) | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1` |
| CPU runtime | `C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe` |
| GPU runtime | `C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe` |
| Hardware | 8 GiB VRAM, 15.2 GiB host RAM |

Do not `cd` to the main `Chat-Bubble` checkout. **One model job at a time.** `Supa/` and
`rineya_reasoning_forge_v1/` are gitignored, as is every `*.safetensors` under the vault.

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
$PY -c "import sys;sys.path.insert(0,'.');from experiments.architecture_benchmark_001.gates import implementation_digest;print(implementation_digest()[:8])"   # expect a329eb7b
git diff --stat HEAD -- experiments/architecture_benchmark_001/     # expect empty
```

### Two host facts that will cost you hours if you do not know them

* **Another Claude session runs CUDA jobs on this host**, from
  `SupaExperiment-nonrecurrent-adaptation-v1`. Contending risks OOM-ing *its* run as well as
  yours. Check before starting anything: `Get-CimInstance Win32_Process -Filter "Name =
  'python.exe'" | Where-Object { $_.CommandLine -match 'SupaExperiment' }`.
* **`bc` is not installed.** Use `awk` for numeric comparison in shell gates. A wait loop
  using `bc` silently evaluates false forever; that cost 90 minutes here.
* **Gate on 4.15 GiB, not 3.6.** `arm_dose` calls `hostbudget.require(3.6)` *after*
  importing torch, and the interpreter costs ~0.5 GiB, so a pre-launch reading of 3.97
  becomes 3.48 at the check.

---

## 2. The deliverable is shipped

**`Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze`** -- a dense HF `model.safetensors`,
Supa 1.7 with blocks 8-27 finetuned under WSD. Registered in `CATALOG.json` and
`MODEL_REGISTRY.md`.

Published number is **73/150, not 74**: the checkpoint is float32, the vault BF16, and the
cast costs one panel row (140/150 byte-identical outputs, 150/150 identical termination).
Thresholds were fixed in `OPALLO_PREREG.md` before measuring.

```bash
$PY experiments/rab001_exploration/materialize_opallo.py --tag <ckpt> --codename X --version 1.0.0-devfreeze
$PY experiments/rab001_exploration/validate_published_core.py --core <dir> --against <tag> \
    --min-correct 70 --min-retained 38
```

---

## 3. What is established

| finding | evidence |
|---|---|
| WSD worth +7.5 panel rows on tail arms | 4/4 seeds, pooled **p=0.00029** |
| WSD is arm-specific | nothing on full B28 (p=0.70) |
| Depth matters | B28 > last8 (p=1e-4), > last12 (p=0.004) |
| Full-parameter training buys nothing over last20 | 76 vs 74, 16-14 discordant, **p=0.8555** |
| Cross-position CED memory harms | P beats H at equal params, p=0.0107 |
| **Reasoning ceiling is not a long-context artifact** | puzzle probe, Supa **102/450** vs floor **0.229** |
| **Opallo lost abstention, not retrieval** | `UNKNOWN` 13 -> 2 of 64; answerable rows 149 -> 150 |
| Reasoning data does not help | 11.5x dose, IID 34 -> 34, p=0.868 |
| Weight averaging inert | soup lands at its best ingredient |

### The finding that should drive the next decision

`splits_large` holds **10,000 rows**; the deliverable used **41** of them -- 0.41%. The four
seeds drew **completely disjoint** row sets scoring 74/62/73/68, and the panel's three
classes appear in **none** of the training data. Selection dominates volume at this dose, and
volume is measured to hurt (15,600 -> 64; 25,000 mixture -> 48).

### Noise floors -- read every result against these

| instrument | discordance |
|---|---|
| panel, 150 rows | **15-24** (mean 18.8); one same-config pair hit p=0.0352 on seed alone |
| forge `test_iid` / `test_transfer`, 200 | 7-12 / 5-10 |
| puzzle, 450 rows | majority-guessing floor **0.229** |
| long-context, 256 rows | deterministic; use paired McNemar |

---

## 4. Tools

```bash
arm_dose.py <arm> <lr> <budget> <seed> <probe_fractions> <schedule> <beta2> <save> [label]
build_splits_curated.py --rule random|balanced|compositional|basic|reasoning|diverse|repair
long_context_probe.py --model <dir> [--checkpoint TAG] --label NAME
puzzle_probe.py [--build-sample] --model <dir> [--checkpoint TAG] --label NAME
forge_probe.py --split test_iid|test_transfer --rows N --checkpoint TAG --label NAME
arm_compare.py <tag> <tag>            # WARNING: overwrites ARM_COMPARISON.json every run
materialize_opallo.py / validate_published_core.py / _register_exploration_core.py
```

Budgets must be `<= max_usable_budget` in the split manifest. `arm_dose` counts *supervised*
targets (len-1 per row).

---

## 5. IN FLIGHT -- the repair experiment

**Blocked on host memory, not on design.** Everything is built, guarded and pre-registered;
nothing has trained.

`REPAIR_PREREG.md` fixes the test. Treatment `splits_curated_repair_a33_b5233_s17` (42 rows,
14 abstention, 5,302 supervised tokens) against control `splits_curated_random_b5233_s17`
(41 rows, which reproduce the deliverable's draw *exactly, in order*), at seeds 17/23/29.

Primary test requires **both**: treatment mean `UNKNOWN` **>= 7 of 64** *and* pooled paired
McNemar **p < 0.05**. Retention constraint: mean panel **>= 62**, mean retained **>= 37**.

Run it with:

```bash
bash <scratchpad>/run_repair.sh      # gates on 4.15 GiB, retries on refusal, skips finished arms
```

or directly, once the host has headroom:

```bash
RAB001_SPLIT_DIR=splits_curated_repair_a33_b5233_s17 $PYG \
  experiments/rab001_exploration/arm_dose.py B28last20 1e-5 5302 17 1.0 wsd:0.2:0.2 0.95 final repairA33
```

Then score each arm with `long_context_probe.py --checkpoint <tag>`, which reports
`says_unknown_on_unanswerable` directly.

**The blocker: host memory, and it got worse over the session.** It read 8.14 GiB free when
the deliverable trained, 3.9-4.0 GiB for most of the repair attempts, and **1.04 GiB** by the
end -- 93% memory load, 23.5 GiB committed against 15.2 physical. `arm_dose` needs 3.6.

Three things worth not re-deriving:

* It is **not reclaimable cache**. Forcing a 3 GiB allocation to evict standby pages moved
  free memory by 0.02 GiB.
* `GlobalMemoryStatusEx.ullAvailPhys` (what `hostbudget` uses) and
  `Win32_OperatingSystem.FreePhysicalMemory` (convenient from PowerShell) **agree** on this
  host -- 1.04 vs 0.99 GiB measured simultaneously. Either is a fine gate. I assumed they
  would diverge and they do not.
* Working sets sum to only ~7 GiB of 15.2, so no single process is the culprit; the commit
  charge is spread across the browser and the several Claude sessions.

**Closing Chrome and any idle Claude sessions unblocks this immediately.** Do not raise or
bypass the `hostbudget` floor to get around it: at 93% load and 8.2 GiB over physical, this
is exactly the thrashing scenario that guard was written from, which cost a measured 10x.

---

## 6. What is worth doing next, and what is not

**Not worth doing.** Reasoning data (falsified on two instruments now). Puzzle training --
the pre-registered rule fires STOP at 102/450. More weight averaging. More single-seed
comparisons. Further CED arms.

**`logic_grid` is not a loophole.** It is the one puzzle family of fifteen that beats its
floor under Bonferroni. Selecting it *after* seeing which family that is would be the
post-hoc selection this programme has withdrawn seven findings for. It needs its own
pre-registration and its own held-out families.

**Worth doing.**

1. **Finish the repair experiment** (section 5). It is the only open confirmatory test.
2. **The five untested selection rules** -- `balanced` (36/36 classes), `diverse`,
   `compositional`, `basic`, `reasoning` -- are built, guarded, and never run. Each needs
   three seeds; single-seed comparisons on this panel are uninformative.
3. **Abstention may be cheaper to bake than to train.** Supa-A1 ships a baked logit bias on
   token 51378 (`UNKNOWN`) and scores 43/64 Showcase. If the repair mixture fails, that
   mechanism is the alternative and it needs no training at all.
4. **Fix `arm_compare.py`'s fixed output path.** It silently destroys the previous
   comparison's evidence; it did so here and the file had to be restored from git.

**The honest ceiling.** The panel went 47 -> 73. General reasoning did not move under any
intervention on *two* independent instruments, one of which was chosen specifically to break
the long-context confound and failed to. That is a capacity limit. What is *not* a capacity
limit, and is the live opportunity, is the abstention behaviour Supa has and Opallo lost.
