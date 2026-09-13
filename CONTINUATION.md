# RAB-001 continuation brief

You are taking over an in-flight research programme. **Verify every factual claim in this
document before you act on it.** It was written from a session that has since ended, runs may
have finished or died, and several numbers in this programme's own history have already been
found wrong by audit. Treat this file as a set of leads with verification commands attached,
never as a source of truth. Where this document and the repository disagree, the repository
wins and you should say so.

---

## 1. Environment

| | |
|---|---|
| Worktree (work ONLY here) | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1` |
| CPU runtime | `C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe` |
| GPU runtime | `C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe` |
| Hardware | 8 GiB VRAM, 15.2 GiB host RAM (≈7 GiB free at rest) |

**Do not `cd` to the main `Chat-Bubble` checkout.** Never create a venv under
`experiments/architecture_benchmark_001/`.

**Run ONE model job at a time.** A 1.156B arm is 4.31 GiB in fp32 and its probe runs on the CPU;
two concurrent jobs will thrash or trip `hostbudget`. Training runs on the GPU and probes run on
the CPU deliberately — the frozen reference is a CPU measurement and CUDA/CPU FP32 differ by
~1e-4, enough to flip a greedy argmax and manufacture a fake retention loss. Every run starts
with a **device-integrity control**: the untrained arm, moved GPU→CPU, must reproduce the frozen
reference on all 150 rows or the run aborts. If that control ever fails, stop and investigate;
do not interpret any number from that run.

TF32 must be forced off (cuDNN defaults it ON and it costs 5.42e-02). `force_tf32_off()` does
this and records the result in every artifact — check the `numerics` field.

---

## 2. Verify these before trusting anything

Run each; do not proceed on the remembered value.

```bash
cd /c/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe

# a) implementation digest — every claim below is tied to it
$PY -c "import sys;sys.path.insert(0,'.');from experiments.architecture_benchmark_001.gates import implementation_digest;print(implementation_digest()[:8])"
# expected a329eb7b. If it differs, SOMEONE EDITED A HASHED FILE and every frozen
# comparison in this programme is void until the ladder is requalified.

# b) no hashed file is dirty
git diff --stat HEAD -- experiments/architecture_benchmark_001/

# c) what runs actually exist
ls experiments/architecture_benchmark_001/evidence/retention-design-v1/ARM_DOSE_*.json
ls experiments/architecture_benchmark_001/outputs/arm-checkpoints/ 2>/dev/null

# d) is a model job still running?
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object {\$_.WorkingSet64 -gt 1GB} | Select-Object Id,WorkingSet64,CPU"

# e) memory headroom before launching anything
$PY -c "import sys;sys.path.insert(0,'experiments/rab001_exploration');import hostbudget;print(hostbudget.describe())"
```

The official weights live at
`C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze`.
A `Supa/` directory of **NTFS hardlinks** to those files exists in the worktree root because
D28-001's `guards.binding()` requires `REPO/Supa`. It is **untracked and must never be
committed** (1.2 GB of weights) — so **never run `git add -A` in this worktree**; stage explicit
paths only. Verify it is intact and not symlinks before relying on it.

---

## 3. Where things stand (verify §2c first)

All arms below ran the identical dose: lr 1e-5, 5,233 supervised tokens, 41 steps, seed 17,
identical rows in identical order. The untrained source scores **47/150**. Read the artifacts;
do not trust this table.

| arm | schedule | panel | kept/47 | +real | +fmt | lost | net | params | KV |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| B28 full | cosine | **73** | 41 | ? | ? | 6 | **+26** | 1.00x | 1.00x |
| P56 | wsd/0.95 | 52 | 32 | 12 | 8 | 15 | +5 | 1.94x | 2.04x |
| P56 | cosine | 48 | 34 | 7 | 7 | 13 | +1 | 1.94x | 2.04x |
| B28last8 | cosine | 47 | **46** | 1 | 0 | **1** | 0 | 1.00x | 1.00x |
| H56 | wsd/0.95 | 37 | 23 | 9 | 5 | 24 | −10 | 1.94x | 2.04x |
| H56 | cosine | 16 | 11 | 4 | 1 | 36 | −31 | 1.94x | 2.04x |

B28 full's artifact is `RETENTION_DOSE_gpu_lr1e-05_b5200_seed17.json` from the older harness; it
has **no per-row record**, so its gains cannot be split into strata and it cannot be paired.

Paired McNemar exact, two-sided, from `ARM_COMPARISON.json`:

    H ~ P             (cosine)   b= 5  c=37  n=42  p=4.4e-07   P better
    H ~ B28last8      (cosine)   b= 5  c=36  n=41  p=5.9e-07   B28last8 better
    P ~ B28last8      (cosine)   b=14  c=13  n=27  p=1.0000    no effect
    H+WSD ~ P+WSD                b= 8  c=23  n=31  p=0.0107    P better
    H+WSD ~ B28last8             b=14  c=24  n=38  p=0.1433    no effect
    P+WSD ~ B28last8             b=20  c=15  n=35  p=0.4996    no effect

**Two load-bearing conclusions.**

1. **Fixing the recipe changed the verdict by four orders of magnitude.** The H↔P
   cross-position penalty reads p=4.4e-07 under a 1.2-step warmup and p=0.0107 under WSD. Most
   of the apparent catastrophe was the optimizer. Anyone quoting 4.4e-07 as the architecture's
   verdict is overstating it enormously.
2. **At matched trainable budget the CED arms buy nothing measurable.** P56+WSD's 52 against
   B28last8's 47 is p=0.50. And **B28 full dominates P56+WSD on BOTH axes** — 87.2% retention
   against 68.1%, 32 gains against 20 — at half the parameters and half the KV cache. There is
   no trade-off frontier on which CED is competitive.

**The real frontier is inside the B28 family and nobody has measured it.** Training 8 of 28
blocks gives 97.9% retention and 1 gain; training all 28 gives 87.2% retention and 32 gains.
Everything between those two points is unexplored.

**Checkpoints on disk** (`outputs/arm-checkpoints/`), trainable tensors only, each with a
`soup_key`: `B28last8_lr1e-05_b5200_seed17`, `H_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95`,
`P_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95`. All three are in **different** soup groups, so none
is averageable with another. **B28 full's weights do not exist** — it predates checkpointing.

**Possibly in flight** (check §2d): `Hlast4` at lr 1e-5, b5200, wsd:0.2:0.2, beta2 0.95. Its
stdout was accidentally sent to /dev/null; the artifact still lands at
`ARM_DOSE_Hlast4_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json`.

---

## 4. Corrections already made — do not re-introduce them

These were all found by audit after being asserted confidently. If you find a document still
carrying an old value, fix the document.

1. **The grammar ceiling is 64/150, not 71/150.** `headroom_audit.py`'s `duration_seconds()`
   used to return each `<number> <unit>` component *and* pairwise sums, so `1 hour and 25
   minutes` (5100 s) was credited with asserting 1500 s. Seven wrong answers were miscounted as
   presentation rejections. Two guards now prevent it; the general one is that a row whose
   scorer reason is `value mismatch` was *parsed* by the grammar and can never be a presentation
   rejection.
2. **The dose frontier is not monotonic.** The 500k-token run peaks near **50k tokens** then
   declines (task hits 3, 7, 20, **23**, 15, 17, 19, 19) while `dev_ce` falls monotonically to
   0.357. Budget work must be a **bracketed search in ~10k–60k tokens**.
3. **`dev_ce` is not a capability proxy and must never pick a stopping point.** TRAIN and DEV are
   the same 36 reply classes; the panel is three *disjoint* classes. Most of the celebrated
   "likelihood/capability decoupling" is this distribution mismatch.
4. **The retention gate scores only loss.** Its ceiling is the source's 47/47, so a capability
   *gain* cannot register on it. "47/47 at init" is vacuous for H56/P56/D56 — with the read
   output projection zero-initialised they *are* the source at step 0, by algebra.
5. **The format confound runs both ways.** Several of H56's 36 lost rows are presentation
   losses, not arithmetic ones. Split losses by stratum, not just gains.
6. **The +26 result is transfer, not in-domain.** TRAIN/DEV contain **zero** rows from the three
   panel classes.

---

## 5. Discipline (non-negotiable)

- **Digest discipline.** `implementation_digest()` hashes every `*.py` under
  `experiments/architecture_benchmark_001/` plus `protocol.json`, `source_lock.json`,
  `requirements*.txt` and the listed amendments. Editing any of them invalidates the whole gate
  ladder. Do not edit them without explicitly deciding to requalify, and never while a run is
  building — that has already stopped a ladder mid-run once.
- **Thresholds are fixed before scores are seen.** If you add a comparison, pre-register it in
  `DISTINGUISHABILITY_PREREG.md` before running it.
- **Run directories are create-only.** Never delete a failed run; never delete an `ACTIVE.lock`
  without confirming no process is running. Three preserved `blocked.json` attempts live in
  `experiments/dual28_ced_v1/outputs/` and must stay.
- **Do not edit other packages.** `dual28_ced_v1`, `full_width_transition_v1`,
  `ced_native_loader_issue4_v1` and `cross_scale_remote_receipt_d82_v1` were merged in from
  other branches. Propose fixes; do not apply them here. (Known open example: D28-001's P1
  cannot run because `full_width_transition_v1/source.py` builds `Qwen3ForCausalLM` under
  `torch.device('meta')` with `load_state_dict(assign=True)`, leaving the non-persistent
  `rotary_emb.inv_freq` on meta; RAB-001 already solved this with `_materialize_native_buffers`
  in `architecture_benchmark_001/run.py`.)
- **Exploratory work stays outside the hashed root**, in `experiments/rab001_exploration/`. It
  opens no gate, writes no receipt, and licenses no release.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## 6. Tools you will use

```bash
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe   # GPU
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe          # CPU

# arm_dose.py <arm> <lr> <budget> <seed> <probe_fractions> <schedule> <beta2> <save>
#   arm      B28 | B28last8 | H | P | Hlast<N> | Plast<N>
#   schedule cosine:<warmup> | wsd:<warmup>:<decay>     (fractions of the token budget)
#   save     final | none    (writes ONLY trainable tensors + MANIFEST with a soup_key)
$PYG experiments/rab001_exploration/arm_dose.py H 1e-5 5200 17 1.0 wsd:0.2:0.2 0.95 final

# paired comparison, McNemar exact two-sided; refuses unmatched data order
$PY experiments/rab001_exploration/arm_compare.py H P

$PY experiments/rab001_exploration/headroom_audit.py          # no model, safe beside a run
$PY experiments/rab001_exploration/build_splits_onpanel.py    # tokenizer only
```

Each probe of a 1.156B arm costs ~379 s on the CPU, so a 5-point curve is most of the wall clock
of an H/P run; pass `1.0` for two probes when you only need endpoints. Probing does **not** touch
weights, optimizer state, or the data cursor, so probe count never changes an arm's trajectory.

---

## 7. The schedule — run in this order

Everything below is one model job at a time. Check `hostbudget` before each launch.

R0, R0b, R1 and R1b are **done** — their results are in §3. Remaining, in priority order:

| # | command | question it answers |
|---|---|---|
| R2 | `arm_dose.py Hlast4 1e-5 5200 17 1.0 wsd:0.2:0.2 0.95 final` | fewer read sites (28→4)? *may be running* |
| **R3** | **add last-12/16/20 arms, then sweep** | **the unmeasured depth frontier — highest value** |
| **R4** | `arm_dose.py B28 1e-5 5200 17 0.25,0.5,0.75,1.0 cosine:0.03 0.999 final` | **determinism check vs 73/150, AND the only way to get B28 full's weights and stratum split** |
| R5 | alpha-interpolation toward the source, no retraining | acquisition/retention decoupling |
| R6 | `build_splits_onpanel.py` | in-domain dev; **required** before any soup |
| R7 | `arm_dose.py B28 1e-5 5200 {23,41} …` + one at lr 3e-6 | soup diversity — first averageable set |
| R8 | `arm_dose.py B28 1e-5 25000 17 …` | best single model, inside the 10k-60k peak window |
| R9 | `model_soup.py` | does averaging beat the best ingredient? |

### R3 — the depth frontier (do this first)

`B28last8` already exists in `arm_dose.py` via the `_ARM_RE` pattern. Extend it to an arbitrary
tail (`B28last<N>`) and run N = 12, 16, 20. B28 probes cost ~150 s, not the 379 s a 1.156B arm
costs, so the whole sweep is well under an hour. This traces the curve between 97.9% retention /
1 gain and 87.2% retention / 32 gains, which is the only place in this programme where a genuine
capability/retention trade-off has been observed.

### R5 — decoupling acquisition from forgetting, without retraining

P56+WSD converts the hard `wrong_answer` rows at **14.3%** against B28last8's **1.2%** — twelve
times the acquisition rate — while retaining 68.1% against 97.9%. Those cancel in the paired
test. If they can be separated, that arm lands near 59/150.

Scale the saved trained delta by alpha in [0, 1] and re-probe. At alpha=0 the model is exactly
the source (47/47 kept); at alpha=1 it is the trained arm. **For P56 this is precisely a post-hoc
gate**, because the source's read output projection is zero, so alpha IS the branch magnitude —
no retraining, the checkpoint is already saved. For B28 it is standard robust-finetuning
interpolation and often recovers lost reference answers while keeping most gains. Reuse
`model_soup.py`'s averaging, with the source as one ingredient.

Related, all untried and all cheap:

- **weight decay.** Every run so far used `wd = 0.0`. For the CED read, decay pulls `o_proj`
  toward zero, which *is* the source, so it is an anti-forgetting regulariser costing one
  argument.
- **early stop at the knee.** P56 under cosine peaked at 2,610 tokens (50, kept 37) and
  *declined* by 5,233 (48, kept 34). The knee is already visible; more milestones locate it.
- **replay.** 6,492 unused on-panel rows exist (§ build_splits_onpanel). Mixing skill-exercising
  data into training is the standard fix for catastrophic forgetting.

### Do not lose what already works

`B28last8` is the retention champion and the only arm that came close to holding the gate open;
its checkpoint is saved. **B28 full is the best model in the programme and its weights do not
exist.** R4 regenerates them, checks determinism against the recorded 73/150, and finally
produces its per-row record so its 32 gains can be split into real versus formatting.

**R4 is a determinism check, not a formality.** CUDA backward uses non-deterministic atomics. If
R4 reproduces the recorded **73/150** exactly, determinism holds and regenerated weights are the
model that scored 73. If it lands elsewhere, say so — the weights are still usable as soup
ingredients but the historical number cannot be attributed to them.

### Model soup rules (R7/R8)

- Two checkpoints may be averaged only if their `soup_key` matches — `mode`, `read_tail`, source
  digest, and the hash of the tensor-name set. B28 (596M) and H/P (1.156B) are different shapes
  and must never be mixed.
- **Greedy soup:** sort candidates by held-out score, add one at a time, keep only if held-out
  improves.
- **Select on the in-domain dev from R3, never on the 150-row panel.** Selecting on the panel
  fits the soup to the test set and makes the reported number meaningless.
- Ingredients must be *diverse*. Every run so far uses seed 17 and therefore the identical data
  order; averaging near-identical fits buys nothing. That is why R5 varies the seed. Label seed
  variants as soup ingredients so they are never mistaken for controlled comparison arms.
- Do not soup a collapsed model. H56 at 16/150 is not an ingredient.

---

## 8. What a good result looks like, and what it does not

The bar is **73/150 at 1.00× parameters and 1.00× KV cache**, from ordinary transfer finetuning
of the 596M source. H56 costs **1.94× parameters and 2.04× KV cache**, so it must beat 73/150
outright — pre-registered as A7, an absolute total and not a net over baseline.

Report gains **split by stratum** (`wrong_answer` vs `format_only` from `HEADROOM_AUDIT.json`);
only `wrong_answer` gains count as capability. `ratio_proportion` has zero `format_only` rows and
is therefore the cleanest class in the panel.

State plainly if the answer is that no architecture arm beats simply training the existing model
on more data. That is a legitimate and valuable result, and this programme's worth depends on its
reports being trustworthy rather than encouraging.

Do not claim a general capability gain from any run trained on the on-panel split from R3 — that
converts the panel from a transfer benchmark into an in-domain one. Report both, labelled.
