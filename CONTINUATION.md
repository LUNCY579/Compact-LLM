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

**Completed and believed sound:**

- `ARM_DOSE_H_lr1e-05_b5200_seed17.json` — H56 trained at the matched dose. Ends **16/150**,
  kept 11/47, gained 5, lost 36, net **−31**. Trajectory 47 → 2 → 6 → 17 → 16 as the cosine
  decayed. Device-integrity control passed.
- `HEADROOM_AUDIT.json` — of the 103 rows the source fails: **84 wrong_answer, 17 format_only,
  2 runaway**. Grammar-limited ceiling **64/150**.
- `DISTINGUISHABILITY_PREREG.md` — thresholds A1–A7, fixed before any arm was trained.
- Long-budget evidence already on disk:
  `outputs/longbudget/LONGBUDGET_source_control_seed17_b500000_lr1e-05.json`.

**In flight when this was written** (may have completed or died — check §2c/§2d): P56 at
lr 1e-5, b5200, cosine, 5 probes. At its 25% milestone it read
`kept 24/47, gained 14, lost 23, correct 38/150, devCE 3.2466` against H56's
`kept 1/47, gained 1, lost 46, correct 2/150, devCE 3.2601` at the same point. If it finished,
its artifact is `ARM_DOSE_P_lr1e-05_b5200_seed17.json`.

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

| # | command | question it answers |
|---|---|---|
| R0 | *(in flight)* `arm_dose.py P 1e-5 5200 17 0.25,0.5,0.75,1.0` | branch vs context |
| R1 | `arm_dose.py H 1e-5 5200 17 1.0 wsd:0.2:0.2 0.95 final` | does correct technique rescue H? |
| R1b | `arm_dose.py P 1e-5 5200 17 1.0 wsd:0.2:0.2 0.95 final` | **control for R1** — required |
| R2 | `arm_dose.py Hlast4 1e-5 5200 17 1.0 cosine:0.03 0.999 final` | is it simultaneity (28→4 reads)? |
| R3 | `build_splits_onpanel.py` | in-domain dev, needed for soup selection |
| R4 | `arm_dose.py B28 1e-5 5200 17 0.25,0.5,0.75,1.0 cosine:0.03 0.999 final` | **determinism check** |
| R5 | `arm_dose.py B28 1e-5 5200 23 1.0 … final` and `… 41 …`, plus one at lr 3e-6 | soup diversity |
| R6 | `arm_dose.py B28 1e-5 25000 17 0.25,0.5,1.0 … final` | best single model, inside the peak window |
| R7 | greedy model soup over R4–R6 | does averaging beat the best ingredient? |
| R8 | CED soup — **only if R1 or R2 is non-catastrophic** | |

**R1b is not optional.** Without a P arm under the same schedule you cannot tell whether WSD
rescued H specifically or simply moved both arms.

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
