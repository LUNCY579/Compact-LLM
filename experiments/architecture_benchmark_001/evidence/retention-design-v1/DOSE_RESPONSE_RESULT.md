# The gate fires while cross-entropy improves — and its threshold is too strict to certify anything

This is the test `G1_RETENTION` / `G2_RETENTION` were built for and had never had. Until now the
training-side gate was mechanically verified and behaviourally untested: the only trained arm it
had seen (`G2_CACHED_B28`, 25 SGD steps at lr 1e-5) changed **0 of 150** answers, so a PASS there
said the stage executed, not that anything was retained.

Two doses, 5,233 supervised target tokens each, AdamW, clip 1.0, seed 17, scored against the
frozen 47-item reference from the `G1_RETENTION_B28` receipt at digest `a329eb7b`.

## Result

| step | lr 1e-4 devCE | kept | lost | changed | | lr 1e-5 devCE | kept | lost | changed |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 5.5782 | 47/47 | 0 | 0 | | 5.5782 | 47/47 | 0 | 0 |
| 10 | 2.6423 | 25 | 22 | 145 | | 3.0608 | 37 | 10 | 128 |
| 19 | 2.2161 | **11** | 36 | 149 | | 2.6280 | 40 | 7 | 134 |
| 31 | 1.8106 | 16 | 31 | 149 | | 2.3293 | 41 | 6 | 139 |
| 41 | **1.6516** | 21 | **26** | 148 | | **2.2481** | **41** | **6** | 140 |

Both bare policies agree at every checkpoint, in both runs.

**1. The gate does what it was built to do.** At lr 1e-4 held-out CE improves monotonically by
**3.93 nats** — 5.5782 to 1.6516 — while the arm loses **26 of its 47** reference answers. Any
likelihood-shaped endpoint reads that run as an unambiguous success. The capability gate reads it
as a failure from the first checkpoint onward. `gate_fired_while_ce_improved: true`.

**2. It discriminates dose, not merely "training happened".** At lr 1e-5 the same budget loses
**6** answers against lr 1e-4's **26**, and the trajectories differ in shape: 1e-4 collapses to 11
kept and then partially recovers as the cosine schedule decays the rate, while 1e-5 degrades
gently to 41 and stabilises. Final retention 44.7% against 87.2%. The damage is dose-dependent
and partly reversible, which is consistent with this programme's "dose, not data" finding.

**3. But the registered threshold cannot certify either run, and that is a defect.**
`max_correct_to_incorrect_per_class` is **0**, so the gate FAILS at lr 1e-5 too — a dose this
programme previously characterised as *preserving* capability (6/6 arithmetic retained). On 47
reference items rather than 6 prompts, lr 1e-5 loses 6. Those are not contradictory: a six-prompt
probe cannot resolve a 13% loss. But it means that **as registered, this gate certifies no
training run that has ever been measured here.**

The threshold is **not** being moved. Choosing a tolerance now, having seen that 6 losses is what
a "good" run costs, is precisely the tuning `forbidden` lists and precisely what HOTFIX-004 had to
retract from HOTFIX-003. It is recorded as the gate's first calibration finding, for a
prospective revision with its own registration.

## Why the numbers are trustworthy

**Device-integrity control, both runs: 0 of 150 answers changed before training.** Training ran on
the GPU and every probe ran on the CPU, so an untrained model that crossed devices had to
reproduce the frozen CPU reference exactly — and did, 47 permissive / 41 strict. Had the transfer
flipped a single greedy argmax, the run aborts, because a device artifact and a capability loss
are indistinguishable once they reach the counts.

**The models genuinely moved**: 140–149 of 150 answers differ from the source, against 0 of 150
for the 25-step engineering lane. This is not the null-dose case again.

## The other thing this shows

At lr 1e-5, **140 of 150 answers changed** while only 6 reference answers were lost. Supervised
fine-tuning rewrites almost every answer's phrasing while mostly preserving its content. A
zero-loss rule scored by a **format-sensitive** grammar is therefore structurally fragile: the
grammar's own known defect — it rejects the source's `It ends at …` phrasing on 7 rows — bites
in exactly this regime, because a reference item whose phrasing drifts into an unparsed form is
recorded as a capability loss it is not. Some part of the 6 losses at lr 1e-5 may be that, and
this run cannot separate the two.

That makes the pending grammar review a prerequisite for calibrating the tolerance, not an
independent chore.

## How it was run, and why the GPU

Three earlier CPU-only attempts were stood down by `hostbudget.guard`. AdamW fp32 on 596M
parameters needs ~9.5 GB and this 15.2 GiB host keeps ~7 GiB for the OS, desktop applications and
another project's job. The GPU was the *less* contended resource — 6.90 GiB free — so the work
split by what each device was short of:

| | placement |
|---|---|
| parameters + gradients | GPU (gradients resident because the recipe clips on global norm) |
| Adam moments, 4.44 GiB | **1.85 GiB on GPU, 2.59 GiB on a disk memmap**, filled to whatever VRAM was actually free at startup |
| every probe | **CPU** |

The disk tier is the point: file-backed pages are evictable and never add to the commit charge
that killed the three CPU attempts. It costs I/O per optimizer step and buys a run that finishes.
Peak host stayed near 4.5 GiB; wall time was ~14 minutes per dose.

The probe is on CPU deliberately. CUDA FP32 differs from CPU FP32 by up to 9.97e-05 per logit with
TF32 disabled, which is inside single-forward parity tolerance but becomes a discrete choice 3,600
times per probe. And the one un-root-caused `CUBLAS_STATUS_EXECUTION_FAILED` in this programme
lives in exactly that workload shape — long runs of sequential cached-decode. TF32 is forced off
and asserted; cuDNN defaults it ON on this card and it costs 5.42e-02, 540× outside tolerance.

## What this does not establish

- **One seed, one budget, one panel.** 5,233 tokens is a small dose; 150 prompts of three numeric
  classes; greedy, 24 new tokens. A finite-panel measurement, not a population claim.
- **Not a receipt.** `retention_dose_gpu.py` is exploratory and opens no stage. The registered
  gate's own runs remain the `a329eb7b` receipts.
- **No claim that lr 1e-5 is safe.** It loses 6 of 47 here. Whether that is real forgetting or
  grammar fragility is unresolved, and resolving it needs the grammar review first.
- **Not comparable to the CPU-trained corpus.** Training ran on GPU; the 500k matched pair and the
  multi-seed frontier ran on CPU. Only the probes share a device with the reference.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/retention_dose_gpu.py 1e-4 5200 17
$PY experiments/rab001_exploration/retention_dose_gpu.py 1e-5 5200 17
```

Needs ~3.6 GiB free host, ~6.9 GiB free VRAM and ~3 GB of disk for the spill; refuses at startup
otherwise. ~14 min per dose. Artifacts:
`evidence/retention-design-v1/RETENTION_DOSE_gpu_lr{1e-4,1e-5}_b5200_seed17.json`.
