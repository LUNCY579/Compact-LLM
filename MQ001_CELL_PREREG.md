# MQ-001 first-cell pre-registration

Written **before** native qualification completed and before any cell was trained. Fixed here
so the choice cannot be made after seeing a number.

Deliberately at the repository root, **not** inside `experiments/model_quality_gpu_v1/`.
`code_identity()` hashes every file in that package — 28 files, currently
`abe7b2f6…` — and the runbook requires qualification and training to share one commit. A
note added there between the two would silently invalidate the qualification.

## Which cell runs first, and why

Six cells: `{whole, response} x {3e-6, 1e-5, 3e-5}`.

**First cell: `whole_1e-5`.**

It is the only one of the six whose result is interpretable against evidence that already
exists. RAB-001 trained this exact inventory — Supa blocks 8-27, 314,618,880 parameters
across 220 tensors — with whole-sequence supervision at lr 1e-5 under WSD with β₂ 0.95, and
measured it repeatedly. Every other cell changes two things at once relative to that history.

This is **not** a claim that the two are comparable as scores. They are not, and the branch
says so: MQ-001 uses a different runner, token-group accumulation instead of one row per
update, a different corpus, and a 64-row DEV of its own. **The historical 73/74 panel numbers
are not inherited and will not be quoted for any MQ cell.** The argument is narrower: if
`whole_1e-5` behaves pathologically — loss not falling, gradients not finite, DEV collapsing —
that is far more likely to be an integration fault than a real finding, because the nearest
neighbour of this configuration is known to train. It is the cell whose failure would be most
informative about the harness.

**Second cell, if the first is healthy: `response_1e-5`.** Same rate, the objective flipped.
That pair is the experiment's actual question, and the two together are worth more than any
one cell at any rate.

## What the first cell does not decide

* **No objective claim from one pair, let alone one cell.** DEV is 64 rows, of which 41 carry
  a scoreable contract. This programme has measured, repeatedly, that instruments of this size
  do not resolve small differences: the 150-row arithmetic panel runs 15-24 discordant rows
  between same-config seeds, and one such pair reached p=0.0352 on seed alone. A 41-row
  scored DEV is smaller than that. Treat single-cell DEV deltas as diagnostics, not results.
* **No rate recommendation** until at least one objective has been run at more than one rate.
* **No release, no promotion, no comparison to Opallo.** Opallo is a published core measured
  on different instruments; MQ-001 must establish its own baseline.

## Pre-registered health criteria for the first cell

These decide only whether to run a second cell — they are not quality claims.

| check | threshold |
|---|---|
| training loss | finite at every recorded update, and lower at the end than at the first update |
| gradients | finite at every update; no NaN or Inf |
| frozen weights | `parameter_digest` over the frozen tensors unchanged from the source |
| DEV termination | at least 90% of DEV rows terminate within the generation bound |
| DEV exact-match | **reported, not gated** |

If loss or gradients go non-finite, or frozen weights move, **stop and investigate** — those
are harness faults, not results. If DEV accuracy is simply poor, that is a finding about a
25k-token screen and is reported as one.

## Hardware, which is the actual constraint

`resources.guard(before_load=True)` requires 6 GiB available host RAM and 6 GiB free GPU and
refuses with "no automatic retry or override". **That guard will not be lowered**; its
threshold comes from a measured 10x paging slowdown in this same programme.

At the time of writing the host has ~1.5 GiB free of 15.2 at 92% load, against 11.0 GiB of
private commit spread across four Claude sessions, Chrome, Defender and Radeon software. The
GPU is fine at ~7.0 GiB free of 8. **Host RAM is the only thing standing between here and a
result, and freeing it means closing applications — an operator action, not an experiment
one.**
