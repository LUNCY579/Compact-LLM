# TITAN-OPALLO-SKILL-VECTOR-001 — result

Date: 2026-09-22

## What was actually run

The exact stored Supa bytes were used as the common parent and upcast once to
FP32. Titan-Sparse was treated as an atomic FP32 sparse endpoint. Opallo was
reconstructed from its exact FP32 `B28last20` trained-tensor checkpoint on the
same parent. The reconstructed Opallo endpoint, cast once to BF16, matched the
published Opallo checkpoint at all `596,049,920` serialized elements.

The nine-arm matrix was materialized into the new `arms/` directory:

```text
BASE, SPARSE, OPALLO, AVG,
ADD-100, ADD-S75, ADD-O75, ADD-75, CONFLICT-LOCAL
```

All nine outputs are FP32, reload successfully, have the expected 310-tensor /
596,049,920-parameter inventory, and contain no non-finite values. The exact
BASE, SPARSE, and OPALLO endpoint checks passed. See `VERIFICATION.json`.

## Geometry result

| quantity | result |
|---|---:|
| `||dSparse||₂` | `0.6940550847` |
| `||dOpallo||₂` | `1.3569596128` |
| global cosine | `0.0025466703` |
| absolute per-tensor interference | `0.0037945863` |
| sparse nonzero entries | `78,654,720` |
| Opallo nonzero entries | `314,616,010` |
| overlapping nonzero entries | `78,654,011` |

The global and absolute-interference values both support the geometric part of
the near-orthogonality theory at this stored-parent precision. Only two tiny
norm tensors crossed the preregistered local-conflict rule (`cosine <= -0.20`):

```text
model.layers.27.self_attn.k_norm.weight
model.layers.27.self_attn.q_norm.weight
```

`CONFLICT-LOCAL` attenuates Opallo to `0.5` only on those two tensors. It is a
diagnostic rescue arm, not a selected model.

## Exploratory functional probe

To get a first functional read without reusing any old bank, a new 24-row suite
was authored in `data/fresh_probe.jsonl`: eight S rows, eight O rows, and eight
joint S×O rows. Each arm was loaded independently on the RTX 4070 Laptop GPU in
FP32, greedy, batch-one mode. The run produced 216 forwards and retained every
raw answer in `FUNCTIONAL_PROBE.json`.

| arm | S / 8 | O / 8 | S×O / 8 |
|---|---:|---:|---:|
| BASE | 1 | 2 | 0 |
| SPARSE | 0 | 2 | 1 |
| OPALLO | 3 | 3 | 2 |
| AVG | 2 | 3 | 2 |
| **ADD-100** | **2** | **3** | **2** |
| ADD-S75 | 3 | 3 | 2 |
| ADD-O75 | 2 | 3 | 2 |
| ADD-75 | 3 | 3 | 1 |
| CONFLICT-LOCAL | 2 | 3 | 2 |

On this small exploratory bank, ADD-100 preserves the Opallo endpoint's O and
joint counts, improves over BASE on S, but remains below OPALLO on S and shows no
joint-task gain beyond OPALLO. SPARSE itself scores `0/8` on this hand-authored
S-bank, so the donor-normalized S preservation ratio is not interpretable here:
its denominator is negative. This probe therefore does not confirm full-strength
skill composition or synergy. The two conflict-localized norm tensors do not
change the aggregate counts.

The suite is deliberately not presented as a formal capability instrument: it is
small, hand-authored after the models existed, and not proven disjoint from every
historical training example. It is a first functional diagnostic, not a release
or confirmation gate.

## Parent precision boundary

The machine does not expose an unrounded FP32 Supa checkpoint. Accordingly, the
common parent here is the exact stored Supa BF16 artifact lifted to FP32. The
Opallo and Titan source tensors were reconstructed against that same parent, and
the published Opallo cast round-trip is exact. This makes the composition
internally consistent, but it is not evidence about an unavailable hidden
unrounded-Supa coordinate system.

## What this does not establish

The functional pass is exploratory only. It does not establish formal functional
independence, retention, quality improvement, or release fitness. ADD-100 is the
primary hypothesis artifact, but it remains an exploratory candidate only.

Existing banks and old experiment authorization were not reused or modified.
