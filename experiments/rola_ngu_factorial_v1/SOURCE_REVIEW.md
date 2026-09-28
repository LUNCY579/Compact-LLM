# Source review — 2026-09-18

## Frozen code ancestry and current evidence

At the first read in this session the dedicated branch pointed at
`fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`, without a package commit. The
pre-write recheck discovered a concurrent publication at
`3a83618a96a82f87255e982e888d0ecb78dd94b0` (2026-09-18T20:15:09Z).
Its package tree `0a1d4b7fb4154329f69f61d05cd64839c098eb89` contains the
original delivered 97-test mechanics bundle, with exact parent `fbec90a0...`.
The integration therefore descends from `3a83618...`, preserves all original
historical evidence bytes, and adds separately dated evidence. No force reset,
history rewrite or unrelated research merge is part of this change.

| Branch | Observed remote head | Relevance |
|---|---|---|
| `research/rola-ngu-factorial-v1`, pre-write recheck | `3a83618a96a82f87255e982e888d0ecb78dd94b0` | Original mechanics published concurrently; immediate integration parent |
| `codex/dataset-retention-audit-v1` | `07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd` | Updated family evidence; confirmation is no longer wholly absent |
| `single-layer-fine-tuning` | `954a0c2637e691b493c5ac3d488653a8c1adf675` | Titan-Sparse REP-001 evidence donor |
| `research/delta-multi-reasoning-factorial-v1` | `b56dd4b1101ba70e5109b712c28ac52c1fb718d8` | Same negative confirmation; exploratory mechanism accounts conflict with later Codex correction |
| `research/sequential-complementary-skills-v1` | `3ad2769a9f8343a9697e1720fcdc9efa18c3845a` | Separate sequential programme; not merged or silently pooled |
| `main` | `a6ee824584a9e28881826ab92a11db48e124d90c` | Historical baseline, not the selected research parent |

Source lock copied from `experiments/model_quality_gpu_v1/source_lock.json` at
`fbec90a0...`; model source is exact ungated Supa 1.7, never Supa-A1.1 or a biased
UNKNOWN head. Neither code ancestry nor an inherited grant authorizes this study.

## What Titan establishes and what it does not

At `954a0c2`, the Titan README records 384 one-step training rows, 54,172 processed
tokens, 28 optimizer updates, tail blocks 8–27, learning rate 1e-5, and no rehearsal.
Its acquisition was 234/384 vs Supa's 14/384. This is learning, not free construction.
Titan-Sparse reuses the trained FP32 delta, keeping top-25% magnitude per tensor.
That derivative needs no new training or model calls to construct, but still
requires sorting, memory/I/O and evaluation. Dense serving is not accelerated.

The seed17 sparse core preserves 41/47 source-correct arithmetic answers, losing
6 and gaining 31. Its old long-context net +3 hides 7 losses and 10 gains. These
are not no-forgetting results. The full trained update is about 0.056% of weight norm;
small parameter displacement does not imply small behavioral displacement.

REP-001 paired observations:

| Draw | Acquisition full / k25 | Retained full / k25 | Long context full / k25 |
|---|---|---|---|
| 17 | 235 / 232 | 40 / 41 | 157 / 164 |
| 23 | 229 / 228 | 38 / 39 | 152 / 152 |
| 29 | 228 / 230 | 36 / 40 | 144 / 159 |

Both preservation endpoints were non-worse in these three draws. This is not
strict improvement on both in all three; seed23 long-context change was zero.
Arithmetic also has regressions. The worst draw had the largest repair, but
three observations do not prove a monotone damage-repair law or a no-harm rate.
Historical long context remains SPENT/report-only.

## New, load-bearing negative confirmation

`Rineya_Published_Cores/FAMILY_PERFORMANCE_AND_RESEARCH.md` at `07e53b0...`,
section 3, records RCI-001: a 512-row, single-authorized-read instrument, now spent.
On 128 unanswerable rows, frozen Supa had 40 literal UNKNOWN successes and
Titan-Sparse had 25: **0 gains, 15 losses; exact paired McNemar p=6.1e-05**.
Both had zero false abstentions on 384 answerable rows. This is a registered
negative confirmation, not a promotion. Do not repeat the old blanket claim that
no confirmation instrument has ever existed.

Two material limits stay attached: both runs used matched FP32 HF generate on
Transformers 5.14.1, but that serving path was unqualified against the older
4.51.3 DenseReference. The aggregate floor was corrected after the read from
0.2773 to 0.3711; the record says the sentinel stratum is immune to this defect.
Do not pool the matched 5.14.1 contrast with the older historical score tables.

The latest Codex record withdraws the role/occupant-conflation explanation still
present in the sibling DMRF head. Answerable subgroup counts also depend on
scoring conventions. Carry forward the agreed negative sentinel result, not a
mechanism or disputed answerable decomposition. No sealed rows/golds were imported.

## Implications for RNGF-001

Keep Supa as the common primary initialization and freeze an independent source
baseline. Keep pruning out of the primary 2x2; generate all paired k25 derivatives
after checkpoint freezing. Keep FP32; casting is a separate audited intervention.
Measure acquisition AND lost source-correct rows AND literal/false abstention.
Loss rates must condition on the source-correct subgroup denominator. Do not
silently use an improvement threshold (e.g. inherited panel>=62 for source=47)
as a retention criterion. Require fresh prospective instruments and meaningful
effect/noninferiority margins; the spent RCI-001 cannot be reopened here.

Papers inspected directly: RoLA v2, arXiv:2609.06712; NGU v1, arXiv:2609.13443.
The causal port and logical-async fixture are explicit deviations from full video
RoLA and production asynchronous RL respectively; no paper speedup transfers.
