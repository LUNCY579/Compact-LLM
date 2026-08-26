# Published Qwen3-0.6B family: first answer-level evidence

Runs: [`halo_family_showcase64_v1_part1`](results/runs/halo_family_showcase64_v1_part1)
(Halo Uncensored, Lyra) and
[`halo_family_showcase64_v1_part2`](results/runs/halo_family_showcase64_v1_part2)
(Nimbus, Pulse).
Classification: `BOUNDED_EXPLORATORY`, unregistered. Addresses ledger row `N-009`.

Before this, only Halo had ever decoded an answer under this runtime. The other
four published cores had a single prompt-only 192-token compatibility smoke each,
which recorded `answer_generation_performed: False`. `N-009` stood at `NOT_RUN`
for the whole program.

## Results

Complete 64-case showcase split, B0 full cache and B3 observation-normalized
compression, identical protocol to Halo. Every core's weight hash and
`CORE.json` identity was verified before load.

| Core | B0 full cache | B3 compressed | lost to compression | B0-conditioned retention |
|---|---:|---:|---:|---:|
| Halo Uncensored 1.5.0 | 31/64 | 21/64 | 12 | 0.61 |
| Nimbus 1.6.0 | 30/64 | 19/64 | 13 | 0.57 |
| **Halo 1.2.0** (reference) | **30/64** | **19/64** | **11** | **0.63** |
| Pulse 1.6.0 | 29/64 | 20/64 | 10 | 0.66 |
| Lyra 1.4.0 | 28/64 | 18/64 | 11 | 0.61 |

## Finding — the compression gap is a family property, not a Halo quirk

The spread is narrow on every axis: full-cache capability 28-31, compressed
19-21, answers lost 10-13, retention 0.57-0.66. Halo sits in the middle of its
own family on all four.

Two consequences.

**The program's Halo findings are not checkpoint-specific.** Every negative
result about the significance mechanism — that it is indistinguishable from a
position counter, that shuffling its scores does not hurt, that `C` is age —
was measured on a core whose compression behaviour is now shown to be typical
of the family. That materially strengthens those conclusions.

**Any fix that works should be expected to transfer.** The pooling-plus-sink
configuration currently under held-out validation was tuned on Halo. Since all
five cores lose 10-13 answers with retention clustered near 0.6, the same
selection deficit exists in each, and a selection-side repair has the same
target in each. That is a prediction, not a result: no core other than Halo has
been run under pooling.

## What this does and does not license

`N-009` asked whether a result on Halo generalizes to the other cores. The
honest answer is now split:

- The **compression-deficit phenomenon** replicates on 4/4 additional cores.
- **No pooled family efficacy claim** is permitted. Each core keeps a separate
  identity, and these are five separate 64-case exploratory measurements, not a
  family-level inferential result.

## Limits

- Exploratory and unregistered; no frozen endpoint, no advance gate.
- Scenario packs are correlated; 64 rows are not 64 independent observations,
  and no cross-core inferential test was run.
- Halo's numbers come from a separate earlier run under the same protocol.
- Part 1 and part 2 executed in different processes. Each core is evaluated
  entirely within one process, so every within-core B0/B3 contrast is
  process-local; `N-034` cross-process drift does not apply to the contrasts
  reported here.
- Part 1 survived an interrupted run. Its two complete cores were preserved
  answer-blind — the answer key was never opened by the interrupted process —
  and scored only in this analysis.
- CPU bf16 local accounting; no deployment or GPU claim.
