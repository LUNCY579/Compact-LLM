# PPT-001 CPU sampler qualification amendment

**Date:** 2026-10-03 (America/Chicago).  
**Scope:** supplied tiny enumerable probabilities and sampler mechanics only.  
**Result:** 28 checks passed: 12 registration and 16 independent sampler tests.  
**Native scoring / native benchmark / reasoning benefit:** unqualified / unrun / unknown.

The owner requested this bounded CPU implementation as the next useful step.
The founding registration, root charter, and other research tracks are retained.
The proposed under-one-CPU-minute and 256-MiB target is measured here for this
fixture; it is not a native admission profile or permission to start replicas.

## Executable fixture and independent oracle

[tiny_sampler.py](../tiny_sampler.py) defines a two-slot horizon, ordinary tokens
`A` and `B`, EOS, and deterministic PAD after EOS. Seven canonical records yield
49 ordered two-replica states. The supplied base conditionals are:

| Prefix | A | B | EOS |
| --- | --- | --- | --- |
| empty | 1/2 | 1/3 | 1/6 |
| A | 1/5 | 1/2 | 3/10 |
| B | 1/2 | 1/8 | 3/8 |

Each EOS is included in base sequence probability. PAD contributes probability
one. Horizon-ending records without EOS are valid and do not receive an invented
EOS score. Targets are normalized whole-sequence probabilities raised to a
positive integer power; tokenwise powered proposal probabilities are separately
normalized at each prefix. This prefix-sensitive table deliberately makes those
two distributions differ at powers greater than one.

For each local proposal, a restart slot is chosen uniformly from slots 0 and 1,
regardless of realized length. A suffix is drawn at the replica's power and
accepted with the whole-sequence target ratio times the reverse/forward suffix
proposal ratio. A restart after EOS produces an exact self-transition. Restarting
at or before EOS permits an earlier termination to be replaced by a longer
record. Rejections retain the prior record and appear in the event record.

Swap acceptance is `min(1, (p(right)/p(left))**(left_power-right_power))`.
Accepted swaps retain powers at their original replica positions and recompute
each received record's proposal score at that receiving power. Stale replica
scores are rejected before exchange. Local and swap event records retain both
attempted and final states and acceptance probabilities.

All probabilities and draws use exact fractions and integer tickets. No
floating-point tolerance or statistical frequency test stands in for exact
kernel comparison. [The independent tests](../tests/test_tiny_sampler.py) use
their own state enumeration, probabilities, proposals, acceptance law, and
transition construction. They compare every local and composed joint transition,
normalization, detailed balance for individual reversible kernels, and target
stationarity for the composed schedule. A composition need not be reversible.

Uniform full restart gives positive support between every pair of canonical
states; positive self-proposal mass prevents periodicity. The local target and
product joint target are thus unique stationary distributions in this finite
fixture. An initialized finite chain is tested separately and need not already
equal equilibrium. This establishes no useful finite-budget mixing rate for
native reasoning.

`LocalEvent.token_draws` is a synthetic count of non-PAD slots in the attempted
suffix, including EOS. The apparatus actually draws one enumerated suffix.
That field is neither a count of RNG calls nor a native forward-work measurement.
Native caches, forward-token costs, transfer costs, and launch budgets remain
unimplemented. There is no general centralized runtime or dataset generator.

## Deliberately broken control

`broken_truncation_kernel` selects restart slots only through the old record's
realized EOS-inclusive length, then incorrectly omits the state-dependent
restart-law ratio from acceptance. Tests must reject it with the same stationarity
criterion used for the correct sampler. It still has normalized transition rows:
normalization alone is insufficient. At power 2 its exact maximum stationarity
residual is **1009/33852**, while the correct kernel has residual zero.

This preserved negative control is a defective implementation, not evidence
against interacting search. It is never used by `TinySampler` or `joint_kernel`.

## Measurement and reproduction

The machine-readable [CPU receipt](cpu_sampler_2026-10-03.json) binds source
hashes, Python/platform identity, test output, measured CPU/wall seconds and
process memory peaks. This run used **0.671875 CPU seconds** and **0.6691212 wall
seconds** for test discovery/execution. Lifetime peak working set/private commit
were below the proposed cap; their maximum was **23.24609375 MiB**. The suite is
in one Python process with no model, GPU,
framework, network, or test child-process work. On Windows, lifetime peak working
set and peak private commit are read with `GetProcessMemoryInfo`; the larger
value is checked against the proposed 256-MiB target. CPU time covers discovery
and execution of this package's tests. These measurements do not predict model
memory, cache fit, latency, or matched-work accuracy.

From the repository root:

```sh
python -B -m unittest discover -s experiments/rineya_ppt_centralized_reasoning_v1/tests -p 'test_*.py' -v
python -B experiments/rineya_ppt_centralized_reasoning_v1/qualify_cpu.py --receipt evidence/cpu_sampler_repeat.json
```

The receipt writer preserves existing receipts and fails if tests, skipped-test,
source-stability, or measured target checks fail. A new filename is required for
each run. The original 12 registration checks remain administrative checks; the
additional sampler tests supply the mechanics evidence.

Remote freshness was attempted before local implementation: `git ls-remote`
for the registered branch returned `Repository not found`. No remote content
was overwritten or published. The receipt records the local base tip plus actual
source hashes; it does not certify the current remote branch.

## Next gate: exact Rineya scoring, then measured fit

No checkpoint is selected by this amendment. Preserve the established inherited
prefix/full failure in
[AMENDMENT_015.md](../../post_training_mimo_centralized_v1/AMENDMENT_015.md) and
[AMENDMENT_016.md](../../post_training_mimo_centralized_v1/AMENDMENT_016.md).
For the recorded Titan BF16 no-cache trace (prompt 128, IDs `[16,22,151645]`),
maximum and mean absolute log-probability errors were **0.0414324105** and
**0.0262993514**, exceeding the unchanged **0.03125** and **0.01** gates. D1
localized the difference upstream of the output head; FP32 head projection alone
did not repair it. This is historical evidence from another track, not present
PPT source qualification. Its D2 diagnostic stopped before model construction
or forwards for insufficient available RAM.

A separately scoped scorer qualification must bind the exact intended Rineya
checkpoint, tokenizer/template, dtype, code, masks, attention, EOS policy,
numerical tolerances, current host admission and launch authorization. Use short
label-free traces, including the retained failing trace, before considering any
512-token multi-replica configuration. Compare per-token base log-probabilities
and their sequence sums across prefix replay, full scoring, and cached incremental
scoring. Argmax agreement is insufficient. Retain failed traces and do not relax
the earlier gates or mix inconsistent scoring paths to manufacture a pass.

Check immediate EOS, horizon EOS, no EOS, post-EOS zero score, suffix replacement,
swapped records, receiver-power proposal normalization, cache ownership, rollback,
and fresh replay after exchange. Measure all attempted generation, rejected
proposals, rescoring, prefix replay and cache transfer work. Old resource receipts
are not current fit evidence. Native scoring, host admission, and benchmark
authorization remain separate gates; CHECK/FINAL, training, and paid-service gates
remain closed.

The research decision is unchanged: exchanges must beat identical uncoupled
replicas (A5 versus A4), and the method must also beat a strong independent-sampling
baseline (A2) at matched total work. Candidate coverage and returned-answer accuracy
remain separate outcomes. A win over greedy decoding alone cannot promote the
method. The CPU receipt supplies no efficacy or model-quality result.
