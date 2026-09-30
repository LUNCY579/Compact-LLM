# Amendment 016: fixed-shape future-token causality diagnostic

30 September 2026. The measured geometry diagnostic reproduced the failed
prefix/full likelihood gap exactly and localized it to differing hidden states
before the output head. Changing BF16 head row count produced zero difference;
a GPU-only FP32 head still failed both 0.03125/0.01 bounds. This did not identify
the upstream mechanism. Original capture and geometry registrations, code and
raw records remain unchanged.

Register a second diagnostic on the same frozen 131-token full sequence. Make
four native forwards: the original sequence, then fixed mutations at positions
128-130, 129-130 and 130. Compare protected final-normalized hidden states and
full-vocabulary logits at prediction positions 127, 127-128 and 127-129
respectively. Require bitwise identity within each equal-length comparison;
record selected target logprobs as an additional check. The baseline must match
the prior bulk logits, logprobs and held-hidden fingerprint.

Record actual first-decoder additive masks, rotary position IDs and final hidden
hook invocations. Verify the BF16 1x1x131x131 lower-triangular mask: permitted
entries are zero, prohibited entries are the finite dtype minimum. Positions
must be exactly 0 through 130. Hooks only observe tensors and return no replacement
values. Retain only the final four hidden rows; do not keep full layer snapshots
or create CPU model copies. Record exposed BF16/FP16 reduction/split-K flags
without changing them. Unsupported flag names must be marked unavailable.

Four forwards imply 524 input positions and 316 logical model/load/copy calls,
with zero additional direct head projections, generation, updates or teacher
calls. Retain the 2 GiB process-tree RSS cap, 1.25 GiB system reserve, 4/0.5 GiB
GPU and 4 GiB disk floors, 600-second ceiling, 100 ms monitoring and Windows
containment. Commit the registration and reviewed source before execution;
preserve all actual/failed accounting and verify final base/source hashes.

A successful diagnostic only establishes fixed-shape causal invariance for these
three protected suffix comparisons. It does not establish prefix/full shape
invariance or identify a numerical layer/kernel cause. It grants no capture,
DEV collection, language grading, E1 or closed-bank access. If these controls
pass, a later explicitly registered exact-prefix teacher-forced scoring scope
may be evaluated with the unchanged likelihood tolerances. Such a successor
must retain the failed bulk test as unqualified and cannot authorize stochastic
RL or declare bulk training likelihood equivalent to generation likelihood.
