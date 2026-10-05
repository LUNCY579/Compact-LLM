# Staged long-context inference component

The new module differs from the preserved original in three lines: its header,
version, and constructor's explicit long envelopes 1024/2048/4096. Root short
inference sources remain byte-identical. Greedy batch-one full FP32 finite logits,
no-cache/no-grad execution, masks, EOS handling, generated-only decoding,
parameter/gradient preservation, attempted/returned accounting, halt semantics,
two exact repeats, bulk selected-logprob comparison and numerical limits remain
the same algorithm. Neither module grants native launch or qualification.

The sparse full-length CPU controls pass all 28 cases: six new controls and 22
existing semantic cases explicitly rebound to this module. Each 992/2016/4064
prompt exercises all 32 generated steps twice and a complete bulk comparison.
Maximum/mean fake logprob error is zero. The sparse tensors represent the full
vocabulary algebraically; they do not allocate actual model logits or prove
native memory, numerical performance or throughput.

An unmodified 134144-position ceiling stops the 4064-token case after its first
32-step trace (130544 positions), before a second model call can overspend. A
future pipeline must keep both repeats and the bulk check. Separate fresh owned
trace-one, trace-two and bulk scopes can retain the existing position ceiling:
including the prospective 390 positions of two short TRAIN parity forwards, a
4096-envelope trace scope needs 34 forwards/130934 positions and the bulk scope
three forwards/4485 positions. Their combined maximum is 71 forwards/266353
positions, including the four extra parity calls imposed by fresh scopes.
These are arithmetic upper bounds, with TRAIN rows and actual scopes unbound.

Independent review remains required. TRAIN-only long prompt construction must
bind exact text/template/tokens without truncation or guard/oracle substitution.
The split pipeline must authenticate unchanged model/zero-B/source/RNG identities,
immutable complete trace records and every selected-token/LP comparison across
fresh workers. It must retain actual cumulative costs, original resource and
numerical limits, owned process cleanup and fresh admission per scope. No native
contract, registration, worker, cross-process verifier, guard enrollment or score
is implemented by this delivery. Qualification cannot be inferred from its CPU
controls, prospective budget or the separate short INFQ001 result.
