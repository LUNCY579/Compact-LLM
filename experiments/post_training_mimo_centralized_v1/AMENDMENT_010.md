# Amendment 010: separate backward memory fit from unresolved padded likelihood

29 September 2026, America/Chicago. The authorized laptop resource pilot loaded
exact Titan and completed a frozen 256-token forward. The full laptop E0 attempt
passed exact loading and 32 bitwise zero-adapter comparisons, then failed the first
left-padding likelihood check before any backward or optimizer work. The host-RAM
guards passed. Preserve both original runs without relabeling the full failure.

The failing check did not retain its observed discrepancy. Add failure-only
telemetry for maximum/mean error, aligned reference/observed logprobs and changed
argmax positions. Preserve the registered 0.03125 maximum and 0.01 mean thresholds,
the input/mask/position operations and full mathematical workload. A new full run
may recover the discrepancy; it is not an automatic pass or a relaxed tolerance.

To answer the user's adapter-training memory question independently, register a
backward_resume resource stage under LAPTOP_GPU_RESIDENT_V1. It retains the exact
loader, constructor seed, 32 base forwards, adapter installation, 32 zero-adapter
forwards, positional EOS/PAD label mechanics, all three 256/512/1024 gradient/update
profiles, the five-call bound, save/reload, bitwise within-stage continuation and
frozen-base verification. Omit native padded forward/causality and sampled-logprob
checks in this explicitly limited stage. Generation budget is zero. The omitted
sampler also omits its RNG draws; the saved RNG state is local to this registered
stage and is not claimed identical to a full E0 trajectory. Training dropout is
zero and the retained adapter initialization occurs before the omitted sampler.

Retain the 1800-second ceiling, 2 GiB system reserve, 2.5 GiB process-tree RSS cap,
dynamic admission, GPU/disk floors, external watchdog and source/output binding.
All five updates are disposable non-evaluation mechanics fixtures. No acquisition,
retention, held-out capability or learning efficacy is measured.

Success has the distinct status PASS_E0_BACKWARD_RESUME_RESOURCE_ONLY. Native
padding/causality, sampled likelihood and full E0 remain explicitly unqualified.
The historical E1 checker rejects this status, and e1_authorized remains false.
This stage measures the feasibility and mechanics of GPU-resident adapter work;
it cannot waive the unresolved numerical issue, preservation or E1 prerequisites.
