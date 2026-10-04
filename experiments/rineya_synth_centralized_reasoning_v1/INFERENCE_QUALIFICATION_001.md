# Four-TRAIN-prompt inference qualification

This scope checks the numerical mechanics needed by a later prompt-only
collector. It cannot establish acquisition, transfer, retention or a score.
The owner delegated pilot decisions; this qualification implements that goal
without opening the study, DEV scoring or protected-bank gates.

Before any model output, select the longest complete R and S TRAIN prompt in
each registered M1/M2 source pool, with lexical row-ID tie breaking. Strip
continuations and labels. The four selected lengths are 419/186/419/186 tokens.
Use a fresh pinned unchanged dense Titan base, the already qualified BF16
resident / FP32 activation path, and fresh zero-B rank-8 q/v factors. No optimizer,
backward, STATE1 restore or adapter checkpoint is permitted.

For each row, capture greedy no-cache likelihoods twice and compare against the
bulk path under the existing numerical tolerances. Preserve every token result,
attempted/returned primitive, partial failure and terminal receipt. Verify all
310 frozen source tensors afterward. The hard scope allows at most 262 forwards
and 134,144 input positions, 512 context tokens and 32 generated tokens.

Reuse the exact Native7 process ownership, Job containment, direct CPU/RAM/NVML
observations, freshness, memory, disk and 600-second deadline bounds. A clean
committed source closure and a new write-once configuration precede launch.
Admission failure is terminal for that attempt. Success requires a normal
supervised exit, complete sampler/tree cleanup, numerical controls, unchanged
base bytes and an independent audit of raw evidence.

Even a pass qualifies only this bounded TRAIN inference mechanics scope.
Full DEV collection, study sustained fit and cross-process continuation,
long-context execution, qualified language review, prior-correct enrollment
and the M0/M1/M2 comparison remain separate requirements.
