# Amendment 009: measured laptop resource qualification

29 September 2026, America/Chicago. The user explicitly requested a plan suited
to a 16 GB laptop, authorized changing the RAM requirement and experimental plan,
and preferred VRAM-heavy work with modest host-memory use. This authorization
supersedes the prohibition on revising host floors for the new named profile below.
The original research objective and exact Titan student are retained.

## Why the earlier guard is revised

The 7.4 GiB admission and 6 GiB ongoing values were inherited from PRLC project
defaults. That workflow keeps a GPU model and a frozen CPU reference; its README
calls those values defaults rather than portable measurements. The centralized
loader constructs BF16 Titan directly on CUDA, streams the checkpoint and has no
full CPU reference model. All earlier E0 attempts stopped at admission. They did
not measure a need for 7.4 GiB of available RAM or establish that Titan cannot fit.

The exact BF16 payload is 1,192,099,840 bytes, approximately 1.110 GiB, primarily
resident in VRAM after loading. Hashing already uses roughly 2 MiB BF16 chunks.
The 32 saved last-token parity logits occupy approximately 18.55 MiB of host RAM.
The q/v adapters have 1,146,880 FP32 parameters: 4.375 MiB; their gradients and two
Adam moments together add 13.125 MiB, primarily in VRAM. Native driver, tokenizer,
mapped checkpoint and activation peaks still require measurement.

## Prospectively registered laptop profile

Profile: LAPTOP_GPU_RESIDENT_V1. Preserve E0_TITAN_BF16.json byte-for-byte as the
historical profile. Two new frozen configs select load_forward or full qualification.
Retain the exact model, BF16 base, FP32 rank-8 q/v adapters, eager attention, RNG
operations, fixtures, loss, optimizer, tolerances and deterministic-resume checks.
No quantization, CPU reference, CPU optimizer offload or attention replacement is
introduced by this amendment.

Keep at least 2 GiB available system RAM throughout and limit the worker process
tree to 2.5 GiB RSS. At post-import admission, reserve the entire remaining worker
RSS allowance: required available RAM = 2 + max(0, 2.5 - current worker RSS) GiB.
Also reject an already over-limit worker. For the previous 0.669 GiB import RSS,
this requires approximately 3.831 GiB available after imports, rather than 7.4.
The 2.5 GiB worker cap is an initial measured-pilot ceiling, not proof of fit.

Retain GPU admission >=4 GiB, GPU ongoing free >=0.5 GiB and disk >=4 GiB. Preserve
the independent 100 ms process-tree watchdog, Windows kill-on-close containment,
GPU lock, clean committed source requirement, durable intent journal and write-once
receipts. Record the new limits in launch evidence. RSS is working-set memory,
not committed memory; the sampled global reserve also covers pressure outside the
worker. This is not an OS hard memory allocation cap or a promise of zero transient
overshoot. Stop and preserve failures if any measured limit is violated.

## Staged experiment

First run a <=600 s exact-load resource pilot, with one 256-token frozen no-cache
forward retaining only the last vocabulary logits. Zero adapters, optimizer calls
and generated tokens; config caps for optimizer, generation and parity loops are
explicitly zero, with only the 256-token probe context. Verify exact checkpoint/assets, every copied tensor, tying,
tokenizer and finite logits. A distinct resource-only PASS cannot qualify E1.

If that pilot fits, run the <=1800 s full original E0 workload under the same laptop
resource profile: 32 base/zero-adapter comparisons, masks/causality/likelihood,
256/512/1024 backward profiles and five bounded reversible optimizer calls including
bitwise checkpoint resume. Workload, numerical checks and budgets are unchanged.
The historical E1 checker must reject the new config until a separately reviewed
validator and current-runtime E1 plan are bound; no resource pass grants training.

Next prioritize one resident GPU worker, microbatch one with accumulation, streamed
data, adapter-only checkpoints and sequential evaluation panels. Register measured
laptop successors for queue/E1 and longer-context plans before their runs. If VRAM
fails, separately register activation checkpointing or a shorter operating context
and requalify numerical behavior. Do not stack such changes into this first pilot.

## Interpretation and preserved evidence

This changes the resource admission policy, not a learned-method comparison.
Earlier admission failures remain valid under their historical policy; they are
not actual model memory-fit failures. All prior files and receipts are preserved.
This amendment does not promote a recipe, waive preservation, open CHECK/FINAL or
fabricate grading. Remaining scientific prerequisites persist after memory fit.
