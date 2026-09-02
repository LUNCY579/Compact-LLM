# Resource report

Measured on NVIDIA GeForce RTX 4070 Laptop, 8188 MiB. One model load at a time.
Supa-A1.1 loaded float16 from the sibling vault. FWA adapters trained in float32.

## Parameters (layer 14, rank 16)

| Arm | Trainable | Recurrent state (B=1) |
|---|---:|---:|
| A | 0 | 0 |
| B | 524,593 | 0 |
| C | 524,593 | 8,192 bytes |
| D | 525,618 | 8,192 bytes |

KV-cache bytes **if used** at T=1024: 28 × 2 × 8 × 1024 × 128 × 2 ≈ 14.7 MB.
This experiment kept `use_cache=False` for the recurrent path, so FWA did **not**
replace the KV cache. Bounded state is 8 KB beside a still-present attention stack.

## Peak VRAM (forward profile)

| Arm | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|
| A | 1.28 GB | 1.36 GB | 1.51 GB |
| B seed 11 | 1.28 GB | 1.36 GB | 1.52 GB |
| C seed 11 | 1.29 GB | 1.37 GB | 1.52 GB |
| D seed 11 | 1.29 GB | 1.37 GB | 1.52 GB |

Training peak (B/C/D, T=512, backward): **~3.02 GB**.
P1 training peak: **2.34 GB**.

No VRAM saving versus Arm A at these lengths. FWA did not shrink memory growth
with length; KV attention is still computed in full.

## Throughput (forward tokens/sec, measured)

Arm A T=256: ~3627 tok/s (no FWA scan).
Arms B/C/D T=256–1024: ~800–1350 tok/s (Python FALCON-2 scan + `use_cache=False`).

Inference is slower with FWA. Do not claim an efficiency improvement.

## Training cost

| Item | Value |
|---|---|
| Completed optimizer steps | 24 × 7 runs (P1 C + P2 B/C/D × 2 seeds) = 168 |
| Training GPU-hours | ≈ 0.64 |
| Seed 33 | not run (OOM while starting the seventh sequential training load) |
| Adapter size | ~1.05 MB (`fwa_adapter.pt`) |

---

# Addendum — corrected measurements (P2b instrumentation)

The table above is retained as the record of what P2 measured, but two of its
numbers should not be used:

1. **The throughput comparison is invalid.** `profile_forward` took no warmup
   pass, so each arm's first measurement absorbed CUDA context creation. That is
   why arm A's recorded tok/s were non-monotonic in length (3627 / 6927 / 5624 at
   256 / 512 / 1024). Re-measured with a warmup pass below.
2. **`kv_cache_bytes_if_used` is identical across arms, and stayed hypothetical.**
   The KV cache was never substituted or reduced by the fast memory, so no
   efficiency claim is testable from those numbers.

Re-measured by `scripts/profile_scaling.py`: warmup excluded, last-position
logits only (full-vocabulary logits for every position is 5 GB at 16K and makes
long-context profiling impossible), peak VRAM reset per measurement.

## Scaling with context length (RTX 4070 Laptop, 8188 MiB)

| Context | Arm A prefill | Arm C prefill | Peak VRAM | Recurrent state | KV cache if used |
|---:|---:|---:|---:|---:|---:|
| 1024 | 5899 tok/s | 1401 tok/s | 1.32 GiB | 8192 B | 112 MiB |
| 2048 | 3630 tok/s | 1222 tok/s | 1.80 GiB | 8192 B | 224 MiB |
| 4096 | 1930 tok/s | 925 tok/s | 3.64 GiB | 8192 B | 448 MiB |
| 8192 | 164 tok/s | 154 tok/s | 10.79 GiB | 8192 B | 896 MiB |
| 16384 | OOM | OOM | — | — | — |

**The 8192 row is not a native measurement.** 10.79 GiB exceeds the card's
8188 MiB, so it is served by host-memory spill; that is why throughput collapses
~6x rather than ~2x. **16K OOMs outright and is not measurable on this
hardware**, so the brief's 8K and 16K tiers are reported as unavailable rather
than silently dropped.

## Answering Q5 honestly

1. **The bounded state is real.** 8192 bytes at rank 16, constant in T, against a
   KV cache that would grow linearly to 896 MiB at 8K. That is the property the
   architecture claims, and it holds.
2. **It buys nothing in this design.** Peak VRAM is identical for arms A and C to
   within 0.01 GiB at every length, because the pathway is *auxiliary*: full
   attention still runs and still dominates the memory. Converting a bounded
   state into an actual memory saving requires Arm E (replacing attention), which
   was not run.
3. **Inference is strictly slower.** Arm C's prefill is 1.5x–4.2x slower than arm
   A at 1K–4K. The gap narrows at 8K only because both arms are spilling.
4. **Generation is worse than the prefill numbers suggest.** The wrapper hard-
   refuses KV-cache mixing, so every decode step recomputes the whole prefix:
   generation is O(new_tokens x T), not O(new_tokens). This is what made
   free-form long-context evaluation unaffordable, and why P2b scores by
   candidate argmax in a single forward pass.

No efficiency improvement may be claimed from this experiment.

## Cost of the P2 work now withdrawn

The 0.64 GPU-hours of P1+P2 training recorded above bought no testable result;
see `reports/MECHANISM_DIAGNOSTIC.md`. It is retained in the cost accounting
because it was really spent.

## Training memory and time (P2b instrumentation, measured)

One optimizer step, arm C, 1024-token contexts (sequence 1152), RTX 4070 Laptop:

| Configuration | Peak VRAM | Wall (1 step) |
|---|---:|---:|
| labels to the model, gradient checkpointing **on** | 6.91 GiB | 15545 s |
| labels to the model, checkpointing off | 4.98 GiB | 20 s |
| `tail_loss`, checkpointing off | **3.09 GiB** | 20 s* |

\* the 3.09 GiB row's wall time is not comparable: it was taken while an
unrelated job had the card at 92 MiB free, and measured 43.8 s/example against
~3.3 s/example uncontended. The **memory** figure is contention-independent
(it is `torch.cuda.max_memory_allocated`).

`tail_loss` saves **1.89 GiB**, which is what brings training inside the
headroom this shared card actually offers. Note the eval phase now dominates:
4096-token scoring peaks at 3.64 GiB.

Two results worth keeping:

**Gradient checkpointing is strictly worse for this configuration** — more memory
*and* ~800x slower. Both configs have declared `gradient_checkpointing: true`
since P1, and no run ever enabled it, so nothing regressed; but had it been
enabled it would have hurt. The cause is specific to a frozen backbone: enabling
it needs `enable_input_require_grads()`, which forces layers 0-13 to build an
autograd graph they otherwise skip entirely (only layer 14 is trainable, so
nothing below it needs one), and the recompute pass re-runs layer 14's
python-loop scan a second time.

**Full-length logits dominated training memory.** Passing `labels` makes the
model compute logits at every position: 1152 x 151936 is ~350 MB in fp16, plus
an fp32 copy inside the cross-entropy and a gradient of the same size — roughly
1.4 GB — to supervise a single answer token. `tail_loss` requests only the
trailing positions covering the supervised span and is numerically identical to
the model's own loss (`tests/test_tail_loss.py`).

This mattered operationally: the card is shared with an unrelated sweep, and a
launch at 4808 MiB free against a 4.98 GiB peak would have OOMed.
