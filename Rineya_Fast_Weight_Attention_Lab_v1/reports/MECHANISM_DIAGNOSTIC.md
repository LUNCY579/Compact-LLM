# Mechanism diagnostic — why the P2 NO-GO tested nothing

Produced by `scripts/diagnose_mechanism.py`, `scripts/profile_scaling.py` and
`scripts/calibrate_suite.py` against the committed P2 adapters. Raw numbers in
`artifacts/supa_a1_1_fast_weight_attention_v1/diagnostics/`.

The P2 report concluded **NO-GO** because arms B, C and D emitted identical
predictions on every scored item. That inference is only valid if the
fast-weight pathway was contributing something the evaluation could see. This
document measures both halves of that conditional. Neither holds.

## 1. The pathway was live, and the recurrence did dominate the control

Measured at layer 14 on real P2 eval items (990 prompt tokens), fp16 forward,
`||alpha * P(m)||` against `||attn_out||` per token:

| Arm | trained alpha | mean contribution / attention output | mean ||P(m)|| | ||attn_out|| |
|---|---|---|---|---|
| B (static)      | 0.051625 | **0.12 %** | 0.295 | 13.37 |
| C (recurrent)   | 0.051610 | **1.41 %** | 3.50  | 13.37 |
| D (sig-gated)   | 0.051618 | **3.22 %** | 8.05  | 13.37 |

The recurrent state carry makes arm C's contribution **11x** arm B's, and the
significance gate makes arm D's **2.3x** arm C's. The arms were mechanically
distinct, in the predicted order.

Final-token logit differences on identical inputs confirm it:

| Pair | max abs logit delta | mean abs delta | mean KL (nats) | argmax agreement |
|---|---|---|---|---|
| A vs B | 0.633 | 0.091 | 0.016 | 2/2 |
| A vs C | 1.438 | 0.221 | 0.071 | 2/2 |
| A vs D | 3.367 | 0.569 | 0.251 | 2/2 |
| B vs C | 0.805 | 0.132 | 0.017 | 2/2 |
| C vs D | 1.930 | 0.353 | 0.038 | 2/2 |

A 3.4-logit shift and a 0.25-nat KL are far above fp16 noise. **The arms
differed; the evaluation could not see it, because every arm was already
correct on every item.** "C = B" was a property of the ruler, not the pathway.

## 2. The memory horizon was ~2 orders of magnitude too short

The trained plasticity/forgetting parameters barely moved from their
initialisation, so the retention horizon was effectively a hyperparameter
choice, not a learned one:

- `beta = 2*sigmoid(beta_raw) = 1.0006` (init 1.0)
- `lambda = softplus(lambda_raw) = 0.1268` (init 0.1269 — unmoved)
- `k_norm` is an RMSNorm over rank 16, so `||x||^2 = 15.95`
- `eta = beta / (||x||^2 + lambda + eps) = 0.0622`
- ridge decay `1 - lambda*eta = 0.99211` per token → **half-life 87.5 tokens**

Worse, `beta = 1.0` is exactly the NLMS deadbeat point. Retention along the
*current key direction* is `decay - eta*||x||^2 = 0.992107 - 0.992129 = -2.2e-5`,
i.e. a write **completely replaces** whatever was stored in that direction. The
memory is a local smoother with an ~87-token ridge horizon and total overwrite
on collision.

The measured state norm confirms saturation rather than accumulation: on a
990-token item `||S_t||_F` rises to ~12.5 by t≈390 and then *declines* to 10.1
by t=990. A fact at relative position 0.05 is ~940 tokens from the query;
`0.99211^940 = 6e-4` of it survives.

**No configuration in P1 or P2 could have carried a fact from the needle to the
query.** Q3 (positional robustness) was never testable.

## 3. The evaluation was at ceiling and had no statistical power

- Filler was **one sentence repeated** 21–42 times, so the needle was the only
  non-duplicated sentence — "spot the odd sentence out", not retrieval.
- Answer space was **10 colour words**.
- `target_length` counted filler *sentences*, not tokens: a "512" item is really
  990 prompt tokens. No evaluated length was what it claimed.
- `run_experiment.py:151` hardcodes `lengths = [256, 512]`, overriding the
  config's `[512, 1024, 2048, 4096]`. The long-context regime was never run.
- n = 14 position items — **one per (length, position) cell**. A paired exact
  test needs 6 of 14 items to flip (42.9 pp) before it can reach p < 0.05. The
  interference battery (7 items, one per category) needs 6 of 7.

Arm A's single recorded "miss" was the string `color slate` against gold
`slate` — a formatting artifact scored as a retrieval failure.

## 4. The corpus that was used was not the corpus that was built

`data/unsealed/` holds the official CC0 unsealed shard: 256 rows, ~2000–2700
token contexts, **207 distinct gold answers**, hashed and committed with a
provenance sidecar. It was never read by any run.

`run_experiment.py` passes `force_synthetic=True` and trains and evaluates on
`data/synthetic/`: 512 rows, **10 distinct gold answers**, one repeated filler
sentence. Final training loss 2e-4 reflects a degenerate task, not convergence.

## 5. Training stopped while the gate was still ramping

`train.py:115` hardcoded `init_alpha=0.05`, overriding **both** configs' declared
`init_alpha: 0.0`. The recorded configs do not describe the recorded run.

(That hardcoded value is also what kept the pathway alive at all: `P.weight` is
zero-initialised, so with `alpha = 0` exactly, `dL/dalpha` and `dL/dP` are both
identically zero and the block is a permanent no-op. The zero-init of `P` with a
non-zero `alpha` is the standard LoRA-style "start as identity" trick and is
fine; the defect is that the config says otherwise.)

Over 24 optimizer steps at lr 1e-4, Adam's total movement bound is
24 x 1e-4 = 2.4e-3. Observed alpha movement was +1.6e-3 to +2.0e-3 — 67–83 % of
the theoretical maximum, in one direction, still rising when training stopped.
`lambda_raw` moved 0.19 %, `beta_raw` 0.10 %. In arm D, `beta_raw`/`lambda_raw`
are bit-identical to init because the significance gate orphans them from the
graph (this is correct behaviour, but it means arm D changes the *base*
forgetting rate as well as adding the gate, so it is not a minimum-mechanism
ablation of arm C).

## 6. Resource scaling (measured, RTX 4070 Laptop, 8188 MiB)

Warmup excluded; last-position logits only; peak VRAM reset per measurement.

| Context | Arm A prefill | Arm C prefill | Peak VRAM | Recurrent state | KV cache if used |
|---|---|---|---|---|---|
| 1024 | 5899 tok/s | 1401 tok/s | 1.32 GiB | 8192 B | 112 MiB |
| 2048 | 3630 tok/s | 1222 tok/s | 1.80 GiB | 8192 B | 224 MiB |
| 4096 | 1930 tok/s | 925 tok/s | 3.64 GiB | 8192 B | 448 MiB |
| 8192 | 164 tok/s | 154 tok/s | 10.79 GiB | 8192 B | 896 MiB |
| 16384 | OOM | OOM | — | — | — |

The 8192 row exceeds physical VRAM and is served by host-memory spill, which is
why throughput collapses ~6x. **16K is not measurable on this hardware.**

Two honest conclusions for Q5:

1. The recurrent state is genuinely bounded — 8192 bytes at rank 16, constant in
   T, against a KV cache that grows linearly to 896 MiB at 8K.
2. **That buys nothing here.** Peak VRAM is identical for A and C to within
   0.01 GiB, because the pathway is *auxiliary*: full attention still runs and
   still dominates. Only Arm E (replacement) could convert the bounded state
   into a memory saving. The P2 numbers cannot support any efficiency claim.

Separately, the wrapper hard-refuses KV-cache mixing, so free generation costs
O(new_tokens x T). That is what made long-context free-form evaluation
unaffordable, and is why P2b scores by candidate argmax in a single forward.

The P2 throughput table is additionally not comparable across arms: it was taken
with no warmup pass, so arm A's first measurement absorbed CUDA context
creation. That is why its recorded tok/s were non-monotonic in length.

## Verdict on the P2 conclusion

The recorded **NO-GO is withdrawn as unsupported**. It is not evidence that
fast-weight memory fails on Supa-A1.1; it is the null result of an experiment in
which

- the fast memory could not span the context (87-token horizon, total overwrite),
- the backbone already answered every item correctly,
- the test could not have reached significance short of a near-total reversal,
- the trained gate was still ramping when training stopped, and
- the corpus was a 10-word closed-answer template.

None of the preregistered hypotheses H1–H5 were actually tested. P2b re-runs the
matched arms with each of these defects fixed; see `PREREGISTRATION.md`
Amendment 1.
