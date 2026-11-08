# This host cannot train this model beyond ~250-token sequences, and four runs found out the hard way

Testing whether exposure to long sequences preserves off-distribution termination requires
training on long sequences. Four attempts aborted. The cause is a hardware ceiling, not a
tuning problem, and the recorded `resource_trace` had the answer from the first abort.

## The measurement

`cuda_peak_allocated_gib` tracks the longest sequence the run has seen, from the fourth
attempt's trace:

    step   tokens   peak GiB   host free GiB
     1-12   95-211      5.88           5.08
       14      555      8.34           3.16
       19      746     10.16           1.52
       21     1000     12.98           0.32   -> stood down

The card has **8 GiB**. A 13 GiB peak exists only because Windows WDDM backs GPU allocations
with system commit, so the excess lands in host RAM and `hostbudget` stops the run.

Fitting the four points gives roughly `peak_GiB ~ 5.0 + 0.008 * tokens`, so staying under 8 GiB
means about **250 tokens**. The programme's existing rows are ~128 tokens, so the usable headroom
for longer training data on this machine is under 2x.

## Why, and why the obvious fixes did not work

Two terms scale with sequence length and neither is avoidable by configuration:

* **attention scores** -- 16 heads x seq^2 x 4 bytes x 28 layers, about 1.8 GiB at 1,000 tokens
  against 72 MB at 200;
* **logits** -- seq x **151,936** vocab x 4 bytes, about 0.6 GiB at 1,000 tokens, materialised
  several times over through `.float()`, `log_softmax` and the backward pass.

`gradient_checkpointing = True` made it **worse**, not better (reserved 13.50 against 10.92):
recomputation adds transient allocations, and the logits are computed *outside* the checkpointed
blocks, so the dominant term is untouched.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did not help either, because the growth is
genuine peak allocation rather than fragmentation. `cuda_allocated_gib` sat at 4.58 throughout
every run; what moved was the peak, in lockstep with sequence length.

## The process failure worth recording

Four aborts, and the first three were guesses:

1. 2,048-token cap -> died at step 11, the moment the first long rows arrived;
2. 1,024-token cap -> died worse, treating a quadratic cost as if halving the input halved it;
3. gradient checkpointing -> died worst, attacking a term that was never dominant;
4. expandable segments -> died, attacking fragmentation that was never occurring.

`resource_trace` records `cuda_allocated_gib`, `cuda_reserved_gib` and `cuda_peak_allocated_gib`
per step, and has since the first abort. Reading it before attempt 2 would have ended this in one
step. The instrumentation existed; it was not consulted. This is the same failure as quoting
`task_strict_hits` when `task_numeric_hits` was the series that answered the question -- reaching
for the plausible number instead of the one that decides.

## What this forecloses, and what it does not

**Forecloses:** training on Rineya Reasoning Forge on this host. Its rows are ~1,400 tokens
median, 8,750 maximum. Nothing above ~250 tokens is trainable here, so the mixed-length
experiment cannot be run as designed, and the "does long-sequence exposure preserve stopping"
hypothesis is untestable on this machine.

**Does not foreclose:** forge as an EVALUATION. Inference needs no optimizer state, no gradients
and no stored activations; the floor probe scored 200 rows up to 2,048 tokens in 110 seconds.
Forge remains the better-powered instrument and every checkpoint can still be scored on it.

**The path to long-context training here**, if it is ever wanted, is chunked cross-entropy:
compute the loss in slices over the sequence so full-vocabulary logits are never materialised at
once. That removes the larger of the two terms. It is real work, it belongs in `arm_dose`
(outside the hashed root), and it should not be attempted while a conclusion depends on it.

---

## Addendum: forge's `test_length` split cannot be run here at all, even at inference

`test_length` is the third forge evaluation split and the only instrument in this repository
that has never been measured. It is the long-input stress set: five rows per family at word
tiers up to 4,096, the same reasoning problems embedded in progressively more distracting
context.

Measured prompt lengths, tokenised with the source tokenizer and its chat template:

    80 rows.  min 11,179   median 15,918   max 35,875 tokens
      <= 2,048 tokens:  0 rows
      <= 8,192 tokens:  0 rows
      <=16,384 tokens: 42 rows

The model supports 40,960 positions, so this is not a context-window limit. It is attention
memory. `model.py:68` computes attention eagerly --
`scores = q @ k.transpose(-1,-2) * d**-0.5` -- which materialises the full `[heads, T, T]`
matrix:

    T =  4,096  ->  1.00 GiB of scores
    T =  8,192  ->  4.00 GiB
    T = 11,179  ->  ~7.4 GiB        (the SHORTEST row in the split)

Plus 2.22 GiB for the model itself. On an 8 GiB card **not one row of this split fits**, and
this is inference: no gradients, no optimizer state, no stored activations. Chunked
cross-entropy does not help, because the bottleneck is attention rather than the head.

The fix would be memory-efficient attention -- `scaled_dot_product_attention` with a flash or
mem-efficient backend, which never materialises the score matrix. That is not available here for
two reasons. `model.py` is inside the hashed root, so changing it moves the implementation digest
and invalidates the gate ladder. And its own docstring at line 53 states the implementation
reproduces "the original arithmetic bit-for-bit"; a different attention kernel changes the
numerics that every frozen comparison in this programme depends on.

So the evaluation picture is complete as far as this hardware allows:

    test_iid       200 rows scored, noise floor measured   -- usable
    test_transfer  200 rows scored, noise floor measured   -- usable
    test_length      0 rows scorable                       -- blocked by attention memory

Recorded rather than worked around. A claim about this model's long-input behaviour cannot be
made from this machine, and the earlier finding that verbosity is a *generation*-length effect
therefore remains untested against *input* length.
