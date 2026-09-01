# Final interpretation

## Decision: NO-GO

After P2 (seeds 11 and 22; seed 33 unaffordable), the preregistered GO rule is
not met.

- C does not beat B. Predictions are identical.
- D does not beat C. Predictions are identical.
- The only C>A difference is one malformed prefix on Arm A (`color slate` vs
  `slate`). B shows the same “fix”. That is extra-parameter / formatting
  behaviour, not recurrent fast-memory.

A negative result is retained.

## Established

- Prefix-aligned FALCON-2 can be attached to frozen Supa-A1.1 as an auxiliary
  pathway. α=0 recovers unmodified logits on GPU.
- Arm B and Arm C have the same trainable count (524,593 per layer).
- Recurrent state is 8,192 bytes/layer at rank 16 and does not grow with T.
- The path can be optimized for 24 steps without NaN if adapters stay float32
  and examples fit the sequence budget.
- On this unsealed synthetic colour-retrieval suite (256/512 tokens), B, C, and
  D all score 14/14 and match on the interference battery.
- Adding FWA did not reduce peak VRAM versus the static control and made
  forward tokens/sec worse than unmodified Supa.

## Suggestive

- Frozen Supa already solves these short closed-context colour items. There was
  almost no remaining error for fast memory to repair.
- INT-D (contradictory update) failed on every arm, including A. The suite does
  not isolate a FWA-specific overwrite bug; the backbone already misses that item.

## Failed hypotheses

- **H2 / Q2.** Recurrent dynamics beat extra parameters: **failed** (C = B).
- **H4 / Q4.** Significance gating helps: **failed** (D = C).
- **H3 / Q3.** Middle-vs-edge robustness: **not supported**. Gap is already 0
  for A on 13/14 items; the remaining miss is not positional.

## Untested

- Contexts 2K–16K with a non-ceilinged, position-controlled bank (PSA
  validation items were not materialised; sealed PSA data was not opened).
- Official CC0 2k/4k rows trained **without** truncation.
- Arm E (attention replacement).
- Seed 33.
- Chunk-parallel FALCON-2 / FALCON-3.
- Whether FWA can replace KV cache (it was never substituted).

## Strongest evidence against the working hypothesis

Identical B and C item-level outputs, plus Arm A already at 13/14 with the
same interference profile. Any apparent C>A gain is shared with the static
control.

## Exact next experiment (only if someone still wants the question)

Use a **non-ceilinged** unsealed position suite (PSA calibration/validation
generation, or CC0 2k rows scored untruncated) at 2K+ with the same A/B/C
adapters, one seed, layer 14, rank 16. If C still equals B, stop. Do not open
showcase64, TEST2, or PSA sealed data to shop for a win.
