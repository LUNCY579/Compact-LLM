# Rineya Fast Weight Attention Lab v1

Exploratory, low-resource test of a FALCON-2-inspired **auxiliary** fast-weight
memory pathway on frozen **Supa-A1.1**. Negative results are valid and must be
kept.

This lab does **not** assume FALCON is beneficial.

## Isolation record

| Field | Value |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\RineyaBeta-supa-fwa-v1` |
| Branch | `research/supa-a1.1-fast-weight-attention-v1` |
| Base branch | `research/positional-sequential-attention` |
| Base commit | `ee13448f11efaa7f14206dab95b277816b912ce6` |
| Worktree created (UTC) | `2026-09-01T22:04:34Z` |
| Source checkpoint | `Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze` |
| Core ID | `rineya.core.supa-a1.1.v1.1.0-devfreeze.28L.qwen3-nimbus-lcr-step200-scale-0p8125-fp16-abstain-b5p15625` |
| Weights SHA-256 | `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026` |
| Store dtype | float16 (mandatory) |
| Architecture | 28-layer Qwen3-family, hidden 1024, 16 Q / 8 KV heads, head_dim 128 |
| Abstention | token 51378, `+5.15625` logit offset in `Qwen3AbstainForCausalLM.forward` |

The checkpoint directory is **immutable**. No file in this lab writes under
`Rineya_Published_Cores/`.

## Scientific questions

- **Q1.** Can a FALCON-style fast-weight memory improve frozen Supa-A1.1?
- **Q2.** If so, is it the recurrent dynamics, or merely extra parameters?
- **Q3.** Does it change long-context positional robustness (middle vs edge)?
- **Q4.** Does significance-gated writing beat ordinary FWA writing?
- **Q5.** What are the memory, latency, throughput, and VRAM tradeoffs?

## Arms

| Arm | Meaning |
|---|---|
| A | Unmodified Supa-A1.1 |
| B | Parameter-matched **static** control (no recurrent state carry) |
| C | Frozen Supa + prefix-aligned FALCON-2 pathway |
| D | Arm C + Rineya significance-gated plasticity/forgetting |
| E | Deferred until A–D are interpretable |

Primary contrasts: **C−A**, **C−B**, **D−C**.

## Layout

```
Rineya_Fast_Weight_Attention_Lab_v1/
  README.md
  PREREGISTRATION.md
  research/PAPER_NOTES.md
  configs/
  src/rineya_fwa/
  scripts/
  tests/
  reports/
artifacts/supa_a1_1_fast_weight_attention_v1/   # gitignored run outputs
```

## Tooling status — what was built and whether it was run

Recorded so a reader does not mistake a committed script for evidence.

| Script | Run? | Notes |
|---|---|---|
| `run_experiment.py` (P1/P2) | yes | produced the **withdrawn** P1/P2 results |
| `diagnose_mechanism.py` | yes | the measurements that withdrew the P2 NO-GO |
| `profile_scaling.py` | yes | resource scaling 1K-16K |
| `calibrate_suite.py` | yes | arm-A difficulty calibration (Amendment 2) |
| `run_p2b.py` | yes | the P2b A/B/C/D run |
| `make_report.py` | yes | `RESULTS_P2B.md` + figures |
| `probe_layers.py` | yes | 28-layer zero-training probe |
| `run_cshort_replication.py` | yes | Amendment 11 step 0 |
| `run_h7_early.py` | yes | Amendment 12 step 1 |
| `run_layer_pilot.py` | **NO** | **cancelled by Amendment 10** before running. Its design was reviewed and rejected: it reused a suite with 9 items of headroom and would have returned four foregone nulls at ~9 GPU-h. Kept as the record of a rejected design, not as pending work. |
| `analyze_interference.py` | **NO** | built to add the "blended" outcome the main scorer cannot see (brief section 14). Not run because Amendment 9 established this battery is both ceilinged for gains and too seed-noisy to attribute anything to an architecture. Running it would produce numbers no claim may rest on. It becomes useful only against a rebuilt battery. |
| `eval_p2_on_v2.py` | **NO** | written as a fallback to ask whether the P2 adapters separate on a discriminative suite. P2b answered that question directly and better. Superseded, not pending. |
| `smoke_babysit.py` | yes | P1-era smoke helper |

**Section 14 of the brief is not answered by this lab.** The interference battery
could not support a claim in either direction, and the tool that would have
characterised failure modes was deliberately not run against it.

## Sealed-data ban

Do not open, train on, or iterate against:

- CC0 `sealed_eval/`
- Invisible-Limit TEST / TEST2
- Demo v1 showcase64 / stability192 as a tuning set
- PSA `data/sealed/`

## Status

**P2b complete. Decision: NO-GO**, at layer 14, in a compressed range.
See `reports/RESULTS_P2B.md`, `reports/FINAL_INTERPRETATION.md`, `reports/figures/`.

A bounded FALCON-2 fast-weight memory on frozen Supa-A1.1 gives **no positional
benefit over a parameter-matched static control**, and **costs** interference
performance.

| Arm | position (of 168) | interference (of 56) |
|---|---:|---:|
| A frozen | 138 | 55 |
| B static control | 159 / 159 | 55 / 51 |
| C recurrent | 160 / 161 | 50 / 46 |
| D significance-gated | 161 / 160 | 55 / 51 |
| C-short (original horizon) | **163** | 55 |

- **C vs B, position: 2/1 and 3/1 discordant of 168.** Null at both seeds. Arm B
  alone repairs the primacy deficit (position 0.05: 0.417 -> 0.917).
- **C loses 5 interference items to B at both seeds**, zero gains in 10
  discordant pairs; worse than the *untrained* backbone (p = 0.0039 at seed 22).
- **D is identical to B on interference** at both seeds. The significance gate's
  whole demonstrated value is cancelling a cost recurrence introduced.
- **C-short — the horizon I argued was too short — is the best arm.** The
  interference cost tracks the horizon, not recurrence. My Amendment 1 horizon
  change caused the only harm attributable to the fast memory.

Limits carried with the decision: one layer (the brief's layer pilot was never
run), two seeds, a range compressed by calibrating against the frozen rather than
the trained control, and an interference battery that is ceilinged for gains and
seed-noisy. The earlier P2 NO-GO remains withdrawn — it measured nothing; this
one measures something.

Prior status, retained: the P2 withdrawal is documented in
`reports/MECHANISM_DIAGNOSTIC.md`; preregistration and its eight amendments,
including four corrections to my own errors and one failed prediction, are in
`PREREGISTRATION.md`.
