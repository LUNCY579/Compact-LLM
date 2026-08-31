# Rineya historical production trial queue

Machine source: [`evidence/production_trial_queue.json`](evidence/production_trial_queue.json)

Schema: [`schemas/production_trial_queue.schema.json`](schemas/production_trial_queue.schema.json)

This queue records mechanisms worth a bounded modern transfer trial. It does
not record promotable models. A queue `PASS` means `TRANSFER_POSITIVE`; shipment
still requires the separate unchanged ten-gate promotion ledger.

## Ranked bounded trials

| Rank | Mechanism | Class | Historical signal | Modern target/status | Why this position |
|---:|---|---|---|---|---|
| 1 | Uniform FP32 reconstructed-delta soup | CHECKPOINT | Phase-4 seed_37/seed_49 soup improved mean quality, preserved worst decile, zero hard/critical failures, human PASS | Supa-A1.1 + Baya midpoint; `QUEUED` | Same base, adapter, tensors, tokenizer, and topology; one candidate; no training/runtime cost |
| 2 | Long-context adapter dose 1.0625 | CHECKPOINT | Nimbus + checkpoint-200 dose region created Supa/Baya and large long-context gains | Supa-A1.1 lineage; `QUEUED` | Exact same adapter lineage and one historically measured dose, but less directly supported than the midpoint soup |
| 3 | Pulse negative task vector | CHECKPOINT | Pulse improved from 31/64 to 41/64 and 117/192 | A1.1 transfer; `QUEUED` only after ranks 1–2 fail | Strong gain and bakeable, but historical evidence says the effect is anchor-dependent |
| 4 | Context compaction | RUNTIME | 29.10% memory-token reduction with complete measured recall and no reported quality loss | Efficiency profile; `QUEUED` | Real product-stack win, but cannot satisfy the immediate one-checkpoint objective |

## Positive mechanisms not entering the immediate queue

| Mechanism | Standing | Reason |
|---|---|---|
| Cross-depth/global/layerwise soup | `ALREADY_INTEGRATED` | Forms the Nimbus ancestry already present in A1.1; no non-arbitrary new donor is selected |
| FP16 + S4 abstention | `ALREADY_INTEGRATED` | It is the exact preferred parent and must be preserved |
| System42/45/46/48 routing | `ALREADY_INTEGRATED` | Nonlinear multi-execution systems cannot honestly be baked into one checkpoint |
| Phase-1 dense SFT | `DEFER_MISSING_ARTIFACT` | Old task/lineage; exact adapter absent; would require new training |
| Phase-2C targeted recovery | `DEFER_MISSING_ARTIFACT` | Bound to the old 19-layer deletion profile |
| Phase-3 DPO | `DEFER_MISSING_ARTIFACT` | Small old-lineage effect; no A1.1-compatible preference adapter |
| Phase-9 v2.2 greedy adapter soup | `DEFER_MISSING_ARTIFACT` | Different compact-student topology; enclosing production gate remained blocked |
| Phase-2B structural reduction | `EXCLUDED` | Valid specialized efficiency frontier, not a quality-first dense improvement; later extra compression was negative |
| Adapter consensus | `EXCLUDED` | System-only local effect, failed strong-expert/cross-corpus tests, high latency, no A1.1 committee |

## Frozen first trial

Registration:
[`configs/historical_soup_a11_baya_midpoint_trial.json`](configs/historical_soup_a11_baya_midpoint_trial.json)

The only candidate is:

```text
0.5 * FP32(Supa-A1.1 weights) + 0.5 * FP32(Baya-1.7 weights)
→ float16 checkpoint
→ retain A1.1 tokenizer, custom class, and S4 abstention offset
```

Transfer requires all of the following:

1. at least one additional strict-exact hit on all 256 rows;
2. no Stability192 hit loss;
3. no all-four-pack loss;
4. complete identity/coverage and no generation failure;
5. fresh-process reload as `Qwen3AbstainForCausalLM`;
6. complete artifact/evaluation hashes.

A tie is `TRANSFER_FAIL`. No coefficient, layer, prompt, or gate adjustment is
permitted after observing the candidate.

## Validation

```powershell
Set-Location Rineya_TGSA_Retrofit_v1
python .\scripts\verify_production_trial_queue.py
python -m pytest .\tests\test_trial_queue.py -q
```
