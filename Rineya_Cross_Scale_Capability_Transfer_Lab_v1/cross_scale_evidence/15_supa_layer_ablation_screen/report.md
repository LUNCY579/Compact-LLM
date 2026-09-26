# Supa-A1.1 Single-Layer Importance Fast Screen — Report

Protocol: `15.0.0-exploratory` (`protocol.md`)

Model: `Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze`  
Operator: identity residual skip via forward hook  
Bank: 8-prompt smoke (`supa_layer_ablation_smoke.jsonl`, SHA `404c866f…`)  
Sealed: **CLOSED**. Donors/merges: none.

## Abstract

Skipping early decoder layers **catastrophically** breaks Supa. Layers **1** and
**2** are the strongest hinges: identity-skip produces empty/whitespace,
fullwidth punctuation loops, Thai/Cyrillic/Korean garbage, and replacement
characters. Layer **0** also trips the critical bar via logit collapse and
mojibake-like mixed scripts, though without a 50% invalid-generation rate.

Most middle layers (especially **5–16**, best: **8**) are nearly redundant under
this operator: logit cosine vs intact stays ≥0.95 and outputs remain coherent.
Late layers **19** and **27** are soft (cosine ~0.71–0.73) but do **not** produce
garbage on this smoke set.

## What was run

| Step | Status |
|---|---|
| Hook verification (layer 14) | PASS — skip changes logits; restore is bit-identical |
| Intact baseline (8 prompts) | complete |
| Leave-one-out layers 0–27 | complete, resumable JSON under `smoke/` |
| Ranking | `ranking.json` / `ranking.md` |

## Critical layers

| Layer | Score | Invalid rate | Weird hits (≥0.25) | Mean logit cos | Mean NLL |
|---:|---:|---:|---:|---:|---:|
| **1** | 5.14 | **1.00** | 4 | 0.303 | 3.16 |
| **2** | 4.69 | **0.50** | **7** | **0.062** | 0.70 |
| **0** | 1.42 | 0.00 | 1 | 0.384 | 1.46 |

### Qualitative examples

Intact baseline remains sensible (`Ice melts and becomes water.`, `87`, valid JSON).

**Skip layer 1:** empty strings; 24 spaces; fullwidth `．．．` loops; Thai consonant
repetition; newline floods.

**Skip layer 2:** Cyrillic `мещ.` loops; U+FFFD replacement floods; mixed Hangul;
many empty completions. Lowest logit alignment of any layer (0.062).

**Skip layer 0:** mixed CJK/Thai/Greek fragments and English debris — incoherent
but not always “invalid” under the repetition rule.

## Soft layers (degraded but not garbage)

`19, 27, 25, 17, 18` — cosine 0.71–0.89, invalid_rate 0, weird_hits 0.  
Example layer 19: still English/JSON/`UNKNOWN`, with mild content drift
(`Ice cubes and ice cubes in warm air.` vs intact melt phrasing).

## Null / robust layers

Best: **8** (cos 0.978). Broad null band **3–16** and most of **20–24, 26**.
Single mid-depth skip rarely destroys fluency on this bank.

## Resources

| Quantity | Value |
|---|---:|
| Wall time | 553.8 s (~9.2 min) |
| Model loads | **1** |
| Peak RSS | 3.10 GiB |
| End RSS | 3.02 GiB |
| Device | CPU float32, 4 threads |
| GPU | refused/unused (card busy at 10% / ~1 GiB) |
| New weight checkpoints | **0** |

## Interpretation

1. **Yes — some single layers are catastrophic hinges**, concentrated at the
   **bottom of the stack (0–2)**, especially **1** and **2**.
2. Supa is **highly redundant** across most middle layers under identity skip.
3. This is inference-time leave-one-out sensitivity, **not** a training
   importance or merge-localization claim.

## Escalation

| Gate | Justified? |
|---|---|
| MERGEDEV-fast on critical + soft + best-null controls | **Yes** (optional follow-up; critical layers found) |
| DEV-FULL / sealed | **No** |
| Multi-layer removal / physical depth deletion | Not required to answer the smoke question |

## Files

- `smoke/baseline_intact.json`, `smoke/layer_00.json` … `layer_27.json`
- `smoke/intact_logits.pt` (gitignored if large; reproducible from baseline reload)
- `ranking.md`, `ranking.json`, `run_manifest.json`, `hook_verification.json`, `resource.json`
