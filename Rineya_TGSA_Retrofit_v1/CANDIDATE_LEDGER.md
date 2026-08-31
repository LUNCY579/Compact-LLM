# Rineya production-retrofit candidate ledger

Machine-readable source: [`evidence/candidate_ledger.json`](evidence/candidate_ledger.json)

Schema: [`schemas/candidate_ledger.schema.json`](schemas/candidate_ledger.schema.json)

Audit snapshot: 2026-08-31 UTC, after `git fetch --prune origin`

Retrofit base: `research/tgsa-rineya-pretrained-retrofit` at `0703ba41e7b7d0a31fb3cc1ad8003fce0a312140`

## Intake result

The first-pass current remote research set contained no eligible mechanism. The
second-pass historical harvest admitted three bounded checkpoint trials. One
was transfer-positive but failed production preservation; two failed transfer.
No candidate is promotable, and no resource/demo promotion work is authorized.

| Candidate | Source commit | Research classification | Exact parent | Decision | Reason |
|---|---|---|---|---|---|
| A1.1/Baya 50/50 soup | `a07abdb` | `ELIGIBLE`; transfer-positive | Supa-A1.1 | `REJECT` | 170→173/256 and 20→27 all-four packs, but 51→42 base/paraphrase both-correct packs and Showcase 45→44 block general promotion. |
| A1.1 adapter dose 1.0625 | `8324995` | `ELIGIBLE`; transfer-fail | Supa-A1.1 | `REJECT` | Exact reconstruction fidelity passes, but 170→170/256 and 51→40 base/paraphrase packs fail strict gain and preservation. |
| A1.1 + Pulse negative vector | `8324995` | `ELIGIBLE`; transfer-fail | Supa-A1.1 | `REJECT` | Zero-scale fidelity passes, but 170→60/256, 125→45 Stability, and 20→0 all-four packs confirm catastrophic anchor incompatibility. |
| TGSA Phase-1 v5 mechanism | `d16ea1c` | `WAITING_FOR_EVIDENCE` | Supa-A1.1 | `DEFER` | V5 is a draft competence preregistration. V3 was inconclusive; v4 rejected the regime before a confirmatory mechanism matrix. |
| TGSA + MTP Phase 2A | `d16ea1c` | `WAITING_FOR_EVIDENCE` | Supa-A1.1 | `DEFER` | CI, rehearsal, and integrity gates are not a positive result; within-study qualification has not produced a GO. |
| PSA / SGSA interventions | `ee13448f` | `WAITING_FOR_EVIDENCE` | Supa-A1.1 | `DEFER` | Measurement phases exist, but no treatment result, model selection, preservation run, sealed test, or final report exists. |
| 4B/8B spectral consensus | `538687f` | `NEGATIVE` | Supa-A1.1 | `REJECT` | The development +1 disappeared on DEV-FULL: 144/180 vs 144/180, one fix and one break, macro delta −0.0045. V2 remained near-null. |
| Normalized significance/count KV retention | `b97cd8c` | `NEGATIVE` for product quality | Halo 1.2 | `REJECT` | The retained-state endpoint improved, but registered answer quality was 5/12 vs 6/12 and later quality screens were null/negative or failed specificity. |
| Supa-A1.1 FP16 + abstention | `34de429` | `ALREADY_INTEGRATED` | Supa-A1 | `ALREADY_INTEGRATED` | Already published as the strongest suitable dense Supa parent: Showcase 45 vs 43 and Stability tied at 125. No significance claim. |
| Supa abstention threshold copied to Baya | `34de429` | `NEGATIVE` | Baya 1.7 | `REJECT` | The source report says the unchanged threshold lowers Baya Showcase from 40 to 35. |

## Recorded measurements

Scores below are quoted only where the source branch contains a directly
comparable parent/candidate result. Blank cells mean the candidate did not pass
the research intake gate, so production measurement was not run.

| Candidate | Metric | Parent | Candidate | Delta | Fidelity | Regression | Artifact | Demo |
|---|---|---:|---:|---:|---|---|---|---|
| A1.1/Baya 50/50 soup | all-256 strict hits | 170 | 173 | +3 | pass | **fail** | `73312805…` | not run |
| A1.1 adapter dose 1.0625 | all-256 strict hits | 170 | 170 | 0 | pass | **fail** | `59f3f5ba…` | not run |
| A1.1 + Pulse negative vector | all-256 strict hits | 170 | 60 | −110 | pass | **fail** | `783c2c94…` | not run |
| 4B/8B spectral consensus | DEV-FULL semantic hits / 180 | 144 | 144 | 0 | not run here | fail | none | not run |
| Normalized KV retention | registered answer hits / 12 | 6 | 5 | −1 | n/a | fail | none | not run |
| Supa-A1.1 FP16 + abstention | Showcase hits / 64 | 43 | 45 | +2 | pass (publisher) | pass (publisher bounds) | `a99f079d…` | not run here |
| Supa threshold on Baya | Showcase hits / 64 | 40 | 35 | −5 | n/a | fail | none | not run |

## Strict promotion predicate

`PROMOTE` is accepted only when every field below is explicitly `true` and the
declared quality delta is finite and strictly positive:

1. `positive_source_evidence`
2. `parent_identity_verified`
3. `conversion_fidelity_pass`
4. `baked_effect_verified`
5. `strict_quality_gain`
6. `no_blocking_regression`
7. `production_budget_pass`
8. `artifact_integrity_pass`
9. `demo_smoke_pass`
10. `provenance_complete`

Missing and `null` gates fail promotion. `NEGATIVE` entries must be `REJECT`;
`WAITING_FOR_EVIDENCE` and `INCOMPATIBLE` entries must be `DEFER`. The validator
also rejects duplicate candidate IDs, mutable/non-SHA commit identities,
malformed hashes, inconsistent score deltas, and unavailable git provenance.

Validate the ledger with:

```powershell
Set-Location Rineya_TGSA_Retrofit_v1
python .\scripts\verify_candidate_ledger.py
```

## Re-intake rule

Refresh this ledger only after a source branch writes an immutable result that
passes its own frozen gate. The new record must pin the source commit,
configuration, result artifact, and exact intended parent. Branch existence,
successful CI, conversion exactness, promising telemetry, or a development-only
gain is not positive evidence.
