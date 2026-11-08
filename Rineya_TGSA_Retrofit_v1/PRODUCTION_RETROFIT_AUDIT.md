# Rineya production-retrofit first-pass audit

## Decision

**`DEFER — RESEARCH EVIDENCE INSUFFICIENT`**

No active research mechanism satisfies the intake rule at the audited remote
commits. The correct production action is therefore to ship no new model. This
audit improves the branch's intake and promotion controls without converting,
training, evaluating, packaging, or demoing an unauthorized candidate.

Audit time: 2026-08-31 UTC, after `git fetch --prune origin`

Retrofit branch: `research/tgsa-rineya-pretrained-retrofit`

Audited base commit: `0703ba41e7b7d0a31fb3cc1ad8003fce0a312140`

Remote: `https://github.com/LUNCY579/ChatBubble.git`

The worktree also contained an untracked
`Rineya_TGSA_Retrofit_v1/PHASE2A_PREREQUISITE_HOLD.md` before this audit. It was
treated as user-owned and left unmodified. Its sequencing rule agrees with the
evidence audit: no real-model TGSA execution is currently authorized.

## 1. Existing retrofit infrastructure

| Surface | Current implementation | Verified capability | Production limitation |
|---|---|---|---|
| Retrofit configuration | `rineya_tgsa/config.py` | Reads explicit Qwen head dimensions and both top-level and transformers-5 nested RoPE layouts; fails closed on unknown RoPE | No source-research configuration importer or immutable configuration identity check |
| Reference operator | `rineya_tgsa/attention.py` | GQA, RoPE, cache layout, sequential head carrier, neutral zero-initialized conditioning, gradients, telemetry | Hand-written reference omits Qwen-native `q_norm`/`k_norm`; it is a test/reference implementation, not the real checkpoint adapter |
| Real Qwen adapter | `rineya_tgsa/qwen3_adapter.py` | Subclasses transformers `Qwen3Attention`, reuses `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, and `k_norm`; consumes model-supplied RoPE | Runtime class exists only in this package; no standard loader registration or self-contained model artifact exists |
| In-memory model conversion | `convert_qwen3_model` | Replaces selected decoder attention modules, rejects out-of-range layers, verifies neutral conditioning | Converts a live Python object only |
| Conversion CLI | `scripts/convert_rineya.py` | Pins source config/weight hashes, converted layer list, new-state hashes, environment, and native-tensor reuse | Writes a manifest and optional attention state only; it does **not** write a standalone Hugging Face checkpoint |
| Source integrity | `scripts/build_source_manifest.py` | Existing evidence hashes 12 source artifacts and 310 tensors and matches the publisher manifest | Manifest currently targets BF16 Supa-A1, not the stronger FP16 Supa-A1.1 |
| Fidelity measurement | `damage.py`, `real_layer_exactness.py`, `real_conversion_damage.py` | Existing Supa-A1 evidence reports neutral eager conversion at zero model-level drift and positive-control sensitivity; real Qwen layers preserve all six native tensors | Evidence covers the older A1 parent, short prompts, and selected layers; no fresh-process standalone-artifact reload exists |
| Generation/cache test | `scripts/generation_test.py` | Native-vs-neutral greedy generation, logits, cached-vs-uncached behavior, and finite outputs | Rebuilds conversion in memory; does not load a baked artifact and does not cover production fixtures or long-context scaling |
| Development training | `scripts/train_retrofit.py` | Bounded R0/R1/R2 freeze stages, tiny repeated-text overfit, finite loss/gradients, resume payload | Not an evidence-bearing bake; no selected research curriculum/config, exact-parent control, complete model save, or baked/runtime equivalence |
| Placement and schedule | `schedule.py` | Deterministic layer-placement candidates and progressive freeze-stage logic are unit tested | No real-model placement/warm-start driver or result; open-ended tuning is forbidden here |
| Telemetry | `scripts/trace_retrofit.py` | Head-wise carrier, velocity, control, conditioning, entropy, top-token trace | Diagnostic only; cannot establish quality gain |
| Candidate intake | Added by this audit: JSON ledger, JSON Schema, semantic validator, git provenance check | Every candidate records source commit/evidence, exact parent, scores/statuses, artifact/demo fields, decision, and reason | Must be refreshed when research branches publish new immutable results |
| Promotion predicate | Added by this audit: `rineya_tgsa/promotion.py` | `PROMOTE` is a strict ten-gate logical AND plus a finite positive quality delta and artifact hash | No candidate currently reaches the predicate |
| CI | Added by this audit: `.github/workflows/rineya-retrofit.yml` | Full git history for provenance checks, pinned transformers compatibility window, ledger verification, all tests, compile check | Real model/artifact jobs remain operator-gated and are not run in CI |

### Preserved Supa foundation

The committed Supa-A1 evidence establishes a useful prerequisite:

- source `model.safetensors`: 1,192,135,096 bytes, SHA-256
  `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`;
- 28 Qwen3 layers, hidden size 1024, 16 query heads, 8 KV heads, head dimension
  128, RoPE theta 1,000,000;
- every native attention tensor is retained by the production adapter;
- neutral real-layer eager comparisons in the persisted report are bitwise
  identical for float64, float32, and bfloat16;
- the full-model short-prompt neutral report records KL 0, top-1 agreement 1.0,
  and max absolute logit drift 0;
- perturbation controls produce measurable drift, so the harness is not
  trivially insensitive.

This proves that the existing A1 converter can preserve its parent on the tested
path. It does not prove TGSA works, transfers, bakes, or improves quality.

## 2. Verification performed in this audit

The initial full retrofit suite returned **56 passed, 1 failed**. The failure
was the hand-written reference test at the real 16-head dimension. It asserted
bitwise equality between batched-head and per-head CPU matrix multiplication,
even though those shapes select different floating-point reduction orders. The
production Qwen adapter already has a separate native-eager bitwise test.

The reference gate now uses a strict float64 `atol=rtol=1e-12`, matching the
brief's requirement for an appropriate strict numerical tolerance. The focused
regression plus the five new promotion tests passes. The complete suite must be
green before commit; see the verification record in the eventual commit handoff.

## 3. Missing production infrastructure

The branch was a credible conversion prototype, not a complete production
retrofit pipeline. The following gaps block any future promotion until filled
for an eligible candidate:

1. **No self-contained artifact writer.** `convert_rineya.py --save-weights`
   writes selected attention state, not a loadable model directory.
2. **No fresh-process artifact verifier.** The former runbook referenced a
   nonexistent `verify_conversion.py`.
3. **No enabled-runtime versus baked-artifact equivalence gate.** There is no
   baked artifact to compare.
4. **No exact-parent production evaluator.** Existing damage/generation tools
   are correctness diagnostics, not target-quality and preservation suites.
5. **No serving benchmark in this branch.** TTFT, decode throughput, total
   latency, model load time, peak VRAM, peak host RAM, and checkpoint-size
   budgets are not measured.
6. **No generic package/release manifest.** Published-core scripts are
   transactional and hash-aware but hard-coded to earlier releases.
7. **No exact-artifact demo binding.** Local evaluation can load arbitrary
   paths, but there is no manifest-locked retrofit demo smoke.
8. **No rollback record for a retrofit release.** Existing publisher scripts
   preserve old cores, but this lab does not bind a candidate to that machinery.
9. **No research-result importer.** Source branch, commit, selected config, and
   GO artifact were previously prose-only; the new ledger is the first
   fail-closed intake layer.
10. **Stale operator documentation.** The runbook described implemented scripts
    as absent while invoking three scripts that do not exist. It has been
    corrected to current commands and explicit gaps.
11. **Runtime dependency omission.** The package imported transformers and
    safetensors operationally without declaring them. The verified 5.14/5.15
    transformers compatibility window and safetensors runtime are now explicit.

These are implementation requirements after a candidate becomes eligible, not
permission to invent or tune a mechanism now.

## 4. Parent priority and compatibility

### Supa — priority 1

The strongest suitable **dense, single-checkpoint** Supa parent is
`Rineya-Supa-A1.1-1.1.0-devfreeze`, not the older A1 used by the current
evidence.

| Property | Supa-A1.1 |
|---|---|
| Weight SHA-256 | `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026` |
| Store/load dtype | float16 |
| Architecture | `Qwen3AbstainForCausalLM` over Qwen3 decoder layers |
| Layers / hidden | 28 / 1024 |
| Query / KV heads | 16 / 8 |
| Head dimension | 128 |
| Tokenizer SHA-256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Runtime requirement | `trust_remote_code=True`, float16 load, greedy decode |
| Published scores | 45/64 Showcase, 125/192 Stability, 170/256 combined |

Its decoder topology, tensor shapes, GQA mapping, tokenizer, and RoPE convention
match the adapter's assumptions. The custom outer class adds a constant
abstention logit offset; it does not change the decoder-layer path used by
`convert_qwen3_model`. Compatibility is therefore **structurally likely but not
verified**. The A1.1 weight is not cached locally, and no actual A1.1 conversion,
fidelity, save/reload, or generation test has run. A future candidate must pin
this exact hash and preserve the custom abstention runtime or implement and test
its portable logit-bias equivalent.

`Rineya-Supa-2.0.0-system48` scores higher as a family system, but it is a
nine-arm ordered consensus cascade rather than one checkpoint. It is not a
valid parent for the current checkpoint transformer unless a product profile
explicitly targets system-level retrofit and compares the exact full system.

### Baya — priority 2

The checkpoint target is `Rineya-Baya-1.7.0-devfreeze`, SHA-256
`1737cd1ee2bdde714c259adf2012ae44c89ad82a06797653343bad7e3cf88bba`.
It is a standard `Qwen3ForCausalLM` BF16 dense core with the same 28-layer,
1024-hidden, 16/8-head, head-dimension-128, tied-tokenizer architecture. Its
top-level `rope_theta=1000000` is supported by `RineyaTGSAConfig`.

This is **structurally compatible in principle but unverified on weights**. The
Baya weight is not cached locally. No real conversion, layer exactness,
serialization, reload, inference, resource, or quality result exists. The
stronger Baya 2.0 System46 is a three-arm routed system, not a standalone
checkpoint. The Supa-specific abstention threshold is explicitly incompatible
as an unchanged Baya transfer: source evidence reports 40/64 dropping to 35/64.

## 5. Research intake audit

The full machine and human records are in `evidence/candidate_ledger.json` and
`CANDIDATE_LEDGER.md`. The research set was inspected at the tips below. Branch
names in the first column are the names as of inspection and are not resolvable
refs; three have since been retired. Resolve any row by its tip sha or archive
tag, never by branch name.

| Branch (at inspection) | Tip | Classification | Ref status |
|---|---|---|---|
| `research/positional-sequential-attention` | `ee13448f11efaa7f14206dab95b277816b912ce6` | `WAITING_FOR_EVIDENCE` | live on `origin` |
| `research/rineya-cross-scale-qwen4b-8b` | `538687fb8ddab999f825e300e7c91d43947c5c4e` | `NEGATIVE` | live on `origin` |
| `research/tgsa-phase1-v4-competence-first` | `8b5737e260b5478427b8bb033cffa394a5f3fc5d` | `NEGATIVE` regime | retired 2026-08-31; tag `archive/tgsa-phase1-v4-regime-rejected` |
| `research/tgsa-phase2-mtp-complementarity` | `81755bf216a383734d3dc599e0cb321a26d74f33` | `WAITING_FOR_EVIDENCE` | retired 2026-08-31; tag `archive/tgsa-phase2a-superseded-2026-08-31` |
| `research/tgsa-phase2a-ci-and-competence` | `d16ea1cdd62ce3c49a99b382b32e4dc9b53f3c4c` | `WAITING_FOR_EVIDENCE` | retired 2026-09-04; tag `archive/tgsa-phase2a-ci-and-competence-superseded-2026-09-03` |
| `research/tgsa-mtp-rineya-complementarity-v1` | `27b54322080be143f8d0b84d98daebec1b4281d8` | integration tooling only; prerequisite result absent | live on `origin` |

The three retired tips (`8b5737e`, `81755bf`, `d16ea1c`) stay reachable as
ancestors of the canonical programme branch
`research/tgsa-v5-1-unified-programme`, so retiring those branch names cost no
evidence.

Key exclusions:

- **TGSA:** v3 was inconclusive; v4's regime family was rejected; v5 is a draft
  competence path whose own claim ceiling says the programme has no mechanism
  evidence.
- **TGSA + MTP:** CI/integrity and rehearsal do not satisfy positive research
  evidence, and the v4 regime may not be revived or pooled into Phase 2.
- **PSA/SGSA:** measurement and attribution artifacts exist, but no intervention
  treatment has completed; model selection, preservation, sealed test, and final
  report are missing, with material audit findings still open.
- **Cross-scale spectral transfer:** the MERGEDEV +1 did not replicate on
  DEV-FULL, and the branch's frozen verdict is no confirmatory GO.
- **Significance/count cache policy:** a retained-state endpoint moved, but
  registered answer quality worsened and later production-relevant screens were
  null/negative or specificity-failed.
- **Supa-A1.1 FP16/abstention:** positive enough to be a published parent and
  already integrated; it is not new intake work. Its unchanged threshold is a
  negative Baya transfer.

## 6. Candidate selection and production results

**Best current retrofit candidate: none.**

Because `positive_source_evidence` is false for every unintegrated mechanism,
the following are intentionally `NOT_RUN` rather than missing favorable results:

| Required result | Status | Reason |
|---|---|---|
| Exact-parent before/after target evaluation | `NOT_RUN` | No eligible mechanism/configuration |
| General preservation and instruction following | `NOT_RUN` | Intake gate failed |
| Generation stability and long-context scaling | `NOT_RUN` | Intake gate failed |
| TTFT, throughput, latency, VRAM/RAM, load time | `NOT_RUN` | Intake gate failed |
| Enabled-runtime vs baked-artifact equivalence | `NOT_RUN` | No authorized bake and no artifact writer |
| Artifact provenance/hash | `NOT_RUN` | No candidate artifact |
| Demo smoke | `NOT_RUN` | No candidate artifact |
| Promotion | `DEFER` | Research evidence insufficient |

This is the required behavior of the absolute intake rule, not an incomplete
attempt to rescue a negative mechanism.

## 7. Demo and release assessment

The repository contains a useful local long-context runner and evaluator under
`Rineya_Long_Context_Demo_v1`; its runner supports local checkpoints,
`trust_remote_code`, explicit dtype, greedy decoding, no-silent-truncation, and
frozen prompt/key files. It can become the exact-parent quality/demo smoke once
a manifest-locked artifact exists.

The remote product branch
`product/rineya-family-cloud-demo-secure-delivery` contains a hardened stateless
gateway, local-model hash manifest, integrity verifier, TTFT/latency/throughput
benchmarking, and mock/transformers/llama.cpp engines. It deliberately loads
transformers models with `trust_remote_code=False`. That means it cannot
currently load Supa-A1.1's custom abstention class or a TGSA custom attention
artifact. A future release must either use a supported standard representation
with the portable abstention logit bias, or add and security-review the minimal
runtime extension. Wholesale merging the product branch is unnecessary.

The existing `_promote_supa_a1.py` and `_publish_family_frontier.py` scripts
demonstrate valuable release practices: pinned hashes, staged writes, collision
refusal, immutable versioned directories, verification, evidence copying, and
rollback. They are release-specific and must not be reused as if they were a
generic retrofit publisher.

## 8. Reproducibility commands

```powershell
git fetch --prune origin
git status --short --branch
git branch --all --verbose --no-abbrev

Set-Location Rineya_TGSA_Retrofit_v1
python -m pip install -e ".[dev]"
python .\scripts\verify_candidate_ledger.py
python -m pytest .\tests -q --tb=short --durations=15
```

Existing A1 evidence can be inspected without running a prohibited candidate:

```powershell
Get-Content -Raw .\evidence\source_checkpoint_manifest.json
Get-Content -Raw .\evidence\real_layer_exactness.json
Get-Content -Raw .\evidence\real_conversion_damage.json
```

When an eligible source result arrives, the next production sequence is:

1. add its exact source branch, commit, configuration, result artifact, and
   intended parent to the ledger;
2. verify the source result satisfies its own frozen positive gate;
3. obtain and hash the exact strongest compatible parent (Supa-A1.1 first);
4. implement only the missing artifact/runtime surface required by that selected
   configuration;
5. run neutral fidelity, runtime/baked equivalence, exact-parent quality,
   regression, systems, integrity, and exact-artifact demo gates;
6. set `PROMOTE` only if the validator sees every mandatory gate as true.
