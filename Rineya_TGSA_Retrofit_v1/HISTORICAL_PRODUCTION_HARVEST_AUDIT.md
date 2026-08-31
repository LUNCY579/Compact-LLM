# Historical Rineya production harvest audit — second pass

## Registration state

**Historical scan complete enough to freeze the first bounded trial. Outcome
generation has not started.**

This document supplements, and does not replace,
`PRODUCTION_RETROFIT_AUDIT.md`. The first audit correctly described active
branch evidence. This second pass expands `positive_source_evidence` for trial
eligibility to credible positive signals anywhere in reachable program history.
The strict ten-gate shipping predicate remains unchanged.

Audit base: `research/tgsa-rineya-pretrained-retrofit` at `3d481b959ff4bb35150a42836ee33f6709f20598`

Historical routing index: `origin/main` program audit at
`a6ee824584a9e28881826ab92a11db48e124d90c`, verified against the cited source
reports and manifests rather than accepted as unsupported summary.

## Historical program scan

The scan covered:

- every local and remote ref returned by `git for-each-ref`;
- nine archive/live tags, including frozen TGSA and cross-scale results;
- `git log --all`, merged and retired branch commits;
- foundational Phase 1–9 stop gates and release audit;
- teacher, global, cross-depth, and layerwise soup reports/manifests;
- published-core and system manifests;
- long-context adapter-dose/task-vector evidence;
- abstention/readout evidence;
- significance/KV, adapter-consensus, PSA, TGSA, PCSM, cross-scale, and cloud-demo records;
- the dated historical program audit added to `main` at `a6ee824`.

Confirmed or operationally credible positive mechanisms found:

| Mechanism | Type | Positive evidence | Current standing |
|---|---|---|---|
| Dense short-response SFT | CHECKPOINT | 0.804→0.995 heuristic score, shorter outputs, no category regression, human PASS | Old-lineage positive; not a cheap A1.1 transfer |
| Progressive structural reduction | CHECKPOINT | 19-layer efficiency frontier with positive proportional value | Specialized efficiency profile; later extra compression negative |
| Targeted recovery | CHECKPOINT | Recovered reduced frontier with ~94.7% trainable reduction for top1 option | Exact-parent-specific old artifact |
| DPO | CHECKPOINT | +0.0048 mean quality, unchanged worst decile, human PASS | Small old-lineage result |
| Uniform reconstructed-delta soup | CHECKPOINT | seed_37/seed_49 soup improved mean quality, no tail/hard/critical regression | **Selected first modern trial** |
| Context compaction | RUNTIME | 29.10% fewer memory tokens, complete measured recall, no reported quality loss | Queued efficiency profile |
| Cross-depth/global/layerwise soup | CHECKPOINT | Restored 28-layer family cores; α=0.40 recipe formed Nimbus | Already integrated in A1.1 ancestry |
| Long-context adapter dose | CHECKPOINT | Nimbus 113/256→Supa 161/256; complementary high dose formed Baya/system gates | Queued second modern trial |
| Pulse negative task vector | CHECKPOINT | Pulse 31/64→41/64, 117/192 | Queued third; anchor-dependence risk |
| FP16 storage + abstention | CHECKPOINT / PRODUCT_POLICY | A1→A1.1 +2 Showcase with equal Stability; stable +4 S4 policy | Already integrated parent behavior |
| Answer-blind cascades | SYSTEM | System48 reached 176/256 versus Nimbus 113/256 | Already integrated system frontier |
| Adapter consensus | SYSTEM | Local sealed improvement over incapable backbone | Excluded: not best expert, transfer negative, high latency |
| Structural pooling/sinks under KV compression | RUNTIME | Narrow structural retention effect | Not a checkpoint-quality win; significance-specific claim failed |

## Exclusions retained from the first audit

- TGSA v3 inconclusive, v4 regime rejected, v5/Phase2 not positive yet.
- PSA/SGSA has measurement but no completed treatment result.
- Cross-scale 4B/8B transfer tied DEV-FULL and remained near-null in v2.
- Normalized significance/count KV retention did not improve registered answers.
- The unchanged Supa abstention threshold harms Baya.
- Sparse-runtime measurement was negative and selected the dense fallback.
- Phase-9 compact student/RAG production candidacy is blocked, including
  recorded zero procedure correctness.

These remain excluded from bounded quality-first checkpoint transfer unless new
evidence changes their status.

## Selected first candidate

**Uniform A1.1/Baya midpoint soup.**

Why it ranks first:

1. The mechanism has an actual historical stop-gate and human PASS.
2. The modern endpoints are not arbitrary siblings: both are Nimbus 1.6 plus
   the same checkpoint-200 rank-16 adapter at two recorded doses.
3. Exact artifacts are present in the sibling canonical ChatBubble checkout.
4. Weight hashes match the published manifests:
   - A1.1: `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`
   - Baya: `1737cd1ee2bdde714c259adf2012ae44c89ad82a06797653343bad7e3cf88bba`
5. Both expose 310 identical tensor keys with identical shapes and tokenizer
   SHA-256 `aeb13307…`.
6. The output is one ordinary FP16 Hugging Face checkpoint with A1.1's existing
   custom/portable abstention policy; rollback is the exact parent.
7. The one-candidate 0.5 coefficient is fixed by the historical uniform-soup
   precedent. No search is needed or permitted.

## Exact parent and transfer gate

Parent: `Rineya-Supa-A1.1-1.1.0-devfreeze`.

The complete outcome-blind contract is
`configs/historical_soup_a11_baya_midpoint_trial.json`. It pins parent and
ingredient weight/config/tokenizer hashes, recipe, runtime assets, runner and
evaluator hashes, prompts/keys, inference protocol, and an AND transfer gate.

The trial is `TRANSFER_POSITIVE` only if the single candidate gains at least one
all-256 strict-exact hit, loses no Stability hit or all-four pack, has complete
generation coverage, reloads fresh as the required custom class, and has full
artifact/evaluation provenance. A tie is failure.

### Outcome-blind compute amendment

The original runner reached only 21/256 parent rows while materializing
full-vocabulary logits for every prompt token. The partial JSONL was archived
unscored; its content and answer keys were not inspected. A first amendment
attempt failed before producing a prediction because the custom wrapper's
generic signature hid Qwen3's supported `logits_to_keep` parameter from
GenerationMixin. The final outcome-blind path loads the identical local weights
as plain Qwen3, requests the last prompt logits only, and applies the published
portable S4 offset through a constant logits processor. The amended runner hash
is `cdf156ef…` and must reproduce identical raw predictions and token counts on
the first four archived parent fixtures before use. All model weights, prompts,
seed, decoding, attention, cache, dtype, threshold, and evaluator settings
remain unchanged.

## Outcome

`PENDING — registration must be committed before merge or inference.`

If the transfer fails, the queue moves to the single historically measured
adapter dose 1.0625, then the Pulse negative-vector trial, without coefficient
rescue or evaluation changes. If all reasonable checkpoint candidates fail,
the runtime-only context-compaction profile is assessed separately rather than
misrepresented as a new model.
