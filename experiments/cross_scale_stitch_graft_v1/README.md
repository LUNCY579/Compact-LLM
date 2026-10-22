# XSG-001 — Cross-scale stitch and FFN-graft framework

**Implemented and CPU-tested offline mechanics. No pretrained experiment, native qualification, or quality improvement is claimed.**

Repository: `LUNCY579/ChatBubble`  
New research branch: `research/cross-scale-stitch-graft-framework-v1`  
Immutable parent: `research/rineya-cross-scale-qwen4b-8b@fde553c908db39e87a15a52ed239d1ba0a68bfd3`

This additive package preserves three independently interpreted experiments:

1. **Donor headroom:** paired, task-level donor/receiver comparison before transfer.
2. **Small-model FFN graft:** keep the entire host; add one frozen donor FFN through two fitted bridges. An ordinary no-donor adapter provides an exactly trainable-parameter-matched control.
3. **Bidirectional stitching:** large prefix → bridge → small suffix, and the reverse, initially at one midpoint boundary.

A fourth, separately authorized stage may train only interface parameters. The code supplies the loss, explicit optimizer allowlist and frozen-component behavior; its only delivered optimizer run is a four-update random-fixture demonstration. It does not provide a native training launcher.

## What is executable now

| Component | Delivered implementation |
|---|---|
| Identities and activation contracts | Model/runtime/template/precision/logit-policy digests; exact token-position pairing; group/example/prompt split isolation |
| Bridge fitting | CPU FP64 centered affine ridge, streaming sufficient statistics, regularization, held-out MAP_CHECK metrics, capture digests |
| Graft | Intact host plus intact frozen donor normalization/FFN; affine input map, bias-free residual return map; explicit bypass |
| Stitch | Retained prefix embedding/blocks and suffix blocks/norm/head; no retained owning donor model; raw boundary conventions |
| Ordinary control | Frozen host normalization plus two-layer SiLU adapter with the same bridge-parameter budget |
| Scoring | Exact complete paired panels, discordant counts, family deltas, cluster bootstrap, format and literal-UNKNOWN regressions |
| Artifacts | New-directory-only, SHA-256-bound safetensors and strict JSON; no pickle; capture/bridge round trips |
| Resource accounting | Analytic model inventories; bounded ridge workspace; CLI payload and host/cgroup-memory guard |
| Tests and demonstration | Source-bound no-skip CPU suite; actual tiny causal attention, GQA and RoPE fixtures; both stitch directions and interface gradients |
| Native adapter candidate | An explicit adapter for **already loaded, frozen stock Qwen3 under Transformers 4.51.3**. Unqualified and not executed here. |

**Historical host identity remains Supa-A1.1.** The parent audit records an FP16 custom `Qwen3AbstainForCausalLM` with a non-weight logit adjustment. It is not interchangeable with Supa Core 1.7 or stock Qwen3. The optional stock-Qwen adapter refuses that custom class rather than silently discarding its behavior. Completing native A1.1 integration is an explicit next gate, not a hidden substitution.

## Run from repository root

Use an existing CPU environment containing the scoped dependencies in `requirements-mechanics.txt`. Commands never download models, inspect sealed banks, start GitHub Actions or write into inherited source directories.

```bash
python -m experiments.cross_scale_stitch_graft_v1 plan
python -m experiments.cross_scale_stitch_graft_v1 verify-source
python -m experiments.cross_scale_stitch_graft_v1 qualify --output ../xsg001-qualification-new
python -m experiments.cross_scale_stitch_graft_v1 demo --output ../xsg001-demo-new
```

Output directories must not exist and must be outside this package. The supervisor has a 120-second timeout and refuses missing, altered, failing or skipped tests. Its status is `PASS_MECHANICS_ONLY`, never a pretrained GO.

The demo creates tiny activation artifacts and a portable fitting request. To exercise the same offline fitting CLI on those artifacts:

```bash
python -m experiments.cross_scale_stitch_graft_v1 fit --request ../xsg001-demo-new/fit_request.json --output ../xsg001-refit-new
```

## Read in this order

- `docs/DESIGN.md`: equations, sites, directions, controls and mathematical limitations.
- `docs/EXPERIMENT_PLAN.md`: small staged studies, proposed budgets, selection/reporting boundaries.
- `docs/NATIVE_CONTRACT.md`: exact missing native prerequisites and supported adapter boundary.
- `docs/RUNBOOK.md`: commands, schemas, artifact rules and failure interpretation.
- `docs/SOURCES_AND_DECISIONS.md`: inspected source locations, corrections to tempting assumptions and publication scope.

`protocol.json` is a concrete **design contract**, not a completed native preregistration. Its unresolved native pins are intentionally null. `SOURCE_MANIFEST.json` freezes every scoped source/document/test file and the exact test IDs; evidence is excluded to avoid self-referential hashes. See `evidence/LOCAL_VERIFICATION.json` for the actual delivered result.

## Important limits

No full-model recovery, weight merging, LoRA, CED composition, scheduler sweep, donor download, production export, HF `generate()`, or KV-cache implementation is included. The runners support only unpadded batch-one full-prefix execution. Passing a cache is an error. A future bounded uncached generation collector could use these forwards, but is not supplied as a qualified native runner.

Do not mistake a good reconstruction metric for transferred capability, an exact zero/bypass result for useful donor behavior, or a small trainable bridge for a small deployed model. `planning.py` makes the latter distinction explicit: the 4B midpoint stitches have about 2.584B parameters, while the analytic one-FFN graft has about 676M including its host.
