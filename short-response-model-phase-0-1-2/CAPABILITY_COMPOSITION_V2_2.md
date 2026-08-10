# v2.2 — Multi-Domain Capability Composition

## Why v2.2 exists

The v2.1.x sequence produced a useful pattern rather than a single clean winner:

1. query-aware retrieval recovered the historical 0/150 Real-RAG procedure collapse without changing weights;
2. narrow procedural fine-tuning could improve hard collision cases but damaged stronger existing behavior;
3. preservation-first procedural training retained the original procedural path but only moved the harder challenge slightly;
4. broad reasoning training produced a much larger reasoning gain, suggesting that the small student still has meaningful learnable capacity, but also raised the risk of reallocating capacity away from specialist behaviors.

The design source of truth for this release therefore reframes the research question:

> Can specialized capabilities coexist when they are trained as independent deltas from one frozen student and composed deliberately, rather than sequentially overwriting the same adapter?

v2.2 keeps the original student as an immutable anchor and implements four governed experiment tracks.

## Source-of-truth lineage

The repository was built directly from `Output(20260810-063134).zip`. The attached chat-history PDF is preserved under `research_sources/v2.2/` together with SHA-256 provenance in `SOURCE_TRUTH_MANIFEST.json`.

## Immediate correctness fix from v2.1.2

v2.1.2 selection could report zero preservation cost when original Phase 9 protected metrics were missing/null. v2.2 forbids that interpretation.

Official selection now requires non-null Real-RAG values for:

- procedure correctness;
- supported-answer correctness;
- evidence faithfulness;
- correct abstention;
- conflict-safe handling;
- social usability.

A missing protected value is **technical invalidity**, not zero loss.

## Dataset redesign

`data/phase_9_student_distillation/multidomain_reasoning_v22/`

- **40,500** deterministic canonical cases;
- **27 domains**;
- **15 transferable reasoning archetypes**;
- TRAIN / DEV / TEST = **32,400 / 4,050 / 4,050**;
- prompt-contained evidence only;
- no chain-of-thought target;
- concise final answers only;
- high-stakes domains use hypothetical rules and conservative uncertainty rather than external professional facts.

Domains:

`general, research, writing, communication, translation, coding, devops, systems, math, data_analysis, science, engineering, education, business, operations, finance, legal, health, consumer, travel, home, creative, media, personal, coaching, tools, agents`

Reasoning archetypes:

`compare_metric, filter_then_rank, aggregate_delta, capacity_fit, state_transition, rule_exception, evidence_binding, insufficient_information, conflicting_evidence, dependency_order, two_hop_relation, constraint_preserving_transform, constraint_satisfaction, evidence_synthesis, diagnostic_priority`

The corpus trains reusable operations, not encyclopedic memorization. In addition, the social specialist receives a separate **3,000-row TRAIN-only conversational rehearsal pool** (acknowledgment, celebration, clarification, brief support, and preference responses) so that the social adapter is not merely another logic adapter with social-domain vocabulary.

## Track A — broad multi-domain LoRA

A fresh rank-16 LoRA is trained from the frozen anchor on a 15,000-row balanced broad view.

Default:

- LR: `1e-5`
- optimizer steps: `320`
- no sequential specialist adapter

Purpose: reproduce and extend the v2.1.2 broad-reasoning effect on a much wider domain distribution.

## Track B — behaviorally anchored broad LoRA

Track B uses a **15,000-row anchored mixed view** rather than the pure Track A view:

- 12,000 v2.2 broad multi-domain reasoning rows;
- 2,000 original Phase 9 TRAIN-only preservation rows;
- 1,000 v2.1.1 procedural TRAIN-only preservation rows.

It then adds a frozen-anchor KL term:

`L_total = L_supervised + lambda * KL(anchor || student)`

Default:

- LR: `1e-5`
- steps: `320`
- rank: `16`
- anchor KL weight: `0.20`
- anchor temperature: `1.5`

This is not teacher distillation. The original frozen student is the behavioral reference, the replay component supplies explicit supervised preservation targets, and the KL term penalizes unnecessary movement while the broad labels teach new reasoning operations. This implements the intended `general + specialist rehearsal + behavioral anchor` experiment rather than KL alone.

`train_student.py` now supports:

- `--anchor-model`
- `--anchor-kl-weight`
- `--anchor-temperature`
- `--anchor-device auto|cpu`

The tokenizer must match exactly because the loss is computed directly over logits. The PowerShell runner exposes `-AnchorDevice auto|cpu`; `auto` is faster when VRAM permits, while `cpu` trades substantial speed for lower GPU memory pressure.

## Track C — parallel specialist LoRAs

Five independent adapters are trained from the **same frozen anchor**, never from one another:

- `general`
- `grounding`
- `procedure`
- `abstention`
- `social`

Train-only specialist views combine new v2.2 operation data with reviewed Phase 9/v2.1.1 rehearsal where relevant.

Default sizes:

- general: 10,000
- grounding: 5,500
- procedure: 5,500
- abstention: 4,500
- social: 4,500

Default specialist training:

- LR: `8e-6`
- steps: `220`
- rank: `16`

This gives clean causal deltas:

`Delta_general, Delta_grounding, Delta_procedure, Delta_abstention, Delta_social`

## Track D — capability composition

### D1. Exact weighted static composition

`compose_lora_adapters_v22.py` materializes:

`W = W_anchor + sum_i alpha_i * Delta_i`

It scales each adapter's LoRA-B tensors by the requested coefficient and sequentially merges the exact deltas into the frozen base. The result is one dense checkpoint with **no runtime adapter overhead**.

Three pre-registered presets are evaluated:

- `balanced`
- `grounding_guarded`
- `reasoning_forward`

The weights are stored in `configs/phase_9_capability_composition_v22.json`.

### D2. Layer-separated specialists

`train_student.py` now supports `--lora-layer-indices`.

`derive_layer_plan_v22.py` selects a central band and splits it into five non-overlapping contiguous groups. This uses the project's middle-layer literature only as a broad search prior. It is explicitly **not** a claim that a named skill is already known to live in a particular layer.

Each capability adapter is trained only in its assigned layer group, and the five non-overlapping deltas are then merged into one checkpoint.

This directly tests whether parameter-space separation reduces cross-skill interference.

### D3. Greedy adapter soup

`run_greedy_adapter_soup_v22.py` performs a real greedy search rather than choosing ingredients from single-adapter scores alone.

At each round:

1. compose each remaining adapter at each configured candidate weight;
2. evaluate the materialized mixture on bounded DEV probes;
3. compute the official v2.2 opportunity margin against the frozen anchor;
4. accept the best addition only if it improves margin by the configured minimum;
5. repeat for at most the configured number of rounds;
6. recompose the accepted ingredients and run full DEV evaluation.

TEST is never touched by greedy search.

## Unified capability matrix

Every candidate is evaluated on the same four axes:

1. v2.2 27-domain direct reasoning;
2. v2.1.2 general-reasoning continuity;
3. original Phase 9 Real-RAG under `query_aware_v21`;
4. v2.1.1 procedural challenge under `query_aware_v21`.

The final report is `CAPABILITY_COMPOSITION_V22_SELECTION.md`.

Candidates include:

- frozen anchor;
- Track A broad adapter;
- Track B anchored adapter;
- five Track C specialists;
- weighted Track D compositions;
- layer-separated composition;
- greedy soup.

Selection reports a Pareto frontier and two policies (`balanced`, `preservation_guarded`). If they disagree, sealed TEST is blocked for human review.

## Sealed TEST discipline

The first run is DEV only.

`-RunSealedTest` may be used only after one DEV winner is frozen. The selected candidate is evaluated once on:

- original Phase 9 TEST with frozen original DEV calibration;
- v2.1.1 procedural TEST with frozen v2.1.1 DEV calibration;
- v2.1.2 direct TEST;
- v2.2 multi-domain TEST.

No TEST-based training, recalibration, merging, or candidate switching is allowed.

## Main command

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_capability_composition_v22.ps1 `
  -StudentModel "C:\path\to\student_base_or_checkpoint" `
  -StudentAdapter "C:\path\to\original_selected_adapter" `
  -RequireCuda
```

If the input student is already a complete checkpoint, omit `-StudentAdapter`.

The full default run is intentionally expensive because it trains and evaluates multiple independent research tracks. To run a controlled subset while preserving the same data/evaluation contract, use:

- `-SkipWeightedComposition`
- `-SkipLayerSeparated`
- `-SkipGreedy`

Do not use `-RunSealedTest` on the first execution.

## Research interpretation

A single scalar score is not expected to describe the whole experiment. The central evidence is the capability matrix and Pareto frontier.

Possible outcomes are all informative:

- **Track B beats Track A:** KL anchoring preserved old capabilities while retaining much of the broad gain.
- **Specialists beat broad adapters on their axes:** independent capability deltas are meaningful.
- **Weighted/greedy composition dominates specialists:** capabilities can coexist in one static checkpoint.
- **Layer-separated composition dominates overlapping adapters:** interference is partly parameter-local.
- **Frozen anchor remains Pareto-optimal:** the small model may have limited simultaneous capacity, even if individual adapters can specialize strongly.

No outcome is automatically a production release decision.
