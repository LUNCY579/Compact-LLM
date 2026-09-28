# RRP-001 source audit — 2026-09-15 UTC

## Authority and target

The current [goal](GOAL.md) targets the approximately **0.6B Rineya model** through verifiable post-training, measured preservation and bounded iteration. It supersedes the previous memo's suggestion to prioritize a larger standalone model as a possible target. Larger models may be separately authorized controls/teachers; they cannot replace this target on the scoreboard. No spending or model-execution budget is granted merely by goal mode.

The source branch was freshly checked at `codex/model-quality-audit-v2@59df72c2e24e6c4d3c7e4e86c1dd4ae143054400`. The new dedicated worktree is `C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1`, branch `research/rineya-reasoning-posttraining-v1`. Its first implementation commit is `58bf64cf93ff918ede4b1079b8a5716ee0e1e312`. Existing source branches and the original worktree's untracked Forge v2 directory were not modified. The newly observed paired-looped-flow host branch at `627253edde9a0e67334034d252e5d9dd9daead9a` contains fixture evidence, not a new trained target; it does not displace the ordinary lane.

The supplied September 8 PDF was read, including the conventional trainability, preservation, measurement and agent-handoff sections; pages11,21,23,24 were rendered for inspection. Source file: `C:/Users/clclc/Downloads/Rineya_Research_Plan_2026-09-08 (1).pdf`, SHA256 `faccd9ada0927822603a4fd6bece4671841f618d7dcc0805e4469bede8e824c8`. The plan rejects universal capacity claims and treats its example thresholds/budgets as prospective suggestions. It does not authorize a model run.

## Immutable inputs verified locally

| Input | Verified SHA256 | Fresh scope |
|---|---|---|
| Forge v1 archive |`3d490c386bf5d0d832a860403253aeff29266ade84f93c384a0a3645cff0eaac`|46 extracted files match archive bytes; local filename omits `(1)` but content identity matches supplied hash|
| Forge RL v2 archive |`d3bcd3d84c2d123c9876d334e867b0387f78cb7e93c7b625eafddbb0c685b663`|96 extracted files match archive;95 entries in its release manifest verified|
| Ungated Supa weights |`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`|File hash and bounded safetensors header inspected; no model loaded|

The Supa header contains **596,049,920 serialized parameters**, BF16 storage; blocks8–27 contain **314,618,880 parameters in220 tensors**. A fresh instantiated native recount, tied-storage check and actual runtime qualification remain required. No Opallo continuation or gated FP16 Supa-A1.1 substitution is made.

Original archives remain untouched in Downloads. Extracted sources remain in the historical worktree and are read only. The new importer verifies archive/tree bytes and rejects unlisted Python sources before importing solver functions. V1's `validate_release` rewrites its original report, so it was deliberately not invoked; its read-only `validate_row`/text solvers were used instead. V2's source tests use temporary fixtures. No remote model code, network teacher, model download or workflow dispatch was enabled.

## Historical state reconciled

- MQ's native/training records are real historical records, not merely a synthetic implementation boundary. The later six executions vary three ordering seeds over the same238 rows at1e-5; four objective/LR combinations remain unrun. Their missing watchdog/checkpoint evidence is not repaired retroactively.
- The 41 original Opallo training rows were recovered in their exact recorded order from the hash-matching10,000-row `splits_large` pool. See [recovery record](evidence/inspection-20260915/OPALLO_DATA_RECOVERY.json). Reusing those rows, replaying the old recipe from Supa and continuing Opallo weights remain distinct interventions. None occurred here.
- `rag_reasoning_v3_large` is a different historical5,900-case corpus, with4,410 TRAIN,752 DEV and738 published test cases. Its “reviewed” filenames contain auto-verified gold, not new human approval. It remains a candidate preservation/data component pending qualification.
- V3 general communication/reasoning, short-response, puzzle and historical Forge banks remain candidate sources. Their earlier provenance and contamination limits in the parent audit remain applicable. `splits_onpanel` is excluded from this clean input path; no quarantine is lifted.
- Published Opallo's task/retention and exact-UNKNOWN corrections remain as documented in the parent audit. This preparation adds no model-quality score and does not infer a capacity ceiling.

## Fresh data evidence

The actual Supa tokenizer was run with Transformers4.57.1 and its exact local assets in a process with Torch/TF/Flax disabled. This is **tokenizer/serialization qualification only**. Across768 unique TRAIN worlds, the preparation measured512 v1 direct views and256 v2 direct/reasoning view pairs. Repeated attempts do not add independent worlds.

The v1 text-side solver freshly checked all512 pilot targets. V2's separate oracle freshly checked256 pilot worlds and complete structured reference traces; this is not a rerun of the entire20,000-world semantic validation. The original release's20,000-world and198,770-transition results remain source-reported. Its222 unit/integration tests were freshly rerun and passed.

At a2,048-token context with128 completion reserve, the original v2 pilot has **zero eligible answerable evidence-chain cases**. Removing only publicly declared other-scope records, while proving the entire canonical target world unchanged, still does not supply the required cases. Neither failure was bypassed by dropping target evidence, shrinking output reserve or silently increasing context. Both failed attempts and token surveys are preserved.

For the preceding engineering microfit, a separate explicit policy selects32 original v1 TRAIN primitives: four each from deduction, relational joins, document reconciliation, signed ledgers, program traces, dialogue state, source boundaries and story state. Full original prompts are retained. They occupy496–851 native full-sequence tokens, total20,907 per exposure and148 supervised response/EOS tokens. There are six exact full `UNKNOWN` targets; task-specific compound missing-field outputs are not relabeled as whole-task UNKNOWN.

Selected32 versus the declared v1 holdouts and historical RAB/MQ DEV instruments have zero overlaps on the listed ID/case/exact/normalized/digit-blinded keys. This does not establish semantic or skeleton disjointness. Forge splits share grammar; distributed test labels are developer-visible. Independent confirmation is **not established**.

## Review and permission boundaries

[REVIEW_PACKET.md](REVIEW_PACKET.md) contains every selected microfit problem. The85-world JSONL packet adds v2 family/status/layout cases; it is not85 selected training worlds. Human review has **not** occurred. Empty review fields do not constitute approval, and source CC0 intent is not a new human rights/label certification.

Only an explicit engineering-only authorization could open this microfit before broader human review. Broad mixed SFT, reasoning acquisition, RL, long-context escalation and confirmation still require their own qualified inputs and budgets. The pending [budget request](BUDGET_REQUEST.json) does not grant anything by existing. The native controller remains to be implemented and qualified; no current command launches the registered native experiment.
