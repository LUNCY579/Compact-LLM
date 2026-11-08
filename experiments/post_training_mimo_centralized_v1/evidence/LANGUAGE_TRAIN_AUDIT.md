# TRAIN-only language source audit

**SOURCE VERIFIED; CANDIDATE UNENROLLED.** Audit: 2026-09-28, code head `43d2ed3c438a64fdd6a4088db0a151f7e604d0ee`. This audit follows the parent-directed authorization for retention qualification and raw TRAIN reads. It reads the prior `RETENTION_REQUIREMENTS.json` and `RETENTION_DESIGN.md` as proposal context; it does not reinterpret their former authorization fields as current native authorization.

The v3 preservation file is usable as a **candidate source**, subject to the exclusions and remaining gates below. It does not contain 14,000 independent language parents. No 256-row replay bank was selected/exported, no model was called, and no protected CHECK/FINAL/test2/fresh_sealed scored content was read. Only this report and `receipts/LANGUAGE_SOURCE_AUDIT.json` were written.

## Verified source and original TRAIN membership

The actual `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v1_v2_preservation_slice_14000.jsonl` contains 14,000 rows, 17,341,591 bytes, SHA256 `5ee0c283b290cf89f6fcfbcf52bd2bcd03f6c63bb06793b9fa4b957d91973d50`. This matches its published local manifest. Every row is marked TRAIN; all IDs are unique; all rows have nonempty message contents and alternate system/user/assistant roles correctly, ending in the assistant target. All retain `human_review_status: pending`.

All rows were joined to the two explicit original TRAIN files by exact original ID. Removing only `metadata.preservation_replay_for` and `metadata.preservation_source_version` makes every joined record structurally equal to its original record, including messages, target and other metadata.

| Original TRAIN file | Original rows / bytes | Preserved exact matches | SHA256 |
| --- | ---: | ---: | --- |
| v1 `data/sft/rineya_short_response_new_train.jsonl` | 10,000 / 9,720,811 | 4,000; zero mismatches | `20d662836e9343d2024486ff89dece0b6a618f53a6c9b8449445aa0340a72dbb` |
| v2 `data/sft/rineya_v2_new_all_train.jsonl` | 32,000 / 39,659,277 | 10,000; zero mismatches | `dc68d9400419e2158144f8832ca52b005628f9b24862f95667e482e34998faee` |

Both original files are entirely TRAIN. The V3 generator's `build_preservation_slice` confirms the 4,000/10,000 stratified origin. This is provenance recovery, not proof of current evaluation decontamination or Titan training exposure.

**Confirmed metadata defect:** all 14,000 preservation rows say `preservation_source_version: v2.0`. The V3 generator at line 875 checks prefixes `train_` and `rineya_short`, while the actual 4,000 V1 IDs start `rineya_srb_v1_`. Recover source version through the verified original-file membership above; never use that faulty field. Historical data was left unchanged.

## Parent units and language coverage

V2 has 452 original `base_template_family` groups among its 10,000 preserved rows. The generator returns these scenario families; `template_family` appends a surface suffix later. Rank suffixes, wrapper variations, persona names and row IDs are not new semantic parents. For an admitted V2 pool, use `v2.0:<base_template_family>`, retain any counterfactual connection, and allow at most one admitted row per group. No counterfactual ID in the audited communication/integrated pool crosses base groups.

The 4,000 V1 rows have 1,299 distinct `template_family` labels but lack V2's `base_template_family`. Their byte provenance is verified; this audit has not recovered the V1 generating semantics. Do not admit them as additional independently recovered parents merely to fill a quota.

The proposed V2 language pool has 5,000 rows in 20 classes: 12 communication and eight integrated classes, each with 250 rows and 16 base families. It has 320 recovered groups and 982 exact target strings. This includes real conversational functions: empathy, repair, follow-up, clarification, disagreement, negotiation, closure, rewriting and practical recommendations, alongside short grounded/context answers. The receipt records per-class multi-turn and target-diversity counts plus 20 actual TRAIN examples, selected by lexicographically smallest ID per class solely for inspection.

These are **conservative training clusters**, not 320 statistically independent natural conversations. Similar decision problems appear across classes, and semantic equivalence across all classes has not been exhaustively resolved. None of this existing training material can become fresh DEV/CHECK/FINAL through relabeling. A later held-out analysis must use newly authored parent worlds and its own independence audit.

## Target-quality exclusions before selection

Inspection combined generator source tables with actual representative and suspicious TRAIN rows. This is a bounded source-quality audit, not exhaustive human approval. Preserve the pending human-review metadata. Some quarantines are conservative ambiguity judgments; they are identified separately from hard provenance defects.

| Family / predicate | Affected actual rows | Concern |
| --- | ---: | --- |
| `v2_rewrite_05`, all | 14 | Adds a handoff, step or location as the cause of a delay when the input states only that a person caused it. |
| `v2_rewrite_07`, all | 12 | Replaces ambiguous referents with current/previous process interpretations; conservative ambiguity quarantine. |
| `v2_rewrite_09`, all | 14 | Adds a frustrating/failed-attempt assertion to a bare try-again sentence; conservative quarantine. |
| `v2_rewrite_11`, all | 19 | Changes willingness to stay late into advance-notice conditions or refusal. |
| `v2_decision_reason_13`, all | 15 | Urgent low-impact task versus tomorrow's team blocker lacks an explicit tradeoff rule; recommendation is plausible but not uniquely supported. |
| `v2_plan_revision_00`, target contains `two` | 9 | Invents exactly two core/must-have features. The two other rows in this family remain candidates. |
| `v2_rewrite_12`, target differs from `Running about 10 minutes late.` | 4 | Changes ten minutes late to arriving in ten minutes or invents heading over now. Six source rows remain. |
| `v2_ack_update_08`, target `Okay, no call today.` | 5 | Adds a date absent from the cancellation statement. |
| `v2_texting_02`, target contains `we survived` | 5 | Implies the assistant attended the meeting. |
| `v2_context_update_00`, original deadline project differs from final-user wrapper project | 9 | All nine preserved rows name one project's deadline then wrap the question in another project's context. The target ignores conflicting task identity. |
| `v2_roleplay_reason_11`, all | 12 | Assumes a stronger implication from the gate's only-for-valid-code rule; conservative logical ambiguity quarantine. |
| `v2_roleplay_reason_10`, target contains `staff escort` | 3 | Invents a staff-escort exception to staff-only access. |

The machine receipt enumerates **every affected row ID**, the family and reproducible predicate. The full non-roleplay recommendation excludes all 250 roleplay rows for scope; the roleplay flags remain recorded evidence. No historical row was repaired or deleted. Source wrappers can introduce additional ambiguity; selected records still require a semantic review of the full stored conversation, not just its final sentence.

## Feasible proposed 256-row allocation

After exact V2 membership, communication/integrated scope, non-roleplay scope and all recorded exclusions, there are **4,644 rows in 298 base families across 19 classes**. This is a candidate population count, not a selected export.

The parent's proposed quotas are feasible at the source level:

- **16 rows/families each:** boundary setting, clarification request, conversation closure, de-escalation, empathy validation, informal texting and negotiation/compromise. Each retains all 16 groups: 112 rows total.
- **12 rows/families each:** the other 12 non-roleplay classes: acknowledgment/update, apology/repair, communication rewrite, comparative recommendation, decision explanation, evidence-grounded dialogue, multi-turn context update, natural follow-up, plan revision, practical next step, preference/opinion and tactful disagreement. These contribute 144 rows.

Communication rewrite retains exactly 12 groups; decision explanation and multi-turn context update retain 15 each; every other class retains 16. The allocation totals **256 rows from 256 recovered groups**, of which 24 are grounded/context short-answer rows and 232 cover other conversational functions. It is still not an independence claim.

Root should freeze the source receipt/hash, exact exclusion IDs, quotas, seed, namespace, canonical serialization, bounded candidate count and ranking before native behavior. A reproducible rule is SHA256 ordering of `[namespace, seed, class, source_version, base_template_family]`, then SHA256 ordering of `[namespace, seed, original_row_id]` inside each group, with lexical tie breaks. Freeze any bounded within-family alternatives at the same time. Neither family order nor target eligibility may be changed after native outcomes. This audit recommends the rule but does not freeze/select it.

Source-correct quota feasibility is **unmeasured**. Seven 16-of-16 classes and rewrite's 12-of-12 quota have no spare groups; a failed family cannot be replaced by another class or evaluation data without a disclosed amendment. A bounded predetermined alternative within the same family is possible only if frozen before model screening. Do not expand the pool adaptively until a desired success count appears.

## Remaining evidence boundaries

The receipt lists fields required for a frozen derived record: source and original-file hashes/line/ID, original-record hash, recovered family/version, counterfactual links, class, system policy, full messages/target, provenance/review state, normalized/tokenized prompt hashes, tokenizer/template identities, actual token lengths, quality/exclusion revisions, rank, rubric and training eligibility.

Current historical fingerprint exclusions remain outstanding. No packaged validator that recursively scans evaluation content was run. Historical local README claims concern older snapshots and cannot establish present-day decontamination. Use authorized ID/hash inventories, record their scope, and disclose unresolvable semantic-overlap limits without opening protected banks.

Native token fit, exact-template handling, truncation rejection, source-correct semantic enrollment and all model behavior remain unmeasured here. Exact response matching is insufficient for empathy, advice, rewrite and other open-ended classes; the fixed rubric must check adequacy, relevance, fluency and role consistency. Fresh language DEV/CHECK and long-context preservation are separate unqualified obligations. This audit supports no retention or efficacy claim.

V1's manifest labels the source a project-internal research artifact and requires third-party/source-model obligation review before redistribution. These local provenance facts do not imply public license clearance. Hashes for the consulted manifests, documentation, generators and prior retention design are preserved in the companion receipt.
