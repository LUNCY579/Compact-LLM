# TRAIN preference-bank audit — 2026-09-17 UTC

**53,000 pairs exist, but they are not 53,000 ready-to-train reasoning examples.** Full-file schema, identity, duplicate and text-contact checks covered all pairs. Tokenizer-only measurement covered all 21,000 v3/behavior pairs and a time-capped Forge prefix. No model, GPU, protected bank or preference-training implementation was used. Main audit elapsed 237 seconds; the short supplementary filter recount took about two seconds.

| Bank | Pairs / distinct prompts | Contact with open arithmetic panel | After applicable exclusions |
|---|---:|---:|---:|
| v3 preferences | 16,000 / 16,000 | 9,343 pairs: 9,197 user eight-word hits, 888 blinded-stem hits, 29 source-ID hits; reasons overlap | **6,579** after both text/ID and panel-class guards: 4,312 single-turn, 2,267 multi-turn |
| Short-response preferences | 5,000 / 5,000 | 22 blinded percentage-stem hits | **4,978**: 4,231 single-turn, 747 multi-turn |
| Forge RL v2 preferences | 32,000 / 16,000 | Zero panel contact after joining canonical TRAIN prompts | **30,464** after excluding 1,536 pairs whose parents belong to either process instrument; window/quality admission remains separate |

These are conservative **text-contact** exclusions. Shared user wrappers can trigger eight-word matches; not every contact is an identical task. Conversely, passing these guards does not prove semantic independence. Counts are within each bank; no cross-bank semantic-deduplication certificate is claimed.

All 53,000 rows have nonempty chosen/rejected strings, valid alternating parsed prompt roles, TRAIN declarations, zero duplicate pair IDs, zero identical full prompt/chosen/rejected triples, and zero chosen==rejected matches both exactly and after stripping whitespace. No target contains the checked `<|` control-delimiter prefix. Forge's 16,000 repeated-prompt occurrences are intentional: each world supplies two different rejection categories. Those pairs are correlated and must stay with their parent in splitting and analysis.

## Corrections to the earlier preference survey

**Forge pairs contain no prompt.** Their schema is `{id,parent_id,group_id,source_split,category,chosen,rejected,synthetic_preference}`. A verified parent join to `data/rl/train.jsonl.gz` supplies the prompt. This audit joined all 32,000 pairs to 16,000 TRAIN parents, two pairs each. The bank contains 16,000 `wrong_concise` and 16,000 `unnecessary_prose` pairs: half explicitly target output style, not a different correct answer.

**Equal-length pairs were previously mistaken for shorter pairs.** Actual character-length directions are:

| Bank | Chosen longer | Equal length | Chosen shorter |
|---|---:|---:|---:|
| v3 | 7,882 (49.3%) | 3,671 (22.9%) | 4,447 (27.8%) |
| Short-response | 1,169 (23.4%) | 450 (9.0%) | 3,381 (67.6%) |
| Forge | 2,475 (7.7%) | 11,019 (34.4%) | 18,506 (57.8%) |

Thus “Forge chosen is shorter in 92%” is wrong. Likewise, v3's median length difference of zero does not establish length-unbiased or content-valid preferences. The short-response bank has **25 preference dimensions**, not one: the old census missed `metadata.preference_dimension`. V3 has 40 reply classes; neither class count establishes broad capability coverage.

## Serialization and usable windows

All 16,000 v3 and 5,000 behavior pairs fit both rendered branches within 2048 tokens, with a 128-token prompt reserve. Maximum branch lengths are 259 and 208 respectively. **All 21,000 stored prompt strings differ from the target tokenizer's thinking-disabled generation prefix.** Their Qwen strings end at the assistant header; the native direct contract includes the empty thinking stub. Parse and re-render a versioned derivative with provenance rather than appending completions to the stored bytes and assuming equivalence. Existing SFT serialization checks do not by themselves qualify a DPO loss/reference-policy path or its multi-turn masks.

Forge tokenization stopped at the prospectively bounded time limit: **21,380/32,000 pairs measured; 8,478 fit both branches and reserve**. Measured maximum is 14,980 tokens. This was a **family-ordered prefix**, not a random sample; no whole-bank fit percentage or representative family coverage is inferred. The 8,478 count precedes process-parent exclusion and is not an admitted training count. Complete its window census before choosing any Forge preference recipe.

## What is worth pursuing

The v3 **4,312 guarded single-turn pairs** are the simplest broad candidate for a later preference pilot. Preserve the 2,267 clean multi-turn pairs for a separately qualified serialization path. Behavior's 4,231 guarded single-turn pairs are another candidate, with substantial style/length objectives that should be reported explicitly. These are candidates for review, not certified high-quality preferences.

For reasoning-specific work, Forge's `wrong_concise` category is easier to verify mechanically than subjective style pairs: rederive the correct answer and prove the rejected answer fails the task's actual scorer. Do not treat `unnecessary_prose` as incorrect reasoning. Preserve parent identities, exclude both process instruments, finish window measurement and avoid shortening away evidence.

Before any preference experiment, audit semantic direction by task: correctness, grounding, missing evidence, instruction compliance and style are different objectives. Verify solver-checkable pairs mechanically; retain an explicit review-needed status elsewhere, especially for safety or subjective communication judgments. Separate length strata and inspect equivalent-answer negatives. Qualify chosen/rejected masks, EOS, reference scores, numerical behavior and actual 8-GiB memory fit before allocating GPU training. No new training path or experiment was started by this audit.

Evidence: [PREFERENCE_AUDIT.json](PREFERENCE_AUDIT.json) contains source hashes, complete counts, role distributions, contact reasons, representative affected IDs and the tokenizer-census limitation. Source files remain unchanged; human semantic review was not performed.
