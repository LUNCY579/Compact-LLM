# Candidate Discovery

## Audit scope

The official read-only repository contains 36 dense model directories and 186 adapter directories. Inventory is machine-readable in `candidate_inventory.json`. Four direct-vote compatibility groups were found. The two excluded v22 mini-labs were neither scanned for evidence nor used.

## Required anchors

| Candidate | Exact official artifact | Lineage | Parameters | Official evidence | Experimental role |
|---|---|---|---:|---|---|
| Frozen anchor | `artifacts/phase_9_capability_composition_v22_lr1e4_s1200/frozen_anchor` | 344M warm initialization plus the selected warm SFT adapter, materialized before v2.2 composition | 344,354,816 bf16 | v2.2 multidomain 0.000; original procedure 0.9533, supported 0.6264, faithfulness 0.9732, abstention 0.8633 | Shared primary backbone and base control |
| D_greedy / greedy_soup | `.../track_d_greedy_soup/final_model` | Frozen anchor plus `0.5 ×` Track C abstention LoRA delta, materialized dense | 344,354,816 bf16 | v2.2 multidomain 0.1242, hard 0.1247, abstention 1.000; selected by both official policies | Required reproduced dense comparator; not interchangeable with frozen-anchor LoRAs |

Both use Qwen3ForCausalLM, 12 layers, hidden size 1024, intermediate size 3072, vocabulary 151,936, and tokenizer SHA256 `be7560…5506`. The lab exactly reproduced 8/8 stored outputs, token counts, and correctness for both anchors.

## Primary compatible committee

All seven experts below declare the exact frozen anchor, share the tokenizer and seven target-module families (`q/k/v/o_proj`, `gate/up/down_proj`), and contain 4,325,376 float32 LoRA parameters at rank 16.

| Expert | Official multidomain DEV | Key strength | Key weakness |
|---|---:|---|---|
| Track A broad | 0.2588 | Largest direct multidomain gain | Severe procedure/grounding preservation loss |
| Track B anchored | 0.2365 | Reasoning with strong procedure preservation | Some faithfulness/supported-answer loss |
| General | 0.1926 | General direct answering | Weak preservation and no abstention |
| Grounding | 0.0736 | Faithfulness/procedure preservation | Low direct reasoning |
| Procedure | 0.0798 | Some procedural-challenge recovery | Large original procedure regression after specialization |
| Abstention | 0.1333 | Perfect official multidomain abstention | Strong false-abstention risk |
| Social | 0.0005 | Best social/preservation specialist | Essentially no direct reasoning |

This complementary error structure—not simply the highest individual score—motivated the K=7 condition.

## Other candidates considered

- Five layer-scoped v2.2 adapters share the frozen base and are technically compatible, but their purpose is layer attribution rather than a clean specialty committee.
- v2.3 residual/procedural adapters attach to D_greedy and are useful for a future second-backbone study, but cannot be mixed with frozen-anchor adapters.
- Earlier Phase 2/2B seed adapters often declare a missing historical Phase 2B base, so their lineage cannot be safely reconstructed for direct voting.
- Dense SFT, layer-reduced, RAG-recovery, memory-reconstruction, and DPO artifacts were inventoried as controls or alternate anchors. None offered a larger verified, same-base specialty pool than v2.2.
- D_greedy itself is a dense merged model, not an independently activatable expert on the frozen anchor.

## Adapter-method capability

Installed PEFT 0.19.1 can execute minimal Qwen LoRA, DoRA, PiSSA, and AdaLoRA forwards and can construct a mixed LoRA/AdaLoRA inference model. The official pool is nevertheless LoRA-only. Matched method training would require a separate controlled budget; no method was substituted or claimed superior without that evidence. See `PEFT_CAPABILITIES.md` and `peft_capability_probe.json`.
