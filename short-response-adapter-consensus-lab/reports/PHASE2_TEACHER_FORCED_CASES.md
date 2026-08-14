# Deterministically selected teacher-forced DEV cases

Status: **DEV_TEACHER_FORCED_QUALITATIVE_SELECTION_COMPLETE**

Cases were selected before narrative interpretation by hashing every eligible example ID within each machine-defined bucket. Counts use all 4,050 examples and 18,760 gold-prefix decisions.

## Bucket inventory

| Bucket | Eligible examples | Selected |
|---|---:|---:|
| minority_expert_correct_but_outvoted | 1,095 | 3 |
| gold_tied_but_tie_break_failed | 294 | 3 |
| tie_break_selected_gold | 157 | 3 |
| overconfident_wrong_dissenter_suppressed | 1,520 | 3 |
| majority_wrong | 1,114 | 3 |
| premature_eos | 90 | 3 |
| failed_to_select_gold_eos | 5 | 3 |
| probability_rescued_hard | 313 | 3 |
| hard_rescued_probability | 204 | 3 |

## minority_expert_correct_but_outvoted

### 1. `v22_devops_diagnostic_priority_0088` at t=1

- Selection hash: `0045fd199a258b672f7c561db423f5e7aac1b787a62949b9a5990a3cbf093e5c`
- Prompt: Devops scenario ca1a73d6: Priority policy: critical -> escalate; else warning -> inspect; else -> monitor. Current observation: critical flag present; warning flag present. Return the selected action.
- Gold answer/token: `escalate.` / `ate` (349)
- Hard / probability token: `.` (13) / `ate` (349)
- Vote counts: `{"13": 2, "16": 1, "349": 1, "3689": 1, "432": 1, "877": 1}`; tied leaders: `[13]`; gold-voting experts: `['track_a']`
- Highest-confidence experts: track_a=`ate` p=0.323 margin=0.101; general=`1` p=0.285 margin=0.089; procedure=`.` p=0.230 margin=0.164

### 2. `v22_tools_two_hop_relation_0088` at t=2

- Selection hash: `006ab13622df6bddfb1897b327357a7cc63d7ba0f0919467cb50ed00f9c0b963`
- Prompt: Tools scenario e4320da9: Relations: Ari -> Team-Falcon; Team-Falcon -> System-Beacon. Follow the two links and name the system.
- Gold answer/token: `System-Beacon.` / `Be` (3430)
- Hard / probability token: `1` (16) / `1` (16)
- Vote counts: `{"16": 2, "18067": 1, "2269": 1, "2727": 1, "3430": 1, "36702": 1}`; tied leaders: `[16]`; gold-voting experts: `['track_a']`
- Highest-confidence experts: grounding=`Em` p=0.298 margin=0.201; track_b=`1` p=0.287 margin=0.246; track_a=`Be` p=0.183 margin=0.021

### 3. `v22_travel_insufficient_information_0081` at t=0

- Selection hash: `0071e43abf8ff59848fec0b96d6951e778f370a76a11b010781c1f631151ed1f`
- Prompt: Travel scenario 32b50992: Known: Beacon=25, Delta=27. Grove's travel time is not provided. Name the highest among all three.
- Gold answer/token: `I don't know based on the available information.` / `I` (40)
- Hard / probability token: `Delta` (20277) / `Delta` (20277)
- Vote counts: `{"20277": 5, "40": 1, "6855": 1}`; tied leaders: `[20277]`; gold-voting experts: `['abstention']`
- Highest-confidence experts: abstention=`I` p=0.999 margin=0.999; grounding=`Delta` p=0.777 margin=0.671; track_a=`Delta` p=0.607 margin=0.501


## gold_tied_but_tie_break_failed

### 1. `v22_writing_capacity_fit_0088` at t=0

- Selection hash: `017e12d80e5831cfb4e9e0c7158712f0f7302fe89f3652499d93204bf75efffe`
- Prompt: Writing scenario 95cec877: Required capacity: 53. Available candidates: Harbor=49, Beacon=56, Cedar=65. Which is the least capacity that still fits?
- Gold answer/token: `Beacon.` / `Be` (3430)
- Hard / probability token: `C` (34) / `C` (34)
- Vote counts: `{"32": 1, "34": 2, "3430": 2, "40": 1, "785": 1}`; tied leaders: `[34, 3430]`; gold-voting experts: `['track_a', 'track_b']`
- Highest-confidence experts: abstention=`I` p=0.995 margin=0.993; grounding=`C` p=0.739 margin=0.697; general=`A` p=0.480 margin=0.415

### 2. `v22_tools_dependency_order_0087` at t=0

- Selection hash: `027c8399d5aee7768a2a1f07f9ef2bf11465d0d735b59e409a0c34428826cef6`
- Prompt: Tools scenario 94ef42d4: Dependencies: Ember <- Grove <- Falcon, where the item on the right must happen first. What is the earliest prerequisite?
- Gold answer/token: `Falcon.` / `F` (37)
- Hard / probability token: `I` (40) / `I` (40)
- Vote counts: `{"32": 2, "37": 2, "38": 1, "40": 2}`; tied leaders: `[32, 37, 40]`; gold-voting experts: `['grounding', 'procedure']`
- Highest-confidence experts: abstention=`I` p=0.999 margin=0.998; grounding=`F` p=0.470 margin=0.388; general=`A` p=0.284 margin=0.202

### 3. `v22_business_diagnostic_priority_0080` at t=1

- Selection hash: `02857a803c347ddd6bbaf57d8f15883d3310bf779b7ade2c0f441996884034b7`
- Prompt: Business scenario 406d859e: Priority policy: critical -> escalate; else warning -> inspect; else -> monitor. Current observation: critical flag present; warning flag present. Return the selected action.
- Gold answer/token: `escalate.` / `ate` (349)
- Hard / probability token: `.` (13) / `ate` (349)
- Vote counts: `{"12528": 1, "13": 1, "16": 1, "349": 1, "4366": 1, "82460": 1, "825": 1}`; tied leaders: `[13, 16, 349, 825, 4366, 12528, 82460]`; gold-voting experts: `['track_b']`
- Highest-confidence experts: social=` shell` p=0.293 margin=0.171; track_a=`1` p=0.217 margin=0.026; general=`version` p=0.189 margin=0.022


## tie_break_selected_gold

### 1. `v22_home_conflicting_evidence_0086` at t=0

- Selection hash: `02b71261ed3780e7e88d5858c2b84e48b20b713445bc922080e3f021fe4f47d1`
- Prompt: Home scenario 9ca5ebf2: For this hypothetical home task, source 1 says Cedar; equally trusted source 2 says Delta. There is no freshness or priority difference. Give the certain answer.
- Gold answer/token: `I don't know based on the available information.` / `I` (40)
- Hard / probability token: `I` (40) / `I` (40)
- Vote counts: `{"20277": 2, "2320": 1, "2753": 1, "40": 2, "8948": 1}`; tied leaders: `[40, 20277]`; gold-voting experts: `['abstention', 'social']`
- Highest-confidence experts: abstention=`I` p=1.000 margin=1.000; grounding=`Delta` p=0.554 margin=0.323; procedure=`No` p=0.348 margin=0.235

### 2. `v22_health_insufficient_information_0084` at t=0

- Selection hash: `0781887a4098c0bbc5998f2edab5bc23011ff62d1024d30aecd985989aa6a8c9`
- Prompt: Health scenario 379bcb87: Known: Cedar=15, Falcon=17. Grove's wait time is not provided. Name the highest among all three.
- Gold answer/token: `I don't know based on the available information.` / `I` (40)
- Hard / probability token: `I` (40) / `I` (40)
- Vote counts: `{"2320": 1, "37": 2, "38": 2, "40": 2}`; tied leaders: `[37, 38, 40]`; gold-voting experts: `['abstention', 'social']`
- Highest-confidence experts: abstention=`I` p=0.999 margin=0.999; grounding=`System` p=0.426 margin=0.094; general=`G` p=0.279 margin=0.163

### 3. `v22_business_conflicting_evidence_0084` at t=0

- Selection hash: `088fff76882e4068ca6429437c71fd07def1759db60d8878528fed4df029a23d`
- Prompt: Business scenario 1c004067: For this hypothetical plan, source 1 says Ember; equally trusted source 2 says Falcon. There is no freshness or priority difference. Give the certain answer.
- Gold answer/token: `I don't know based on the available information.` / `I` (40)
- Hard / probability token: `I` (40) / `I` (40)
- Vote counts: `{"2320": 2, "34": 1, "37": 1, "40": 2, "8948": 1}`; tied leaders: `[40, 2320]`; gold-voting experts: `['abstention', 'social']`
- Highest-confidence experts: abstention=`I` p=1.000 margin=1.000; grounding=`System` p=0.602 margin=0.483; social=`I` p=0.557 margin=0.376


## overconfident_wrong_dissenter_suppressed

### 1. `v22_health_constraint_preserving_transform_0085` at t=0

- Selection hash: `001d6aee9b98dc07e6697c2d98963dfd147615ba481e1bcdfaee22a4c98706a9`
- Prompt: Health scenario e6de4585: Preserve meaning, not wording. Facts: Falcon is approved; deadline is day 5. Candidate A: Falcon approved; deadline day 5. Candidate B: Harbor approved; deadline day 5. Candidate C: Falcon approved; deadline day 6. Which candidate is faithful?
- Gold answer/token: `A.` / `A` (32)
- Hard / probability token: `A` (32) / `A` (32)
- Vote counts: `{"32": 3, "37": 2, "40": 2}`; tied leaders: `[32]`; gold-voting experts: `['track_a', 'track_b', 'general']`
- Highest-confidence experts: abstention=`I` p=0.958 margin=0.935; track_a=`A` p=0.921 margin=0.885; general=`A` p=0.737 margin=0.700

### 2. `v22_home_insufficient_information_0083` at t=2

- Selection hash: `0070ebee2d0b97c86dc7d6e18bb0e7baea22eb4a6068e8e6f1ba1557cc5c666a`
- Prompt: Home scenario 7152f903: Known: Harbor=37, Cedar=39. Beacon's time is not provided. Name the highest among all three.
- Gold answer/token: `I don't know based on the available information.` / `'t` (944)
- Hard / probability token: `'t` (944) / `'t` (944)
- Vote counts: `{"13": 1, "944": 6}`; tied leaders: `[944]`; gold-voting experts: `['track_a', 'track_b', 'general', 'grounding', 'abstention', 'social']`
- Highest-confidence experts: abstention=`'t` p=1.000 margin=1.000; social=`'t` p=0.999 margin=0.999; procedure=`.` p=0.999 margin=0.998

### 3. `v22_travel_insufficient_information_0083` at t=2

- Selection hash: `007d23b8fe2fe544bbd093643bcd72b57cfb632b80b2ea21cbc7a7052d332188`
- Prompt: Travel scenario 0a02b2bb: Known: Delta=39, Falcon=41. Atlas's travel time is not provided. Name the highest among all three.
- Gold answer/token: `I don't know based on the available information.` / `'t` (944)
- Hard / probability token: `'t` (944) / `'t` (944)
- Vote counts: `{"11": 1, "13": 1, "944": 5}`; tied leaders: `[944]`; gold-voting experts: `['track_a', 'track_b', 'general', 'abstention', 'social']`
- Highest-confidence experts: abstention=`'t` p=1.000 margin=1.000; procedure=`.` p=1.000 margin=1.000; general=`'t` p=0.999 margin=0.999


## majority_wrong

### 1. `v22_general_two_hop_relation_0083` at t=2

- Selection hash: `00c41e52c19f353ed7f6d172a65b825b808a2ffa87b72ce094be99003f23c6ba`
- Prompt: General scenario 2e895e54: Relations: Dara -> Team-Delta; Team-Delta -> System-Falcon. Follow the two links and name the system.
- Gold answer/token: `System-Falcon.` / `alcon` (46214)
- Hard / probability token: `.` (13) / `.` (13)
- Vote counts: `{"12528": 1, "13": 5, "2968": 1}`; tied leaders: `[13]`; gold-voting experts: `[]`
- Highest-confidence experts: procedure=`.` p=0.974 margin=0.962; general=`.` p=0.924 margin=0.915; track_a=`.` p=0.849 margin=0.787

### 2. `v22_health_dependency_order_0081` at t=0

- Selection hash: `016a094a057b17950438d20c9d8b1db20a7f1c5a33b826713e6329674872d56f`
- Prompt: Health scenario 1aaa7505: Dependencies: Beacon <- Delta <- Grove, where the item on the right must happen first. What is the earliest prerequisite?
- Gold answer/token: `Grove.` / `G` (38)
- Hard / probability token: `Delta` (20277) / `Delta` (20277)
- Vote counts: `{"20277": 5, "40": 1, "6855": 1}`; tied leaders: `[20277]`; gold-voting experts: `[]`
- Highest-confidence experts: abstention=`I` p=0.994 margin=0.991; grounding=`Delta` p=0.874 margin=0.840; track_a=`Delta` p=0.769 margin=0.730

### 3. `v22_legal_filter_then_rank_0081` at t=2

- Selection hash: `0179da379475e6ea5a1d11320a29254cdfb62f7f00a270ed9e5c0785356be572`
- Prompt: Legal scenario e9de7ef3: Minimum quality is 28. Filter Grove(31, $121), Ember(26, $81), Harbor(35, $101); then pick the cheapest qualifying policy option.
- Gold answer/token: `Harbor.` / `.` (13)
- Hard / probability token: `<|im_end|>` (151645) / `.` (13)
- Vote counts: `{"13": 3, "151645": 4}`; tied leaders: `[151645]`; gold-voting experts: `['track_a', 'general', 'grounding']`
- Highest-confidence experts: track_b=`<|im_end|>` p=0.924 margin=0.884; general=`.` p=0.920 margin=0.862; track_a=`.` p=0.898 margin=0.814


## premature_eos

### 1. `v22_coaching_filter_then_rank_0081` at t=2

- Selection hash: `01657db7ef71dcea9b9c538671ab6305d227df3b5d8c959756f907ce478cf9c4`
- Prompt: Coaching scenario b7351a75: Minimum quality is 31. Filter Beacon(34, $121), Ember(29, $81), Harbor(38, $101); then pick the cheapest qualifying practice plan.
- Gold answer/token: `Harbor.` / `.` (13)
- Hard / probability token: `<|im_end|>` (151645) / `<|im_end|>` (151645)
- Vote counts: `{"151645": 6, "57810": 1}`; tied leaders: `[151645]`; gold-voting experts: `[]`
- Highest-confidence experts: track_b=`<|im_end|>` p=0.902 margin=0.863; procedure=`<|im_end|>` p=0.900 margin=0.850; general=`<|im_end|>` p=0.871 margin=0.767

### 2. `v22_education_diagnostic_priority_0080` at t=2

- Selection hash: `01be2532a053847d12709546ab07084e3d18658052aec0735771510a8f717445`
- Prompt: Education scenario 85a366e1: Priority policy: critical -> escalate; else warning -> inspect; else -> monitor. Current observation: critical flag present; warning flag present. Return the selected action.
- Gold answer/token: `escalate.` / `.` (13)
- Hard / probability token: `<|im_end|>` (151645) / `<|im_end|>` (151645)
- Vote counts: `{"13": 2, "151645": 5}`; tied leaders: `[151645]`; gold-voting experts: `['track_a', 'track_b']`
- Highest-confidence experts: grounding=`<|im_end|>` p=0.879 margin=0.774; general=`<|im_end|>` p=0.828 margin=0.665; procedure=`<|im_end|>` p=0.796 margin=0.639

### 3. `v22_home_diagnostic_priority_0080` at t=2

- Selection hash: `063b6c7af3980ab3fdab472656a9899564b8e7b77e4650a0f8b9e95e65a34a6e`
- Prompt: Home scenario e66dc09c: Priority policy: critical -> escalate; else warning -> inspect; else -> monitor. Current observation: critical flag present; warning flag present. Return the selected action.
- Gold answer/token: `escalate.` / `.` (13)
- Hard / probability token: `<|im_end|>` (151645) / `<|im_end|>` (151645)
- Vote counts: `{"13": 2, "151645": 5}`; tied leaders: `[151645]`; gold-voting experts: `['track_a', 'procedure']`
- Highest-confidence experts: grounding=`<|im_end|>` p=0.825 margin=0.663; general=`<|im_end|>` p=0.788 margin=0.588; track_a=`.` p=0.719 margin=0.455


## failed_to_select_gold_eos

### 1. `v22_math_constraint_preserving_transform_0083` at t=2

- Selection hash: `220ebe5b43bbdb8545d4fd8a5930e03873dc29c4619e3b79332ddd2f900bd747`
- Prompt: Math scenario 55355ed3: Preserve meaning, not wording. Facts: Harbor is approved; deadline is day 3. Candidate A: Harbor approved; deadline day 3. Candidate B: Cedar approved; deadline day 3. Candidate C: Harbor approved; deadline day 4. Which candidate is faithful?
- Gold answer/token: `A.` / `<|im_end|>` (151645)
- Hard / probability token: `3` (18) / `<|im_end|>` (151645)
- Vote counts: `{"14817": 1, "151645": 2, "18": 2, "35520": 1, "358": 1}`; tied leaders: `[18, 151645]`; gold-voting experts: `['track_a', 'general']`
- Highest-confidence experts: general=`<|im_end|>` p=0.979 margin=0.976; track_a=`<|im_end|>` p=0.979 margin=0.977; abstention=` I` p=0.409 margin=0.239

### 2. `v22_translation_aggregate_delta_0083` at t=3

- Selection hash: `58ce9d615835df3580fe094f8d5569a18f3a5a7999caae5212cdf1d59886a753`
- Prompt: Translation scenario d986a944: Track the phrase: initial 33; +9; +6; -4. Final total?
- Gold answer/token: `44.` / `<|im_end|>` (151645)
- Hard / probability token: `.` (13) / `.` (13)
- Vote counts: `{"13": 3, "151645": 2, "358": 1, "96371": 1}`; tied leaders: `[13]`; gold-voting experts: `['track_a', 'track_b']`
- Highest-confidence experts: grounding=`.` p=0.954 margin=0.950; procedure=`.` p=0.876 margin=0.853; general=`.` p=0.754 margin=0.538

### 3. `v22_math_constraint_preserving_transform_0082` at t=2

- Selection hash: `9597e49107bcfdd597f3a5d7cd6b81fc1aa219b73c80a40523107125080b94ae`
- Prompt: Math scenario abd71d12: Preserve meaning, not wording. Facts: Grove is approved; deadline is day 2. Candidate A: Grove approved; deadline day 2. Candidate B: Harbor approved; deadline day 2. Candidate C: Grove approved; deadline day 3. Which candidate is faithful?
- Gold answer/token: `A.` / `<|im_end|>` (151645)
- Hard / probability token: ` ` (220) / `<|im_end|>` (151645)
- Vote counts: `{"13": 1, "151645": 2, "17": 1, "220": 2, "358": 1}`; tied leaders: `[220, 151645]`; gold-voting experts: `['track_a', 'general']`
- Highest-confidence experts: general=`<|im_end|>` p=0.920 margin=0.868; track_a=`<|im_end|>` p=0.898 margin=0.871; grounding=`2` p=0.342 margin=0.308


## probability_rescued_hard

### 1. `v22_consumer_filter_then_rank_0089` at t=2

- Selection hash: `003283e2697db95b0b97cfa62e15fc11ee249a0ee196a12cda7feba12e457ba1`
- Prompt: Consumer scenario 15729408: Minimum quality is 25. Filter Beacon(28, $129), Ember(23, $89), Harbor(32, $109); then pick the cheapest qualifying product.
- Gold answer/token: `Harbor.` / `.` (13)
- Hard / probability token: `<|im_end|>` (151645) / `.` (13)
- Vote counts: `{"13": 3, "151645": 3, "57810": 1}`; tied leaders: `[13, 151645]`; gold-voting experts: `['track_a', 'general', 'grounding']`
- Highest-confidence experts: track_a=`.` p=0.917 margin=0.859; general=`.` p=0.821 margin=0.695; track_b=`<|im_end|>` p=0.815 margin=0.748

### 2. `v22_communication_aggregate_delta_0085` at t=2

- Selection hash: `004c44a14ff813a675d69e469e70f5c2c2d1b6dda0104e7442043fc2d3c03fe6`
- Prompt: Communication scenario 1d4e0f94: Track the message: initial 49; +4; +4; -1. Final total?
- Gold answer/token: `56.` / `.` (13)
- Hard / probability token: ` GB` (18865) / `.` (13)
- Vote counts: `{"13": 3, "18865": 3, "6605": 1}`; tied leaders: `[13, 18865]`; gold-voting experts: `['track_a', 'track_b', 'general']`
- Highest-confidence experts: track_a=`.` p=0.960 margin=0.958; abstention=` GB` p=0.795 margin=0.738; general=`.` p=0.746 margin=0.704

### 3. `v22_research_conflicting_evidence_0087` at t=0

- Selection hash: `015d60a1af927d0f78a28fbb2784fa6b842f634aa6af15f74469fa2f8136703f`
- Prompt: Research scenario 4a559856: For this hypothetical analysis, source 1 says Harbor; equally trusted source 2 says Grove. There is no freshness or priority difference. Give the certain answer.
- Gold answer/token: `I don't know based on the available information.` / `I` (40)
- Hard / probability token: `Har` (26490) / `I` (40)
- Vote counts: `{"26490": 2, "32": 1, "34": 1, "38": 1, "40": 2}`; tied leaders: `[40, 26490]`; gold-voting experts: `['abstention', 'social']`
- Highest-confidence experts: abstention=`I` p=0.999 margin=0.999; grounding=`Har` p=0.368 margin=0.215; social=`I` p=0.253 margin=0.171


## hard_rescued_probability

### 1. `v22_devops_constraint_preserving_transform_0080` at t=0

- Selection hash: `00250dfd223bcf1d8eab5c00417a57278f98fe83d45beb2570064036b3f95e59`
- Prompt: Devops scenario 07a7a1ea: Preserve meaning, not wording. Facts: Grove is approved; deadline is day 9. Candidate A: Grove approved; deadline day 9. Candidate B: Beacon approved; deadline day 9. Candidate C: Grove approved; deadline day 10. Which candidate is faithful?
- Gold answer/token: `A.` / `A` (32)
- Hard / probability token: `A` (32) / `I` (40)
- Vote counts: `{"2320": 1, "32": 3, "38": 1, "40": 1, "785": 1}`; tied leaders: `[32]`; gold-voting experts: `['track_a', 'track_b', 'general']`
- Highest-confidence experts: abstention=`I` p=0.955 margin=0.938; track_a=`A` p=0.712 margin=0.661; general=`A` p=0.360 margin=0.326

### 2. `v22_home_two_hop_relation_0083` at t=0

- Selection hash: `00e089f4791a8248bf22c97de705cfeb52b7102c5125cbfb5977fa2ff1a3dc7f`
- Prompt: Home scenario 6ccbc5e4: Relations: Dara -> Team-Harbor; Team-Harbor -> System-Cedar. Follow the two links and name the system.
- Gold answer/token: `System-Cedar.` / `System` (2320)
- Hard / probability token: `System` (2320) / `I` (40)
- Vote counts: `{"10141": 1, "2320": 3, "34": 1, "38": 1, "40": 1}`; tied leaders: `[2320]`; gold-voting experts: `['track_a', 'track_b', 'grounding']`
- Highest-confidence experts: abstention=`I` p=0.999 margin=0.999; grounding=`System` p=0.530 margin=0.396; track_b=`System` p=0.235 margin=0.109

### 3. `v22_data_analysis_constraint_preserving_transform_0081` at t=0

- Selection hash: `021c7711cd5aabb3b66c76a420974eb62ca15cc5ac7ecab00270ea76357659d7`
- Prompt: Data Analysis scenario 8f16377e: Preserve meaning, not wording. Facts: Grove is approved; deadline is day 1. Candidate A: Grove approved; deadline day 1. Candidate B: Ember approved; deadline day 1. Candidate C: Grove approved; deadline day 2. Which candidate is faithful?
- Gold answer/token: `A.` / `A` (32)
- Hard / probability token: `A` (32) / `I` (40)
- Vote counts: `{"32": 3, "38": 2, "40": 2}`; tied leaders: `[32]`; gold-voting experts: `['track_a', 'track_b', 'general']`
- Highest-confidence experts: abstention=`I` p=0.990 margin=0.986; track_a=`A` p=0.775 margin=0.718; general=`A` p=0.585 margin=0.530

## Interpretation boundary

These are local gold-prefix examples. They support case-level mechanism inspection but do not identify free-running counterfactual trajectories or causal expert effects.
