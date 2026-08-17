# Curriculum Map

Rineya v2.0 is organized as 40 balanced capability classes. Each class contributes 800 new SFT rows and 300 DPO preference pairs. The DPO rejection is designed around a class-specific failure mode rather than generic verbosity alone.

| Group | Class | Learning objective | Representative use cases | Primary DPO contrast |
|---|---|---|---|---|
| reasoning | `arithmetic_chain` | Track several numeric operations and return only the final value. | Multi-operation totals, grouping, doubling, remaining balances. | Arithmetic slip or extra explanation. |
| reasoning | `percentage_reasoning` | Apply percentages to totals, discounts, increases, and completion rates. | Prices, progress, growth, portions. | Wrong base value or percent direction. |
| reasoning | `ratio_proportion` | Scale quantities while preserving ratios and proportions. | Recipes, printing, mixtures, packs. | Uses the original rather than scaled amount. |
| reasoning | `unit_conversion` | Convert common units with grammatically correct compact answers. | Time, distance, mass, volume. | Wrong factor or mismatched unit. |
| reasoning | `time_duration` | Add and subtract durations across clock times. | Meetings, trips, breaks, deadlines. | Clock arithmetic error. |
| reasoning | `schedule_ordering` | Respect prerequisites, fixed times, and ordering constraints. | Work queues, appointments, task sequences. | Violates a dependency or fixed slot. |
| reasoning | `comparison_ranking` | Rank options from explicit measurements and resolve ties correctly. | Scores, prices, speeds, quantities. | Reverses the requested ordering. |
| reasoning | `logical_implication` | Apply stated implications without affirming the consequent. | Eligibility, permissions, status rules. | Invalid reverse inference. |
| reasoning | `conditional_rules` | Apply positive and negative if-then rules to concrete cases. | Access, alerts, workflow decisions. | Ignores the condition or negation. |
| reasoning | `contradiction_detection` | Detect when two supplied claims cannot both be true. | Schedules, ownership, quantities, locations. | Accepts contradictory facts. |
| reasoning | `sequence_pattern` | Continue simple numeric and symbolic patterns. | Arithmetic, alternating, repeating, stepped sequences. | Applies the wrong pattern. |
| reasoning | `counting_sets` | Compute unions, overlaps, and remaining counts. | Attendance, inventory, memberships, selections. | Double-counts overlap. |
| reasoning | `constraint_selection` | Choose an option satisfying multiple explicit constraints. | Budget, size, time, feature combinations. | Optimizes one condition while violating another. |
| reasoning | `causal_reasoning` | Identify the directly supported likely cause without embellishment. | Everyday failures and simple events. | Invents an unrelated cause. |
| reasoning | `physical_commonsense` | Use basic object and physical-state relations. | Containers, balance, temperature, motion, fit. | Contradicts ordinary physical behavior. |
| reasoning | `spatial_relations` | Compose relative positions and directions. | Left/right, above/below, route movement. | Reverses or collapses the relation. |
| reasoning | `estimation_sanity` | Reject implausible magnitudes and choose reasonable estimates. | Time, distance, count, capacity, price. | Accepts an order-of-magnitude error. |
| reasoning | `evidence_inference` | Infer only what explicit evidence supports. | Mini records, observations, stated evidence. | Adds an unsupported conclusion. |
| reasoning | `insufficient_information` | Recognize underdetermination and abstain instead of guessing. | Missing rates, identities, causes, comparison data. | Fabricated certainty. |
| reasoning | `troubleshooting_diagnosis` | Select the most relevant first check from symptoms and changes. | Software, devices, network, files, peripherals. | Jumps to an unrelated fix. |
| communication | `acknowledgment_update` | Acknowledge updates naturally while preserving the changed fact. | Status changes, completion, files, routes, participants. | Robotic overresponse or lost update. |
| communication | `natural_follow_up` | Ask one socially natural question that advances the conversation. | News, projects, travel, outcomes, experiences. | Stiff acknowledgment that ends the exchange. |
| communication | `empathy_validation` | Validate emotion briefly without diagnosing or overpromising. | Stress, disappointment, exhaustion, nerves. | Dismissal, minimization, or clinical overreach. |
| communication | `tactful_disagreement` | Disagree directly but respectfully and give a concise reason. | Plans, priorities, designs, interpretations. | Hostile or unsupported contradiction. |
| communication | `clarification_request` | Ask the smallest useful clarification when a request is ambiguous. | Unclear referents, versions, goals, recipients, speed. | Acts confidently on missing information. |
| communication | `preference_opinion` | Give a direct preference aligned with the user-stated criterion. | Products, formats, plans, styles, options. | Hedges or chooses against the stated priority. |
| communication | `apology_repair` | Correct an earlier mistake, acknowledge it, and use the newest fact. | Times, files, locations, totals, choices. | Repeats the superseded answer. |
| communication | `deescalation` | Lower conversational temperature without surrendering the issue. | Work conflict, delays, misunderstandings, frustration. | Escalatory or accusatory language. |
| communication | `negotiation_compromise` | Propose a compact compromise that preserves both sides’ main goals. | Budget versus quality, speed versus review, scope versus time. | One-sided surrender. |
| communication | `boundary_setting` | Refuse fabrication, dishonesty, unsafe shortcuts, or false claims concisely. | False verification, invented memory, fake citations, misrepresentation. | Unsafe or dishonest compliance. |
| communication | `informal_texting` | Use clear, casual, human-sounding short replies. | Everyday texting, quick reactions, lightweight coordination. | Needlessly formal or verbose phrasing. |
| communication | `conversation_closure` | Close completed exchanges naturally without reopening them. | Thanks, resolved tasks, completed updates, farewells. | Unnecessary follow-up or extended lecture. |
| integrated | `decision_explanation` | Make a choice and attach one compact reason tied to the criterion. | Cost, reliability, time, simplicity, quality. | Choice without criterion or lengthy analysis. |
| integrated | `multi_turn_context_update` | Follow corrections and later facts across multiple turns. | Changed names, destinations, totals, formats, plans. | Uses stale conversation state. |
| integrated | `plan_revision` | Revise a plan when a constraint or fact changes. | Deadlines, availability, budgets, dependencies. | Preserves an invalid original plan. |
| integrated | `comparative_recommendation` | Compare options and recommend one under explicit tradeoffs. | Purchases, schedules, tools, routes, project choices. | Generic answer that ignores the tradeoff. |
| integrated | `practical_next_step` | Turn a diagnosed situation into the smallest useful next action. | Troubleshooting, preparation, coordination, safety. | Vague advice or premature large action. |
| integrated | `evidence_grounded_dialogue` | Answer conversationally from supplied evidence and abstain when unsupported. | RAG-style snippets, records, notes, procedures. | Hallucinated or contradicted answer. |
| integrated | `communication_rewrite` | Rewrite a message to meet a stated social goal while preserving content. | Friendlier, clearer, firmer, shorter, more tactful. | Changes the meaning or misses the requested tone. |
| integrated | `roleplay_reasoning` | Stay in a lightweight role while using explicit scene facts correctly. | Coordinator, guide, teammate, clerk, game roles. | Character break or fact inconsistency. |

## Cross-cutting design

- Every target is the final assistant turn; the prompt is masked during SFT.
- No chain-of-thought is supervised. Reasoning rows teach compact conclusions, explicit insufficiency, and evidence boundaries.
- Prompt surfaces vary across systems, domains, names, locations, conversational preludes, and multi-turn state changes.
- Counterfactual families change one fact or condition while holding the surrounding structure stable, encouraging boundary learning rather than phrase memorization.
- The 8,000-row v1.0 preservation slice protects the original short-response, safety, empathy, clarification, and roleplay behavior.
- Open-ended communication accepts multiple reasonable human strategies; the supplied target is one high-quality training target, not a claim that it is the only valid response.

## Review emphasis

Human reviewers should pay extra attention to empathy, de-escalation, boundary setting, informal language, roleplay, and DPO social contrasts. Deterministic audits can recompute numerical and explicit-contract answers, but cannot certify that every social wording is optimal.
