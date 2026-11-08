# Phase 2b — Instrument calibration

Before spending GPU-hours on Phase 3, the benchmark was checked for **dynamic
range**: a task the model fails identically at every position measures nothing about
position, and a metric that saturates measures nothing about an intervention. The
check found one protocol bug, one metric that is unusable as a primary, and one task
family sitting on the floor. All three are fixed or scoped before Phase 3, and all
three are recorded here rather than silently corrected.

Pilot: 162 items, 2K natural-prose bucket, Qwen3-0.6B bf16, greedy, batch 1.

## 1. The `<think>` trap — a protocol bug, caught

Qwen3 is a hybrid reasoning model. Rendering the chat template without
`enable_thinking=False` makes it open a `<think>` block and spend the entire
generation budget reasoning, so the answer never appears:

```
[MISS] exact_retrieval  gold='402eb5c6-44e9-3b20-b89ea57e9d94'  pred='<think>'
```

That scored **0/27 across every family and every position** — a clean, symmetric,
entirely artefactual null that would have looked like "the benchmark is too hard"
rather than "the harness is wrong". With `enable_thinking=False`, the same items
score non-zero immediately.

The project already has a canonical protocol for this
(`rineya_long_context_demo_v1_bf16_greedy_no_think_24_4096`), and the lab's
`evaluation/runner.py` and `benchmark/tokenize.py` were already using it. The bug was
in a hand-rolled pilot script that bypassed the runner. The lesson is kept as a rule:
**any script that renders a prompt without going through the runner is a bug.**

## 2. Accuracy is at floor; gold NLL is not

Teacher-forced gold NLL per answer token against evidence position, `exact_retrieval`
(n = 6 per cell):

| p | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **gold NLL** | **0.221** | 0.408 | 0.502 | 0.482 | 0.426 | 0.449 | **0.197** |
| strict EM | 2/6 | 0/6 | 0/6 | 1/6 | 0/6 | 0/6 | 2/6 |
| contains | 3/6 | 2/6 | 2/6 | 3/6 | 1/6 | 1/6 | 4/6 |

The NLL row is U-shaped with an edge–middle gap of about **0.27 nats**, on six items
per cell. The accuracy rows carry the same shape at a fraction of the resolution and
would need an order of magnitude more items to reach the same confidence.

**Decision:** gold NLL and answer margin are the primary metrics for every
mechanistic phase; accuracy is reported alongside but is never the quantity an
intervention is selected on. The brief's §10 called for this as a precaution; the
pilot makes it a measured requirement.

This is a preview, not a result. It is one model, one family, one length, n=6, and no
confidence intervals. Phase 3 does it properly.

## 3. Two scoring artefacts

Both of these are the model finding the answer and the scorer refusing it:

- **Format non-compliance** — gold `the loom overseer`, prediction
  `The answer is: **The loom overseer**.`
- **Truncation** — the model restates the question, then runs out of budget:
  `The tally mark for casket 06c83ff1-506e-034e-bd1c6205c30a is`

`evaluation/metrics.py` therefore reports three accuracy columns together and never
collapses them: `strict`, `contains`, and `first_line`. **`contains` is a ceiling,
not a score** — a sufficiently verbose model inflates it — and it may not be used
alone to accept an intervention.

## 4. Multi-hop is on the floor

| family | strict | contains | gold NLL |
| --- | ---: | ---: | ---: |
| `exact_retrieval` | 3/42 | 16/42 | 0.21–0.50 |
| `semantic_single_hop` | 0/42 | 17/42 | 1.71–2.70 |
| `state_constraint` | 0/42 | 15/42 | 3.25 |
| `multi_hop` | **0/36** | **0/36** | **4.38–5.07** |

`multi_hop` has no signal whatsoever. The brief's §41 requires "at least one genuine
multi-hop middle-context condition must improve", which cannot be evaluated against a
floor — an intervention that helped would be indistinguishable from one that did not.

**Fix:** a `multi_hop_guided` variant with explicit linking language and the bridge
entity repeated verbatim in both evidence sentences, so a 0.6B model can follow the
chain. The hard variant is retained; both are reported. If guided multi-hop is *also*
at floor, that is a capability limit of the panel, and the §41 criterion is declared
unmeetable **in the report** rather than quietly dropped.

## 5. Exact retrieval was testing transcription, not retrieval

Two distinct failure modes were scoring identically:

- retrieving the **wrong record** — a well-formed but different UUID
  (`9de80e8d-…` for gold `402eb5c6-…`);
- retrieving the **right record** and mis-copying one character
  (`902eb5c6-…` for gold `402eb5c6-…`).

The first is the positional failure the study is about. The second is a transcription
failure over a 13–16 token string, and it is noise. Answers become **short nonce keys
(6–8 characters)**; the *lookup keys inside the context* stay long and UUID-like, so
the retrieval problem is unchanged and only the reproduced string shortens. This stays
within the brief's §7A menu. Answer token length is recorded per item so the
transcription-length confound can be checked rather than assumed away.

## 6. What the benchmark got right

Worth stating, since the above is a list of problems. The generator's construction
holds up: filler is original, digit-free and semantically inert (which is what makes
the digit-bearing evidence unambiguous); positional variants of a scenario reuse an
identical multiset of filler chunks, so only order changes; distractors are
format-matched (same field different key, same key different field, and a
"withdrawn draft" negative); and `state_constraint` items require honouring a later
correction over an earlier value — 74 + 5 (amended from 16) − 11 = 68 — rather than
summing everything in sight.

## 7. Post-fix validation on the full bank

The regenerated bank is 4752 items over 308 scenarios (calibration 1752, validation
1413, sealed 1587), including 720 guided and 720 hard multi-hop items. Checked on the
real data rather than on unit-test fixtures:

| property | result |
| --- | --- |
| filler multiset identical across all positional variants of a scenario | **0 violations / 864 groups** |
| chunk order actually varies across positions within a group | 864 / 864 |
| prompt token budget within ±2 % of target | 4752 / 4752, mean error 0.47 %, max 1.95 % |
| ‖achieved − target‖ normalised evidence position | mean **0.0070**, p95 0.0230, max 0.0578 |
| scenarios spanning more than one split | 0 |
| `exact_retrieval` answer length | exactly 7 characters |

Position grid spacing is 0.15–0.17, so a worst-case placement error of 0.058 keeps
every evidence position comfortably inside its own cell.

One correction worth recording, because it is the kind of mistake that reads as a
catastrophic result: a first pass at the position check compared
`meta.primary_achieved_char` — the *decisive* evidence sentence — against
`target_positions[0]`, the *first* sentence. For `multi_hop` and `state_constraint`
those are different sentences, and the comparison produced an apparent mean error of
0.17 and a max of 0.97, i.e. "position control is completely broken". Comparing like
with like gives the table above. The generator was correct; the check was not.
