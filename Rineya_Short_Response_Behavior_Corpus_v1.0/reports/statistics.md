# Rineya Short-Response Behavior Corpus v1.0 — Statistics

## Corpus sizes

| Artifact | Rows | Purpose |
|---|---:|---|
| New behavior SFT | 10,000 | New coverage-balanced short-response behavior |
| Preservation SFT | 2,000 | Existing accepted train-only behavior retained against forgetting |
| Combined SFT | 12,000 | Recommended first SFT input |
| DPO preferences | 5,000 | Chosen concise answer versus targeted failure mode |

## New-SFT structure

- Response classes: **25**
- Multi-turn rows: **1,498**
- Unique rendered prompts: **10,000/10,000**
- Proxy target length: median **7**, P90 **11**, maximum **17**
- Human review status: **pending**

## Length modes

| Mode | Rows |
|---|---:|
| ESCALATE | 1,921 |
| SENTENCE | 2,679 |
| SHORT | 3,611 |
| TINY | 1,789 |

## Leakage isolation

- Exact rendered-prompt overlap with repository DEV/SEALED: **0**
- Exact latest-user overlap with repository DEV/SEALED: **0**
- RapidFuzz ratio ≥94 near-overlap: **0**
- Template-family overlap: **0**

## New-SFT class distribution

| Class | Rows | DPO pairs |
|---|---:|---:|
| `acknowledgment` | 350 | 175 |
| `apology_repair` | 200 | 100 |
| `arithmetic_short_answer` | 400 | 200 |
| `celebration` | 300 | 150 |
| `clarification` | 700 | 350 |
| `comparison_recommendation` | 350 | 175 |
| `complex_escalation` | 200 | 100 |
| `concise_explanation` | 400 | 200 |
| `confirmation` | 250 | 125 |
| `conflict_deescalation` | 200 | 100 |
| `context_recall` | 650 | 325 |
| `context_update` | 350 | 175 |
| `direct_preference` | 350 | 175 |
| `empathy` | 450 | 225 |
| `factual_compression` | 1,000 | 500 |
| `gentle_disagreement` | 300 | 150 |
| `informal_texting` | 450 | 225 |
| `knowledge_boundary` | 350 | 175 |
| `natural_follow_up` | 300 | 150 |
| `output_only` | 250 | 125 |
| `practical_one_step` | 450 | 225 |
| `pronoun_resolution` | 250 | 125 |
| `roleplay_short` | 400 | 200 |
| `safety_boundary` | 650 | 325 |
| `uncertainty` | 450 | 225 |
