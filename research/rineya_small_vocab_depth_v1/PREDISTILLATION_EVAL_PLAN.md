# SMALLVD-001 — pre-distillation evaluation plan (Phase 10, Phase 13-15)

**Status: PLAN. Nothing in this document has been run on a student.** It fixes what the
surgery-only arms A0-A3 are measured on, before any recovery training, so immediate structural
damage is recorded permanently and cannot be hidden by recovery afterwards.

## 1. The obstacle, and the only honest way round it

Every *qualified* probe (process, long-context, HSKILL2 via `run._load`; retention via
`hf_checkpoint`) hard-locks the teacher geometry: 28 blocks, 151,936 rows, tied, and sidecars
byte-identical to Supa. **A pruned student cannot enter them**, and editing them would change the
`implementation_digest` (a329eb7b) and break every recorded comparison.

**Plan: an evaluation shim, qualified against A0 before it may score anything else.**

- **Path:** HF `generate` (greedy, fp32, TF32 off, `eager` attention, batch 1) on prompt strings
  **pre-rendered once under transformers 4.51.3 with the teacher's chat template**, exactly the
  QWEN-001 stage-2 design, which reproduced six qualified values exactly. Prompt strings are
  tokenizer-agnostic; each arm tokenizes them with its OWN tokenizer (every chat special token and
  the empty think block are retained by construction, G1).
- **Scorers:** imported unchanged from the qualified packages (`retention.py:score_prediction`,
  `boundary_rule.py` at its pinned sha, the long-context first-line strict matcher, the RCI and
  alias scorers) and hash-pinned in the run manifest. The shim generates; it never re-implements a
  scorer.
- **Qualification gate (per instrument, before any student row is generated):** A0 through the
  shim must reproduce the recorded Supa per-row predictions **text-identically on every row** and
  the recorded score exactly. G0b (SELECT-VIEWS, 384 rows) is the first instance of this gate. An
  instrument whose A0 replay is not exact is REPORT-ONLY for this track, with its mismatch rate.

## 2. Instruments, and why each is admissible

| instrument | rows | skill | Supa A0 recorded | admissibility / caveat |
|---|---:|---|---|---|
| HSKILL-VIEWS-002 | 224 | abstention/boundary + short arithmetic | boundary 71/112, answerable 40/112 | DEV-allowed, unspent. **All 224 rows are training-corpus rows** (v1 SRB + v3), so it measures retention of trained behaviour, not generalisation |
| RCI-001 calibration twin | 128 | long-context extraction + literal UNKNOWN | 63/128 (abstention 10/32) | spent for confirmation, DEV-allowed |
| RCI-002 calibration twin | 256 | same | 91/256 | as above |
| ALIAS-001 | 360 | binding | 270/360 | open mechanism-scout bank; the "high" condition sits at its 1.0 floor |
| retention/arithmetic panel | 150 | concise arithmetic + typed-answer format | 47 permissive / 41 strict | **every row is a training row**; permissive policy registered; the period-strip convention is load-bearing (`SCORER_SURFACE_CONVENTION_HAZARD.md`) |
| long-context Demo | 256 | long-context retrieval, extraction, abstention | 161 (UNKNOWN 13/64) | SPENT, **report-only**; Supa's own LoRA was selected on it |
| SELECT-VIEWS-001 | 384 | JSON state transition | 14/384 | Supa is **below its floor (49)**: a student cannot be shown to lose ground here. Report-only canary |
| chat DEV | 510 | instruction following, writing, conversation | **never run for Supa** | heuristic scorer (quality v4.5), not a judge; A0 establishes the baseline |

**Excluded:** RCI-001/RCI-002 sealed banks (spent), HSKILL-CONFIRM-002 (sealed, unread, reserved),
Invisible-Limit test sets, and the RAG DEV (its knowledge base is missing from every checkout).

## 3. Protected skills with NO instrument — named, not papered over

| skill | status | what this track measures instead |
|---|---|---|
| coding | no instrument; no coding class in the corpora | held-out Python/TypeScript **bits per byte** (tokenizer-independent), calibration HOLDOUT |
| factuality | no closed-book factual QA | none; reported as UNMEASURED |
| writing transformation | no dedicated instrument | v3 `communication_rewrite` rows in the layer panels (flip rate vs teacher); chat DEV |
| verifiable instruction following | none (MRS-FP-1 has 8 items) | none; a verifiable-constraint bank is recommended before Phase 12 |
| grounded answering | RAG DEV not runnable | cc0 validation rows + v3 evidence classes in the layer panels |

## 4. Tokenizer-independent language-modelling measurements

On the calibration HOLDOUT (never used to choose tokens or blocks): **bits per byte** of
reference answers and of plain text (prose, Python, TypeScript, Markdown) for A0-A3. BPB is the
only likelihood quantity comparable across a vocabulary change; per-token perplexity is not, and
is not reported across vocabularies.

## 5. Generation stability and format

Per arm and instrument: EOS-termination rate, cap-hit rate, mean output length, repetition rate
(any 4-gram repeated ≥ 3 times), JSON-parse rate where the instrument expects JSON, literal
`UNKNOWN` rate, and prompt token counts under the arm's own tokenizer (context consumption).

## 6. Statistics

Per instrument, paired per row against A0: exact two-sided McNemar on the credited rows, with the
resolution precondition derived from the multiplicity actually applied (`MIN_ND_FOR(m)`; an
UNRESOLVED comparison satisfies no clause). Clopper-Pearson intervals on each score. Additivity
(A3 vs A1 + A2) is descriptive only. No significance claim is registered beyond these.

## 7. Deployment measurements (Phase 15), fixed hardware

RTX 4070 Laptop 8 GB, one runtime (.venv-rab001-cu126), batch 1, fp32 and bf16: unique
parameters (measured), checkpoint bytes (measured), weight memory, peak VRAM, prefill throughput
at 512/2048/4096 prompt tokens, decode tokens/s, time to first token, end-to-end latency on a fixed
64-token generation, and KV-cache bytes per token
(`2 · n_layers · n_kv_heads · head_dim · bytes`: 28 blocks → 114,688 B/token in fp32; 18 blocks →
73,728 B/token). The KV reduction is 18/28 of the teacher's and follows from depth alone.

## 8. Gate G4 (registered in SMALLVD001_PREREG.md §7)

PASS when surgery-only damage is **measured and recorded separately** for A1, A2 and A3 on the
HOLDOUT panels, AND on every instrument whose shim replay of A0 is exact. G4 is a gate about
*measurement*, not about the size of the damage: a heavily damaged 18-block student PASSES G4 if its
damage is recorded.
