# CONFIRM-002 stored answerable audit

This is descriptive development analysis of a spent bank, using stored
generations and the exact frozen scorer. It changes no historical score, scorer,
receipt, bank, or claim. Audit/model-generation calls: **zero**. No significance,
equivalence, noninferiority, or new confirmation analysis was performed.

All 445 bank IDs are unique and covered exactly once by each supplied arm
(SPARSE, TH25, and the co-stored THFULL); the compared answerable subset contains
109 complete pairs, all in `arithmetic_short_answer`. Row types/families, the full
runtime records, and SPARSE/TH25 EOS lists agree. Re-scored flags, receipt summaries,
and the historical aggregate have **zero mismatches** for the compared arms.

| Frozen correctness outcome | Pairs |
|---|---:|
| Both correct | 35 |
| Both incorrect | 72 |
| SPARSE correct, TH25 incorrect | 2 |
| SPARSE incorrect, TH25 correct | 0 |
| SPARSE / TH25 attempts | 109 / 109 |
| SPARSE / TH25 correct | 37 / 35 |

Source-correct loss is 2/37; source-incorrect gains are 0/72. These denominators
describe the observed bank. The aggregate difference of -2/109 cannot identify
its cause without the pairs and does not establish paired significance.

The two losses are fully localized:

| ID suffix | Prompt operation | Reference | SPARSE | TH25 | Text evidence |
|---|---|---|---|---|---|
| `arithmetic_short_answer_00292` | Average of 25 and 19 | `22.` | `22` | `44` | Arithmetic error: TH25 returns the sum rather than the average. |
| `arithmetic_short_answer_00370` | 406 units less 97 | `309.` | `309` | `309 units.` | Correct numeric value with extra unit wording; frozen canonical exact match rejects it. |

Both full IDs begin `HSKILL3S:rineya_srb_v1_`. They occur at bank lines 78 and 99;
the corresponding prediction records begin at lines 729 and 918 in both
`B_H_SPARSE.json` and the TH25 section of `B_H_COMPOSE.json`.
Calling the second row a scoring/format artifact describes the text; it does not
re-credit that row or amend the historical scorer. There is one evident numeric
arithmetic loss and one evident numeric-value-preserving exact-match loss among
the two frozen-score losses. This narrow inspection is not a semantic grade of
all 109 responses.

Of 109 pairs, 75 have identical raw output and 91 have identical canonical first
lines. Sixteen changed outputs differ only in punctuation handled by the frozen
normalization. All answerable outputs are single-line and nonempty; none reaches
the 48-token cap (observed maxima: SPARSE 17, TH25 16). Strict first-line equality
scores 0 versus 8, while frozen canonical correctness is 37 versus 35, because
references carry punctuation. Strict equality is a diagnostic, not a replacement
endpoint. The frozen rule folds case/whitespace and strips trailing `. ! ?` from
the first line; it does not strip unit words. Attempt scoring uses the historical
abstention predicate on that first line.

## Evidence provenance

The original evidence is under
`C:/Users/clclc/BusinessProject/ChatBubble-titan-model-soup-mini-v1/experiments/composition_confirmation_v2/`.
That directory is currently untracked in its dirty soup checkout. It was read
without modification. The committed copy is under the same relative path in
`C:/Users/clclc/BusinessProject/ChatBubble-post-training-Mimo/` and at integration
`03db218ceebc61a2bc04f2482cf7e094045e01c1`.

The result was recorded in commit
`1d54b93fd2a9f8e0fe6e98af2e2c7fae17442275` on 2026-09-25. The bank, both generation
receipts, aggregate result, preflight, and chain log have no content changes
between that result commit and integration. All 18 files in the copy manifest
are byte-identical across original, centralized, and isolated copies. Original
`run_chain.log` is committed as `RUN_CHAIN_LOG.txt`.

| Source | SHA256 |
|---|---|
| `HSKILL_SOUP_V3.jsonl` | `686df45b94fa000be8df8d15e1f026a7f74e22deadfa4aacda96f5c6244c8ad5` |
| `B_H_SPARSE.json` | `57837957dd1eefe8e3b5fdcb2930828b3dae82ace82fff616cb6d7d086a92899` |
| `B_H_COMPOSE.json` | `123c405f7297181db8bd340fe1dd2ce15b5b70e1614b9aab9a62fb26842bec7b` |
| Frozen `boundary_rule.py` | `416cfb6aec8aa28240752d3db2fc1b8af85115c4962a873582765c009adc3bf0` |
| `B_PREFLIGHT.json` | `ab88924c1c8038e3622bc722686bdcefff35d732da37a2ef286b0d94f311a9e2` |
| `CONFIRM002_RESULT.json` | `a5cf0a6dd5714d5de82738d0d9218d71c709cca7736526034237cacdd1f5e401` |
| `RUN_CHAIN_LOG.txt` / original `run_chain.log` | `d5b47e7480cf983f5a7ddbe23f4181cb052c982e5ecdc1572aa6612442cb02ec` |

The frozen rule is at
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002/boundary_rule.py`.
The helper verifies its full hash before executing those exact bytes, with no
import search or bytecode write. The historical runtime is Python 3.11.15,
torch 2.6.0+cu124, transformers 5.14.1, FP32, greedy, 48 new tokens, thinking
disabled, TF32 off; EOS is `[151645, 151643]`. The new audit only uses the installed
CPython 3.11.15 standard library, with `-B -I -S` and no torch/transformers import.

Historical `run_stage_b.py` binds SPARSE to
`experiments/titan_opallo_skill_vector_v1/arms/SPARSE` and TH25 to
`experiments/composition_confirmation_v2/arms/TH25`, records preflight hashes, and
then calls `score_boundary.py` for the stored outputs. SPARSE's build/preflight
weight hash is
`6ae85f5bdab9437edc06b20716e6e7008d8aa402ec95a0a65ebc7bf8da1e94ed`;
TH25's is
`a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567`.
The TH25 deletion tombstone retains that hash and states the confirmation is
complete and its read budget spent. No weights were restored or reconstructed.

Generation receipts omit direct weight, tokenizer/template, and harness hashes.
The driver, preflight, build receipts, and chain log provide historical
attribution, rather than a fresh end-to-end verification of those omitted
bindings. Raw token IDs and actual stopping causes are also absent. The bank is
a small arithmetic competence proxy; this audit cannot establish broad capability
or resolve dependence/generalization from one spent bank.

The SPARSE `BUILD.json` is present only in the original soup checkout at
`experiments/titan_opallo_skill_vector_v1/arms/SPARSE/BUILD.json` (SHA256
`b14000efe4fa9fdd371224914e0d32da8d8d2e7cc0db1a472e147ee7f07fca41`);
it is absent from the centralized/integration copy. Its hash and preflight match
are included in the derived provenance manifest. This does not block the stored
output audit, whose bank, generations, and scorer are available and pinned.

## Local implementation and baseline state

The new isolated branch is
`codex/confirm002-paired-correctness-audit-20260930`, based on integration
`03db218c`, at
`C:/Users/clclc/Documents/Codex/2026-09-30/task-2/ChatBubble-confirm002-paired-audit`.
Only `experiments/confirm002_paired_audit_v1/` is added. The prior CLARE worktree
and commit `669ba121` were preserved and not duplicated. No push, PR, merge, or
deployment occurred. See `README.md` for reproducible CPU-only commands and
`SOURCE_PROVENANCE.json` for exact source paths, hashes, and copy checks.

Validation: all **30 synthetic standard-library unittest tests passed** in
0.089 seconds. They exercise all four correctness cells, reordered IDs,
full-bank/boundary coverage, duplicate/missing/extra IDs and arms, incompatible
bank/scorer/runtime/EOS/metadata, missing provenance, stored flag/summary
disagreements, empty source-correct denominators, first-line/format diagnostics,
strict JSON, scorer pinning, malformed historical-result shapes, and create-only
output preservation. The real stored-output audit separately passes with no
flag, summary, or historical aggregate mismatches. A static review found one
malformed-result error-path defect; dictionary guards and regression tests
resolved it before the final run. No broader/native test suite was invoked.

Before CPU validation, relevant process/resource checks showed no Python,
torch, accelerate, or native research job, approximately 4.18 GiB free host RAM,
and GPU utilization 0% (836 MiB resident) at 2026-09-30 11:15Z. Process inspection
needed read-only sandbox escalation; no process was stopped. Initial checkout
status and worktree lists were inspected and existing dirty edits preserved.

Centralized MiMo is now at `e6f5f8905e0a73b40f6787358c1a982fff8ed3c7`, following
the `d269f649` baseline. Later commits `e2a67abc` and `e6f5f890` register short
greedy capture and freeze DEV inputs. The existing untracked capture scripts and
receipts were left intact. Latest read-only receipts at 2026-09-30T11:01:33Z are
`CAPTURE_SHORT_V1_DESIGN_REVIEW.json` (`PROSPECTIVE_DESIGN_REVIEW_NO_NATIVE_GRANT`)
and `LANGUAGE_GRADER_AUTHORITY_AUDIT_V1.json`
(`QUALIFIED_LANGUAGE_GRADER_AND_LABELS_MISSING`), under
`experiments/post_training_mimo_centralized_v1/receipts/`. Native launch, E1, and
language grading remain false. No later native/capture result was found in the
bounded inspection. Historical CE512 eight-check/four-update qualification and
all 5,120 TRAIN joins remain saved passes; they were not rerun.

SupaExperiment's dirty main checkout and research worktrees were preserved.
RineyaArchitecture is clean at `2d50f33c`; its stored native-random mechanical
receipt establishes structural checks, not pretrained quality. Repository
instructions and relevant local memories were read. No unrelated application or
game source was examined or changed.
