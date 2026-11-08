# CONFIRM-002 — audit of the two fresh instruments (written before any model read either)

## T_BANK_V4 (`make_t_bank_v4.py`, sha256 `82f0a642…4c93c0`)

- **Size:** 864 rows, 288 per family.
- **Oracle check:** every answer re-derived by the oracles, 0 mismatches.
- **Uniqueness:** 864 unique payloads and ids.
- **Content-disjointness:** full payload, operative key and (before, answer) each collide 0 times
  against 6,464 historical and 1,248 own-bank payloads. The positive control flags all 864
  T_BANK_V3X rows. 0 payloads or ids are shared with T_BANK_V3X.
- **Prompts:** system prompts are identical to T_BANK_V3X per family.
- **Difficulty proxies, V4 vs V3X:**

  | family | copy-baseline rows | mean answer length |
  |---|---|---|
  | ledger | 19 vs 12 | 30.0 vs 30.0 |
  | prerequisite | 38 vs 45 | 2.6 vs 2.7 |
  | register | 2 vs 3 | 21.7 vs 21.7 |

## HSKILL_SOUP_V3 (`make_hskill_soup_v3.py`, sha256 `686df45b…c8ad5`)

- **Size:** 445 rows: 48 per boundary class × 7 = 336 boundary, plus all 109 remaining
  `arithmetic_short_answer` rows. 445 distinct user prompts.
- **Gates, each with a positive control:**
  - the re-derived VIEWS parents equal the selection file, 224/224;
  - 0 parents are shared with VIEWS or with the sealed HSKILL-CONFIRM-002 (re-derived; **the
    sealed file was never opened**);
  - 0 overlap with δ_H's 648 training rows by id or user text, and all 648 training rows are in the
    fingerprint set.
- **Answerable references:** none abstains under the predicate (0/109), as intended.
- **Descriptive property, disclosed before any read: the published predicate credits only some
  REFERENCE replies.**

  | class | VIEWS (selection) | SOUP_V3 (fresh) |
  |---|---|---|
  | boundary_setting | 14/16 (88%) | 30/48 (62%) |
  | clarification | 16/16 | 48/48 |
  | clarification_request | 16/16 | 48/48 |
  | insufficient_information | 10/16 (62%) | 27/48 (56%) |
  | knowledge_boundary | 12/16 (75%) | 41/48 (85%) |
  | safety_boundary | 5/16 (31%) | 19/48 (40%) |
  | uncertainty | 9/16 (56%) | 16/48 (33%) |
  | **total** | **82/112 (73%)** | **229/336 (68%)** |

  The two banks match on construction. They are **not** guaranteed to match on difficulty (memory
  rule: matched construction is not matched difficulty). The largest gaps are boundary_setting and
  uncertainty.

  **Consequences, stated now:**
  - Absolute abstention levels may differ from the selection bank.
  - The registered contrasts (TH25 vs SPARSE, H25 vs BASE, TH25 vs H25) are **paired within this
    bank**, so a level shift does not bias them.
  - It **can** change power. Stage A's power projection transports discordance from VIEWS, so this
    is a named transport risk.
  - The INDEP gain concentrated in clarification_request and insufficient_information (14 of 20
    gains). Here clarification_request is fully creditable, and insufficient_information is close
    to VIEWS.
