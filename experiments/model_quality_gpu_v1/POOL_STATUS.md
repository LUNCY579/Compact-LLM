# MQ data and execution status

**Authoritative continuation: [MODEL_QUALITY_STATUS.md](../../MODEL_QUALITY_STATUS.md).** The previous status is [preserved verbatim](../../docs/history/model-quality-audit-2026-09-15/experiments/model_quality_gpu_v1/POOL_STATUS.md).

## Existing pool: development use only

`mq-data-001` is marked superseded locally. `mq-data-002` supplied the punctuation-brittle41-scoreable DEV first pair. `mq-data-003` corrected punctuation variants and balanced polar golds4YES/4NO; it supplies51scoreable/64DEVrows and the same2,000TRAINcandidates. Plans003/004/005 select the same238rows with selectionseed17, vary order17/23/29, and compile24,910processed/24,672whole/1,944response targets in13updates. Full hashes and byte-preserved raw records are in the [audit archive](../../docs/model-quality/audit-2026-09-15/historical-mq-archive-manifest.json).

Historical named operator reviews exist and bind those exact plans. Their source-rights and semantic/generator statements are attestations; the compiler's exactID/family/prompt/digit-blind audit does not independently establish them. The corpus is selected from v3single-turn rows and excludes historical panel/splits_large prompt collisions. It is not the originally proposed800grounded/600task/400preservation/200contract mixture.

## Newly audited limitations

- No explicitUNKNOWNtask or matched answerable/unanswerable pairs. The builder's generic `answerable=True` field does not establish epistemic answerability for natural missing-information rows.
- Forty-three of 64 DEV rows share underlying `source_base_template_family` with the full 2,000-row TRAIN pool; 25 of 64 share them with the 238 consumed rows (22 of 51 scoreable DEV rows). Declared v3 family IDs are distinct. No exact source-index overlap was found. Context/name substitutions can leave the same substantive task; see the [diagnostic](../../docs/model-quality/audit-2026-09-15/corpus-payload-audit.json).
- DEV was redesigned after first-pair outputs. It is development, not confirmation. Literal short-reference variants are not a semantic correctness rubric;51scored rows do not provide a broad quality card.
- The later repeated cells test ordering only, and all historical checkpoint directories are absent. Memory is not the only unresolved prerequisite.

## Next bank

Follow the proposed **QTB-001** composition and source/solver/shortcut/manifest gates in the [memo](../../MODEL_QUALITY_STATUS.md#proposed-qtb-001-corpus-specification--not-yet-built). It is not yet built, frozen or independently approved. Preserve old banks/reviews and create new ones; do not reinterpret their hashes or reuse their approvals. Freeze actual underlying generator/scenario partitions, paired selection and all token/update accounting before training. Extended DEVcapacity requires a separately registered scientific successor.
