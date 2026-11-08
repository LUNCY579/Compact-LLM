# Source scope — 16 September 2026

## Supplied commission

`ORIGINAL_COMMISSION.md` is an unchanged copy of the attached `Pasted markdown(20260916-174409).md`. It requests 6,000 long-context retention rehearsal examples, actual target-tokenizer measurements, independent answer checking, controlled composition, and evaluation-exclusion/shortcut audits. This package does not fulfill the corpus-generation deliverable.

## Connected repository reads

Repository: `LUNCY579/ChatBubble`.
Observed branch: `codex/dataset-retention-audit-v1`.
Observed revision: `dbcc3d6e81898d7d637f181948e3971db511433a`.

Relevant files read at that revision:

- `REHEARSAL_COMPOSITION.md`
- `REHEARSAL_BLEND.md`
- `DATASET_REQUIREMENTS.md`
- `experiments/model_quality_gpu_v1/source_lock.json`
- `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/tokenizer_config.json`

The source lock supplies the four tokenizer hashes in `TARGET_TOKENIZER_LOCK.json`. Its historical source revision is `a36de96b30eb03265db3a18e464a2e2a23ee67bf`; the lock was inspected at the current audit revision above. The lock also includes model/config identities, but the collector deliberately does not require or copy model weights.

The rehearsal reports are historical, source-reported model measurements. This turn did not rerun them, independently adjudicate their statistical interpretations, or infer a universal rehearsal threshold. The commissioned data remains a hypothesis-driven retention intervention, not a demonstrated remedy.

A directory listing located `Rineya_Long_Context_Demo_v1` and its `data` directory. A listing does not constitute a corpus-wide exclusion audit. Complete locally materialized prompt banks, entity/renderer inventories, and a declared exclusion scope remain necessary; any quarantined or sealed materials retain their existing access restrictions.

The Hugging Face connector confirmed that `Luncy1978/rineya-published-cores` exists and is private. Available metadata access did not materialize tokenizer files. No account credentials, signed download URLs, or private model weights are included in this package.

## Execution boundary

The working container did not have the target tokenizer assets or a usable installed Transformers/tokenizers runtime. Attempts to obtain public assets/install the runtime failed. Reading tokenizer configuration text or a published asset hash is not equivalent to executing the tokenizer on all corpus rows.

Only the standard-library input collector's synthetic mechanics tests were executed. The optional native fixture probe must run in an already qualified local environment and was not executed here. Six fixture encodings, even if collected later, will not establish the token profile of an ungenerated corpus.

No repository write, branch creation, model invocation, training update, or corpus generation was performed in this turn.
