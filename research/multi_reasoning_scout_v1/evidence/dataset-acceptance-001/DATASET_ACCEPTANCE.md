# Dataset P0 acceptance: STOP — not accepted

The user-confirmed `Rineya_LCR_Full_v1_6000` candidate is not a qualified training input. **P0 fails; P1/P2 stay closed and no training ran.** Fresh checks resolve the missing-tokenizer question for train/dev and expose 41 actual token-band violations. Structural isolation, nontrivial abstention and external exclusion remain unqualified.

Audit scope was exactly 4,800 train and 600 dev rows. Holdout, `data/all.jsonl`, sealed, ANY_READ and confirmation payloads were neither read nor hashed. Producer aggregate metadata and manifests were read; their holdout claims remain unverified. No whole-package verifier or test command that could open holdout was run.

| Requirement | Fresh evidence | Decision |
|---|---|---|
| Permitted-file integrity | Train/dev data, messages-only SFT and worlds match exact manifest hashes; all explicitly read metadata/code hashes match | PASS scoped |
| Row and answer contract | 5,400 unique scenarios; exact SFT projections; system/user/assistant; one-word answers; literal UNKNOWN | PASS train/dev |
| Answer rederivation | 5,400 answers and absence reasons, 75,073 parsed facts, zero disagreements | PASS train/dev; shared author/grammar limitation |
| Native tokenizer | All four locked Supa tokenizer assets match; Transformers 4.57.1, tokenizers 0.22.2, Qwen2TokenizerFast | PASS tokenizer only |
| 2,048 window / 128 reserve | Max rendered prompt 1,470; max full chat 1,476; zero violations, filtering or truncation | PASS train/dev |
| Context length bands | **41 short-band contexts exceed 600 native tokens: 4 train, 37 dev** | **FAIL** |
| Basic split isolation | Zero train/dev ID, group, entity, template-ID, exact/NFKC/digit-blinded whole context/question overlap | PASS narrow scope |
| Strict structural/template isolation | **392 shared name-blind sentence forms**; coarse typed-graph count signatures disjoint | NOT ESTABLISHED |
| Answerability balance | UNKNOWN: 1,440/4,800 train, 180/600 dev; four absence reasons equally allocated | MEASURED |
| Question-only screen | Train-fitted unigram naive Bayes: dev balanced accuracy 47.90%; train 50.44% | No positive shortcut finding from this finite screen |
| Surface screen | Train-selected `context_words` threshold: dev 49.60%, train 52.48% balanced accuracy | Finite screen only; no significance claim |
| Nontrivial abstention | **Cheap selector plus explicit validity rule: 87.5% balanced accuracy in both splits** | Qualification not established |
| Conditional balance | Producer reports max within-band relative deviation 5.263% against literal 3%; fresh permitted marginals reproduced | Full gate not reexecuted; producer failure unresolved |
| Protected-bank and frozen-scout contamination | No qualifying blind exclusion receipt | UNVERIFIED, never zero |
| Holdout/all-row inventory and token census | Deliberately not accessed | UNVERIFIED |

The 41 band failures are enumerated in `DATASET_ACCEPTANCE.json` and `native_train_dev_rows.json`. Examples include `lcr_v1_latest_verified_870ce2ce` (611 context tokens) and `lcr_v1_latest_verified_36188298` (614), both assigned to short. The literal inclusive band convention was used; two shared-endpoint rows are explicitly flagged and do not explain these failures.

The independent solver receives only model-visible user text before comparing its result with the stored answer/reason. It is a separate supplied implementation with shared authorship and grammar, not independent human certification. First/last entity, copy-most-recent, majority and UNKNOWN baselines were recomputed. Producer logistic-regression/nearest-neighbor attacks and every detailed evidence-deletion/difficulty check were not rerun; no stronger completeness claim is made.

Publish a new immutable dataset revision that corrects the 41 native band violations without truncation, resolves the strict structure/template and easy-abstention failures, and satisfies the balance rule or a prospectively authorized rounding convention. Obtain blind, aggregate exclusion receipts against all protected Rineya banks and the frozen scout, plus authorized holdout integrity/token/isolation receipts. Then rerun P0 before freezing any training recipe. This audit does not modify the corpus or its producer receipts.

Dataset directory: `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/Rineya_LCR_Full_v1_6000`.

Manifest SHA-256: `65995575f13187ef78464e4120d1768696dd9114c7dfd8a379ed3ca6e28b266d`.

Attached specification SHA-256: `041f50d4a35ecf50dc255ade4a725b5f18e23db314dfb0cbf1440c79ceecbf21`.

Audit script SHA-256: `23fb08454e4b8734f6596d1d6b9e17dc937b9793bd3570bda4453dd8f7e4d4f3`.

Execution: 46.27 seconds, zero model calls and zero optimizer/training steps. Full explicit input hashes, runtime, source branch/parent and output hashes are recorded in the JSON companion and `partial_audit_receipt.json`. Software audit completed; native tokenizer execution passed; model-quality, confirmatory and release claims remain false.
