# Training and Evaluation Guide

## Recommended first SFT

Use `data/sft/rineya_v3_recommended_combined_124000.jsonl` for one controlled QLoRA epoch from the accepted v2-trained teacher checkpoint. Preserve the parent checkpoint and write the new adapter to a separate directory.

If adding the approved RAG training corpus, append it exactly once and shuffle deterministically. Never add RAG DEV/TEST or chat DEV/SEALED files.

## DPO

Apply `data/dpo/rineya_v3_preferences_train_16000.jsonl` only after an SFT candidate passes the frozen chat and RAG DEVs. Train DPO as a child artifact with rollback available.

## Required evaluation comparisons

- current accepted teacher versus v3 SFT;
- v3 SFT versus v3 SFT+DPO;
- chat aggregate and per-class tails;
- task adequacy and prompt-function match;
- supported, abstention, and procedure correctness under Oracle and real RAG;
- median, P90, and collapse-frequency statistics;
- relationship continuity, trust/secrets, later-correction, roleplay, and evidence-grounded audit subsets.
