# Research Questions and Decision Status

The primary hypothesis is that independently evaluated parameter-efficient experts can use hard next-token plurality to improve a shared short-response model while every expert sees the same accepted prefix.

## Primary confirmatory questions

1. Does exact hard next-token voting improve the frozen backbone?
2. Does it improve the best individual adapter, not merely the mean adapter?
3. Does it outperform standard probability/logit fusion?
4. Is any improvement compatible with the short-response objective and worth its latency/storage cost?

The frozen 32-case sealed packet supports question 1, rejects question 2 for this committee, and rejects a hard-voting advantage over probability mean. It is too small to establish a general scaling law.

## Mechanistic and secondary questions

- Does specialty diversity help? Partially tested with official broad, anchored, general, grounding, procedure, abstention, and social LoRAs.
- Does committee size help? Tested at K=1, 2, 3, 5, and 7 on a paired balanced DEV pilot.
- Does disagreement predict error? Tested with AUROC, AUPRC, Brier score, ECE, and reliability bins.
- Can disagreement allocate compute? Tested separately by a K=3-to-K=7 adaptive consultation pilot.
- Does EOS need special handling? Ordinary EOS is primary; strict majority, mean-confidence threshold, and minimum-token policies are ablations.
- Does hard voting resist overconfidence? Tested with an artificially sharpened weak expert.
- Does sequence-level consensus explain the result? Tested with an independent-candidate edit-distance medoid.
- Does oracle evidence erase differences? Tested with official oracle-RAG prompts and functional/grounding metrics.

## Explicitly unresolved questions

- Random-seed diversity was not isolated: the official compatible pool has no matched same-task seed family.
- Mixed LoRA/DoRA/PiSSA/AdaLoRA quality was not tested: software support was probed, but no matched trained adapters exist.
- A second backbone was not evaluated with a fair matched committee. D_greedy is a dense merge with different adapter lineage.
- Real retrieval, memory reconstruction, a learned router, and a trained verifier remain future experiments.

These absences are recorded as limitations, not silently treated as negative findings.
