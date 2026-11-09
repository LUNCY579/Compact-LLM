# Source-anchor competence passed; cache reconstruction check failed

Measured source anchors:144/168 correct (binding110/112, arithmetic25/28, time9/28).
Binding includes54/56 correct answerable and56/56 literal UNKNOWN on missing rows.
These are anchor-selection observations, not evaluation/confirmation outcomes.

No training, new-skill evaluation, or old-development evaluation ran. The strict
teacher-head reconstruction check failed at3.9577484130859375e-5 versus1e-5.
The reference projected the full prompt+answer matrix before selecting answer
positions, whereas the cache projected only answer positions. Different GEMM row
counts can produce different FP32 rounding. Pilot004 aligns the native HF reference
using its supported logits_to_keep=position_indices argument. Tolerance is unchanged;
the fix will itself be tested. Preserve the source snapshot and failed receipt.
