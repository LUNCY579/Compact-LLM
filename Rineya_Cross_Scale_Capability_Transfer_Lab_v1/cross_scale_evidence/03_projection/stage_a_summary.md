# Projection Stage A

Decision: **P1_NO_GO**

All 8 arms passed tensor/key/parameter/finite-logit checks; 0 passed the functional coherence and EOS gate.

| Arm | Structural | Functional | EOS | Loop | Smoke output |
|---|---|---|---|---|---|
| P1_D1_W2_4B_embedding | True | NO_GO | False | False | `owie般根基根基根基根基根基根基` |
| P1_D1_W2_8B_embedding | True | NO_GO | False | True | ` Gard Gard Gard Gard Gard Gard Gard Gard` |
| P1_D2_W2_4B_embedding | True | NO_GO | False | True | `港区 haute港区 haute港区 haute港区 haute` |
| P1_D2_W2_8B_embedding | True | NO_GO | False | True | ` regular regular regular regular regular regular regular regular` |
| P1_D2_W3_4B_embedding | True | NO_GO | False | False | ` addCriterion点燃点燃点燃点燃ifaxifaxifax` |
| P1_D2_W3_8B_embedding | True | NO_GO | False | True | `相伴相伴相伴相伴相伴相伴相伴相伴` |
| P1_D3_W2_4B_embedding | True | NO_GO | False | True | `pansepansepansepansepansepansepansepanse` |
| P1_D3_W2_8B_embedding | True | NO_GO | False | False | `析析析析析析 gases gases` |

W1 prefix slicing is numerically identical to W2 for these shapes: target Q/K/V dimensions are exact multiples of the shared 128 head dimension, so W2 merely verifies that W1 boundaries are complete heads.

Depth choice and magnitude-ranked structured subsets did not restore a coherent donor-only model. The tested independent-model coordinate bases are not compatible under deterministic intersection/truncation.

No P1 projection is authorized for MERGEDEV or for constructing an absolute/task-vector merge.
