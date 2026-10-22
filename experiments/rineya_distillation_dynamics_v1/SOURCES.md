# Primary sources

1. Julianna Piskorz, Antonin Berthon, Mihaela van der Schaar.
   **On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics.**
   arXiv:2609.35259v1, submitted 2026-09-28; preprint, under review.
   https://arxiv.org/abs/2609.35259
   https://arxiv.org/html/2609.35259v1
   Consulted for the independent rollout/KL/learning-rate design, detached-sampling
   semantics, full-vocabulary objectives, and limits of sparsity/forgetting claims.
   No author benchmark, dataset or implementation is copied into this experiment.
2. Hugging Face **Transformers v4.57.1 model documentation**.
   https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model
   Consulted for local-only safetensors loading and missing/unexpected-weight
   diagnostics. A documented API does not qualify the exact Rineya runtime.
3. Repository primary implementations inspected through GitHub are identified by
   path, ref and blob SHA in `RECONNAISSANCE.md`.

Paper conclusions are motivating hypotheses, not Rineya measurements. Reported
native efficacy must come from separately qualified, registered runs; fixture
scores and passing unit tests cannot substitute for that evidence.
