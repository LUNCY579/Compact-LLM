# Projection Policy Freeze 1

Frozen after the architecture audit and before construction of any projected
checkpoint.

The audit established that Qwen3-0.6B, Supa, and Qwen3-4B use tied input/output
embeddings. Qwen3-8B is untied and serializes a distinct `lm_head.weight`.
Mapping 8B into a tied 0.6B target therefore cannot preserve both matrices.

The deterministic P1 baseline uses `tied_projection=embedding`: project the 8B
input embedding into target width and use that single matrix for both input and
tied output roles. Two controlled secondary variants are predeclared:

- `output_head`: project the 8B output head and tie it for both roles;
- `mean`: project both matrices in FP32, average them, and tie the result.

No policy is presented as semantically lossless. Results must report the policy,
and 4B/8B comparisons use the same named policy where defined. Embedding/output
head variants are not selected on sealed data.
