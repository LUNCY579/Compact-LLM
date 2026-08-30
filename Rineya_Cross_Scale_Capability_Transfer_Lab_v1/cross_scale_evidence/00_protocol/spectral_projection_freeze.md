# Target-Basis Spectral Projection Freeze

Frozen after all P1 arms and the 4B/8B embedding-aligned basis arms failed the
functional gate, and before constructing or evaluating a spectral checkpoint.

The remaining training-free arm does not assume that donor and target singular
vectors share coordinates. For each corresponding 2D decoder matrix:

1. retain the official Qwen3-0.6B target singular vectors;
2. estimate the top eight normalized donor singular values by deterministic
   randomized SVD on the D2-mapped donor layer;
3. impose that top-eight spectral energy profile in the target basis;
4. rescale the residual target spectrum to preserve the target Frobenius norm;
5. keep target embeddings, tied output head, RMS norms, and all 1D tensors.

The confirmatory exploratory configuration is `P4-D2-W4-rank8-strength1.0` for
both donors. It uses no benchmark/calibration text. A projected checkpoint must
still pass the same coherence/EOS gate before any delta or merge is permitted.
