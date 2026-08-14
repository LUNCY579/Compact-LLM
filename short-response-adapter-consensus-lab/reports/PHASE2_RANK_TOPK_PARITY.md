# Rank top-k implementation parity

Status: **RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED**

The memory-safe stable partial top-k returns the exact prefix of the original stable full-vocabulary descending sort. The check covers every 7-token vector over a three-value grid at every rank depth, randomized/tie-stressed matrices, and the model's realistic 151,936-token vocabulary at K=7 experts and depth 5.
Because rank points are a deterministic function of these identical per-expert rankings and the unchanged secondary score uses the same raw log probabilities, each accepted token is identical. Induction on the shared accepted prefix then gives an identical full trajectory.

- Exhaustive vector/rank checks: **15,309**
- Randomized row/rank checks: **4,096**
- Realistic-vocabulary expert rows: **7**
- Original full-sort index allocation proxy: **8,508,416 bytes**
- Partial result index allocation: **280 bytes**

The post-example CUDA cache release occurs outside generation timing and only returns unused allocator blocks; it cannot change logits or decoded outputs.
