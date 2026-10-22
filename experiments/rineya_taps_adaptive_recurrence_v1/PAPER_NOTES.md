# Paper audit: TAPS

Source version: arXiv:2609.36653v1, submitted 2026-09-29. Boyuan Wang,
Chengyao Yu, Jiaxi Ren, Hongxin Wei, Bingyi Jing, Yuxin Tao.
Audit date: 2026-10-01. The full HTML, main text and appendices A-D were read.
PDF equation pages were inspected; subsequent PDF screenshot requests failed
with cache-miss errors, so remaining tables were read in the full HTML.

## Primary sources and official implementation

- https://arxiv.org/html/2609.36653v1 (Sections 3-5; Appendices A-D, Algorithm 7)
- https://arxiv.org/pdf/2609.36653
- https://huggingface.co/papers/2609.36653 (discovery only; AI summary not authority)
- https://huggingface.co/ElvisWang111/TAPS-TRM-Sudoku-Codesign
- https://huggingface.co/ElvisWang111/TAPS-TRM-Maze-Codesign
- https://huggingface.co/ElvisWang111/TAPS-TRM-Sudoku-Codesign/blob/main/trm.py
- https://huggingface.co/ElvisWang111/TAPS-TRM-Sudoku-Codesign/blob/main/all_config.yaml

Official code DOES exist in the author checkpoint repository: `trm.py`,
`losses.py`, and configuration files. The complete Sudoku `trm.py`, including
`_adaptive_l_step`, reset/carry and forward paths, was inspected, together with
its configuration. Code-file history identifies revision `f742461`; inspected
Sudoku/Maze repository tips were `3712df8` / `1f1b219`. These are observed short
revisions, not asserted full immutable digests. Maze file retrieval and
`losses.py` retrieval failed; their contents were not inferred. No separate
official GitHub TAPS repository was located by title/identifier/author searches.
The available release is not a verified complete experiment/fused-kernel bundle.
No checkpoint was downloaded and no author code is vendored or executed here.

## Independently implementable derivation

For one sequence, let h be a token-by-channel state, u fixed conditioning, and
F(h;u) the frozen recurrent proposal. Define the RAW update d=F(h;u)-h, not the
previously scaled movement. Ordinary recurrence takes the proposal; TAPS takes
h + alpha*d. Each arm must recompute proposals on its own current state.

For observation index t=1,2,..., with initially zero biased moments:

    m_t = beta*m_(t-1) + (1-beta)*d_t
    v_t = beta*v_(t-1) + (1-beta)*mean(d_t**2)
    c_t = 1-beta**t
    M_t = m_t/c_t; V_t = v_t/c_t
    P_t = mean(M_t**2)
    S_t = max(V_t-P_t, 0)
    B_t = (P_t-gamma*S_t)/(P_t+gamma*S_t+eps)
    alpha_t = clip(1+rho*B_t, alpha_min, alpha_max)

Warmup observations update moments but use alpha=1. Expanding the recurrence
gives weights w_j=(1-beta)*beta**(t-j)/(1-beta**t), whose sum is one. Thus S is
exactly weighted mean squared deviation from M in real arithmetic; clipping
only handles floating-point cancellation. This is Algorithm 7's normalized
EMA, implemented using equivalent biased moments plus correction. It is NOT
Adam optimization or coordinate-wise division of d by sqrt(v). Reduction is
within each sequence, never across the batch. Large persistent moments raise
the multiplier; fluctuation lowers it. Zero update gives neutral balance.

The author code confirms FP32 moment arithmetic, eps=1e-12, nonnegative variance,
per-example carry/reset, exclusion of puzzle-ID prefix positions from statistics,
and scaling of the complete proposed low-level state. TRM carries low/high states
z_L/z_H; its low proposal is conditioned on z_H and input embeddings. The high
update remains separate. Controller history can cross TRM outer cycles until a
sample resets; that is different from resetting each autoregressive LM forward.

## Defaults are host-specific, not universal

Appendix C.1's Qwen intermediate-block configuration: beta=.8, gamma=1.5,
rho=.275, bounds [.8,1.2], first-stage warmup. Crucially its actual Euler step is
eta/3 for three stages on layers 15-18. It is NOT full repeated-block recurrence.
The published TRM co-design configuration instead uses gamma=1, rho=.2, warmup=2,
and beta=.8 with the same bounds. Its Sudoku YAML has batch 768 and max steps 16;
Table 6 describes a batch-384 training experiment. Do not conflate released
checkpoint metadata with every paper experiment. This delivery uses explicitly
named settings; it does not claim these are optimal Rineya settings.

Appendix C.4 studies refresh q in {1,2,4}; coarser updates do not guarantee speed.
This implementation observes every update, refreshes alpha at the first
post-warmup step and each q steps thereafter, and holds it in between. That
explicit convention is a Rineya adaptation; exact author q>1 parity is unverified.

## What the theory does and does not imply

With terminal task loss L, the exact sensitivity is
`dL/dalpha_k = <G_(k+1), d_k>`, where G is a downstream adjoint. Weighted averaging
and centering split its negative into signed task-dependent persistent and
fluctuating contributions. Those oracle quantities are NOT the nonnegative P/S
observable energies above. Theorems require conditional comparison bounds linking
the two, bounded temporal mismatch, and sufficient directional margin. Finite
steps additionally pay a curvature term proportional to (alpha-1)^2. A telescoping
hybrid-policy argument gives cumulative expected-loss/target-depth conclusions
only under those assumptions. No generic correctness, convergence or speed
promise follows from aligned updates alone.

Failure hypotheses: coherent movement toward a wrong answer, useful alternating
updates, near-zero existing gates, stale EMA, changing conditioning, padding
contamination, BF16 cancellation, unsafe extrapolation, readout distribution
shift, and controller overhead on a very short trajectory. Stable-wrong and
zero-update fixtures explicitly guard against a flattering interpretation.

## Inference versus co-design

Inference-only changes no weights. Co-design adds
`mu * sum(relu(S-kappa*P))` to task plus ACT objectives while training through the
controller; Appendix C.2 reports mu=.001 and kappa=4. That is a distinct training
experiment, not a consequence of plugging in the inference scheduler.

Paper-reported outcomes are mixed across hosts/controllers. In Table 4 the
Qwen block unit baseline averages 59.22%, while TAPS-Adam averages 58.71% and BB
59.37%. These are PAPER-REPORTED, not Rineya or reproduced results. Selective
positive examples would obscure the most relevant counterexample. Inference-only
and co-designed wall-clock gains must not be mixed. Comparisons with only .8 and
1.2 fixed factors do not settle the stronger calibrated-fixed/oracle question.

## Adaptation labels

Reproduced mathematically: Algorithm 7 moment/energy/balance rule at q=1.
Not an empirical reproduction: no paper checkpoint or benchmark execution.
Rineya adaptation: frozen full-prefix ARMR block, intact first pass seeds history,
full residual rather than eta/3, exact unit-step bypass, diagnostic safety checks,
restricted DEV evaluation, optional masked statistics and held-refresh convention.
Hypothesis only: trajectory-dependent scaling improves native Rineya reasoning.
