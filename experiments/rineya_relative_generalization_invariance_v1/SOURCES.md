# Sources and evidence boundaries

## Primary methodological sources

1. **Relative Generalization Invariance of LLM Pretraining**, arXiv:2609.33016v1,
   September 26, 2026. https://arxiv.org/html/2609.33016v1
   Definition 4.1, Proposition 4.2 and Appendix A motivate offset-only loss comparisons,
   centered concordance and pair-discrepancy diagnostics. The main aggregate results use
   1,024-token units; individual-token results are separate. This implementation applies
   the measurement idea to post-training, not a reproduced pretraining theorem or study.
   Its weighted single-run discrepancy does not estimate the paper's expectation over runs.

2. **Hugging Face Transformers: Qwen3 model documentation.**
   https://huggingface.co/docs/transformers/model_doc/qwen3
   Reference for dense Qwen3 causal-model inputs, loss masking and selective-logit behavior.
   The collector evaluates the decoder once and projects only answer positions, using a
   tested next-token shift. Actual Transformers loading is optional and still unqualified
   in the delivery environment; see the skipped integration tests.

3. **Hugging Face PEFT: LoRA API documentation.**
   https://huggingface.co/docs/peft/package_reference/lora
   Reference for frozen-base adapters and weighted adapter combinations. The implementation
   uses `combination_type="cat"` and never dense-merges the sparse host. Unsupported DoRA,
   saved host modules and bias-updating adapters are refused rather than treated as ordinary LoRA.

4. **Task arithmetic / weight-disentanglement research**, arXiv:2305.12827.
   https://arxiv.org/abs/2305.12827
   Conceptual context for studying effects through the model's function rather than treating
   low flattened-weight cosine as a sufficient composition criterion. No code is copied.

All new source code and miniature fixtures in this directory are independently implemented.
No paper text, pretrained weights, restricted datasets or private result binaries are bundled.

## Rineya evidence sources (read-only)

- `LUNCY579/rineya-world-model-research`:
  `rcwm/experiments/RCWM-T1/REPORT.md`,
  `rcwm/experiments/RCWM-T1/runs/bank_confirm2/analysis/PREREG_2_RESULTS.md`,
  and `bwm/README.md`.
  https://github.com/LUNCY579/rineya-world-model-research
  These motivated separate rule-case, format, self-rollout and protected-behavior measurements.
  The originating records define their own evidence classes and limitations. They are not
  rerun or newly confirmed by RGI-001. The registered adapter's host/precision/unmerged-loading
  constraints must be taken from its own current freeze records, not inferred from a model name.

- `LUNCY579/rineya-dataset-retention-records`:
  `indep002-20260921/INDEP002_RESULT.md`.
  https://github.com/LUNCY579/rineya-dataset-retention-records
  Motivation for matching coefficients and measuring protected endpoints independently.
  Its selection-grade composition evidence is not upgraded to confirmation by this branch.

- Implementation parent:
  https://github.com/LUNCY579/ChatBubble/commit/03db218ceebc61a2bc04f2482cf7e094045e01c1
  The existing `post-training-Mimo` integration branch is left unchanged. No game, vault,
  published core, historical evidence file or other research lane is modified.

## Claims this source review does not license

No proof of unrestricted skill stacking, identical knowledge from similar loss profiles,
no-forgetting from centered CCC, or a latent world model from an activation offset is claimed.
The new experimental protocol, matching policy, analysis weights, guardrails and software
fixtures are proposed Rineya methodology. Their scientific utility requires native tests.
