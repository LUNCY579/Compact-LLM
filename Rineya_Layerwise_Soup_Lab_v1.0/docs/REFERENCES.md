# Research References

The implementation was designed from the user-supplied research materials and source archive, especially:

1. Mitchell Wortsman et al. **Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.** ICML 2022. The lab reproduces the uniform/global interpolation control and adapts the paper’s learned-by-layer idea to two compatible language-model endpoints.
2. Zijian Zhang et al. **Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training.** 2026 preprint. The lab uses the paper as motivation for nonuniform layer sensitivity, while explicitly distinguishing single-layer RL training from post-hoc layer replacement and checkpoint interpolation.
3. Kai Yao et al. **Layer-wise Importance Matters: Less Memory for Better Performance in Parameter-efficient Fine-tuning of Large Language Models.** The response-suppression/importance principle motivates intervention-based layer profiling and task-specific reliability.
4. **Short-Response Model — Research and Engineering Blueprint v1.0.** The lab follows its requirements for compatible lineages, FP32 delta/weight merging, frozen evaluation, ablation controls, provenance, and quantization only after candidate acceptance.
5. The supplied **Quality Evaluator v4.5** source and policy files. These are vendored with provenance and checksums.

Additional uploaded papers on contextual sparsity, SliceGPT, Sheared LLaMA, memory reconstruction, and agent-system engineering are relevant to adjacent Rineya research tracks but are not silently claimed as evidence for the layer-wise soup result.
