# Source and implementation map

## Primary reference

Luis Zuin, Alexis Huet, Dario Rossi, Zied Ben Houidi. *Disentangling Self-Distillation:
Measuring and Modeling Acquisition and Retention*. arXiv:2609.39494v1, 2026-09-30.

- https://arxiv.org/abs/2609.39494
- https://arxiv.org/html/2609.39494v1
- https://arxiv.org/pdf/2609.39494
- Official code: https://github.com/lhzuin/disentangling_self_distillation
- Inspected official main: `352f0898616e2a8b1b2c24419f25ae133b48d2a6`.

The article is CC BY 4.0. This branch independently implements its mathematical
construction. No upstream trainer/kernel code, model weights, original training
corpora or paper PDF are copied into this repository. Follow original licenses
and dataset permissions when separately obtaining those assets.

| Reference | Local implementation | Qualification boundary |
|---|---|---|
| Eq1 shared trajectory/different context | engine.py, data.py | Native Qwen3 loader still unexecuted |
| Eq2 detached full-vocabulary KL | objectives.py, engine.py | FP64 analytic gradient checks and BF16 fixtures executed |
| Eq3 EMA after student step | teacher.py | Parameter bank represents changed subspace, not full-model adaptation |
| Section4 20 method cells plus SFT | plan.py | Default Rineya plan has 126 cells, not the 1,200-run source study |
| Section4 accuracy endpoints | analysis.py | Local scorers, not original five-benchmark evaluation harness |
| Appendix B.3 clockwise spatial pair | data.py | Fresh simpler diagnostic; same prompt/parent, reference conventions differ |
| Appendix C privileged template | data.py | Independently written fixed user-message template, not exact prompt reproduction |
| Appendix E.1 feature/readout geometry | controlled.py | Exact statewise cosine does not imply disjoint global spans |
| Eqs7-10 attainable target/context | controlled.py | Least-squares target; context displacement calibrated only initially |
| Eq11 exact detached occupancy | controlled.py | No trajectory-sampling gradient or refreshed target mixture |
| Appendix E.4 metrics and cross-fit | controlled.py, analysis.py | Seed-level pairing, complete fixed-stratum grid required |
| Appendix F 18-rate controlled grid | CONTROLLED_RATES, CLI --paper-grid | Distinct from five-rate LLM grid |

Official `distil_config.py` uses `alpha` for a divergence/JSD choice and
`ref_model_mixup_alpha` for EMA. Our API deliberately uses separate `kl_direction`
and `ema_rate` names so they cannot be accidentally interchanged.

## API references (not native execution evidence)

- https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3
- https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.sdpa_kernel.html
- https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.kl_div.html

Only PyTorch 2.10.0+cpu was actually executed. Transformers installation in an
isolated test environment was attempted but blocked by network/DNS access; no
native compatibility or GPU performance claim follows from documentation review.

## Intentional numerical implementation detail

KL's analytical logit gradient is `p-q` forward and
`p*(log(p/q)-sum(p*log(p/q)))` reverse. The custom autograd function recomputes that
gradient and supports student double derivatives, avoiding cancellation of unit
mass terms at p=q. Exact equal represented distributions therefore have zero
update signal. This changes no mathematical objective. Tests compare values,
first derivatives and second derivatives against independent calculations.
