# Licensing and attribution

SPDX-License-Identifier: GPL-3.0-only

This isolated LGR-001 experiment package is distributed under GNU GPL version 3;
see `COPYING`. This notice does not relicense any preexisting ChatBubble file.

The VPAD equations and numerical policy in `regularization.py` are adapted from
`deep-spin/len-gen-regularization`, commit
`d9749d54347f585abfd471d9a71015ac25562780`, specifically
`dropout/vpad_core.py` (GNU GPL version 3).
Research attribution: Pavlo Vasylenko, Matthias Lindemann, Andre F. T. Martins,
and Marcos Treviso, "Length Generalization Needs Proper Regularization."

Changes here: explicit Qwen block-entry placement rather than a broad RMSNorm
walker; original-forward identity paths; reversible patching; validation and
numerical tests; an isolated LoRA experiment runner; data, safety and evidence
contracts. No upstream checkpoint, dataset or full model implementation is
redistributed. Do not silently relicense this package or copy it into differently
licensed release code without an appropriate license-compatibility review.
