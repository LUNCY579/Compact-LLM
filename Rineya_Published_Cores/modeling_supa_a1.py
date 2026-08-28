"""Qwen3 with the abstention decision folded into the output logits.

The gate this implements is

    abstain iff  logP(UNKNOWN) - logP(argmax) >= threshold

Both terms share the same softmax normaliser, so it cancels and the rule reduces
to a comparison of two raw logits:

    z_UNKNOWN + b >= z_argmax,   b = -threshold

Under greedy decoding the emitted token is the argmax, so adding the constant
``b`` to one row of the logit vector *is* the gate. No wrapper, no second pass,
no post-processing: the abstention behaviour is a property of the model.

Two conditions make this exact, and both hold for this suite:

* ``UNKNOWN`` is a single token in the Qwen3 vocabulary (id 51378);
* decoding is greedy, so the decision is an argmax comparison.

One deliberate deviation: the offset is applied at every decode step, while the
gate was defined on the first generated token only. Whether that changes any
prediction is measured, not assumed -- see the checkpoint's evaluation report.
"""

from __future__ import annotations

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class Qwen3AbstainForCausalLM(Qwen3ForCausalLM):
    """Qwen3 plus a constant logit offset on the abstention token."""

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        token_id = getattr(self.config, "abstain_token_id", None)
        bias = float(getattr(self.config, "abstain_logit_bias", 0.0) or 0.0)
        logits = getattr(outputs, "logits", None)
        if token_id is not None and bias and logits is not None:
            logits[..., int(token_id)] = logits[..., int(token_id)] + bias
        return outputs
