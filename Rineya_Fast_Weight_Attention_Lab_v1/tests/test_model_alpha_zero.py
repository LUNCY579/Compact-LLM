"""Optional GPU test: α=0 recovers unmodified Supa-A1.1 logits."""

from __future__ import annotations

import pytest
import torch

from rineya_fwa.arms import apply_arm
from rineya_fwa.loading import load_supa, load_tokenizer
from rineya_fwa.paths import SUPA_PATH


@pytest.mark.model
@pytest.mark.gpu
def test_supa_alpha_zero_logits():
    if not (SUPA_PATH / "model.safetensors").is_file():
        pytest.skip("Supa-A1.1 weights are not on disk")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tokenizer = load_tokenizer()
    model = load_supa(device="cuda", eval_mode=True)
    ids = tokenizer("The harbour ledger is ordinary.", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base = model(input_ids=ids).logits
    handle = apply_arm(model, "C", layers=[14], rank=16, init_alpha=0.0)
    assert handle is not None
    model.eval()
    with torch.no_grad():
        wrapped = model(input_ids=ids).logits
    handle.uninstall()
    torch.testing.assert_close(wrapped, base, rtol=1e-3, atol=1e-2)
