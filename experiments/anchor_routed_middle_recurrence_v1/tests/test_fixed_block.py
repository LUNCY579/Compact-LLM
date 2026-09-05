import pytest
import torch

from armr.hooks import Reapplication
from armr.recurrence import FixedBlock
from test_mechanics import model


@pytest.mark.parametrize("apps", [1,2,3])
def test_block_order_and_per_forward_reset(model, apps):
    observed=[]
    for i, block in enumerate(model.model.layers):
        original=block.forward
        def spy(*args, _i=i, _original=original, **kwargs):
            observed.append((_i,kwargs["position_ids"],kwargs["position_embeddings"]))
            return _original(*args,**kwargs)
        block.forward=spy
    ids=torch.tensor([[1,4,6]])
    with torch.inference_mode():
        base=model(ids,use_cache=False).logits
        observed.clear()
        with FixedBlock(model,[3,4],apps) as wrapper:
            for _ in range(2):
                before=len(observed)
                got=model(ids,use_cache=False).logits
                entries=observed[before:]
                assert [e[0] for e in entries] == [0,1,2]+[3,4]*apps
                assert all(e[1] is entries[0][1] and e[2] is entries[0][2] for e in entries)
            assert wrapper.extra_layer_applications == 2*(apps-1)
        assert torch.equal(base,model(ids,use_cache=False).logits)
        if apps==1:
            assert torch.equal(base,got)


@pytest.mark.parametrize("layers", [[2,3],[3,3],[3,5],[3],[4,3]])
def test_invalid_blocks(model,layers):
    with pytest.raises(ValueError):
        FixedBlock(model,layers)


def test_wrapper_reuse_disabled_after_enabled(model):
    ids=torch.tensor([[1,4]])
    wrapper=Reapplication(model,3,2)
    with torch.inference_mode():
        base=model(ids,use_cache=False).logits
        with wrapper:
            model(ids,use_cache=False)
        wrapper.enabled=False
        with wrapper:
            assert torch.equal(base,model(ids,use_cache=False).logits)
    assert not model._forward_pre_hooks
