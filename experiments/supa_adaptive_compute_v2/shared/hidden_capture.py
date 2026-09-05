"""The actual installed donor decoder, streamed without loading a whole donor."""
from .legacy import legacy_graft


def capture_native(path, encoded, layers, limit=6, progress=None, *, device='cpu',max_cuda_memory_gib=6.5):
    if str(device) == 'cpu':
        return legacy_graft().capture_native(path,encoded,layers,limit,progress)
    import gc
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    from rineya_cross_scale.streaming_forward import StreamingWeights
    from .runtime import synchronize, cuda_memory
    from .integrity import frozen_inventory, state_hash
    legacy = legacy_graft()
    config = AutoConfig.from_pretrained(path,local_files_only=True)
    config._attn_implementation = 'eager'
    rope = Qwen3RotaryEmbedding(config).to(device).eval().requires_grad_(False)
    captured = {}
    with torch.inference_mode(), StreamingWeights(path) as weights:
        states = [weights.rows('model.embed_tokens.weight',ids[0].cpu()).unsqueeze(0) for ids in encoded]
        weights.close()
        for index in range(max(layers)+1):
            layer = legacy.load_layer(weights,config,index).to(device)
            frozen_inventory(layer)
            before = state_hash(layer)
            for row,host in enumerate(states):
                hidden = host.to(device)
                positions,mask = legacy.positions_mask(hidden)
                value = legacy.finite(layer(hidden,attention_mask=mask,position_ids=positions,
                    position_embeddings=rope(hidden,positions),use_cache=False))
                states[row] = value.cpu()
                synchronize(device)
                legacy.memory_guard(limit)
                cuda_memory({'device':str(device),'max_cuda_memory_gib':max_cuda_memory_gib})
                del hidden,value,positions,mask
            if state_hash(layer) != before:
                raise RuntimeError('streamed GPU layer mutated')
            if index in layers:
                captured[index] = [h[0].clone() for h in states]
            del layer
            weights.close()
            gc.collect()
            if progress:
                progress(index,max(layers))
    return {index:[h.clone() for h in values] for index,values in captured.items()}


def load_layer(*args, **kwargs):
    return legacy_graft().load_layer(*args, **kwargs)


def positions_mask(*args, **kwargs):
    return legacy_graft().positions_mask(*args, **kwargs)
