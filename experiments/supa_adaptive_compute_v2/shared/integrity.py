"""Frozen backbone identity and gradients are checked independently of results."""
from . import legacy
from armr.provenance import state_hash, checkpoint_files


def frozen_inventory(model):
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    gradients = [name for name, p in model.named_parameters() if p.grad is not None]
    if model.training or trainable or gradients:
        raise RuntimeError(f'backbone is not frozen: trainable={trainable}; gradients={gradients}')
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': 0, 'backbone_gradients': 0}


def assert_unchanged(model, expected_state, model_path, expected_files):
    inventory = frozen_inventory(model)
    after_state, after_files = state_hash(model), checkpoint_files(model_path)
    if after_state != expected_state or after_files != expected_files:
        raise RuntimeError('canonical checkpoint or model state mutation')
    return {**inventory, 'state_sha256_after': after_state, 'checkpoint_files_after': after_files,
            'canonical_unchanged': True}


def validate_inference_receipt(value, config):
    """Model/control prerequisites before another arm consumes baseline evidence."""
    if value.get('status') != 'COMPLETE' or not value.get('canonical_unchanged'):
        raise RuntimeError('incomplete baseline/control receipt')
    for key in ('seed','threads','dtype','device','max_prompt_seconds'):
        if value.get(key) != config[key]:
            raise RuntimeError(f'incompatible inference receipt runtime: {key}')
    if value.get('torch') != '2.12.1+cu126' or value.get('transformers') != '5.14.1' or value.get('cache_mode') != 'full_prefix_no_cache':
        raise RuntimeError('unqualified inference receipt runtime')
    if value.get('model_class') != 'Qwen3AbstainForCausalLM' or value.get('trainable_parameters') != 0 or value.get('backbone_gradients') != 0:
        raise RuntimeError('unqualified or unfrozen inference model')
    if not value.get('state_sha256_before') or value['state_sha256_before'] != value.get('state_sha256_after'):
        raise RuntimeError('inference state identity missing or changed')
    files = value.get('checkpoint_files_before',{})
    if not files or files != value.get('checkpoint_files_after') or files.get('model.safetensors') != config['checkpoint_sha256']:
        raise RuntimeError('inference checkpoint identity missing or changed')
    if value.get('arm') == 'AM':
        for key in ('operator_trainable_parameters','operator_gradients','operator_trainable_parameters_after','operator_gradients_after'):
            if value.get(key) != 0:
                raise RuntimeError('compute-control freeze receipt incomplete')
        if not value.get('operator_state_sha256'):
            raise RuntimeError('compute-control identity missing')
    return True
