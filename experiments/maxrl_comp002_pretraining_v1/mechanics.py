"""Tensor-level qualification helpers; no loader, optimizer, training loop or grant."""
from __future__ import annotations
from .preflight import require, digest


def completion_labels(ids, attention, prompt_lengths, completion_lengths, eos_ids,
                      stop_reasons, max_new_tokens):
    """Labels cover actual sampled tokens, including first EOS but excluding padding.

    Supports one unpacked prompt/completion per row with left OR right padding.
    Lengths are counts of real tokens, not decoded-string or padded tensor offsets.
    A length-capped response is not repaired by appending EOS or changing its reward.
    """
    import torch
    require(ids.ndim == attention.ndim == 2 and ids.shape == attention.shape,
            'input/mask shape')
    require(ids.dtype == torch.long and attention.dtype == torch.bool, 'tensor dtypes')
    require(ids.device == attention.device, 'tensor devices')
    require(type(max_new_tokens) is int and max_new_tokens > 0, 'response cap')
    require(type(eos_ids) in (list, tuple) and len(eos_ids) > 0
            and len(set(eos_ids)) == len(eos_ids)
            and all(type(x) is int and x >= 0 for x in eos_ids), 'EOS inventory')
    batch = ids.shape[0]
    require(batch > 0 and all(len(v) == batch for v in
            (prompt_lengths, completion_lengths, stop_reasons)), 'batch lengths')
    labels = torch.full_like(ids, -100)
    for i, (p, c, reason) in enumerate(zip(prompt_lengths, completion_lengths, stop_reasons)):
        require(type(p) is int and type(c) is int and p > 0 and 0 < c <= max_new_tokens,
                'prompt/completion counts')
        positions = attention[i].nonzero(as_tuple=True)[0].tolist()
        require(len(positions) == p + c and positions == list(range(positions[0], positions[-1]+1)),
                'attention span: packing/gaps/truncation unsupported')
        start = positions[0] + p
        tokens = ids[i, start:start+c].tolist()
        require(all(t >= 0 for t in tokens), 'negative completion ID')
        eos_positions = [j for j, t in enumerate(tokens) if t in eos_ids]
        require(reason in ('eos', 'length'), 'unbound stop reason')
        if reason == 'eos':
            require(eos_positions == [c-1], 'first EOS must terminate generated span')
        else:
            require(not eos_positions and c == max_new_tokens, 'fabricated EOS/length stop')
        labels[i, start:start+c] = ids[i, start:start+c]
    return labels


def selected_logprobs(logits, ids, labels):
    """Exactly one causal shift; return only masked token log probabilities."""
    import torch
    require(logits.ndim == 3 and ids.ndim == labels.ndim == 2
            and logits.shape[:2] == ids.shape == labels.shape, 'logprob shapes')
    require(ids.dtype == labels.dtype == torch.long and logits.is_floating_point(), 'logprob dtypes')
    require(logits.device == ids.device == labels.device, 'logprob devices')
    require(bool(torch.isfinite(logits).all()), 'nonfinite logits')
    require(bool(((ids >= 0) & (ids < logits.shape[-1])).all()), 'ID out of vocabulary')
    require(bool(((labels == -100) | (labels == ids)).all()), 'label/ID mismatch')
    require(bool((labels[:, 0] == -100).all()), 'unpredictable first-token label')
    mask = labels[:, 1:] != -100
    require(bool(mask.any(dim=1).all()), 'empty completion loss')
    selected = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return selected[mask]


def parameter_inventory(model):
    """Observe actual requires_grad and aliases; never guesses names or toggles flags.

    Same-Parameter tied names count once. Distinct Parameters sharing storage are
    rejected until an explicit view-alias contract is supplied.
    """
    grouped = {}
    storage_owners = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        require(parameter.device.type != 'meta' and parameter.numel() > 0,
                'unloaded/empty parameter')
        storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
        require(storage_owners.setdefault(storage, id(parameter)) == id(parameter),
                'unsupported distinct-Parameter shared storage')
        entry = grouped.setdefault(id(parameter), {
            'names': [], 'shape': list(parameter.shape), 'dtype': str(parameter.dtype),
            'device': str(parameter.device), 'numel': parameter.numel(),
            'requires_grad': bool(parameter.requires_grad)})
        entry['names'].append(name)
    entries = sorted(grouped.values(), key=lambda x: sorted(x['names'])[0])
    require(entries and any(e['requires_grad'] for e in entries), 'empty trainable scope')
    for entry in entries:
        entry['names'].sort()
    return {'parameters': entries, 'inventory_sha256': digest(entries),
            'unique_numel': sum(e['numel'] for e in entries),
            'trainable_numel': sum(e['numel'] for e in entries if e['requires_grad']),
            'status': 'OBSERVED_INVENTORY_NOT_NATIVE_QUALIFICATION'}
