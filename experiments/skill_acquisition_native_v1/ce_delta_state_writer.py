"""Retain actual full-state names/aliases and tiny non-base bytes, closed entry.

Selected masters are reused from the verified parent checkpoint. No selected
copy or framework import occurs before the existing native execution permit.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path

from . import ce_native_authority as authority, ce_qualification_artifacts as raw
from .ce_delta_rebuild import SCHEMA, DTYPES

require = authority.require


def retain_full_state(executor):
    # Admission precedes traversing backend/model/framework or any source file.
    from .ce_native_qualification import NativeQualificationExecutor
    require(type(executor) is NativeQualificationExecutor, 'exact admitted native qualification executor')
    executor._admit('delta_roundtrip')
    backend = executor.backend; context = executor.context
    authority.require_execution_permit(executor.permit, context, 'delta_roundtrip', operation=executor.operation, gate=executor.gate)
    from experiments.maxrl_comp002_pretraining_v1 import warmstart_checkpoint as streams
    from . import update
    base_path = Path(backend.spec['model_dir']) / 'model.safetensors'
    with executor._span('qualification_artifact', read_bytes_upper=raw.CHUNK + 8):
        executor.gate('qualification_artifact')
        with base_path.open('rb') as stream:
            first = stream.read(8); require(len(first) == 8, 'actual pinned base header prefix')
            count = int.from_bytes(first, 'little'); require(0 < count <= raw.CHUNK, 'bounded actual pinned base header')
            header = raw._json(stream.read(count)); require(type(header) is dict, 'actual base tensor header')
        base_bytes = base_path.stat().st_size
    named = dict(backend.model.named_parameters(remove_duplicate=False))
    buffers = dict(backend.model.named_buffers(remove_duplicate=False)); require(not set(named) & set(buffers), 'full state parameter/buffer collision')
    full = dict(named, **buffers); selected = set(backend.trainable_names)
    dtype_names = {DTYPES[key][1]: key for key in DTYPES}
    inventory, extra = [], {}
    with executor._span('qualification_artifact', identity_hash_attempts_upper=3, identity_hash_bytes_upper=3 * raw.MAX_JSON):
        aliases = {}
        for name in sorted(full):
            tensor = full[name]; dtype = dtype_names.get(str(tensor.dtype))
            require(dtype is not None and tensor.is_contiguous(), 'bounded supported actual runtime state dtype/layout')
            source = name if name in header else None
            if name == 'lm_head.weight' and source is None:
                require(tensor is named.get('model.embed_tokens.weight'), 'actual tied lm_head alias only')
                source = 'model.embed_tokens.weight'
            require(source is not None or name in buffers, 'actual parameter absent from pinned base')
            if source is None: extra[name] = tensor
            alias = aliases.get(id(tensor)); aliases.setdefault(id(tensor), name)
            inventory.append({'name': name, 'shape': list(tensor.shape), 'dtype': dtype,
                'kind': 'parameter' if name in named else 'buffer', 'source_name': source, 'alias_of': alias})
        require(selected == {name for name, value in named.items() if value.requires_grad}, 'actual full-state selected inventory')
    extra_ref = None
    if extra:
        maximum = context['bindings']['scope']['qualifier_execution'].get('nonbase_state_bytes_upper')
        size = sum(value.numel() * value.element_size() for value in extra.values())
        require(type(maximum) is int and 0 < size <= maximum <= 16 * raw.CHUNK, 'tiny explicit non-base state byte cap')
        specs = {name: (dtype_names[str(value.dtype)], list(value.shape)) for name, value in extra.items()}
        # Existing streamed writer accepts F32/U8; unusual model buffer encodings
        # require a reviewed writer extension, never omitted state or hashes.
        require(all(dtype in ('F32', 'U8') for dtype, shape in specs.values()), 'unsupported retained non-base buffer encoding')
        path = executor._target('delta_nonbase_state', '.safetensors')
        with executor._span('qualification_tensor_write', write_bytes_upper=maximum + raw.CHUNK + 8,
                read_bytes_upper=2 * (maximum + raw.CHUNK + 8), identity_hash_attempts_upper=3,
                identity_hash_bytes_upper=3 * (maximum + raw.CHUNK + 8)):
            executor.gate('qualification_tensor_write'); executor._owned_path(path); path.parent.mkdir(parents=True, exist_ok=True)
            written = streams.stream_tensors(path, specs, lambda name: streams.tensor_chunks(extra[name]))
            extra_ref = {'path': str(path), 'sha256': written['file_sha256'], 'bytes': written['bytes']}
    terminal_owner = context['native_services'].get('checkpoint_terminal_owner')
    from .ce_checkpoint_processes import CheckpointProcesses
    require(type(terminal_owner) is CheckpointProcesses and terminal_owner.restore_record is not None,
            'actual retained parent U2 checkpoint/master source before delta')
    parent = terminal_owner.parent
    payload = {'schema': SCHEMA, 'full_state_inventory': inventory, 'selected_inventory': copy.deepcopy(backend.inventory),
        'nonbase_ref': extra_ref, 'parent_manifest_ref': copy.deepcopy(parent['manifest_ref']),
        'checkpoint_binding': copy.deepcopy(parent['binding']), 'parent_restore_record': copy.deepcopy(terminal_owner.restore_record),
        'base_ref': {'path': str(base_path), 'sha256': context['bindings']['anchors']['base_weights_sha256'], 'bytes': base_bytes},
        'warm_ref': {key: backend.spec['warm_delta_ref'][key] for key in ('path', 'sha256', 'bytes')},
        'common_anchor_sha256': context['bindings']['anchors']['common_anchor_sha256'],
        'warm_data_sha256': backend.spec['warm_delta_ref']['data_sha256'], 'warm_state_sha256': backend.spec['warm_delta_ref']['state_sha256'],
        'frozen_runtime_sha256': backend.frozen_sha256, 'native_qualified': False, 'execution_verified': False}
    reference = executor._write_json('delta_full_state', payload, 'delta_full_state')
    return payload, reference, ([] if extra_ref is None else [{**extra_ref, 'role': 'delta_buffers', 'binding_sha256': reference['binding_sha256']}])
