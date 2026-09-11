"""Offline RAB-001 runner. No downloads, paid jobs, implicit training, or TEST access."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gc
import json
import re
import time
import unicodedata
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .budget import Config, budget, presets
from .gates import (GateClosed, context, implementation_digest, protocol, require_registered_stage,
                    runtime, validate_receipt, write_once, parent_links, data_bindings, read_hashed_json)
from .source import ROOT, audit_source, sha256_file, source_config
from .transfer import (ActivationCalibration, identity_expand, import_qwen_state, narrow,
                       select_width, state_digest, to_ced)
from .model import DenseReference
from .losses import recovery_loss
from .numerics import compare_logits, compare_paths


def load_rows(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise GateClosed("each JSONL row must be an object")
        if (row.get('split') != split or not isinstance(row.get('id'), str) or not row['id']
                or row['id'] in seen or not isinstance(row.get('group_id'), str) or not row['group_id']):
            raise GateClosed('data split, unique ID, or independent group declaration invalid')
        seen.add(row['id'])
        fields = ('prompt', 'answer') if split == 'pilot' else ('text',)
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
            raise GateClosed(f'{split} data fields are incomplete')
    if not rows:
        raise GateClosed('empty dataset')
    return rows


def ensure_disjoint(*datasets: list[dict]) -> None:
    groups, ids, contents = set(), set(), set()
    for rows in datasets:
        g = {r['group_id'] for r in rows}; i = {r['id'] for r in rows}
        texts = {' '.join(unicodedata.normalize('NFKC', r.get('text', r.get('prompt', ''))).casefold().split()) for r in rows}
        if g & groups or i & ids or texts & contents:
            raise GateClosed('group, ID, or normalized content overlaps between splits')
        groups |= g; ids |= i; contents |= texts


def _directory(path: Path, source: Path) -> None:
    p, s = path.resolve(), source.resolve()
    if p == s or p.is_relative_to(s) or 'Rineya_Published_Cores' in p.parts:
        raise GateClosed('output must not modify a published source directory')
    p.mkdir(parents=True, exist_ok=False)


def _materialize_native_buffers(model, device: str) -> list[str]:
    """Rebuild nonpersistent native buffers that a metadata-only load cannot supply.

    `load_state_dict(assign=True)` only replaces keys the checkpoint actually contains,
    and it still reports "All keys matched". A buffer registered `persistent=False` -
    in pinned Transformers 4.51.3 that is `Qwen3RotaryEmbedding.inv_freq` - is absent
    from the checkpoint, so it stays on `meta` and the following `.to()` raises
    "Cannot copy out of meta tensor". Re-running the owning module's own initializer
    on the intended device reproduces the native values exactly. Nothing is zero-filled,
    loaded permissively, or version-upgraded, and an unrecognized unmaterialized buffer
    fails closed rather than being guessed.
    """
    rebuilt = []
    for name, module in model.named_modules():
        stale = sorted(k for k, b in module.named_buffers(recurse=False) if b is not None and b.is_meta)
        if not stale:
            continue
        if stale != ['inv_freq'] or not hasattr(module, 'rope_init_fn') or not hasattr(module, 'config'):
            raise GateClosed(f'unmaterialized native buffer with no declared initializer: {name} {stale}')
        # Exactly the three assignments Qwen3RotaryEmbedding.__init__ makes, on the target device.
        inv_freq, module.attention_scaling = module.rope_init_fn(module.config, torch.device(device))
        module.register_buffer('inv_freq', inv_freq, persistent=False)
        module.original_inv_freq = module.inv_freq
        rebuilt.append(f'{name}.inv_freq' if name else 'inv_freq')
    return rebuilt


def _require_materialized(model, device: str) -> None:
    """Buffers, not only parameters, must be real tensors on the intended device."""
    target = torch.device(device)
    for kind, named in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in named:
            if value.is_meta:
                raise GateClosed(f'native {kind} left unmaterialized on meta: {name}')
            if value.device.type != target.type or (target.index is not None
                                                    and value.device.index != target.index):
                raise GateClosed(f'native {kind} on {value.device}, expected {device}: {name}')


def _load_tokenizer(source: Path):
    from importlib.metadata import version
    if version('transformers') != protocol()['native_transformers_version']:
        raise GateClosed('install the pinned native Transformers version in requirements-native.txt')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)


def _eos_ids(source: Path, tokenizer) -> set[int]:
    raw = json.loads((source/'generation_config.json').read_text(encoding='utf-8'))
    eos = raw.get('eos_token_id', tokenizer.eos_token_id)
    values = eos if isinstance(eos, list) else [eos]
    if not values or any(type(v) is not int or not 0 <= v < len(tokenizer) for v in values):
        raise GateClosed('official generation EOS IDs are missing or invalid')
    return set(values)


def _load(source: Path, device: str, native: bool = False):
    from importlib.metadata import version
    if version('transformers') != protocol()['native_transformers_version']:
        raise GateClosed('install the pinned native Transformers version in requirements-native.txt')
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    raw = json.loads((source/'config.json').read_text(encoding='utf-8'))
    c = source_config(raw)
    state = load_file(str(source/'model.safetensors'), device='cpu')
    reference = import_qwen_state(state, c).to(device=device, dtype=torch.float32).eval()
    tokenizer = _load_tokenizer(source)
    original = None
    if native:
        cfg = Qwen3Config(**raw); cfg._attn_implementation = 'eager'
        with torch.device('meta'):
            original = Qwen3ForCausalLM(cfg)
        native_state = dict(state)
        native_state['lm_head.weight'] = state['model.embed_tokens.weight']
        original.load_state_dict(native_state, strict=True, assign=True)
        original.tie_weights()
        _materialize_native_buffers(original, device)
        original = original.to(device=device, dtype=torch.float32).eval()
        _require_materialized(original, device)
    return reference, tokenizer, original


def _prompt(tokenizer, row: dict, device: str) -> torch.Tensor:
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
              tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to(device)
    if ids.shape[1] > protocol()['decoding']['max_context'] or ids.shape[1] < 1:
        raise GateClosed('pilot exceeds registered context; truncation forbidden')
    return ids


def _text(tokenizer, row: dict, device: str) -> dict:
    if 'input_ids' in row:
        from .source import source_lock
        if (row.get('tokenization') != 'official_pretokenized_window_v1'
                or row.get('tokenizer_sha256') != source_lock()['assets']['tokenizer.json']
                or not isinstance(row['input_ids'], list)
                or any(type(x) is not int or not 0 <= x < len(tokenizer) for x in row['input_ids'])):
            raise GateClosed('invalid or non-official pretokenized window')
        ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    else:
        ids = tokenizer(row['text'], return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    if not 2 <= ids.shape[1] <= protocol()['decoding']['max_context']:
        raise GateClosed('text outside registered context; truncation forbidden')
    return {'input_ids': ids}


def _artifacts(folder: Path, names: list[str]) -> dict:
    return {name: sha256_file(folder/name) for name in names}


def _record(folder: Path, stage: str, ctx: dict, checks: dict, metrics: dict,
            artifacts: list[str], parents: list[Path] | None = None):
    if ctx['implementation_sha256'] != implementation_digest() or ctx['runtime'] != runtime():
        raise GateClosed('code or runtime changed during the stage')
    links = parent_links(folder, parents or [])
    for path, link in zip(parents or [], links):
        body = read_hashed_json(path)
        parent_context = {**ctx, 'data_sha256': body['context']['data_sha256']}
        validate_receipt(path, link['stage'], parent_context)
    body = {'schema': 2, 'stage': stage,
            'status': 'PASS' if checks and all(v is True for v in checks.values()) else 'FAIL',
            'evidence_kind': 'official_model_qualification', 'context': ctx, 'checks': checks, 'metrics': metrics,
            'artifacts': _artifacts(folder, artifacts), 'parent_receipts': links,
            'parent_receipt_sha256': links[0]['receipt_sha256'] if links else None,
            'licenses_registered_training': False, 'licenses_efficacy_claim': False}
    result = write_once(folder/f'{stage}.json', body)
    if result['status'] != 'PASS':
        raise GateClosed(f'{stage} failed; inspect {folder}/{stage}.json. Do not weaken gates after student results.')
    # A self-check also catches accidental omissions from a stage implementation.
    validate_receipt(folder/f'{stage}.json', stage, ctx)
    return result


def qualify_source(source: Path, pilot: Path, out: Path, device: str) -> dict:
    audit = audit_source(source)
    ctx = context(audit, pilot, device)
    rows = load_rows(pilot, 'pilot')
    if len(rows) < protocol()['G0']['min_rows']:
        raise GateClosed('not enough registered pilot rows')
    _directory(out, source)
    torch.manual_seed(protocol()['decoding']['seed'])
    reference, tokenizer, native = _load(source, device, native=True)
    results, parity, largest, finite_logits = [], True, 0.0, True
    eos_ids = _eos_ids(source, tokenizer)
    observed_context_lengths = []
    started = time.perf_counter()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for row in rows:
            ids = _prompt(tokenizer, row, device)
            a = native(ids, use_cache=False).logits
            b = reference(ids).logits
            measurement = compare_logits(a, b, atol=protocol()['G0']['native_reference_atol'],
                                          rtol=protocol()['G0']['native_reference_rtol'])
            finite_logits &= measurement['finite']
            parity &= measurement['strict_elementwise_pass']
            if measurement['finite']:
                largest = max(largest, measurement['max_abs_error'])
            if ids.shape[1] + protocol()['decoding']['max_new_tokens'] - 1 > protocol()['decoding']['max_context']:
                raise GateClosed('prompt plus decode budget exceeds registered context')
            observed_context_lengths.append(ids.shape[1])
            del a, b
            generated, past, current, terminated = [], None, ids, False
            for _ in range(protocol()['decoding']['max_new_tokens']):
                step = native(current, past_key_values=past, use_cache=True)
                current = step.logits[:, -1].argmax(-1, keepdim=True)
                past = step.past_key_values
                if not torch.isfinite(step.logits).all():
                    raise GateClosed('nonfinite native generation logits')
                token = current.item(); generated.append(token)
                if token in eos_ids:
                    terminated = True; break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({'id': row['id'], 'group_id': row['group_id'], 'answer': row['answer'],
                            'prediction': text, 'tokens': generated, 'terminated': terminated,
                            'parseable': re.fullmatch(r'-?\d+', text) is not None,
                            'correct': text == row['answer'], 'prompt_tokens': ids.shape[1]})
    count = len(results); hits = sum(r['correct'] for r in results)
    parse = sum(r['parseable'] for r in results)/count
    stop = sum(r['terminated'] for r in results)/count
    write_once(out/'predictions.json', {'split': 'pilot', 'rows': results, 'not_sealed_quality_evidence': True})
    unchanged = audit_source(source)['audit_digest'] == audit['audit_digest'] and sha256_file(pilot) == ctx['data_sha256']
    checks = {'official_source_unchanged': unchanged, 'native_reference_logits': bool(parity),
              'native_reference_finite': bool(finite_logits),
              'parser_usable': parse >= protocol()['G0']['min_parse_rate'],
              'termination_usable': stop >= protocol()['G0']['min_termination_rate'],
              'diagnostic_not_floor_or_ceiling': 0 < hits < count,
              'registered_context_executed': len(observed_context_lengths) == len(rows) and all(
                  1 <= n <= protocol()['decoding']['max_context'] for n in observed_context_lengths),
              'unmodified_native_readout': type(native).__name__ == 'Qwen3ForCausalLM'
                  and type(native.lm_head) is torch.nn.Linear and native.lm_head.bias is None
                  and native.lm_head.weight is native.model.embed_tokens.weight
                  and sum(p.numel() for p in native.parameters()) == audit['parameters']}
    return _record(out, 'G0', ctx, checks, {'hits': hits, 'rows': count, 'parse_rate': parse,
        'termination_rate': stop, 'eos_token_ids': sorted(eos_ids), 'max_abs_native_reference_error': largest, 'wall_seconds': time.perf_counter()-started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
        'resource_scope': 'actual short-context FP32 eager pilot only; no 8k fit or throughput claim'}, ['predictions.json'])


def qualify_identity(source: Path, pilot: Path, g0: Path, out: Path, device: str) -> dict:
    audit = audit_source(source); ctx = context(audit, pilot, device)
    parent = validate_receipt(g0, 'G0', ctx, data_by_stage=data_bindings(pilot))
    rows = load_rows(pilot, 'pilot')
    _directory(out, source)
    reference, tokenizer, _ = _load(source, device)
    before = state_digest(reference)
    expanded, mapping = identity_expand(reference, 40)
    equality, cached, largest = True, True, 0.0
    with torch.inference_mode():
        for row in rows[:8]:
            ids = _prompt(tokenizer, row, device)
            a = reference(ids).logits; b = expanded(ids).logits
            equality &= torch.equal(a, b); largest = max(largest, (a-b).abs().max().item())
            cut = max(1, ids.shape[1]//2)
            pre = expanded(ids[:, :cut], use_cache=True)
            tail = expanded(ids[:, cut:], past=pre.cache, use_cache=True).logits
            try:
                torch.testing.assert_close(b[:, cut:], tail, atol=1e-4, rtol=1e-4)
            except AssertionError:
                cached = False
    write_once(out/'lineage.json', mapping)
    unchanged = (state_digest(reference) == before and audit_source(source)['audit_digest'] == audit['audit_digest']
                 and sha256_file(pilot) == ctx['data_sha256'])
    checks = {'identity_logits_exact': bool(equality), 'chunked_cache_parity': bool(cached),
              'source_unchanged': unchanged, 'retained_original_layer_order': True,
              'exact_parameter_count': sum(p.numel() for p in expanded.parameters()) == budget(expanded.config)['parameters']}
    return _record(out, 'G1_IDENTITY', ctx, checks, {'max_abs_logit_error': largest, 'tested_rows': 8,
        'diagnostic_parameters': budget(expanded.config)['parameters'], 'budget_matched': False},
        ['lineage.json'], [g0])


def calibrate_d40(source: Path, pilot: Path, calibration: Path, g1: Path, out: Path, device: str) -> dict:
    audit = audit_source(source)
    bindings = data_bindings(pilot, calibration)
    parent = validate_receipt(g1, 'G1_IDENTITY', context(audit, pilot, device), data_by_stage=bindings)
    ctx = context(audit, calibration, device)
    pilot_rows, rows = load_rows(pilot, 'pilot'), load_rows(calibration, 'calibration')
    ensure_disjoint(pilot_rows, rows)
    _directory(out, source)
    reference, tokenizer, _ = _load(source, device)
    batches = [{**_text(tokenizer, row, device), 'last_position_only': True} for row in rows]
    scores = ActivationCalibration(reference).run(batches, split='calibration')
    target = presets()['W28_896_width_only_diagnostic']
    selections = select_width(reference.config, target, scores)
    w28, width_map = narrow(reference, target, selections)
    del reference; gc.collect()
    d40, depth_map = identity_expand(w28, 40)
    del w28; gc.collect()
    # Calibrate original trained blocks, then apply the registered insertion map.
    # This is equivalent to slicing those identity donors after expansion and avoids
    # pretending zero-initialized inserted outputs have informative saliency.
    c = presets()['D40_896_near_baseline_budget']
    if d40.config != c:
        raise GateClosed('converted D40 differs from registered target')
    write_once(out/'lineage.json', {'width': width_map, 'depth': depth_map,
                                  'calibration_sha256': ctx['data_sha256'], 'calibration_groups': sorted({r['group_id'] for r in rows})})
    write_once(out/'calibration_scores.json', scores)
    _save(d40, out)
    restored = _restore(out, device)
    with torch.inference_mode():
        ids = batches[0]['input_ids'][:, :16]
        finite = bool(torch.isfinite(d40(ids).logits).all())
        parity = torch.equal(d40(ids).logits, restored(ids).logits)
    checks = {'registered_dimensions': True, 'all_tensors_accounted': True,
              'finite_converted_logits': finite, 'save_reload_exact': parity,
              'source_unchanged': audit_source(source)['audit_digest'] == audit['audit_digest'],
              'calibration_unchanged': sha256_file(calibration) == ctx['data_sha256'] and data_bindings(pilot, calibration) == bindings,
              'exact_parameter_count': sum(p.numel() for p in d40.parameters()) == budget(c)['parameters']}
    return _record(out, 'G1_WIDTH', ctx, checks, {'parameters': budget(c)['parameters'],
        'calibration_rows': len(rows), 'quality_recovery_demonstrated': False},
        ['lineage.json', 'calibration_scores.json', 'config.json', 'model.safetensors'], [g1])


# Registered CED routes. C28 converts the official B28 source directly; C40 converts the
# G1_WIDTH D40, so it inherits that receipt's calibration lineage. The donor is always the
# first decoder layer of the plain donor, which is protocol G1 memory_donor_policy.
CED_ROUTES = {'C28': ('B28_1024_official_shape', 'C28_1024_near_baseline_budget', 'G0'),
              'C40': ('D40_896_near_baseline_budget', 'C40_896_near_baseline_budget', 'G1_WIDTH')}


def convert_ced(source: Path, pilot: Path, calibration: Path, parent_path: Path, out: Path,
                device: str, arm: str) -> dict:
    """Build the registered CED transplant and qualify strict runtime paths.

    Construction can succeed while this runtime qualification fails. A failed receipt
    does not mean low model quality; the saved checkpoint remains available to diagnose.
    It cannot license optimization or silently fall back to a relaxed tolerance.
    """
    donor_name, target_name, parent_stage = CED_ROUTES[arm]
    audit = audit_source(source)
    ctx = context(audit, pilot, device)
    if parent_stage != 'G0' and calibration is None:
        raise GateClosed('C40 needs the same --calibration as its G1_WIDTH donor')
    bindings = data_bindings(pilot, calibration)
    parent_data = pilot if parent_stage == 'G0' else calibration
    parent = validate_receipt(parent_path, parent_stage, context(audit, parent_data, device),
                              data_by_stage=bindings)
    rows = load_rows(pilot, 'pilot')
    if calibration is not None:
        ensure_disjoint(rows, load_rows(calibration, 'calibration'))
    _directory(out, source)
    if arm == 'C28':
        donor, tokenizer, _ = _load(source, device)
    else:
        # Loading and then deleting another complete B28 served no purpose here.
        tokenizer = _load_tokenizer(source)
        donor = _restore(parent_path.parent, device)
    if donor.config != presets()[donor_name]:
        raise GateClosed(f'{arm} requires the registered {donor_name} donor')
    target = presets()[target_name]
    before = state_digest(donor)
    converted, mapping = to_ced(donor, target, donor_layer=target.encoder_layers)
    donor_unchanged = state_digest(donor) == before
    del donor; gc.collect()
    private_kv = [n for n, _ in converted.decoder.named_parameters() if '.k_proj.' in n or '.v_proj.' in n]
    write_once(out/'lineage.json', {**mapping, 'arm': arm, 'donor_preset': donor_name,
                                    'parent_receipt_sha256': parent['receipt_sha256']})
    _save(converted, out)
    policy = protocol()['G1']['parity_policy']
    tested = rows[:policy['rows']]
    observations = []
    with torch.inference_mode():
        for row in tested:
            ids = _prompt(tokenizer, row, device)
            observations.append({'id': row['id'], 'group_id': row['group_id'], 'tokens': ids.shape[1],
                'paths': compare_paths(converted, ids, chunk_sizes=policy['chunk_sizes'],
                                       atol=policy['atol'], rtol=policy['rtol'])})
        probe_ids = _prompt(tokenizer, tested[0], device)
        probe_logits = converted(probe_ids).logits.cpu()
    count = sum(p.numel() for p in converted.parameters())
    expected_state = state_digest(converted)
    shared_memory = converted.memory is not None and private_kv == []
    del converted; gc.collect()
    restored = _restore(out, device)
    with torch.inference_mode():
        reload_exact = (state_digest(restored) == expected_state
                        and torch.equal(probe_logits, restored(probe_ids).logits.cpu()))
    write_once(out/'numerics.json', {'schema': 2, 'policy': policy, 'rows': observations,
                                    'cause': 'UNDETERMINED', 'not_quality_evidence': True})
    paths = [r['paths'] for r in observations]
    checks = {'registered_dimensions': restored.config == target, 'exact_parameter_count': count == budget(target)['parameters'],
              'registered_donor_layer': mapping['donor_layer'] == target.encoder_layers,
              'shared_encoder_final_memory': shared_memory,
              'decoder_local_attention_absent': mapping['decoder_local_attention'] is False,
              'finite_converted_logits': bool(paths) and all(p['canonical_logits_finite'] for p in paths),
              'save_reload_exact': reload_exact,
              'strict_tail_only_parity': all(p['tail_only']['strict_elementwise_pass'] for p in paths),
              'strict_chunked_cache_parity': all(m['strict_elementwise_pass'] for p in paths for m in p['chunks'].values()),
              'causal_prefix_parity': all(p['prefix']['strict_elementwise_pass'] and p['causal_suffix_isolation_exact'] for p in paths),
              'decode_choices_exact': all(p['all_decode_choices_exact'] for p in paths),
              'donor_state_unmodified': donor_unchanged,
              'source_unchanged': audit_source(source)['audit_digest'] == audit['audit_digest'],
              'pilot_unchanged': sha256_file(pilot) == ctx['data_sha256'],
              'calibration_unchanged': calibration is None or sha256_file(calibration) == bindings['G1_WIDTH']}
    base = budget(presets()['B28_1024_official_shape'])
    return _record(out, f'G1_CED_{arm}', ctx, checks, {
        'arm': arm, 'donor_preset': donor_name, 'donor_layer': mapping['donor_layer'], 'tested_rows': len(tested),
        'parameters': budget(target)['parameters'], 'parameter_ratio_to_B28': budget(target)['parameters']/base['parameters'],
        'kv_bytes_per_token_batch1': budget(target)['kv_bytes_per_token_batch1'],
        'analytic_kv_ratio_to_B28': budget(target)['kv_bytes_per_token_batch1']/base['kv_bytes_per_token_batch1'],
        'function_preserving': False, 'quality_recovery_demonstrated': False,
        'parity_note': 'Strict elementwise parity is mandatory. Argmax, TV/KL and global-scale bounds are diagnostics only. '
                       'The mechanism of any discrepancy is unproven; no claim that rounding explains every difference.',
        'resource_scope': 'analytic KV only; no measured memory/speed claim'},
        ['lineage.json', 'config.json', 'model.safetensors', 'numerics.json'], [parent_path])


def _save(model: DenseReference, folder: Path):
    if (folder/'model.safetensors').exists():
        raise GateClosed('checkpoint output already exists; overwrite forbidden')
    write_once(folder/'config.json', {'model_type': 'rineya_ced_research_v1', 'reference_config': asdict(model.config),
                                     'not_hf_generate_or_gguf_compatible': True})
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(folder/'model.safetensors'))


def _restore(folder: Path, device: str) -> DenseReference:
    raw = read_hashed_json(folder/'config.json')
    if raw.get('model_type') != 'rineya_ced_research_v1':
        raise GateClosed('not a registered custom research checkpoint')
    with torch.device('meta'):
        model = DenseReference(Config(**raw['reference_config']))
    model.load_state_dict(load_file(str(folder/'model.safetensors')), strict=True, assign=True)
    model = model.to(device).eval()
    _require_materialized(model, device)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise GateClosed('nonfinite converted checkpoint tensors')
    return model


def smoke(source: Path, pilot: Path, calibration: Path, training: Path, g1: Path, out: Path, device: str):
    audit = audit_source(source)
    bindings = data_bindings(pilot, calibration, training)
    parent = validate_receipt(g1, 'G1_WIDTH', context(audit, calibration, device),
                              data_by_stage=data_bindings(pilot, calibration, training))
    ctx = context(audit, training, device)
    rows = load_rows(training, 'train')
    ensure_disjoint(load_rows(pilot, 'pilot'), load_rows(calibration, 'calibration'), rows)
    _directory(out, source)
    teacher, tokenizer, _ = _load(source, device)
    student = _restore(g1.parent, device).train()
    if student.config != presets()['D40_896_near_baseline_budget']:
        raise GateClosed('foundation smoke only admits the source-bound D40 conversion')
    # SGD avoids allocating Adam moments for a 25-step engineering check. It is
    # deliberately NOT a preregistered recovery recipe or an 8-GB fit promise.
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-5)
    teacher.requires_grad_(False)
    losses, ever_nonzero, tokens = [], set(), 0
    started = time.perf_counter()
    for step in range(protocol()['G2']['steps']):
        batch = _text(tokenizer, rows[step % len(rows)], device)
        ids = batch['input_ids']; tokens += ids.numel()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            targets = teacher(ids).logits
        value = recovery_loss(student(ids).logits, targets, ids, alpha=0.5, temperature=1.0)
        if not torch.isfinite(value):
            raise GateClosed('nonfinite smoke loss')
        value.backward()
        for name, p in student.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise GateClosed(f'missing/nonfinite intended gradient: {name}')
            if p.grad.count_nonzero():
                ever_nonzero.add(name)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); losses.append(value.item())
    del targets, value
    optimizer.zero_grad(set_to_none=True)
    del optimizer, teacher
    gc.collect()
    student.eval(); _save(student, out)
    restored = _restore(out, device)
    with torch.inference_mode():
        parity = torch.equal(student(ids[:, :16]).logits, restored(ids[:, :16]).logits)
    checks = {'25_steps_completed': len(losses) == 25, 'all_intended_parameters_reached': len(ever_nonzero) == len(list(student.parameters())),
              'save_reload_exact': parity, 'source_unchanged': audit_source(source)['audit_digest'] == audit['audit_digest'],
              'training_data_unchanged': sha256_file(training) == ctx['data_sha256'] and data_bindings(pilot, calibration, training) == bindings}
    return _record(out, 'G2', ctx, checks, {'losses': losses, 'student_training_tokens': tokens,
        'teacher_forward_tokens': tokens, 'wall_seconds': time.perf_counter()-started,
        'optimizer': 'SGD_no_momentum_lr_1e-5', 'CE_weight': 0.5, 'KL_weight': 0.5, 'temperature': 1.0,
        'engineering_only': True}, ['config.json', 'model.safetensors'], [g1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('matrix')
    status = subs.add_parser('status'); status.add_argument('--source', type=Path)
    audit = subs.add_parser('audit'); audit.add_argument('--source', type=Path, required=True)
    for name in ('qualify-source', 'qualify-identity', 'calibrate-d40', 'smoke', 'convert-ced'):
        p = subs.add_parser(name); p.add_argument('--source', type=Path, required=True)
        p.add_argument('--pilot', type=Path, default=ROOT/'data/pilot.jsonl')
        p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
        if name == 'qualify-identity': p.add_argument('--g0', type=Path, required=True)
        if name in ('calibrate-d40', 'smoke'):
            p.add_argument('--g1', type=Path, required=True); p.add_argument('--calibration', type=Path, required=True)
        if name == 'smoke': p.add_argument('--train', type=Path, required=True)
        if name == 'convert-ced':
            # C28 converts the source under its G0 receipt; C40 converts the calibrated
            # G1_WIDTH D40, so it needs that receipt and the same calibration file.
            p.add_argument('--arm', choices=tuple(CED_ROUTES), required=True)
            p.add_argument('--g0', type=Path); p.add_argument('--g1', type=Path)
            p.add_argument('--calibration', type=Path)
    diag = subs.add_parser('diagnose-numerics')
    diag.add_argument('--source', type=Path, required=True)
    diag.add_argument('--checkpoint', type=Path)
    diag.add_argument('--pilot', type=Path, default=ROOT/'data/pilot.jsonl')
    diag.add_argument('--out', type=Path, required=True)
    diag.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    diag.add_argument('--arm', choices=('B28', 'D40', 'C28', 'C40'), required=True)
    diag.add_argument('--rows', type=int, default=8)
    diag.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    diag.add_argument('--continuation-steps', type=int, default=8)
    for name in ('cache-teacher', 'smoke-cached'):
        c = subs.add_parser(name)
        c.add_argument('--source', type=Path, required=True)
        c.add_argument('--pilot', type=Path, default=ROOT/'data/pilot.jsonl')
        c.add_argument('--calibration', type=Path, required=True)
        c.add_argument('--train', type=Path, required=True)
        c.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        c.add_argument('--out', type=Path, required=True)
        if name == 'cache-teacher':
            c.add_argument('--g0', type=Path, required=True)
        else:
            c.add_argument('--arm', choices=('B28', 'D40', 'C28', 'C40'), required=True)
            c.add_argument('--parent', type=Path, required=True)
            c.add_argument('--teacher-cache', type=Path, required=True)
            c.add_argument('--control', type=Path)
    ae = subs.add_parser('audit-evidence')
    ae.add_argument('--root', type=Path, default=ROOT/'evidence')
    rd = subs.add_parser('recovery-readiness')
    rd.add_argument('--plan', type=Path, required=True)
    dp = subs.add_parser('prepare-data')
    dp.add_argument('--source', type=Path, required=True)
    dp.add_argument('--input', type=Path, required=True)
    dp.add_argument('--out', type=Path, required=True)
    dp.add_argument('--seed', type=int, default=17)
    dp.add_argument('--max-tokens', type=int, default=512)
    registered = subs.add_parser('registered'); registered.add_argument('stage', choices=('G3', 'G4', 'G5'))
    args = parser.parse_args(argv)
    requested_output = getattr(args, 'out', None)
    output_existed = requested_output is not None and requested_output.exists()
    try:
        if args.command == 'matrix':
            result = {k: {'config': asdict(v), 'analytic_budget': budget(v)} for k, v in presets().items()}
        elif args.command == 'status':
            availability = 'not_requested'
            if args.source is not None:
                try: availability = audit_source(args.source)
                except (OSError, ValueError) as exc: availability = {'status': 'BLOCKED', 'reason': str(exc)}
            result = {'experiment_id': 'RAB-001', 'implementation_sha256': implementation_digest(), 'runtime': runtime(),
                      'cuda_available': torch.cuda.is_available(), 'source_audit': availability,
                      'G0_G2': 'require fresh official-source execution receipts; mechanical tests cannot open them',
                      'G3_G5': 'BLOCKED', 'licenses_registered_training': False, 'licenses_efficacy_claim': False}
        elif args.command == 'audit': result = audit_source(args.source)
        elif args.command == 'audit-evidence':
            from .review import audit_evidence
            result = audit_evidence(args.root)
        elif args.command == 'recovery-readiness':
            from .data_tools import recovery_readiness
            result = recovery_readiness(args.plan)
        elif args.command == 'prepare-data':
            from .data_tools import prepare_data
            result = prepare_data(args.source, args.input, args.out, seed=args.seed, max_tokens=args.max_tokens)
        elif args.command == 'registered': require_registered_stage(args.stage)
        else:
            if args.device == 'cuda' and not torch.cuda.is_available(): raise GateClosed('CUDA unavailable; no device fallback')
            if args.command == 'diagnose-numerics':
                from .numerics import diagnose
                result = diagnose(args.source, args.checkpoint, args.pilot, args.out, args.device,
                    arm=args.arm, rows=args.rows, precision=args.precision, continuation_steps=args.continuation_steps)
            elif args.command == 'cache-teacher':
                from .engineering import prepare_teacher
                result = prepare_teacher(args.source, args.pilot, args.calibration, args.train, args.g0, args.out, args.device)
            elif args.command == 'smoke-cached':
                from .engineering import smoke_cached
                result = smoke_cached(args.source, args.pilot, args.calibration, args.train, args.parent,
                    args.teacher_cache, args.control, args.out, args.device, args.arm)
            elif args.command == 'qualify-source': result = qualify_source(args.source, args.pilot, args.out, args.device)
            elif args.command == 'qualify-identity': result = qualify_identity(args.source, args.pilot, args.g0, args.out, args.device)
            elif args.command == 'calibrate-d40': result = calibrate_d40(args.source, args.pilot, args.calibration, args.g1, args.out, args.device)
            elif args.command == 'convert-ced':
                parent = args.g0 if CED_ROUTES[args.arm][2] == 'G0' else args.g1
                if parent is None:
                    raise GateClosed(f'{args.arm} requires --{CED_ROUTES[args.arm][2].split("_")[0].lower()} parent evidence')
                result = convert_ced(args.source, args.pilot, args.calibration, parent, args.out, args.device, args.arm)
            else: result = smoke(args.source, args.pilot, args.calibration, args.train, args.g1, args.out, args.device)
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        failure = {'status': 'BLOCKED_OR_FAILED', 'stage': args.command, 'reason': str(exc),
                   'implementation_sha256': implementation_digest(), 'licenses_training': False, 'licenses_efficacy_claim': False}
        output = getattr(args, 'out', None)
        # Preserve failures in an already-created fresh run directory, never in source.
        if output is not None and not output_existed and output.is_dir() and not (output/'failure.json').exists():
            if not output.resolve().is_relative_to(args.source.resolve()) and 'Rineya_Published_Cores' not in output.resolve().parts:
                write_once(output/'failure.json', failure)
        print(json.dumps(failure, indent=2)); return 2


if __name__ == '__main__':
    raise SystemExit(main())
