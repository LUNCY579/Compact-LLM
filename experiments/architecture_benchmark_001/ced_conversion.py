"""Official C28/C40 construction only: python -m ...ced_conversion --help.

G1_CED is a mechanical initialization receipt, NOT recovery, CED G2, or efficacy.
The existing G0-G2 implementation and every G3-G5 gate remain unchanged.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import torch
from torch.nn import functional as F
from safetensors import SafetensorError
from . import run
from .budget import budget, presets
from .gates import GateClosed, context, protocol, runtime, validate_receipt, write_once
from .model import DenseReference
from .source import ROOT, audit_source, sha256_file
from .transfer import native_key_map, state_digest, to_ced

# Fixed, answer-free mechanics; no prompt/seed/donor search or quality endpoint.
PROBE_IDS = (1, 7, 11, 3, 9, 5, 13, 2)
ATOL = RTOL = 1e-4
REQUIRED = {
    'G0': ({'official_source_unchanged', 'native_reference_logits', 'parser_usable',
            'termination_usable', 'diagnostic_not_floor_or_ceiling',
            'registered_context_executed', 'unmodified_native_readout'}, {'predictions.json'}),
    'G1_IDENTITY': ({'identity_logits_exact', 'chunked_cache_parity', 'source_unchanged',
                     'retained_original_layer_order', 'exact_parameter_count'}, {'lineage.json'}),
    'G1_WIDTH': ({'registered_dimensions', 'all_tensors_accounted', 'finite_converted_logits',
                  'save_reload_exact', 'source_unchanged', 'calibration_unchanged'},
                 {'lineage.json', 'calibration_scores.json', 'config.json', 'model.safetensors'}),
}


def _configs(arm: str):
    if arm not in ('C28', 'C40'):
        raise GateClosed('only the registered C28/C40 arms are admitted')
    p = protocol()
    if (p['G1']['memory_donor_policy'] != 'first_decoder_layer_of_plain_donor'
            or p['G1']['ffn_growth_policy'] != 'cyclic_gate_up_donors_zero_down_columns'
            or p['baseline_precision'] != 'float32' or p['attention_backend'] != 'eager'
            or p['G3_G5']['execution_enabled'] is not False):
        raise GateClosed('conversion contract drift; no training/evaluation amendment is admitted here')
    names = {'C28': ('B28_1024_official_shape', 'C28_1024_near_baseline_budget'),
             'C40': ('D40_896_near_baseline_budget', 'C40_896_near_baseline_budget')}
    donor, target = names[arm]
    if p['primary_arms'][arm] != target:
        raise GateClosed('registered target drift')
    return presets()[donor], presets()[target]


def _parent(path: Path, stage: str, ctx: dict, previous: str | None) -> dict:
    body = validate_receipt(path, stage, ctx)
    checks, artifacts = REQUIRED[stage]
    if (type(body.get('schema')) is not int or body['schema'] != 1
            or not checks <= body['checks'].keys() or not artifacts <= body['artifacts'].keys()
            or body.get('parent_receipt_sha256') != previous
            or body.get('licenses_registered_training') is not False
            or body.get('licenses_efficacy_claim') is not False):
        raise GateClosed(f'incomplete or broken {stage} prerequisite chain')
    return body


def _chain(arm, source_audit, pilot, calibration, g0, identity, width, device):
    ctx = context(source_audit, pilot, device)
    parents = {'G0': _parent(g0, 'G0', ctx, None)}
    if arm == 'C40':
        parents['G1_IDENTITY'] = _parent(identity, 'G1_IDENTITY', ctx, parents['G0']['receipt_sha256'])
        parents['G1_WIDTH'] = _parent(width, 'G1_WIDTH', context(source_audit, calibration, device),
                                      parents['G1_IDENTITY']['receipt_sha256'])
    return parents


def _tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    h.update(memoryview(value.view(torch.uint8).numpy()).cast('B'))
    return h.hexdigest()


def verify_transfer(donor, converted, primitive):
    """Check every target byte against its declared immediate donor and growth rule."""
    c, target = donor.config, converted.config
    src, dst = donor.state_dict(), converted.state_dict()
    with torch.device('meta'):
        inventory = DenseReference(target).state_dict()
    if (set(dst) != set(inventory) or primitive['accounted_target_tensors'] != sorted(dst)
            or primitive['donor_layer'] != target.encoder_layers
            or primitive['target_config'] != asdict(target)):
        raise GateClosed('CED inventory, target, or fixed memory donor mismatch')
    rows = [i % c.intermediate_size for i in range(target.intermediate_size - c.intermediate_size)]
    donor_storages = {v.untyped_storage().data_ptr() for v in src.values()}
    target_storages, mapping = set(), {}
    for name, value in dst.items():
        if name.startswith('memory.'):
            tail = name[len('memory.'):]
            key = f'encoder.{target.encoder_layers}.' + ('input_norm.weight' if tail == 'norm.weight' else tail)
        elif name.startswith('decoder.'):
            _, i, tail = name.split('.', 2)
            key = f'encoder.{target.encoder_layers + int(i)}.' + tail
        else:
            key = name
        original = src[key]
        expected, operation = original, 'copy'
        if rows and name.endswith(('mlp.gate_proj.weight', 'mlp.up_proj.weight')):
            indices = torch.tensor(rows, device=original.device)
            expected = torch.cat((original, original.index_select(0, indices)), 0)
            operation = 'copy_then_append_ffn_extra_row_indices'
        elif rows and name.endswith('mlp.down_proj.weight'):
            expected = torch.cat((original, original.new_zeros(c.hidden_size, len(rows))), 1)
            operation = 'copy_then_append_zero_down_columns'
        storage = value.untyped_storage().data_ptr()
        if (value.is_meta or value.dtype != torch.float32 or value.device != original.device
                or value.shape != inventory[name].shape or not torch.isfinite(value).all()
                or storage in donor_storages or storage in target_storages
                or _tensor_hash(value) != _tensor_hash(expected)):
            raise GateClosed(f'CED tensor copy/growth/shape/dtype/finiteness/alias check failed: {name}')
        target_storages.add(storage)
        mapping[name] = {'donor_tensor': key, 'operation': operation,
                         'donor_shape': list(original.shape), 'target_shape': list(value.shape),
                         'donor_tensor_sha256': _tensor_hash(original), 'target_tensor_sha256': _tensor_hash(value)}
    return {'tensor_map': mapping, 'ffn_extra_row_indices': rows,
            'source_config': asdict(c), 'target_config': asdict(target),
            'function_preserving': False, 'recovery_required': True,
            'requires_matched_plain_continuation_control': True}


def mechanical_checks(model):
    """Short-context causal/cache probes on fixed token IDs, never answer accuracy."""
    device = model.embed_tokens.weight.device
    ids = torch.tensor([PROBE_IDS], device=device)
    checks, errors = {}, {}

    def close(name, a, b):
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        errors[name] = (a - b).abs().max().item() if finite else None
        checks[name] = finite and bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))

    with torch.inference_mode():
        normalized = []
        handle = model.final_norm.register_forward_hook(lambda m, a, o: normalized.append(o))
        try:
            full = model(ids).logits
        finally:
            handle.remove()
        checks['finite_logits'] = bool(torch.isfinite(full).all())
        checks['tied_embedding_readout_exact'] = torch.equal(full, F.linear(normalized[0], model.embed_tokens.weight))
        changed = ids.clone(); changed[:, 4:] = changed[:, 4:].flip(1)
        close('future_token_isolation', full[:, :4], model(changed).logits[:, :4])
        pos = torch.zeros_like(ids)
        close('repeated_position_causality', model(ids, position_ids=pos).logits[:, :4],
              model(changed, position_ids=pos).logits[:, :4])
        mask = torch.ones_like(ids); mask[:, :2] = 0
        changed = ids.clone(); changed[:, :2] = changed[:, :2].flip(1)
        close('left_padding_isolation', model(ids, attention_mask=mask).logits[:, 2:],
              model(changed, attention_mask=mask).logits[:, 2:])
        seg = torch.zeros_like(ids); seg[:, 4:] = 1
        pos = torch.arange(8, device=device).remainder(4)[None]
        changed = ids.clone(); changed[:, :4] = changed[:, :4].flip(1)
        close('packed_document_isolation', model(ids, position_ids=pos, segment_ids=seg).logits[:, 4:],
              model(changed, position_ids=pos, segment_ids=seg).logits[:, 4:])
        pre = model(ids[:, :3], use_cache=True)
        tail = model(ids[:, 3:], past=pre.cache, use_cache=True)
        close('chunked_cache_parity', full, torch.cat((pre.logits, tail.logits), 1))
        cached, outputs = None, []
        for i in range(ids.shape[1]):
            step = model(ids[:, i:i+1], past=cached, use_cache=True)
            outputs.append(step.logits); cached = step.cache
        close('token_cache_parity', full, torch.cat(outputs, 1))
        close('tail_only_prefill_parity', full[:, -1:], model(ids, last_position_only=True).logits)
        pre = model(ids[:, :-1], use_cache=True, last_position_only=True)
        close('tail_prefill_decode_parity', full[:, -1:],
              model(ids[:, -1:], past=pre.cache, use_cache=True).logits)
        # Verify cache rejection without mutating source weights or applying an optimizer.
        for name, bad in [('foreign_cache_rejected', replace(cached, owner=-1)),
                          ('stale_cache_rejected', replace(cached, signature=()))]:
            try:
                model(ids[:, :1], past=bad, use_cache=True)
            except ValueError:
                checks[name] = True
            else:
                checks[name] = False
        banks = list(cached.encoder) + ([] if cached.memory is None else [cached.memory])
        expected_shape = (1, model.config.num_key_value_heads, len(PROBE_IDS), model.config.head_dim)
        checks['shared_cache_topology'] = (len(cached.encoder) == model.config.encoder_layers
            and cached.memory is not None and all(tuple(v.shape) == expected_shape for pair in banks for v in pair))
        cache_bytes = sum(v.numel() * v.element_size() for pair in banks for v in pair)
        checks['persistent_kv_bytes_exact'] = cache_bytes == budget(model.config, 4)['kv_bytes_per_token_batch1'] * len(PROBE_IDS)
        checks['no_decoder_private_kv'] = all(not hasattr(b, 'k_proj') and not hasattr(b, 'v_proj') for b in model.decoder)
    return checks, {'max_abs_parity_errors': errors, 'probe_persistent_kv_bytes': cache_bytes,
                    'probe_tokens': list(PROBE_IDS), 'atol': ATOL, 'rtol': RTOL,
                    'scope': 'eight fixed token IDs; not long-context, quality, latency, or total-memory evidence'}


def qualify_ced(*, source: Path, arm: str, pilot: Path, g0: Path, out: Path,
                device: str = 'cpu', identity: Path | None = None,
                width: Path | None = None, calibration: Path | None = None) -> dict:
    expected_donor, target = _configs(arm)
    if (arm == 'C28' and any(p is not None for p in (identity, width, calibration))) or (
            arm == 'C40' and any(p is None for p in (identity, width, calibration))):
        raise GateClosed('C28 needs only G0; C40 needs G0, G1_IDENTITY, G1_WIDTH and calibration')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise GateClosed('unsupported/unavailable device; no fallback')
    if runtime()['transformers'] != protocol()['native_transformers_version']:
        raise GateClosed('pinned native Transformers runtime required')
    audit = audit_source(source)
    ctx = context(audit, pilot, device)
    run.load_rows(pilot, 'pilot')
    if calibration is not None:
        run.ensure_disjoint(run.load_rows(pilot, 'pilot'), run.load_rows(calibration, 'calibration'))
    parents = _chain(arm, audit, pilot, calibration, g0, identity, width, device)
    inputs = {'pilot': pilot, 'G0': g0}
    if arm == 'C40':
        inputs.update(calibration=calibration, G1_IDENTITY=identity, G1_WIDTH=width)
    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    protected = [source.resolve(), g0.parent.resolve()]
    protected += [p.parent.resolve() for p in (identity, width) if p is not None]
    if out.is_symlink() or any(out.resolve().is_relative_to(p) for p in protected):
        raise GateClosed('output may not overwrite or nest inside source/prerequisite evidence')
    run._directory(out, source)
    stage = f'G1_CED_{arm}'
    try:
        if arm == 'C28':
            donor, tokenizer, native = run._load(source, device)
            del tokenizer, native
        else:
            donor = run._restore(width.parent, device)
        if donor.config != expected_donor:
            raise GateClosed('ordinary donor differs from the registered arm; G2/recovered/arbitrary donors forbidden')
        run._require_materialized(donor, device)
        before = state_digest(donor)
        converted, primitive = to_ced(donor, target, donor_layer=target.encoder_layers)
        if converted.config != target:
            raise GateClosed('converted dimensions differ from the registered CED target')
        lineage = verify_transfer(donor, converted, primitive)
        lineage.update(primitive=primitive, arm=arm, donor_state_sha256=before,
                       donor_status='official_B28_unmodified' if arm == 'C28' else 'G1_WIDTH_D40_unrecovered')
        checks = {'donor_state_unchanged': state_digest(donor) == before,
                  'registered_parameter_count': sum(p.numel() for p in converted.parameters()) == budget(target)['parameters'],
                  'complete_verified_tensor_lineage': set(lineage['tensor_map']) == set(converted.state_dict())}
        del donor; gc.collect()  # Do not retain three full-size models during reload.
        state_before = state_digest(converted)
        probe_checks, metrics = mechanical_checks(converted)
        checks.update(probe_checks)
        checks['conversion_state_unchanged_by_probes'] = state_digest(converted) == state_before
        with torch.inference_mode():
            probe = torch.tensor([PROBE_IDS], device=device)
            logits_before = converted(probe).logits.detach().cpu()
        run._save(converted, out)
        del converted; gc.collect()
        restored = run._restore(out, device)
        checks['save_reload_state_exact'] = state_digest(restored) == state_before
        checks['save_reload_config_exact'] = restored.config == target
        with torch.inference_mode():
            checks['save_reload_logits_exact'] = torch.equal(logits_before, restored(probe).logits.cpu())
        del restored; gc.collect()
        lineage['target_state_sha256'] = state_before
        artifacts = ['config.json', 'model.safetensors', 'lineage.json', 'provenance.json']
        if arm == 'C40':
            shutil.copyfile(width.parent/'lineage.json', out/'ordinary_lineage.json')
            checks['ordinary_lineage_snapshot_exact'] = sha256_file(out/'ordinary_lineage.json') == parents['G1_WIDTH']['artifacts']['lineage.json']
            artifacts.append('ordinary_lineage.json')
        # Rehash source, inputs, full parent artifacts, runtime and implementation before PASS.
        checks['official_source_unchanged'] = audit_source(source) == audit
        checks['input_bytes_unchanged'] = {name: sha256_file(path) for name, path in inputs.items()} == input_hashes
        checks['context_unchanged'] = context(audit, pilot, device) == ctx
        checks['prerequisite_chain_unchanged'] = _chain(arm, audit, pilot, calibration, g0, identity, width, device) == parents
        write_once(out/'lineage.json', lineage)
        write_once(out/'provenance.json', {'schema': 1, 'official_source_audit': audit,
            'input_file_sha256': input_hashes, 'validated_parent_receipts': parents,
            'official_native_to_reference_keys': native_key_map(presets()['B28_1024_official_shape']),
            'donor_checkpoint_sha256': audit['assets']['model.safetensors'] if arm == 'C28' else parents['G1_WIDTH']['artifacts']['model.safetensors'],
            'parent_snapshots_are_not_standalone_receipts': True,
            'build_mode': 'mechanical_initialization_only', 'C40_recovered_donor_requirement_waived': False,
            'optimizer_steps': 0, 'quality_evaluation_executed': False})
        parent = parents['G0'] if arm == 'C28' else parents['G1_WIDTH']
        result = write_once(out/f'{stage}.json', {'schema': 1, 'stage': stage, 'arm': arm,
            'status': 'PASS' if all(v is True for v in checks.values()) else 'FAIL',
            'evidence_kind': 'official_model_qualification', 'context': ctx,
            'parent_receipt_sha256': parent['receipt_sha256'], 'checks': checks,
            'metrics': {**metrics, 'parameters': budget(target)['parameters'], 'optimizer_steps': 0,
                        'function_preserving': False, 'quality_recovery_demonstrated': False},
            'artifacts': {name: sha256_file(out/name) for name in artifacts},
            'licenses_CED_G2': False, 'licenses_registered_training': False,
            'licenses_quality_evaluation': False, 'licenses_efficacy_claim': False})
        if result['status'] != 'PASS':
            raise GateClosed('CED mechanical checks failed; preserve this run and inspect its receipt')
        return result
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError, SafetensorError) as exc:
        write_once(out/'failure.json', {'stage': stage, 'status': 'BLOCKED_OR_FAILED', 'reason': str(exc),
            'context': ctx, 'licenses_registered_training': False, 'licenses_efficacy_claim': False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('C28', 'C40'), required=True)
    for name in ('source', 'g0', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=ROOT/'data/pilot.jsonl')
    parser.add_argument('--g1-identity', dest='identity', type=Path)
    parser.add_argument('--g1-width', dest='width', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    try:
        result = qualify_ced(**vars(args))
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError, SafetensorError) as exc:
        result = {'status': 'BLOCKED_OR_FAILED', 'reason': str(exc),
                  'licenses_registered_training': False, 'licenses_efficacy_claim': False}
        print(json.dumps(result, indent=2)); return 2
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
