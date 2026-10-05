"""Independent inert loader-order fixtures. No model/encoder/decoder/binary IO."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/short_phase_audit_work'
sys.path.insert(0, str(STAGE))
import phase_replay as phase

def markers_for(profile, assets, first_order, *, final_order=None, asset_order=None):
    # This generator is independent of the auditor's tree traversal and of the
    # production loader/phase recorder. The returned metadata is deliberately inert.
    entries, next_id = [], 0
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    def node(name, children=None):
        nonlocal next_id
        next_id += 1
        ordinal = next_id
        def emit(event):
            index = len(entries)
            value = {'ordinal': ordinal, 'operation': name, 'event': event,
                     'monotonic': index / 100000.,
                     'utc': (start + timedelta(microseconds=index)).isoformat(),
                     'historical_cause_established': False}
            if event == 'returned':
                value.update(result=None, error=None)
            entries.append({'name': 'phase_%04d_%s' % (ordinal, 'start' if event == 'attempt' else 'end'), 'phase': value})
        emit('attempt')
        if children:
            children()
        emit('returned')
    def guard(name):
        node('guard_' + name)
    def api(name, children=None):
        def inside():
            guard('before_' + name)
            node('primitive_' + name, children)
            guard('after_' + name)
        node('api_' + name, inside)
    def chunks(name):
        return math.ceil(2 * math.prod(profile['shapes'][name]) / profile['chunk_bytes'])
    def digest(name):
        for index in range(chunks(name)):
            guard('device_digest_before:' + name)
            guard('device_digest_after:' + name)
    def load():
        guard('load_titan_before')
        for name in asset_order or profile['asset_order']:
            iterations = math.ceil(assets[name]['bytes'] / profile['chunk_bytes']) + 1
            for index in range(iterations):
                guard('asset_hash_before:' + name)
                guard('asset_hash_after:' + name)
        guard('config_load_before')
        guard('config_load_after')
        def construct():
            for name in ('model_construct_before', 'model_to_empty_before', 'model_construct_after'):
                guard(name)
        api('model_construct', construct)
        def checkpoint():
            guard('checkpoint_load_before')
            for name in first_order:
                def copy(name=name):
                    for index in range(chunks(name)):
                        guard('tensor_copy_before:' + name)
                        guard('tensor_copy_after:' + name)
                api('tensor_copy', copy)
                digest(name)
            guard('checkpoint_load_after')
        api('checkpoint_load', checkpoint)
        guard('load_titan_after')
    node('main_load_titan', load)
    def final():
        guard('verify_frozen_base_before')
        for name in final_order or profile['final_digest_order']:
            digest(name)
        guard('verify_frozen_base_after')
    api('verify_frozen_base', final)
    return entries

def main():
    source_path = STAGE / 'phase_replay.py'
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    profile = phase.loader_source((PACKAGE / 'native_loader.py').read_bytes())
    # Assign offsets in reverse lexicographic order so first-copy order differs
    # both from canonical JSON order and from source shape insertion order.
    first_order = sorted(profile['shapes'], reverse=True)
    offset, entries = 0, {}
    for name in first_order:
        size = 2 * math.prod(profile['shapes'][name])
        entries[name] = {'dtype': 'BF16', 'shape': profile['shapes'][name], 'data_offsets': [offset, offset + size]}
        offset += size
    header = {'entries': entries, 'metadata': {}, 'header_bytes': 128,
              'payload_bytes': offset, 'file_bytes': offset + 136}
    assets = {name: {'bytes': 127, 'sha256': pin} for name, pin in profile['asset_pins'].items()}
    assets['model.safetensors']['bytes'] = header['file_bytes']
    loader = {'asset_hashes': dict(sorted(assets.items())),
              'header_bytes': header['header_bytes'], 'payload_bytes': offset, 'file_bytes': header['file_bytes'],
              'source_tensor_sha256': {name: 'f' * 64 for name in profile['shapes']},
              'payload_chunk_count': sum(math.ceil(2 * math.prod(shape) / profile['chunk_bytes']) for shape in profile['shapes'].values()),
              'device_digest_maximum_chunk_bytes': profile['chunk_bytes']}
    good = markers_for(profile, assets, first_order)
    positive = phase.loader_tree(good, loader, profile, header, assets)
    controls = []
    def deny(name, *, markers=None, changed_loader=None, changed_header=None, changed_assets=None):
        try:
            phase.loader_tree(markers or deepcopy(good), changed_loader or deepcopy(loader), profile,
                              changed_header or deepcopy(header), changed_assets or deepcopy(assets))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            controls.append({'id': name, 'rejected': True, 'error': str(exc)})
        else:
            controls.append({'id': name, 'rejected': False})
    deny('canonical_asset_execution_order', markers=markers_for(profile, assets, first_order, asset_order=sorted(assets)))
    wrong_first = list(first_order)
    wrong_first[0], wrong_first[1] = wrong_first[1], wrong_first[0]
    deny('same310_names_wrong_first_copy_order', markers=markers_for(profile, assets, wrong_first))
    deny('same310_names_wrong_final_digest_order', markers=markers_for(profile, assets, first_order, final_order=sorted(profile['shapes'])))
    bad_loader = deepcopy(loader)
    bad_loader['source_tensor_sha256'].pop(first_order[0])
    deny('missing_source_tensor_digest', changed_loader=bad_loader)
    bad_assets = deepcopy(assets)
    bad_assets['config.json']['sha256'] = '0' * 64
    deny('forged_asset_hash', changed_assets=bad_assets)
    bad_loader = deepcopy(loader)
    bad_loader['asset_hashes']['config.json']['bytes'] += 1
    deny('same_pins_wrong_asset_size_receipt', changed_loader=bad_loader)
    bad_header = deepcopy(header)
    bad_header['entries'][first_order[0]]['data_offsets'][1] -= 2
    deny('tensor_span_mismatch', changed_header=bad_header)
    bad_header = deepcopy(header)
    bad_header['entries'][first_order[0]]['dtype'] = 'F32'
    deny('wrong_source_dtype', changed_header=bad_header)
    bad_loader = deepcopy(loader)
    bad_loader['payload_chunk_count'] -= 1
    deny('wrong_complete_payload_chunk_denominator', changed_loader=bad_loader)
    after = hashlib.sha256(source_path.read_bytes()).hexdigest()
    forbidden = [name for name in ('torch', 'transformers', 'tokenizers') if name in sys.modules]
    receipt = {'schema_version': 1, 'kind': 'INDEPENDENT_CPU_LOADER_ORDER_MUTATION_REVIEW',
               'positive_inert_metadata': positive, 'controls': controls,
               'all_negatives_rejected': all(c['rejected'] for c in controls),
               'canonical_receipt_asset_order': list(loader['asset_hashes']), 'source_asset_order': profile['asset_order'],
               'copy_order_bound_to_inert_offsets': True, 'final_digest_order_bound_to_pinned_source': True,
               'source_before': before, 'source_after': after, 'source_bytes_unchanged': before == after,
               'forbidden_runtime_imports': forbidden, 'model_or_runtime_binary_reads': False,
               'model_GPU_decoder_encoder_execution': False, 'native_PASS_or_study_result': False}
    receipt['sha256'] = phase.digest(receipt)
    target = HERE / ('independent_loader_review_%d.json' % time.time_ns())
    target.write_text(phase.canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'negative_controls': len(controls),
                      'failed_controls': [c['id'] for c in controls if not c['rejected']],
                      'source_bytes_unchanged': before == after, 'forbidden_runtime_imports': forbidden}))
    return 0 if receipt['all_negatives_rejected'] and before == after and not forbidden else 1

if __name__ == '__main__':
    raise SystemExit(main())
