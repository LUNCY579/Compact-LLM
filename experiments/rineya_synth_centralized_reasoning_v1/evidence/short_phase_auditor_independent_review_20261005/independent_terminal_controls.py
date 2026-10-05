"""Re-signed filesystem/source/terminal controls, without native or binary IO."""
from copy import deepcopy
import hashlib, importlib.util, json, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/short_phase_audit_work'
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(PACKAGE))
import audit_adapter as adapter
import phase_replay as phase
import fixture_support as fixtures

def source_hashes():
    files = list(STAGE.glob('*.py')) + [adapter.DEPENDENCY, STAGE / 'pinned_dependencies/PRESERVATION.json']
    return {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def write(path, value):
    path.write_bytes((phase.canonical(value) + '\n').encode('ascii'))

def seal(value):
    value['sha256'] = phase.digest({k: v for k, v in value.items() if k != 'sha256'})
    return value

def reseal(frame):
    package, config, leaf, saved, registration, unused_config, unused_pres, unused_profile, profile, header, assets = frame
    scope = phase.read_json(config)
    write(config, seal(scope))
    config_sha = phase.file_info(config)['sha256']
    worker = phase.read_json(leaf / 'worker_result.json')
    worker['config_sha256'] = config_sha
    write(leaf / 'worker_result.json', worker)
    sup = phase.read_json(leaf / 'SUPERVISOR_RECEIPT.json')
    sup['config_sha256'] = config_sha
    summary = phase.read_json(leaf / 'PHASE_ACCOUNTING_FINAL.json')
    summary['identity']['config_sha256'] = config_sha
    write(leaf / 'PHASE_ACCOUNTING_FINAL.json', seal(summary))
    sup['partial_outputs'] = [{'path': p.name, **phase.file_info(p)}
        for p in sorted(leaf.iterdir()) if p.is_file() and p.name != 'SUPERVISOR_RECEIPT.json']
    write(leaf / 'SUPERVISOR_RECEIPT.json', sup)
    lease = Path(sup['lease_path'])
    reservation = phase.read_json(lease / 'reservation.json')
    reservation['config_sha256'] = config_sha
    write(lease / 'reservation.json', reservation)
    (lease / 'terminal.json').write_bytes((leaf / 'SUPERVISOR_RECEIPT.json').read_bytes())
    (saved / 'config_snapshot.json').write_bytes(config.read_bytes())
    pres = phase.read_json(saved / 'preservation_receipt.json')
    pres['config_snapshot'] = phase.file_info(config)
    pres['raw_files'] = {p.name: phase.file_info(p) for p in leaf.iterdir() if p.is_file()}
    for p in leaf.iterdir():
        if p.is_file():
            (saved / p.name).write_bytes(p.read_bytes())
    write(saved / 'preservation_receipt.json', seal(pres))
    registration.update(config_file_sha256=config_sha)
    seal(registration)
    return [package, config, leaf, saved, registration, config_sha, pres['sha256'],
            phase.digest(scope['short_phase_observation_profile']), profile, header, assets]

def context(frame):
    p, c, l, s, r, ch, ph, pf, profile, header, assets = frame
    with patch.object(adapter, 'PACKAGE', p):
        return adapter._terminal_context(c, l, s, r, ('atomic_RS_DEV', 4), ch, ph, pf, 'a' * 40, fixture=True)

def mutate_json(frame, name, change):
    path = frame[2] / name
    value = phase.read_json(path)
    change(value)
    write(path, value)

def main():
    before = source_hashes()
    spec = importlib.util.spec_from_file_location('terminal_review_supplier', PACKAGE / 'tests/test_short_phase_collection.py')
    supplier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supplier)
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        unused, cuda, markers, summary, worker, journal, identity = supplier.complete_collection(directory)
    values = fixtures.native_shaped((markers, summary, worker, journal))
    results = []
    source, changes = adapter.prepared_source()
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        frame = fixtures.filesystem(directory, values)
        ctx = context(frame)
        p, c, l, s, r, ch, ph, pf, profile, header, assets = frame
        with patch.object(adapter, 'PACKAGE', p), patch.object(adapter.subprocess, 'run',
                side_effect=lambda command, **kwargs: SimpleNamespace(stdout='' if command[1] == 'status' else 'a' * 40)):
            positive = adapter._finish_composition(ctx, {'status': 'CPU_INERT_HEAVY_CLOSURE', 'scoring_authorized': False},
                source, changes, profile, header, assets, r, 'a' * 40, fixture=True)
            try:
                adapter._finish_composition(ctx, {'status': 'CPU_INERT_HEAVY_CLOSURE', 'scoring_authorized': False},
                    source, changes, profile, header, assets, r, 'a' * 40)
            except ValueError as exc:
                results.append({'id': 'CPU_context_native_finish_conversion', 'rejected': True, 'error': str(exc)})
            else:
                results.append({'id': 'CPU_context_native_finish_conversion', 'rejected': False})
        assert positive['status'] == 'PASS_CPU_FIXTURE_TERMINAL_COMPOSITION_ONLY'
        assert positive['native_raw_audit_complete'] is False
    def negative(name, change):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            frame = list(fixtures.filesystem(directory, values))
            change(frame)
            frame = reseal(frame)
            try:
                context(frame)
            except (ValueError, KeyError, TypeError, IndexError, FileNotFoundError) as exc:
                results.append({'id': name, 'rejected': True, 'error': str(exc)})
            else:
                results.append({'id': name, 'rejected': False})
    negative('runtime64_core_digest_forged', lambda f: f[4].update(qualified_runtime64_sha256='0' * 64))
    negative('inventory_count_boolean', lambda f: f[4]['counts'].update(source=True))
    negative('inventory_hash_not_hex', lambda f: f[4]['inventories']['decoder'].__setitem__(next(iter(f[4]['inventories']['decoder'])), 'z' * 64))
    def bad_extension(f):
        value = phase.read_json(f[1])
        value['short_RAM_monitor_profile']['runtime_extension_pins'] = {'UNREGISTERED/other.py': 'f' * 64}
        write(f[1], value)
    negative('AST_extension_wrong_path', bad_extension)
    negative('reversed_CPU_GPU_sample_order', lambda f: mutate_json(f, 'RESOURCE_OBSERVATIONS.json', lambda v: v.reverse()))
    negative('negative_worker_wall_seconds', lambda f: mutate_json(f, 'worker_result.json', lambda v: v.update(wall_seconds=-1.)))
    negative('fractional_CPU_byte_amount', lambda f: mutate_json(f, 'RESOURCE_OBSERVATIONS.json',
        lambda v: v[0].update(host_available_bytes=float(4 * 1024**3) + .5)))
    negative('cleanup_unjoined_sampler', lambda f: mutate_json(f, 'SUPERVISOR_RECEIPT.json', lambda v: v.update(unjoined_sampler_threads=['CPU_FAKE_LIVE_THREAD'])))
    negative('cleanup_owned_identity_alive', lambda f: mutate_json(f, 'SUPERVISOR_RECEIPT.json', lambda v: v.update(owned_worker_identity_currently_alive=True)))
    negative('cleanup_NVML_unverified', lambda f: mutate_json(f, 'SUPERVISOR_RECEIPT.json', lambda v: v.update(gpu_telemetry_cleanup_verified=False)))
    negative('cleanup_Job_not_closed', lambda f: mutate_json(f, 'SUPERVISOR_RECEIPT.json', lambda v: v.update(job_closed_kill_on_close=False)))
    negative('terminal_GPU_diagnostic_disagrees', lambda f: mutate_json(f, 'worker_GPU_final.json', lambda v: v.update(free_bytes_final=7 * 1024**3)))
    negative('worker_config_source_identity_mismatch', lambda f: mutate_json(f, 'worker_result.json', lambda v: v.update(source_commit='b' * 40)))
    negative('final_phase_self_authorizes_native_fit', lambda f: mutate_json(f, 'PHASE_ACCOUNTING_FINAL.json', lambda v: v.update(native_fit_established=True)))
    negative('final_phase_self_authorizes_scoring', lambda f: mutate_json(f, 'PHASE_ACCOUNTING_FINAL.json', lambda v: v.update(scoring_authorized=True)))
    def prompt_oracle(f):
        value = phase.read_json(f[1])
        value['rows'][0]['oracle_target'] = 'CPU_EXPOSED_UNREGISTERED_LABEL'
        value['source_packet_sha256'] = phase.digest(value['rows'])
        write(f[1], value)
        mutate_json(f, 'worker_result.json', lambda v: v['collection'].update(source_packet_sha256=value['source_packet_sha256']))
    negative('prompt_input_contains_oracle_target', prompt_oracle)
    negative('extra_raw_file_denominator', lambda f: write(f[2] / 'CPU_EXTRA_UNREGISTERED.json', {'CPU_EXTRA': True}))
    def missing_fragment(f):
        path = next(f[2].glob('stage_raw_row*.json'))
        if path.resolve().parent != f[2].resolve():
            raise ValueError('owned CPU temporary fragment path')
        path.unlink()
    negative('missing_raw_token_fragment_denominator', missing_fragment)
    def root_registration(directory):
        package = Path(directory) / 'FAKE_ROOT/experiments/CPU_SOURCE_PACKAGE'
        package.mkdir(parents=True)
        root = package.parents[1]
        for p in [package / 'short_phase_contract.py', package / 'PILOT_DECISIONS.json',
                  package / 'pilot_trainable_inventory.json', package / 'NATIVE_QUALIFICATION_PROPOSAL.json',
                  package / 'AGENTS.md', root / 'AGENTS.md', root / 'RINEYA_SYNTH_CENTRALIZED_REASONING.md']:
            p.write_bytes(b'CPU_INERT_SOURCE_IDENTITY_ONLY\n')
        pins = {str(p.resolve()): phase.file_info(p)['sha256'] for p in package.iterdir() if p.is_file()}
        pins.update({str((root / name).resolve()): phase.file_info(root / name)['sha256'] for name in ('AGENTS.md', 'RINEYA_SYNTH_CENTRALIZED_REASONING.md')})
        dependencies = [STAGE / name for name in ('audit_adapter.py', 'phase_replay.py', 'test_phase_audit.py')]
        # fixture_support is a CPU-only supplier; if registered by the reviewed
        # native module include its exact byte identity in this test anchor.
        if 'fixture_support.py' in (STAGE / 'audit_adapter.py').read_text():
            dependencies.append(STAGE / 'fixture_support.py')
        dependencies += [adapter.DEPENDENCY, STAGE / 'pinned_dependencies/PRESERVATION.json']
        registration = {'root_path': str(root.resolve()), 'package_path': str(package.resolve()),
            'source_commit': 'a' * 40, 'config_file_sha256': 'b' * 64, 'profile_body_sha256': 'c' * 64,
            'inventories': {'source': pins}, 'auditor_dependency_pins': {str(p.resolve()): phase.file_info(p)['sha256'] for p in dependencies}}
        target = package / 'artifacts/short_m0_audit_registrations/CPU_INERT.json'
        target.parent.mkdir(parents=True)
        write(target, seal(registration))
        return package, registration, target
    for mode in ('positive', 'source_omitted', 'source_changed', 'auditor_dependency_omitted', 'dirty_HEAD', 'different_HEAD', 'alias_module'):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            p, r, target = root_registration(directory)
            if mode == 'source_omitted':
                r['inventories']['source'].pop(next(iter(r['inventories']['source'])))
            if mode == 'source_changed':
                next(iter(r['inventories']['source']))
                (p / 'short_phase_contract.py').write_bytes(b'CHANGED_CPU_SOURCE\n')
            if mode == 'auditor_dependency_omitted':
                r['auditor_dependency_pins'].pop(str(adapter.DEPENDENCY.resolve()))
            write(target, seal(r))
            def git(command, **kwargs):
                if command[1] == 'status':
                    return SimpleNamespace(stdout=' M CPU_SOURCE.py\n' if mode == 'dirty_HEAD' else '')
                return SimpleNamespace(stdout=('d' if mode == 'different_HEAD' else 'a') * 40)
            cache = SimpleNamespace(__file__=str((STAGE if mode == 'alias_module' else p) / 'short_phase_contract.py'))
            try:
                with patch.object(adapter, 'PACKAGE', p), patch.object(adapter.subprocess, 'run', side_effect=git), patch.dict(sys.modules, {'short_phase_contract': cache}):
                    adapter.prerequisites(terminal_signal=adapter.SIGNAL, inventory_path=target,
                        inventory_sha=phase.file_info(target)['sha256'], profile_sha='c' * 64, config_sha='b' * 64,
                        preservation_sha='e' * 64, source_commit='a' * 40)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                results.append({'id': mode, 'rejected': True, 'expected_outcome': 'accepted' if mode == 'positive' else 'rejected',
                    'passed': mode != 'positive', 'error': str(exc)})
            else:
                results.append({'id': mode, 'rejected': False, 'expected_outcome': 'accepted' if mode == 'positive' else 'rejected',
                    'passed': mode == 'positive'})
    forbidden = [name for name in ('torch', 'transformers', 'tokenizers') if name in sys.modules]
    after = source_hashes()
    receipt = {'schema_version': 1, 'kind': 'INDEPENDENT_CPU_TERMINAL_FILESYSTEM_AND_ADMISSION_REVIEW',
        'positive_composition': positive, 'controls': results, 'all_controls_expected': all(r.get('passed', r['rejected']) for r in results),
        'source_before': before, 'source_after': after, 'source_bytes_unchanged': before == after,
        'forbidden_runtime_imports': forbidden, 'real_Git_queries': False, 'real_heavy_inherited_audit': False,
        'model_runtime_binary_reads_or_GPU_decoder_encoder_execution': False, 'native_PASS_or_study_result': False}
    receipt['sha256'] = phase.digest(receipt)
    target = HERE / ('independent_terminal_review_%d.json' % time.time_ns())
    write(target, receipt)
    print(json.dumps({'receipt': str(target), 'controls': len(results),
        'failed_controls': [r['id'] for r in results if not r.get('passed', r['rejected'])], 'source_bytes_unchanged': before == after,
        'forbidden_runtime_imports': forbidden}))
    return 0 if receipt['all_controls_expected'] and before == after and not forbidden else 1

if __name__ == '__main__':
    raise SystemExit(main())
