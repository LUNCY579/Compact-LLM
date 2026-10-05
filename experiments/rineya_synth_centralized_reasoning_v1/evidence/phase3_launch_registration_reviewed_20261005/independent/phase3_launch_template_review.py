"""Independent template/future-freeze controls; real Git and writes forbidden."""
from copy import deepcopy
import hashlib, importlib.util, io, json, sys, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/phase3_launch_registration_work'
EXPECTED = {'INVENTORY_TEMPLATE.json': '9d316cabd8170576030d0f94cbb70453341a5248cfe74955670d2aa6beb89ba4',
    'prepare_registration.py': '3e9962b5c6028263ed23c98c07d4a0e9976ead63b3092f912ead0571b0fd4f66',
    'test_registration.py': '5634dcdc1add9e9de5ccfe3b07cf75aaaa895a32bcc8817e2cb4e5039aaac64c'}
FINAL_HEAD = 'dc4e52ef3a7b6b6cde51f1ab54a68b2d63be5510'
def hashes():
    return {name: hashlib.sha256((STAGE / name).read_bytes()).hexdigest() for name in EXPECTED}

def main():
    before = hashes()
    assert before == EXPECTED
    spec = importlib.util.spec_from_file_location('independent_phase3_registration_prepare', STAGE / 'prepare_registration.py')
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    cfg = prepare.scope()
    template = prepare.pure.read_json(prepare.TEMPLATE)
    check_fields = {'schema_version', 'id', 'contract', 'root_path', 'package_path', 'config_path',
        'config_file_sha256', 'profile_body_sha256', 'source_commit', 'inventories', 'counts',
        'qualified_runtime64_sha256', 'auditor_dependency_pins', 'sha256'}
    assert set(template) == check_fields
    assert template['sha256'] == 'c3b02346bc45f49e18b782b869ecb123a635e4fcae5378673a80882349bf3ba7'
    sections = {'source': 'source_pins', 'runtime': 'runtime_pins', 'decoder': 'decoder_runtime_pins', 'data': 'data_pins', 'model': 'model_pins'}
    assert template['inventories'] == {name: cfg[key] for name, key in sections.items()}
    assert template['counts'] == {'source': 131, 'runtime': 65, 'decoder': 17, 'data': 5, 'model': 7}
    assert len(template['auditor_dependency_pins']) == 6
    assert template['qualified_runtime64_sha256'] == '7cc9daa0ed13d78c2c78583902b515066fca579a3dd47d054d59c1cdc2e88d71'
    results = []
    # Actual template/source/admission methods execute. Any real Git subprocess
    # or final publication is an immediate control failure.
    with (patch.object(prepare.subprocess, 'run', side_effect=AssertionError('real Git/probe forbidden')) as real_git,
            patch.object(prepare, 'write_final', side_effect=AssertionError('owned final registration write forbidden')) as writer):
        positive = prepare.validate_template(template, cfg)
        assert positive['launch_HEAD_frozen'] is positive['native_launch_authorized'] is False
        for section in sections:
            bad = deepcopy(template)
            entries = bad['inventories'][section]
            name = next(iter(entries))
            pin = entries.pop(name)
            entries['CPU_REPLACEMENT_SAME_COUNT'] = pin
            bad = prepare.seal(bad)
            try:
                prepare.validate_template(bad, cfg)
            except ValueError as exc:
                results.append({'id': 'same_count_' + section, 'passed': True, 'error': str(exc)})
            else:
                results.append({'id': 'same_count_' + section, 'passed': False})
        try:
            with patch.object(prepare.pure, 'read_bytes', side_effect=AssertionError('read before placeholder rejection')):
                prepare.auditor.prerequisites(terminal_signal=prepare.auditor.SIGNAL, inventory_path='UNUSED',
                    inventory_sha='a' * 64, config_sha=prepare.CONFIG_SHA, profile_sha=prepare.PROFILE_SHA,
                    preservation_sha='b' * 64, source_commit=template['source_commit'])
        except ValueError as exc:
            results.append({'id': 'public_native_placeholder_denial_before_reads', 'passed': True, 'error': str(exc)})
        else:
            results.append({'id': 'public_native_placeholder_denial_before_reads', 'passed': False})
        output = io.StringIO()
        with patch.object(sys, 'argv', ['prepare_registration.py']), redirect_stdout(output):
            prepare.main()
        assert json.loads(output.getvalue())['native_launch_authorized'] is False
        results.append({'id': 'default_CLI_no_Git_no_owned_write', 'passed': real_git.call_count == writer.call_count == 0})
        # Run git_identity itself against explicit inert command responses. Each
        # negative changes only one substantive gate, not its input length alone.
        for mode in ('valid', 'dirty', 'different_HEAD', 'committed_config_drift', 'only8hex', 'nonhex40'):
            expected = FINAL_HEAD if mode not in ('only8hex', 'nonhex40') else ('dc4e52ef' if mode == 'only8hex' else 'z' * 40)
            calls = []
            def git(command, **kwargs):
                calls.append(command)
                assert kwargs.get('cwd') == prepare.ROOT
                if command[1] == 'status':
                    return SimpleNamespace(stdout=' M CPU_SOURCE.py\n' if mode == 'dirty' else '')
                if command[1] == 'rev-parse':
                    return SimpleNamespace(stdout='a' * 40 if mode == 'different_HEAD' else FINAL_HEAD)
                assert command == ['git', 'show', FINAL_HEAD + ':' + prepare.CONFIG.relative_to(prepare.ROOT).as_posix()]
                return SimpleNamespace(stdout=b'ALTERED_CPU_CONFIG\n' if mode == 'committed_config_drift' else prepare.pure.read_bytes(prepare.CONFIG))
            try:
                with patch.object(prepare.subprocess, 'run', side_effect=git):
                    head = prepare.git_identity(expected)
            except ValueError as exc:
                results.append({'id': 'git_gate_' + mode, 'passed': mode != 'valid', 'error': str(exc), 'mocked_commands': len(calls)})
            else:
                results.append({'id': 'git_gate_' + mode, 'passed': mode == 'valid' and head == FINAL_HEAD and len(calls) == 3,
                    'mocked_commands': len(calls)})
        def valid_git(command, **kwargs):
            if command[1] == 'status':
                return SimpleNamespace(stdout='')
            if command[1] == 'rev-parse':
                return SimpleNamespace(stdout=FINAL_HEAD)
            return SimpleNamespace(stdout=prepare.pure.read_bytes(prepare.CONFIG))
        with patch.object(prepare.subprocess, 'run', side_effect=valid_git):
            final = prepare.final_body(template, cfg, FINAL_HEAD)
        differences = {key for key in template if template[key] != final[key]}
        assert differences == {'source_commit', 'sha256'} and final['source_commit'] == FINAL_HEAD
        assert writer.call_count == real_git.call_count == 0
        results.append({'id': 'final_body_source_only_no_write_only_HEAD_and_checksum_change', 'passed': True})
    after = hashes()
    forbidden = [name for name in ('torch', 'transformers', 'tokenizers', 'psutil') if name in sys.modules]
    receipt = {'schema_version': 1, 'kind': 'INDEPENDENT_PHASE3_TEMPLATE_SOURCE_AND_FREEZE_SEMANTICS_CPU_REVIEW',
        'positive_template': positive, 'controls': results, 'all_controls_passed': all(r['passed'] for r in results),
        'source_before': before, 'source_after': after, 'source_bytes_unchanged': before == after,
        'root_supplied_final_launch_HEAD': FINAL_HEAD, 'actual_HEAD_or_committed_blob_queried': False,
        'real_Git_or_OS_probe_calls': 0, 'owned_final_registry_write_calls': 0,
        'final_body_is_inert_control_only': True, 'forbidden_runtime_imports': forbidden,
        'runtime_model_decoder_binary_reads_or_execution': False, 'native_ready': False, 'native_PASS_or_study_result': False}
    receipt['sha256'] = prepare.pure.digest(receipt)
    target = HERE / ('phase3_launch_template_review_%d.json' % time.time_ns())
    target.write_text(prepare.pure.canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'controls': len(results), 'failed_controls': [r['id'] for r in results if not r['passed']],
        'source_bytes_unchanged': before == after, 'real_Git_probe_and_final_write_calls': 0, 'native_ready': False}))
    return 0 if receipt['all_controls_passed'] and before == after and not forbidden else 1

if __name__ == '__main__':
    raise SystemExit(main())
