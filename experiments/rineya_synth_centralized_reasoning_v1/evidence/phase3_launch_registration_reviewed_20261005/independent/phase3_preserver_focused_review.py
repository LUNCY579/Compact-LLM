"""Focused inert copy/readback review. Never accesses real raw scope or native audit."""
import hashlib, importlib.util, io, json, sys, time, unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/phase3_launch_registration_work'
PINS = {'preserve_terminal.py': '29790d005ee98073125f35748b1878eeafbbc4f209e93d0083ab385446a0eac1',
    'test_preserve_terminal.py': 'cf56123160e3a4019f5e4e48b2b46e2a6825d99d31c71474303294b6e7b17570',
    'audit_preserved_success.py': '6367758e92d3cab07afc06da837f37b30f8db988f48428e3c2e35e8c618a07c8'}
def hashes():
    return {name: hashlib.sha256((STAGE / name).read_bytes()).hexdigest() for name in PINS}
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def main():
    before = hashes()
    assert before == PINS
    tests = load('independent_preservation_focused_tests', STAGE / 'test_preserve_terminal.py')
    wrapper = load('independent_root_only_audit_wrapper', STAGE / 'audit_preserved_success.py')
    methods = ['test_source131_flat_success_copy_and_write_once_are_CPU_only',
               'test_late_change_of_earlier_copied_target_refuses_final_receipt']
    output = io.StringIO()
    with (patch.object(tests.h.registration.subprocess, 'run', side_effect=AssertionError('real Git/OS probe forbidden')) as probe,
            patch.object(tests.h.registration.auditor, 'audit_terminal', side_effect=AssertionError('real native heavy audit forbidden')) as audit):
        suite = unittest.TestSuite(tests.Controls(name) for name in methods)
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        # Wrapper's success path is Root-only. Missing explicit Root signal must
        # reject before any preservation read or native audit.
        with (patch.object(sys, 'argv', ['audit_preserved_success.py', '--preservation-body-sha256', 'a' * 64]),
                patch.object(wrapper.preserve.pure, 'read_json', side_effect=AssertionError('preserved read before Root signal')) as read):
            try:
                wrapper.main()
            except ValueError as exc:
                assert 'explicit Root' in str(exc)
            else:
                raise AssertionError('audit wrapper accepted missing Root signal')
            assert read.call_count == 0
        assert probe.call_count == audit.call_count == 0
    after = hashes()
    forbidden = [n for n in ('torch', 'transformers', 'tokenizers', 'psutil', 'native_supervisor', 'native_process') if n in sys.modules]
    assert not forbidden
    receipt = {'schema_version': 1, 'kind': 'INDEPENDENT_FROZEN_PRESERVER_FOCUSED_CPU_COPY_READBACK_REVIEW',
        'methods': methods, 'tests_run': result.testsRun, 'successful': result.wasSuccessful(), 'skipped': len(result.skipped),
        'output': output.getvalue(), 'source_before': before, 'source_after': after, 'source_bytes_unchanged': before == after,
        'wrapper_missing_Root_signal_rejected_before_reads': True, 'actual_Git_OS_probe_and_native_audit_calls': 0,
        'actual_native_raw_leaf_read_or_modified': False, 'actual_preserved_destination_created': False,
        'frozen_six_auditor_dependencies_modified': False, 'forbidden_runtime_imports': forbidden,
        'native_PASS_or_grade_or_study_result': False}
    receipt['sha256'] = tests.p.digest(receipt)
    target = HERE / ('phase3_preserver_focused_review_%d.json' % time.time_ns())
    target.write_text(tests.p.canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'receipt_sha256': receipt['sha256'], 'tests_passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'source_bytes_unchanged': before == after, 'actual_native_probe_audit_calls': 0}))
    return 0 if result.wasSuccessful() and result.testsRun == 2 and not result.skipped and before == after else 1

if __name__ == '__main__':
    raise SystemExit(main())
