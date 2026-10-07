"""Real CPU writer/core -> preserved bytes -> terminal consumer replay.

Native frameworks/APIs and native worker actions are never executed here.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
GRAPH = HERE / 'final_placement_category004'
PRODUCER_PINS = {'prefix_input_owned_worker.py': 'b71075e252b5d2c18fe32644d3d22257131ba23a931cdedc52597f4f43e02afc',
    'prefix_native_pipeline.py': '9efb6f15444a6449734fe85d5eddd72d3cefb2a76c0b7af46264b44ed68cbb2b',
    'prefix_tensor_contract.py': '0d12fd01bb1f06c29054943a3a114efc6eac9619c33989e5ce030d97a3cf4263'}
FIXTURE_LAYOUT = HERE.parent / 'entry_auditor_source_fixture_category004'
FIXTURE_LAYOUT.mkdir(exist_ok=True)
for _name in ('prefix_input_owned_worker.py', 'prefix_native_pipeline.py', 'prefix_tensor_contract.py', 'registration_git_join.py'):
    _source = (GRAPH / _name).read_bytes()
    if _name in PRODUCER_PINS:
        assert hashlib.sha256(_source).hexdigest() == PRODUCER_PINS[_name]
    _target = FIXTURE_LAYOUT / _name
    if _target.exists():
        assert _target.read_bytes() == _source
    else:
        _target.write_bytes(_source)


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    raw = path.read_bytes()
    module.__compiled_source_descriptor__ = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'role': 'python_source'}
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


class NoNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('ctypes', 'torch', 'psutil', 'native_worker'):
            raise AssertionError('native import forbidden: ' + fullname)


TRAP = NoNative()
sys.meta_path.insert(0, TRAP)
W = load(FIXTURE_LAYOUT / 'prefix_input_owned_worker.py', 'CPU_entry_audit_worker')
P = load(FIXTURE_LAYOUT / 'prefix_native_pipeline.py', 'CPU_entry_audit_pipeline')
A = load(HERE / 'prefix_input_owned_auditor.py', 'CPU_entry_audit_consumer')
F = load(W.C.CORE / 'fixture_support.py', 'CPU_entry_audit_inert_api')
PREFIX = json.loads(W.source_core()['C'].INPUT.read_bytes())['prefix_ids']


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    path.write_text(A.canonical(value) + '\n', encoding='utf-8')


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='CPU_entry_audit_')
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root / 'original'
        self.output.mkdir()
        self.preserved = self.root / 'preserved'
        self.ledger, self.phases = W._entry_journal_bootstrap(self.output)
        self.selected = {item['path']: {'bytes': item['bytes'], 'sha256': item['sha256']}
            for item in self.ledger.entry_bootstrap_sources}
        self.selected[str(FIXTURE_LAYOUT / 'prefix_tensor_contract.py')] = descriptor(FIXTURE_LAYOUT / 'prefix_tensor_contract.py')
        self.report = None
        finish = self.ledger.finish_owned
        def capture(report):
            self.report = report
            return finish(report)
        self.ledger.finish_owned = capture

    def tearDown(self):
        if not self.ledger.closed:
            self.ledger.close()
        self.tmp.cleanup()

    def run_entry(self, failed=None, cleanup=False, publication=False, mixed=False):
        actions = {name: lambda state, costs: None for name in self.phases}
        if mixed:
            source = self.root / 'inert.py'
            cache = self.root / 'inert.pyc'
            source.write_bytes(b'VALUE=7\n')
            cache.write_bytes(b'cpu-cache-only')
            def source_action(state, costs):
                contract = load(FIXTURE_LAYOUT / 'prefix_tensor_contract.py', 'CPU_mixed_contract')
                state['source_meter'].bind_contract(contract)
                contract.read_bytes(source)
                contract.file_descriptor(source, {**descriptor(source), 'role': 'python_source'})
                contract.read_bytes(cache)
            actions[self.phases[0]] = source_action
            actions[self.phases[-1]] = lambda state, costs: state['source_meter'].restore()
        if failed is not None:
            actions[self.phases[failed]] = lambda *args: (_ for _ in ()).throw(RuntimeError('selected source failure'))
        if cleanup:
            actions[self.phases[-1]] = lambda *args: (_ for _ in ()).throw(OSError('owned cleanup failure'))
        if publication:
            durable = self.ledger.durable
            def fail_final(name, value):
                if name == 'PHASE_ACCOUNTING_FINAL.json':
                    raise OSError('final publication failure')
                return durable(name, value)
            self.ledger.durable = fail_final
        def execute(state, ledger):
            run, namespace, proof = P._bind_shared_core_executor(W, ledger)
            api = F.InjectedAPI()
            self.report = run(api, api.guard, self.output)
            return self.report
        try:
            self.report = W._entry_phase_body(self.ledger, self.phases, actions, execute)
        except (RuntimeError, OSError):
            # The actual owner retains the object passed to finish_owned even
            # when the entry function raises. No missing tensor facts invented.
            self.assertIsNotNone(self.report)
        write(self.output / 'CPU_ENTRY_RETURN.json', self.report)
        shutil.copytree(self.output, self.preserved)

    def audit(self):
        raw = {name: descriptor(self.output / name) for name in A.RAW_CORE if (self.output / name).exists()}
        return A._CPU_shared_entry_bytes(self.preserved, original_directory=self.output,
            raw_files=raw, report_file='CPU_ENTRY_RETURN.json',
            report_descriptor=descriptor(self.output / 'CPU_ENTRY_RETURN.json'), prefix=PREFIX,
            selected_source_pins=self.selected)

    def rewrite_pair(self, name, value):
        for root in (self.output, self.preserved):
            write(root / name, value)

    def test_actual_full_core_writer_preserved_reader_and_replay(self):
        self.run_entry()
        value = self.audit()
        self.assertTrue(value['source_outcome_completed'])
        self.assertFalse(value['resource_Job_death_or_library_evidence_validated'])
        self.assertFalse(value['native_authority_issued'])

    def test_actual_mixed_source_cache_cost_derivation_and_resealed_forgery(self):
        self.run_entry(mixed=True)
        value = self.audit()
        self.assertTrue(value['source_outcome_completed'])
        final = json.loads((self.output / 'PHASE_ACCOUNTING_FINAL.json').read_bytes())
        costs = final['startup_subcosts'][self.phases[0]]
        self.assertEqual(costs, {'source_reads': 2, 'source_bytes': 16, 'cache_reads': 1,
            'cache_bytes': 14, 'API_calls': 0, 'IO_calls': 3})
        report = deepcopy(self.report)
        final['startup_subcosts'][self.phases[0]]['cache_bytes'] += 1
        report.update(final_phase_summary=final, startup_subcosts=final['startup_subcosts'])
        rows = [json.loads(line) for line in (self.output / 'PHASE_OBSERVATIONS.jsonl').read_bytes().splitlines()]
        for row in rows:
            if row['operation'] == self.phases[0] and row['event'] == 'returned':
                row['result']['startup_subcosts'] = final['startup_subcosts'][self.phases[0]]
        raw = b''.join((A.canonical(row) + '\n').encode('utf-8') for row in rows)
        final['marker_bytes'] = len(raw)
        report['final_phase_summary'] = final
        for root in (self.output, self.preserved):
            (root / 'PHASE_OBSERVATIONS.jsonl').write_bytes(raw)
        self.rewrite_pair('PHASE_ACCOUNTING_FINAL.json', final)
        self.rewrite_pair('CPU_ENTRY_RETURN.json', report)
        with self.assertRaisesRegex(ValueError, 'independently derived'):
            self.audit()

    def test_resealed_cost_join_forgery_refused(self):
        self.run_entry()
        final = json.loads((self.output / 'PHASE_ACCOUNTING_FINAL.json').read_bytes())
        report = deepcopy(self.report)
        final['startup_subcosts'][self.phases[0]]['source_bytes'] = 1
        report.update(final_phase_summary=final, startup_subcosts=final['startup_subcosts'])
        self.rewrite_pair('PHASE_ACCOUNTING_FINAL.json', final)
        self.rewrite_pair('CPU_ENTRY_RETURN.json', report)
        with self.assertRaisesRegex(ValueError, 'durable returned marker startup costs'):
            self.audit()

    def test_resealed_order_and_source_pin_override_refused(self):
        self.run_entry()
        report = deepcopy(self.report)
        report['entry_bootstrap_sources'][0]['sha256'] = '0' * 64
        self.rewrite_pair('CPU_ENTRY_RETURN.json', report)
        with self.assertRaisesRegex(ValueError, 'actual called source pin'):
            self.audit()
        self.rewrite_pair('CPU_ENTRY_RETURN.json', self.report)
        final = json.loads((self.output / 'PHASE_ACCOUNTING_FINAL.json').read_bytes())
        report = deepcopy(self.report)
        final['whole_scope_order'][0:2] = list(reversed(final['whole_scope_order'][0:2]))
        report.update(final_phase_summary=final, whole_scope_order=final['whole_scope_order'])
        self.rewrite_pair('PHASE_ACCOUNTING_FINAL.json', final)
        self.rewrite_pair('CPU_ENTRY_RETURN.json', report)
        with self.assertRaisesRegex(ValueError, 'actual first-attempt whole scope order'):
            self.audit()

    def test_actual_precore_failure_absent_worker_result_is_partial(self):
        self.run_entry(failed=0)
        self.assertFalse((self.output / 'worker_result.json').exists())
        value = self.audit()
        self.assertFalse(value['source_outcome_completed'])
        self.assertTrue(value['durable_final_summary_present'])

    def test_actual_cleanup_failure_retains_failed_outcome(self):
        self.run_entry(cleanup=True)
        value = self.audit()
        self.assertFalse(value['source_outcome_completed'])
        self.assertEqual(value['operations'][self.phases[-1]]['failed'], 1)

    def test_actual_final_publication_failure_is_never_complete(self):
        self.run_entry(publication=True)
        self.assertFalse((self.output / 'PHASE_ACCOUNTING_FINAL.json').exists())
        value = self.audit()
        self.assertFalse(value['source_outcome_completed'])
        self.assertFalse(value['durable_final_summary_present'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    sources = [HERE / 'prefix_input_owned_auditor.py', Path(__file__).resolve()] + [GRAPH / name for name in
        ('prefix_input_owned_worker.py', 'prefix_native_pipeline.py', 'prefix_tensor_contract.py')]
    for name, pin in PRODUCER_PINS.items():
        assert hashlib.sha256((GRAPH / name).read_bytes()).hexdigest() == pin
    receipt = {'kind': 'CPU_SHARED_ENTRY_TERMINAL_CONSUMER_CONTROLS_PRODUCER004', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'all_controls_completed': result.wasSuccessful(),
        'actual_original_writer_core_and_consumer_read_replay': True,
        'native_API_or_framework_called': False, 'whole_native_terminal_consumer_qualified': False,
        'native_authority_issued': False, 'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    if len(sys.argv) > 1:
        with Path(sys.argv[1]).open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
