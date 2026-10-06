"""Independent authored-only interface controls; no native or data inputs."""
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import argparse, ast, builtins, hashlib, io, json, sys, tempfile, unittest

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == 'rineya_synth_centralized_reasoning_v1')
SOURCE = None
MODULE = None


class IndependentControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='owned_review_', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def context(self):
        output = self.root / 'fresh_owned_leaf'
        output.mkdir()
        config = self.root / 'authored_config.json'
        config.write_bytes(b'{"CPU_authored":true}')
        ref = {'path': str(config), 'bytes': config.stat().st_size,
               'sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
               'value_sha256': MODULE.digest({'CPU_authored': True})}
        value = {'output': str(output), 'output_root': str(self.root), 'engine_config': ref}
        context = MODULE.Context(value, None, {}, capability=MODULE._CPU_capability())
        MODULE.write_new(output / MODULE.CONTEXT, {'launch_ref': None, 'value': value})
        MODULE.write_new(output / 'supervisor_ownership.json',
                         {'config_sha256': ref['sha256'], 'worker_pid': 7001, 'supervisor_pid': 6001})
        MODULE.write_new(output / 'supervisor_heartbeat.jsonl', {'CPU_authored': True})
        # Names are taken from the actual original producer, not invented receipt keys.
        tree = ast.parse((PACKAGE / 'native_supervisor.py').read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_WindowsOwnedWorker')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        tuples = [n for n in ast.walk(init) if isinstance(n, ast.Tuple) and
                  all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in n.elts)]
        names = next(tuple(x.value for x in n.elts) for n in tuples
                     if tuple(x.value for x in n.elts) == ('worker.stdout.log', 'worker.stderr.log'))
        for name in names:
            (output / name).write_bytes(b'')
        return context, output

    def test_original_supervisor_logs_are_accepted_only_in_exact_owned_leaf(self):
        context, output = self.context()
        self.assertEqual(MODULE._output_check(context, worker=True), output)
        (output / 'foreign_unselected.txt').write_bytes(b'CPU_authored_unselected')
        with self.assertRaisesRegex(ValueError, 'foreign existing output'):
            MODULE._output_check(context, worker=True)

    def test_staged_dispatch_and_main_reject_before_any_selected_input_read(self):
        with patch.object(MODULE, 'read_ref', side_effect=AssertionError('read before stage gate')):
            with self.assertRaisesRegex(ValueError, 'staged owned STUDY runtime'):
                MODULE.run_registered({})
            with self.assertRaisesRegex(ValueError, 'staged owned STUDY worker'):
                MODULE.main()

    def test_durable_publication_is_consumed_by_original_scientific_verifier(self):
        sys.path.insert(0, str(PACKAGE))
        import study_native_chunk as chunks
        payload = {'CPU_authored': True, 'scope': 'no_native_authority'}
        path = self.root / 'scientific_publication.json'
        descriptor = MODULE.write_new(path, payload)
        self.assertEqual(set(descriptor), {'path', 'bytes', 'sha256', 'payload_sha256', 'durable'})
        self.assertEqual(chunks._verify_publication(descriptor, payload), descriptor)
        with self.assertRaises(FileExistsError):
            MODULE.write_new(path, payload)
        path.write_bytes(b'changed authored bytes')
        with self.assertRaisesRegex(ValueError, 'published bytes/hash changed'):
            chunks._verify_publication(descriptor, payload)

    def test_source_uses_exact_original_supervisor_adaptations_and_eight_callbacks(self):
        context, _ = self.context()
        context.prepared = {'config': {'resources': {}}}
        supervise, counts = MODULE.supervisor_body(
            {'_new_output': lambda p: Path(p), '_acquire_lease': lambda *a: None}, context)
        self.assertTrue(callable(supervise))
        self.assertEqual(counts, {'worker': 2, 'scope': 1})
        tree = ast.parse(SOURCE.read_bytes())
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_owned_boundaries')
        assigned = next(n for n in ast.walk(builder) if isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Name) and t.id == 'boundaries' for t in n.targets))
        self.assertIsInstance(assigned.value, ast.Dict)
        keys = {n.value for n in assigned.value.keys}
        self.assertEqual(keys, {'frameworks', 'call', 'guard', 'preserve', 'admission',
                               'evidence', 'cleanup', 'terminal_audit'})
        returns = [n for n in builder.body if isinstance(n, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertEqual(ast.unparse(returns[0].value), 'boundaries')

    def test_framework_growth_guard_precedes_original_cuda_initialization(self):
        tree = ast.parse(SOURCE.read_bytes())
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_owned_boundaries')
        frameworks = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == 'frameworks')
        body = ast.unparse(frameworks)
        self.assertLess(body.index('whole owned RSS growth reserve before CUDA'), body.index('torch.cuda.init()'))
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker_body')
        worker_source = ast.unparse(worker)
        self.assertIn('engine._boundaries(prepared)', worker_source)
        self.assertIn('engine._run(prepared, producer, boundaries)', worker_source)


def main():
    global SOURCE, MODULE
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    reads, denied, imports = [], [], []
    original_open, original_builtin, original_import = io.open, builtins.open, builtins.__import__

    def reader(path, mode='r', *a, **kw):
        if isinstance(path, (str, bytes, Path)) and not any(c in str(mode) for c in ('w', 'a', 'x', '+')):
            selected = Path(path).resolve()
            reads.append(str(selected))
            if not (selected.is_relative_to(HERE) or selected == SOURCE or
                    (selected.parent == PACKAGE and selected.suffix == '.py')):
                denied.append(str(selected))
                raise AssertionError('non-authored input read forbidden: ' + str(selected))
        return original_open(path, mode, *a, **kw)

    def importer(name, *a, **kw):
        if name.split('.', 1)[0] in {'torch', 'transformers', 'tokenizers', 'safetensors',
                                    'ctypes', 'psutil', 'numpy'}:
            imports.append(name)
            raise AssertionError('native/model import forbidden: ' + name)
        return original_import(name, *a, **kw)

    MODULE = ModuleType('independent_owned_runtime_candidate')
    MODULE.__file__ = str(SOURCE)
    log = io.StringIO()
    with patch('io.open', side_effect=reader), patch.object(builtins, 'open', side_effect=lambda p, *a, **kw:
            reader(p, *a, **kw) if isinstance(p, (str, bytes, Path)) else original_builtin(p, *a, **kw)), \
            patch.object(builtins, '__import__', side_effect=importer):
        exec(compile(raw, str(SOURCE), 'exec'), MODULE.__dict__)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(IndependentControls))
    value = {'kind': 'INDEPENDENT_OWNED_STUDY_FOCUSED_INTERFACE_CONTROLS',
             'source_sha256': args.sha256, 'source_unchanged': hashlib.sha256(SOURCE.read_bytes()).hexdigest() == args.sha256,
             'tests': result.testsRun, 'success': result.wasSuccessful(), 'reads': len(reads),
             'denied_inputs': denied, 'forbidden_imports': imports, 'log': log.getvalue(),
             'fixture_only': True, 'native_PASS': False, 'native_run': False,
             'quality_gain_established': False,
             'limits': 'Source adaptation and authored filesystem/original publication protocol only; no Windows or model execution.'}
    value['sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode('ascii') + b'\n'
    args.receipt.open('xb').write(data)
    print(json.dumps({'receipt': str(args.receipt), 'file_sha256': hashlib.sha256(data).hexdigest(),
                      'body_sha256': value['sha256'], 'tests': result.testsRun,
                      'success': result.wasSuccessful(), 'source_unchanged': value['source_unchanged'],
                      'denied_inputs': denied, 'forbidden_imports': imports}))
    if not result.wasSuccessful():
        print(log.getvalue())
    return 0 if result.wasSuccessful() and value['source_unchanged'] and not denied and not imports else 1


if __name__ == '__main__':
    raise SystemExit(main())
