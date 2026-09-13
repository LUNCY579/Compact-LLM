"""Small soup checks. Default tests use only stdlib; no model or torch import.

Set RAB001_TEST_TORCH=1 only after hostbudget permits a tiny CPU tensor test.
"""
import ast
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
NAMES = {'normalise', 'tensor_header', 'slices', 'check_outputs', 'paired',
         'assert_disjoint', 'json_rows', 'write_json', 'freeze_dev',
         'arm_spec', 'expected_trainables', 'verify_bound_files'}
tree = ast.parse((HERE / 'model_soup.py').read_text(encoding='utf-8'))
pure = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES],
                  type_ignores=[])
scope = {'Path': Path, 'json': json, 'math': math, 'struct': struct, 'Fraction': Fraction, 're': re,
         '_WS': re.compile(r'\s+'), 'CHUNK_ELEMENTS': 8,
         'sha256_file': lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
         'canonical_hash': lambda data: hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}
exec(compile(pure, str(HERE / 'model_soup.py'), 'exec'), scope)


def fixture(path, header, data=b'\0' * 16):
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + data)


class SoupPureTests(unittest.TestCase):
    def test_inventory_rejects_truncation_and_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x.safetensors'
            header = {'a': {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]}}
            fixture(p, header)
            self.assertEqual(scope['tensor_header'](p), header)
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                scope['tensor_header'](p)
            header['a']['shape'] = [3, 2]
            fixture(p, header)
            with self.assertRaises(ValueError):
                scope['tensor_header'](p)

    def test_inventory_rejects_overlaps_and_nonfloat(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'x.safetensors'
            header = {'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]},
                      'b': {'dtype': 'F32', 'shape': [2], 'data_offsets': [4, 12]}}
            fixture(p, header)
            with self.assertRaises(ValueError):
                scope['tensor_header'](p)
            header = {'a': {'dtype': 'BF16', 'shape': [8], 'data_offsets': [0, 16]}}
            fixture(p, header)
            with self.assertRaises(ValueError):
                scope['tensor_header'](p)

    def test_chunk_partition_handles_matrix_and_vector_tails(self):
        self.assertEqual([(x.start, x.stop) for x in scope['slices']([5, 3])], [(0, 2), (2, 4), (4, 5)])
        self.assertEqual([(x.start, x.stop) for x in scope['slices']([17])], [(0, 8), (8, 16), (16, 17)])

    def test_controls_compare_termination_and_exact_inventory(self):
        expected = {'x': {'text': '3', 'terminated': True}}
        self.assertTrue(scope['check_outputs']([{'id': 'x', **expected['x']}], expected, 'test')['passed'])
        with self.assertRaises(ValueError):
            scope['check_outputs']([{'id': 'x', 'text': '3', 'terminated': False}], expected, 'test')
        with self.assertRaises(ValueError):
            scope['check_outputs']([], expected, 'test')

    def test_paired_boundary_and_no_difference(self):
        a = {str(i): {'correct': True} for i in range(6)}
        b = {str(i): {'correct': False} for i in range(6)}
        result = scope['paired'](a, b)
        self.assertEqual(result['p_exact'], '1/32')
        self.assertTrue(result['soup_improvement_demonstrated'])
        self.assertEqual(scope['paired'](a, a)['p_float'], 1)
        with self.assertRaises(ValueError):
            scope['paired'](a, {})

    def test_disjoint_checks_family_and_id_independently(self):
        with self.assertRaises(ValueError):
            scope['assert_disjoint']([{'id': 'a', 'group_id': 'f'}],
                                     [{'id': 'b', 'group_id': 'f'}], 'test')
        with self.assertRaises(ValueError):
            scope['assert_disjoint']([{'id': 'a', 'group_id': 'f'}],
                                     [{'id': 'a', 'group_id': 'g'}], 'test')

    def test_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'out.json'
            scope['write_json'](p, {'ok': True})
            with self.assertRaises(FileExistsError):
                scope['write_json'](p, {'ok': False})
            self.assertEqual(json.loads(p.read_text()), {'ok': True})

    def test_actual_reserved_and_large_dev_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope['ROOT'], scope['ONPANEL'] = root, root / 'outputs' / 'splits_onpanel'
            dev = {'id': 'dev', 'group_id': 'dev-family', 'answer': '4',
                   'messages': [{'role': 'user', 'content': '2+2'}]}
            panel = [{'id': 'panel', 'group_id': 'panel-family'}]
            paths = ['data/monitor_probe.jsonl', 'data/calibration.jsonl', 'data/pilot.jsonl',
                     'outputs/splits_large/train.jsonl', 'outputs/splits_large/dev.jsonl',
                     'outputs/splits_onpanel/train.jsonl', 'outputs/splits_onpanel/dev.jsonl']
            for index, name in enumerate(paths):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                row = dev if name.endswith('splits_onpanel/dev.jsonl') else {
                    'id': str(index), 'group_id': 'family-' + str(index)}
                p.write_text(json.dumps(row) + '\n')
            self.assertEqual(scope['freeze_dev'](1, panel)[0], [dev])
            for name in ('data/calibration.jsonl', 'data/pilot.jsonl', 'outputs/splits_large/dev.jsonl'):
                path = root / name
                original = path.read_text()
                path.write_text(json.dumps({'id': 'different', 'group_id': 'dev-family'}) + '\n')
                with self.assertRaises(ValueError):
                    scope['freeze_dev'](1, panel)
                path.write_text(original)

    def test_expected_inventory_distinguishes_tail_from_full_model(self):
        names = ['embed_tokens.weight', 'final_norm.weight'] + [f'encoder.{i}.x' for i in range(28)]
        model = SimpleNamespace(encoder=[None] * 28,
                                named_parameters=lambda: [(name, None) for name in names])
        full = scope['expected_trainables'](model, scope['arm_spec']('B28'))
        tail = scope['expected_trainables'](model, scope['arm_spec']('B28last28'))
        self.assertEqual(full - tail, {'embed_tokens.weight', 'final_norm.weight'})
        self.assertEqual(len(scope['expected_trainables'](model, scope['arm_spec']('B28last16'))), 16)
        with self.assertRaises(ValueError):
            scope['arm_spec']('B28last29')

    def test_final_input_hash_verification_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text('original')
            bindings = {str(path): scope['sha256_file'](path)}
            scope['verify_bound_files'](bindings)
            path.write_text('changed')
            with self.assertRaises(ValueError):
                scope['verify_bound_files'](bindings)


@unittest.skipUnless(os.environ.get('RAB001_TEST_TORCH') == '1', 'torch test requires parent memory clearance')
class SoupTensorTests(unittest.TestCase):
    def test_stream_average_and_saved_roundtrip(self):
        import importlib.util
        import sys
        from unittest import mock
        import torch
        from safetensors.torch import load_file, save_file
        sys.path.insert(0, str(HERE))
        spec = importlib.util.spec_from_file_location('soup_test_module', HERE / 'model_soup.py')
        soup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(soup)
        from experiments.rab001_exploration import tensor_stream
        soup.CHUNK_ELEMENTS = 3
        soup.hostbudget.guard = lambda *args, **kwargs: None
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.matrix = torch.nn.Parameter(torch.zeros(5, 2))
                self.vector = torch.nn.Parameter(torch.zeros(7))
        model = Tiny()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            members = []
            for index in range(3):
                path = root / str(index)
                path.mkdir()
                tensors = {'matrix': torch.arange(10, dtype=torch.float32).reshape(5, 2) + index,
                           'vector': torch.arange(7, dtype=torch.float32) - index}
                # fp32 summation would erase the middle ingredient's contribution.
                tensors['matrix'][0, 0] = (1e10, 1.0, -1e10)[index]
                save_file(tensors, str(path / 'trainable.safetensors'))
                members.append({'_dir': path, 'tensor_shapes': {k: list(v.shape) for k, v in tensors.items()},
                                '_provenance': {'tag': str(index)}, '_key': {'mode': 'B28'}})
            reads = []
            original_reader = tensor_stream.TensorFile
            class ObservedReader(original_reader):
                def read(self, name, start, count):
                    reads.append((name, start, count))
                    return super().read(name, start, count)
            with mock.patch.object(tensor_stream, 'TensorFile', ObservedReader), \
                    mock.patch('safetensors.safe_open', side_effect=AssertionError('averaging must not map weights')):
                soup.apply_average(model, members)
            self.assertTrue(reads)
            self.assertTrue(all(0 < count <= 3 for _, _, count in reads))
            self.assertIn(('matrix', 0, 3), reads)
            self.assertIn(('matrix', 9, 1), reads)
            expected_matrix = torch.arange(10, dtype=torch.float32).reshape(5, 2) + 1
            expected_matrix[0, 0] = 1 / 3
            torch.testing.assert_close(model.matrix, expected_matrix, rtol=0, atol=0)
            torch.testing.assert_close(model.vector, torch.arange(7, dtype=torch.float32) - 1)
            soup.save_trainables(model, members, root / 'saved', {'assets': {}})
            restored = load_file(str(root / 'saved' / 'trainable.safetensors'))
            torch.testing.assert_close(restored['matrix'], model.matrix, rtol=0, atol=0)
            torch.testing.assert_close(restored['vector'], model.vector, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
