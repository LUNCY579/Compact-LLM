"""Focused controls and tiny-tensor checks; never loads a model/tokenizer checkpoint.

Run after other model jobs finish (the tensor tests import torch)::
    python -m unittest experiments.rab001_exploration.test_alpha_sweep

The MetadataTests class alone is stdlib-only and safe during model execution.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rab001_exploration import alpha_sweep as alpha


class MetadataTests(unittest.TestCase):
    def test_endpoint_requires_termination_identity(self):
        expected = {'row': {'text': '42', 'terminated': True}}
        actual = {'row': {'text': '42', 'terminated': False}}
        result = alpha.endpoint_control(actual, expected, 'source')
        self.assertFalse(result['passed'])
        self.assertEqual(result['text_mismatch_ids'], [])
        self.assertEqual(result['termination_mismatch_ids'], ['row'])

    def test_endpoint_requires_text_identity_and_complete_inventory(self):
        expected = {'row': {'text': '42', 'terminated': True}}
        actual = {'row': {'text': '42.', 'terminated': True}}
        self.assertFalse(alpha.endpoint_control(actual, expected, 'source')['passed'])
        with self.assertRaisesRegex(ValueError, 'inventory'):
            alpha.endpoint_control({}, expected, 'source')
        with self.assertRaisesRegex(ValueError, 'termination'):
            alpha.endpoint_control({'row': {'text': '42'}}, expected, 'source')

    def test_legacy_source_key_is_not_mislabeled_as_asset_digest(self):
        audit = {'audit_digest': 'asset-audit', 'assets': {'model.safetensors': 'weights'}}
        self.assertIn('NOT_source_asset_digest', alpha.source_key_kind('impl', 'impl', audit, {}))
        self.assertEqual(alpha.source_key_kind('weights', 'impl', audit, {}),
                         'verified_source_weight_file_sha256')
        with self.assertRaisesRegex(ValueError, 'source key'):
            alpha.source_key_kind('stale', 'impl', audit, {})

    def test_legacy_and_current_b28_tail_identity(self):
        manifest = {'arm': 'B28last8', 'soup_key': {'mode': 'B28last8', 'read_tail': None},
                    'reconstruct': {'mode': 'B28last8', 'read_tail': None}}
        self.assertEqual(alpha.arm_identity(manifest, 'b28-wise'), ('B28', 8))
        manifest['soup_key']['mode'] = manifest['reconstruct']['mode'] = 'B28'
        self.assertEqual(alpha.arm_identity(manifest, 'b28-wise'), ('B28', 8))
        with self.assertRaisesRegex(ValueError, 'requires --kind'):
            alpha.arm_identity(manifest, 'p-output-gate')

    def test_p_tail_must_match_checkpoint_metadata(self):
        manifest = {'arm': 'Plast4', 'soup_key': {'mode': 'P', 'read_tail': 4},
                    'reconstruct': {'mode': 'P', 'read_tail': 4}}
        self.assertEqual(alpha.arm_identity(manifest, 'p-output-gate'), ('P', 4))
        manifest['soup_key']['read_tail'] = None
        with self.assertRaisesRegex(ValueError, 'read_tail'):
            alpha.arm_identity(manifest, 'p-output-gate')

    def test_source_assets_are_hashed_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'source'
            source.mkdir()
            weights = source / 'model.safetensors'
            weights.write_bytes(b'fixture weights')
            lock = folder / 'source_lock.json'
            lock.write_text(json.dumps({'core_id': 'fixture', 'weight_bytes': weights.stat().st_size,
                'assets': {'model.safetensors': hashlib.sha256(weights.read_bytes()).hexdigest()}}))
            verified = alpha.verify_source_assets(source, lock)
            self.assertEqual(verified['audit_digest'], alpha.canonical_hash(
                {'core_id': 'fixture', 'assets': verified['assets']}))
            weights.write_bytes(b'changed weights')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                alpha.verify_source_assets(source, lock)

    def test_create_only_artifact_never_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alpha_0.json'
            alpha.write_json(path, {'completed': False}, exclusive=True)
            with self.assertRaises(FileExistsError):
                alpha.write_json(path, {'completed': True}, exclusive=True)
            self.assertFalse(alpha.read_json(path)['completed'])


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from safetensors.torch import save_file
        cls.torch = torch
        cls.save_file = staticmethod(save_file)

    def test_b28_endpoints_exact_and_intermediate_resets_from_files(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = torch.nn.Linear(2, 2, bias=False)
            source = torch.tensor([[1.5, -2.0], [0.5, 4.0]], dtype=torch.bfloat16)
            trained = torch.tensor([[-0.25, 0.75], [1.25, -3.5]], dtype=torch.float32)
            self.save_file({'native.weight': source}, str(folder / 'source.safetensors'))
            self.save_file({'weight': trained}, str(folder / 'trained.safetensors'))
            for value in (0.0, 1.0, 0.25, 0.5, 0.75, 0.25, 0.0, 1.0):
                alpha.apply_alpha(model, folder / 'trained.safetensors', ['weight'], value,
                    'b28-wise', source_weights=folder / 'source.safetensors',
                    source_names={'weight': 'native.weight'})
                expected = ((1 - value) * source.float() + value * trained)
                self.assertTrue(torch.equal(model.weight, expected))

    def test_p_scales_only_outputs_and_does_not_compound_gate(self):
        torch = self.torch
        model = torch.nn.Module()
        read = torch.nn.Module()
        read.o_proj = torch.nn.Linear(2, 2, bias=False)
        read.q_proj = torch.nn.Linear(2, 2, bias=False)
        model.reads = torch.nn.ModuleList([read])
        model.memory = torch.nn.Linear(2, 2, bias=False)
        trained = {name: torch.full_like(value, float(index + 1))
                   for index, (name, value) in enumerate(model.named_parameters())}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'trained.safetensors'
            self.save_file(trained, str(checkpoint))
            for value in (0.0, 1.0, 0.25, 0.5, 0.75, 0.25):
                alpha.apply_alpha(model, checkpoint, sorted(trained), value, 'p-output-gate')
                for name, parameter in model.named_parameters():
                    expected = trained[name] * value if alpha.is_read_output(name) else trained[name]
                    self.assertTrue(torch.equal(parameter, expected), name)

    def test_corrupt_tensor_fails_before_probe(self):
        torch = self.torch
        model = torch.nn.Linear(2, 2, bias=False)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'trained.safetensors'
            self.save_file({'weight': torch.full((2, 2), float('nan'))}, str(checkpoint))
            with self.assertRaisesRegex(ValueError, 'invalid trained tensor'):
                alpha.apply_alpha(model, checkpoint, ['weight'], 1.0, 'b28-wise',
                                  source_weights=checkpoint, source_names={'weight': 'weight'})

    def test_inventory_requires_all_trainable_names_and_exact_shapes(self):
        torch = self.torch
        model = torch.nn.Module()
        model.encoder = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False),
                                            torch.nn.Linear(2, 2, bias=False)])
        names = ['encoder.1.weight']
        manifest = {'tensor_shapes': {'encoder.1.weight': [2, 2]}, 'tensors': 1,
                    'parameters': 4, 'soup_key': {'tensor_names_sha256':
                        hashlib.sha256('\n'.join(names).encode()).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'trained.safetensors'
            self.save_file({names[0]: torch.ones(2, 2)}, str(checkpoint))
            self.assertEqual(alpha.validate_tensor_inventory(model, checkpoint, manifest, 'B28', 1), names)
            manifest['tensor_shapes'][names[0]] = [4]
            with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                alpha.validate_tensor_inventory(model, checkpoint, manifest, 'B28', 1)

    def test_chunked_alpha_has_no_mmap_and_keeps_float32_formula(self):
        from experiments.rab001_exploration import tensor_stream
        torch = self.torch
        model = torch.nn.Linear(3, 3, bias=False)
        source = torch.arange(9, dtype=torch.float32).reshape(3, 3).to(torch.bfloat16)
        trained = torch.arange(9, dtype=torch.float32).reshape(3, 3) * -0.75 + 0.125
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.save_file({'source.weight': source}, str(folder / 'source.safetensors'))
            self.save_file({'weight': trained}, str(folder / 'trained.safetensors'))
            reads, guard_calls = [], []
            original_read = tensor_stream.TensorFile.read
            def tracked_read(reader, name, start, count):
                reads.append((name, start, count))
                return original_read(reader, name, start, count)
            with patch.object(tensor_stream, 'MAX_ELEMENTS', 4), \
                    patch.object(tensor_stream.TensorFile, 'read', tracked_read), \
                    patch('safetensors.safe_open', side_effect=AssertionError('mmap loader forbidden')):
                for value in (0.0, 1.0, 0.25, 0.5, 0.75):
                    alpha.apply_alpha(model, folder / 'trained.safetensors', ['weight'], value,
                        'b28-wise', source_weights=folder / 'source.safetensors',
                        source_names={'weight': 'source.weight'}, guard=lambda: guard_calls.append(1))
                    expected = source.float().mul_(1 - value).add_(trained, alpha=value)
                    self.assertTrue(torch.equal(model.weight, expected))
            self.assertTrue(reads)
            self.assertTrue(all(count <= 4 for _, _, count in reads))
            self.assertIn(('weight', 8, 1), reads)
            self.assertEqual(len(guard_calls), 15)


if __name__ == '__main__':
    unittest.main()
