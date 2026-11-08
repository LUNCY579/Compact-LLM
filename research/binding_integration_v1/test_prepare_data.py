"""Model-free integration layout, mask, denominator and byte-custody checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.binding_integration_v1 import prepare_data as p


def fixture_entry(label, index, length=4, supervised=2, newline=b'\n'):
    prompt = [10] * (length - supervised)
    ids = prompt + [20] * (supervised - 1) + [151645]
    row = {'id': label + '-' + str(index), 'parent_id': label + '-parent-' + str(index),
           'source_split': 'train', 'mode': 'direct', 'truncated': False,
           'ids': ids, 'labels': [-100] * len(prompt) + ids[len(prompt):],
           'prompt_ids': prompt, 'answer_start': len(prompt), 'answer_end': len(ids) - 1,
           'processed_tokens': len(ids), 'supervised_tokens': supervised}
    raw = json.dumps(row, separators=(', ', ': ')).encode() + newline
    return {'row': row, 'raw': raw, 'source': label, 'source_index': index,
            'source_row_sha256_including_newline': hashlib.sha256(raw).hexdigest()}


def prior_mix(native, binding, sequence):
    sources = {'native_dual899': native, 'binding_v2': binding}
    lines, membership = [], []
    for index, (label, source_index) in enumerate(sequence):
        entry = sources[label][source_index]
        lines.append(entry['raw'])
        membership.append({'mixture_index': index, 'id': entry['row']['id'], 'parent_id': entry['row']['parent_id'],
                           'source': label, 'source_index': source_index,
                           'source_row_sha256_including_newline': entry['source_row_sha256_including_newline']})
    return lines, membership


class IntegrationPreparationTests(unittest.TestCase):
    def test_floor_partition_preserves_each_source_order_and_all_views(self):
        order = [2, 0, 4, 1, 3]
        execution, ng, bg, combined = p.layout_indices([[0, 1], [2], [3, 4]], order)
        self.assertEqual([i for label, i in execution if label == 'native_dual899'], list(range(5)))
        self.assertEqual([i for label, i in execution if label == 'binding_v2'], order)
        self.assertEqual(list(map(len, bg)), [1, 2, 2])
        self.assertEqual([i for group in combined for i in group], list(range(10)))
        self.assertEqual(combined, [n + b for n, b in zip(ng, bg)])

    def test_invalid_native_partition_and_binding_permutation_fail(self):
        for native, binding in [([[0, 2]], [0]), ([[0], []], [0]), ([[0]], [0, 0]), ([[0]], [1]), ([], [0])]:
            with self.assertRaises(ValueError):
                p.layout_indices(native, binding)

    def test_original_membership_checks_exact_bytes_and_order(self):
        native = [fixture_entry('native_dual899', i) for i in range(2)]
        binding = [fixture_entry('binding_v2', i) for i in range(2)]
        sequence = [('native_dual899', 0), ('binding_v2', 1), ('native_dual899', 1), ('binding_v2', 0)]
        lines, membership = prior_mix(native, binding, sequence)
        order, previous = p.original_binding_order(native, binding, lines, membership)
        self.assertEqual(order, [1, 0]); self.assertEqual(previous['binding_v2', 1], 1)
        changed = list(lines); changed[0] = lines[0].replace(b': ', b':', 1)
        with self.assertRaisesRegex(ValueError, 'line bytes differ'):
            p.original_binding_order(native, binding, changed, membership)

    def test_forged_membership_hash_is_refused(self):
        native = [fixture_entry('native_dual899', 0)]; binding = [fixture_entry('binding_v2', 0)]
        lines, membership = prior_mix(native, binding, [('native_dual899', 0), ('binding_v2', 0)])
        membership[1]['source_row_sha256_including_newline'] = '0' * 64
        with self.assertRaises(ValueError):
            p.original_binding_order(native, binding, lines, membership)

    def test_native_denominator_and_clock_do_not_include_binding(self):
        native = [fixture_entry('native_dual899', 0, length=5, supervised=3), fixture_entry('native_dual899', 1, length=6, supervised=2)]
        binding = [fixture_entry('binding_v2', 0, length=7, supervised=4), fixture_entry('binding_v2', 1, length=8, supervised=5)]
        previous = {(label, index): i for i, (label, index) in enumerate([('native_dual899', 0), ('binding_v2', 1), ('native_dual899', 1), ('binding_v2', 0)])}
        ordered, membership, groups = p.assemble(native, binding, [[0], [1]], [1, 0], previous, [0.01, 0.02])
        self.assertEqual(groups['native_denominators'], [3, 2])
        self.assertEqual(groups['binding_denominators'], [5, 4])
        self.assertEqual(groups['native_processed_cumulative'], [5, 11])
        self.assertEqual(groups['processed_tokens'], 26)
        self.assertEqual(groups['applied_native_learning_rates'], [0.01, 0.02])
        self.assertEqual(groups['effective_binding_mean_coefficients'], [5 / 3, 2])
        self.assertEqual([m['execution_index'] for m in membership], list(range(4)))
        self.assertEqual([m['optimizer_group_index'] for m in membership], [0, 0, 1, 1])

    def test_combined_accumulation_may_exceed_sequence_cap(self):
        native = [fixture_entry('native_dual899', 0, length=2000)]
        binding = [fixture_entry('binding_v2', 0, length=1000)]
        _, _, g = p.assemble(native, binding, [[0]], [0], {('native_dual899', 0): 0, ('binding_v2', 0): 1}, [0.001])
        self.assertEqual(g['max_sequence_tokens'], 2000)
        self.assertEqual(g['sequence_token_limit'], 2048)
        self.assertEqual(g['maximum_combined_processed_tokens_per_update'], 3000)
        self.assertTrue(g['combined_token_count_is_accumulation_not_sequence_length'])

    def test_compiled_entry_preserves_whitespace_and_crlf(self):
        entry = fixture_entry('native_dual899', 0, newline=b'\r\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.jsonl'; path.write_bytes(entry['raw'])
            got = p.compiled_entries(path, p.sha(path), 'native_dual899')
        self.assertEqual(got[0]['raw'], entry['raw'])
        self.assertEqual(got[0]['source_row_sha256_including_newline'], entry['source_row_sha256_including_newline'])

    def test_changed_mask_or_dev_split_fails(self):
        for key, value in [('labels', [-100, -100, -100, 151645]), ('source_split', 'dev'), ('truncated', True)]:
            item = copy.deepcopy(fixture_entry('native_dual899', 0)['row']); item[key] = value
            raw = json.dumps(item).encode() + b'\n'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'source.jsonl'; path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    p.compiled_entries(path, p.sha(path), 'native_dual899')

    def test_write_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'; p.write_new(path, {'first': 1}); before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                p.write_new(path, {'second': 2})
            self.assertEqual(path.read_bytes(), before)

    def test_weight_payload_is_never_eligible_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'; path.write_bytes(b'not read')
            with self.assertRaisesRegex(ValueError, 'weight payload'):
                p.sha(path)

    def test_full_frozen_train_inventory_in_memory_only(self):
        native = p.compiled_entries(p.NATIVE, p.PINS[p.NATIVE], 'native_dual899')
        binding = p.compiled_entries(p.BINDING_TRAIN, p.PINS[p.BINDING_TRAIN], 'binding_v2')
        lines = (p.MIXTURE / 'prepared/microfit-train.jsonl').read_bytes().splitlines(keepends=True)
        order, previous = p.original_binding_order(native, binding, lines, p.read(p.MIXTURE / 'prepared/SOURCE_MEMBERSHIP.json'))
        plan = p.read(p.CONTROL / 'PLAN.json')
        steps = [json.loads(line) for line in (p.CONTROL / 'journal/runs/train-001/steps.jsonl').read_text().splitlines()]
        ordered, membership, groups = p.assemble(native, binding, plan['update_groups'], order, previous, [s['lr'] for s in steps])
        self.assertEqual((len(ordered), len(groups['combined_update_groups'])), (3465, 193))
        self.assertEqual(len({e['row']['parent_id'] for e in ordered}), 2761)
        self.assertEqual((groups['processed_tokens'], groups['supervised_tokens']), (577417, 39502))
        self.assertEqual((groups['native_processed_tokens'], groups['native_supervised_tokens'], groups['binding_supervised_tokens']), (380921, 37090, 2412))
        self.assertEqual((groups['max_sequence_tokens'], groups['maximum_combined_processed_tokens_per_update']), (277, 3114))
        self.assertEqual(sum(len(g) for g in groups['native_update_groups'][:2]), 28)
        self.assertEqual(groups['native_processed_cumulative'][1], 4032)
        self.assertEqual(sum(groups['native_denominators'][:2]), 388)
        self.assertEqual([m['source_index'] for m in membership if m['source'] == 'native_dual899'], list(range(2697)))
        self.assertEqual([m['source_index'] for m in membership if m['source'] == 'binding_v2'], order)
        self.assertTrue(all(e['raw'] == (native if e['source'] == 'native_dual899' else binding)[e['source_index']]['raw'] for e in ordered))


if __name__ == '__main__':
    unittest.main()
