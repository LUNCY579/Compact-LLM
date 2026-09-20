"""Mechanics-only admission tests; no real banks, tokenizer, weights or outputs."""
import copy
import tempfile
from pathlib import Path
import unittest

from research.binding_diagnostic_v1 import prepare_data as p
from research.binding_diagnostic_v1 import binding, arithmetic


def row(identifier, text, parent=None):
    return {'id': identifier, 'parent_id': parent or identifier + '-parent',
            'prompt': [{'role': 'system', 'content': 'fixture'}, {'role': 'user', 'content': text}]}


class Guard:
    """Stub native disclosures; the mandatory comparer remains independently tested."""
    @staticmethod
    def audit_and_filter(rows, bank):
        return [], {'dropped': [{'id': r['id'], 'reasons': ['user_8gram', 'digit_blinded_panel_stem'],
                               'matched_panel_stem_ids': [bank[0]['id']]} for r in rows]}


class Tokenizer:
    def __init__(self, prompt_tokens=10, answer_tokens=3):
        self.prompt_tokens, self.answer_tokens = prompt_tokens, answer_tokens

    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs == {'tokenize': True, 'add_generation_prompt': True, 'enable_thinking': False}
        return [3] * self.prompt_tokens

    def encode(self, reference, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return [4] * self.answer_tokens


class PreparationTests(unittest.TestCase):
    def test_short_native_contacts_are_disclosed_without_blocking(self):
        r = p.compare_bank([row('new', 'A wholly different question.')], [row('old', 'Old task.')], Guard)
        self.assertEqual(r['mandatory_contact_rows'], 0)
        self.assertEqual(r['disclosure_contact_rows'], 1)
        self.assertEqual(r['contacts'][0]['native_short_stem_bank_ids'], ['old'])

    def test_cross_field_parent_identity_blocks(self):
        r = p.compare_bank([row('new', 'New.', parent='old')], [row('old', 'Old.')], Guard)
        self.assertEqual(r['mandatory_contact_rows'], 1)
        self.assertEqual(r['contacts'][0]['mandatory_matches']['exact_cross_field_ids'], ['old'])

    def test_full_unicode_prompt_equality_blocks(self):
        r = p.compare_bank([row('new', 'Ａ  NUMBER\n 7?')], [row('old', 'a number 7?')], Guard)
        self.assertIn('full_NFKC_casefold_user_prompt', r['contacts'][0]['mandatory_matches'])

    def test_full_digit_blinded_template_blocks(self):
        r = p.compare_bank([row('new', 'Determine 18 / 3.')], [row('old', 'Determine 12 / 4.')], Guard)
        matches = r['contacts'][0]['mandatory_matches']
        self.assertIn('full_digit_blinded_user_template', matches)
        self.assertNotIn('full_NFKC_casefold_user_prompt', matches)

    def test_common_system_and_assistant_text_are_ignored(self):
        a, b = row('new', 'New question.'), row('old', 'Other question.')
        for value in (a, b):
            value['prompt'].append({'role': 'assistant', 'content': 'same answer'})
        self.assertEqual(p.compare_bank([a], [b], Guard)['mandatory_contact_rows'], 0)

    def test_literal_contact_is_phrase_bounded_and_nfkc(self):
        manifest = {key: [] for key in p.LITERAL_FIELDS}
        manifest['forbidden_proper_nouns'] = ['Alma Reed']
        manifest['forbidden_domains'] = ['orbital_station']
        rows = [row('hit', 'Ａｌｍａ Reed enters orbital-station.'), row('clean', 'Alma Reedling stays elsewhere.')]
        result = p.literal_audit(rows, manifest)
        self.assertEqual(result['mandatory_contact_rows'], 1)
        self.assertEqual(len(result['contacts']), 2)
        self.assertTrue(all(x['candidate_ids'] == ['hit'] for x in result['contacts']))

    def test_create_only_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'receipt.json'
            p.write_new(target, {'first': True})
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                p.write_new(target, {'second': True})
            self.assertEqual(target.read_bytes(), before)

    def test_duplicate_fields_and_nonfinite_json_are_refused(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                p.decode(value)

    def test_compilation_has_only_four_policy_keys_and_exact_budget(self):
        rows = [row('x', 'Example.')]
        gold = [{'id': 'x', 'reference': 'answer', 'secret': 'never copied'}]
        compiled, audit = p.compile_inputs(rows, gold, Tokenizer())
        self.assertEqual(set(compiled[0]), {'id', 'parent_id', 'prompt', 'prompt_ids'})
        self.assertEqual(audit['reserved_processed_input_tokens_per_model'], 33)
        self.assertEqual(audit['max_answer_content_tokens'], 3)
        self.assertNotIn('answer', compiled[0])

    def test_prompt_or_answer_overflow_is_refused_without_truncation(self):
        rows = [row('x', 'Example.')]; gold = [{'reference': 'answer'}]
        for tokenizer in (Tokenizer(prompt_tokens=2025), Tokenizer(answer_tokens=24)):
            with self.assertRaisesRegex(ValueError, 'truncation prohibited'):
                p.compile_inputs(rows, gold, tokenizer)

    def test_full_factor_bundle_oracles_and_gold_separation(self):
        b, a = binding.build_rows(seed=31), arithmetic.build_rows(seed=31)
        rows, gold, meta = p.validate_bundles(b, a, binding, arithmetic)
        self.assertEqual((len(rows), len(gold)), (896, 896))
        self.assertEqual((len(meta['binding']), len(meta['arithmetic'])), (32, 96))
        self.assertTrue(all(set(r) == {'id', 'parent_id', 'prompt'} for r in rows))
        self.assertEqual([g['section'] for g in gold], ['binding'] * 512 + ['arithmetic'] * 384)
        self.assertNotIn('section', a['gold'][0])

    def test_changed_gold_fails_visible_input_oracle(self):
        b, a = binding.build_rows(seed=31), arithmetic.build_rows(seed=31)
        changed = copy.deepcopy(b)
        changed['gold'][0]['reference'] = 'WRONG'
        with self.assertRaises(ValueError):
            p.validate_bundles(changed, a, binding, arithmetic)

    def test_weight_payload_reads_are_explicitly_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'model.safetensors'
            target.write_bytes(b'not read')
            with self.assertRaisesRegex(ValueError, 'prohibited weight read'):
                p.sha(target)


if __name__ == '__main__':
    unittest.main()
