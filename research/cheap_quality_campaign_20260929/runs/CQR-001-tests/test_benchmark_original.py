"""Bounded synthetic checks; never run the 192-case retrieval study here."""
import inspect
import os
import unittest
from types import SimpleNamespace

from experiments.cheap_quality_retrieval_v1.benchmark import (
    METHODS, NUMERIC_METRICS, PACKAGE, aggregate, digest, evidence_metrics,
    pack_context, pad_exact, public_view, rank_public, read_json, token_count)


class CharacterTokenizer:
    """Independent packing fixture: one character per token, no special tokens."""
    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError('packing must exclude special tokens')
        return SimpleNamespace(ids=list(text))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(PACKAGE/'config.json')
        self.sentences = [
            {'sentence_id': 's407', 'text': 'Courier C1900 carries parcel P2900.'},
            {'sentence_id': 's162', 'text': 'Parcel P2900 is stored in locker L3900.'},
            {'sentence_id': 's893', 'text': 'Courier C8800 carries parcel P7800.'},
            {'sentence_id': 's551', 'text': 'Parcel P7800 is stored in locker L6800.'},
            {'sentence_id': 's725', 'text': 'Parcel P4800 is stored in locker L5800.'}]

    def test_public_schema_rejects_support_answer_and_type_leakage(self):
        case = {'case_id': 'synthetic', 'question': 'Which locker?', 'sentences': self.sentences}
        public_view(case)
        for forbidden in ('support_sentence_ids', 'gold_answer', 'world_type', 'position'):
            with self.assertRaisesRegex(ValueError, 'extra fields'):
                public_view({**case, forbidden: 'secret'})
        with self.assertRaisesRegex(ValueError, 'evaluator metadata'):
            public_view({**case, 'sentences': [{**self.sentences[0], 'is_support': True}]})
        self.assertEqual(list(inspect.signature(rank_public).parameters),
                         ['question', 'sentences', 'method', 'base_seed'])

    def test_duplicate_sentence_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'duplicate public sentence ID'):
            public_view({'case_id': 'synthetic', 'question': '?',
                         'sentences': [self.sentences[0], self.sentences[0]]})

    def test_all_public_methods_are_deterministic_exact_permutations(self):
        question = 'Which locker stores the parcel carried by courier C1900?'
        for method in METHODS:
            first = rank_public(question, self.sentences, method, 710233)
            self.assertEqual(first, rank_public(question, self.sentences, method, 710233))
            self.assertEqual(sorted(first), list(range(len(self.sentences))))
        self.assertEqual(rank_public(question, self.sentences, 'bm25_one_hop', 0)[0], 0)
        self.assertEqual(rank_public(question, self.sentences, 'bm25_two_pass', 0)[:2], [0, 1])

    def test_random_identity_ranking_is_paired_across_context_positions(self):
        rearranged = list(reversed(self.sentences))
        first = rank_public('public query', self.sentences, 'seeded_random', 710233)
        second = rank_public('public query', rearranged, 'seeded_random', 710233)
        self.assertEqual([self.sentences[index]['sentence_id'] for index in first],
                         [rearranged[index]['sentence_id'] for index in second])

    def test_restore_source_order_and_exact_budget_independently(self):
        facts = [{'sentence_id': 'a', 'text': 'AAAA'}, {'sentence_id': 'b', 'text': 'BBBB'},
                 {'sentence_id': 'c', 'text': 'CCCC'}]
        packed = pack_context(facts, [2, 0, 1], 10, CharacterTokenizer(), self.config)
        self.assertEqual(packed['selected_sentence_ids'], ['a', 'c'])
        self.assertEqual(packed['selected_source_indices'], [0, 2])
        self.assertEqual(packed['selected_context_text'], 'AAAA\nCCCC')
        self.assertEqual(packed['consumed_tokens'], 10)
        self.assertEqual(packed['net_padding_tokens'], 1)
        self.assertEqual(token_count(CharacterTokenizer(), packed['context_text']), 10)

    def test_padding_failure_does_not_relax_budget(self):
        with self.assertRaisesRegex(ValueError, 'cannot add one token'):
            pad_exact('AAAA', 5, CharacterTokenizer(), ['long filler'])

    def test_two_support_full_recall_requires_both(self):
        one = evidence_metrics(['first'], ['first', 'second'])
        self.assertEqual((one['full_evidence_recall'], one['any_evidence_recall']), (0, 1))
        both = evidence_metrics(['first', 'second', 'irrelevant'], ['first', 'second'])
        self.assertEqual((both['full_evidence_recall'], both['any_evidence_recall']), (1, 1))
        self.assertAlmostEqual(both['support_precision'], 2/3)
        self.assertEqual(evidence_metrics([], ['first'])['support_precision'], 0)
        with self.assertRaises(ValueError):
            evidence_metrics(['first'], [])

    def test_world_clusters_keep_three_positions_as_one_world(self):
        rows = []
        for world, recall in [('world-a', [0, 1, 1]), ('world-b', [1, 1, 1])]:
            for position, full in zip(('early', 'middle', 'late'), recall):
                rows.append({'method': 'synthetic', 'budget': 128, 'world_id': world,
                             'world_type': 'two_hop', 'position': position,
                             **{key: 1 for key in NUMERIC_METRICS}, 'full_evidence_recall': full})
        aggregates, clusters = aggregate(rows)
        combined = next(row for row in aggregates if row['stratum'] == 'all')
        self.assertEqual((combined['case_count'], combined['world_count']), (6, 2))
        self.assertEqual(len(clusters), 2)
        self.assertAlmostEqual(combined['world_cluster_full_recall_mean'], 5/6)
        self.assertAlmostEqual(combined['world_cluster_full_recall_min'], 2/3)

    def test_native_tokenizer_exact_consumed_budgets(self):
        # Raw tokenizer integration is mandatory; no skip or approximate token substitute.
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import tokenizers
        declared = self.config['tokenizer']
        self.assertEqual(tokenizers.__version__, declared['tokenizers_version'])
        self.assertEqual(digest(declared['path']), declared['sha256'])
        tokenizer = tokenizers.Tokenizer.from_file(declared['path'])
        for method in METHODS:
            ranking = rank_public('Which locker stores parcel P2900?', self.sentences, method, 710233)
            for native_budget in (128, 256, 512):
                packed = pack_context(self.sentences, ranking, native_budget, tokenizer, self.config)
                self.assertEqual(token_count(tokenizer, packed['context_text']), native_budget)
                self.assertEqual(packed['consumed_tokens'], native_budget)
                self.assertEqual(packed['selected_context_tokens']+packed['net_padding_tokens'], native_budget)


if __name__ == '__main__':
    unittest.main()
