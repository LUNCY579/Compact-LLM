"""Exact item-order trie counts; never estimates model token cost or runs a model."""
from itertools import permutations
import math


def counts(n):
    if type(n) is not int or n < 1:
        raise ValueError('positive integer set size required')
    by_depth = [math.factorial(n) // math.factorial(n - k) for k in range(n + 1)]
    return {
        'set_size': n,
        'valid_compact_orderings': math.factorial(n),
        'item_prefix_nodes_by_depth': by_depth,
        'item_prefix_nodes_including_root': sum(by_depth),
        'item_prefix_edges': sum(by_depth) - 1,
        'single_teacher_item_edges': n,
        'subset_states_including_empty': 2 ** n,
        'unit': 'complete-item order prefixes, not tokenizer prefixes or model operations',
    }


def enumerate_small(n):
    if type(n) is not int or not 1 <= n <= 6:
        raise ValueError('finite independent enumeration bounded to set sizes 1..6')
    paths = list(permutations(range(n)))
    prefixes = {path[:k] for path in paths for k in range(n + 1)}
    by_depth = [sum(len(prefix) == k for prefix in prefixes) for k in range(n + 1)]
    expected = counts(n)
    if len(paths) != expected['valid_compact_orderings'] or by_depth != expected['item_prefix_nodes_by_depth']:
        raise AssertionError('closed form disagrees with explicit permutation trie')
    return {'set_size': n, 'enumerated_paths': len(paths), 'enumerated_prefixes': len(prefixes),
            'nodes_by_depth': by_depth, 'closed_form_equal': True}


def train_panel(size_counts):
    if type(size_counts) is not dict or not size_counts:
        raise ValueError('explicit nonempty set-size histogram required')
    rows = []
    for n, multiplicity in sorted(size_counts.items()):
        if type(n) is not int or type(multiplicity) is not int or multiplicity < 1:
            raise ValueError('integer set sizes and positive integer row counts required')
        rows.append(dict(counts(n), prompt_rows=multiplicity))
    total = lambda field: sum(row['prompt_rows'] * row[field] for row in rows)
    return {'strata': rows, 'prompt_rows': sum(size_counts.values()),
            'valid_compact_orderings_total': total('valid_compact_orderings'),
            'item_prefix_nodes_total_including_roots': total('item_prefix_nodes_including_root'),
            'item_prefix_edges_total': total('item_prefix_edges'),
            'single_teacher_item_edges_total': total('single_teacher_item_edges'),
            'classification': 'combinatorial representation counts only; no native runtime/IO/memory estimate'}
