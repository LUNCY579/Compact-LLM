"""Offline descriptive accounting of the frozen mixture's loss normalization."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from paths import OUT, HERE, sha, read, write


def main():
    prepared = OUT / 'prepared'
    rows = [json.loads(line) for line in (prepared / 'microfit-train.jsonl').read_text().splitlines()]
    members = read(prepared / 'SOURCE_MEMBERSHIP.json')
    grouping = read(prepared / 'UPDATE_GROUPS.json')
    plan = read(OUT / 'PLAN.json')
    groups = plan['update_groups']
    assert [i for group in groups for i in group] == list(range(len(rows)))
    assert len(groups) == 296 and len(rows) == len(members) == 3465
    supervised = [sum(value != -100 for value in row['labels'][1:]) for row in rows]
    totals = {source: Counter() for source in ('binding_v2', 'native_dual899')}
    for index, (row, member) in enumerate(zip(rows, members)):
        assert member['mixture_index'] == index and member['id'] == row['id']
        counter = totals[member['source']]
        counter['views'] += 1
        counter['processed_tokens'] += len(row['ids'])
        counter['supervised_tokens'] += supervised[index]
    fractions = []
    per_group = []
    for index, group in enumerate(groups):
        denominator = sum(supervised[i] for i in group)
        binding_numerator = sum(supervised[i] for i in group if members[i]['source'] == 'binding_v2')
        fraction = binding_numerator / denominator
        fractions.append(fraction)
        per_group.append({'group': index + 1, 'views': len(group),
                          'binding_views': sum(members[i]['source'] == 'binding_v2' for i in group),
                          'supervised_denominator': denominator,
                          'binding_supervised_numerator': binding_numerator,
                          'binding_loss_coefficient_fraction': fraction})
    for key in ('views', 'processed_tokens', 'supervised_tokens'):
        denominator = sum(value[key] for value in totals.values())
        for value in totals.values():
            value[key + '_fraction'] = value[key] / denominator
    assert sum(supervised) == 39502
    assert totals['binding_v2']['supervised_tokens'] == 2412
    output = {'schema': 1, 'record': 'BINDING_MIXTURE_EXPOSURE_DESCRIPTION',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'COMPLETE_DATA_ONLY',
        'input_files_sha256': {str(p): sha(p) for p in (
            prepared / 'microfit-train.jsonl', prepared / 'SOURCE_MEMBERSHIP.json',
            prepared / 'UPDATE_GROUPS.json', OUT / 'PLAN.json', HERE / 'train.py')},
        'analyst_sha256': sha(Path(__file__)), 'source_totals': totals,
        'binding_group_loss_fraction': {'minimum': min(fractions), 'maximum': max(fractions),
            'mean_over_optimizer_groups': statistics.mean(fractions), 'median': statistics.median(fractions)},
        'groups_containing_binding': sum(value['binding_views'] > 0 for value in per_group),
        'groups': per_group,
        'interpretation': [
            'Each group sums response-plus-EOS token losses and divides by all supervised tokens in that group.',
            'These fractions describe coefficients in the scalar loss, not gradient norm, optimizer influence, or learned skill.',
            'Long binding prompts consume processed-token budget, but masked prompt tokens do not enter the supervised denominator.',
            'This descriptive analysis does not modify the fixed mixture, schedule, criteria, or training run.'],
        'model_calls': 0, 'quality_claim': False}
    write(OUT / 'EXPOSURE_DESCRIPTION.json', output)
    print(json.dumps({key: output[key] for key in ('status', 'source_totals', 'binding_group_loss_fraction', 'groups_containing_binding')}))


if __name__ == '__main__':
    main()
