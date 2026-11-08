"""Post-run descriptive template and answer-type analysis; no new efficacy rule."""
from collections import Counter, defaultdict
from pathlib import Path
from paths import ROOT
from common import read, read_jsonl, sha, write_new, require
import binding


def norm(text):
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main():
    result = read(ROOT / 'RESULTS.json')
    require(result['status'] == 'COMPLETE_COMPARABLE', 'comparable results required')
    inputs = {row['id']: row for row in read_jsonl(ROOT / 'candidate-v1/DIAGNOSTIC_INPUTS.jsonl')}
    counts, errors = {}, {}
    for role, endpoint in result['endpoints'].items():
        cells = defaultdict(Counter)
        categories = Counter()
        false_worlds = defaultdict(set)
        for row in endpoint['rows'].values():
            if row['section'] != 'binding':
                continue
            parsed = binding.parse_policy_input(inputs[row['id']]['prompt'])
            cell = cells[(row['domain'], row['template'], row['state'])]
            cell['n'] += 1
            cell['correct'] += row['canonical_complete_exact']
            cell['false_unknown'] += row['false_exact_unknown']
            cell['exact_roster_person_on_unknown'] += row['exact_roster_person_on_unknown']
            if row['false_exact_unknown']:
                false_worlds[row['template']].add(row['parent_id'])
            if not row['canonical_complete_exact']:
                category = ('false_unknown' if row['false_exact_unknown'] else
                            'exact_roster_person_on_unknown' if row['exact_roster_person_on_unknown'] else
                            'exact_queried_role_instead_of_person_or_unknown' if norm(row['prediction']) == norm(parsed['queried_role']) else
                            'other_unclassified_error')
                categories[category] += 1
        counts[role] = [{'domain': d, 'template': t, 'state': s, **dict(cells[(d, t, s)])} for d, t, s in sorted(cells)]
        errors[role] = {'categories': dict(categories), 'false_unknown_parent_counts_by_template':
                        {str(template): len(false_worlds[template]) for template in (0, 1)}}
    record = {'schema': 1, 'status': 'COMPLETE_DESCRIPTIVE_POST_RUN_ANALYSIS',
        'results_sha256': sha(ROOT / 'RESULTS.json'), 'input_sha256': sha(ROOT / 'candidate-v1/DIAGNOSTIC_INPUTS.jsonl'),
        'analyst_sha256': sha(Path(__file__).resolve()), 'cells': counts, 'errors': errors,
        'primary_endpoint_changed': False, 'new_significance_tests': 0, 'model_calls': 0, 'quality_claim': False}
    write_new(ROOT / 'BINDING_ERROR_ANALYSIS.json', record)
    lines = [
        '# Fresh binding diagnostic: wording sensitivity and answer types', '',
        'All three models remain at zero complete16-view parents. The registered primary contrast is (net0,nd0,p1); the mixture band8–24 failed. This analysis does not substitute an easier endpoint.', '',
        '| Model | Template | Occupied correct /128 | False UNKNOWN /128 | Missing UNKNOWN /64 | Vacant UNKNOWN /64 |',
        '|---|---:|---:|---:|---:|---:|']
    for role in ('source', 'lora', 'mixed'):
        for template in (0, 1):
            rows = [row for row in counts[role] if row['template'] == template]
            summed = lambda state, field: sum(row[field] for row in rows if row['state'] == state)
            lines.append(f"| {role} | {template} | {summed('occupied','correct')} | {summed('occupied','false_unknown')} | {summed('missing','correct')} | {summed('vacant','correct')} |")
    lines += ['',
        'These are descriptive view counts. The independent unit remains32 parents; domain/template derivatives are correlated.', '',
        'The mixture produces50 of its52 false UNKNOWNs under template1, across both vocabularies. Template0 has two. The full crossing holds each parent’s facts, names and answers fixed while changing the wording bundle; it supports a sensitivity to that bundle on these fixed inputs. Occupancy wording, duty wording and question wording change together, so the result does not identify one phrase or a training cause.', '',
        'Answerable accuracy nevertheless rises from175/256 for LoRA to187/256 for the mixture. Correct UNKNOWN rises from6/256 to161/256. Neither descriptive difference meets the registered all16 selectivity endpoint, and no row-level iid significance test is performed.', '',
        '| Model | False UNKNOWN | Exact roster person on UNKNOWN target | Exact queried role instead of requested answer | Other unclassified errors |',
        '|---|---:|---:|---:|---:|']
    for role in ('source', 'lora', 'mixed'):
        value = errors[role]['categories']
        lines.append('| ' + role + ' | ' + ' | '.join(str(value.get(key, 0)) for key in
            ('false_unknown', 'exact_roster_person_on_unknown', 'exact_queried_role_instead_of_person_or_unknown', 'other_unclassified_error')) + ' |')
    lines += ['',
        'The role-label diagnostic is exact normalized text only. It does not strip articles, append/remove “post”, repair spelling, or reinterpret such outputs as correct. Other errors remain unclassified. Raw responses and exact primary/secondary parent maps remain in RESULTS.json.', '',
        'The mixture has8–14 correct views per parent, but no parent has16. The conjunction exposes incomplete selectivity across conditions; it does not mean no individual responses improved. These post-run descriptions are exploratory and provide no promotion, preservation or confirmation claim.']
    with (ROOT / 'BINDING_ERROR_ANALYSIS.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('BINDING_ERROR_ANALYSIS_COMPLETE')


if __name__ == '__main__':
    main()
