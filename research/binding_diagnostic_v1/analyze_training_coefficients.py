"""Data-only audit of applied native loss coefficients and a zero-addend design."""
from collections import Counter
import json
from pathlib import Path
import statistics
from paths import BASE, ROOT
from common import sha, read, write_new, require


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    records = BASE / 'rineya-dataset-retention-records'
    original = records / 'highdose-lora001-resume-20260920'
    mixture = records / 'binding-mixture001-20260920'
    native_data = records / 'doseint001-20260919/prepared-dual899/microfit-train.jsonl'
    files = [native_data, original / 'PLAN.json', original / 'journal/runs/train-001/steps.jsonl',
             mixture / 'PLAN.json', mixture / 'prepared/microfit-train.jsonl',
             mixture / 'prepared/SOURCE_MEMBERSHIP.json', mixture / 'journal/runs/train-001/steps.jsonl']
    pins = {str(path): sha(path) for path in files}
    native = rows(native_data)
    mixed = rows(mixture / 'prepared/microfit-train.jsonl')
    membership = json.loads((mixture / 'prepared/SOURCE_MEMBERSHIP.json').read_text())
    old_plan, new_plan = read(original / 'PLAN.json'), read(mixture / 'PLAN.json')
    old_steps = rows(original / 'journal/runs/train-001/steps.jsonl')
    new_steps = rows(mixture / 'journal/runs/train-001/steps.jsonl')
    require(len(native) == 2697 and len(mixed) == 3465 and len(old_steps) == 193 and len(new_steps) == 296,
            'registered training geometry differs')
    supervised = lambda row: sum(value != -100 for value in row['labels'][1:])
    def coefficients(data, groups, steps, membership_=None):
        ledger = {}
        per_group = []
        for index, (group, step) in enumerate(zip(groups, steps)):
            require(step['updates'] == index + 1 and step['row_ids'] == [data[i]['id'] for i in group], 'step/group order differs')
            denominator = sum(supervised(data[i]) for i in group)
            require(denominator > 0 and step['lr'] > 0, 'invalid denominator/applied LR')
            counts = Counter()
            for row_index in group:
                row = data[row_index]
                source = 'native_dual899' if membership_ is None else membership_[row_index]['source']
                counts[source] += supervised(row)
                ledger[row['id']] = {'group': index + 1, 'supervised_tokens': supervised(row),
                    'applied_lr': step['lr'], 'group_supervised_denominator': denominator,
                    'lr_times_loss_coefficient_per_supervised_token': step['lr'] / denominator}
            per_group.append({'group': index + 1, 'applied_lr': step['lr'], 'supervised_denominator': denominator,
                'source_supervised_counts': dict(counts),
                'source_loss_coefficient_fractions': {source: n / denominator for source, n in counts.items()}})
        require(len(ledger) == len(data), 'duplicate or missing training views')
        return ledger, per_group
    old, old_groups = coefficients(native, old_plan['update_groups'], old_steps)
    new, new_groups = coefficients(mixed, new_plan['update_groups'], new_steps, membership)
    comparisons = []
    for row_index, row in enumerate(mixed):
        member = membership[row_index]
        require(member['id'] == row['id'] and member['mixture_index'] == row_index, 'source membership differs')
        if member['source'] != 'native_dual899':
            continue
        require(row == native[member['source_index']], 'original native row changed')
        a, b = old[row['id']], new[row['id']]
        comparisons.append({'id': row['id'], 'original': a, 'mixture': b,
            'mixture_divided_by_original_coefficient': b['lr_times_loss_coefficient_per_supervised_token'] / a['lr_times_loss_coefficient_per_supervised_token']})
    require(len(comparisons) == 2697, 'native comparison inventory differs')
    ratios = sorted(row['mixture_divided_by_original_coefficient'] for row in comparisons)
    budget = {}
    for label, groups in (('original', old_groups), ('mixture', new_groups)):
        totals = Counter(); weighted = Counter()
        for group in groups:
            for source, fraction in group['source_loss_coefficient_fractions'].items():
                totals[source] += fraction
                weighted[source] += group['applied_lr'] * fraction
        budget[label] = {'sum_group_loss_coefficient_fractions': dict(totals),
                         'sum_applied_lr_times_group_loss_coefficient_fraction': dict(weighted)}
    # One data-only construction: all original native optimizer groups remain
    # intact. Existing shuffled binding order is partitioned over those groups.
    binding_indices = [i for i, member in enumerate(membership) if member['source'] == 'binding_v2']
    proposal = []
    for index, native_group in enumerate(old_plan['update_groups']):
        extra = binding_indices[index * len(binding_indices) // 193:(index + 1) * len(binding_indices) // 193]
        denominator = sum(supervised(native[i]) for i in native_group)
        binding_supervised = sum(supervised(mixed[i]) for i in extra)
        proposal.append({'group': index + 1, 'native_row_indices': native_group, 'binding_mixture_row_indices': extra,
            'native_ids': [native[i]['id'] for i in native_group], 'binding_ids': [mixed[i]['id'] for i in extra],
            'native_supervised_denominator': denominator, 'binding_supervised_tokens': binding_supervised,
            'effective_binding_mean_coefficient': binding_supervised / denominator,
            'applied_native_lr': old_steps[index]['lr'],
            'native_processed_tokens': sum(len(native[i]['ids']) for i in native_group),
            'binding_processed_tokens': sum(len(mixed[i]['ids']) for i in extra),
            'coefficient_per_supervised_token_for_both_sources': old_steps[index]['lr'] / denominator})
    require([i for group in proposal for i in group['binding_mixture_row_indices']] == binding_indices,
            'proposed binding partition repeats or drops a view')
    require(sum(group['native_processed_tokens'] + group['binding_processed_tokens'] for group in proposal) == 577417,
            'proposal changes total data cost')
    require(all(sha(Path(path)) == value for path, value in pins.items()), 'source evidence changed during coefficient audit')
    result = {'schema': 1, 'status': 'COMPLETE_DATA_ONLY_COEFFICIENT_AUDIT', 'source_files_sha256': pins,
        'analyst_sha256': sha(Path(__file__).resolve()), 'original_updates': 193, 'mixture_updates': 296,
        'native_row_count': len(comparisons), 'native_coefficient_ratio_distribution': {
            'minimum': ratios[0], 'p10': ratios[int(.1 * (len(ratios) - 1))], 'median': statistics.median(ratios),
            'p90': ratios[int(.9 * (len(ratios) - 1))], 'maximum': ratios[-1]},
        'coefficient_budgets': budget, 'native_rows': comparisons,
        'proposed_integration': {'status': 'DESIGN_ONLY_NOT_QUALIFIED_OR_GRANTED',
            'rule': 'For each original native group, divide the native and appended binding summed token losses by the ORIGINAL native supervised denominator; retain original native applied-LR schedule and193 optimizer updates.',
            'separate_tuned_task_weight': False, 'binding_order': 'Existing mixture binding subsequence, partitioned by floor(g*768/193).',
            'native_coefficients_preserved_by_construction': True,
            'zero_addend_requirement': 'Skip all binding forward/backward work and RNG use when disabled; independently prove exact recovery of original control optimizer/checkpoint states before accepting active integration.',
            'maximum_combined_processed_tokens_per_update': max(g['native_processed_tokens'] + g['binding_processed_tokens'] for g in proposal),
            'binding_mean_coefficient_range': [min(g['effective_binding_mean_coefficient'] for g in proposal), max(g['effective_binding_mean_coefficient'] for g in proposal)],
            'groups': proposal},
        'limitations': ['Scalar loss coefficients are not measured gradient norms, parameter movement or causal influence.',
            'Global clipping and Adam moments couple both sources; active binding changes subsequent native gradients.',
            'Original control already failed retention at38/47. Exact zero-addend recovery is a qualification limit, not proof of a preservation solution.',
            'Holding the binding corpus fixed tests integration and does not directly fix unseen-domain/wording coverage.'],
        'model_calls': 0, 'weights_read': False, 'training_permission': False, 'quality_claim': False}
    write_new(ROOT / 'NATIVE_COEFFICIENT_AUDIT.json', result)
    lines = ['# Native training coefficient audit', '',
        'The original high-dose LoRA used193 optimizer groups; the mixture used296. Both presented every native view once. This audit uses the LR actually applied in each saved step and the exact shifted response-plus-EOS denominator.', '',
        '| Native per-token coefficient ratio: mixture / original | Value |', '|---|---:|']
    lines += [f'| {key} | {value:.9g} |' for key, value in result['native_coefficient_ratio_distribution'].items()]
    lines += ['', '| Run/source | Sum of group coefficient fractions | Sum of applied LR × fraction |', '|---|---:|---:|']
    for label, values in budget.items():
        for source, value in values['sum_group_loss_coefficient_fractions'].items():
            lines.append(f"| {label}/{source} | {value:.9g} | {values['sum_applied_lr_times_group_loss_coefficient_fraction'][source]:.9g} |")
    lines += ['', 'These are coefficients in a nonlinear optimizer, not measurements of gradient influence or proof of why retention changed.', '',
        'A bounded next design can retain the original193 native groups, original native denominators, native-token LR clock and update count. Append the existing768 binding views over those groups, using the SAME per-token denominator for both sources. This introduces no tuned task-loss coefficient or post-hoc weight transform. A data-only proposed partition is retained in the JSON ledger.', '',
        'Qualification must demonstrate exact original-control recovery when binding work is disabled, including optimizer state, update order and RNG behavior. Binding work must still be fully charged, and active gradients still share clipping/Adam. The existing control failed38/47 retention; restoring its limiting trajectory does not itself meet the target. No training grant or model promotion is provided by this audit.']
    with (ROOT / 'NATIVE_COEFFICIENT_AUDIT.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'native_coefficient_ratio_distribution': result['native_coefficient_ratio_distribution'],
                      'coefficient_budgets': budget, 'proposed_max_group_tokens': result['proposed_integration']['maximum_combined_processed_tokens_per_update']}))


if __name__ == '__main__':
    main()
