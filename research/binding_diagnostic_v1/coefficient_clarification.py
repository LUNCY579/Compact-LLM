"""Create a transparent post-closeout clarification without changing prior evidence."""
from datetime import datetime, timezone
from pathlib import Path
import math
import statistics
from paths import ROOT
from common import read, sha, write_new, require


def main():
    ledger = read(ROOT / 'NATIVE_COEFFICIENT_AUDIT.json')
    weighted = [row['mixture_divided_by_original_coefficient'] for row in ledger['native_rows']
                for _ in range(row['original']['supervised_tokens'])]
    require(len(weighted) == 37090, 'native supervised-token inventory changed')
    proposal = ledger['proposed_integration']['groups']
    new_binding = sum(group['applied_native_lr'] * group['binding_supervised_tokens'] /
                      group['native_supervised_denominator'] for group in proposal)
    old_binding = ledger['coefficient_budgets']['mixture']['sum_applied_lr_times_group_loss_coefficient_fraction']['binding_v2']
    current_native = {row['id']: row['original']['lr_times_loss_coefficient_per_supervised_token'] for row in ledger['native_rows']}
    for group in proposal:
        require(all(current_native[identifier] == group['coefficient_per_supervised_token_for_both_sources']
                    for identifier in group['native_ids']), 'proposal fails exact native coefficient preservation')
    value = {'schema': 1, 'status': 'VERIFIED_COEFFICIENT_WEIGHTING_AND_PROPOSAL_CLARIFICATION',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_ledger_sha256': sha(ROOT / 'NATIVE_COEFFICIENT_AUDIT.json'),
        'view_weighted_median_ratio': ledger['native_coefficient_ratio_distribution']['median'],
        'native_views': 2697, 'native_supervised_tokens': len(weighted),
        'supervised_token_weighted_median_ratio': statistics.median(weighted),
        'observed_mixture_binding_lr_weighted_coefficient_sum': old_binding,
        'proposed_binding_lr_weighted_coefficient_sum': new_binding,
        'proposed_divided_by_observed_binding_coefficient_budget': new_binding / old_binding,
        'native_coefficients_preserved_exactly_by_proposed_partition': True,
        'maximum_accumulated_tokens_per_update': 3114, 'maximum_sequence_context': 2048,
        'interpretation': 'Same binding views do not imply unchanged binding coefficient exposure. This is an integration-regime test, not isolation of native weighting at fixed binding exposure. Coefficients are not gradient influence.',
        'primary_results_or_criteria_changed': False, 'model_calls': 0, 'training_permission': False,
        'analyst_sha256': sha(Path(__file__).resolve())}
    require(math.isclose(new_binding / old_binding, 0.677169196409, rel_tol=0, abs_tol=1e-11), 'independent reviewed ratio differs')
    write_new(ROOT / 'COEFFICIENT_CLARIFICATION.json', value)
    old = (ROOT / 'INTERPRETATION.md').read_text(encoding='utf-8')
    original = ('For each native supervised token, the audit compared the actual applied LR\n'
        'divided by its group’s supervised denominator. Mixture/original ratios have\n'
        'median1.45411672, range0.82687838–2.49591258.')
    replacement = ('Each native view’s supervised tokens share an applied-LR/group-denominator\n'
        'coefficient. Mixture/original ratios have median1.45411672 across2697 views\n'
        'and median1.43220339 across37090 supervised tokens; range0.82687838–2.49591258.')
    require(original in old, 'original interpretation wording changed')
    revised = old.replace(original, replacement)
    sentence = 'processed tokens per update would be3114, while sequence lengths stay unchanged.'
    require(sentence in revised, 'original proposal wording changed')
    revised = revised.replace(sentence, sentence + '\n'
        'This3114 is accumulated over serial sequences, not a3114-token context;\n'
        'the model context bound remains2048. The proposal also reduces the summed\n'
        'LR-weighted binding coefficient from0.00146714048 to0.00099350234, a ratio\n'
        'of0.67716920. It holds the views fixed while changing binding coefficient\n'
        'exposure too. This tests an integration regime, not an isolated change to\n'
        'native weighting at constant binding exposure.')
    revised = ('Revision2 clarifies coefficient weighting and the proposed binding exposure.\n'
               'The original interpretation and closeout remain preserved; scientific results\n'
               'and all failed criteria are unchanged. See COEFFICIENT_CLARIFICATION.json.\n\n' + revised)
    with (ROOT / 'INTERPRETATION_v2.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(revised)
    write_new(ROOT / 'CLOSEOUT_ADDENDUM.json', {'schema': 1, 'status': 'ANALYSIS_CLARIFICATION_ONLY',
        'original_closeout_sha256': sha(ROOT / 'CLOSEOUT.json'),
        'original_results_sha256': sha(ROOT / 'RESULTS.json'),
        'coefficient_clarification_sha256': sha(ROOT / 'COEFFICIENT_CLARIFICATION.json'),
        'latest_interpretation': str(ROOT / 'INTERPRETATION_v2.md'),
        'latest_interpretation_sha256': sha(ROOT / 'INTERPRETATION_v2.md'),
        'original_goal_achieved': False, 'models_changed': False, 'criteria_changed': False,
        'model_execution_permission': False, 'training_permission': False})
    print({'weighted_median': value['supervised_token_weighted_median_ratio'], 'binding_coefficient_ratio': new_binding / old_binding})


if __name__ == '__main__':
    main()
