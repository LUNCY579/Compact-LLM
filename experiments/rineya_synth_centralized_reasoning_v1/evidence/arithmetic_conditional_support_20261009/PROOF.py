"""Conditional integer planning proof; no data, model or candidate imports."""
import hashlib
import json
from pathlib import Path

bp = Path('C:/Users/clclc/BusinessProject')
out = bp / 'rineya-arithmetic-support-planning-20261009-01'
policy_path = bp / 'ChatBubble-rineya-synth-centralized-reasoning-v1/experiments/rineya_synth_centralized_reasoning_v1/SYNTH_ARITHMETIC_GUARD_POLICY001.json'
raw = policy_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '42ee9c26f0fe1ab42c368618a5c25385fe09dac0f5777bab92df5c34c5fa05b7'
policy = json.loads(raw.decode('utf-8'))
assert policy['parent_count'] == 128 and policy['row_count'] == 512
assert policy['precision']['final_domain_ids'] is None
assert policy['precision']['eligible_parent_count'] is None
assert policy['precision']['rule'] == 'alpha=0.05/(2*D), D is frozen explicit domain ID count; zero-loss support ceil(log(alpha)/log(0.95)); siblings count one parent'

def satisfies(n, domains):
    # U=1-alpha**(1/n)<=1/20 iff (19/20)**n<=alpha=1/(40D).
    return 40 * domains * (19 ** n) <= 20 ** n

rows = []
for domains in (3, 4, 5, 6, 8, 12, 16, 17, 18):
    n = next(n for n in range(1, 1025) if satisfies(n, domains))
    assert satisfies(n, domains) and not satisfies(n - 1, domains)
    rows.append({'hypothetical_frozen_domain_count': domains,
                 'minimum_eligible_independent_parents_if_zero_loss': n,
                 'maximum_ineligible_of_128_if_zero_loss': 128 - n if n <= 128 else None,
                 'potentially_feasible_at_128_only_if_all_assumptions_hold': n <= 128,
                 'passing_integer_comparison': f'{40 * domains}*19^{n} <= 20^{n}',
                 'preceding_integer_comparison_fails': f'{40 * domains}*19^{n-1} > 20^{n-1}'})
maximum_domains = (20 ** 128) // (40 * (19 ** 128))
assert maximum_domains == 17
assert satisfies(128, 17) and not satisfies(128, 18)
result = {
    'schema': 'RINEYA_CONDITIONAL_RETENTION_SUPPORT_PLANNING_1',
    'status': 'EXACT_INTEGER_CONDITIONAL_DESIGN_PROOF_NOT_MODEL_EVIDENCE',
    'source_policy': {'path': str(policy_path).replace('\\', '/'), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()},
    'proof': 'For n>0 independent eligible parents and zero observed losses, the declared one-sided upper bound U=1-(1/(40D))^(1/n) is at most0.05 iff40D*19^n<=20^n. Integer comparisons avoid floating-log rounding. The table proves each minimal n using its immediate predecessor. It does not prove that these binomial/sampling assumptions apply to the actual candidate population.',
    'conditional_rows': rows,
    'maximum_hypothetical_D_at_n128_and_zero_loss': maximum_domains,
    'actual_state': {'frozen_domain_ids': None, 'eligible_independent_M0_correct_parents': None, 'actual_losses': None, 'actual_upper_bound': None, 'retention_precision': 'NOT_MEASURED', 'model_gain': 'NOT_MEASURED'},
    'interpretation': 'Hypothetical D4 requires99 eligible independent parents; D5 requires104. D18 cannot meet this declared bound with at most128 eligible parents even if zero losses. These are conditional capacity facts, not selected domain counts, enrolled units, independent support or measured preservation.512 grouped views cannot replace canonical independent parents. Missing native outputs/scorer qualification/lineage/independence mean even128 is only a candidate cap.',
    'next_action': 'Existing owner freezes the actual explicit domain-ID roster and reviews scorer, parent aggregation, sampling/lineage independence and enrollment before candidate outcomes. Preserve all predetermined candidates and all M0 outcomes. Insufficient M0-correct support stays inconclusive; no post-M0 replacement, sibling inflation, merging domains, changed alpha/margin or automatic larger draw.',
    'boundaries': 'No source policy, population, domain roster, budget, margin, training dose or evaluator is changed. This source-bound calculation is an offline planning receipt only. It grants no scoring/tokenizer/native/enrollment authority or parent/RSD eligibility, and does not establish broad arithmetic coverage or Rineya gains.',
    'native_execution_authorized': False, 'protected_bank_access_authorized': False,
    'training_authorized': False, 'enrollment_authorized': False,
    'quality_improvement_established': False,
    'root_candidate_imports_or_execution': 0,
}
with (out / 'REPORT.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
report_raw = (out / 'REPORT.json').read_bytes()
print(json.dumps({'path': str(out / 'REPORT.json').replace('\\', '/'), 'bytes': len(report_raw), 'sha256': hashlib.sha256(report_raw).hexdigest(), 'rows': len(rows)}))
