"""Re-score completed LORA-596 artifacts without model execution or mutation.

Run as python -m experiments.continual_skill_audit_v1.audit_lora596
  --pilot-root PATH --records-root PATH --output NEW_JSON_PATH
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from .metrics import Outcome, compare, loss_upper_bound


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unique(rows):
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate artifact IDs')
    return indexed


class Inputs:
    def __init__(self):
        self.receipts = {}

    def read(self, path, jsonl=False):
        raw = path.read_bytes()
        self.receipts[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        text = raw.decode('utf-8')
        return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)


def audit(pilot: Path, records: Path) -> dict:
    inputs = Inputs()
    package = pilot / 'experiments/architecture_benchmark_001'
    evidence = package / 'evidence/retention-design-v1'
    lora = records / 'lora596-20260919'
    scorer_path = package / 'retention.py'
    inputs.receipts[str(scorer_path.resolve())] = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('historical_typed_scorer', scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    bank = unique(inputs.read(package / 'data/monitor_probe.jsonl', jsonl=True))
    labels = inputs.read(evidence / 'PROBE_LABEL_AUDIT.json')
    label_rows = unique(labels['rows'])
    source = inputs.read(package / 'outputs/retention-a329eb7b/B28/retention.json')
    source_rows = unique(source['rows'])
    candidate = unique(inputs.read(lora / 'eval/lora596-retention/retention_predictions.json')['predictions'])
    published = inputs.read(evidence / 'RETENTION_PROBE_int001_lora596.json')
    if not (bank.keys() == source_rows.keys() == candidate.keys() == label_rows.keys()):
        raise ValueError('arithmetic artifact coverage differs')
    bank_hash = inputs.receipts[str((package / 'data/monitor_probe.jsonl').resolve())]
    if published['probe_sha256'] != bank_hash:
        raise ValueError('arithmetic bank hash mismatch')
    arithmetic = {}
    for policy in scorer.BARE_POLICIES:
        overlay = scorer.build_overlay(labels, policy)
        old, new = [], []
        for key, row in bank.items():
            if fingerprint(row['messages']) != label_rows[key]['messages_sha256']:
                raise ValueError(f'label/prompt hash mismatch: {key}')
            target = overlay['targets'][key]
            stamp = fingerprint([row['messages'], target, policy])
            old_correct = scorer.score_prediction(source_rows[key]['source']['raw'],
                                                 source_rows[key]['source']['terminated'], target)['correct']
            new_correct = scorer.score_prediction(candidate[key]['text'], candidate[key]['terminated'], target)['correct']
            old.append(Outcome(key, row['group_id'], stamp, old_correct))
            new.append(Outcome(key, row['group_id'], stamp, new_correct))
        if {r.id for r in old if r.correct} != set(source['reference_ids_by_policy'][policy]):
            raise ValueError('source correct set does not reproduce frozen reference')
        result = compare(old, new)
        saved = published['by_bare_policy'][policy]
        if (result['source_correct'], result['candidate_correct'], result['lost']) != (
                saved['source_correct_total'], saved['arm_correct_total'], saved['lost_total']):
            raise ValueError('re-scored arithmetic differs from published outcome')
        result['published_reference_size'] = saved['reference_size']
        result['published_reference_is_source_candidate_union'] = saved['reference_size'] == result['source_correct'] + result['gained']
        result['per_class'] = {}
        for cls in sorted({r['reply_class'] for r in bank.values()}):
            ids = {k for k, r in bank.items() if r['reply_class'] == cls}
            result['per_class'][cls] = compare([r for r in old if r.id in ids], [r for r in new if r.id in ids])
        # Template grouping is a sensitivity analysis, not an iid template population.
        template_groups = {key: row['reply_class'] + ':' + label_rows[key]['independent_recomputation']['template']
                           for key, row in bank.items()}
        result['template_sensitivity'] = compare(
            [Outcome(r.id, template_groups[r.id], r.fingerprint, r.correct) for r in old],
            [Outcome(r.id, template_groups[r.id], r.fingerprint, r.correct) for r in new])
        arithmetic[policy] = result

    suite = pilot / 'Rineya_Long_Context_Demo_v1/data'
    prompts = unique(inputs.read(suite / 'all_prompts.jsonl', jsonl=True))
    targets = unique(inputs.read(suite / 'all_answer_key.jsonl', jsonl=True))
    lc_source = inputs.read(evidence / 'LONGCTX_supa.json')
    lc_arm = inputs.read(evidence / 'LONGCTX_int001_lora596.json')
    fields = ['implementation_digest', 'device', 'max_context', 'max_new_tokens', 'transformers', 'numerics']
    if any(lc_source[k] != lc_arm[k] for k in fields):
        raise ValueError('long-context runtime mismatch')
    if lc_source['rows_skipped_context_overflow'] or lc_arm['rows_skipped_context_overflow']:
        raise ValueError('context-overflow omissions')
    def normalize(text):
        return ' '.join(text.strip().lower().replace('.', ' ').split())
    outcomes = []
    for run in [lc_source, lc_arm]:
        if run['predictions'].keys() != targets.keys() or prompts.keys() != targets.keys():
            raise ValueError('long-context coverage mismatch')
        scored = []
        for key, target in targets.items():
            record = run['predictions'][key]
            if record['status'] != 'ok':
                raise ValueError(f'failed generation: {key}')
            first = record['prediction'].splitlines()[0] if record['prediction'] else ''
            accepted = target.get('accepted_answers') or [target['answer']]
            hit = normalize(first) in {normalize(a) for a in accepted}
            if hit != run['per_row_correct'][key]:
                raise ValueError(f'long-context scoring drift: {key}')
            scored.append(Outcome(key, target['scenario_id'], fingerprint([prompts[key], target]), hit))
        outcomes.append(scored)
    long_context = compare(*outcomes)
    long_context['per_variant'] = {}
    long_context['per_task'] = {}
    for field, destination in [('variant', 'per_variant'), ('task_type', 'per_task')]:
        for value in sorted({r[field] for r in targets.values()}):
            ids = {k for k, r in targets.items() if r[field] == value}
            long_context[destination][value] = compare(*[[r for r in rows if r.id in ids] for rows in outcomes])
    long_context['abstention'] = {}
    for name, run in [('source', lc_source), ('candidate', lc_arm)]:
        insufficient = [k for k, r in targets.items() if not r['answerable']]
        sufficient = [k for k, r in targets.items() if r['answerable']]
        literal = {k: r['prediction'].strip() == 'UNKNOWN' for k, r in run['predictions'].items()}
        long_context['abstention'][name] = {
            'insufficient_n': len(insufficient), 'literal_true_positive': sum(literal[k] for k in insufficient),
            'sufficient_n': len(sufficient), 'literal_false_positive': sum(literal[k] for k in sufficient),
            'literal_unanswerable_ids': sorted(k for k in insufficient if literal[k]),
        }

    views_path = records / 'select-views-001/SELECT_VIEWS.jsonl'
    views = unique(inputs.read(views_path, jsonl=True))
    acquisition = inputs.read(evidence / 'PROCESS_int001_lora596.json')
    if acquisition['views_sha256'] != inputs.receipts[str(views_path.resolve())]:
        raise ValueError('acquisition bank hash mismatch')
    acq_preds = unique(acquisition['predictions'])
    if acq_preds.keys() != views.keys():
        raise ValueError('acquisition coverage mismatch')
    hits = Counter()
    totals = Counter()
    for key, row in views.items():
        prediction = acq_preds[key]
        try:
            hit = json.dumps(json.loads(prediction['prediction']), sort_keys=True, separators=(',', ':')) == row['reference']
        except (ValueError, TypeError):
            hit = False
        if hit != prediction['canonical'] or prediction['reference'] != row['reference']:
            raise ValueError(f'acquisition scoring drift: {key}')
        hits[row['family']] += hit
        totals[row['family']] += 1
    acq_report = {'correct': sum(hits.values()), 'rows': sum(totals.values()), 'per_family': {}}
    for family, n in totals.items():
        floor_rec = acquisition['floors_published_before_any_model_call'][family]
        floor = max(floor_rec['copy_before'], floor_rec['constant_majority']) / floor_rec['n']
        lower = 1 - loss_upper_bound(n - hits[family], n, alpha=.025)
        acq_report['per_family'][family] = {'correct': hits[family], 'n': n, 'cp95_lower': lower,
                                            'floor': floor, 'clears_floor': lower > floor}
    acq_report['floors_cleared'] = sum(r['clears_floor'] for r in acq_report['per_family'].values())

    train = inputs.read(lora / 'arm-lora-r16/TRAIN.json')
    materialization = inputs.read(lora / 'arm-lora-r16-merged/MATERIALIZATION.json')
    for path in [records / 'qwenft001-20260918/train_qwen_lora.py', lora / 'merge_lora.py', lora / 'LORA596_PREREG.md']:
        inputs.receipts[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'schema': 1, 'evidence_kind': 'independent_cpu_rescore_of_existing_exploratory_outputs',
        'model_calls': 0, 'acquisition': acq_report, 'arithmetic': arithmetic, 'long_context': long_context,
        'training_receipt': {k: v for k, v in train.items() if k != 'steps'},
        'stored_dtype': materialization['stored_dtype'],
        'comparability': {
            'lora_vs_full_ft_is_single_variable': False,
            'confounds': ['BF16 training base vs registered FP32; FP32 export does not change training history',
                         'all 28 blocks vs control tail20',
                         'update-index linear WSD vs processed-token square-root WSD',
                         'same learning rate does not match effective delta norm or acquisition'],
            'training_lr_log_is_post_scheduler_step': True,
            'long_context_runtime_fields_equal': fields,
            'long_context_original_bank_hash_binding': 'not stored by historical probe; current bank rescoring reproduces all row flags',
            'weights_rehashed_in_this_audit': False,
        },
        'limitations': ['Historical selection/report-only data; no fresh confirmation.',
                        'Conditional bounds assume iid sampling at the stated unit and are unadjusted.',
                        'Template analysis is sensitivity only; templates were not randomly sampled.',
                        'LC normalization ignores periods and all but the first line; literal abstention is separate.',
                        'Raw answer rescoring verifies scores, not broad intelligence or train/test independence.'],
        'quality_improvement_established': False, 'release_permitted': False, 'input_sha256': inputs.receipts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--records-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite an audit')
    result = audit(args.pilot_root, args.records_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for label, result in [('arithmetic/permissive', result['arithmetic']['permissive']),
                          ('arithmetic/strict', result['arithmetic']['strict']),
                          ('long_context', result['long_context'])]:
        print(label, {k: result[k] for k in ['source_correct', 'candidate_correct', 'lost', 'gained', 'churn', 'damaged_groups']})


if __name__ == '__main__':
    main()
