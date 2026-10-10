"""Exploratory leave-one-WORLD-out prediction of already inspected DEV outputs.

Fit one of seven selectors to saved source OUTPUTS, never to target correctness.
Candidate order and tie-break are fixed to the order in the task request:
first, middle, last, max payload, min payload, max revision, min revision.
All 16 worlds were previously inspected. Withholding a world computationally
does not make it an untouched holdout, confirmation, model-generalization test
or evidence of an intervention gain. Native strict/typed update scores stay 0.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CANDIDATES = ('first', 'middle', 'last', 'max_payload', 'min_payload', 'max_revision', 'min_revision')
EXPECTED_RESULTS = '0806a6a6a5704e09ba4aca12865157b71bbacd794e4a773df2c1c7bfc316eba5'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def numeric_code_lexeme(raw):
    """Frozen anchor outputs are JSON integers; no target-dependent conversion."""
    value = json.loads(raw)
    require(type(value) is int, 'frozen update output must be a JSON integer')
    return compact(value)


def select(name, events):
    if name == 'first':
        event = events[0]
    elif name == 'middle':
        event = events[1]
    elif name == 'last':
        event = events[2]
    elif name == 'max_payload':
        event = max(events, key=lambda e: int(e['value']))
    elif name == 'min_payload':
        event = min(events, key=lambda e: int(e['value']))
    elif name == 'max_revision':
        event = max(events, key=lambda e: e['version'])
    elif name == 'min_revision':
        event = min(events, key=lambda e: e['version'])
    else:
        raise ValueError('unknown predeclared selector')
    return event['value']


def fit(examples):
    """Only world, observed-output lexeme and visible events enter fitting."""
    require(len(examples) == 360 and len({e['world_id'] for e in examples}) == 15,
            'each fit must contain exactly 15 whole worlds and 360 rows')
    scores = {name: sum(select(name, e['events']) == e['output_lexeme'] for e in examples)
              for name in CANDIDATES}
    ranked = sorted(CANDIDATES, key=lambda name: (-scores[name], CANDIDATES.index(name)))
    best = ranked[0]
    return best, scores, [name for name in CANDIDATES if scores[name] == scores[best]]


def tally(rows):
    n = len(rows)
    return {'rows': n,
        'selected_rule_saved_output_agreements': sum(r['rule_matches_observed_output'] for r in rows),
        'selected_rule_saved_output_disagreements': sum(not r['rule_matches_observed_output'] for r in rows),
        'selected_rule_saved_output_agreement_fraction': sum(r['rule_matches_observed_output'] for r in rows)/n if n else None,
        'observed_output_latest_payload_lexeme_matches': sum(r['observed_output_matches_gold_lexeme'] for r in rows),
        'selected_rule_latest_payload_lexeme_matches': sum(r['rule_matches_gold_lexeme'] for r in rows),
        'native_original_strict_correct': sum(r['native_strict_correct'] for r in rows),
        'native_original_typed_correct': sum(r['native_typed_correct'] for r in rows),
        'observed_revision_numeral_rows': sum(r['observed_revision_numeral'] for r in rows),
        'disagreement_classes': dict(Counter(r['observed_class'] for r in rows if not r['rule_matches_observed_output']))}


def analyze():
    path = ROOT/'anchor-patterns.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    data = receipt['results']
    require(digest(data) == receipt['results_sha256'] == EXPECTED_RESULTS, 'preserved pattern results changed')
    require(sha(ROOT/'anchor_patterns.py') == receipt['analyzer_sha256'], 'preserved input analyzer changed')
    for reference in receipt['input_files'].values():
        require(sha(reference['path']) == reference['sha256'], 'frozen native input bytes changed')
    rows = data['update_rows']
    require(len(rows) == 384 and len({r['id'] for r in rows}) == 384, 'registered update row count changed')
    worlds = sorted({r['world_id'] for r in rows})
    require(len(worlds) == 16, 'exactly 16 dependent worlds required')
    examples = []
    by_id = {}
    for row in rows:
        events = row['events']
        require(len(events) == 3 and len({e['value'] for e in events}) == 3
                and len({e['version'] for e in events}) == 3, 'selector tie/geometry changed')
        require(all(isinstance(e['value'], str) and len(e['value']) == 3 and e['value'].isdigit()
                    and e['value'][0] != '0' and type(e['version']) is int for e in events), 'fixed numeric code/revision schema changed')
        output = numeric_code_lexeme(row['prediction'])
        require(output == row['decoded']['code_lexeme'], 'numeric decoder differs from preserved diagnostic')
        require(row['strict'] is False and row['typed'] is False, 'original update correctness changed')
        # Exclude all gold/correctness/role fields from training examples.
        examples.append({'id': row['id'], 'world_id': row['world_id'],
                         'events': events, 'output_lexeme': output})
        by_id[row['id']] = row
    folds, predicted = [], []
    for held_out in worlds:
        train = [e for e in examples if e['world_id'] != held_out]
        test = [e for e in examples if e['world_id'] == held_out]
        require(len(test) == 24 and not {e['world_id'] for e in train}.intersection(e['world_id'] for e in test), 'world leakage')
        chosen, scores, tied = fit(train)
        test_rows = []
        for example in sorted(test, key=lambda e: e['id']):
            row = by_id[example['id']]
            emitted = select(chosen, example['events'])
            # Gold is read only AFTER fitting and is never used for selection.
            gold = max(example['events'], key=lambda e: e['version'])['value']
            require(gold == row['latest_payload'], 'independent gold oracle differs')
            recorded = {
                'id': example['id'], 'world_id': held_out, 'role': row['role'],
                'counterfactual_id': row['counterfactual_id'], 'permutation': row['permutation'],
                'latest_position': row['latest_position'], 'gold_payload_numeric_rank': row['latest_payload_numeric_rank'],
                'gold_vs_max_payload_conflict': row['comparison_key_conflict'],
                'selected_rule': chosen, 'rule_predicted_payload_lexeme': emitted,
                'observed_raw_prediction': row['prediction'], 'observed_output_numeric_lexeme': example['output_lexeme'],
                'gold_latest_payload_lexeme': gold,
                'rule_matches_observed_output': emitted == example['output_lexeme'],
                'observed_output_matches_gold_lexeme': example['output_lexeme'] == gold,
                'rule_matches_gold_lexeme': emitted == gold,
                'native_strict_correct': row['strict'], 'native_typed_correct': row['typed'],
                'observed_revision_numeral': row['selection_class'] == 'visible_revision_numeral',
                'observed_class': row['selection_class'], 'events': example['events'],
            }
            test_rows.append(recorded)
            predicted.append(recorded)
        folds.append({'computationally_withheld_world': held_out,
            'training_worlds': sorted({e['world_id'] for e in train}), 'training_rows':360,'test_rows':24,
            'selected_rule': chosen, 'training_output_agreements_by_candidate': scores,
            'tied_best_rules_in_fixed_order': tied,
            'selected_training_output_agreement_fraction': scores[chosen]/360,
            'test_output_prediction': tally(test_rows)})
    require(len(predicted) == 384 and len({r['id'] for r in predicted}) == 384, 'each spent DEV row must be computationally withheld once')
    global_scores = {}
    for name in CANDIDATES:
        global_scores[name] = {
            'saved_output_lexeme_agreements': sum(select(name,e['events']) == e['output_lexeme'] for e in examples),
            'latest_payload_gold_lexeme_agreements': sum(select(name,e['events']) == max(e['events'],key=lambda v:v['version'])['value'] for e in examples),
            'source_agreement_by_role':{role:sum(select(name,e['events']) == e['output_lexeme'] for e in examples if by_id[e['id']]['role']==role)
                                        for role in ('near_control','distant_heldout')},
            'rows':384,
        }
    disagreed = [r for r in predicted if not r['rule_matches_observed_output']]
    results = {
        'global_descriptive_candidate_scores_not_crossvalidated':global_scores,
        'selected_rules_by_fold':dict(Counter(f['selected_rule'] for f in folds)),
        'folds_with_training_ties':sum(len(f['tied_best_rules_in_fixed_order'])>1 for f in folds),
        'leave_one_world_out_saved_output_prediction':tally(predicted),
        'computationally_withheld_output_prediction_by_role':{role:tally([r for r in predicted if r['role']==role]) for role in ('near_control','distant_heldout')},
        'computationally_withheld_output_prediction_by_gold_rank':{str(rank):tally([r for r in predicted if r['gold_payload_numeric_rank']==rank]) for rank in range(3)},
        'computationally_withheld_output_prediction_by_gold_max_payload_conflict':{str(conflict):tally([r for r in predicted if r['gold_vs_max_payload_conflict'] is conflict]) for conflict in (False,True)},
        'computationally_withheld_output_prediction_by_latest_text_position':{str(position):tally([r for r in predicted if r['latest_position']==position]) for position in range(3)},
        'counterexamples':{
            'selected_rule_disagrees_with_correct_latest_payload_lexeme':sorted([r for r in disagreed if r['observed_output_matches_gold_lexeme']],key=lambda r:r['id'])[:8],
            'revision_number_outputs_no_payload_selector_can_predict':sorted([r for r in disagreed if r['observed_revision_numeral']],key=lambda r:r['id'])[:8],
            'selected_rule_disagrees_with_other_visible_payload':sorted([r for r in disagreed if not r['observed_revision_numeral'] and not r['observed_output_matches_gold_lexeme']],key=lambda r:r['id'])[:8],
        },
        'folds':folds, 'computationally_withheld_row_predictions':predicted,
    }
    return {
        'schema':'RINEYA-EXPLORATORY-SPENT-DEV-LOOWO/v1',
        'status':'COMPUTATIONAL_FALSIFICATION_OF_SIMPLE_OUTPUT_SELECTORS_ONLY',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'candidate_selectors_and_fixed_tie_break':list(CANDIDATES),
        'tie_break_origin':'The candidate order stated in the task request, declared before this calculation; not selected from observed fold results.',
        'fit_objective':'Maximize exact agreement with saved output numeric lexemes on 15 whole worlds; never optimize gold correctness.',
        'decoder':'Only finite JSON integer scalar from the fixed 384 update rows; canonical JSON text matched to visible quoted code lexemes. Type erasure is diagnostic only.',
        'training_fields':['id','world_id','events','output_lexeme'],
        'design':{'worlds':16,'rows_per_world':24,'rows':384,'training_worlds_per_fold':15,'training_rows_per_fold':360,'computationally_withheld_worlds_per_fold':1,'computationally_withheld_rows_per_fold':24},
        'input_receipt':{'path':str(path),'sha256':sha(path),'results_sha256':EXPECTED_RESULTS},
        'raw_native_inputs':receipt['input_files'],'frozen_bindings':receipt['frozen_bindings'],
        'analyzer_sha256':sha(__file__),'results_sha256':digest(results),'results':results,
        'interpretation_boundary':['Agreement predicts which visible payload lexeme the saved source emitted; it is not model correctness.',
            'Selector gold agreement is a property of a scripted rule on parsed facts, not a Rineya capability or intervention gain.',
            '43 visible revision-numeral outputs cannot be generated by any of these payload-returning selectors.',
            'The global rule selector is chosen from 15 computationally retained worlds and tested on one computationally withheld world; all worlds have already been inspected.',
            'This is exploratory cross-validation within spent synthetic DEV, not an untouched holdout, confirmation bank, new native run, proof of generalization or internal algorithm.'],
        'actions':{'model_or_gpu_calls':False,'native_retry':False,'source_corpus_metrics_proposal_edits':False,'new_model_gain_measured':False}}


def self_check():
    events=[{'value':'101','version':3},{'value':'909','version':1},{'value':'505','version':2}]
    expected={'first':'101','middle':'909','last':'505','max_payload':'909','min_payload':'101','max_revision':'101','min_revision':'909'}
    require({name:select(name,events) for name in CANDIDATES}==expected,'selector definitions changed')
    require(numeric_code_lexeme('101')=='101','numeric decoder changed')
    try:
        numeric_code_lexeme('"101"')
    except ValueError:
        pass
    else:
        raise ValueError('numeric-only decoder accepted a string')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'loowo.json'))
    args=parser.parse_args()
    self_check()
    output=Path(args.output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists() and output.parent.is_dir(),'new exclusive output below the pattern root required')
    value=analyze()
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':value['status'],'output':str(output),'results_sha256':value['results_sha256'],
        'selected_rules_by_fold':value['results']['selected_rules_by_fold'],
        'crossvalidated_saved_output_prediction':value['results']['leave_one_world_out_saved_output_prediction']},sort_keys=True))
