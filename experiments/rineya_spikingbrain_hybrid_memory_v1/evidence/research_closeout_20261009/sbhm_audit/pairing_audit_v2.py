"""Offline SBHM-001 closeout: raw-prompt replay, paired DEV scores and exposure.

Only Python stdlib; never imports experiment code, a tokenizer, or a model.
Reads explicit unprotected TRAIN/DEV and existing native-run receipts. Output
uses exclusive creation. Token distances are receipt/metadata cross-checks,
not a fresh tokenization or causal proof that recurrence conveyed evidence.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import random
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compact(value):
    return json.dumps(value, separators=(",", ":"))


def norm(value):
    return value.strip().lower()


def parse_record(text):
    patterns = [
        (r'(\w+): (\w+)="([^"]+)"; version=(\d+)\.', (0, 1, 2, 3)),
        (r'At revision (\d+), (\w+) recorded (\w+) "([^"]+)"\.', (1, 2, 3, 0)),
        (r'Revision (\d+) \| (\w+) \| (\w+) \| "([^"]+)"\.', (1, 2, 3, 0)),
        (r'(\w+) of (\w+) was "([^"]+)" in release (\d+)\.', (1, 0, 2, 3)),
        (r'(\w+): (\w+)="([^"]+)"\.', (0, 1, 2, None)),
        (r'(\w+) recorded (\w+) "([^"]+)"\.', (0, 1, 2, None)),
        (r'(\w+) \| (\w+) \| "([^"]+)"\.', (0, 1, 2, None)),
        (r'The (\w+) of (\w+) is "([^"]+)"\.', (1, 0, 2, None)),
    ]
    for pattern, order in patterns:
        m = re.fullmatch(pattern, text)
        if m:
            parts = m.groups()
            value = dict(subject=parts[order[0]], field=parts[order[1]], value=parts[order[2]])
            if order[3] is not None:
                value['version'] = int(parts[order[3]])
            return value
    raise ValueError('Unrecognized factual record: ' + text)


def evidence_oracle(prompt):
    blocks = prompt.split('\n---\n')
    if len(blocks) != 4:
        raise ValueError('Expected complete explicit evidence/filler/query blocks')
    q = blocks[-1]
    m = re.search(r'\b([A-Z]{5})\b', q)
    if not m:
        raise ValueError('No query subject')
    subject = m[1]
    evidence = next(b for b in blocks[1:3] if '"' in b)
    facts = [parse_record(line) for line in evidence.splitlines()]
    lookup = {(f['subject'], f['field']): f['value'] for f in facts}
    if 'north then south' in q:
        out = dict(a=lookup[(subject, 'north')], b=lookup[(subject, 'south')])
        fields = ['north', 'south']
    elif 'route then destination' in q:
        middle = lookup[(subject, 'route')]
        terminal = lookup[(middle, 'destination')]
        out = dict(v=lookup[(terminal, 'code')])
        fields = ['route', 'destination', 'code']
    elif 'greatest numbered revision' in q:
        matches = [f for f in facts if f['subject'] == subject and f['field'] == 'code']
        out = dict(v=max(matches, key=lambda f: f['version'])['value'])
        fields = ['code']
    else:
        fields = [next(field for field in ['shipment', 'measurement', 'code'] if field in q)]
        out = dict(v=lookup.get((subject, fields[0])))
        if out['v'] is None and '{"v":null} when absent' not in q:
            raise ValueError('Absent query lacks explicit null policy')
    return compact(out), facts, subject, fields


def retention_oracle(prompt):
    p = prompt.split(' Return only ')[0]
    if m := re.fullmatch(r'Compute (\d+) (plus|minus) (\d+)\.', p):
        value = int(m[1]) + int(m[3]) if m[2] == 'plus' else int(m[1]) - int(m[3])
    elif m := re.fullmatch(r'What is (\d+) multiplied by (\d+)\?', p):
        value = int(m[1]) * int(m[2])
    elif m := re.fullmatch(r'Add (\d+) and (\d+), then subtract (\d+)\.', p):
        value = int(m[1]) + int(m[2]) - int(m[3])
    elif m := re.fullmatch(r'A=(true|false); B=(true|false)\. Evaluate (AND\(A,B\)|XOR\(A,B\)|OR\(A,B\)|NOT\(AND\(A,B\)\))\.', p):
        a, b = m[1] == 'true', m[2] == 'true'
        op = m[3]
        value = {'AND(A,B)': a and b, 'XOR(A,B)': a != b, 'OR(A,B)': a or b, 'NOT(AND(A,B))': not(a and b)}[op]
        value = str(value).lower()
    elif m := re.fullmatch(r'Rule: if reading is (at least|less than|greater than|at most) (\d+), output (\w+); otherwise output (\w+)\. Reading=(\d+)\.', p):
        x, threshold = int(m[5]), int(m[2])
        truth = {'at least': x >= threshold, 'less than': x < threshold, 'greater than': x > threshold, 'at most': x <= threshold}[m[1]]
        value = m[3] if truth else m[4]
    elif m := re.fullmatch(r'List (\[[\d,]+\])\. Return the (minimum|maximum|median|second smallest) number\.', p):
        numbers = sorted(json.loads(m[1]))
        value = numbers[{'minimum': 0, 'maximum': -1, 'median': len(numbers)//2, 'second smallest': 1}[m[2]]]
    elif p.startswith('Table: '):
        body, subject = re.fullmatch(r'Table: (.+)\. Copy the value assigned to (\w+)\.', p, re.S).groups()
        pairs = re.findall(r'([A-Z]{5})=(\d+)', body)
        pairs += re.findall(r'([A-Z]{5}) records (\d+)', body)
        pairs += re.findall(r'\| ([A-Z]{5}) \| (\d+) \|', body)
        pairs += [(b, a) for a, b in re.findall(r'(\d+) belongs to ([A-Z]{5})', body)]
        if len(pairs) != 3:
            raise ValueError('Expected three table records')
        value = dict(pairs)[subject]
    elif m := re.fullmatch(r'Input "([a-zA-Z]+)"\. (.+)\.', p):
        text, op = m.groups()
        operations = {'Convert every letter to uppercase': str.upper, 'Convert every letter to lowercase': str.lower,
                      'Reverse all characters; preserve their case': lambda x: x[::-1],
                      'Keep only the final three characters; preserve their order and case': lambda x: x[-3:]}
        value = operations[op](text)
    elif m := re.fullmatch(r'(.+)\. Return the status at the (latest|earliest|middle revision)\.', p):
        revisions = sorted((int(a), b) for a, b in re.findall(r'revision (\d+): (\w+)', m[1]))
        if len(revisions) != 3:
            raise ValueError('Expected three temporal records')
        value = revisions[{'latest': -1, 'earliest': 0, 'middle revision': 1}[m[2]]][1]
    elif m := re.fullmatch(r'One unit A equals (\d+) units B\. Convert (\d+) units A to units B\.', p):
        value = int(m[1]) * int(m[2])
    else:
        raise ValueError('Unknown retention prompt: ' + p)
    return compact(dict(v=str(value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    inputs, checks, failures = {}, collections.Counter(), []

    def read(rel):
        path = args.package / rel
        raw = path.read_bytes()
        inputs[str(path.resolve())] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        return json.loads(raw)

    def check(cond, label, row=None):
        checks[label] += 1
        if not cond:
            failures.append(dict(check=label, row=row))

    root = pathlib.Path('corpora/distant_evidence_v2')
    manifest, design = read(root/'manifest.json'), read(root/'design.json')
    read(root/'review.json')
    config = read('configs/titan_transition.v2.memory_efficient.json')
    geom = read('evidence/qualification_20261003/corpus_v2_frozen_token_geometry.json')
    geometry = {r['id']: r for r in geom['records']}
    corpus = {}
    for split in ['train', 'dev']:
        path = args.package/root/f'{split}.jsonl'
        raw = path.read_bytes()
        inputs[str(path.resolve())] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        corpus[split] = [json.loads(line) for line in raw.decode().splitlines()]
        check(inputs[str(path.resolve())]['sha256'] == manifest['splits'][split]['sha256'], 'corpus_hash_matches_admission', split)
        check(manifest['splits'][split]['protected'] is False and manifest['splits'][split]['role'] == split, 'unprotected_explicit_split', split)
        check(inputs[str(path.resolve())]['sha256'] == geom['corpus_files_sha256'][f'{split}.jsonl'], 'historical_token_geometry_corpus_hash', split)
    for name in ['design', 'review']:
        check(inputs[str((args.package/root/f'{name}.json').resolve())]['sha256'] == manifest[name]['sha256'], 'metadata_hash_matches_admission', name)
    check(inputs[str((args.package/root/'design.json').resolve())]['sha256'] == geom['corpus_files_sha256']['design.json'], 'historical_token_geometry_design_hash')
    check(manifest['reviewed_for_sbhm'] is True, 'corpus_admitted')
    rows = {r['id']: r for split in corpus.values() for r in split}
    check(len(rows) == 768 and len(corpus['train']) == 256 and len(corpus['dev']) == 512, 'corpus_counts')
    check(set(rows) == set(design['rows']) == set(geometry), 'row_metadata_geometry_exact_coverage')
    for field in ['id', 'family_id', 'prompt']:
        normalize = lambda x: ' '.join(x.casefold().split())
        check(not {normalize(r[field]) for r in corpus['train']} & {normalize(r[field]) for r in corpus['dev']}, 'train_dev_separation_'+field)
    for field in ['world_id', 'template_id']:
        check(not {m[field] for m in design['rows'].values() if m['split']=='train'} & {m[field] for m in design['rows'].values() if m['split']=='dev'}, 'train_dev_separation_'+field)
    cf, position, worlds = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    oracle = {}
    for id, r in rows.items():
        m, g = design['rows'][id], geometry[id]
        check(digest(r) == m['row_sha256'], 'row_digest', id)
        check(r['family_id'] == m['family_id'], 'family_metadata', id)
        check(m['token_geometry']['prompt_tokens'] == g['prompt_tokens'] and m['token_geometry']['target_tokens']+1 == g['answer_plus_eos_tokens'] and m['token_geometry']['train_tokens'] == g['training_input_tokens'], 'token_geometry_metadata_consistency', id)
        check(g['training_input_tokens'] == g['prompt_tokens']+g['answer_plus_eos_tokens'], 'answer_eos_input_arithmetic', id)
        try:
            if m['role'] == 'retention':
                result = retention_oracle(r['prompt'])
            else:
                result, facts, subject, fields = evidence_oracle(r['prompt'])
                check(facts == m['events'], 'facts_parsed_from_prompt_match_metadata', id)
                check(subject == m['query']['subject'], 'subject_parsed_from_query_match_metadata', id)
                for block in ['evidence', 'filler', 'query']:
                    start, end = m['block_spans'][block]
                    check(r['prompt'][start:end] == m[block+'_block_text' if block != 'query' else 'query_text'], 'raw_prompt_block_span', id)
                for span in m['evidence_spans']:
                    check(r['prompt'][span['start_char']:span['end_char']] == span['record_text'], 'raw_record_span', id)
                for key in ['evidence_to_query_gap_tokens', 'earliest_required_to_answer_tokens', 'latest_required_to_answer_tokens']:
                    check(g[key] == m['token_geometry'][key], 'recorded_distance_consistency', id)
                check(g['evidence_to_query_gap_tokens'] > 128 if m['role']=='distant_heldout' else g['earliest_required_to_answer_tokens'] <= 128, 'registered_distance_contract', id)
                cf[(m['split'],m['group_id'],m['role'])].append(id)
                position[(m['split'],m['position_pair_id'])].append(id)
                worlds[(m['split'],m['group_id'])].append(id)
            oracle[id] = result
            check(result == r['target'], 'independent_actual_prompt_target_replay', id)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            failures.append(dict(check='oracle_or_span_exception',row=id,error=str(e)))
    for key, ids in worlds.items():
        check(len(ids)==4 and {(design['rows'][id]['role'],design['rows'][id]['counterfactual_id']) for id in ids} == {('near_control',0),('near_control',1),('distant_heldout',0),('distant_heldout',1)}, 'world_has_four_dependent_variants', str(key))
    for key, ids in position.items():
        check(len(ids)==2, 'position_pair_count', str(key))
        a,b = [design['rows'][id] for id in ids]
        check({a['role'],b['role']} == {'near_control','distant_heldout'}, 'position_pair_roles', str(key))
        for field in ['events','evidence_block_text','filler_block_text','query_text','query','template_id','counterfactual_id']:
            check(a[field] == b[field], 'position_pair_same_'+field, str(key))
        check(rows[ids[0]]['target'] == rows[ids[1]]['target'] and a['token_geometry']['prompt_tokens']==b['token_geometry']['prompt_tokens'], 'position_pair_target_token_count', str(key))
        blocks = [rows[id]['prompt'].split('\n---\n') for id in ids]
        check(blocks[0][0] == blocks[1][0] and blocks[0][1:3] == blocks[1][1:3][::-1] and blocks[0][3] == blocks[1][3], 'position_pair_only_block_order_changes', str(key))
    for key, ids in cf.items():
        check(len(ids)==2, 'counterfactual_pair_count', str(key))
        a,b = [design['rows'][id] for id in ids]
        check({a['counterfactual_id'],b['counterfactual_id']} == {0,1}, 'counterfactual_pair_ids', str(key))
        for field in ['filler_block_text','query_text','query','template_id']:
            check(a[field] == b[field], 'counterfactual_same_'+field, str(key))
        changed = [(x,y) for x,y in zip(a['events'],b['events']) if x!=y]
        expected_changes = 2 if a['category']=='multiple_facts' else 1
        check(len(a['events'])==len(b['events']) and len(changed)==expected_changes, 'counterfactual_expected_changed_record_count', str(key))
        check(rows[ids[0]]['target'] != rows[ids[1]]['target'], 'counterfactual_truth_changes', str(key))
        for x,y in changed:
            field = 'subject' if a['category']=='missing' else 'value'
            check({k:v for k,v in x.items() if k!=field} == {k:v for k,v in y.items() if k!=field}, 'counterfactual_changes_only_expected_field', str(key))
        if a['category']=='missing':
            check(sorted(oracle[id] for id in ids).count('{"v":null}')==1, 'missing_pair_one_present_one_absent', str(key))
    groups = {group:{r['id'] for r in corpus['dev'] if r['family_id'] in families} for group,families in manifest['evaluation_groups'].items()}
    check(sum(map(len,groups.values()))==len(set.union(*groups.values()))==512, 'DEV_groups_disjoint_cover')
    for id in groups['distant_heldout'] | groups['near_control'] | groups['retention']:
        check(id in groups[design['rows'][id]['role']], 'DEV_group_matches_role', id)
    run = pathlib.Path('runs/titan_transition_v2_20261004_memory_efficient')
    evaluation = {arm:read(run/f'{arm}-evaluation/receipt.json') for arm in 'ABCD'}
    predictions, correct, normalized = {}, {}, {}
    for arm,e in evaluation.items():
        predictions[arm] = {p['id']:p for p in e['predictions']}
        check(len(e['predictions'])==len(predictions[arm])==512 and set(predictions[arm])=={r['id'] for r in corpus['dev']}, 'DEV_prediction_coverage', arm)
        check([p['id'] for p in e['predictions']]==[r['id'] for r in corpus['dev']], 'DEV_corpus_order', arm)
        correct[arm], normalized[arm] = set(),set()
        for id,p in predictions[arm].items():
            check(p['target']==rows[id]['target'] and p['family_id']==rows[id]['family_id'], 'prediction_target_family_identity', arm+':'+id)
            strict, normal = p['prediction']==rows[id]['target'], norm(p['prediction'])==norm(rows[id]['target'])
            check(p['strict_exact_match'] is strict and p['normalized_exact_match'] is normal, 'prediction_score_flags_recomputed', arm+':'+id)
            check(p['prompt_tokens']==geometry[id]['prompt_tokens'], 'DEV_prompt_length_geometry', arm+':'+id)
            if strict: correct[arm].add(id)
            if normal: normalized[arm].add(id)
        check(e['strict_exact_match']==len(correct[arm])/512 and e['normalized_exact_match']==len(normalized[arm])/512, 'DEV_aggregate_recomputed', arm)
        check(correct[arm]==normalized[arm], 'strict_normalized_same_cases', arm)
    scored = {}
    for group,ids in groups.items():
        a_correct = ids & correct['A']
        scored[group] = dict(rows=len(ids), counts={arm:len(ids & correct[arm]) for arm in 'ABCD'},
            source_correct_cases=len(a_correct), retention={arm:dict(retained=len(a_correct & correct[arm]),damaged=len(a_correct-correct[arm]),repaired=len(ids & (correct[arm]-correct['A']))) for arm in 'BCD'},
            C_only=sorted(ids & (correct['C']-correct['D'])), D_only=sorted(ids & (correct['D']-correct['C'])),
            D_minus_C_count=len(ids & correct['D'])-len(ids & correct['C']))
    row_detail = lambda id: dict(id=id,target=rows[id]['target'],predictions={arm:predictions[arm][id]['prediction'] for arm in 'ABCD'})
    retention = dict(source_correct_rows=sorted(groups['retention'] & correct['A']),
        damaged={arm:[row_detail(id) for id in sorted(groups['retention'] & (correct['A']-correct[arm]))] for arm in 'BCD'},
        repaired={arm:[row_detail(id) for id in sorted(groups['retention'] & (correct[arm]-correct['A']))] for arm in 'BCD'})
    distant_near_lost = {g:[row_detail(id) for id in scored[g]['C_only']] for g in ['distant_heldout','near_control']}
    missing = {}
    for role in ['distant_heldout','near_control']:
        ids = {id for id in groups[role] if design['rows'][id]['category']=='missing'}
        present = {id for id in ids if oracle[id]!='{"v":null}'}
        absent = ids-present
        missing[role] = dict(present_rows=len(present),absent_rows=len(absent),arms={arm:dict(present_correct=len(present & correct[arm]),absent_correct=len(absent & correct[arm]),null_predictions=sum(predictions[arm][id]['prediction']=='{"v":null}' for id in ids)) for arm in 'ABCD'})
    pairing_outcomes = {}
    for role in ['distant_heldout','near_control']:
        pairs = [ids for (split,world,position_role),ids in cf.items() if split=='dev' and position_role==role]
        pairing_outcomes[role] = {arm:dict(worlds=len(pairs),both_counterfactuals_correct=sum(set(ids)<=correct[arm] for ids in pairs),one_counterfactual_correct=sum(len(set(ids)&correct[arm])==1 for ids in pairs),neither_correct=sum(not set(ids)&correct[arm] for ids in pairs),identical_predictions_across_truth_change=sum(predictions[arm][ids[0]]['prediction']==predictions[arm][ids[1]]['prediction'] for ids in pairs)) for arm in 'ABCD'}
    stratification = {}
    for role in ['distant_heldout','near_control']:
        stratification[role] = {}
        for category in sorted({design['rows'][id]['category'] for id in groups[role]}):
            ids = {id for id in groups[role] if design['rows'][id]['category']==category}
            stratification[role][category] = dict(rows=len(ids),counts={arm:len(ids & correct[arm]) for arm in 'ABCD'},D_only=sorted(ids & (correct['D']-correct['C'])))
    exposure = {}
    order = list(range(len(corpus['train'])))
    random.Random(config['seed']).shuffle(order)
    expected = [corpus['train'][order[i%len(order)]]['id'] for i in range(config['limits']['steps'])]
    check(expected==geom['training_exposure']['ordered_row_ids'], 'independent_shuffle_matches_frozen_geometry')
    check(set(collections.Counter(expected).values())=={2}, 'every_TRAIN_row_exactly_two_exposures')
    for stage in ['distill','recover']:
        traces = {}
        for arm in 'BCD':
            e = read(run/f'{arm}-{stage}/receipt.json')
            check(e['status']=='TRAINING_COMPLETED_NOT_EFFICACY' and e['stage']==stage and len(e['records'])==512, 'training_completed_512', arm+stage)
            check([r['step'] for r in e['records']]==list(range(512)) and [r['row_id'] for r in e['records']]==expected, 'training_step_row_order', arm+stage)
            traces[arm] = [(r['row_id'],r['input_tokens'],r['supervised_positions']) for r in e['records']]
            for r in e['records']:
                g = geometry[r['row_id']]
                check(r['input_tokens']==g['training_input_tokens'] and r['supervised_positions']==(g['training_input_tokens'] if stage=='distill' else g['answer_plus_eos_tokens']), 'training_length_label_count_contract', arm+stage+':'+str(r['step']))
                check(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']), 'training_record_finite', arm+stage+':'+str(r['step']))
            check(e['optimizer']=='AdamW(.9,.95), wd=0, eps=1e-8, clip=1', 'optimizer_recipe', arm+stage)
            check(e['binding']['config_sha256']==digest(config), 'training_canonical_config_digest', arm+stage)
            if stage=='recover':
                check(e['parent_record']==read(run/f'{arm}-distill/receipt.json'), 'recovery_parent_exact_distillation_record', arm)
                check(evaluation[arm]['adaptation_record']==e, 'evaluation_embeds_exact_recovery_record', arm)
            exposure[arm+'_'+stage] = dict(steps=512,input_tokens_total=sum(r['input_tokens'] for r in e['records']),supervised_positions_total=sum(r['supervised_positions'] for r in e['records']),input_tokens_min=min(r['input_tokens'] for r in e['records']),input_tokens_max=max(r['input_tokens'] for r in e['records']),unique_rows=len(set(expected)),trainable_parameters=e['trainable_parameters'],ordered_trace_sha256=digest(traces[arm]),optimizer=e['optimizer'],learning_rate=config['learning_rate'])
        check(traces['B']==traces['C']==traces['D'], 'all_arms_identical_recorded_exposure', stage)
    read(run/'comparison.json')
    for rel in ['sbhm/data.py','sbhm/pipeline.py','sbhm/native.py']:
        path=args.package/rel; raw=path.read_bytes(); inputs[str(path.resolve())]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    report = dict(schema='SBHM-001-independent-pairing-closeout/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='stdlib raw-prompt parser independently reconstructs all 768 targets; strict strings independently score 2048 predictions; no generator import, tokenizer execution or native execution',
        status='PASS' if not failures else 'FAIL',checks=dict(sorted(checks.items())),assertions=sum(checks.values()),failures=failures,input_hashes=inputs,
        script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),canonical_config_sha256=digest(config),
        corpus=dict(rows=len(rows),train=len(corpus['train']),dev=len(corpus['dev']),position_pairs=len(position),counterfactual_pairs=len(cf),worlds=dict(train=sum(k[0]=='train' for k in worlds),dev=sum(k[0]=='dev' for k in worlds)),DEV_position_pairs=sum(k[0]=='dev' for k in position),DEV_counterfactual_pairs=sum(k[0]=='dev' for k in cf),DEV_distant_gap_range=[min(geometry[id]['evidence_to_query_gap_tokens'] for id in groups['distant_heldout']),max(geometry[id]['evidence_to_query_gap_tokens'] for id in groups['distant_heldout'])],near_required_to_answer_max=max(geometry[id]['earliest_required_to_answer_tokens'] for id in groups['near_control'])),
        scores=scored,retention_row_details=retention,C_to_D_lost_row_details=distant_near_lost,missing_answerability=missing,counterfactual_world_outcomes=pairing_outcomes,task_stratification=stratification,training_exposure=exposure,
        exact_exposure_evidence_limit='No per-stage token-ID/label-ID hashes are stored in training traces. Matching immutable row/prompt/target hashes, tokenizer/config/code bindings, common deterministic encode recipe and exact input/supervision counts support matched exposure; this offline audit does not freshly tokenize or reconstruct model inputs.',
        historical_manifest_note='20261003 frozen geometry hashes the pre-admission manifest; review.json records that proposal hash. Current admitted manifest differs after review admission; exact train/dev/design bytes remain bound to the historical geometry.',
        limitations=['One source, one converted site, one seed 17 and posthoc DEV inspection; no confirmation evidence or uncertainty/promotion claim.','112 DEV evidence worlds have four dependent counterfactual/position variants; rows are not independent samples.','768 corpus checks include 256 TRAIN rows; only 512 rows are native DEV measurements.','64 synthetic retention rows establish narrow source-correct retention/damage only; repaired source-incorrect cases do not negate damage.','Distances are cross-checked from historical exact-token geometry metadata, not freshly tokenized in this offline inspection.','Other 27 layers retain full attention; pairing does not demonstrate an exclusive recurrent route.','Compact JSON strict and outer-whitespace/case normalized exact match do not parse semantic JSON correctness.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print(compact(dict(status=report['status'],assertions=report['assertions'],failures=len(failures),output=str(args.output),scores=scored,missing=missing)))
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
