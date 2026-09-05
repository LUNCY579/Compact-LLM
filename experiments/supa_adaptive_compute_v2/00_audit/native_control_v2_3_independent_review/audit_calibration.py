"""Independent stdlib-only audit of fixed native control calibration evidence."""
import collections
import datetime
import decimal
import hashlib
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROGRAMME = HERE.parents[1]
REPO = PROGRAMME.parents[1]
EVIDENCE = PROGRAMME / 'native_armr/evidence/00_control_calibration/20260905T223621_219545Z_a6c46c00'
REGISTRATION_ID = 'cf1440a9bbf3344bcc0cca316a1d5f9cb172a448b06bf1696b27646fc673fe57'
FORMULA = 'sqrt(mean_scenario_mean_ordinary_token_relative_A3_energy / mean_scenario_mean_ordinary_token_relative_M_energy)'
PROMPT_FIELDS = ('example_id', 'scenario_id', 'family', 'template_id', 'prompt', 'prompt_sha256', 'alignment_partition')
METRICS = ('native_energy', 'control_energy', 'hidden_rms', 'native_delta_rms', 'control_delta_rms')
checks = []

def check(name, value):
    checks.append({'check': name, 'pass': bool(value)})

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(65536):
            value.update(block)
    return value.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def relative(path):
    return path.relative_to(REPO).as_posix()

def time_record(path):
    stat = path.stat()
    utc = lambda ns: datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc).isoformat()
    return {'path': relative(path), 'last_write_ns': stat.st_mtime_ns,
        'last_write_utc': utc(stat.st_mtime_ns),
        'windows_creation_ns': stat.st_ctime_ns,
        'windows_creation_utc': utc(stat.st_ctime_ns)}

report = read(EVIDENCE / 'report.json')
manifest = read(EVIDENCE / 'run_manifest.json')
started = read(EVIDENCE / 'started_manifest.json')
fit_scale = read(EVIDENCE / 'fit_scale.json')
regpath = REPO / report['registration_path']
registration = read(regpath)
contract = registration['contract']
config = read(REPO / contract['calibration_config_path'])
check('registration frozen canonical digest', registration['status'] == 'FROZEN' and canonical(contract) == registration['registration_sha256'] == report['registration_sha256'] == REGISTRATION_ID)
check('report hash bound to complete terminal receipt', sha(EVIDENCE / 'report.json') == manifest['report_sha256'] and manifest['status'] == report['status'] == 'COMPLETE' and manifest['failure_receipt'] is None)
check('contract bound to started/final manifests', all(started.get(key) == manifest.get(key) == value for key, value in contract.items()))
check('report provenance equals contract', all(report[key] == contract[key] for key in ('source_files', 'calibration_config_path', 'calibration_config_sha256', 'dataset_manifest_path', 'dataset_manifest_sha256', 'calibration_data_path', 'calibration_data_sha256')))
check('config content and byte hash', config == contract['config'] and sha(REPO / contract['calibration_config_path']) == contract['calibration_config_sha256'])
check('fixed fit-only unclipped formula', report['formula'] == config['formula'] == fit_scale['formula'] == FORMULA and all(report[key] is False and config[key] is False for key in ('alpha_search', 'alpha_clipping', 'holdout_selects_alpha')))
check('no labels/generation/training declared', report['label_free'] is True and report['answer_labels_used'] is False and report['generated_tokens'] == report['trainable_parameters'] == report['backbone_gradients'] == 0)
check('CPU reference runtime', all(report[key] == config[key] for key in ('device', 'dtype', 'threads', 'use_cache')) and report['runtime_identity'] == {'attention':'eager','device':'cpu','dtype':'float32','seed':17,'threads':4,'torch':'2.12.1+cu126','transformers':'5.14.1','use_cache':False})
bound = dict(contract['source_files'])
bound.update({contract[key + '_path']: contract[key + '_sha256'] for key in ('calibration_config', 'dataset_manifest', 'calibration_data')})
source_snapshot = {}
for name, expected in bound.items():
    observed = sha(REPO / name)
    source_snapshot[name] = {'registered_sha256': expected, 'observed_sha256': observed}
    check('registered file bytes ' + name, observed == expected)

bank_manifest = read(REPO / contract['dataset_manifest_path'])
manifest_payload = dict(bank_manifest)
manifest_digest = manifest_payload.pop('manifest_sha256')
check('dataset internal digest and boundary', canonical(manifest_payload) == manifest_digest and bank_manifest['track'] == 'native_armr' and bank_manifest['status'] == 'FROZEN_UNSEALED' and bank_manifest['sealed_access'] is False)
info = bank_manifest['datasets']['calibration']
check('calibration JSONL bytes', sha(REPO / contract['calibration_data_path']) == info['sha256'] == contract['calibration_data_sha256'])
prompts = []
with (REPO / contract['calibration_data_path']).open(encoding='utf-8') as handle:
    for line in handle:
        if not line.strip():
            continue
        raw = json.loads(line)
        check('own original calibration ' + raw['example_id'], (raw['track'], raw['split'], raw['variant']) == ('native_armr', 'calibration', 'original'))
        row = {key: raw[key] for key in PROMPT_FIELDS}
        del raw
        check('prompt and frozen partition ' + row['example_id'], canonical(row['prompt']) == row['prompt_sha256'] and row['alignment_partition'] == ('holdout' if int(row['scenario_id'].rsplit('_',1)[1]) % 4 == 3 else 'fit'))
        prompts.append(row)
prompts.sort(key=lambda row: row['example_id'])
check('128 unique scenarios and examples', len(prompts) == len({x['example_id'] for x in prompts}) == len({x['scenario_id'] for x in prompts}) == 128)
check('family/template inventory', dict(collections.Counter(x['family'] for x in prompts)) == info['families'] and dict(collections.Counter(x['template_id'] for x in prompts)) == info['templates'])
check('complete case filename inventory', set(report['case_files']) == {canonical(x['example_id']) + '.json' for x in prompts} == {x.name for x in (EVIDENCE/'cases').glob('*.json')})
by_id = {x['example_id']:x for x in prompts}
cases = []
case_index = []
stratum_discrepancies = []
special_ids = set(report['special_control_token_ids'])
check('special-token inventory sorted unique', report['special_control_token_ids'] == sorted(special_ids) and len(special_ids) == 26)
for name, expected in report['case_files'].items():
    path = EVIDENCE / 'cases' / name
    case = read(path)
    row = by_id[case['example_id']]
    check('case hash and identity ' + name, sha(path) == expected and all(case[k] == row[k] for k in PROMPT_FIELDS if k != 'prompt'))
    check('case contains no response/answer/scoring fields ' + name, not set(case).intersection({'answer','scorer','score','correct','response','gold_nll'}))
    tokens = case['token_ids']
    n = len(tokens)
    ordinary = sum(token not in special_ids for token in tokens)
    check('complete prompt and ordinary token coverage ' + name, 0 < n <= config['max_input_tokens'] and ordinary > 0 and case['full_prompt_tokens'] == case['retained_prompt_tokens'] == n and case['metrics']['all']['token_count'] == n and case['metrics']['ordinary']['token_count'] == ordinary and case['metrics']['final_prompt']['token_count'] == 1)
    special = case['metrics']['special_control']
    check('special stratum token coverage ' + name, special is None if n == ordinary else special['token_count'] == n-ordinary)
    for stratum, values in case['metrics'].items():
        if values is not None:
            check('finite nonnegative metrics '+name+' '+stratum, all(type(values[k]) in (int,float) and math.isfinite(values[k]) and values[k] >= 0 for k in METRICS))
    for key in ('embedding_rows_sha256','hidden_sha256','native_delta_sha256','control_delta_sha256','original_positions_sha256','original_mask_sha256'):
        check('tensor digest syntax '+name+' '+key, re.fullmatch('[0-9a-f]{64}',case[key]) is not None)
    check('RoPE digest pair '+name, len(case['original_rope_sha256']) == 2 and all(re.fullmatch('[0-9a-f]{64}',x) for x in case['original_rope_sha256']))
    # Independently check scalar consistency of all/ordinary/special strata;
    # this does not recover per-token tensor arithmetic from hashes.
    for key in METRICS:
        left = case['metrics']['all'][key]
        right = (ordinary * (case['metrics']['ordinary'][key] ** (2 if key.endswith('_rms') else 1)) + (n-ordinary) * (special[key] ** (2 if key.endswith('_rms') else 1) if special else 0)) / n
        if key.endswith('_rms'):
            right = math.sqrt(right)
        delta = abs(left-right)
        stratum_discrepancies.append(delta)
        check('stratum scalar reconciliation '+name+' '+key, math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-14))
    cases.append(case)
    case_index.append({'example_id':case['example_id'],'partition':case['alignment_partition'],'sha256':expected,**time_record(path)})

aggregates = {}
for partition, expected_count in (('fit',96),('holdout',32)):
    selected = [x for x in cases if x['alignment_partition']==partition]
    recomputed = {'scenario_ids':sorted(x['scenario_id'] for x in selected),
        'example_ids':sorted(x['example_id'] for x in selected), 'scenarios':len(selected),
        'ordinary_tokens':sum(x['metrics']['ordinary']['token_count'] for x in selected)}
    for key in METRICS:
        recomputed[key] = math.fsum(x['metrics']['ordinary'][key] for x in selected) / len(selected)
    aggregates[partition] = recomputed
    check(partition+' exact count and preregistered IDs',len(selected)==expected_count and recomputed['example_ids']==contract[partition+'_example_ids'])
    check(partition+' exact independently recomputed aggregate',recomputed=={k:v for k,v in report[partition].items() if k!='scaled_control_energy'})
check('fit holdout scenarios disjoint',not set(aggregates['fit']['scenario_ids']).intersection(aggregates['holdout']['scenario_ids']))
alpha = math.sqrt(aggregates['fit']['native_energy']/aggregates['fit']['control_energy'])
check('exact independently recomputed alpha no clipping',math.isfinite(alpha) and alpha>0 and alpha==report['alpha']==fit_scale['alpha']==manifest['alpha'])
for partition in ('fit','holdout'):
    aggregates[partition]['scaled_control_energy'] = alpha**2 * aggregates[partition]['control_energy']
    check(partition+' exact scaled control energy',aggregates[partition]==report[partition])
check('fit scale hash and payload',sha(EVIDENCE/'fit_scale.json')==report['fit_scale_sha256'] and fit_scale=={'alpha':alpha,'formula':FORMULA,'fit':{k:v for k,v in aggregates['fit'].items() if k!='scaled_control_energy'}})
with decimal.localcontext() as ctx:
    ctx.prec=80
    values = {key:sum((decimal.Decimal.from_float(x['metrics']['ordinary'][key]) for x in cases if x['alignment_partition']=='fit'),decimal.Decimal(0))/96 for key in ('native_energy','control_energy')}
    alpha_decimal=(values['native_energy']/values['control_energy']).sqrt()
    high_precision = {'native_energy':str(values['native_energy']),'control_energy':str(values['control_energy']),'alpha':str(alpha_decimal),'alpha_rounded_float':float(alpha_decimal),'absolute_difference_from_report':abs(float(alpha_decimal)-report['alpha'])}

layers = report['layer_state_receipts']
check('all prefix and replay layers covered',[(x['stage'],x['layer']) for x in layers]==[('ordinary_prefix',i) for i in range(14)]+[('replay',12),('replay',13)])
state_receipts = layers + [report['rotary_state_receipt'],report['control_state_receipt']]
for index,item in enumerate(state_receipts):
    check('state/freeze/buffers unchanged '+str(index),item['state_sha256_before']==item['state_sha256_after'] and re.fullmatch('[0-9a-f]{64}',item['state_sha256_before']) is not None and item['buffer_sha256_before']==item['buffer_sha256_after'] and item['trainable_parameters']==item['backbone_gradients']==0 and all(re.fullmatch('[0-9a-f]{64}',x) for x in item['buffer_sha256_before'].values()))
for index in (12,13):
    check('replay layer equals prefix '+str(index),layers[index]['state_sha256_before']==layers[index+2]['state_sha256_before'] and layers[index]['buffer_sha256_before']==layers[index+2]['buffer_sha256_before'])
referencep=REPO/'experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json'
reference=read(referencep)
check('checkpoint and HF source identity equals preserved reference',report['checkpoint_files_before']==report['checkpoint_files_after']==reference['checkpoint_files_before'] and report['checkpoint_files_before']['model.safetensors']==config['checkpoint_sha256'] and report['qwen_implementation_sha256']==reference['qwen_implementation_sha256'] and report['model_config_sha256']==report['checkpoint_files_before']['config.json'])
qual=read(PROGRAMME/'native_armr/evidence/00_runtime_qualification/20260905T191958_868458Z_82f1162e/report.json')
check('original seeded control hash retained',report['control_state_sha256']==report['control_state_receipt']['state_sha256_before']==qual['cpu_identity']['operator_hashes']['AM']==qual['gpu_identity']['operator_hashes']['AM'] and report['control_state_receipt']['parameters']==31457280)
check('canonical post-run integrity reported',report['canonical_unchanged'] is True and manifest['canonical_unchanged'] is True)

scale_time=time_record(EVIDENCE/'fit_scale.json')
fit_times=[x for x in case_index if x['partition']=='fit']
holdout_times=[x for x in case_index if x['partition']=='holdout']
chronology={
    'fit_scale':scale_time,
    'last_fit_write_utc':max(fit_times,key=lambda x:x['last_write_ns'])['last_write_utc'],
    'first_holdout_creation_utc':min(holdout_times,key=lambda x:x['windows_creation_ns'])['windows_creation_utc'],
    'first_holdout_write_utc':min(holdout_times,key=lambda x:x['last_write_ns'])['last_write_utc'],
    'all_fit_writes_precede_fit_scale_creation':max(x['last_write_ns'] for x in fit_times)<=scale_time['windows_creation_ns'],
    'fit_scale_write_precedes_every_holdout_creation':scale_time['last_write_ns']<min(x['windows_creation_ns'] for x in holdout_times),
    'fit_scale_write_precedes_every_holdout_write':scale_time['last_write_ns']<min(x['last_write_ns'] for x in holdout_times),
    'qualification':'Filesystem times corroborate the registered source order. They are mutable and are not cryptographic timestamps; hashes bind contents, not historical chronology.',
}
for key in ('all_fit_writes_precede_fit_scale_creation','fit_scale_write_precedes_every_holdout_creation','fit_scale_write_precedes_every_holdout_write'):
    check('filesystem chronology '+key,chronology[key])

result={
    'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL',
    'checks_run':len(checks),'failed_checks':[x['check'] for x in checks if not x['pass']],
    'script_sha256':sha(pathlib.Path(__file__)),
    'evidence_directory':relative(EVIDENCE),'registration_id':REGISTRATION_ID,
    'registration_file':relative(regpath),'registration_sha256':sha(regpath),
    'report_sha256':sha(EVIDENCE/'report.json'),'run_manifest_sha256':sha(EVIDENCE/'run_manifest.json'),
    'started_manifest_sha256':sha(EVIDENCE/'started_manifest.json'),'fit_scale_sha256':sha(EVIDENCE/'fit_scale.json'),
    'recorded_execution_commit':manifest['commit_sha'],'git_commit_status_checked':False,
    'source_files_checked':len(contract['source_files']),'bound_files_checked':len(bound),
    'case_count':len(cases),'recomputed_aggregates':aggregates,'recomputed_alpha':alpha,
    'high_precision_crosscheck':high_precision,
    'max_stratum_scalar_reconciliation_error':max(stratum_discrepancies),
    'stratum_reconciliation_tolerance':{'relative':1e-12,'absolute':1e-14},
    'aggregate_and_alpha_comparisons':'exact Python float equality; no matching tolerance or clipping',
    'state_receipts_checked':len(state_receipts),'control_state_sha256':report['control_state_sha256'],
    'checkpoint_reference_file':relative(referencep),'checkpoint_reference_sha256':sha(referencep),
    'checkpoint_weight_bytes_reread':False,
    'chronology':chronology,'source_file_snapshot':source_snapshot,'case_file_snapshot':case_index,
    'models_loaded':0,'torch_imported':'torch' in sys.modules,'gpu_execution':False,
    'programme_helpers_imported':False,'architecture_accuracy_inspected':False,
    'limitations':[
        'Raw hidden/native-delta/control-delta tensors, embeddings, masks, positions and RoPE tensors are represented by hashes, not stored tensor bytes. Per-token energy construction cannot be independently rerun from these artifacts.',
        'Scalar aggregates and alpha were independently recomputed from recorded per-scenario measurements, without model rerun. Stratum checks reconcile reported scalars but do not recover the underlying tensors.',
        'Checkpoint/control/layer/buffer hashes and zero parameter-gradient inventories were reconciled from saved receipts. No weights were loaded and no live gradient or state inspection occurred. Separate buffer-gradient fields are not present; buffer preservation is evidenced by before/after digests.',
        'Filesystem times only corroborate ordering. The fit-scale receipt has no cryptographic timestamp.',
        'This is engineering calibration designed after an architecture failure. It is neither efficacy evidence nor GPU requalification and does not rehabilitate original v2.2 failures.'
    ],
}
with (HERE/'receipt.json').open('x',encoding='utf-8',newline='\n') as handle:
    json.dump(result,handle,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    handle.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_file_snapshot','case_file_snapshot','recomputed_aggregates')},indent=2))
print('AGGREGATES',json.dumps({k:{key:value for key,value in v.items() if key not in ('scenario_ids','example_ids')} for k,v in aggregates.items()},indent=2))
