from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import json, hashlib, math, importlib.util, sys

ROOT=Path('C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-mixture001-20260920')
CODE=Path('C:/Users/clclc/BusinessProject/ChatBubble-dose-param/research/binding_mixture_v1')
evidence={}
def need(value,message):
    if not value: raise ValueError(message)
def sha(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<32*2**20,'not small regular metadata/eval artifact: '+str(path))
    need(path.suffix.lower() not in ('.safetensors','.pt','.bin','.ced'),'weight access forbidden')
    with path.open('rb') as f: result=hashlib.file_digest(f,'sha256').hexdigest()
    evidence[str(path)]=result
    return result
def read(path):
    sha(path); return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rows(path):
    sha(path)
    result={}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        value=json.loads(line);need(value['id'] not in result,'duplicate evaluation row');result[value['id']]=value
    return result
def index(values):
    result={value['id']:value for value in values};need(len(result)==len(values),'duplicate prediction row');return result
def bound(binding,key):
    path=Path(binding[key+'_path']);need(sha(path)==binding[key+'_sha256'],'bound artifact changed: '+key);return path
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def pair(left,right):
    need(left.keys()==right.keys() and left and all(type(x) is bool for m in (left,right) for x in m.values()),'invalid paired rows')
    win=sum(right[k] and not left[k] for k in left);loss=sum(left[k] and not right[k] for k in left);nd=win+loss
    def p(w,l):return min(Fraction(1),Fraction(2*sum(math.comb(w+l,i) for i in range(min(w,l)+1)),2**(w+l)))
    exact=p(win,loss)
    minimum=next((n for n in range(nd%2,nd+1,2) if p((nd+n)//2,(nd-n)//2)<=Fraction(1,20)),None)
    minimum_holm=next((n for n in range(nd%2,nd+1,2) if p((nd+n)//2,(nd-n)//2)<=Fraction(1,40)),None)
    return {'n':len(left),'reference_correct':sum(left.values()),'candidate_correct':sum(right.values()),'wins':win,'losses':loss,'net':win-loss,'nd':nd,
            'exact_p':float(exact),'exact_p_fraction':str(exact),'minimum_significant_abs_net_at_observed_nd':minimum,
            'minimum_initial_Holm2_abs_net_at_observed_nd':minimum_holm,'nominally_significant':exact<=Fraction(1,20)}

result=read(ROOT/'RESULTS.json');criteria=read(ROOT/'CRITERIA.json');gate=read(ROOT/'BUILD_GATE.json');freeze=read(ROOT/'PRELAUNCH_BINDING.json')
need(result['status']=='COMPLETE_COMPARABLE' and sha(ROOT/'BUILD_GATE.json')==result['build_gate_sha256']==freeze['build_gate_sha256'],'mixed report/gate identity differs')
need(sha(ROOT/'CRITERIA.json')==gate['criteria_sha256']==result['criteria_sha256'],'criteria pin differs')
need(sha(CODE/'report.py')==result['reporter_sha256'],'frozen reporter changed')
for name,value in freeze['code_files_sha256'].items():need(sha(CODE/name)==value,'frozen executable changed')
reloc=read(ROOT/'SOURCE_RELOCATION.json');need(sha(ROOT/'SOURCE_RELOCATION.json')==gate['source_relocation_sha256']==result['source_relocation_sha256'],'relocation pin differs')
roots={Path(reloc['original_root']).resolve(),Path(reloc['archive_root']).resolve()}
for name,value in reloc['qualification_files_sha256'].items():
    for folder in roots:need(sha(folder/name)==value,'qualified source code changed')
aliases={}
for value in reloc['files']:
    a,b=Path(value['original_path']).resolve(),Path(value['archive_path']).resolve()
    need(sha(a)==sha(b)==value['sha256'],'code/input alias changed')
    if value['role']=='additional_input': aliases[str(a)]=aliases[str(b)]=str(a)
def normalized_inputs(binding):
    values=binding['additional_input_sha256_before'];need(values==binding['additional_input_sha256_after'],'additional inputs changed during execution')
    result={}
    for name,value in values.items():
        need(sha(name)==value,'additional input changed');key=aliases.get(str(Path(name).resolve()),str(Path(name).resolve()));need(key not in result,'alias collision');result[key]=value
    return result

control_root=Path(result['control_root']);control_report=read(control_root/'RESULTS.json')
need(control_report['status']=='COMPLETE_COMPARABLE' and sha(control_root/'RESULTS.json')==gate['control_results_sha256'],'authoritative control result changed')
loaded={};custody={};contexts=set()
for role in ('mixed','lora'):
    loaded[role]={};custody[role]={};weights=set();training=[]
    for leg,kind,n in (('dose','process',384),('select','process',384),('retention','retention',150),('long_context','long_context',256)):
        declared=result['custody'][role][leg];folder=Path(declared['directory']);bp=folder/'PROBE_WEIGHT_BINDING.json'
        binding=read(bp);need(sha(bp)==declared['binding_sha256'],'binding is not reported authoritative artifact')
        need(binding['status']=='COMPLETE' and binding['complete'] is True and type(binding['returncode']) is int and binding['returncode']==0 and binding['kind']==kind,'probe completion absent')
        rawpath=bound(binding,'raw_result');need(rawpath.resolve()==folder/'result.json' and sha(rawpath)==declared['result_sha256'],'raw result differs')
        raw=read(rawpath);original=bound(binding,'original_raw_result');need(sha(original)==sha(rawpath),'preserved raw copy differs')
        armpath=bound(binding,'arm_receipt');materialpath=bound(binding,'materialization');arm=read(armpath);material=read(materialpath)
        weight=binding['weight_sha256'];weights.add(weight)
        need(weight==declared['weight_sha256']==binding['weight_sha256_before']==binding['weight_sha256_after']==arm['arm_weight_sha256']==material['out_weight_sha256'],'recorded weight custody differs')
        need(arm['stored_dtype']==material['stored_dtype']=='float32' and arm['source_assets_sha256']==material['source_assets_sha256'],'arm precision/source custody differs')
        need(arm['checkpoint_manifest_sha256']==material['checkpoint_manifest_sha256'] and arm['checkpoint_binding_sha256']==material['checkpoint_binding_sha256'],'checkpoint receipt identity differs')
        training.append(binding['training_snapshot_before']);need(binding['training_snapshot_before']==binding['training_snapshot_after'],'training snapshot changed')
        model=raw[{'process':'model_dir','retention':'target','long_context':'model_directory'}[kind]]
        if kind=='retention':need(model.startswith('hf:'),'retention loader not hf');model=model[3:];need(raw['hf_candidate_identity']['weight_sha256']==weight,'retention loaded weight differs')
        need(Path(model).resolve()==Path(binding['arm_dir']).resolve(),'raw arm path differs')
        for stem in ('script','wrapper','supervisor_code','supervisor','terminal'):bound(binding,stem)
        supervisor=read(binding['supervisor_path']);terminal=read(binding['terminal_path'])
        need(supervisor['status']=='EXITED' and supervisor['returncode']==0 and terminal['status']=='COMPLETED' and terminal['complete'] is True and terminal['system_exit_code']==0 and terminal['failure'] is None,'successful observed terminal absent')
        need(terminal['script_unchanged'] is True and terminal['script_hash_error'] is None and terminal['script_sha256_after']==binding['script_sha256'] and Path(terminal['result']).resolve()==original.resolve(),'terminal result/script identity differs')
        need(binding['sidecar_sha256_before']==binding['sidecar_sha256_after']==binding['source_sidecar_sha256_before']==binding['source_sidecar_sha256_after'],'sidecar snapshots differ')
        for name,value in binding['sidecar_sha256_before'].items():need(sha(Path(binding['arm_dir'])/name)==value,'arm sidecar changed')
        bg=read(binding['build_gate_path']);need(sha(binding['build_gate_path'])==binding['build_gate_sha256'],'own training gate changed')
        need(binding['script_sha256']==bg['probe_script_sha256'][kind] and binding['wrapper_sha256']==bg['wrapper_sha256'],'probe code differs from own gate')
        views=rows(bound(binding,'views'));need(len(views)==n,'evaluation view count differs')
        inputs=normalized_inputs(binding)
        pre=read(folder/'PRECHECK.json');need(pre['status']=='PRECHECK_PASSED' and pre['kind']==kind and pre['arm_before']['weight_sha256']==weight,'precheck differs')
        need(Path(pre['command'][0]).resolve()==Path(reloc['python_path']).resolve() and Path(pre['cwd']).resolve() in roots,'unqualified interpreter/cwd')
        contexts.add((str(Path(pre['command'][0]).resolve()),'qualified_clone_pair'))
        field='reference_digest' if kind=='retention' else 'implementation_digest';need(raw[field]==binding['implementation_digest'],'scorer digest differs')
        if role=='lora':need(declared['binding_sha256']==control_report['custody']['lora'][leg]['binding_sha256'],'control attempt changed')
        loaded[role][leg]={'raw':raw,'binding':binding,'views':views,'inputs':inputs}
        custody[role][leg]={'binding_sha256':sha(bp),'result_sha256':sha(rawpath),'weight_sha256_from_successful_receipts':weight,'rows':n}
    need(len(weights)==1 and all(v==training[0] for v in training),'within-arm model/training identity changed')
need(len(contexts)==1,'execution contexts differ')
for leg in loaded['mixed']:
    a,b=loaded['mixed'][leg],loaded['lora'][leg];need(a['views'].keys()==b['views'].keys() and a['inputs']==b['inputs'],'paired row/additional inputs differ')
    for key in ('views_sha256','implementation_digest','script_sha256','wrapper_sha256'):need(a['binding'][key]==b['binding'][key],'evaluation binding identity differs')
    fields=('device','runtime','numerics','max_new_tokens','instrument') if leg in ('dose','select') else (('device','transformers','numerics','max_context','max_new_tokens') if leg=='long_context' else ('device','registered_bare_policy','frozen_reference_sizes','probe_rows'))
    for key in fields:need(a['raw'][key]==b['raw'][key],'paired execution setting differs '+leg+'/'+key)
need(loaded['mixed']['dose']['binding']['training_snapshot_before']!=loaded['lora']['dose']['binding']['training_snapshot_before'],'distinct training draws collapsed')

# Custody and scientific comparability pass before independent endpoint work.
scorer_path=Path(reloc['archive_root'])/'experiments/architecture_benchmark_001/retention.py'
spec=importlib.util.spec_from_file_location('independent_standard_retention_audit',scorer_path);scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
need('torch' not in sys.modules,'model runtime imported')
maps={role:{} for role in loaded};endpoints={role:{} for role in loaded};references={}
for role,legs in loaded.items():
    for leg in ('dose','select'):
        raw=legs[leg]['raw'];views=legs[leg]['views'];pred=index(raw['predictions']);need(pred.keys()==views.keys(),'acquisition prediction coverage differs')
        values={};strict={};terminated={}
        for key,row in pred.items():
            reference=views[key]['reference'];need(row['reference']==reference,'acquisition reference differs from frozen view')
            try:correct=canonical(json.loads(row['prediction']))==reference
            except (ValueError,TypeError):correct=False
            need(type(row['canonical']) is bool and row['canonical']==correct and row['strict']==(row['prediction']==reference),'acquisition recorded row score differs from independent exact JSON parsing')
            values[key]=correct;strict[key]=row['prediction']==reference;terminated[key]=row['terminated']
        need(sum(values.values())==raw['aggregate']['canonical_correct']==result['endpoints'][role][leg]['correct'],'acquisition count differs')
        maps[role][leg]=values;endpoints[role][leg]={'n':384,'canonical_correct':sum(values.values()),'strict_correct':sum(strict.values()),'terminated':sum(terminated.values())}
    r=legs['retention'];views=r['views'];binding=r['binding'];raw=r['raw'];saved=read(bound(binding,'retention_predictions'));pred=index(saved['predictions'])
    need(saved['scope']=='UNCHANGED_LEGACY_GENERATOR_RETURN' and saved['scorer_modified'] is False and len(pred)==150 and pred.keys()==views.keys(),'retention capture differs')
    additional={Path(name).name:Path(name) for name in binding['additional_input_sha256_before']}
    audit=read(additional['PROBE_LABEL_AUDIT.json']);frozen=read(additional['retention.json']);source=index(frozen['rows']);need(source.keys()==views.keys(),'source reference coverage differs')
    fixed=set(frozen['reference_ids_by_policy']['permissive']);need(len(fixed)==47,'registered fixed47 changed')
    if references:need(fixed==set(references['fixed47_ids']),'reference cohorts differ across arms')
    references['fixed47_ids']=sorted(fixed);references['strict_own41_ids']=sorted(frozen['reference_ids_by_policy']['strict'])
    policy_endpoints={}
    for policy in ('permissive','strict'):
        targets=scorer.build_overlay(audit,policy)['targets'];need(targets.keys()==views.keys(),'target coverage differs')
        def hit(text,eos,target):
            need(type(eos) is bool,'non-boolean EOS flag');scorer.validate_target(target)
            if not eos:return False
            want=target['minute_of_day'] if target['kind']=='clock' else scorer.parse_rational(str(target['value']))
            try:return scorer.parse_prediction(text,target)==want
            except scorer.ParseError:return False
        baseline={key:hit(source[key]['source']['raw'],source[key]['source']['terminated'],targets[key]) for key in views}
        values={key:hit(pred[key]['text'],pred[key]['terminated'],targets[key]) for key in views}
        own={key for key,value in baseline.items() if value};need(own==set(frozen['reference_ids_by_policy'][policy]) and len(own)==(47 if policy=='permissive' else 41),'typed source recount does not reproduce frozen own-policy cohort')
        panel=sum(values.values());retained=sum(values[key] for key in fixed);own_retained=sum(values[key] for key in own)
        declared=result['endpoints'][role]['retention']['policies'][policy]
        need((panel,retained,own_retained)==(declared['panel_correct'],declared['references_retained'],declared['own_policy_source_references_retained']),'retention reported endpoint differs')
        need(sorted(key for key in fixed if not values[key])==declared['registered_source_lost_ids'] and sorted(key for key in own if not values[key])==declared['own_policy_source_lost_ids'],'retention lost ID sets differ')
        need(raw['by_bare_policy'][policy]['arm_correct_total']==panel and raw['by_bare_policy'][policy]['lost_total']==len(own)-own_retained,'raw retention aggregate differs')
        classes={}
        for cls in sorted({row['reply_class'] for row in views.values()}):
            ids={key for key,row in views.items() if row['reply_class']==cls};need(len(ids)==50,'retention class geometry differs')
            count=sum(values[key] for key in ids);need(count==declared['per_class'][cls]['correct'],'retention class count differs');classes[cls]=count
        policy_endpoints[policy]={'panel_n':150,'panel_correct':panel,'fixed_reference_n':47,'fixed47_retained':retained,'own_reference_n':len(own),'own_reference_retained':own_retained,'per_class_correct':classes}
        maps[role]['panel_'+policy]=values;maps[role]['references_retained_'+policy]={key:values[key] for key in fixed};maps[role]['ratio_proportion_'+policy]={key:values[key] for key in views if views[key]['reply_class']=='ratio_proportion'}
    maps[role]['termination']={key:pred[key]['terminated'] for key in views};endpoints[role]['retention']={'policies':policy_endpoints,'terminated':sum(maps[role]['termination'].values())}
    lc=legs['long_context'];raw=lc['raw'];goldpath=next(Path(name) for name in lc['binding']['additional_input_sha256_before'] if Path(name).name=='all_answer_key.jsonl');gold=rows(goldpath)
    need(gold.keys()==lc['views'].keys()==raw['predictions'].keys()==raw['per_row_correct'].keys(),'LC coverage differs')
    def normalize(text):return ' '.join(str(text).strip().lower().replace('.',' ').split())
    total={};unknown={};unanswerable={};skipped=0
    for key,target in gold.items():
        pred=raw['predictions'][key];need(pred['id']==key and pred['status'] in ('ok','skipped_context_overflow'),'invalid LC prediction')
        text=pred['prediction'];first=text.splitlines()[0] if text else '';accepted=target.get('accepted_answers') or [target['answer']]
        total[key]=normalize(first) in {normalize(x) for x in accepted};need(type(raw['per_row_correct'][key]) is bool and total[key]==raw['per_row_correct'][key],'independent LC score differs')
        skipped+=pred['status']=='skipped_context_overflow'
        if target['variant']=='unanswerable':unanswerable[key]=total[key];unknown[key]='unknown' in normalize(text)
    counts={'n':len(total),'correct':sum(total.values()),'skipped':skipped,'unanswerable_n':len(unanswerable),'unanswerable_correct':sum(unanswerable.values()),'literal_unknown':sum(unknown.values())}
    need(counts==result['endpoints'][role]['long_context'] and len(total)==256 and len(unknown)==64,'LC aggregate differs')
    endpoints[role]['long_context']={**counts,'report_only':True};maps[role].update(long_context_total=total,long_context_unanswerable_correct=unanswerable,long_context_literal_unknown=unknown)

contrasts={key:pair(maps['lora'][key],maps['mixed'][key]) for key in maps['mixed']}
running=Fraction(0)
for rank,key in enumerate(sorted(('dose','select'),key=lambda key:Fraction(contrasts[key]['exact_p_fraction']))):
    running=max(running,min(Fraction(1),(2-rank)*Fraction(contrasts[key]['exact_p_fraction'])));v=contrasts[key]
    v.update(holm_adjusted_p=float(running),holm_family_size=2,cross_training_draw_abs_net_floor=20,cross_training_draw_floor_cleared=abs(v['net'])>=20,
             structural_claim_eligible=abs(v['net'])>=20 and running<=Fraction(1,20))
for key,v in contrasts.items():
    declared=result['acquisition_contrasts_mixed_minus_control'][key] if key in ('dose','select') else result['secondary_contrasts_mixed_minus_control'][key]
    for field,value in v.items():
        if field not in ('exact_p_fraction','minimum_initial_Holm2_abs_net_at_observed_nd'):need(declared[field]==value,'paired statistic differs '+key+'/'+field)

candidate=endpoints['mixed'];observed={'dose':candidate['dose']['canonical_correct'],'select':candidate['select']['canonical_correct'],
    'references_retained':candidate['retention']['policies']['permissive']['fixed47_retained'],'arithmetic_panel':candidate['retention']['policies']['permissive']['panel_correct'],
    'long_context_report_only':candidate['long_context']['correct'],'literal_unknown_report_only':candidate['long_context']['literal_unknown']}
predictions={}
for name,value in observed.items():
    low,high=criteria['prediction_bands'][name];outcome={'observed':value,'lower':low,'upper':high,'status':'PASS' if low<=value<=high else 'FAILED_PREDICTION'}
    need(outcome==result['predictions']['bands'][name],'standard band verdict differs');predictions[name]=outcome
primary=criteria['primary_engineering'];checks={'dose':observed['dose']>=primary['each_acquisition_at_least'],'select':observed['select']>=primary['each_acquisition_at_least'],
    'fixed47_retained':observed['references_retained']>=primary['fixed47_retained_at_least'],'arithmetic150':observed['arithmetic_panel']>=primary['arithmetic150_at_least']}
need(checks['fixed47_retained'] is False and result['predictions']['primary_engineering_conjunction']=='FAILED_PREDICTION','joint failure does not follow failed fixed47 criterion')
for bank in ('dose','select'):
    value=result['predictions']['acquisition_cost_engineering'][bank];need(value['net']==contrasts[bank]['net'] and value['minimum']==-20 and value['status']=='PASS' and value['establishes_preservation'] is False,'cost-bound interpretation differs')
need(result['predictions']['long_context_used_for_selection_or_confirmation'] is False,'LC enters selection')

cost=read(ROOT/'COST_REPORT.json');terminal=read(ROOT/'posttrain/TERMINAL.json');need(cost['status']=='COMPLETE_VERIFIED_COST_REPORT' and terminal['status']=='COMPLETE_FIVE_PROBES','complete cost/probe terminal absent')
need(not (ROOT/'ATTEMPT_MAP.json').exists(),'audit expected no inference retry')
attempts=[]
for leg,n,maximum in (('dose',384,128),('select',384,128),('retention',150,24),('long_context',256,24),('binding',384,24)):
    record=read(ROOT/'posttrain'/('mixed-'+leg+'.json'));need(record in terminal['attempts'] and record['returncode']==0 and record['reserved_rows']==n and record['reserved_generation_tokens']==n*maximum,'cost attempt differs')
    supervisor=read(Path(record['out'])/('supervisor.json' if leg=='binding' else 'worker/supervisor.json'))
    need(supervisor['status']=='EXITED' and supervisor['returncode']==0 and math.isclose(record['worker_seconds_charged'],supervisor['wall_seconds'],abs_tol=1e-9),'cost observed worker wall differs')
    attempts.append(record)
totals={'attempts':5,'successful_attempts':5,'failed_attempts':0,'worker_seconds_charged':sum(v['worker_seconds_charged'] for v in attempts),
    'rows_reserved':sum(v['reserved_rows'] for v in attempts),'generation_tokens_reserved':sum(v['reserved_generation_tokens'] for v in attempts)}
need(totals==cost['costs']['probes']['totals'] and totals['rows_reserved']==1558 and totals['generation_tokens_reserved']==117264,'costledger all-attempt totals differ')
need(math.isclose(totals['worker_seconds_charged'],terminal['probe_worker_seconds'],abs_tol=1e-9),'posttrain worker total differs')
need({Path(v['out']).name for v in attempts}=={p.name for p in (ROOT/'eval').iterdir()},'extra evalattempt omitted')
need(len(terminal['attempts'])==5,'duplicate costattempt')
need('torch' not in sys.modules and 'transformers' not in sys.modules,'model runtime imported')
for name,expected in list(evidence.items()):need(sha(name)==expected,'audited artifact changed during recount')
audit={'schema':1,'record':'BINDING_MIXTURE_STANDARD_NUMERICAL_AUDIT','status':'PASS_INDEPENDENT_STANDARD_RECOUNT',
 'created_utc':datetime.now(timezone.utc).isoformat(),'results_sha256':sha(ROOT/'RESULTS.json'),'audit_script_sha256':sha(Path(__file__)),
 'custody_gate':'PASS_RECORDED_EXECUTION_AND_CURRENT_SMALL_ARTIFACT_IDENTITIES_BEFORE_SCORING','custody':custody,
 'verification_scope':'All successful standard execution/binding/code/input/sidecar receipts revalidated; no live weight bytes read. Candidate/control training registrations deliberately differ.',
 'independent_methods':{'acquisition':'Reparse every prediction with JSON and compare sorted compact JSON to registered reference; strict/termination counted separately.',
   'retention':'Frozen hash-qualified typed grammar target construction and parse primitives; independently apply EOS plus exact rational/clock match and rederive both source cohorts.',
   'long_context':'Independent implementation of frozen first-line normalization/accepted-answer rule and legacy whole-output UNKNOWN substring; report-only.',
   'statistics':'Independent integer binomial-tail Fraction calculation, wins/losses/net, nominal observed-discordance thresholds; separate Holm2 only on dose and SELECT.'},
 'endpoints':endpoints,'retention_reference_cohorts':references,'paired_mixed_minus_lora':contrasts,'standard_prediction_bands':predictions,
 'primary_standard_checks':checks,'full_primary_conjunction_independently_forced_to_fail_by_standard_component':'fixed47 retained 37 < 40; binding cannot rescue this failure',
 'binding_numerical_audit_scope':'DELEGATED_SEPARATELY; no BIND row/world recount performed here',
 'cost_ledger_small_receipt_recheck':{'status':'PASS','five_probe_totals':totals,'measured_gpu_active_seconds':None,'time_semantics':'observed supervisor wall, not GPU-active time'},
 'implementation_bugs_found':[],'source_artifact_sha256':evidence,'model_execution':False,'weight_files_read':False,'sealed_raw_sources_read':False,
 'quality_claim':False,'selection_permitted':False,'release_permitted':False}
destination=ROOT/'STANDARD_NUMERICAL_AUDIT.json'
with destination.open('x',encoding='utf-8',newline='\n') as f:json.dump(audit,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':audit['status'],'path':str(destination),'standard_endpoints':endpoints,'probe_cost_totals':totals},indent=2))
