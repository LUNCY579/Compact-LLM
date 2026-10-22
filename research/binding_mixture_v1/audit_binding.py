from collections import Counter,defaultdict
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,importlib.util,json,math,os,re,sys,time,unicodedata
from pathlib import Path
START=time.monotonic()
BASE=Path('C:/Users/clclc/BusinessProject');CODE=BASE/'ChatBubble-dose-param';REC=BASE/'rineya-dataset-retention-records'
ROOT=REC/'binding-mixture001-20260920';OLD=REC/'binding-dev-baseline001-20260920';OUT=ROOT/'eval/mixed-binding'
OBJ=Path('C:/Users/clclc/.codex/attachments/9253cba1-b908-4858-98f1-34c710d720e8/goal-objective.md')
CATS=('person_answerable','person_missing','person_vacant','role_answerable');NEG={'person_missing','person_vacant'};evidence={}
def need(x,m):
 if not x:raise ValueError(m)
def safe(p):
 p=Path(p).resolve();need(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'unsafe/missing evidence')
 need(p.suffix not in {'.safetensors','.pt','.bin'},'weight payload reads prohibited')
 need('Rineya_Long_Context_Demo_v1' not in p.parts,'long context/sealed source reads prohibited')
 return p
def sha(p):
 p=safe(p)
 with p.open('rb') as f:v=hashlib.file_digest(f,'sha256').hexdigest()
 evidence[str(p)]=v;return v
def read(p):return json.loads(safe(p).read_text(encoding='utf-8-sig'))
def jl(p):return [json.loads(x) for x in safe(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
for name in ['BINDING_NUMERICAL_AUDIT.json','BINDING_ERROR_ANALYSIS.md']:need(not (ROOT/name).exists(),'create-only audit output exists')
need('without forgetting' in OBJ.read_text(encoding='utf-8'),'objective not read')
prereg=(ROOT/'PREREG.md').read_text(encoding='utf-8');need('false UNKNOWN0–10/192' in prereg,'registered false-UNKNOWN band absent')
report=read(ROOT/'RESULTS.json');gate=read(ROOT/'BUILD_GATE.json');frozen=read(ROOT/'PRELAUNCH_BINDING.json');criteria=read(ROOT/'CRITERIA.json')
need(report['status']=='COMPLETE_COMPARABLE','complete report required')
for key,name in [('build_gate_sha256','BUILD_GATE.json'),('criteria_sha256','CRITERIA.json'),('prelaunch_binding_sha256','PRELAUNCH_BINDING.json'),('source_relocation_sha256','SOURCE_RELOCATION.json')]:need(report[key]==sha(ROOT/name),'result governing artifact differs')
for name,expected in frozen['code_files_sha256'].items():need(sha(CODE/'research/binding_mixture_v1'/name)==sha(ROOT/'code_before_training'/name)==expected,'frozen mixture code differs')
need(report['reporter_sha256']==sha(CODE/'research/binding_mixture_v1/report.py'),'reporter identity differs')
for name,expected in frozen['artifact_files_sha256'].items():need(sha(ROOT/name)==expected,'prelaunch artifact differs')
for key,name in [('plan_sha256','PLAN.json'),('grant_sha256','GRANT.json'),('registration_sha256','PREREG.md'),('session_envelope_sha256','SESSION_ENVELOPE.json'),('build_gate_sha256','BUILD_GATE.json')]:need(sha(ROOT/name)==frozen[key],'prelaunch direct pin differs')
need(sha(OLD/'BASELINE_SCORES.json')==gate['binding_baseline_scores_sha256'] and sha(OLD/'SCORE_PLAN.json')==gate['binding_score_plan_sha256'],'frozen baseline report/score plan differs')
baseline_audit=read(OLD/'NUMERICAL_AUDIT.json');need(sha(OLD/'NUMERICAL_AUDIT.json')=='2ff82c59b44c116ee1c89bcddcbb39e55f67d89b978d8dd97b6ffd18876aa5cc','prior independent baseline audit changed')
for p,expected in baseline_audit['evidence_files_sha256'].items():need(sha(p)==expected,'baseline audited evidence changed')
base=read(OLD/'BASELINE_PLAN.json');config=read(OLD/'SCORE_PLAN.json');old_report=read(OLD/'BASELINE_SCORES.json');old_grant=read(OLD/'BASELINE_GRANT.json')
for key in ['baseline_plan','grant','gold','generator']:need(sha(config[key+'_path'])==config[key+'_sha256'],'frozen score input differs')
need(sha(CODE/'research/binding_signal_eval_v1/score.py')==config['scorer_sha256'],'frozen scoring implementation differs')
sys.path.insert(0,str(CODE))
from research.binding_signal_eval_v1 import common as c
from research.binding_mixture_v1 import binding_common as bc
c.validate_plan(base);c.validate_grant(OLD/'BASELINE_PLAN.json',old_grant);code=c.code_snapshot(base)
inputs=c.validate_inputs(jl(base['input_path']));input_by_id={x['id']:x for x in inputs};gold_rows=jl(config['gold_path']);gold={g['id']:g for g in gold_rows}
need(sha(base['input_path'])==base['input_sha256']==old_report['input_sha256'] and len(gold)==len(gold_rows)==384 and set(gold)==set(input_by_id),'view/gold binding differs')
counts=defaultdict(Counter)
for key,g in gold.items():
 need(set(g)=={'id','parent_id','category','reference'} and g['parent_id']==input_by_id[key]['parent_id'] and g['category'] in CATS,'gold schema/world differs')
 need((g['reference']=='UNKNOWN')==(g['category'] in NEG),'gold category/answerability mismatch');counts[g['parent_id']][g['category']]+=1
need(len(counts)==32 and all(v==Counter({cat:3 for cat in CATS}) for v in counts.values()),'32-world geometry differs')
decoder=load('_binding_audit_frozen_policy_decoder',Path(config['generator_path']))
parsed={k:decoder.decode_policy_input(i['prompt'][1]['content']) for k,i in input_by_id.items()}
need(all(parsed[k]['answer']==gold[k]['reference'] for k in gold),'policy-only oracle and gold differ')
bundle={};custody={}
for role in ['source','lora']:
 out=OLD/role;b=read(out/'BASELINE_BINDING.json');t=read(out/'worker_terminal.json');s=read(out/'supervisor.json');g=read(out/'GENERATIONS.json')
 need(b['status']=='COMPLETE' and b['complete'] is True and b['controller_returncode']==0 and b['failure'] is None,'old baseline unsuccessful')
 c.verify_terminal(s,t)
 need(all(b[k]==v for k,v in c.verify_launch(out,OLD/'BASELINE_PLAN.json',OLD/'BASELINE_GRANT.json',base,role,s).items()),'old launch custody differs')
 for stem,name in [('supervisor','supervisor.json'),('worker_terminal','worker_terminal.json'),('generations','GENERATIONS.json')]:need(b[stem+'_sha256']==sha(out/name),'baseline raw receipt SHA differs')
 need(g['before']==g['after']==t['before']==t['after'] and g['before']['code']==code and g['before']['input_sha256']==base['input_sha256'] and g['before']['runtime_versions']==base['runtime_versions'],'baseline snapshot mismatch')
 need(g['before']['model']['weight_sha256']==base['models'][role]['weight_sha256'] and g['before']['model']['receipts_sha256']==base['models'][role]['receipt_sha256'],'baseline model receipt identity differs')
 need(report['binding_custody'][role]==old_report['custody'][role],'baseline control provenance changed in report')
 for key,name in [('binding_sha256','BASELINE_BINDING.json'),('generations_sha256','GENERATIONS.json'),('supervisor_sha256','supervisor.json')]:need(report['binding_custody'][role][key]==sha(out/name),'reported baseline custody differs')
 bundle[role]=g;custody[role]={'binding_sha256':sha(out/'BASELINE_BINDING.json'),'generation_sha256':sha(out/'GENERATIONS.json'),'weight_sha256_recorded':g['before']['model']['weight_sha256']}
# check_live=False forbids the helper from opening the candidate weight payload.
mixed,binding,candidate_base=bc.verify_binding(OUT,check_live=False)
need(candidate_base==base,'candidate generation plan differs')
bundle['mixed']=mixed;bind_grant=read(ROOT/'BIND_PROBE_GRANT.json');arm_root=ROOT/'arms/mixed';arm=read(arm_root/'ARM.json');materialization=read(arm_root/'MATERIALIZATION.json')
need(binding['training_before']==binding['training_after'],'candidate training snapshot changed')
before=mixed['before'];training=binding['training_before']
for key,name in [('arm_receipt_sha256','ARM.json'),('materialization_sha256','MATERIALIZATION.json')]:need(before[key]==training[key]==sha(arm_root/name),'candidate provenance receipt differs')
for role in ['source','lora','mixed']:
 model_root=arm_root if role=='mixed' else Path(base['models'][role]['path'])
 need({name:sha(model_root/name) for name in c.SIDECARS}==base['sidecars_sha256'],'three-arm tokenizer/config sidecars differ')
 if role!='source':
  for name in ['ARM.json','MATERIALIZATION.json']:
   receipt=read(model_root/name);need(receipt['source_weight_sha256']==base['models']['source']['weight_sha256'] and receipt['out_weight_sha256']==receipt['arm_weight_sha256']==(before['weight_sha256'] if role=='mixed' else base['models'][role]['weight_sha256']) and receipt['stored_dtype']=='float32','arm/source/precision identity differs')
need(before['weight_sha256']==training['weight_sha256']==bind_grant['arm_weight_sha256'],'candidate recorded weight binding differs')
need(training['sidecar_sha256']==training['source_sidecar_sha256']==before['sidecars_sha256']==base['sidecars_sha256'],'candidate sidecar snapshots differ')
td=training['training']
need(sha(td['data_contract_path'])==td['data_contract_sha256'] and sha(td['plan_path'])==td['plan_file_sha256'] and digest(read(td['plan_path']))==td['plan_sha256'],'candidate training contract differs')
for p,expected in td['data_files_sha256'].items():need(sha(p)==expected,'candidate training data file differs')
need(sha(ROOT/'TRAINING_AUDIT.json')=='12640055bf9d937ef66ef44a8453fca6f7dcc387652ceefabbb4f49cb72ae0e5','completed training audit changed')
for rec in [arm,materialization]:
 need(rec['source_assets_sha256']==gate['source_assets_sha256'] and rec['checkpoint_manifest_sha256']==sha(Path(rec['checkpoint'])/'manifest.json'),'candidate source/checkpoint identity differs')
 need(rec['checkpoint_binding_sha256']==digest(rec['checkpoint_binding']),'candidate checkpoint binding digest differs')
 for stem in ['training_result','training_supervisor','training_run_summary','training_launch','training_spawn','training_worker_claim','merge_qualification']:need(sha(rec[stem+'_path'])==rec[stem+'_sha256'],'candidate training provenance link differs')
 need(read(rec['merge_qualification_path'])['dense_tensor_sha256']==rec['merged_tensor_sha256'],'candidate canonical merge receipt differs')
for key,name in [('binding_sha256','BINDING.json'),('generations_sha256','GENERATIONS.json')]:need(report['binding_custody']['mixed'][key]==sha(OUT/name),'mixed report custody differs')
custody['mixed']={'binding_sha256':sha(OUT/'BINDING.json'),'generation_sha256':sha(OUT/'GENERATIONS.json'),'weight_sha256_recorded':before['weight_sha256'],'arm_sha256':sha(arm_root/'ARM.json'),'materialization_sha256':sha(arm_root/'MATERIALIZATION.json')}
tjson=read(Path(base['source_dir'])/'tokenizer.json');tcfg=read(Path(base['source_dir'])/'tokenizer_config.json')
need(tjson['decoder']['type']=='ByteLevel' and tjson['model']['type']=='BPE' and tcfg.get('clean_up_tokenization_spaces') is False,'unsupported stdlib token decoding')
vocab={v:k for k,v in tjson['model']['vocab'].items()};added={v['id']:v for v in tjson['added_tokens']};special={k for k,v in added.items() if v['special']}
bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
for b in range(256):
 if b not in bs:bs.append(b);cs.append(256+n);n+=1
byte_inverse={chr(u):b for b,u in zip(bs,cs)}
def decode_tokens(tokens):
 joined=''.join(added[t]['content'] if t in added else vocab[t] for t in tokens if t not in special)
 return bytes(byte_inverse[ch] for ch in joined).decode('utf-8',errors='replace')
for role,g in bundle.items():
 need(g['rows']==384 and g['worlds']==32 and g['numerics']==c.NUMERICS and g['eos_ids']==[151643,151645] and g['gold_file_read'] is False and g['extra_warmup_or_qualification_calls']==0,'generation numerical/EOS/gold scope differs')
 need(len(g['predictions'])==len({p['id'] for p in g['predictions']})==384 and {p['id'] for p in g['predictions']}==set(gold),'prediction coverage differs')
 for p in g['predictions']:
  tokens=p['generated_ids_including_eos'];i=input_by_id[p['id']]
  need(p['parent_id']==i['parent_id'] and p['input_tokens']==len(i['prompt_ids']) and 1<=len(tokens)==p['generated_tokens']==p['forward_calls']<=24 and all(type(t) is int and 0<=t<151936 for t in tokens),'row/token count differs')
  need(type(p['terminated']) is bool and p['terminated']==(tokens[-1] in g['eos_ids']) and not any(t in g['eos_ids'] for t in tokens[:-1]),'row EOS custody differs')
  need(p['processed_input_tokens']==len(i['prompt_ids'])+len(tokens)-1 and decode_tokens(tokens)==p['prediction'],'raw string/token accounting differs')
 need(g['actual_budget_charges']=={'rows':384,'generated_tokens_including_eos':sum(p['generated_tokens'] for p in g['predictions']),'forward_calls':sum(p['forward_calls'] for p in g['predictions']),'processed_input_tokens':sum(p['processed_input_tokens'] for p in g['predictions'])},'generation costs differ')

# Custody is established above before recomputing any correctness endpoint.
def norm(s,fold=True):
 x=' '.join(unicodedata.normalize('NFKC',s).split());return x.casefold() if fold else x
recomputed={};maps={};roster_maps={}
for role,gen in bundle.items():
 details={};by_world=defaultdict(lambda:defaultdict(list))
 for p in gen['predictions']:
  g=gold[p['id']];unknown=norm(p['prediction'])=='unknown';canonical_exact=norm(p['prediction'])==norm(g['reference']);strict_exact=norm(p['prediction'],False)==norm(g['reference'],False);negative=g['category'] in NEG
  roster=decoder.decode_policy_input(input_by_id[p['id']]['prompt'][1]['content'])['world']['people'];need(len(roster)==len(set(roster))==6,'roster parse differs')
  d={'id':p['id'],'parent_id':g['parent_id'],'category':g['category'],'canonical_text_exact':canonical_exact,'strict_text_exact':strict_exact,'terminated':p['terminated'],'canonical_complete_exact':canonical_exact and p['terminated'],'strict_complete_exact':strict_exact and p['terminated'],'canonical_exact_unknown':unknown,'false_unknown_on_positive':unknown and not negative,'exact_roster_person_on_unanswerable':negative and norm(p['prediction']) in {norm(x) for x in roster}}
  details[p['id']]=d;by_world[g['parent_id']][g['category']].append(d)
 fields=('canonical_text_exact','strict_text_exact','canonical_complete_exact','strict_complete_exact','terminated')
 categories={cat:{'rows':96,**{f:sum(r[f] for r in details.values() if r['category']==cat) for f in fields}} for cat in CATS}
 role_maps={policy:{cat:{} for cat in CATS} for policy in ('canonical_complete_exact','strict_complete_exact')}
 role_roster={}
 for policy in role_maps:
  role_maps[policy].update(unknown_combined_all6={},no_false_unknown_all6_positive_controls={})
  for world,cats in by_world.items():
   for cat in CATS: role_maps[policy][cat][world]=all(r[policy] for r in cats[cat])
   negative=cats['person_missing']+cats['person_vacant'];positive=cats['person_answerable']+cats['role_answerable']
   role_maps[policy]['unknown_combined_all6'][world]=all(r[policy] for r in negative)
   role_maps[policy]['no_false_unknown_all6_positive_controls'][world]=not any(r['false_unknown_on_positive'] for r in positive)
   role_roster[world]=any(r['exact_roster_person_on_unanswerable'] for r in negative)
 totals={f:sum(r[f] for r in details.values()) for f in fields}
 totals.update(rows=384,worlds=32,positive_control_rows=192,unanswerable_rows=192,false_unknown_on_positive=sum(r['false_unknown_on_positive'] for r in details.values()),exact_roster_person_on_unanswerable=sum(r['exact_roster_person_on_unanswerable'] for r in details.values()))
 recomputed[role]={'categories':categories,'roster_person_world_any_count':sum(role_roster.values()),'rows':details,'totals':totals,'world_counts':{p:{e:sum(v.values()) for e,v in endpoints.items()} for p,endpoints in role_maps.items()}}
 maps[role]=role_maps;roster_maps[role]=role_roster
need(recomputed==report['binding']['endpoints'],'independent row/category/world scores differ')
need(maps==report['binding']['world_maps'] and roster_maps==report['binding']['roster_person_world_any_maps'],'independent world maps differ')


# Independent exact statistics use integer combinations/Fractions, not the reporter's math.
def p_exact(w,l):
 n=w+l;return min(Fraction(1),Fraction(2*sum(math.comb(n,k) for k in range(min(w,l)+1)),2**n))
def minimum(nd,alpha=Fraction(1,20)):
 return next((net for net in range(nd%2,nd+1,2) if p_exact((nd+net)//2,(nd-net)//2)<=alpha),None)
def pair(ref,cand):
 need(len(ref)==len(cand)==32 and set(ref)==set(cand),'paired worlds differ')
 wins=sum(cand[k] and not ref[k] for k in ref);losses=sum(ref[k] and not cand[k] for k in ref);nd=wins+losses;p=p_exact(wins,losses)
 return {'n':32,'reference_correct':sum(ref.values()),'candidate_correct':sum(cand.values()),'wins':wins,'losses':losses,'net':wins-losses,'nd':nd,'exact_p':float(p),'alpha':.05,'nominally_significant':p<=Fraction(1,20),'minimum_significant_abs_net_at_observed_nd':minimum(nd),'threshold_structure':'fixed observed within-pair discordance; not prospective power','relationship':'cross_training_draw'}
contrasts={policy:{name:pair(maps['lora'][policy][name],values) for name,values in endpoints.items()} for policy,endpoints in maps['mixed'].items()}
for endpoints in contrasts.values():
 running=0.0
 for rank,cat in enumerate(sorted(NEG,key=lambda k:endpoints[k]['exact_p'])):
  running=max(running,min(1.,(2-rank)*endpoints[cat]['exact_p']))
  endpoints[cat].update(holm_adjusted_p=running,holm_family_size=2,holm_nominally_significant=running<=.05)
need(contrasts==report['binding']['world_contrasts_mixed_minus_control'],'independent paired exact/threshold/Holm statistics differ')
need(pair(roster_maps['lora'],roster_maps['mixed'])==report['binding']['roster_person_world_any_mixed_minus_control'],'roster paired contrast differs')
for role in ['source','lora']:
 need(recomputed[role]==old_report['endpoints'][role] and maps[role]==old_report['world_maps'][role] and roster_maps[role]==old_report['roster_person_world_any_maps'][role],'recomputed baseline differs')
observed={cat:recomputed['mixed']['categories'][cat]['canonical_complete_exact'] for cat in CATS}
observed['false_unknown_on_positive']=recomputed['mixed']['totals']['false_unknown_on_positive']
bands={k:{'observed':v,'lower':criteria['prediction_bands'][k][0],'upper':criteria['prediction_bands'][k][1],'status':'PASS' if criteria['prediction_bands'][k][0]<=v<=criteria['prediction_bands'][k][1] else 'FAILED_PREDICTION'} for k,v in observed.items()}
need(all(report['predictions']['bands'][k]==v for k,v in bands.items()),'registered binding band verdict differs')
rule=criteria['binding_efficacy'];need(rule['primary_categories']==['person_missing','person_vacant'] and rule['holm_family_size']==2 and rule['positive_net_required'] is True and rule['positive_control_and_false_unknown_bands_required'] is True,'binding efficacy rule differs')
controls_pass=all(bands[k]['status']=='PASS' for k in ['person_answerable','role_answerable','false_unknown_on_positive'])
primary=contrasts['canonical_complete_exact']
statistical_gain=any(primary[k]['net']>0 and primary[k]['holm_adjusted_p']<=rule['adjusted_exact_p_at_most'] for k in rule['primary_categories'])
verdict='ESTABLISHED_EXPLORATORY_DEV_GAIN' if controls_pass and statistical_gain else 'NOT_ESTABLISHED'
need(verdict==report['predictions']['binding_efficacy'] and controls_pass==report['predictions']['binding_positive_controls_pass'],'binding efficacy conjunction differs')
need(not controls_pass and not all(v['status']=='PASS' for v in bands.values()) and report['predictions']['primary_engineering_conjunction']=='FAILED_PREDICTION','failed BIND gate was rescued')
row_maps={role:value['rows'] for role,value in recomputed.items()}
false_ids=sorted(k for k,r in row_maps['mixed'].items() if r['false_unknown_on_positive'])
need(len(false_ids)==13,'false UNKNOWN inventory differs')
positive_ids=sorted(k for k,g in gold.items() if g['category'] not in NEG)
def metadata(key):
 text=input_by_id[key]['prompt'][1]['content'];lines=[s.strip() for s in text.splitlines() if s.strip()]
 info=parsed[key];grammar=decoder.INPUT_GRAMMARS[lines[0]];target=info['query']['target_role'];kind=info['query']['kind']
 questions=[pattern for qkind,pattern in grammar['questions'] if qkind==kind]
 indexes=[i for i,pattern in enumerate(questions,1) if re.fullmatch(pattern,lines[-1])]
 need(len(indexes)==1,'question pattern ambiguous')
 duty=[i for i,line in enumerate(lines) if re.fullmatch(grammar['duty'],line)];need(len(duty)==1,'duty fact ambiguous')
 target_positions=[]
 for i,line in enumerate(lines):
  match=re.fullmatch(grammar['occupied'],line)
  if match and (match[1] if grammar['role_first'] else match[2])==target:target_positions.append(i)
 condition=decoder.condition_of(info['world'],info['query'])
 need((condition=='occupied')==(len(target_positions)==1),'target assignment metadata differs')
 ordering='target_occupied_before_duty' if target_positions and target_positions[0]<duty[0] else ('duty_before_target_occupied' if target_positions else 'no_target_occupied_fact_required_for_role_question')
 evidence_type='two_hop_explicit_duty_and_explicit_occupied_person' if kind=='person' else 'direct_explicit_duty_to_role'
 return {'category':gold[key]['category'],'parent_id':gold[key]['parent_id'],'template_family':info['template_family'],'question_pattern':kind+'_'+str(indexes[0]),'question_regex':questions[indexes[0]-1],'target_assignment_condition':condition,'binding_evidence_type':evidence_type,'assignment_duty_order':ordering,'duty_line_index_zero_based':duty[0],'target_occupied_line_index_zero_based':target_positions[0] if target_positions else None,'prompt_tokens':len(input_by_id[key]['prompt_ids'])}
positive_meta={k:metadata(k) for k in positive_ids};preds={role:{p['id']:p for p in g['predictions']} for role,g in bundle.items()}
error_rows=[]
for key in false_ids:
 error_rows.append({'id':key,**positive_meta[key],'source_correct':row_maps['source'][key]['canonical_complete_exact'],'control_correct':row_maps['lora'][key]['canonical_complete_exact'],'source_false_unknown':row_maps['source'][key]['false_unknown_on_positive'],'control_false_unknown':row_maps['lora'][key]['false_unknown_on_positive'],'mixed_terminated':row_maps['mixed'][key]['terminated'],'source_prediction_sha256':hashlib.sha256(preds['source'][key]['prediction'].encode()).hexdigest(),'control_prediction_sha256':hashlib.sha256(preds['lora'][key]['prediction'].encode()).hexdigest(),'mixed_prediction_sha256':hashlib.sha256(preds['mixed'][key]['prediction'].encode()).hexdigest(),'gold_reference_sha256':hashlib.sha256(gold[key]['reference'].encode()).hexdigest()})
def slices(fields):
 groups=defaultdict(list)
 for key,meta in positive_meta.items():groups[tuple(meta[f] for f in fields)].append(key)
 result=[]
 for values,keys in sorted(groups.items()):
  errs=[k for k in keys if k in false_ids]
  result.append({**dict(zip(fields,values)),'positive_rows':len(keys),'false_unknown_rows':len(errs),'false_unknown_fraction':len(errs)/len(keys),'positive_worlds':len({gold[k]['parent_id'] for k in keys}),'false_unknown_worlds':len({gold[k]['parent_id'] for k in errs}),'mixed_correct':sum(row_maps['mixed'][k]['canonical_complete_exact'] for k in keys),'control_correct':sum(row_maps['lora'][k]['canonical_complete_exact'] for k in keys),'source_correct':sum(row_maps['source'][k]['canonical_complete_exact'] for k in keys)})
 return result
sliced={name:slices(fields) for name,fields in [('category',['category']),('template_family',['template_family']),('category_template',['category','template_family']),('category_question_pattern',['category','question_pattern']),('template_question',['template_family','question_pattern']),('assignment_duty_order',['assignment_duty_order']),('category_template_order',['category','template_family','assignment_duty_order']),('person_assignment_duty_order',['category','assignment_duty_order']),('binding_evidence',['binding_evidence_type']),('target_assignment_condition',['category','target_assignment_condition'])]}
def transitions(reference,keys):
 states=Counter((row_maps[reference][k]['canonical_complete_exact'],row_maps['mixed'][k]['canonical_complete_exact']) for k in keys)
 return {'rows':len(keys),'reference_correct':sum(row_maps[reference][k]['canonical_complete_exact'] for k in keys),'mixed_correct':sum(row_maps['mixed'][k]['canonical_complete_exact'] for k in keys),'correct_gains':states[False,True],'correct_losses':states[True,False],'both_correct':states[True,True],'both_wrong':states[False,False],'net_correct_gain':states[False,True]-states[True,False],'false_unknown_from_previously_correct':sum(k in false_ids and row_maps[reference][k]['canonical_complete_exact'] for k in keys),'false_unknown_from_previously_wrong':sum(k in false_ids and not row_maps[reference][k]['canonical_complete_exact'] for k in keys)}
transition_data={ref:{'all_positive_rows':transitions(ref,positive_ids),**{cat:transitions(ref,[k for k in positive_ids if gold[k]['category']==cat]) for cat in ['person_answerable','role_answerable']}} for ref in ['source','lora']}
error_worlds=Counter(gold[k]['parent_id'] for k in false_ids)
need(len(error_worlds)==32-sum(maps['mixed']['canonical_complete_exact']['no_false_unknown_all6_positive_controls'].values()),'false-UNKNOWN world count differs')
diagnosis={'false_unknown_rows':13,'false_unknown_worlds':len(error_worlds),'category_counts':dict(Counter(gold[k]['category'] for k in false_ids)),'world_error_multiplicity':dict(sorted(Counter(error_worlds.values()).items())),'error_worlds':dict(sorted(error_worlds.items())),'same_id_reference_correctness':{ref:{'formerly_correct':sum(row_maps[ref][k]['canonical_complete_exact'] for k in false_ids),'formerly_wrong':sum(not row_maps[ref][k]['canonical_complete_exact'] for k in false_ids),'formerly_false_unknown':sum(row_maps[ref][k]['false_unknown_on_positive'] for k in false_ids)} for ref in ['source','lora']},'positive_correctness_transitions_descriptive':transition_data,'slices_descriptive_no_tests':sliced,'error_rows':error_rows,'all_error_rows_terminated':all(row_maps['mixed'][k]['terminated'] for k in false_ids),'literal_error_outputs':dict(Counter(preds['mixed'][k]['prediction'] for k in false_ids)),'no_template_order_causal_claim':True,'no_threshold_or_endpoint_change':True}
need(not any(m in sys.modules for m in ['torch','transformers','tokenizers','safetensors']),'unexpected model/tokenizer import')


for p in [OBJ,ROOT/'RESULTS.json',ROOT/'PREREG.md',OLD/'BASELINE_SCORES.json',OLD/'NUMERICAL_AUDIT.json',ROOT/'TRAINING_AUDIT.json',ROOT/'BIND_PROBE_GRANT.json']:sha(p)
for dirname in [OLD/'source',OLD/'lora',OUT]:
 for name in ['GENERATIONS.json','supervisor.json','worker_terminal.json','launch.json','spawn.json','worker_claim.json','PRECHECK.json']:sha(dirname/name)
script_path=Path(__file__).resolve();script_sha=sha(script_path)
audit={'schema':1,'record':'BINDING_MIXTURE_INDEPENDENT_BINDING_NUMERICAL_AUDIT','created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_RECORDED_CUSTODY_AND_INDEPENDENT_NUMERICS','experiment':'BINDING-MIXTURE-001','audit_script_path':str(script_path),'audit_script_sha256':script_sha,'objective_path':str(OBJ),'preregistration_path':str(ROOT/'PREREG.md'),'result_path':str(ROOT/'RESULTS.json'),'result_sha256':sha(ROOT/'RESULTS.json'),'comparability':{'status':'PASS_RECORDED_CUSTODY','prior_baseline_independent_audit_sha256':sha(OLD/'NUMERICAL_AUDIT.json'),'same_384_inputs_gold_ids_and_32_worlds':True,'gold_free_worker_inputs_and_full_row_coverage':True,'same_code_numerics_runtime_and_eos_settings':True,'all_generations_decode_exactly_from_saved_token_ids':True,'all_three_live_sidecar_hashes_match':True,'candidate_source_training_checkpoint_and_merge_receipt_links_match':True,'successful_supervisors_and_exact_launch_identity_verified':True,'recorded_weight_hashes_agree_before_after_and_with_plan_grant_arm_materialization':True,'weight_payload_reads':False,'weight_scope':'Contemporaneous recorded hashes and current receipt/sidecar bytes verified; no weight payload was opened or rehashed.','custody':custody},'independent_recomputation':{'rows_per_role':384,'roles':['source','lora','mixed'],'row_flag_records_recomputed':1152,'all_reported_row_flags_and_totals_match':True,'all_world_maps_match':True,'all_exact_net_discordance_p_values_and_thresholds_match':True,'all_Holm2_values_match':True,'published_binding_verdicts_match':True,'token_decode_method':'stdlib ByteLevel inverse with frozen tokenizer vocabulary; no tokenizer or model library imported','math_method':'integer binomial combinations with Fraction; doubled smaller tail, capped at one'},'endpoints':recomputed,'world_maps':maps,'world_contrasts_mixed_minus_control':contrasts,'roster_person_world_any_maps':roster_maps,'roster_person_world_any_mixed_minus_control':pair(roster_maps['lora'],roster_maps['mixed']),'registered_binding_verdicts':{'bands':bands,'unknown_category_statistical_gain':statistical_gain,'positive_controls_and_false_unknown_pass':controls_pass,'binding_efficacy':verdict,'all_binding_row_bands_pass':all(v['status']=='PASS' for v in bands.values()),'false_unknown_band_failed':True,'primary_engineering_conjunction_cannot_pass_with_this_binding_failure':True,'scope':'BIND-only; other raw probes not rescored here.'},'structural_mde':{'paired_unit':'32 independent parent worlds; category success requires all three views','primary_Holm_family':['person_missing','person_vacant'],'other_endpoints':'nominal descriptive only','minimum_first_Holm2_unanimous_world_gain_from_zero_control':7,'seven_world_fraction':7/32,'observed_discordance_thresholds':{k:{'nd':primary[k]['nd'],'nominal_alpha_0_05_min_abs_net':minimum(primary[k]['nd']),'initial_Holm2_alpha_0_025_min_abs_net':minimum(primary[k]['nd'],Fraction(1,40))} for k in sorted(NEG)},'threshold_is_not_prospective_power':True,'cross_draw_20_view_acquisition_floor_applies_to_this_32_world_bank':False},'descriptive_false_unknown_diagnosis':diagnosis,'evidence_files_sha256':dict(sorted(evidence.items())),'audit_compute_wall_seconds':time.monotonic()-START,'model_calls':0,'weight_payloads_read':False,'sealed_or_long_context_sources_opened':False,'new_data_created':False,'code_edited':False,'existing_evidence_modified':False,'criteria_or_thresholds_changed':False,'quality_claim':False,'release_permitted':False,'limitations':['Exploratory developer-visible DEV; template/order error slices are post-hoc descriptive, not new hypothesis tests.','Template family, parent worlds and entities co-vary in this fixed bank; concentration does not isolate a causal wording or order effect.','Strong UNKNOWN-category statistical gains do not rescue the failed false-UNKNOWN positive-control criterion.','Current weight payload integrity is not rehashed; contemporaneous execution identity and provenance records supply the allowed custody scope.']}
md=[
'The independent audit reproduced all 1,152 row records across source, LoRA control, and mixed candidate, every 32-world map, and all exact paired/Holm statistics. Receipt, code, DEV input/gold, sidecar and recorded model identities passed before scoring. All saved token sequences decoded exactly to their recorded strings. No weight payload or sealed/long-context source was opened.',
'',
'The registered BIND efficacy verdict remains **NOT_ESTABLISHED**. UNKNOWN-category gains pass their statistical tests, but false UNKNOWN is **13/192 against the frozen 0–10 band**. That failed positive-control criterion cannot be rescued by another endpoint.',
'',
'| UNKNOWN endpoint | LoRA worlds | Mixed worlds | Net | nd | Exact p | Holm2 p | Nominal minimum significant net at observed nd |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for cat in ['person_missing','person_vacant']:
 v=primary[cat];md.append(f"| {cat} | {v['reference_correct']}/32 | {v['candidate_correct']}/32 | +{v['net']} | {v['nd']} | {v['exact_p']:.12g} | {v['holm_adjusted_p']:.12g} | {v['minimum_significant_abs_net_at_observed_nd']} |")
md+=['',
'Each category-world requires all three related views canonical-exact and EOS-terminated. There are 32 paired units, not 384 independent trials. The two UNKNOWN contrasts form the registered Holm2 family. Seven unanimous gains against a zero-world control can yield the first Holm2 rejection; this attainable threshold is not prospective power. The 20-view acquisition floor belongs to separate 384-parent banks.',
'',
'All 13 false UNKNOWNs are literal UNKNOWN with EOS on **occupied-person answerable queries**, across **nine worlds**: six worlds have one error, two have two, and one has all three views wrong. Every answer is supported by an explicit occupied-person assignment plus an explicit duty-to-role designation. These require a two-hop lookup through stated facts. No false UNKNOWN occurs on direct answerable-role queries, including roles with vacant or missing occupants.',
'',
'On those same 13 IDs, the LoRA control was correct on six and wrong on seven; source was correct on four and wrong on nine. Neither reference already emitted false UNKNOWN on any of them. Thus six are new correctness losses versus LoRA, while seven change the form of an existing error.',
'',
'| Positive slice | Source correct | LoRA correct | Mixed correct | Gains vs LoRA | Losses vs LoRA | Mixed false UNKNOWN |',
'|---|---:|---:|---:|---:|---:|---:|']
for cat,label in [('person_answerable','Occupied person /96'),('role_answerable','Answerable role /96'),('all_positive_rows','All positives /192')]:
 t=transition_data['lora'][cat];src=transition_data['source'][cat]
 md.append(f"| {label} | {src['reference_correct']} | {t['reference_correct']} | {t['mixed_correct']} | {t['correct_gains']} | {t['correct_losses']} | {t['false_unknown_from_previously_correct']+t['false_unknown_from_previously_wrong']} |")
md+=['',
'Across positives, mixed gains 19 correct answers and loses six versus LoRA; versus source it gains 32 and loses four. Every new correctness loss is a false UNKNOWN. These descriptive net gains do not repair the failed false-UNKNOWN band.',
'',
'**All 13 errors occur in dev_sewing_circle occupied-person prompts**, affecting 9/15 sewing worlds. Sewing-person correctness nevertheless improves 20 to 29/45, and craft-person correctness improves 49 to 51/51. The observed behavior is therefore concentrated in a template-conditioned occupied-person binding task, rather than blanket abstention or failed EOS/format. Template family, worlds and entities co-vary; this does not establish a causal wording effect.',
'',
'| Positive category / template | Rows | Source correct | LoRA correct | Mixed correct | False UNKNOWN |',
'|---|---:|---:|---:|---:|---:|']
for v in sliced['category_template']:md.append(f"| {v['category']} / {v['template_family']} | {v['positive_rows']} | {v['source_correct']} | {v['control_correct']} | {v['mixed_correct']} | {v['false_unknown_rows']} |")
md+=['',
'All three sewing-person question forms incur errors: 5/15 for “Who holds…?”, 2/15 for “Who is recorded…?”, and 6/15 for “Whose…?”. Errors also occur in both assignment/duty orders. These correlated post-hoc slices carry no new significance test or causal ordering claim.',
'',
'| Occupied-person template | Relative fact order | Rows | False UNKNOWN |',
'|---|---|---:|---:|']
for v in sliced['category_template_order']:
 if v['category']=='person_answerable':
  label='Duty before occupied-person fact' if v['assignment_duty_order']=='duty_before_target_occupied' else 'Occupied-person fact before duty'
  md.append(f"| {v['template_family']} | {label} | {v['positive_rows']} | {v['false_unknown_rows']} |")
md+=['',
'All 13 affected IDs are listed below; full world, template, ordering and prediction-hash metadata are in BINDING_NUMERICAL_AUDIT.json.',
'',
'| Row ID | Source correct + EOS | LoRA correct + EOS |',
'|---|---|---|']
for r in error_rows:md.append(f"| {r['id']} | {'Yes' if r['source_correct'] else 'No'} | {'Yes' if r['control_correct'] else 'No'} |")
md+=['',
f"Evidence SHA256: RESULTS.json {sha(ROOT/'RESULTS.json')}; mixed GENERATIONS.json {sha(OUT/'GENERATIONS.json')}; input {base['input_sha256']}; gold {config['gold_sha256']}; criteria {sha(ROOT/'CRITERIA.json')}.",
'',
f"Reproduction script: {script_path}; SHA256 {script_sha}.",
'',
'No criteria, thresholds, endpoints, data, model, or existing artifact changed. This offline exploratory diagnosis establishes no confirmation, repair, preservation or release permission.']
audit_bytes=canonical(audit)+b'\n';markdown='\n'.join(md)+'\n'
for p,expected in list(evidence.items()):need(sha(p)==expected,'evidence changed during audit')
with (ROOT/'BINDING_NUMERICAL_AUDIT.json').open('xb') as f:f.write(audit_bytes);f.flush();os.fsync(f.fileno())
with (ROOT/'BINDING_ERROR_ANALYSIS.md').open('x',encoding='utf-8',newline='\n') as f:f.write(markdown);f.flush();os.fsync(f.fileno())
print(json.dumps({'status':audit['status'],'BINDING_NUMERICAL_AUDIT_SHA256':sha(ROOT/'BINDING_NUMERICAL_AUDIT.json'),'BINDING_ERROR_ANALYSIS_SHA256':sha(ROOT/'BINDING_ERROR_ANALYSIS.md'),'audit_script_sha256':script_sha,'binding_verdict':verdict,'false_unknown_rows':len(false_ids),'false_unknown_worlds':len(error_worlds),'sewing_person_order':[v for v in sliced['category_template_order'] if v['category']=='person_answerable' and v['template_family']=='dev_sewing_circle'],'compute_seconds':time.monotonic()-START},indent=2))
