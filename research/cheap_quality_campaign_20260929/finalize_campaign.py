"""Requirement-by-requirement completion checks and inspectable six-study synthesis."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=Path(__file__).resolve().parent
OUT=CAMPAIGN/'runs/final-001'
WORKTREES=ROOT.parents[1]
EXPECTED_HEAD='ac9b79ac07ad650d8815c76eb4194a180f23e8d4'
SPECS={
 'CQS-001':('rineya-quality-stability','cheap_quality_stability_v1','stability-20260929-001','summary.json','report.html','CQS-001-tests',8,'CQS-001-analysis'),
 'CQC-001':('rineya-quality-complementarity','cheap_quality_complementarity_v1','run-20260929-01','analysis.json','report.html','CQC-001-tests-repair1',7,'CQC-001-analysis'),
 'CQA-001':('rineya-quality-abstention','cheap_quality_abstention_v1','abstention-002/report','report.json','REPORT.html','CQA-001-tests-resume1',16,'CQA-001-evaluate-resume1'),
 'CQD-001':('rineya-quality-shortcuts','cheap_quality_shortcuts_v1','study-001','report.json','REPORT.html','CQD-001-tests',6,'CQD-001-analysis'),
 'CQT-001':('rineya-quality-token-fit','cheap_quality_token_fit_v1','token-fit-20260929-001','summary.json','report.html','CQT-001-tests',8,'CQT-001-analysis'),
 'CQR-001':('rineya-quality-retrieval','cheap_quality_retrieval_v1','run-20260929-01','analysis.json','report.html','CQR-001-tests-resume1',11,'CQR-001-analysis')}

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def write(name,value):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def relative(path):return os.path.relpath(path,OUT).replace('\\','/')

assert not OUT.exists()
OUT.mkdir(parents=True)
started=time.monotonic()
study_data={};completion={};cost_receipts=[];report_paths={};test_total=0
for identity,(tree,package,run,report,view,test_run,test_count,analysis_run) in SPECS.items():
 repo=WORKTREES/tree/'ChatBubble-rvs001'
 assert repo.exists()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
 inherited_diff=subprocess.check_output(['git','diff','--name-only'],cwd=repo,text=True).strip()
 assert head==EXPECTED_HEAD and branch.startswith('codex/') and not inherited_diff
 folder=repo/'experiments'/package/'runs'/run
 payload=read(folder/report)
 study_data[identity]=payload;report_paths[identity]=folder/view
 assert (folder/view).is_file()
 resources=repo/'research/cheap_quality_campaign_20260929/runs'
 tested=read(resources/test_run/'SUPERVISOR.json');measured=read(resources/analysis_run/'SUPERVISOR.json')
 assert tested['status']==measured['status']=='PASS_CPU_PROCESS'
 log=(resources/test_run/'stderr.log').read_text()
 assert re.search(r'Ran '+str(test_count)+r' tests',log) and '\nOK' in log
 assert 'skipped' not in log and not re.search(r'\.\.\. (FAIL|ERROR)',log)
 test_total+=test_count
 assert measured['peak_observed_tree_rss_bytes']<=512*2**20
 assert measured['minimum_observed_host_available_bytes']>=1.5*2**30
 assert measured['elapsed_seconds']<=900 and measured['worker_output_bytes']<=50*2**20
 assert measured['before']['host_available_bytes']>=2*2**30
 assert measured['after']['disk_free_bytes']>=4*2**30
 assert measured['native_model_loads']==measured['gpu_worker_launches']==0
 completion[identity]={'status':'VERIFIED_COMPLETE_SCOPED_STUDY','worktree':str(repo),'head':head,'branch':branch,
  'inherited_files_unchanged':True,'report_path':str(folder/report),'report_sha256':sha(folder/report),
  'tests_passed':test_count,'test_receipt':str(resources/test_run/'SUPERVISOR.json'),
  'analysis_receipt':str(resources/analysis_run/'SUPERVISOR.json'),'analysis_wall_seconds':measured['elapsed_seconds'],
  'sampled_tree_rss_bytes':measured['peak_observed_tree_rss_bytes'],'output_bytes':measured['worker_output_bytes']}
 for receipt in sorted(resources.glob('*/SUPERVISOR.json')):
  value=read(receipt)
  cost_receipts.append({'study':identity,'path':str(receipt),**value})

s,c,a,d,t,r=(study_data[k] for k in SPECS)
assert s['status']=='CACHED_EVIDENCE_REPRODUCED' and s['unique_evaluation_items']==150
assert c['status']=='COMPLETE_OFFLINE_ANALYSIS'
assert a['responses']==512 and a['worlds']==64 and a['disagreements']==a['unresolved']==1
assert a['status']=='TWO_REVIEW_DESCRIPTIVE_REPORT_NO_ADJUDICATION'
assert d['status']=='PASS_DATA_VERIFIER_DIAGNOSTIC'
assert d['controls']['total']==2560 and d['controls']['false_accepts']==d['controls']['false_rejects']==0
assert d['costs']['heuristic_candidates']==3072 and d['population']=={'selection':256,'DEV':256,'per_family_per_role':64}
assert t['status']=='MEASURED_TOKENIZER_FIT_ONLY' and t['parents']==128 and t['rendered_rows']==256 and t['fit_decisions']==768
assert not t['model_loaded'] and not t['frameworks_imported']
assert all(p['retained']==256 and p['rejected']==0 for p in t['profile_totals'])
assert len(r['case_results'])==2304 and len(r['world_clusters'])==768
assert r['answer_correctness_measured'] is False and r['native_total_envelope_qualified'] is False
assert all(row['consumed_tokens']==row['budget'] for row in r['case_results'])
eq_path=Path(completion['CQA-001']['worktree'])/'experiments/cheap_quality_abstention_v1/runs/abstention-002/PACKET_EQUIVALENCE.json'
eq=read(eq_path)
assert eq['status']=='PASS_BYTE_IDENTICAL_REBOUND_PACKET' and all(v['identical'] for v in eq['files'].values())
assert len(eq['files'])==39
for identity,payload in study_data.items():
 completion[identity]['population_and_scope_checks']='PASS'
assert test_total==56
analysis_times={k:v['analysis_wall_seconds'] for k,v in completion.items()}
admitted=[x for x in cost_receipts if x.get('worker_started')]
blocked=[x for x in cost_receipts if not x.get('worker_started')]
failed=[x for x in admitted if x['status']!='PASS_CPU_PROCESS']
costs={'six_analysis_supervised_wall_seconds':sum(analysis_times.values()),'all_admitted_supervisor_wall_seconds':sum(x['elapsed_seconds'] for x in admitted),
 'admitted_worker_invocations':len(admitted),'successful_worker_invocations':len(admitted)-len(failed),'failed_test_invocations':len(failed),
 'blocked_before_worker_invocations':len(blocked),'blocked_preflight_wall_seconds':'NOT_RECORDED; not fabricated as zero',
 'highest_sampled_tree_rss_bytes':max(x['peak_observed_tree_rss_bytes'] for x in admitted),
 'lowest_host_available_bytes_during_admitted_workers':min(x['minimum_observed_host_available_bytes'] for x in admitted),
 'retrieval_OS_process_peak_working_set_bytes':max(x['peak_rss_bytes'] for x in r['resource_samples']),
 'canonical_six_study_output_bytes':sum(x['output_bytes'] for x in completion.values()),
 'review_a_reported_wall_seconds':a['costs']['review_a_reported']['wall_seconds'],
 'review_b_reported_wall_seconds':a['costs']['review_b_reported']['wall_seconds'],
 'review_tokens_and_call_counts':'UNMETERED_CODEX_ASSISTED_REVIEWS; no human certification',
 'new_Rineya_model_loads':0,'new_Rineya_generation_tokens':0,'new_optimizer_steps':0,'external_paid_services':0,
 'historical_model_costs':'Existing source-recorded costs were reused, not incurred by this campaign',
 'unmetered':'Operator, agent implementation/review, tool overhead and earlier small agent-local fixture checks. Goal elapsed accounting is separate.'}
all_models=next(x for x in c['by_policy']['permissive']['subset_oracles'] if len(x['models'])==4)
set_floor=next(x for x in d['shortcut_curves'] if x['family']=='finite_set_queries' and x['predictor']=='all_numbers')
recall=[x for x in r['aggregates'] if x['stratum']=='all']
findings={'stability':{p:{k:len(v) for k,v in values.items() if k.endswith('_ids')} for p,values in s['stability'].items()},
 'complementarity_permissive':{'oracle_union':all_models['oracle_union'],'best_single':all_models['best_single'],'rows':150},
 'abstention':{model:{key:values[key] for key in ('communicative_abstention_on_missing_evidence_lower','communicative_abstention_on_missing_evidence_upper','assertions_on_missing_evidence_lower','assertions_on_missing_evidence_upper','semantic_false_abstention_on_answerables_lower','semantic_false_abstention_on_answerables_upper')} for model,values in a['by_model'].items()},
 'set_public_number_copying_floor':set_floor,'verifier_controls':d['controls'],'token_fit_profiles':t['profile_totals'],
 'token_distributions':t['token_distributions'],'paired_wording_cost_delta':t['paired_variant_delta'],
 'retrieval_recall_curves':[{key:x[key] for key in ('method','budget','full_evidence_recall','any_evidence_recall','support_precision','selected_sentence_count')} for x in recall]}
recommendations=[
 {'priority':1,'action':'Prepare fresh TRAIN worlds for clock-endpoint arithmetic, explicit units, and paired missing-evidence contracts. Preserve monitor/DEV as evaluation-only; do not train on these exposed failures.','basis':'Three repeated clock-value losses, strict unit-only losses, and Opallo communicative deferral3–4/64 versus Supa13/64.'},
 {'priority':2,'action':'Design a separately frozen public-information selector between independently competent source/last20/full-B28 candidates; fit on disjoint selection parents and count all inference/switching costs.','basis':'Gold-informed union96 versus best76 establishes headroom, not a deployable selector; Opallo/last20 offers little added diversity.'},
 {'priority':3,'action':'Use simple query expansion as a compact-system hypothesis; next add missing-support controls and then qualify a matched-format native comparison.','basis':'Two-pass retrieval retains all required synthetic evidence at128tokens. No Rineya answer accuracy was measured.'},
 {'priority':4,'action':'Keep operator-sensitive counterfactuals and shortcut baselines in future task registrations; do not tune the current frozen RVS bank.','basis':'A public-number copying rule solves21/64 set cases; all2560 verifier controls pass.'},
 {'priority':5,'action':'Use the measured short-token envelope and response+EOS mask as data preparation evidence for a later qualified small adaptation; retain an ordinary SFT control.','basis':'All256 renderings fit512with128reserve; processed maximum183. This is not learnability or native runtime qualification.'}]
summary={'status':'ALL_SIX_CPU_STUDIES_VERIFIED_COMPLETE','objective_scope':'Execute six prioritized cheap studies with hardware monitoring and a separate worktree each; identify scoped follow-up hypotheses for compact Rineya quality.',
 'completion_audit':completion,'latest_scoped_tests_passed':test_total,'costs':costs,'findings':findings,'recommendations':recommendations,
 'interpretation':'Historical exposed-output analyses, Codex-assisted communicative annotations, and fresh CPU data/tokenizer/retrieval diagnostics. No new Rineya weights, native accuracy improvement, deployment qualification, broad intelligence gain or promotion is established.',
 'native_RVS':'Still requires its own fresh7.4GiB post-import host admission and inference qualification; these CPU successes do not waive native guards.',
 'preservation':'All failed tests, blocked preflight, prior packet/source/reviews and outputs retained. Shared environment/source assets/historical files unchanged.'}
write('FINAL_SUMMARY.json',summary);write('COMPLETION_AUDIT.json',completion);write('FULL_COSTS.json',costs);write('SUPERVISOR_LEDGER.json',cost_receipts)
csv_rows=[{'study':k,**{q:v[q] for q in ('tests_passed','analysis_wall_seconds','sampled_tree_rss_bytes','output_bytes')}} for k,v in completion.items()]
with (OUT/'STUDY_COSTS.csv').open('x',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
brief=[('Gain/loss stability','19 stable gains;9 recurrent source losses;3 universal losses under permissive scoring.'),
 ('Cached complementarity','96/150 oracle union versus76/150 best single. Oracle headroom is not selection accuracy.'),
 ('Communicative abstention','On64 missing-evidence rows: Supa13 deferrals; Opallo3–4. One unresolved review disagreement retained.'),
 ('Shortcut/verifier audit','Public number-copying solves21/64 set queries. All2560 verifier controls pass, no false accepts/rejects.'),
 ('Tokenizer fit','All256 renderings/768 profiles fit, including512context with128reserve. Largest sequence183tokens.'),
 ('Retrieval evidence','Two-pass public query expansion retains100% required synthetic evidence at128/256/512tokens; no model answers measured.')]
table=''.join('<tr><td>'+html.escape(name)+'</td><td>'+html.escape(text)+'</td><td>'+str(completion[k]['tests_passed'])+'</td><td><a href="'+relative(report_paths[k])+'">Report</a></td></tr>' for k,(name,text) in zip(SPECS,brief))
cost_table=''.join('<tr><td>'+k+'</td><td>'+f'{v["analysis_wall_seconds"]:.3f}'+'</td><td>'+f'{v["sampled_tree_rss_bytes"]/2**20:.2f}'+'</td><td>'+f'{v["output_bytes"]/2**20:.2f}'+'</td></tr>' for k,v in completion.items())
recall_table=''.join('<tr><td>'+x['method']+'</td><td>'+str(x['budget'])+'</td><td>'+f'{100*x["full_evidence_recall"]:.2f}%'+'</td><td>'+f'{100*x["any_evidence_recall"]:.2f}%'+'</td></tr>' for x in recall)
actions=''.join('<li><b>'+html.escape(x['action'])+'</b><br>'+html.escape(x['basis'])+'</li>' for x in recommendations)
doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Rineya cheap-quality campaign: six completed studies</title><style>body{{font:16px/1.5 system-ui;color:#172235;max-width:1150px;margin:36px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #cbd3df;padding:8px;text-align:left}}th{{background:#edf2f7}}.scope{{padding:18px;background:#e9f3ed;border-left:5px solid #2b7b4e}}li{{margin:16px 0}}a{{color:#155ab4}}code{{overflow-wrap:anywhere}}</style><h1>Six cheap Rineya quality studies completed</h1><p>September29,2026 · six separate managed worktrees · existing pinned interpreter · sequential monitored CPU workers.</p><p class="scope">All six scoped studies executed and verified. All56 latest scoped tests passed. These results guide compact-model research; they establish no new trained checkpoint, native quality improvement or broad intelligence gain.</p><table><tr><th>Study</th><th>Result</th><th>Tests</th><th>Evidence</th></tr>{table}</table><h2>Practical research priorities</h2><ol>{actions}</ol><h2>Measured retrieval curves</h2><p>64 synthetic worlds reused in three positions,192contexts and2304conditions. Two-hop full recall needs both supports. Budgets are raw selected context plus fixed irrelevant padding; query/chat overhead is excluded.</p><table><tr><th>Method</th><th>Context tokens</th><th>Full evidence recall</th><th>Any evidence recall</th></tr>{recall_table}</table><h2>Costs and hardware</h2><table><tr><th>Study</th><th>Supervised analysis seconds</th><th>Sampled tree RSS MiB</th><th>Output MiB</th></tr>{cost_table}</table><p>The six analyses total {costs['six_analysis_supervised_wall_seconds']:.3f} supervised seconds. All {len(admitted)} admitted supervisor invocations, including tests, preparations, equality check and two preserved failed test attempts, total {costs['all_admitted_supervisor_wall_seconds']:.3f} seconds. One blocked preflight launched no worker; its elapsed cost was not recorded.</p><p>Largest sampled process-tree RSS: {costs['highest_sampled_tree_rss_bytes']/2**20:.2f} MiB. Retrieval also records an OS process peak of {costs['retrieval_OS_process_peak_working_set_bytes']/2**20:.2f} MiB, versus its90.2MiB sampled tree peak. Sampling can miss brief peaks. Lowest available host RAM during admitted workers: {costs['lowest_host_available_bytes_during_admitted_workers']/2**30:.3f} GiB. Limits remained512MiB tree RSS,15-minute study commands,50MiB output/study,2GiB host admission,1.5GiB ongoing host headroom and4GiB free disk.</p><p>Two Codex-assisted annotation reviews report177.531s and138s. Review token/call counts, operator/agent implementation and tool overhead are unmetered; no human certification is claimed. Historical inference/training costs remain source-reported and were not incurred again. New local Rineya loads/tokens, optimizer steps, GPU workers, weight writes, downloads, external paid calls and protected-payload access are zero.</p><h2>Integrity and interpretation</h2><p>All original/sibling worktrees and historical files were preserved. Inputs, tokenizer assets, code, outputs and reviews are hash-bound. Late abstention evaluation hardening is documented; the original source/packet survives, and39 reviewer/evaluator artifacts were proven byte-identical before annotation reuse. Disagreement is retained without adjudication. Raw private reports remain local.</p><p>Historical panels are exposed and repeatedly inspected. Accuracy/retention counts depend on typed scoring, with human target review pending. The semantic review classifies communicative acts without context. Token fit does not prove learning or native fit; synthetic retrieval recall does not prove model use of evidence. Native RVS still needs its unchanged resource and inference-qualification gates.</p><p><a href="FINAL_SUMMARY.json">Complete structured summary</a> · <a href="COMPLETION_AUDIT.json">Requirement audit</a> · <a href="FULL_COSTS.json">Full cost scope</a> · <a href="SUPERVISOR_LEDGER.json">All process receipts</a> · <a href="STUDY_COSTS.csv">Cost CSV</a></p></html>'''
with (OUT/'FINAL_REPORT.html').open('x',encoding='utf-8') as f:f.write(doc)
write('FINAL_ARTIFACTS.json',{'status':'PASS_FINAL_SYNTHESIS','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file()},
 'aggregation_elapsed_seconds':time.monotonic()-started,'source_script_sha256':sha(Path(__file__))})
print(json.dumps({'status':summary['status'],'tests':test_total,'analyses_seconds':costs['six_analysis_supervised_wall_seconds'],
 'all_admitted_supervisor_seconds':costs['all_admitted_supervisor_wall_seconds'],'output':str(OUT)},indent=2))
