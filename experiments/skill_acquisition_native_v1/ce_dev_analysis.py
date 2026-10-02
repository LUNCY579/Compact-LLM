"""Pure paired DEV/guard joins over complete frozen greedy scoring receipts.

No files, banks or model outputs are opened here. Strict scores are joined from
the existing scoped scorer receipts and bound to exact token journals. This is
an exposed-bank diagnostic report, not an execution grant or global forgetting
claim. Statistical kernels are the existing stats_exact functions.
"""
from __future__ import annotations

import copy

from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer as scorer,stats_exact as stats

SCHEMA='CE8-DEV-PAIRED-ANALYSIS-1'
SCOPE='CE8-DEV-EVALUATION-SCOPE-1'
READ='CE8-DEV-GREEDY-READ-1'
BANKS={'dev_core_R':250,'dev_core_S':250,'dev_core_P':250,'guard_dev_legacy':1200}
BASELINES=('Supa','warm_anchor')
MODELS=BASELINES+('CE375_donor',)
GREEDY={'batch_size':1,'do_sample':False,'num_beams':1,'max_new_tokens':128,
    'eos_token_id':[151645,151643],'pad_token_id':151643,'enable_thinking':False,
    'add_generation_prompt':True,'add_special_tokens':False,'use_model_defaults':False}
digest=scorer.digest


def require(condition,message):
    if not condition:raise ValueError(message)


def _hex(value):return type(value) is str and len(value)==64 and not set(value)-set('0123456789abcdef')


def _scope(scope):
    require(type(scope) is dict and scope.get('schema')==SCOPE and scope.get('choice')=='canonical_only'
            and scope.get('greedy')==GREEDY and type(scope.get('banks')) is dict and set(scope['banks'])==set(BANKS)
            and _hex(scope.get('manifest_sha256')),'exact frozen canonical DEV/guard scope')
    for name,n in BANKS.items():
        bank=scope['banks'][name];ids=bank.get('ids');parents=bank.get('parent_by_id');prompts=bank.get('prompt_hash_by_id')
        require(type(ids) is list and len(ids)==n and ids==sorted(set(ids)) and all(type(item) is str and item for item in ids)
            and type(parents) is dict and set(parents)==set(ids) and all(type(value) is str and value for value in parents.values())
            and len(set(parents.values()))==n and type(prompts) is dict and set(prompts)==set(ids)
            and all(_hex(value) for value in prompts.values()),'fixed complete independent parent/prompt membership: '+name)
        require(bank.get('role')==('guard_dev' if name=='guard_dev_legacy' else 'dev') and _hex(bank.get('bank_sha256'))
            and bank.get('ids_sha256')==digest(ids) and bank.get('parents_sha256')==digest(sorted(set(parents.values()))),
            'bank and exact ID/parent digest: '+name)
        if name=='guard_dev_legacy':
            require(bank.get('scope_receipt_sha256') is None and type(bank.get('family_by_id')) is dict
                and set(bank['family_by_id'])==set(ids) and all(type(value) is str and value for value in bank['family_by_id'].values()),
                'explicit frozen guard family membership')
        else:require(_hex(bank.get('scope_receipt_sha256')),'complete frozen DEV R/S/P scope receipt')
    return scope


def _read(read,scope,name,model):
    bank=scope['banks'][name];binding=scope['model_bindings'][model]
    require(type(read) is dict and read.get('schema')==READ and read.get('bank')==name and read.get('model_name')==model
        and _hex(read.get('process_identity_sha256')),'actual bound complete read identity')
    require(type(binding) is dict and set(binding)=={'model_sha256','tokenizer_sha256','decoding_sha256','scorer_bundle_sha256'}
        and all(_hex(value) for value in binding.values()),'frozen model/deployed arm/runtime/scorer binding')
    receipt=read.get('scoring_receipt');rows=read.get('journal_rows')
    require(type(receipt) is dict and receipt.get('kind')=='scoped_scoring_receipt' and receipt.get('role')==bank['role']
        and receipt.get('bank_sha256')==bank['bank_sha256'] and receipt.get('manifest_sha256')==scope['manifest_sha256']
        and receipt.get('scope_receipt_sha256')==bank['scope_receipt_sha256']
        and receipt.get('diagnostic') is False and receipt.get('strict_endpoint')=='experiments.maxrl_composable_skills_v2.corpus.reward (unchanged)',
        'original scope/bank/strict scorer receipt differs')
    frozen_scope=receipt.get('scope')
    require(type(frozen_scope) is dict and frozen_scope.get('ids_sha256')==bank['ids_sha256']
        and frozen_scope.get('parents_sha256')==bank['parents_sha256'] and frozen_scope.get('n')==len(bank['ids'])
        and frozen_scope.get('parents')==len(bank['ids']) and frozen_scope.get('one_row_per_parent_per_signature') is True,
        'scorer scope partial/repeated parents')
    if 'scoring_scope' in bank:
        require(frozen_scope=={key:value for key,value in bank['scoring_scope'].items() if key!='ids'},
                'scorer filtering differs from exact frozen bank declaration')
    require(receipt.get('run')=={key:binding[key] for key in ('model_sha256','tokenizer_sha256','decoding_sha256')}
        and receipt.get('code_bundle_sha256')==binding['scorer_bundle_sha256']
        and type(receipt.get('code_sha256')) is dict and digest(receipt['code_sha256'])==binding['scorer_bundle_sha256'],
        'read runtime/deployed-arm/scorer code mismatch')
    require(type(rows) is list and len(rows)==len(bank['ids']) and [row.get('id') for row in rows]==bank['ids'],
        'every planned journal ID exactly once in frozen order')
    scores=receipt.get('rows');require(type(scores) is list and [row.get('id') for row in scores]==bank['ids'],'complete exact scored ID set')
    responses=[];by_id={}
    for row,score in zip(rows,scores):
        rid=row['id'];tokens=row.get('completion_ids')
        require(row.get('parent_id')==score.get('parent_id')==bank['parent_by_id'][rid]
            and row.get('prompt_hash')==bank['prompt_hash_by_id'][rid] and _hex(row.get('prompt_ids_sha256')),
            'wrong parent or rendered prompt binding')
        require(type(tokens) is list and 0<len(tokens)<=128 and all(type(value) is int and value>=0 for value in tokens)
            and type(row.get('generated_tokens')) is int and row['generated_tokens']==len(tokens)
            and row.get('stop_reason')==score.get('stop_reason') and type(row.get('response')) is str,
            'raw token/response length/stop linkage')
        eos=[index for index,value in enumerate(tokens) if value in (151645,151643)]
        require(row['stop_reason'] in ('eos','length') and (eos==[len(tokens)-1] if row['stop_reason']=='eos' else not eos and len(tokens)==128),
            'positional terminal EOS and fixed cap')
        require(type(score.get('strict')) is int and score['strict'] in (0,1) and type(score.get('content_pass')) is bool
            and score.get('generated_tokens')==row['generated_tokens'],'strict/content scorer values are distinct')
        responses.append({key:row[key] for key in scorer.RESPONSE_FIELDS});by_id[rid]=(row,score)
    require(receipt.get('responses_sha256')==digest(sorted(responses,key=lambda row:row['id']))
        and receipt.get('strict_correct_ids')==[rid for rid in bank['ids'] if by_id[rid][1]['strict']],
        'scoring responses or source-correct membership changed')
    return by_id


def _counts(rows):
    return {'strict':sum(score['strict'] for _,score in rows.values()),
            'content':sum(score['content_pass'] for _,score in rows.values()),'rows':len(rows)}


def _paired(base,donor):
    better=[rid for rid in base if not base[rid][1]['strict'] and donor[rid][1]['strict']]
    worse=[rid for rid in base if base[rid][1]['strict'] and not donor[rid][1]['strict']]
    content_better=[rid for rid in base if not base[rid][1]['content_pass'] and donor[rid][1]['content_pass']]
    content_worse=[rid for rid in base if base[rid][1]['content_pass'] and not donor[rid][1]['content_pass']]
    return {'strict_better_ids':better,'strict_worse_ids':worse,'strict_net':len(better)-len(worse),
            'content_better_ids':content_better,'content_worse_ids':content_worse,'content_net':len(content_better)-len(content_worse)}


def analyze_dev(scope,enrollments=None,donor_reads=None):
    """Join already produced data; no bank/response file reads or authority."""
    _scope(scope)
    missing=[]
    if type(scope.get('model_bindings')) is not dict or set(scope['model_bindings'])!=set(MODELS):missing.append('frozen Supa/warm/deployed-donor bindings')
    elif any(type(scope['model_bindings'][model]) is not dict or any(value is None for value in scope['model_bindings'][model].values())
             for model in MODELS):missing.append('actual completed deployed donor/baseline bindings')
    if enrollments is None:missing.append('two independent complete baseline reads and first-read source memberships')
    if donor_reads is None:missing.append('actual complete donor reads')
    if missing:return {'schema':SCHEMA,'status':'NOT_RUN','missing_dependencies':missing,'native_qualified':False,'evaluation_authorized':False}
    require(type(enrollments) is dict and set(enrollments)==set(BASELINES) and type(donor_reads) is dict and set(donor_reads)==set(BANKS),
            'exact two baseline enrollments and donor four-bank reads')
    baselines={};enrollment_records={}
    for model in BASELINES:
        entry=enrollments[model]
        require(type(entry) is dict and set(entry)=={'first_reads','second_reads','memberships','zero_token_churn'}
            and entry['zero_token_churn'] is True and all(type(entry[key]) is dict and set(entry[key])==set(BANKS)
                for key in ('first_reads','second_reads','memberships')),'complete frozen baseline enrollment/churn')
        baselines[model]={};enrollment_records[model]={}
        process1={read['process_identity_sha256'] for read in entry['first_reads'].values()}
        process2={read['process_identity_sha256'] for read in entry['second_reads'].values()}
        require(len(process1)==len(process2)==1 and process1.isdisjoint(process2),'two distinct independent full-bank baseline processes')
        for name in BANKS:
            first=_read(entry['first_reads'][name],scope,name,model);second=_read(entry['second_reads'][name],scope,name,model)
            for rid in first:
                left,right=first[rid],second[rid]
                require(all(left[0][field]==right[0][field] for field in ('parent_id','prompt_hash','prompt_ids_sha256','completion_ids','response','stop_reason'))
                    and left[1]['strict']==right[1]['strict'] and left[1]['content_pass']==right[1]['content_pass'],
                    'STOP_NONDETERMINISTIC: baseline token/scoring churn')
            derived=scorer.enrollment_from_receipt(entry['first_reads'][name]['scoring_receipt'])
            require(entry['memberships'][name]==derived,'frozen membership differs from baseline first complete receipt')
            baselines[model][name]=first;enrollment_records[model][name]=derived
    donor={name:_read(donor_reads[name],scope,name,'CE375_donor') for name in BANKS}
    require(len({read['process_identity_sha256'] for read in donor_reads.values()})==1,'one complete donor process for all four banks')
    for name in BANKS:
        for model in BASELINES:
            require(all(baselines[model][name][rid][0]['prompt_ids_sha256']==donor[name][rid][0]['prompt_ids_sha256']
                for rid in donor[name]),'paired models rendered different prompt IDs')
    h1=_paired(baselines['warm_anchor']['dev_core_R'],donor['dev_core_R'])
    limb=stats.mcnemar_limb(len(h1['strict_better_ids']),len(h1['strict_worse_ids']),level=.05)
    passed=limb['state']=='PASS' and h1['strict_net']>=13
    h1.update(exact_mcnemar=limb,state='PASS' if passed else 'UNCERTIFIABLE' if limb['state']=='UNCERTIFIABLE' else 'FAIL',
        strict_net_required=13,alpha_exact=[1,20],endpoint='strict body pass@1; content change is separate')
    retention={};warm_preexisting={}
    for model in BASELINES:
        retention[model]={}
        for name,margin in (('dev_core_S',.10),('dev_core_P',.10),('guard_dev_legacy',.05)):
            source=baselines[model][name];members=enrollment_records[model][name]['source_correct_ids']
            losses=[rid for rid in members if not donor[name][rid][1]['strict']]
            repairs=[rid for rid in source if not source[rid][1]['strict'] and donor[name][rid][1]['strict']]
            report={'source_correct_ids':copy.deepcopy(members),'source_correct_ids_sha256':digest(members),'n':len(members),
                'loss_ids':losses,'repair_ids':repairs,'losses':len(losses),'repairs':len(repairs),
                'loss_limb':stats.loss_rate_limb(len(losses),len(members),margin,.95),
                'minimum_zero_loss_certifiable_n':59 if name=='guard_dev_legacy' else 29,'full_bank_pairing':_paired(source,donor[name])}
            if name=='guard_dev_legacy':
                families=scope['banks'][name]['family_by_id'];report['by_family']={}
                for family in sorted(set(families.values())):
                    correct=[rid for rid in members if families[rid]==family]
                    report['by_family'][family]={'source_correct_n':len(correct),'losses':sum(rid in losses for rid in correct),
                        'repairs':sum(families[rid]==family for rid in repairs),'pooled_margin_does_not_certify_this_family':True}
            retention[model][name]=report
    for name in BANKS:warm_preexisting[name]=_paired(baselines['Supa'][name],baselines['warm_anchor'][name])
    return {'schema':SCHEMA,'status':'FIXED_EXPOSED_BANK_REPORT','acquisition_H1':h1,'source_correct_retention':retention,
        'warm_preexisting_change':warm_preexisting,'counts':{model:{name:_counts(baselines[model][name]) for name in BANKS}
            for model in BASELINES}|{'CE375_donor':{name:_counts(donor[name]) for name in BANKS}},
        'baseline_churn':{'Supa':0,'warm_anchor':0},'strict_content_separate':True,
        'claim':'exact exposed DEV/guard bank changes; conditional binomial diagnostics only, not global forgetting, language preservation or composition',
        'assumptions':['Within each endpoint one row per independent parent/world; DEV R/S/P shared worlds are not750 independent samples',
            'Clopper-Pearson population interpretation needs a justified conditional independent Bernoulli/binomial model'],
        'native_qualified':False,'evaluation_authorized':False}
