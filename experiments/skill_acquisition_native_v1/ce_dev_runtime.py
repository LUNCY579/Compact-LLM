"""Fixed five-read DEV runtime; source verifier registration defaults CLOSED.

No caller decoder/model/lease callback can authorize a read. The entry owns the
actual observed worker, one runtime lease and finite costs. Existing generation,
FP32 loading, delta addition and strict response kernels remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import time
import uuid

from . import ce_qualification_artifacts as raw, ce_worker_evidence as worker
from . import update

CONTEXT = 'CE8-DEV-READ-CONTEXT-1'
POLICY = 'CE8-DEV-READ-POLICY-1'
READ = 'CE8-DEV-GREEDY-READ-1'
_REGISTERED_DEV_VERIFIERS = frozenset()
_SEAL = object()
RUNS = (('Supa-read-1','Supa',1),('Supa-read-2','Supa',2),
        ('warm_anchor-read-1','warm_anchor',1),('warm_anchor-read-2','warm_anchor',2),('CE375_donor-read-1','CE375_donor',1))
BANKS = ('dev_core_R','dev_core_S','dev_core_P','guard_dev_legacy')
GREEDY = {'batch_size':1,'do_sample':False,'num_beams':1,'max_new_tokens':128,'eos_token_id':[151645,151643],
          'pad_token_id':151643,'enable_thinking':False,'add_generation_prompt':True,'add_special_tokens':False,'use_model_defaults':False}
KINDS = frozenset(('dev_module_import','dev_worker','dev_source_read','dev_tokenizer','dev_configure','dev_model_load',
    'dev_delta_rebuild','dev_fingerprint','dev_decode','dev_journal','dev_receipt','dev_cleanup'))
require, canonical, digest = raw.require, raw.canonical, raw.digest


def context_preimage(context):
    require(type(context) is dict, 'explicit DEV read context')
    return {key:value for key,value in context.items() if key not in ('dev_read_permit','native_services')}


def _reference(value):
    require(type(value) is dict and type(value.get('path')) is str and Path(value['path']).is_absolute()
            and raw._hex(value.get('sha256')) and raw._int(value.get('bytes'),1), 'exact bounded source DEV reference')


def validate_context(context):
    require(type(context) is dict and context.get('schema') == CONTEXT and context.get('evaluation_authorized') is True
            and context.get('production_training_authorized') is False
            and all(raw._hex(context.get(key)) for key in ('request_sha256','scope_sha256')), 'separate source DEV evaluation grant')
    scope, plan = context['scope'], context['plan']
    require(scope.get('schema') == 'CE8-DEV-EVALUATION-SCOPE-1' and scope.get('evaluation_authorized') is True
            and scope.get('production_training_authorized') is False and scope.get('greedy') == GREEDY
            and context['scope_sha256'] == digest(scope), 'exact registered greedy DEV scope')
    require(plan.get('schema') == 'CE8-DEV-FIXED-READ-PLAN-1' and plan.get('evaluation_authorized') is True
            and plan.get('scope_sha256') == context['scope_sha256'] and plan.get('rows_per_read') == 1950
            and plan.get('total_greedy_completions') == 9750 and plan.get('max_new_tokens') == 128
            and plan.get('refill_allowed') is False and plan.get('checkpoint_selection_allowed') is False
            and plan.get('allowed_roles') == ['dev','guard_dev'], 'five fixed reads, no refill/checkpoint selection')
    reads = plan.get('reads')
    require(type(reads) is list and [(r.get('read_id'),r.get('model_name'),r.get('read_index')) for r in reads] == list(RUNS)
            and all(r.get('fresh_process_required') is True and r.get('bank_names') == list(BANKS)
                    and type(r.get('rows')) is int and r['rows'] == 1950 for r in reads), 'exact five source read roles/order/dose')
    require(type(scope.get('banks')) is dict and set(scope['banks']) == set(BANKS), 'only DEV R/S/P and legacy guard banks')
    for name,n in zip(BANKS,(250,250,250,1200)):
        bank = scope['banks'][name]; ids = bank['ids']
        require(type(ids) is list and len(ids) == n and ids == sorted(set(ids)) and all(type(rid) is str and rid for rid in ids)
                and bank['ids_sha256'] == digest(ids) and set(bank['parent_by_id']) == set(ids)
                and set(bank['prompt_hash_by_id']) == set(ids)
                and all(raw._hex(value) for value in bank['prompt_hash_by_id'].values()), 'exact complete DEV bank identities')
    runtime = context['runtime']
    require(runtime['decoding_preimage']['device'] == 'cuda:0' and runtime['decoding_preimage']['dtype'] == 'float32'
            and runtime['decoding_sha256'] == digest(runtime['decoding_preimage']), 'fixed FP32 DEV runtime')
    model = context['model']
    require(type(model['model_dir']) is str and Path(model['model_dir']).is_absolute(), 'fixed base model directory')
    for key in ('asset_receipt_ref','tokenizer_receipt_ref','warm_delta_ref','donor_ref'): _reference(model[key])
    require(raw._hex(model['warm_delta_ref'].get('data_sha256')) and raw._hex(model['donor_ref'].get('data_sha256')),
            'actual saved warm/R data-section hashes before model loading')
    require(all(raw._hex(model.get(key)) for key in ('base_weights_sha256','Supa_sha256','Supa_state_fingerprint_sha256','warm_anchor_sha256','CE375_donor_sha256'))
            and model['Supa_sha256'] == model['base_weights_sha256']
            and model['donor_ref']['deployed_arm_sha256'] == model['CE375_donor_sha256'], 'fixed model/common/deployed donor identities')
    for name in ('Supa','warm_anchor','CE375_donor'):
        bound = scope['model_bindings'][name]
        require(bound['model_sha256'] == model[name + '_sha256'] and bound['tokenizer_sha256'] == model['tokenizer_receipt_ref']['sha256']
                and bound['decoding_sha256'] == runtime['decoding_sha256'], 'resolved source model/tokenizer/runtime read binding')
    policy = context['policy']
    require(policy.get('schema') == POLICY and type(policy.get('output_directory')) is str
            and Path(policy['output_directory']).is_absolute() and raw._finite(policy.get('physical_wall_seconds_upper'))
            and policy['physical_wall_seconds_upper'] > 0 and raw._int(policy.get('journal_bytes_upper'),1)
            and policy['journal_bytes_upper'] <= 256 * raw.CHUNK and raw._int(policy.get('receipt_bytes_upper'),1)
            and policy['receipt_bytes_upper'] <= raw.MAX_JSON and set(policy.get('operation_profiles',{})) == KINDS,
            'finite DEV output/elapsed/IO/hash policy before framework')
    from .ce_native_services import _profile
    for profile in policy['operation_profiles'].values(): _profile(profile)
    require(policy['operation_profiles']['dev_cleanup']['wall_s_upper'] >= policy['physical_wall_seconds_upper'],
            'prepaid cleanup covers entire loaded-model lifetime')
    canonical(context_preimage(context))
    return context


class DevReadPermit:
    __slots__ = ('_seal','_context','_issuer','_verifier','_operation','_gate','_worker','_lease')
    def __init__(self, context, issuer, verifier, operation, gate):
        require(any(verifier is value for value in _REGISTERED_DEV_VERIFIERS), 'source DEV verifier remains closed')
        require(issuer is not None and not isinstance(issuer,(dict,list,str,int,float,bool)), 'original source entry issuer')
        validate_context(context)
        require(callable(operation) and callable(gate) and issuer.operation is operation and issuer.gate is gate
                and type(issuer.dev_worker_handle) is worker.WinWorkerHandle and issuer.dev_lease_owner is not None
                and not isinstance(issuer.dev_lease_owner,(dict,list,str,int,float,bool)), 'source-owned DEV worker/lease/observer identities')
        preimage = digest(context_preimage(context))
        require(verifier(issuer,preimage,context['scope_sha256'],context['request_sha256']) is True, 'fixed source DEV issuer verification')
        for key,value in {'_seal':_SEAL,'_context':preimage,'_issuer':issuer,'_verifier':verifier,'_operation':operation,
                '_gate':gate,'_worker':issuer.dev_worker_handle,'_lease':issuer.dev_lease_owner}.items(): object.__setattr__(self,key,value)
    def __setattr__(self,name,value): raise AttributeError('immutable source DEV permit')
    def require(self,context,operation,gate,worker_handle,lease_owner):
        require(self._seal is _SEAL and any(self._verifier is value for value in _REGISTERED_DEV_VERIFIERS)
                and operation is self._operation and gate is self._gate and worker_handle is self._worker and lease_owner is self._lease
                and digest(context_preimage(context)) == self._context, 'same source context/actual worker/owned lease before DEV work')
        require(self._verifier(self._issuer,self._context,context['scope_sha256'],context['request_sha256']) is True,
                'source DEV worker/lease remains admitted')


def model_load_steps(model_name):
    require(model_name in ('Supa','warm_anchor','CE375_donor'), 'fixed read model role')
    return {'Supa':['base_FP32'], 'warm_anchor':['base_FP32','add_warm_FP32'],
            'CE375_donor':['base_FP32','add_warm_FP32','add_saved_R_FP32']}[model_name]


def _read_ref(reference, maximum):
    _reference(reference)
    require(reference['bytes'] <= maximum, 'DEV source read byte upper')
    path = Path(reference['path']); require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'regular source DEV file')
    with path.open('rb') as stream: blob = stream.read(reference['bytes']+1)
    require(len(blob) == reference['bytes'] and hashlib.sha256(blob).hexdigest() == reference['sha256'], 'DEV source bytes/hash changed')
    return blob


def _source_rows(context):
    refs = context['scope']['metadata_refs']
    rows = {}
    for name in ('canonical_dev','canonical_guard_dev'):
        for line in _read_ref(refs[name],32*raw.CHUNK).splitlines():
            row = raw._json(line); require(row['role'] in ('dev','guard_dev') and row['id'] not in rows, 'DEV-only unique source row')
            rows[row['id']] = row
    selected = []
    for bank in BANKS:
        entry = context['scope']['banks'][bank]
        for rid in entry['ids']:
            row = rows[rid]
            require(row['role'] == entry['role'] and row['parent_id'] == entry['parent_by_id'][rid]
                    and row['prompt_hash'] == entry['prompt_hash_by_id'][rid], 'actual source row differs from frozen DEV identity')
            selected.append((bank,row))
    require(len(selected) == 1950, 'exact complete DEV plan')
    return selected


def validate_churn(first, second):
    require(type(first) is list and type(second) is list and len(first) == len(second) == 1950
            and [row['id'] for row in first] == [row['id'] for row in second], 'two exact complete baseline reads before churn comparison')
    changed = [a['id'] for a,b in zip(first,second) if a['completion_ids'] != b['completion_ids'] or a['stop_reason'] != b['stop_reason']]
    return {'rows':1950,'changed_ids':changed,'zero_generated_token_churn':not changed}


def fresh_journal_paths(directory):
    """A fixed read belongs to one process; retain and refuse any partial read."""
    require(directory.is_absolute() and not directory.is_symlink(), 'regular source DEV run directory')
    targets = (directory/'TOKEN_JOURNAL.jsonl', directory/'ATTEMPTS.jsonl')
    require(all(not path.exists() or not path.is_symlink() and path.is_file() and path.stat().st_size == 0 for path in targets),
            'fixed one-process DEV read already contains retained work; no implicit resume')
    return targets


def run_dev_read(context, run_name, *, operation, gate, worker_handle, lease_owner):
    permit = context.get('dev_read_permit') if type(context) is dict else None
    require(type(permit) is DevReadPermit, 'source DEV read permit before IO/framework/model')
    permit.require(context,operation,gate,worker_handle,lease_owner)
    validate_context(context)
    reads = {row['read_id']:row for row in context['plan']['reads']}
    require(run_name in reads and context['read_id'] == run_name, 'one exact source-bound fresh read')
    read, policy = reads[run_name], context['policy']; model_name = read['model_name']
    origin = time.monotonic(); profiles = policy['operation_profiles']; identifier = uuid.uuid4().hex
    model = tokenizer = torch = None; primary = None
    operation('dev_cleanup','started',operation_id=identifier,**profiles['dev_cleanup'])
    def span(kind):
        permit.require(context,operation,gate,worker_handle,lease_owner)
        require(time.monotonic()-origin <= policy['physical_wall_seconds_upper'], 'DEV physical deadline')
        return update._span(operation,kind,dict(profiles[kind],read_id=run_name))
    try:
        with span('dev_worker'):
            proof = worker.verify_worker_handle(worker_handle, {'request_sha256':context['request_sha256'],'scope_sha256':context['scope_sha256'],
                'worker_identity_sha256':worker_handle.identity_sha256}, require_native=True)
            process = {'process_identity_sha256':proof['worker_identity_sha256'],
                       'os_identity':copy.deepcopy(proof['os_identity'])}
        with span('dev_journal'):
            directory = Path(policy['output_directory'])/run_name
            path, attempts_path = fresh_journal_paths(directory)
        with span('dev_module_import'):
            gate('dev_module_import')
            from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as rt, s0_competence as comp, arm_build, scoped_scorer
            rt.assert_runtime()
            import torch
        with span('dev_source_read'):
            gate('dev_source_read'); source_rows = _source_rows(context)
            require(scoped_scorer.code_sha256() == context['scope']['source']['scorer_code_sha256'], 'unchanged pinned scorer code')
        with span('dev_configure'):
            gate('dev_configure'); rt.configure('cuda:0')
            actual = rt.decoding_preimage('cuda:0'); require(actual == context['runtime']['decoding_preimage'], 'actual registered DEV decoding preimage')
        assets = context['model']
        with span('dev_tokenizer'):
            gate('dev_tokenizer'); tokenizer = rt.load_tokenizer(assets['model_dir'],assets['tokenizer_receipt_ref']['path'],asset_receipt=assets['asset_receipt_ref']['path'])
        with span('dev_model_load'):
            gate('dev_model_load')
            warm = assets['warm_delta_ref']
            kwargs = {} if model_name == 'Supa' else {'delta_path':warm['path'],'delta_sha256':warm['data_sha256'],'arm_sha256':assets['warm_anchor_sha256']}
            model, identity = rt.load_anchor(assets['model_dir'],'cuda:0',assets['asset_receipt_ref']['path'],**kwargs)
            require(identity['weights_sha256'] == assets['base_weights_sha256'], 'actual pinned Supa base weights')
        if model_name == 'CE375_donor':
            with span('dev_delta_rebuild'):
                gate('dev_delta_rebuild')
                saved = arm_build.read_delta(assets['donor_ref']['path'],assets['donor_ref']['data_sha256'])
                try: rt.add_delta(model,saved)
                finally: saved.clear(); del saved
        with span('dev_fingerprint'):
            gate('dev_fingerprint'); deployed = rt.state_fingerprint(model)
            expected_state = assets['Supa_state_fingerprint_sha256'] if model_name == 'Supa' else assets[model_name+'_sha256']
            require(deployed == expected_state, 'actual base+warm+saved-R sequential FP32 deployed identity')
        model.eval(); device = next(model.parameters()).device
        with span('dev_journal'):
            gate('dev_journal'); generate = comp.generate_module(); contract = generate.generation_contract(128)
            require(not directory.is_symlink(), 'regular owned DEV run directory'); directory.mkdir(parents=True,exist_ok=True)
            bind = {'schema':READ,'read_id':run_name,'model_name':model_name,'read_index':read['read_index'],
                'model_sha256':assets[model_name+'_sha256'],'deployed_state_fingerprint_sha256':deployed,
                'tokenizer_sha256':assets['tokenizer_receipt_ref']['sha256'],
                'decoding_sha256':context['runtime']['decoding_sha256'],'scope_sha256':context['scope_sha256'],
                'generation_contract_sha256':contract['contract_sha256'],'source_sha256':digest(context['source']),
                'rows':1950,'greedy':GREEDY,'process_identity_sha256':proof['worker_identity_sha256']}
            fresh_journal_paths(directory)
            journal = generate.ResumableJsonl(path,key='id',bind=bind)
            attempts = generate.ResumableJsonl(attempts_path,key='event',bind=bind)
            ids = [row['id'] for _,row in source_rows]; start = journal.next_index(ids)
            require(start == 0, 'one source read starts before its first row')
            attempts.append({'event':proof['worker_identity_sha256']+':start','kind':'start','worker_proof':proof,'process':process,'resume_index':start})
        for index,(bank,row) in enumerate(source_rows[start:],start):
            with span('dev_decode'):
                gate('dev_decode'); text, prompt = rt.render_prompt(tokenizer,row['prompt'])
                result, elapsed = comp.timed(device,lambda:generate.greedy_decode(model,prompt,max_new_tokens=128))
                response = comp.response_text(tokenizer,result['completion_ids'],result['stop_reason'])
                record = {'id':row['id'],'index':index,'bank':bank,'parent_id':row['parent_id'],'prompt_hash':row['prompt_hash'],
                    'prompt_ids_sha256':digest(prompt),'render_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':len(prompt),
                    'completion_ids':result['completion_ids'],'stop_reason':result['stop_reason'],'generated_tokens':result['generated_tokens'],
                    'response':response['response'],'body_special_ids':response['body_special_ids'],'body_unknown_ids':response['body_unknown_ids'],
                    'decoder':result['decoder'],'wall_s':elapsed,'process':process,
                    'process_identity_sha256':proof['worker_identity_sha256'],'raw_step_logprobs':result['raw_step_logprobs']}
            with span('dev_journal'):
                require(path.stat().st_size <= policy['journal_bytes_upper']-profiles['dev_journal']['write_bytes_upper'], 'permanent DEV journal size bound')
                journal.append(record)
        with span('dev_fingerprint'):
            gate('dev_fingerprint'); require(rt.state_fingerprint(model) == deployed, 'readonly DEV run changed model state')
        with span('dev_journal'):
            require(journal.next_index(ids) == 1950, 'all exact DEV rows completed')
            attempts.append({'event':proof['worker_identity_sha256']+':end','kind':'end','outcome':'COMPLETE','rows':1950,'process':process})
        with span('dev_receipt'):
            value = {'schema':READ,'status':'DEV_BODY_COMPLETE','read_id':run_name,'model_name':model_name,'read_index':read['read_index'],
                'scope_sha256':context['scope_sha256'],'model_sha256':assets[model_name+'_sha256'],
                'deployed_state_fingerprint_sha256':deployed,'tokenizer_sha256':assets['tokenizer_receipt_ref']['sha256'],
                'decoding_sha256':context['runtime']['decoding_sha256'],'source':context['source'],'generation_contract':contract,
                'worker_proof':proof,'process_identity_sha256':proof['worker_identity_sha256'],
                'model_fingerprint_unchanged':True,'rows':1950,'token_journal':str(path),'attempts_journal':str(attempts_path),
                'greedy':GREEDY,'os_terminal_verified':False,'native_qualified':False,'learning_claim':False,
                'model_load_order':model_load_steps(model_name),'classification':'readonly body tokens; independent scoring/terminal/cost consumer required'}
            blob = canonical(value)+b'\n'; require(len(blob) <= policy['receipt_bytes_upper'], 'DEV body receipt cap')
            target = directory/'DEV_BODY_COMPLETE.json'
            with target.open('xb') as stream: require(stream.write(blob)==len(blob),'short DEV receipt'); stream.flush(); os.fsync(stream.fileno())
            with target.open('rb') as stream: require(stream.read(len(blob)+1)==blob,'DEV receipt exact readback')
            reference = {'path':str(target),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
        return {'body_ref':reference,'token_journal':str(path),'attempts_journal':str(attempts_path),'rows':1950,
                'native_qualified':False,'os_terminal_verified':False,'learning_claim':False}
    except BaseException as error: primary = error; raise
    finally:
        model = tokenizer = None
        if torch is not None:
            try:
                import gc
                gc.collect(); torch.cuda.empty_cache()
            except BaseException as failure:
                if primary is not None: raw._accounting(primary,failure); raise primary from failure
                raise
        phase = 'failed' if primary is not None else 'returned'
        try: operation('dev_cleanup',phase,operation_id=identifier,wall_s=time.monotonic()-origin)
        except BaseException as failure:
            if primary is not None: raw._accounting(primary,failure); raise primary from failure
            raise
