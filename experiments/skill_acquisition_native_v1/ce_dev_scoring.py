"""Paid readback of one completed fixed DEV read, using the unchanged scorer.

Completion is supplied by the source-selected supervisor consumer. Saved OS
records are historical evidence; this module neither reopens a PID nor grants
evaluation. Fixtures use the same byte/journal checks but never native admission.
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import ce_qualification_artifacts as raw, ce_execution_verifier as files
from . import ce_dev_runtime as runtime, ce_worker_evidence as worker

SCHEMA = 'CE8-DEV-SCORED-COMPLETION-1'
MAX_JOURNAL, MAX_LINE, MAX_BANK = 256 * raw.CHUNK, 131072, 32 * raw.CHUNK
require, canonical, digest = raw.require, raw.canonical, raw.digest


def _binding(context, proof):
    return {'stage':'dev_scoring','request_sha256':context['request_sha256'],
        'scope_sha256':context['scope_sha256'],'static_recipe_sha256':digest(runtime.GREEDY),
        'code_sha256':context['source']['code_sha256'],
        'worker_identity_sha256':proof['worker_identity_sha256'],'attempt_id':context['read_id']}


def _journal(reader, reference, binding, role, expected, key):
    """Read canonical original row hashes without ResumableJsonl's repair/write."""
    require(reference['bytes'] <= MAX_JOURNAL, 'bounded fixed DEV journal')
    ref = files._decorated(reader, reference, binding, role)
    with reader._overhead('DEV_journal_rows_and_hashes', work_bytes=32*reference['bytes'],
            hash_attempts=1952, hash_bytes=12*reference['bytes']):
        stream=reader._blocks(ref,binding,role); pending=b''; rows=[]; seen=set(); primary=None
        try:
            for block in stream:
                pending+=block
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    require(0<len(line)<=MAX_LINE, 'bounded nonempty DEV journal line')
                    row=raw._json(line);require(type(row) is dict and canonical(row)==line,'canonical original DEV journal row')
                    body=dict(row); recorded=body.pop('row_sha256',None)
                    require(raw._hex(recorded) and digest(body)==recorded,'original DEV row hash mismatch')
                    rid=body.get(key);require(type(rid) is str and rid and rid not in seen,'duplicate DEV journal identity')
                    require(all(body.get(name)==value for name,value in expected.items()),'DEV journal frozen bind differs')
                    seen.add(rid);rows.append(body)
                    require(len(rows)<=1950 if key=='id' else len(rows)<=2,'fixed DEV journal count exceeded')
                require(len(pending)<=MAX_LINE,'bounded DEV journal partial line')
            require(not pending,'partial DEV journal tail retained; cannot score')
        except BaseException as error:primary=error;raise
        finally:
            try:stream.close()
            except BaseException as failure:
                if primary is not None:raw._accounting(primary,failure);raise primary from failure
                raise
        return rows


def _contract(contract):
    require(type(contract) is dict and raw._hex(contract.get('contract_sha256'))
        and digest({k:v for k,v in contract.items() if k!='contract_sha256'})==contract['contract_sha256'],
        'original complete generation contract digest')
    require(contract.get('transformers')=='4.57.1' and type(contract.get('max_new_tokens')) is int
        and contract['max_new_tokens']==128 and contract.get('eos_token_ids')==[151645,151643]
        and contract.get('pad_token_id')==151643 and contract.get('use_model_defaults') is False
        and contract.get('logits_processors')==['_RawStepRecorder']
        and contract.get('stopping_criteria')==['MaxLengthCriteria','EosTokenCriteria']
        and contract.get('greedy')=={'do_sample':False,'num_beams':1,'batch_size':1,'use_cache':True,
            'raw_logits_source':'generate(output_logits=True) float32 step logits, cross-checked bitwise against the pass-through recorder'},
        'unchanged raw argmax greedy contract')


def _decoder(row, attention):
    decoded=row.get('decoder');require(type(decoded) is dict and decoded.get('mode')=='greedy'
        and decoded.get('transformers')=='4.57.1' and decoded.get('attn_implementation')==attention
        and decoded.get('raw_logits_source')=='output_logits' and decoded.get('argmax_replay_verified') is True
        and decoded.get('logits_processors')==['_RawStepRecorder']
        and decoded.get('stopping_criteria')==['MaxLengthCriteria','EosTokenCriteria'],
        'source-bound observed raw argmax decoder')
    expected={'do_sample':False,'num_beams':1,'num_return_sequences':1,'max_new_tokens':128,
        'eos_token_id':[151645,151643],'pad_token_id':151643,'use_cache':True,
        'output_logits':True,'output_scores':False,'return_dict_in_generate':True}
    config=decoded.get('effective_config');require(type(config) is dict
        and all(type(config.get(k)) is type(v) and config[k]==v for k,v in expected.items()),
        'actual effective greedy generation settings')
    require(raw._int(row.get('prompt_tokens'),1) and type(config.get('max_length')) is int
        and config['max_length']==row['prompt_tokens']+128,'effective max_length equals prompt tokens plus128')
    tokens=row.get('completion_ids');prob=row.get('raw_step_logprobs')
    require(type(tokens) is list and 0<len(tokens)<=128 and all(type(t) is int and t>=0 for t in tokens)
        and type(row.get('generated_tokens')) is int and row['generated_tokens']==len(tokens)
        and type(prob) is list and len(prob)==len(tokens) and all(raw._finite(p) and p<=0 for p in prob),
        'complete finite per-token raw logprob accounting')
    eos=[i for i,t in enumerate(tokens) if t in (151645,151643)]
    require((row.get('stop_reason')=='eos' and eos==[len(tokens)-1]) or
        (row.get('stop_reason')=='length' and not eos and len(tokens)==128),'first terminal EOS or exact128 token cap')
    require(type(row.get('response')) is str and type(row.get('body_special_ids')) is list
        and type(row.get('body_unknown_ids')) is list and raw._int(row.get('prompt_tokens'),1)
        and raw._finite(row.get('wall_s')) and row['wall_s']>=0
        and raw._hex(row.get('prompt_ids_sha256')) and raw._hex(row.get('render_sha256')),'raw rendered/token/response metadata')


def _body(context, body, completion, fixture):
    runtime.validate_context(context)
    proof=body['worker_proof'];require(type(proof) is dict and proof.get('schema')==worker.PROOF
        and proof.get('request_sha256')==context['request_sha256'] and proof.get('scope_sha256')==context['scope_sha256']
        and body.get('process_identity_sha256')==proof.get('worker_identity_sha256')
        and raw._hex(proof.get('worker_identity_sha256')),'one original DEV worker request/scope identity')
    require(digest(worker.worker_identity_preimage(proof['os_identity'],proof['sources'],context['request_sha256'],context['scope_sha256']))
        ==proof['worker_identity_sha256']
        and {name:ref['sha256'] for name,ref in proof['sources']['source_files'].items()}==context['source']['code_sha256'],
        'historical worker stable OS/raw source identity changed')
    model_name,index=next((m,i) for r,m,i in runtime.RUNS if r==context['read_id'])
    expected={'schema':runtime.READ,'status':'DEV_BODY_COMPLETE','read_id':context['read_id'],'model_name':model_name,
        'read_index':index,'scope_sha256':context['scope_sha256'],'model_sha256':context['model'][model_name+'_sha256'],
        'tokenizer_sha256':context['model']['tokenizer_receipt_ref']['sha256'],
        'decoding_sha256':context['runtime']['decoding_sha256'],'source':context['source'],'rows':1950,
        'greedy':runtime.GREEDY,'model_load_order':runtime.model_load_steps(model_name)}
    require(all(body.get(k)==v for k,v in expected.items()),'completed DEV body frozen model/runtime/source/plan differs')
    expected_state=context['model']['Supa_state_fingerprint_sha256'] if model_name=='Supa' else context['model'][model_name+'_sha256']
    require(body.get('deployed_state_fingerprint_sha256')==expected_state
        and body.get('model_fingerprint_unchanged') is True and body.get('os_terminal_verified') is False
        and body.get('native_qualified') is False and body.get('learning_claim') is False,'readonly body identity and honest unverified OS classification')
    for key in ('read_id','model_name','read_index','scope_sha256','model_sha256','tokenizer_sha256','decoding_sha256','greedy','source'):
        require(completion.get(key)==body[key],'source completion differs from original DEV body: '+key)
    require(completion.get('completed_pair_verified') is (not fixture)
        and completion.get('execution_reverified') is False,'source-selected terminal/cost dependency required, no fresh OS claim')
    if fixture:require(completion.get('synthetic_pair_consistent') is True,'explicit synthetic completion dependency')
    else:require(type(completion.get('service_replays')) is dict and completion['service_replays']
        and type(completion.get('completion_ref')) is dict and type(completion.get('cost_seal_ref')) is dict,
        'actual original ended service/completion cost pair')
    _contract(body['generation_contract'])
    bind={k:body[k] for k in ('schema','read_id','model_name','read_index','model_sha256','deployed_state_fingerprint_sha256',
        'tokenizer_sha256','decoding_sha256','scope_sha256','rows','greedy','process_identity_sha256')}
    bind.update(generation_contract_sha256=body['generation_contract']['contract_sha256'],source_sha256=digest(context['source']))
    return proof,bind


def _banks(scope, reader, binding):
    with reader._overhead('DEV_unchanged_scorer_import',work_bytes=64*raw.CHUNK,
            hash_attempts=32,hash_bytes=64*raw.CHUNK):
        _bounded_scorer_sources(scope)
        from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer as scorer
        require(scorer.code_sha256()==scope['source']['scorer_code_sha256']
            and digest(scope['source']['scorer_code_sha256'])==scope['source']['scorer_bundle_sha256']
            and scorer.SOURCE_LOCK_SHA256==scope['source']['source_lock_sha256'],'frozen unchanged scorer/source lock')
    manifest=_metadata_json(reader,scope['metadata_refs']['dataset_manifest'],binding,'DEV_dataset_manifest')
    require(scope['metadata_refs']['dataset_manifest']['sha256']==scope['manifest_sha256']==scorer.MANIFEST,'literal canonical DEV manifest')
    banks={}
    for role in ('dev','guard_dev'):
        key='canonical_'+role;ref=scope['metadata_refs'][key]
        require(ref['bytes']<=MAX_BANK and ref['sha256']==manifest['files']['canonical/'+role+'.jsonl']['sha256'],
            'fixed canonical DEV/guard file binding')
        decorated=files._decorated(reader,ref,binding,key)
        blob=b''.join(reader._blocks(decorated,binding,key))
        with reader._overhead('DEV_bank_parse',work_bytes=16*len(blob)):
            rows=[]
            for line in blob.splitlines():
                require(0<len(line)<=MAX_LINE,'bounded canonical DEV row');row=raw._json(line)
                require(row.get('role')==role,'only DEV/guard rows');rows.append(row)
            require(len(rows)==manifest['counts'][role] and len({r['id'] for r in rows})==len(rows),'complete canonical DEV identity set')
        banks[role]=scorer.Bank(rows,role=role,bank_sha256=ref['sha256'],manifest_sha256=scope['manifest_sha256'])
    return scorer,banks


def _bounded_scorer_sources(scope):
    checkout=Path(__file__).resolve().parents[2];code=scope['source']['scorer_code_sha256']
    require(type(code) is dict and len(code)==14,'exact unchanged fourteen-file scoring bundle')
    for name in (*code,'experiments/maxrl_comp002_pretraining_v1/source_lock.json'):
        relative=Path(name);path=checkout/relative
        require(not relative.is_absolute() and '..' not in relative.parts and not path.is_symlink()
            and path.is_file() and path.resolve().is_relative_to(checkout)
            and 0<path.stat().st_size<=raw.MAX_JSON,'bounded paid unchanged scoring source file')


def _metadata_json(reader, reference, binding, role):
    # Governing dataset metadata has an existing pretty-printed raw encoding.
    # Hash-bound bytes are checked without replacing that original encoding.
    require(reference['bytes']<=raw.MAX_JSON,'bounded original DEV metadata JSON')
    decorated=files._decorated(reader,reference,binding,role)
    blob=b''.join(reader._blocks(decorated,binding,role))
    with reader._overhead('DEV_original_metadata_JSON',work_bytes=16*len(blob)):
        value=raw._json(blob);require(type(value) is dict,'original DEV metadata object')
    return value


def _score(completion, *, reader, bank_reader, outer_preflight, fixture):
    require(type(completion) is dict and callable(outer_preflight),'verified source completion and priced scoring preflight')
    refs={name:completion[name+'_ref'] for name in ('context','body','token_journal','attempts_journal')}
    for ref in refs.values():files._reference(ref)
    require(refs['context']['bytes']<=raw.MAX_JSON and refs['body']['bytes']<=raw.MAX_JSON
        and all(refs[k]['bytes']<=MAX_JOURNAL for k in ('token_journal','attempts_journal')),'finite DEV completed input sizes')
    binding=completion['verification_binding'];raw.binding(binding)
    outer_preflight('completed_DEV_scoring',{'schema':SCHEMA,'rows':1950,'banks':list(runtime.BANKS),
        'journal_byte_upper':MAX_JOURNAL,'line_byte_upper':MAX_LINE,'max_new_tokens':128,
        'source_completion_ref':completion.get('completion_ref'),'fresh_OS_observation':False})
    context=files._read_json(reader,refs['context'],binding,'DEV_context')
    body=files._read_json(reader,refs['body'],binding,'DEV_body')
    with reader._overhead('DEV_source_body_contract',work_bytes=128*raw.MAX_JSON,hash_attempts=256,hash_bytes=128*raw.MAX_JSON):
        proof,bind=_body(context,body,completion,fixture)
        require(binding['worker_identity_sha256']==proof['worker_identity_sha256']
            and binding['request_sha256']==context['request_sha256'] and binding['scope_sha256']==context['scope_sha256']
            and binding['code_sha256']==context['source']['code_sha256'],'exact completion/raw worker source binding')
        require(body['token_journal']==refs['token_journal']['path'] and body['attempts_journal']==refs['attempts_journal']['path'],
            'completed body original journal paths')
    rows=_journal(reader,refs['token_journal'],binding,'DEV_tokens',bind,'id')
    attempts=_journal(reader,refs['attempts_journal'],binding,'DEV_attempts',bind,'event')
    process={'process_identity_sha256':proof['worker_identity_sha256'],'os_identity':proof['os_identity']}
    require(len(attempts)==2 and attempts[0].get('event')==proof['worker_identity_sha256']+':start'
        and attempts[0].get('kind')=='start' and attempts[0].get('worker_proof')==proof
        and type(attempts[0].get('resume_index')) is int and attempts[0]['resume_index']==0
        and attempts[1].get('event')==proof['worker_identity_sha256']+':end' and attempts[1].get('kind')=='end'
        and attempts[1].get('outcome')=='COMPLETE' and type(attempts[1].get('rows')) is int
        and attempts[1]['rows']==1950 and all(a.get('process')==process for a in attempts),
        'single original fresh complete DEV attempt; no retries or process churn')
    scope=context['scope'];planned=[(name,rid) for name in runtime.BANKS for rid in scope['banks'][name]['ids']]
    require(len(rows)==1950 and [(r.get('bank'),r.get('id')) for r in rows]==planned,'all1950 exact sorted bank rows once')
    with reader._overhead('DEV_tokens_process_decoder',work_bytes=32*refs['token_journal']['bytes']):
        for index,row in enumerate(rows):
            bank=scope['banks'][row['bank']]
            require(type(row.get('index')) is int and row['index']==index and row.get('process')==process
                and row.get('process_identity_sha256')==proof['worker_identity_sha256']
                and row.get('parent_id')==bank['parent_by_id'][row['id']]
                and row.get('prompt_hash')==bank['prompt_hash_by_id'][row['id']], 'row plan/parent/prompt/current worker differs')
            _decoder(row,context['runtime']['decoding_preimage']['attn_implementation'])
    scope_receipt=_metadata_json(reader,completion['scope_receipt_ref'],binding,'DEV_RSP_scope_receipt')
    with reader._overhead('DEV_scope_receipt_digest',work_bytes=12*raw.MAX_JSON,hash_attempts=1,hash_bytes=12*raw.MAX_JSON):
        require(digest(scope_receipt)==scope['dev_scope_receipt_sha256'],'frozen DEV R/S/P scoped receipt digest')
    scorer,banks=_banks(scope,bank_reader,binding)
    reads={};model=body['model_name'];run={k:body[k] for k in ('model_sha256','tokenizer_sha256','decoding_sha256')}
    for name in runtime.BANKS:
        selected=[r for r in rows if r['bank']==name];bank=scope['banks'][name]
        responses=[{key:r[key] for key in scorer.RESPONSE_FIELDS} for r in selected]
        with bank_reader._overhead('DEV_unchanged_scoped_scoring',work_bytes=512*raw.CHUNK,
                hash_attempts=128,hash_bytes=512*raw.CHUNK):
            _bounded_scorer_sources(scope)
            receipt=scorer.score_scope(banks[bank['role']],bank['scoring_scope'],responses,run,
                scope_receipt=None if bank['role']=='guard_dev' else scope_receipt,
                expected_code_bundle=scope['source']['scorer_bundle_sha256'],
                expected_scope_receipt_sha256=bank['scope_receipt_sha256'])
        reads[name]={'schema':runtime.READ,'bank':name,'model_name':model,
            'process_identity_sha256':proof['worker_identity_sha256'],'journal_rows':selected,'scoring_receipt':receipt}
    return {'schema':SCHEMA,'read_id':body['read_id'],'model_name':model,'read_index':body['read_index'],
        'scope_sha256':context['scope_sha256'],'rows':1950,'reads':reads,'completion_ref':completion.get('completion_ref'),
        'cost_seal_ref':completion.get('cost_seal_ref'),'original_input_refs':refs,
        'completion_pair_verified':not fixture,'historical_terminal_bound':not fixture,
        'synthetic_pair_consistent':fixture,'execution_reverified':False,'native_qualified':False,'learning_claim':False,
        'interpretation':'unchanged strict/content scoped scoring; raw argmax provenance is original source execution, not a fresh logits replay'}


def score_completed_dev(selection, *, reader, bank_reader, outer_preflight):
    """The fixed supervisor source pin refuses before scoring reads by default."""
    require(type(selection) is dict,'source selected completed DEV pair absent; scorer closed')
    from . import ce_dev_supervisor as supervisor
    completion=supervisor.verify_completed_dev(selection,reader=reader,outer_preflight=outer_preflight)
    return _score(completion,reader=reader,bank_reader=bank_reader,outer_preflight=outer_preflight,fixture=False)


def score_fixture_read(completion, *, reader, bank_reader, outer_preflight):
    """Explicit metadata/token fixtures cannot satisfy native completion."""
    return _score(completion,reader=reader,bank_reader=bank_reader,outer_preflight=outer_preflight,fixture=True)
