"""Read-only source-selected consumption of completed CE375 body/cost pairs.

This replays retained costs and byte links. Historical terminal records are
source-selected evidence, never a fresh OS observation or a learning claim.
Actual DEV permission remains a separately authorized source entry decision.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import MappingProxyType

from . import ce_qualification_artifacts as raw, ce_execution_verifier as files
from . import ce_qualified_receipt as generic, ce_finalizer_accounting as costs
from . import ce_training_supervisor as supervisor, ce_training_runtime as runtime
from . import ce_delta_artifacts as delta, ce_worker_evidence as worker
from . import ce_service_bootstrap_verifier as service_replay

SCHEMA = 'CE8-COMPLETED-TRAINING-SELECTION-1'
RESULT = 'CE8-COMPLETED-TRAINING-PAIR-1'
_SOURCE_SELECTION = _SOURCE_SELECTION_SHA256 = None
REFS = frozenset(('completion','receipt','preclose','header','journal','seal','body','CE_header','CE_journal',
                  'scope','ledger_spec','anchors','runtime','source','donor','service_spec'))
require, canonical, digest = raw.require, raw.canonical, raw.digest


def selected_training():
    require(type(_SOURCE_SELECTION) is MappingProxyType and raw._hex(_SOURCE_SELECTION_SHA256),
            'completed training source pin absent; consumer closed')
    return _SOURCE_SELECTION


def _shape(selection):
    require(type(selection) is dict and set(selection) == {'schema','owned_root','binding','references','finalizer_specification',
            'runtime_sha256','anchors_sha256'} and selection['schema'] == SCHEMA, 'exact completed training source selection')
    raw.binding(selection['binding'])
    require(selection['binding']['stage'] == 'training_supervisor'
            and selection['binding']['attempt_id'] == 'independent-production-completion', 'exact original training completion binding')
    require(type(selection['owned_root']) is str and Path(selection['owned_root']).is_absolute()
            and raw._hex(selection['runtime_sha256']) and raw._hex(selection['anchors_sha256']), 'source completed root/runtime/anchor pins')
    refs = selection['references']; require(type(refs) is dict and set(refs) == REFS, 'all completed training bytes and cost refs')
    for name, ref in refs.items():
        files._reference(ref)
        maximum = 8 * 1024**3 if name == 'donor' else supervisor.MAX_JOURNAL if name in ('journal','CE_journal') else raw.MAX_JSON
        require(set(ref) == {'path','sha256','bytes'} and ref['bytes'] <= maximum, 'bounded exact completed training ref: '+name)
    require(refs['scope']['sha256'] == selection['binding']['scope_sha256'], 'original raw production scope SHA')
    spec = costs.validate_specification(selection['finalizer_specification'],selection['owned_root'])
    for name, filename in (('receipt','TRAINING_COMPLETION.json'),('preclose','TRAINING_PRE_CLOSE_REPLAY.json'),
                           ('header','FINALIZER_HEADER.json'),('journal','FINALIZER.jsonl'),('seal','FINALIZER_SEAL.json')):
        require(Path(refs[name]['path']) == Path(spec['owned_directory'])/filename, 'actual original finalizer filename: '+name)
    return selection


def _cost_pair(selection, header, seal, evidence, reader, fixture):
    refs, binding, spec = selection['references'],selection['binding'],selection['finalizer_specification']
    shadow,state = generic._replay(header,refs['header'],refs['journal'],selection,reader,fixture=fixture)
    expected = {'schema':costs.SCHEMA,'request_sha256':binding['request_sha256'],'scope_sha256':binding['scope_sha256'],
        'header_ref':refs['header'],'head':state['head'],'permanent_reserved_bounds':spec['self_bounds'],
        'consumed_declared_bounds':shadow.counts,'pending':[],'failed':[],'original_origin':state['origin'],
        'last_monotonic':state['last'],'fatal':False,'closed':False,'receipt_refs':state['receipts'],'native_qualified':False}
    require(seal == expected, 'training cost seal refund/pending/clock differs from original raw chain')
    require(type(evidence) is dict and evidence.get('schema') == costs.SCHEMA and evidence.get('owner_closed') is True
            and evidence.get('native_qualified') is False and evidence.get('receipt_refs') == state['receipts']
            and evidence.get('permanent_reserved_bounds') == spec['self_bounds']
            and all(evidence.get(key+'_ref') == refs[key] for key in ('header','journal','seal')), 'original closed cost/header/journal/seal pairing')
    shadow._charge({'operation_attempts_upper':1,'io_write_bytes_upper':refs['seal']['bytes'],'metadata_bytes_upper':refs['seal']['bytes']})
    for name in ('header','journal','seal'):
        size=refs[name]['bytes']+1
        shadow._charge({'operation_attempts_upper':1,'io_read_bytes_upper':size,'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':size})
    consumed=generic._counts(evidence['consumed_declared_bounds'])
    require(all(consumed[key] == shadow.counts[key] for key in costs.COST_FIELDS if key != 'wall_seconds_upper'),
            'original finalizer close/readback costs refunded or changed')
    suffix = consumed['wall_seconds_upper']-shadow.counts['wall_seconds_upper']
    require(generic._finite(suffix), 'finalizer inspection elapsed cost refunded')
    shadow._charge({'wall_seconds_upper':suffix})
    require(state['last']+suffix-state['origin'] <= spec['physical_wall_seconds_upper']
            and state['last']+suffix-state['suffix_origin'] <= spec['finalization_bounds']['wall_seconds_upper'], 'actual retained final cost suffix deadline')
    require(state['receipts'].get('TRAINING_COMPLETION.json') == refs['receipt']
            and state['receipts'].get('TRAINING_PRE_CLOSE_REPLAY.json') == refs['preclose'], 'original returned receipt/preclose writes missing')
    return shadow.counts,state


def _service_pair(selection, observed, state, reader, preflight, fixture):
    """Read both original ended service ledgers; DTO totals are not evidence."""
    binding=selection['binding']; preclose=observed['preclose']; receipt=observed['receipt']
    specs={'worker':observed['service_spec']['bootstrap']}
    specs['supervisor']=dict(copy.deepcopy(specs['worker']),
        owned_directory=str(Path(specs['worker']['owned_directory']).parent/'supervisor-service-bootstrap'))
    ends=[op for op in state['operations'].values() if op['kind']=='finalizer_service_end']
    require(len(ends)==1,'one original supervisor service-end event')
    contracts={'worker':preclose['worker_service_refs'],'supervisor':ends[0]['result']['evidence']}
    expected={'worker':preclose['worker_service_replay'],'supervisor':receipt['supervisor_service_replay']}
    ctx={'mode':'synthetic_fixture' if fixture else 'registered_scope','request_sha256':binding['request_sha256'],'verification_binding':binding}
    results={}
    for name in ('worker','supervisor'):
        refs=contracts[name]
        results[name]=service_replay.verify_service_bootstrap(ctx,specs[name],refs['header_ref'],refs['journal_ref'],refs['seal_ref'],
            reader=reader,outer_preflight=lambda kind,contract:preflight('completed_training_service',contract))
        require(results[name]==expected[name],'original ended '+name+' service cost replay differs from retained DTO')
    return results


def verify_donor_bytes(reader, reference, binding, inventory):
    """Only exact bytes/layout digests; no220M-scalar arithmetic replay."""
    wrapped=files._decorated(reader,reference,binding,'completed_training_donor')
    expected={row['name']:row['shape'] for row in inventory}
    require(len(expected)==len(inventory) and expected,'complete distinct saved donor inventory')
    prefix=b''; target=None; data_hash=hashlib.sha256(); count=0; ready=False
    with reader._overhead('completed_donor_header_and_data_digest',work_bytes=32*raw.CHUNK,
                         hash_attempts=1,hash_bytes=reference['bytes']):
        stream=reader._blocks(wrapped,binding,'completed_training_donor');primary=None
        try:
            for block in stream:
                if ready:data_hash.update(block);count+=len(block);continue
                prefix+=block
                if target is None and len(prefix)>=8:
                    size=int.from_bytes(prefix[:8],'little');require(0<size<=raw.CHUNK,'bounded saved donor header')
                    target=8+size
                if target is None or len(prefix)<target:continue
                header=raw._json(prefix[8:target]);require(type(header) is dict and set(header)==set(expected),'saved donor tensor names')
                end=0
                for name,row in sorted(header.items(),key=lambda item:item[1]['data_offsets'][0]):
                    require(type(row) is dict and set(row)=={'dtype','shape','data_offsets'} and row['dtype']=='F32'
                            and row['shape']==expected[name] and all(type(n) is int and n>0 for n in row['shape']), 'saved donor exact FP32 shape')
                    size=4*math.prod(row['shape'])
                    require(row['data_offsets']==[end,end+size],'saved donor contiguous exact data offsets')
                    end+=size
                require(reference['bytes']==target+end,'saved donor exact header/data byte count')
                data=prefix[target:];data_hash.update(data);count+=len(data);prefix=b'';ready=True
            require(ready and count==reference['bytes']-target,'saved donor complete data section')
        except BaseException as error:primary=error;raise
        finally:
            try:stream.close()
            except BaseException as error:
                if primary is not None:raw._accounting(primary,error);raise primary from error
                raise
    return {'file_sha256':reference['sha256'],'data_section_sha256':data_hash.hexdigest(),'data_bytes':count,
            'numel':count//4,'tensors':len(expected),'numeric_recomputed':False}


def _verify(selection, reader, preflight, *, fixture, source_pin=None):
    require(type(reader) is raw.RawEvidenceReader and callable(preflight), 'bounded paid completed training consumer')
    preflight('completed_training_pair', {'reader_bounds':dict(reader.remaining),'max_events':generic.MAX_EVENTS,
        'max_line_bytes':costs.MAX_JSON,'coverage':['raw files','finalizer and CE chains','donor hash/norm','linkage','verification wall']})
    with reader._overhead('completed_training_fixed_selection',work_bytes=128*raw.MAX_JSON,hash_attempts=128,hash_bytes=128*raw.MAX_JSON):
        selection=_shape(copy.deepcopy(selection))
        require(fixture or digest(selection) == source_pin, 'immutable completed training literal SHA changed')
        require(Path(reader.root) == Path(selection['owned_root']), 'exact original source-selected output root')
    refs,binding=selection['references'],selection['binding']
    observed={name:files._read_json(reader,reference,binding,'completed_training_'+name) for name,reference in refs.items()
              if name not in ('journal','CE_journal','donor')}
    with reader._overhead('completed_training_source_pair_links',work_bytes=128*raw.MAX_JSON,hash_attempts=256,hash_bytes=128*raw.MAX_JSON):
        completion,receipt,preclose=observed['completion'],observed['receipt'],observed['preclose']
        require(completion.get('schema') == supervisor.SCHEMA and completion.get('status') == 'TRAINING_BODY_EXIT_COSTS_VERIFIED'
                and completion.get('completed') is True and type(completion.get('updates')) is int and completion['updates'] == 375
                and completion.get('native_qualified') is False and completion.get('learning_claim') is False
                and completion.get('receipt_ref') == refs['receipt'], 'actual completed body and owned cost pair; flags alone cannot complete')
        require(receipt.get('schema') == supervisor.SCHEMA and receipt.get('status') == 'BODY_EXIT_VERIFIED_REQUIRES_COST_SEAL'
                and receipt.get('updates') == 375 and receipt.get('body_ref') == refs['body']
                and receipt.get('independent_replay_ref') == refs['preclose'] and receipt.get('finalizer_header_ref') == refs['header']
                and receipt.get('expected_finalizer_seal_path') == refs['seal']['path']
                and receipt.get('cost_owner_binding_sha256') == digest(observed['header'])
                and all(receipt.get(key) == binding[key] for key in ('request_sha256','scope_sha256'))
                and receipt.get('native_qualified') is False and receipt.get('learning_claim') is False, 'fixed receipt/body/source/seal pair')
    consumed,state=_cost_pair(selection,observed['header'],observed['seal'],completion['cost_evidence'],reader,fixture)
    with reader._overhead('completed_training_terminal_and_source_links',work_bytes=128*raw.MAX_JSON,hash_attempts=256,hash_bytes=128*raw.MAX_JSON):
        body,scope,artifact=observed['body'],observed['scope'],observed['ledger_spec']
        require(body.get('schema') == runtime.PUBLICATION and body.get('status') == 'TRAINING_BODY_COMPLETE'
                and preclose.get('binding') == binding and preclose.get('body_ref') == refs['body']
                and preclose.get('original_journal_refs') == body['raw_refs']
                and body['raw_refs'] == {'donor':refs['donor'],'header':refs['CE_header'],'journal':refs['CE_journal']}, 'original training body/journal/donor links')
        require(all(body.get(key) == binding[key] for key in ('request_sha256','scope_sha256','static_recipe_sha256'))
                and scope['native_ledger_ref']['sha256'] == refs['ledger_spec']['sha256']
                and artifact['bindings']['source'] == observed['source']['code_sha256'] == binding['code_sha256']
                and artifact['bindings']['runtime'] == observed['runtime']['decoding_preimage']
                and digest(artifact['bindings']['recipe']) == binding['static_recipe_sha256']
                and digest(observed['runtime']) == selection['runtime_sha256'] and digest(observed['anchors']) == selection['anchors_sha256'],
                'original source/runtime/recipe/common-anchor changed')
        proof=preclose['actual_worker_terminal']
        require(proof.get('schema') == worker.TERMINAL and all(proof.get(key) == binding[key] for key in ('request_sha256','scope_sha256','worker_identity_sha256'))
                and type(proof.get('exit_code')) is int and proof['exit_code'] == 0 and proof.get('status_file_used') is False,
                'source-pinned past original terminal record; not fresh OS evidence')
        operations=state['operations']; terminal=[op for op in operations.values() if op['kind']=='finalizer_terminal_observation']
        ended=[op for op in operations.values() if op['kind']=='finalizer_service_end']
        writes={Path(op['result']['reference']['path']).name:op for op in operations.values() if op['kind']=='finalizer_json_write'}
        require(len(terminal)==len(ended)==1 and terminal[0]['result']['terminal_proof']==proof
                and terminal[0]['returned_seq'] < writes['TRAINING_PRE_CLOSE_REPLAY.json']['started_seq']
                < ended[0]['started_seq'] < ended[0]['returned_seq'] < writes['TRAINING_COMPLETION.json']['started_seq'],
                'past observed terminal/preclose/source-end/receipt causal linkage')
        snapshot=receipt['supervisor_service_replay']['derived_snapshot']; suffix=receipt['supervisor_terminal_suffix']
        require(receipt['supervisor_service_replay']['recorded_clock_ended'] is True and snapshot['clock_ended'] is True
                and not snapshot['pending'] and snapshot['fatal'] is False and ended[0]['result']['evidence']['snapshot']==snapshot
                and snapshot['request_sha256']==binding['request_sha256']
                and ended[0]['result']['observed_before'] <= snapshot['last_monotonic'] <= ended[0]['result']['observed_after']
                and suffix['recorded_end']==snapshot['last_monotonic'] and suffix['observed_after']==ended[0]['result']['observed_after']
                and suffix['observed_suffix_seconds_upper']==suffix['observed_after']-suffix['recorded_end']
                and 0 <= suffix['observed_suffix_seconds_upper'] <= suffix['suffix_upper_seconds'], 'original supervisor service/clock closure differs')
    ce_state=supervisor.replay_journal(artifact,refs['CE_header'],refs['CE_journal'],body['journal_snapshot'],binding,
        reader=reader,preflight=lambda name,contract:preflight('completed_training_CE_journal',contract))
    runtime.validate_completion(ce_state.snapshot(),body['donor'],{'bindings':{'anchors':observed['anchors']}})
    checkpoints=supervisor.validate_checkpoint_schedule(ce_state,scope['execution'])
    require(ce_state.current_invocation is None and ce_state.invocations[body['invocation_id']]['outcome']=='COMPLETE'
            and body['donor']['final_master_sha256']==ce_state.progress['history'][-1]['final_policy_id']
            and preclose['derived_journal']==ce_state.snapshot() and preclose['checkpoint_saves']==checkpoints,
            'exact CE375 final history/seven saved checkpoint source readback')
    services=_service_pair(selection,observed,state,reader,preflight,fixture)
    saved=preclose['saved_donor']
    verified_bytes=verify_donor_bytes(reader,refs['donor'],binding,scope['native_backend']['parameter_inventory'])
    require(all(verified_bytes[key]==saved[key] for key in ('data_section_sha256','numel','tensors'))
            and all(saved[key]==body['donor'][key] for key in ('data_section_sha256','state_sha256','numel','tensors'))
            and raw._hex(saved.get('state_sha256')) and generic._finite(saved.get('delta_norm'))
            and saved['delta_norm']==body['donor'].get('delta_norm')
            and body['donor']['file_sha256']==refs['donor']['sha256'] and body['donor']['path']==refs['donor']['path']
            and raw._hex(body['donor'].get('deployed_arm_sha256')), 'actual saved donor bytes/hash/endpoint differ')
    donor={'path':refs['donor']['path'],'sha256':refs['donor']['sha256'],'bytes':refs['donor']['bytes'],
           'data_sha256':saved['data_section_sha256'],'state_sha256':saved['state_sha256'],
           'deployed_arm_sha256':body['donor']['deployed_arm_sha256']}
    return {'schema':RESULT,'completed_pair_verified':not fixture,'synthetic_pair_consistent':fixture,'updates':375,
        'checkpoint_boundaries':sorted({item['C'] for item in checkpoints}),'static_recipe_sha256':binding['static_recipe_sha256'],
        'source_code_sha256':copy.deepcopy(binding['code_sha256']),'runtime':copy.deepcopy(observed['runtime']),
        'anchors':copy.deepcopy(observed['anchors']),'donor_ref':donor,'final_master_sha256':body['donor']['final_master_sha256'],
        'warm_common_sha256':observed['anchors']['common_anchor_sha256'],
        'composition_order':['base_BF16_to_FP32','add_warm_FP32','add_saved_R_FP32'],
        'completion_ref':refs['completion'],'receipt_ref':refs['receipt'],'finalizer_seal_ref':refs['seal'],
        'derived_finalizer_costs':copy.deepcopy(consumed),'original_service_replays':services,
        'saved_donor_byte_verification':verified_bytes,'original_saved_donor_results':copy.deepcopy(saved),
        'historical_terminal_bound':True,'execution_reverified':False,
        'native_qualified':False,'learning_claim':False,'evaluation_authorized':False,
        'classification':'source-selected CE375 saved body and actual ended-cost linkage; separate DEV authorization required'}


def verify_completed_training(selection, *, reader, outer_preflight):
    require(type(_SOURCE_SELECTION) is MappingProxyType and selection is _SOURCE_SELECTION and raw._hex(_SOURCE_SELECTION_SHA256),
            'exact immutable completed training source selection required; consumer closed')
    return _verify(dict(selection),reader,outer_preflight,fixture=False,source_pin=_SOURCE_SELECTION_SHA256)


def verify_fixture_pair(selection, *, reader, outer_preflight):
    """Explicit source-free raw-cost/metadata fixture; never admits evaluation."""
    return _verify(selection,reader,outer_preflight,fixture=True)
