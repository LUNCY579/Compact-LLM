"""Lightweight stdlib-only reader rejection/geometry/termination controls."""
import copy
import contextlib
import hashlib
import struct
import math
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import reader
from experiments.rineya_centralized_reasoning_system_v1 import reader_execution as execution


def run_fixture(cfg):
    """Pure CPU protocol identities, with no process or host observation."""
    binding=execution.binding(cfg,reader.sha(reader.CONTRACT),'a'*40)
    output=str((reader.PACKAGE/'results/CPU_SYNTHETIC_INTERFACE_NOT_NATIVE').resolve())
    payload={'run_id':'CPU_SCHEMA_ONLY','source_commit':binding['source_commit'],'contract_sha256':binding['contract_sha256'],
             'source_review':{'path':str((reader.PACKAGE/'results/CPU_SOURCE_REVIEW.json').resolve()),'sha256':'e'*64},'output':output}
    run={'run_id':'CPU_SCHEMA_ONLY','job_key':'CPU_SCHEMA_JOB','epoch':'CPU_SCHEMA_EPOCH','owner':'CPU_SCHEMA_ONLY','fence':1,
         'host_db':str(execution.host_state_path().resolve()),'reserved_at':100.0,'reserved_seconds':600.0,
         'registration_payload':payload,'request_hash':reader.digest({'payload':payload,'epoch':'CPU_SCHEMA_EPOCH','dependencies':[]})}
    run['command_sha256']=reader.digest(execution.registered_worker_command(cfg,run))
    binaries={p.replace('\\','/').casefold():h for p,h in cfg['runtime_file_sha256'].items() if p.lower().endswith('.exe')}
    members=[{'pid':10+i,'creation_time':100.0+i,'executable':exe,'executable_sha256':binaries[exe.replace('\\','/').casefold()],
              'exact_owned_job_member':True} for i,exe in enumerate((cfg['runtime']['executable'],cfg['runtime_worker_executable']))]
    launcher={'pid':10,'creation_time':'100.0','command_hash':run['command_sha256']}
    context={'binding':binding,'run_identity':run,'worker_identity':launcher,'runtime_worker_identity':members[1]}
    transfer={'own_job_query_handle':100,'query_handle_target_pid':11,'query_handle_target_creation_time':101.0,
              'duplicate_access_mask':4,'duplicate_inheritable':False}
    go={'contract_sha256':binding['contract_sha256'],'owner':run['owner'],'fence':1,'supervisor_pid':9,'worker_tree_pid':10,
        **transfer,**{k:v for k,v in context.items() if k!='binding'}}
    return context,members,transfer,go


def completion_fixture(cfg):
    """Synthetic CPU interface witness, never a native qualification artifact."""
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting
    context,_,_,_=run_fixture(cfg);events=[]
    def emit(kind, **details):events.append({'kind':kind,'time':100.0+len(events)*.0005,**context,**details})
    calls=CallAccounting(emit,maximum_optimizer_calls=0)
    calls.call('model_construct',lambda:None,stage='reader_construct')
    def copies():
        for name in cfg['model_tensor_names']:calls.call('tensor_copy',lambda:None,stage='tensor_copy:'+name)
    calls.call('checkpoint_load',copies,stage='reader_load')
    long_generated=[151645];plan=reader.actual_forward_plan(cfg,long_generated);records=[];mask_cache={}
    for expected in plan:
        ids,keep,stage=expected['input_ids'],expected['logits_to_keep'],expected['stage'];length=len(ids)
        calls.call('forward',lambda:None,stage=stage,input_positions=length)
        if length not in mask_cache:mask_cache[length]=execution._mask_sha(length)
        token,lp,logit=151645,-.1,1.0
        if stage.startswith('historical_') and 'bulk' not in stage:
            index=int(stage.rsplit('_',1)[-1]);token=cfg['historical_generated_ids'][index];lp=cfg['historical_online_logprobs'][index];logit=cfg['historical_selected_raw_logits'][index]
        logits_sha=cfg['historical_bulk_logits_sha256'] if stage=='historical_bulk_DIAGNOSTIC_ONLY' else 'a'*64
        hidden_sha='b'*64
        if stage.startswith('max576_mutate_'):logits_sha=reader.digest([stage,'CPU_logits']);hidden_sha=reader.digest([stage,'CPU_hidden'])
        record={'stage':stage,'input_ids':ids,'input_ids_sha256':reader.digest(ids),'forward_call_id':calls.last_call_id,
                'input_length':length,'logits_to_keep':keep,'logits_shape':[1,keep,151936],'hidden_shape':[1,keep,1024],
                'raw_logits_sha256':logits_sha,'raw_hidden_sha256':hidden_sha,'argmax_id':token,'lowest_id_argmax_verified':True,
                'selected_raw_logit':logit,'selected_model_logprob':lp,
                'witnesses':{'mask':{'shape':[1,1,length,length],'dtype':'bfloat16','raw_sha256':mask_cache[length],
                                     'rows':length,'columns':length,'allowed_uint16':0,'forbidden_uint16':65407,
                                     'allowed_condition':'column <= row','byte_order':'little_endian'},
                             'positions':{'shape':[1,length],'dtype':'int64','ids':[list(range(length))],
                                          'raw_sha256':hashlib.sha256(struct.pack('<'+str(length)+'q',*range(length))).hexdigest()}}}
        records.append(record);emit('reader_forward_measurement',record=record)
    by_stage={r['stage']:r for r in records};rollouts={}
    for label,generated in [('historical',cfg['historical_generated_ids']),('long',long_generated)]:
        bundle={}
        for kind in ['owned','stock','replay']:
            steps=[by_stage[label+'_owned_owned_step_'+str(i) if kind=='owned' else label+'_'+kind+'_'+str(i)] for i in range(len(generated))]
            bundle[kind]=steps if kind=='replay' else {'generated_ids':generated,'steps':steps,'stop_reason':'eos',
                'terminal_eos_id':generated[-1],'raw_text':'CPU_SCHEMA_ONLY','visible_text':'CPU_SCHEMA_ONLY',
                'model_logprobs':[r['selected_model_logprob'] for r in steps]}
        bundle['stock']['policy']={'processors':[],'stops':['MaxLengthCriteria','EosTokenCriteria']};rollouts[label]=bundle
    blocks=[]
    for index,ceiling in enumerate(reader.qualifier_plan(cfg)['blocks']):
        selected=[r for r,p in zip(records,plan) if p['block']==index]
        blocks.append({**ceiling,'status':'COMPLETE','actual_forwards':len(selected),
                       'actual_positions':sum(r['input_length'] for r in selected),'forward_call_ids':[r['forward_call_id'] for r in selected]})
    causal=[{'mutation_start':start,'protected_positions':list(range(572,start)),'unprotected_positions':list(range(start,576)),
             'left_logits_sha256':reader.digest(['CPU protected logits',start]),'right_logits_sha256':reader.digest(['CPU protected logits',start]),
             'left_hidden_sha256':reader.digest(['CPU protected hidden',start]),'right_hidden_sha256':reader.digest(['CPU protected hidden',start]),
             'left_unprotected_logits_sha256':reader.digest(['CPU baseline logits',start]),
             'right_unprotected_logits_sha256':reader.digest(['CPU changed logits',start]),
             'left_unprotected_hidden_sha256':reader.digest(['CPU baseline hidden',start]),
             'right_unprotected_hidden_sha256':reader.digest(['CPU changed hidden',start]),
             'baseline_record':by_stage['max576_baseline'],'record':by_stage['max576_mutate_'+str(start)]} for start in (573,574,575)]
    resource={'stage':'complete','rss_gib':1.0,'host_available_gib':4.0,'gpu_free_gib':2.0,'disk_free_gib':10.0,
              'peak_gpu_allocated_gib':1.4,'peak_gpu_reserved_gib':1.5};emit('resource',**resource)
    result={'evidence_class':'CPU_SYNTHETIC_INTERFACE_NOT_NATIVE','status':'COMPLETE_READER_QUALIFICATION_WORKER_ONLY','scope':reader.SCOPE,
            **context,'wall_seconds':.9,'base_unchanged':True,'model_loads':1,'bulk_capture_qualified':False,'training_qualified':False,
            'rollouts':rollouts,'forward_records':records,'blocks':blocks,'accounting':calls.snapshot(),'resources':[resource],
            'checks':{'maximum_shape_causality':causal},'bulk_diagnostics':{
                label:{'record':by_stage[label+'_bulk_DIAGNOSTIC_ONLY'],'operation_eligible':False,'old_gate_relabelled':False}
                for label in ('historical','long')}}
    emit('reader_control_manifest',controls={k:result[k] for k in ('blocks','checks','rollouts','bulk_diagnostics')})
    return result,events


def artifact_fixture(cfg):
    """CPU schema fixture only; no native publication or measured host data."""
    context,members,transfer,go=run_fixture(cfg);binding=context['binding'];run=context['run_identity']
    artifacts={
      'worker_result':{'evidence_class':'CPU_SYNTHETIC_INTERFACE_NOT_NATIVE','wall_seconds':.9,**context},
      'source_launch_review':{'status':'APPROVED_FOR_READER_QUALIFICATION','binding':binding,'reviewer':'/root/exception_science'},
      'supervisor_result':{**context,'job_key':run['job_key'],'owner':run['owner'],'fence':run['fence'],
          'status':'SUCCEEDED','exit_code':0,'launch_attempted':True,'worker_dispatched':True,'subprocess_creation_outcome_unknown':False,
          'reason':None,'wall_seconds':1.0,'owned_tree_exit':{'owned_tree_exited':True,'remaining_owned_pids':[],'observed_at':102.0},
          'peak_process_tree_rss_bytes':1024,'watchdog_poll_seconds':.1},
      'containment':{**context,'contract_sha256':binding['contract_sha256'],'in_exact_supervisor_job':True,
          'kill_on_close':True,'breakaway_allowed':False,'query_handle_inheritable':False,'query_handle_access':'JOB_OBJECT_QUERY_ONLY',
          'query_handle_closed_before_framework':True,'exact_owned_limit_flags':8192,'pid':11,'creation_time':101.0,
          **{k:go[k] for k in ('owner','fence','supervisor_pid','worker_tree_pid')}},
      'tree_containment':{**{k:v for k,v in context.items() if k!='binding'},'all_members_in_exact_owned_job':True,
          'before_resume':[members[0]],'before_go_members':members,'query_handle_transfer':transfer},'go':go,
      'runtime':{**context,'runtime':cfg['runtime'],'cuda':'12.6','gpu_name':'CPU_SCHEMA_ONLY','dtype':'bfloat16','attention':'eager',
          'cache':False,'adapters':False,'allow_tf32_fixed':False,'reduction_flags_changed':False,
          'reduction_flags':{k:{'available':False,'value':None} for k in ('allow_bf16_reduced_precision_reduction','allow_fp16_reduced_precision_reduction',
          'allow_bf16_reduced_precision_reduction_split_k','allow_fp16_reduced_precision_reduction_split_k','allow_fp16_accumulation')}},
      'model':{**context,'asset_sha256':cfg['asset_sha256'],'unique_parameters':596049920,'tied':True,'trainable_parameters':0,
          'dtype':'bfloat16','tensor_sha256':{k:'a'*64 for k in cfg['model_tensor_names']},
          'loader':{'loader_mode':'stream_read_2MiB_v1','serialized_tensors':310,'read_buffer_bytes':2097152,'maximum_payload_read_bytes':2097152,
          'memory_mapped':False,'cpu_model_allocated':False,'blocking_chunk_copies':True}},
      'tokenizer':{**context,'tokenizer_sha256':binding['tokenizer_sha256'],'template_sha256':binding['template_sha256'],
          'enable_thinking':False,'rendered_utf8':'CPU_SCHEMA_ONLY','rendered_utf8_sha256':hashlib.sha256(b'CPU_SCHEMA_ONLY').hexdigest(),
          'token_ids':cfg['historical_prompt_ids'][-2:],'roundtrip_text':'Qualification: 7 + 5 = 12.','roundtrip_ids':cfg['tokenizer_roundtrip_ids'],
          'roundtrip_verified':True,'chat_token_equivalence_verified':True,'historical_suffix_equivalence_verified':True},
      'launch_intent':{'binding':binding,'run_identity':run,'scope':reader.SCOPE,'model_dispatch':True,
          'command':execution.registered_worker_command(cfg,run),'job_key':run['job_key']},
      'launch_attempt':{'job_key':run['job_key'],'owner':run['owner'],'fence':run['fence'],'command_sha256':run['command_sha256'],
          'attempted_at':100.5,'outcome':'PENDING_SUBPROCESS_CREATION'}}
    proof={'observer':'CRS_READER_SUPERVISOR_V1','run_identity':run,'runtime_worker_identity':context['runtime_worker_identity'],
           'worker_identity':context['worker_identity'],'request_hash':run['request_hash'],'owner':run['owner'],'fence':run['fence'],
           'process_exited':True,'owned_tree_exited':True,'owned_tree_exit':artifacts['supervisor_result']['owned_tree_exit'],
           'elapsed_native_seconds':1.0,'exit_observed_at':103.0}
    receipt={'schema_version':1,'kind':'native_outcome_observation','key':run['job_key'],'payload':proof}
    proof_path=str((Path(run['registration_payload']['output'])/'receipts'/('native_outcome_observation-'+hashlib.sha256(run['job_key'].encode()).hexdigest()+'.json')).resolve())
    row={'key':run['job_key'],'state':'SUCCEEDED','epoch':run['epoch'],'request_hash':run['request_hash'],'owner':run['owner'],'fence':run['fence'],
         'reserved_seconds':600.0,'reserved':100.0,'payload':run['registration_payload'],'worker':context['worker_identity'],
         'actual_seconds':1.0,'finished':104.0,'result':{'evidence':{'supervisor_receipt_sha256':reader.digest(artifacts['supervisor_result']),
         'native_outcome_proof_ref':{'path':proof_path,'sha256':reader.digest(receipt)}},'outcome':'SUCCEEDED','actual_native_seconds':1.0,'reservation_exceeded':False}}
    artifacts['job_outcome']={**context,'job':row};artifacts['native_outcome']=receipt
    return binding,artifacts


def supervisor_reference(artifacts):
    """Actual unchanged native writer bytes in a temporary CPU protocol run."""
    output=Path(artifacts['go']['run_identity']['registration_payload']['output'])
    output.mkdir(parents=True,exist_ok=True)
    path=output/'SUPERVISOR_RESULT.json'
    execution.write_new(path,artifacts['supervisor_result'])
    reference={'path':str(path.resolve()),'sha256':reader.sha(path)}
    artifacts['job_outcome']['job']['result']['evidence']['supervisor_receipt_sha256']=reference['sha256']
    return {'supervisor_result':reference}


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.cfg = reader.read_json(reader.CONTRACT)

    def fake_forward(self, tokens):
        iterator = iter(tokens)
        def call(ids, keep, stage):
            self.assertEqual(keep, 1)
            return {"argmax_id": next(iterator), "lowest_id_argmax_verified": True,
                    "input_ids_sha256": reader.digest(ids), "input_length": len(ids),
                    "selected_model_logprob": -.1, "raw_logits_sha256": "a" * 64,
                    "raw_hidden_sha256": "b" * 64}
        return call

    def test_public_token_boundary_rejects_evaluator_fields_and_unsupported_shapes(self):
        base = {"input_ids": [16], "attention_mask": [1], "max_new_tokens": 64}
        for request in ({**base, "gold": "12"}, {**base, "max_new_tokens": 65},
                        {**base, "input_ids": [True]}, {**base, "attention_mask": [0]},
                        {**base, "input_ids": [16] * 513, "attention_mask": [1] * 513}):
            with self.assertRaises(reader.ReaderBlocked): reader.validate_request(request)

    def test_each_eos_id_and_eos_at_cap_take_precedence(self):
        for eos in reader.EOS_IDS:
            request = {"input_ids": [eos], "attention_mask": [1], "max_new_tokens": 2}
            out = reader.manual_greedy(request, self.fake_forward([17, eos]))
            self.assertEqual(out["generation_count"], 2)
            self.assertEqual(out["stop_reason"], "eos")
            self.assertEqual(out["visible_ids"], [17])
            self.assertEqual(out["behavior_logprobs"], [0.0, 0.0])

    def test_prompt_eos_is_not_output_termination_and_cap_never_emits_extra(self):
        request = {"input_ids": [151645] * 512, "attention_mask": [1] * 512, "max_new_tokens": 64}
        out = reader.manual_greedy(request, self.fake_forward([17] * 64))
        self.assertEqual(len(out["generated_ids"]), 64)
        self.assertEqual(out["stop_reason"], "cap")
        self.assertEqual(out["steps"][-1]["input_length"], 575)

    def test_invalid_argmax_nonfinite_and_foreign_prefix_rejected(self):
        request = {"input_ids": [16], "attention_mask": [1], "max_new_tokens": 1}
        for change in ({"lowest_id_argmax_verified": False}, {"argmax_id": True},
                       {"selected_model_logprob": float("nan")}, {"input_ids_sha256": "foreign"}):
            original = self.fake_forward([17])
            def bad(ids, keep, stage): return {**original(ids, keep, stage), **change}
            with self.assertRaises(reader.ReaderBlocked): reader.manual_greedy(request, bad)

    def test_exact_278_forward_upper_schedule_and_nonemitting576_geometry(self):
        plan = reader.qualifier_plan(self.cfg)
        self.assertEqual(sum(b["max_forwards"] for b in plan["blocks"]), 278)
        self.assertEqual(sum(b["max_input_positions"] for b in plan["blocks"]), 146316)
        self.assertEqual(plan["max_logical_calls"], 590)
        lengths = [len(ids) for ids in plan["forced_mechanical_prefixes"]]
        self.assertEqual(lengths, list(range(512, 577)))
        self.assertFalse(plan["mechanical_is_model_generation"])
        self.assertEqual(sum(lengths), 35360)

    def test_original_likelihood_bounds_are_not_replaced_by_replay_identity(self):
        observed = reader.likelihood_measurement([-.5, -.41], [-.46, -.37])
        self.assertFalse(observed["passed"])
        self.assertEqual(observed["tolerances"], {"max_abs": .03125, "mean_abs": .01})
        self.assertEqual(self.cfg["historical_bulk_capture"]["status"], "FAILED_UNRESOLVED")
        with self.assertRaises(reader.ReaderBlocked): reader.likelihood_measurement([float("nan")], [0])

    def test_same_prefix_raw_hash_not_merely_equal_argmax(self):
        first = [{"input_ids_sha256": "a", "raw_logits_sha256": "b", "raw_hidden_sha256": "c", "selected_model_logprob": -.1}]
        second = copy.deepcopy(first); second[0]["raw_hidden_sha256"] = "different"
        with self.assertRaises(reader.ReaderBlocked): reader.require_same_prefix(first, second)

    def test_caller_pass_or_d2_receipt_cannot_open_reader(self):
        for evidence in (None, {"status": "PASS"}, {"scope": "D2", "native_qualified": True}):
            with self.assertRaises(reader.ReaderBlocked): reader.verify_qualification(evidence, {})

    def test_builder_review_and_wrong_binding_fail_without_model_import(self):
        expected = {"source_commit": "a" * 40}
        for review in ({"status": "APPROVED_FOR_READER_QUALIFICATION", "binding": expected, "reviewer": "/root/runtime_audit"},
                       {"status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC", "binding": expected, "reviewer": "independent"}):
            with self.assertRaises(reader.ReaderBlocked): execution.review_matches(review, expected)

    def test_stock_policy_is_no_cache_greedy_without_processors(self):
        policy = self.cfg["stock_generation_config"]
        self.assertFalse(policy["do_sample"]); self.assertFalse(policy["use_cache"])
        self.assertEqual(policy["num_beams"], 1); self.assertEqual(policy["repetition_penalty"], 1)
        self.assertIsNone(policy["forced_eos_token_id"])
        self.assertEqual(policy["min_new_tokens"], 0)
        self.assertEqual(policy["eos_token_id"], [151645, 151643])

    def test_no_torch_transformers_or_tokenizer_loaded_by_this_module(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("tokenizers", sys.modules)

    def test_source_contract_drift_rejected_before_framework_import(self):
        with self.assertRaises(reader.ReaderBlocked): execution.load_contract("0" * 64)

    def test_complete_cpu_interface_fixture_and_missing_control_rejections(self):
        result,events=completion_fixture(self.cfg)
        validated=execution.validate_qualification_result(result,self.cfg,'a'*40,events)
        self.assertEqual(validated['actual_forwards'],89)
        changes=[lambda r:r.update(binding={'source_commit':'a'*40}),
                 lambda r:r['blocks'][3].update(actual_forwards=64),
                 lambda r:r['checks']['maximum_shape_causality'][0].update(protected_positions=[999]),
                 lambda r:r['rollouts']['long']['owned'].update(generated_ids=[17]),
                 lambda r:r['forward_records'][30].update(input_length=1),
                 lambda r:r['forward_records'][0].update(raw_logits_sha256='not-a-hash')]
        for mutate in changes:
            bad=copy.deepcopy(result);mutate(bad)
            with self.assertRaises((reader.ReaderBlocked,ValueError,KeyError)):
                execution.validate_qualification_result(bad,self.cfg,'a'*40,events)
        missing=[e for e in events if e['kind']!='reader_forward_measurement']
        with self.assertRaises(reader.ReaderBlocked):execution.validate_qualification_result(result,self.cfg,'a'*40,missing)

    def test_historical_coherent_prefix_drift_rejects_unchanged_bounds(self):
        observed=reader.likelihood_measurement(self.cfg['historical_online_logprobs'],[-.1]*3)
        self.assertFalse(observed['passed'])
        self.assertEqual(self.cfg['historical_selected_raw_logits'],[20.625,20.25,25.25])

    def test_coherent_geometry_types_missing_and_duplicate_controls_rejected(self):
        original,events=completion_fixture(self.cfg)
        for change in [lambda r:r['forward_records'][0]['logits_shape'].__setitem__(0,True),
                       lambda r:r['forward_records'][0]['witnesses']['mask'].__setitem__('allowed_uint16',False),
                       lambda r:r['forward_records'][0]['witnesses']['positions']['ids'][0].__setitem__(0,False),
                       lambda r:r['rollouts']['long']['owned']['steps'][0].__setitem__('argmax_id',17),
                       lambda r:r['accounting']['operations']['model_construct'].__setitem__('attempted',True),
                       lambda r:r['resources'][0].__setitem__('gpu_free_gib',float('nan'))]:
            bad=copy.deepcopy(original);change(bad)
            # Coherently rewrite journal copies so semantic checks, not only hashes, reject.
            journal=copy.deepcopy(events)
            measured=iter(bad['forward_records'])
            for event in journal:
                if event['kind']=='reader_forward_measurement':event['record']=next(measured)
                elif event['kind']=='reader_control_manifest':event['controls']={k:bad[k] for k in ('blocks','checks','rollouts','bulk_diagnostics')}
                elif event['kind']=='resource':event.update(bad['resources'][0])
            with self.assertRaises((reader.ReaderBlocked,ValueError)):
                execution.validate_qualification_result(bad,self.cfg,'a'*40,journal)
        for journal in ([e for e in events if e['kind']!='reader_control_manifest'],events+[events[-1]]):
            with self.assertRaises(reader.ReaderBlocked):execution.validate_qualification_result(original,self.cfg,'a'*40,journal)

    def test_status_only_unexecuted_native_artifacts_cannot_open_serving(self):
        binding=execution.binding(self.cfg,reader.sha(reader.CONTRACT),'a'*40)
        minimal={k:{'binding':binding,'status':'SUCCEEDED'} for k in reader.EVIDENCE_NAMES if k!='events'}
        minimal['source_launch_review']={'status':'APPROVED_FOR_READER_QUALIFICATION','binding':binding,'reviewer':'/root/exception_science'}
        with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(minimal,self.cfg,binding,{})

    def test_complete_artifact_schema_and_source_tree_loader_tokenizer_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            package=Path(td)
            with patch.object(reader,'PACKAGE',package),patch.object(execution,'PACKAGE',package):
                binding,artifacts=artifact_fixture(self.cfg);refs=supervisor_reference(artifacts)
                self.assertTrue(execution.validate_native_artifacts(artifacts,self.cfg,binding,refs))
                self.check_artifact_rejections(binding,artifacts,refs)

    def check_artifact_rejections(self,binding,artifacts,refs):
        mutations=[lambda a:a['supervisor_result'].update(worker_dispatched=False),
                   lambda a:a['supervisor_result'].update(exit_code=False),
                   lambda a:a['containment'].update(query_handle_closed_before_framework=False),
                   lambda a:a['containment'].update(exact_owned_limit_flags=12288),
                   lambda a:a['go'].update(fence=2),
                   lambda a:a['tree_containment']['before_go_members'][1].update(executable_sha256='b'*64),
                   lambda a:a['model']['tensor_sha256'].pop(next(iter(a['model']['tensor_sha256']))),
                   lambda a:a['model']['loader'].update(memory_mapped=True),
                   lambda a:a['runtime'].update(cache=True),
                   lambda a:a['tokenizer'].update(roundtrip_verified=False),
                   lambda a:a['tokenizer'].update(rendered_utf8='changed'),
                   lambda a:a['source_launch_review'].update(reviewer='')]
        mutations += [lambda a:a['supervisor_result'].update(job_key='OTHER_RUN',owner='OTHER',fence=999),
                      lambda a:a['supervisor_result'].update(worker_identity={'pid':999,'creation_time':'999','command_hash':'a'*64}),
                      lambda a:a['tree_containment'].update(before_go_members=[a['tree_containment']['before_go_members'][1]]*2),
                      lambda a:a['launch_intent'].update(command=['different program']),
                      lambda a:a['native_outcome']['payload'].update(elapsed_native_seconds=True),
                      lambda a:a['job_outcome']['job'].update(actual_seconds=0)]
        for mutate in mutations:
            bad=copy.deepcopy(artifacts);mutate(bad)
            with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(bad,self.cfg,binding,refs)

    def test_actual_supervisor_newline_bytes_and_same_object_whitespace_drift_reject(self):
        with tempfile.TemporaryDirectory() as td:
            package=Path(td)
            with patch.object(reader,'PACKAGE',package),patch.object(execution,'PACKAGE',package):
                binding,artifacts=artifact_fixture(self.cfg);refs=supervisor_reference(artifacts)
                path=Path(refs['supervisor_result']['path']);original=path.read_bytes()
                self.assertTrue(original.endswith(b'\n'))
                self.assertNotEqual(refs['supervisor_result']['sha256'],reader.digest(artifacts['supervisor_result']))
                self.assertTrue(execution.validate_native_artifacts(artifacts,self.cfg,binding,refs))
                for missing in (None,{}, {'supervisor_result':{'path':str(path),'sha256':'invalid'}}):
                    with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(artifacts,self.cfg,binding,missing)
                other=path.parent/'WRONG_SUPERVISOR_RESULT.json';other.write_bytes(original)
                with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(artifacts,self.cfg,binding,
                    {'supervisor_result':{'path':str(other),'sha256':reader.sha(other)}})
                canonical_only=copy.deepcopy(artifacts)
                canonical_only['job_outcome']['job']['result']['evidence']['supervisor_receipt_sha256']=reader.digest(artifacts['supervisor_result'])
                with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(canonical_only,self.cfg,binding,refs)
                path.write_bytes(original+b' ')
                self.assertEqual(reader.read_json(path),artifacts['supervisor_result'])
                with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(artifacts,self.cfg,binding,refs)
                updated={'supervisor_result':{'path':str(path),'sha256':reader.sha(path)}}
                with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(artifacts,self.cfg,binding,updated)
                path.write_bytes(original)
                changed=copy.deepcopy(artifacts);changed['supervisor_result']['wall_seconds']=1.1
                with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(changed,self.cfg,binding,refs)

    def test_constant_response_or_missing_unprotected_causal_controls_reject(self):
        original,events=completion_fixture(self.cfg)
        mutations=[lambda c:c.update(right_unprotected_logits_sha256=c['left_unprotected_logits_sha256']),
                   lambda c:c.update(right_unprotected_hidden_sha256=c['left_unprotected_hidden_sha256']),
                   lambda c:c.pop('baseline_record'),
                   lambda c:c.update(unprotected_positions=[999]),
                   lambda c:c['record'].update(raw_logits_sha256=c['baseline_record']['raw_logits_sha256']),
                   lambda c:c['record'].update(raw_hidden_sha256=c['baseline_record']['raw_hidden_sha256'])]
        for mutate in mutations:
            bad=copy.deepcopy(original);mutate(bad['checks']['maximum_shape_causality'][0]);journal=copy.deepcopy(events)
            measured=iter(bad['forward_records'])
            for event in journal:
                if event['kind']=='reader_forward_measurement':event['record']=next(measured)
                if event['kind']=='reader_control_manifest':event['controls']={k:bad[k] for k in ('blocks','checks','rollouts','bulk_diagnostics')}
            with self.assertRaises((reader.ReaderBlocked,KeyError)):
                execution.validate_qualification_result(bad,self.cfg,'a'*40,journal)

    def test_normalized_builder_aliases_reject_at_both_review_stages(self):
        aliases=['runtime_audit','/root/runtime_audit','\\root\\runtime_audit','/root/runtime_audit/',
                 ' \\ROOT\\Runtime_Audit\\ ','root','/root/','scientific_audit','data_audit','exception_builder',
                 '/root/exception_builder/','unknown','/other/exception_science','independent']
        expected={'source_commit':'a'*40};refs={'CPU_ONLY':{'path':'CPU_ONLY','sha256':'a'*64}}
        for alias in aliases:
            source={'status':'APPROVED_FOR_READER_QUALIFICATION','binding':expected,'reviewer':alias}
            post={'status':'APPROVED_READER_NATIVE_EVIDENCE','review_kind':'POST_RUN_NATIVE_EVIDENCE',
                  'actual_native_evidence_reviewed':True,'binding':expected,'evidence_refs':refs,'reviewer':alias}
            with self.assertRaises(reader.ReaderBlocked):execution.review_matches(source,expected)
            with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review(post,expected,refs)
        for alias in ('exception_science',' /ROOT\\EXCEPTION_SCIENCE/ ','/root/exception_ops/'):
            self.assertIn(reader.independent_reviewer_id(alias),('/root/exception_science','/root/exception_ops'))

    def test_invalid_source_review_path_rejects_before_inner_dispatch(self):
        args=SimpleNamespace(review=reader.PACKAGE.parent/'OUTSIDE_READER_RESULTS.json',output=reader.PACKAGE/'results/CPU_ONLY',
                             job_key='CPU_ONLY',epoch='CPU_ONLY',host_db=execution.host_state_path(),contract_sha256=reader.sha(reader.CONTRACT))
        with patch.object(execution,'_supervise_reader_worker',side_effect=AssertionError('dispatch occurred')) as inner:
            with self.assertRaises(reader.ReaderBlocked):execution.supervise(args,self.cfg)
            inner.assert_not_called()

    def test_existing_output_uses_recovery_and_never_relaunches(self):
        with tempfile.TemporaryDirectory() as td:
            args=SimpleNamespace(output=Path(td))
            with patch.object(execution,'validate_publication_inputs'),patch.object(execution,'recover_pending_publication',return_value={'CPU_ONLY':True}) as recover,\
                 patch.object(execution,'_supervise_reader_worker',side_effect=AssertionError('model retry')) as inner:
                self.assertEqual(execution.supervise(args,self.cfg),0);recover.assert_called_once();inner.assert_not_called()

    def test_create_only_publication_identical_concurrent_retry_and_lost_response(self):
        with tempfile.TemporaryDirectory() as td:
            package=Path(td);root=package/'results/run';root.mkdir(parents=True);path=root/'PENDING.json';value={'CPU_ONLY':True}
            with patch.object(execution,'PACKAGE',package):
                with ThreadPoolExecutor(max_workers=4) as pool:refs=list(pool.map(lambda _:execution.publish_exact(path,value),range(4)))
                self.assertTrue(all(ref==refs[0] for ref in refs));self.assertEqual(path.read_bytes(),reader.canonical(value))
                with self.assertRaises(reader.ReaderBlocked):execution.publish_exact(path,{'conflict':True})
                lost=root/'LOST_RESPONSE.json';original=execution.os.link
                def after_link(a,b):original(a,b);raise OSError('CPU lost publication response')
                with patch.object(execution.os,'link',side_effect=after_link):
                    with self.assertRaises(reader.ReaderBlocked):execution.publish_exact(lost,value)
                self.assertEqual(execution.publish_exact(lost,value)['sha256'],reader.digest(value))
                partial=root/'PARTIAL.json';partial.write_bytes(b'{')
                with self.assertRaises(reader.ReaderBlocked):execution.publish_exact(partial,value)
                self.assertEqual(partial.read_bytes(),b'{')

    def test_child_requires_canonical_exact_running_reserved_job_and_read_only_adapter(self):
        from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsStore
        with tempfile.TemporaryDirectory() as td:
            package=Path(td);(package/'results').mkdir();db=package/'state.sqlite3';review=package/'results/CPU_SOURCE_REVIEW.json';review.write_bytes(b'{}')
            with patch.object(reader,'PACKAGE',package),patch.object(execution,'PACKAGE',package),patch.object(execution,'host_state_path',return_value=db):
                context,_,_,go=run_fixture(self.cfg);run=context['run_identity']
                run['registration_payload']['source_review']['sha256']=reader.sha(review)
                run['request_hash']=reader.digest({'payload':run['registration_payload'],'epoch':run['epoch'],'dependencies':[]})
                run['command_sha256']=reader.digest(execution.registered_worker_command(self.cfg,run));context['worker_identity']['command_hash']=run['command_sha256']
                args=SimpleNamespace(host_db=db,output=Path(run['registration_payload']['output']),review=review,**{k:run[k] for k in ('job_key','epoch','run_id','request_hash','owner','fence')})
                store=OperationsStore(db,package/'results/receipts',clock=lambda:100.0)
                try:
                    store.register_job(run['job_key'],run['registration_payload'],run['epoch']);store.ready(run['job_key'])
                    lease=store.acquire_lease(run['owner'],120);store.reserve(run['job_key'],run['owner'],lease['fence'],600)
                    with self.assertRaises(reader.ReaderBlocked):execution.require_worker_run(store,args,go,context['binding'],pid=11,creation_time=101.0)
                    store.mark_running(run['job_key'],run['owner'],lease['fence'],context['worker_identity'])
                    execution.require_worker_run(store,args,go,context['binding'],pid=11,creation_time=101.0)
                    for change in ({'host_db':package/'other.sqlite3'},{'job_key':'OTHER_JOB'},{'fence':999},{'epoch':'OTHER_EPOCH'}):
                        with self.assertRaises(reader.ReaderBlocked):execution.require_worker_run(store,SimpleNamespace(**{**vars(args),**change}),go,context['binding'],pid=11,creation_time=101.0)
                    with self.assertRaises(reader.ReaderBlocked):execution.require_worker_run(store,args,go,context['binding'],pid=12,creation_time=101.0)
                    readonly=execution.ReadOnlyOperationsStore(db,package/'results/receipts')
                    try:
                        self.assertEqual(readonly.get_job(run['job_key'])['state'],'RUNNING')
                        with self.assertRaises(execution.sqlite3.OperationalError):readonly.db.execute('UPDATE jobs SET actual_seconds=0')
                    finally:readonly.close()
                    store.mark_unknown(run['job_key'],'CPU fixture interruption')
                    with self.assertRaises(reader.ReaderBlocked):execution.require_worker_run(store,args,go,context['binding'],pid=11,creation_time=101.0)
                finally:store.close()

    def test_pending_recovery_from_retained_cpu_protocol_bytes_preserves_job_cost_and_never_dispatches(self):
        """Temporary protocol simulation; no native proof is retained or serving opened."""
        from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsStore
        with tempfile.TemporaryDirectory() as td:
            package=Path(td);(package/'results').mkdir();db=package/'state.sqlite3'
            with patch.object(reader,'PACKAGE',package),patch.object(execution,'PACKAGE',package),patch.object(execution,'host_state_path',return_value=db):
                binding,artifacts=artifact_fixture(self.cfg);context={k:artifacts['worker_result'][k] for k in
                    ('binding','run_identity','worker_identity','runtime_worker_identity')};run=context['run_identity']
                output=Path(run['registration_payload']['output']);output.mkdir()
                source_review=Path(run['registration_payload']['source_review']['path']);source_review.write_bytes(reader.canonical(artifacts['source_launch_review']))
                run['registration_payload']['source_review']['sha256']=reader.sha(source_review)
                run['request_hash']=reader.digest({'payload':run['registration_payload'],'epoch':run['epoch'],'dependencies':[]})
                run['command_sha256']=reader.digest(execution.registered_worker_command(self.cfg,run));context['worker_identity']['command_hash']=run['command_sha256']
                artifacts['launch_intent']['command']=execution.registered_worker_command(self.cfg,run)
                artifacts['launch_attempt']['command_sha256']=run['command_sha256']
                artifacts['native_outcome']['payload']['request_hash']=run['request_hash']
                result,events=completion_fixture(self.cfg);result.update(context)
                for event in events:event.update(context)
                artifacts['worker_result']=result
                files={'worker_result':'READER_QUAL_RESULT.json','supervisor_result':'SUPERVISOR_RESULT.json',
                       'containment':'CONTAINMENT_RECEIPT.json','tree_containment':'TREE_CONTAINMENT.json','go':'GO.json',
                       'runtime':'RUNTIME_RECEIPT.json','model':'MODEL_RECEIPT.json','tokenizer':'TOKENIZER_RECEIPT.json',
                       'launch_intent':'LAUNCH_INTENT.json','launch_attempt':'LAUNCH_ATTEMPT.json'}
                for name,filename in files.items():execution.write_new(output/filename,artifacts[name])
                self.assertTrue((output/'SUPERVISOR_RESULT.json').read_bytes().endswith(b'\n'))
                self.assertNotEqual(reader.sha(output/'SUPERVISOR_RESULT.json'),reader.digest(artifacts['supervisor_result']))
                (output/'EVENTS.jsonl').write_bytes(b''.join(reader.canonical(event)+b'\n' for event in events))
                verified=execution.validate_qualification_result(result,self.cfg,'a'*40,events)
                verified['events_sha256']=reader.sha(output/'EVENTS.jsonl');execution.write_new(output/'COMPLETION_VERIFICATION.json',verified)
                clock=[100.0];store=OperationsStore(db,output/'receipts',clock=lambda:clock[0])
                try:
                    store.register_job(run['job_key'],run['registration_payload'],run['epoch']);store.ready(run['job_key'])
                    lease=store.acquire_lease(run['owner'],120);store.reserve(run['job_key'],run['owner'],lease['fence'],600)
                    store.mark_running(run['job_key'],run['owner'],lease['fence'],context['worker_identity'])
                    clock[0]=104.0;proof_ref=store.write_receipt('native_outcome_observation',run['job_key'],artifacts['native_outcome']['payload'])
                    evidence={'supervisor_receipt_sha256':reader.sha(output/'SUPERVISOR_RESULT.json'),'native_outcome_proof_ref':proof_ref}
                    store.reconcile(run['job_key'],'SUCCEEDED',1.0,evidence,owner=run['owner'],fence=lease['fence'])
                    terminal=store.get_job(run['job_key']);events_before_retry=[dict(row) for row in store.db.execute('SELECT * FROM events ORDER BY id')]
                    self.assertEqual(store.reconcile(run['job_key'],'SUCCEEDED',1.0,evidence,owner=run['owner'],fence=lease['fence']),terminal)
                    self.assertEqual([dict(row) for row in store.db.execute('SELECT * FROM events ORDER BY id')],events_before_retry)
                    store.release_lease(run['owner'],lease['fence'])
                    original_row=store.get_job(run['job_key'])
                    original_events=[dict(row) for row in store.db.execute('SELECT * FROM events ORDER BY id')]
                    original_cost=store.budget_snapshot(run['epoch'])['epoch_charged_seconds']
                    original_receipt=Path(proof_ref['path']).read_bytes()
                    self.assertEqual(original_receipt,reader.canonical(artifacts['native_outcome']))
                    self.assertEqual(original_cost,1.0)
                finally:store.close()
                args=SimpleNamespace(output=output,review=source_review,job_key=run['job_key'],epoch=run['epoch'],host_db=db,
                                     contract_sha256=reader.sha(reader.CONTRACT))
                publish=execution.publish_exact
                def interrupted(path,value):
                    if Path(path).name=='READER_EVIDENCE_PENDING_REVIEW.json':raise reader.ReaderBlocked('CPU interrupted pending publication')
                    return publish(path,value)
                with contextlib.ExitStack() as guards:
                    guards.enter_context(patch.object(execution,'committed_snapshot',return_value={'commit':'a'*40,'dirty':False}))
                    spawn=guards.enter_context(patch.object(execution.subprocess,'Popen',side_effect=AssertionError('model redispatch')))
                    mutations=[guards.enter_context(patch.object(OperationsStore,name,side_effect=AssertionError('recovery mutation: '+name)))
                               for name in ('reserve','acquire_lease','reconcile','release_lease')]
                    with patch.object(execution,'publish_exact',side_effect=interrupted):
                        with self.assertRaises(reader.ReaderBlocked):execution.recover_pending_publication(args,self.cfg)
                    first=execution.recover_pending_publication(args,self.cfg);second=execution.recover_pending_publication(args,self.cfg)
                    self.assertEqual(first,second);self.assertEqual(reader.read_json(first['path'])['status'],'COMPLETE_READER_EVIDENCE_PENDING_INDEPENDENT_REVIEW')
                    Path(first['path']).write_bytes(b'{')
                    with self.assertRaises(reader.ReaderBlocked):execution.recover_pending_publication(args,self.cfg)
                    self.assertEqual(Path(first['path']).read_bytes(),b'{');spawn.assert_not_called()
                    for mutation in mutations:mutation.assert_not_called()
                readonly=execution.ReadOnlyOperationsStore(db,output/'receipts')
                try:
                    self.assertEqual(readonly.get_job(run['job_key']),original_row)
                    self.assertEqual([dict(row) for row in readonly.db.execute('SELECT * FROM events ORDER BY id')],original_events)
                    self.assertEqual(readonly.budget_snapshot(run['epoch'])['epoch_charged_seconds'],original_cost)
                    self.assertEqual(Path(proof_ref['path']).read_bytes(),original_receipt)
                finally:readonly.close()

    def test_quarantined_inherited_path_blocks_preflight_before_host_checks(self):
        with patch.object(execution,'load_contract',return_value=self.cfg),patch.object(execution,'collect_telemetry',side_effect=AssertionError('new host check')):
            result=execution.preflight('CPU_ONLY')
        self.assertEqual(result['status'],'BLOCKED')
        self.assertTrue(any('QUARANTINED' in message for message in result['reasons']))

    def test_blank_reviewer_is_not_independent_approval(self):
        with self.assertRaises(reader.ReaderBlocked):
            execution.review_matches({'status':'APPROVED_FOR_READER_QUALIFICATION','binding':{},'reviewer':''},{})

    def test_source_review_cannot_be_reused_as_actual_native_evidence_approval(self):
        binding={'source_commit':'a'*40};refs={'worker_result':{'path':'CPU_ONLY','sha256':'a'*64}}
        source={'status':'APPROVED_FOR_READER_QUALIFICATION','review_kind':'PRELAUNCH_SOURCE_ONLY',
                'actual_native_evidence_reviewed':False,'binding':binding,'evidence_refs':refs,'reviewer':'/root/exception_science'}
        with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review(source,binding,refs)
        post={**source,'status':'APPROVED_READER_NATIVE_EVIDENCE','review_kind':'POST_RUN_NATIVE_EVIDENCE','actual_native_evidence_reviewed':True}
        self.assertTrue(reader.validate_native_evidence_review(post,binding,refs))
        with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review(post,binding,{'other_run':refs['worker_result']})
        with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review({**post,'reviewer':' '},binding,refs)


if __name__ == "__main__":
    unittest.main()
