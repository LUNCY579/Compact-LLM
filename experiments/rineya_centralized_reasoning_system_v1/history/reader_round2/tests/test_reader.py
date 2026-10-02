"""Lightweight stdlib-only reader rejection/geometry/termination controls."""
import copy
import hashlib
import struct
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import reader
from experiments.rineya_centralized_reasoning_system_v1 import reader_execution as execution


def completion_fixture(cfg):
    """Synthetic CPU interface witness, never a native qualification artifact."""
    from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting
    events = []
    def emit(kind, **details): events.append({'kind':kind,'time':float(len(events)),**details})
    calls = CallAccounting(emit,maximum_optimizer_calls=0)
    for op,n in [('model_construct',1),('checkpoint_load',1),('tensor_copy',310)]:
        for _ in range(n): calls.call(op,lambda:None,stage='CPU_SYNTHETIC_INTERFACE')
    long_generated = [151645];plan = reader.actual_forward_plan(cfg,long_generated);records=[];mask_cache={}
    for expected in plan:
        ids,keep,stage=expected['input_ids'],expected['logits_to_keep'],expected['stage'];length=len(ids)
        calls.call('forward',lambda:None,stage=stage,input_positions=length)
        if length not in mask_cache:mask_cache[length]=execution._mask_sha(length)
        token,lp,logit=151645,-.1,1.0
        if stage.startswith('historical_') and 'bulk' not in stage:
            index=int(stage.rsplit('_',1)[-1]);token=cfg['historical_generated_ids'][index];lp=cfg['historical_online_logprobs'][index];logit=cfg['historical_selected_raw_logits'][index]
        record={'stage':stage,'input_ids':ids,'input_ids_sha256':reader.digest(ids),'forward_call_id':calls.last_call_id,
                'input_length':length,'logits_to_keep':keep,'logits_shape':[1,keep,151936],'hidden_shape':[1,keep,1024],
                'raw_logits_sha256':cfg['historical_bulk_logits_sha256'] if stage=='historical_bulk_DIAGNOSTIC_ONLY' else 'a'*64,
                'raw_hidden_sha256':'b'*64,'argmax_id':token,'lowest_id_argmax_verified':True,'selected_raw_logit':logit,'selected_model_logprob':lp,
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
            if kind=='replay':bundle[kind]=steps
            else:bundle[kind]={'generated_ids':generated,'stop_reason':'eos','terminal_eos_id':generated[-1],
                              'raw_text':'CPU_SYNTHETIC','visible_text':'CPU_SYNTHETIC','steps':steps,
                              'model_logprobs':[r['selected_model_logprob'] for r in steps]}
        bundle['stock']['policy']={'processors':[],'stops':['MaxLengthCriteria','EosTokenCriteria']};rollouts[label]=bundle
    blocks=[]
    for index,ceiling in enumerate(reader.qualifier_plan(cfg)['blocks']):
        selected=[r for r,p in zip(records,plan) if p['block']==index]
        blocks.append({**ceiling,'status':'COMPLETE','actual_forwards':len(selected),
                       'actual_positions':sum(r['input_length'] for r in selected),'forward_call_ids':[r['forward_call_id'] for r in selected]})
    causal=[{'mutation_start':start,'protected_positions':list(range(572,start)),
             'left_logits_sha256':'c'*64,'right_logits_sha256':'c'*64,'left_hidden_sha256':'d'*64,'right_hidden_sha256':'d'*64,
             'record':by_stage['max576_mutate_'+str(start)]} for start in (573,574,575)]
    resource={'stage':'complete','rss_gib':1.0,'host_available_gib':4.0,'gpu_free_gib':2.0,'disk_free_gib':10.0,
              'peak_gpu_allocated_gib':1.4,'peak_gpu_reserved_gib':1.5}
    emit('resource',**resource)
    result={'evidence_class':'CPU_SYNTHETIC_INTERFACE_NOT_NATIVE','status':'COMPLETE_READER_QUALIFICATION_WORKER_ONLY','scope':reader.SCOPE,
            'binding':execution.binding(cfg,reader.sha(reader.CONTRACT),'a'*40),'base_unchanged':True,'model_loads':1,
            'bulk_capture_qualified':False,'training_qualified':False,'rollouts':rollouts,'forward_records':records,'blocks':blocks,
            'accounting':calls.snapshot(),'resources':[resource],'checks':{'maximum_shape_causality':causal},
            'bulk_diagnostics':{label:{'record':by_stage[label+'_bulk_DIAGNOSTIC_ONLY'],'operation_eligible':False,'old_gate_relabelled':False}
                                for label in ('historical','long')}}
    emit('reader_control_manifest',controls={k:result[k] for k in ('blocks','checks','rollouts','bulk_diagnostics')})
    return result,events


def artifact_fixture(cfg):
    """CPU schema fixture only; no retained native success or measured host values."""
    binding=execution.binding(cfg,reader.sha(reader.CONTRACT),'a'*40)
    exes=[(p,h) for p,h in cfg['runtime_file_sha256'].items() if p.lower().endswith('.exe')][:2]
    members=[{'pid':10+i,'creation_time':100.0+i,'executable':p,'executable_sha256':h,'exact_owned_job_member':True}
             for i,(p,h) in enumerate(exes)]
    transfer={'own_job_query_handle':100,'query_handle_target_pid':11,'query_handle_target_creation_time':101.0,
              'duplicate_access_mask':4,'duplicate_inheritable':False}
    go={'contract_sha256':binding['contract_sha256'],'owner':'CPU_SCHEMA_ONLY','fence':1,'supervisor_pid':9,'worker_tree_pid':10,**transfer}
    return binding,{
      'source_launch_review':{'status':'APPROVED_FOR_READER_QUALIFICATION','binding':binding,'reviewer':'independent'},
      'supervisor_result':{'binding':binding,'status':'SUCCEEDED','exit_code':0,'launch_attempted':True,'worker_dispatched':True,
          'subprocess_creation_outcome_unknown':False,'reason':None,'wall_seconds':1.0,
          'owned_tree_exit':{'owned_tree_exited':True,'remaining_owned_pids':[],'observed_at':102.0},
          'peak_process_tree_rss_bytes':1024,'watchdog_poll_seconds':.1},
      'containment':{'binding':binding,'contract_sha256':binding['contract_sha256'],'in_exact_supervisor_job':True,
          'kill_on_close':True,'breakaway_allowed':False,'query_handle_inheritable':False,'query_handle_access':'JOB_OBJECT_QUERY_ONLY',
          'query_handle_closed_before_framework':True,'exact_owned_limit_flags':8192,'pid':11,'creation_time':101.0,
          **{k:go[k] for k in ('owner','fence','supervisor_pid','worker_tree_pid')}},
      'tree_containment':{'all_members_in_exact_owned_job':True,'before_go_members':members,'query_handle_transfer':transfer},'go':go,
      'runtime':{'binding':binding,'runtime':cfg['runtime'],'cuda':'12.6','gpu_name':'CPU_SCHEMA_ONLY','dtype':'bfloat16','attention':'eager',
          'cache':False,'adapters':False,'allow_tf32_fixed':False,'reduction_flags_changed':False,
          'reduction_flags':{k:{'available':False,'value':None} for k in ('allow_bf16_reduced_precision_reduction','allow_fp16_reduced_precision_reduction',
          'allow_bf16_reduced_precision_reduction_split_k','allow_fp16_reduced_precision_reduction_split_k','allow_fp16_accumulation')}},
      'model':{'binding':binding,'asset_sha256':cfg['asset_sha256'],'unique_parameters':596049920,'tied':True,'trainable_parameters':0,
          'dtype':'bfloat16','tensor_sha256':{k:'a'*64 for k in cfg['model_tensor_names']},
          'loader':{'loader_mode':'stream_read_2MiB_v1','serialized_tensors':310,'read_buffer_bytes':2097152,'maximum_payload_read_bytes':2097152,
          'memory_mapped':False,'cpu_model_allocated':False,'blocking_chunk_copies':True}},
      'tokenizer':{'binding':binding,'tokenizer_sha256':binding['tokenizer_sha256'],'template_sha256':binding['template_sha256'],
          'enable_thinking':False,'rendered_utf8':'CPU_SCHEMA_ONLY','rendered_utf8_sha256':hashlib.sha256(b'CPU_SCHEMA_ONLY').hexdigest(),
          'token_ids':cfg['historical_prompt_ids'][-2:],'roundtrip_text':'Qualification: 7 + 5 = 12.','roundtrip_ids':cfg['tokenizer_roundtrip_ids'],
          'roundtrip_verified':True,'chat_token_equivalence_verified':True,'historical_suffix_equivalence_verified':True}}


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
        minimal['source_launch_review']={'status':'APPROVED_FOR_READER_QUALIFICATION','binding':binding,'reviewer':'independent'}
        with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(minimal,self.cfg,binding)

    def test_complete_artifact_schema_and_source_tree_loader_tokenizer_rejections(self):
        binding,artifacts=artifact_fixture(self.cfg)
        self.assertTrue(execution.validate_native_artifacts(artifacts,self.cfg,binding))
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
        for mutate in mutations:
            bad=copy.deepcopy(artifacts);mutate(bad)
            with self.assertRaises(reader.ReaderBlocked):execution.validate_native_artifacts(bad,self.cfg,binding)

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
                'actual_native_evidence_reviewed':False,'binding':binding,'evidence_refs':refs,'reviewer':'independent'}
        with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review(source,binding,refs)
        post={**source,'status':'APPROVED_READER_NATIVE_EVIDENCE','review_kind':'POST_RUN_NATIVE_EVIDENCE','actual_native_evidence_reviewed':True}
        self.assertTrue(reader.validate_native_evidence_review(post,binding,refs))
        with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review(post,binding,{'other_run':refs['worker_result']})
        with self.assertRaises(reader.ReaderBlocked):reader.validate_native_evidence_review({**post,'reviewer':' '},binding,refs)


if __name__ == "__main__":
    unittest.main()
