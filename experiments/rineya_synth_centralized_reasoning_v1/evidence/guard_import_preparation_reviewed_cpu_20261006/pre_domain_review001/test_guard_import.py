"""Whole authored guard fixtures, original helpers, read/import traps; no outcomes."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import builtins, hashlib, io, json, sys, tempfile, unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')

class GuardControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reads=[];cls.denied=[];cls.imports=[]
        original_io=io.open;original_open=builtins.open;original_import=builtins.__import__
        def checked(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and not any(c in str(mode) for c in ('w','a','x','+')):
                selected=Path(path).resolve();cls.reads.append(str(selected))
                if not selected.is_relative_to(HERE) and not(selected.parent==PACKAGE and selected.suffix=='.py'):
                    cls.denied.append(str(selected));raise AssertionError('non-authored guard/data/model read forbidden: '+str(selected))
            return original_io(path,mode,*args,**kwargs)
        def opened(path,mode='r',*args,**kwargs):
            return checked(path,mode,*args,**kwargs) if isinstance(path,(str,bytes,Path)) else original_open(path,mode,*args,**kwargs)
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','transformers','tokenizers','safetensors','ctypes','psutil','numpy'}:
                cls.imports.append(name);raise AssertionError('model/tokenizer/native import forbidden: '+name)
            return original_import(name,*args,**kwargs)
        cls.io_trap=patch.object(io,'open',side_effect=checked);cls.io_trap.start()
        cls.open_trap=patch.object(builtins,'open',side_effect=opened);cls.open_trap.start()
        cls.import_trap=patch.object(builtins,'__import__',side_effect=imports);cls.import_trap.start()
        path=HERE/'guard_import.py';cls.T=ModuleType('CPU_authored_guard_import');cls.T.__file__=str(path)
        exec(compile(path.read_bytes(),str(path),'exec'),cls.T.__dict__)
        cls.modules=cls.T._helpers();cls.adapter=cls.modules['score_import'];cls.cap=cls.T._CPU_capability()
        cls.policy={'CPU_authored':True,'retention':{'required_domain_classes':['task','language','long_context']}}
        cls.task_rows,cls.task_meta,cls.task_domains=cls.population('task')
        cls.long_rows,cls.long_meta,cls.long_domains=cls.population('long')
        cls.task_source=cls.T._CPU_source('task',cls.task_rows,cls.task_meta,cls.task_domains,_capability=cls.cap)
        cls.long_source=cls.T._CPU_source('long',cls.long_rows,cls.long_meta,cls.long_domains,_capability=cls.cap)
        cls.task_batches=cls.batches('task',cls.task_rows)
        cls.long_batches=cls.batches('long',cls.long_rows)
        cls.task_capture=cls.T._CPU_capture(cls.task_source,cls.task_batches,_capability=cls.cap)
        cls.long_capture=cls.T._CPU_capture(cls.long_source,cls.long_batches,_capability=cls.cap)
        cls.task_import=cls.T._CPU_import(cls.task_source,cls.task_capture,_capability=cls.cap)
        cls.long_import=cls.T._CPU_import(cls.long_source,cls.long_capture,_capability=cls.cap)
    @classmethod
    def tearDownClass(cls):
        cls.import_trap.stop();cls.open_trap.stop();cls.io_trap.stop()
        assert cls.imports==[],cls.imports
        assert len(cls.denied)==20,cls.denied
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_guard_',dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    @classmethod
    def population(cls,kind):
        T=cls.T;rows=[];metadata=[]
        if kind=='task':
            ids=[1];mask=[1]
            for mechanism in T.MECHANISMS:
                for world in range(64):
                    component='fixture_component_task_'+mechanism+'_%02d'%world
                    for variant in ('CPU_authored_original','CPU_authored_counterfactual'):
                        for rendering in ('json_lines','typed_ledger'):
                            row='fixture_task_'+mechanism+'_%02d_'%world+variant+'_'+rendering
                            rows.append({'row_id':row,'prompt_sha256':T.digest(row),'prompt_ids':ids,'prompt_tokens':1,'attention_mask':mask})
                            metadata.append({'row_id':row,'component_id':component,'mechanism':mechanism,'variant':variant,'rendering':rendering})
            domain=T.TASK_DOMAIN;category='task'
        else:
            arrays={length:[1]*length for length in T.ENVELOPES};masks={length:[1]*length for length in T.ENVELOPES}
            for length,cap in T.ENVELOPES.items():
                for family in T.FAMILIES:
                    for world in range(96):
                        component='fixture_component_long_'+family+'_%02d'%world
                        for variant,(answerable,adversarial) in T.LONG_VARIANTS.items():
                            row='fixture_long_'+family+'_%02d_%04d_'%(world,length)+variant
                            rows.append({'row_id':row,'prompt_sha256':T.digest(row),'prompt_ids':arrays[length],'prompt_tokens':length,'attention_mask':masks[length]})
                            metadata.append({'row_id':row,'component_id':component,'family':family,'placement':T.PLACEMENTS[world%4],'missing_index':world%3,'total_sequence_envelope':cap,'variant':variant,'answerable':answerable,'adversarial':adversarial})
            # The caller freezes this ID. Production source declares no default.
            domain='CPU_AUTHORED_EXTERNALLY_FROZEN_LONG_DOMAIN';category='long_context'
        domains=[{'domain_id':domain,'class':category,'row_ids':[r['row_id'] for r in rows],'component_ids':[m['component_id'] for m in metadata]}]
        return rows,metadata,domains
    @classmethod
    def batches(cls,kind,rows):
        T=cls.T;batches=[]
        for offset in range(0,len(rows),64):
            inputs=rows[offset:offset+64];packet=T.digest(inputs);outputs=[]
            for index,row in enumerate(inputs,offset):
                ids=[7]*32 if index%11==0 else [7,151645];eos=ids[-1]==151645
                trace=[{'token':token,'logprob':-1.0,'prefix_length':row['prompt_tokens']+step} for step,token in enumerate(ids)]
                outputs.append({'row_id':row['row_id'],'prompt_sha256':row['prompt_sha256'],'prompt_ids_sha256':T.digest(row['prompt_ids']),'input_row_sha256':T.digest(row),'source_packet_sha256':packet,'status':'RAW_RESPONSE_COLLECTED_UNSCORED','scope':cls.adapter.FIXTURE,'capability_correctness_scored':False,'oracle_inputs_read':False,'prompt_truncated':False,'raw_generated_ids':ids,'finish_reason':'eos' if eos else 'generation_cap','truncated':not eos,'eos_token_id':151645 if eos else None,'context_cap':512 if kind=='task' else T.ENVELOPES[row['prompt_tokens']],'maximum_generated_tokens':32,'decoded_response':'CPU_AUTHORED_WRONG' if index%17==0 else 'CPU_AUTHORED_RESPONSE','trace':trace,'selected_logprobs':[-1.0]*len(ids)})
            batches.append({'ordinal':offset//64+1,'source_kind':'task_guard_DEV' if kind=='task' else 'long_guard_DEV','inputs':inputs,'outputs':outputs,'source_packet_sha256':packet,'fixture_only':True})
        return batches
    def ref(self,name,value):
        path=self.root/name;raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii');path.write_bytes(raw)
        return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'value_sha256':self.T.digest(value)}
    def evidence(self,packets=None):
        proofs={}
        for packet in [self.task_import,self.long_import] if packets is None else packets:
            value=packet.value
            for domain in value['declared_domains']:
                name=domain['domain_id'];source=self.root/(name+'_scorer.py');code=b'def score_fixture(response, label):\n    return response == label\n';source.write_bytes(code)
                scorer={'path':str(source),'bytes':len(code),'sha256':hashlib.sha256(code).hexdigest()}
                labels=self.ref(name+'_labels.json',{'kind':'CPU_AUTHORED_GUARD_LABELS','fixture_only':True,'domain_id':name,'source_sha256':value['source_sha256'],'rows':{row:'CPU_AUTHORED_RESPONSE' for row in domain['row_ids']}})
                qualification=self.ref(name+'_qualification.json',{'kind':'CPU_AUTHORED_GUARD_SCORER_QUALIFICATION_FIXTURE','fixture_only':True,'domain_id':name,'source_sha256':value['source_sha256'],'scorer_source_sha256':scorer['sha256'],'labels_sha256':labels['sha256'],'qualified':True,'native_grader_qualified':False,'controls':[{'response':'same','label':'same','expected':True},{'response':'wrong','label':'same','expected':False}]})
                proofs[name]={'source_sha256':value['source_sha256'],'scorer_source':scorer,'labels':labels,'qualification':qualification}
        return proofs
    def enrollment(self,evidence,imports=None):
        return self.T._CPU_enrollment([self.task_import,self.long_import] if imports is None else imports,evidence,baseline_id='CPU_AUTHORED_M0_BASELINE',guard_policy=self.policy,_capability=self.cap)
    def test_whole_task_roster_and_unmodified_original_raw_contract(self):
        value=self.task_import.value;self.assertEqual(len(value['rows']),768);self.assertEqual(len({r['guard_metadata']['component_id'] for r in value['rows']}),192)
        self.assertEqual(value['declared_domains'],self.task_domains)
        self.assertEqual({r['raw']['finish_reason'] for r in value['rows']},{'eos','generation_cap'})
        self.assertEqual([r['raw'] for r in value['rows']],[r for batch in self.task_batches for r in batch['outputs']])
        self.assertFalse(value['native_launch_authorized']);self.assertFalse(value['quality_gain_established'])
    def test_whole_long_roster_original_strata_and_external_domain_retained(self):
        value=self.long_import.value;self.assertEqual(len(value['rows']),2304);self.assertEqual(len({r['guard_metadata']['component_id'] for r in value['rows']}),192)
        self.assertEqual(value['declared_domains'],self.long_domains)
        self.assertEqual({r['raw']['context_cap'] for r in value['rows']},{1024,2048,4096})
        self.assertEqual([r['guard_metadata'] for r in value['rows']],self.long_meta)
        self.assertFalse(value['scoring_authorized']);self.assertFalse(value['quality_gain_established'])
    def test_original_enrollment_reproduces_source_correct_membership_missing_language(self):
        result=self.enrollment(self.evidence());enrollment=result['enrollment']
        self.assertEqual(enrollment['status'],'INCONCLUSIVE');self.assertEqual(enrollment['issues'],['MISSING_REQUIRED_CLASS:language']);self.assertEqual(enrollment['missing_m0_row_ids'],[])
        self.assertEqual(len(enrollment['m0_scores']),3072);self.assertTrue(any(v is False for v in enrollment['m0_scores'].values()))
        originals={d['domain_id']:d for d in self.task_domains+self.long_domains}
        for domain in enrollment['declared_domains']:
            original=originals[domain['domain_id']];self.assertEqual(domain['class'],original['class'])
            self.assertEqual(set(zip(domain['row_ids'],domain['component_ids'])),set(zip(original['row_ids'],original['component_ids'])))
        for domain in enrollment['domains']:
            enrolled={row for rows in domain['eligible_components'].values() for row in rows}
            self.assertEqual(enrolled,{row for row in domain['row_ids'] if enrollment['m0_scores'][row] is True})
        self.assertFalse(result['native_membership_admitted']);self.assertFalse(result['human_language_qualification_established'])
    def test_public_native_gates_and_forged_capability_before_helpers(self):
        with patch.object(self.T,'_helpers',side_effect=AssertionError('no helper/input reads')):
            for call in (self.T.prepare_task_import,self.T.prepare_long_import,self.T._original_native_task_batches):
                with self.subTest(call=call.__name__),self.assertRaisesRegex(ValueError,'gate closed before reads'):call([str(PACKAGE/'weights.safetensors')])
            with self.assertRaisesRegex(ValueError,'private authored CPU'):self.T._CPU_import(self.task_source,self.task_capture,_capability=object())
    def test_original_native_treatment_import_rejection_retained(self):
        registration=self.adapter._seal({'scope':self.adapter.NATIVE})
        for arm in ('M1','M2'):
            with self.subTest(arm=arm),self.assertRaisesRegex(ValueError,'native M1/M2 collector/audit schema unregistered'):
                self.adapter.prepare_import(registration,[],arm=arm)
    def test_missing_extra_and_wrong_batch_ordinals_reject(self):
        for change in (lambda b:b.pop(),lambda b:b.append(b[-1]),lambda b:b[0].update(ordinal=True),lambda b:b[1].update(ordinal=1),lambda b:b[0].update(source_kind='atomic_RS_DEV')):
            batches=deepcopy(self.task_batches);change(batches);capture=self.T._CPU_capture(self.task_source,batches,_capability=self.cap)
            with self.subTest(),self.assertRaises(ValueError):self.T._CPU_import(self.task_source,capture,_capability=self.cap)
    def test_wrong_row_order_short_batch_and_prompt_binding_reject(self):
        changes=(lambda b:b[0]['outputs'].reverse(),lambda b:b[0]['outputs'].pop(),lambda b:b[0]['outputs'][0].update(prompt_sha256='0'*64),lambda b:b[0]['outputs'][0].update(input_row_sha256='0'*64),lambda b:b[0]['outputs'][0].update(source_packet_sha256='0'*64))
        for change in changes:
            batches=deepcopy(self.task_batches);change(batches);capture=self.T._CPU_capture(self.task_source,batches,_capability=self.cap)
            with self.subTest(),self.assertRaises(ValueError):self.T._CPU_import(self.task_source,capture,_capability=self.cap)
    def test_partial_native_or_bad_trace_finish_capture_reject(self):
        changes=(lambda r:r.update(scope='UNQUALIFIED_NATIVE_IMPLEMENTATION'),lambda r:r.update(status='RAW_TOKEN_PARTIAL_UNSCORED'),lambda r:r.update(oracle_inputs_read=True),lambda r:r.update(finish_reason='eos'),lambda r:r['trace'][0].update(prefix_length=3),lambda r:r['trace'][0].update(logprob=float('inf')))
        for change in changes:
            batches=deepcopy(self.task_batches);change(batches[0]['outputs'][0])
            if any(not __import__('math').isfinite(t['logprob']) for t in batches[0]['outputs'][0]['trace']):
                with self.assertRaises(ValueError):self.T._CPU_capture(self.task_source,batches,_capability=self.cap)
            else:
                capture=self.T._CPU_capture(self.task_source,batches,_capability=self.cap)
                with self.subTest(),self.assertRaises(ValueError):self.T._CPU_import(self.task_source,capture,_capability=self.cap)
    def test_task_component_domain_and_category_strata_reject(self):
        for change in (lambda m:m[0].update(component_id=m[4]['component_id']),lambda m:m[0].update(mechanism='language'),lambda m:m[0].update(rendering='invented'),lambda m:m[0].update(extra='source drift')):
            metadata=deepcopy(self.task_meta);change(metadata)
            with self.subTest(),self.assertRaises(ValueError):self.T._CPU_source('task',self.task_rows,metadata,self.task_domains,_capability=self.cap)
        domains=deepcopy(self.task_domains);domains[0]['domain_id']='invented_task'
        with self.assertRaisesRegex(ValueError,'original task guard domain'):self.T._CPU_source('task',self.task_rows,self.task_meta,domains,_capability=self.cap)
    def test_long_external_domain_and_joined_world_strata_reject(self):
        with self.assertRaisesRegex(ValueError,'externally frozen'):self.T._CPU_source('long',self.long_rows,self.long_meta,[],_capability=self.cap)
        for change in (lambda m:m[0].update(total_sequence_envelope=512),lambda m:m[0].update(answerable=False),lambda m:m[0].update(placement='late'),lambda m:m[0].update(variant='invented')):
            metadata=deepcopy(self.long_meta);change(metadata)
            with self.subTest(),self.assertRaises(ValueError):self.T._CPU_source('long',self.long_rows,metadata,self.long_domains,_capability=self.cap)
        domains=deepcopy(self.long_domains);first=domains[0];domains.append({**first,'domain_id':'CPU_OTHER_FROZEN_LONG_DOMAIN','row_ids':[first['row_ids'].pop()],'component_ids':[first['component_ids'].pop()]})
        with self.assertRaisesRegex(ValueError,'cannot cross declared domains'):self.T._CPU_source('long',self.long_rows,self.long_meta,domains,_capability=self.cap)
    def test_exact_domain_and_source_evidence_reject(self):
        proofs=self.evidence()
        for change in (lambda e:e.pop(self.T.TASK_DOMAIN),lambda e:e.update(extra=next(iter(e.values()))),lambda e:e[self.T.TASK_DOMAIN].update(source_sha256='0'*64)):
            selected=deepcopy(proofs);change(selected)
            with self.subTest(),self.assertRaises(ValueError):self.enrollment(selected)
    def test_stale_source_labels_or_qualification_bytes_reject(self):
        for key in ('scorer_source','labels','qualification'):
            selected=self.evidence();path=Path(selected[self.T.TASK_DOMAIN][key]['path']);path.write_bytes(b'CHANGED_AUTHORED_EVIDENCE')
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'evidence byte identity'):self.enrollment(selected)
    def test_wrong_label_population_and_unqualified_or_wrong_domain_receipt_reject(self):
        for key,change in (('labels',lambda v:v['rows'].pop(next(iter(v['rows'])))),('labels',lambda v:v.update(source_sha256='0'*64)),('qualification',lambda v:v.update(qualified=False)),('qualification',lambda v:v.update(domain_id='wrong')),('qualification',lambda v:v.update(native_grader_qualified=True))):
            selected=self.evidence();ref=selected[self.T.TASK_DOMAIN][key];value=self.adapter._json(Path(ref['path']).read_bytes());change(value);selected[self.T.TASK_DOMAIN][key]=self.ref('changed_'+key+'.json',value)
            with self.subTest(key=key),self.assertRaises(ValueError):self.enrollment(selected)
    def test_semantically_wrong_scorer_rejects_reproduced_negative_control(self):
        selected=self.evidence();proof=selected[self.T.TASK_DOMAIN];code=b'def score_fixture(response, label):\n    return True\n';path=Path(proof['scorer_source']['path']);path.write_bytes(code);proof['scorer_source'].update(bytes=len(code),sha256=hashlib.sha256(code).hexdigest())
        qualifier=self.adapter._json(Path(proof['qualification']['path']).read_bytes());qualifier['scorer_source_sha256']=proof['scorer_source']['sha256'];proof['qualification']=self.ref('bad_scorer_qualification.json',qualifier)
        with self.assertRaisesRegex(ValueError,'qualification reproduces'):self.enrollment(selected)
    def test_actual_public_data_and_model_read_paths_are_trapped(self):
        paths=[PACKAGE/'DEV_STATISTICS_POLICY.json',PACKAGE/'TASK_RETENTION_GUARD_POLICY.json',PACKAGE/'evidence/long_context_guards_root1_20261004/policy_inputs.jsonl',PACKAGE/'data/crossed_source2_v1_20261003/train.json',PACKAGE/'weights.safetensors']
        for path in paths:
            for reader in (lambda p:builtins.open(p,'rb'),lambda p:io.open(p,'rb'),lambda p:p.open('rb'),lambda p:p.read_bytes()):
                with self.subTest(path=path.name),self.assertRaisesRegex(AssertionError,'non-authored guard/data/model read forbidden'):reader(path)

if __name__=='__main__':unittest.main()
