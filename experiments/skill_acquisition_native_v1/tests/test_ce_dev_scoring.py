"""Real retained synthetic token bytes scored on exposed DEV metadata only."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_dev_scoring as d, ce_dev_scope as scope
from experiments.skill_acquisition_native_v1.tests.test_ce_dev_runtime import context_fixture


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[3]/'experiments/skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.json'
        ref={'path':str(path),'sha256':scope.registration.PROPOSAL_SHA,'bytes':path.stat().st_size}
        proposal,banks,policy=scope.load_metadata(ref)
        cls.docs=scope.build_documents(proposal,banks,policy_ref=policy)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
        context=context_fixture();actual=copy.deepcopy(self.docs['DEV_EVALUATION_SCOPE.json'])
        actual.update(evaluation_authorized=True,approved=True)
        for name in actual['model_bindings']:
            actual['model_bindings'][name].update(context['scope']['model_bindings'][name])
        context['scope']=actual;context['scope_sha256']=d.digest(actual)
        context['plan']['scope_sha256']=context['scope_sha256']
        context['runtime']['decoding_preimage']['attn_implementation']='eager'
        context['runtime']['decoding_sha256']=d.digest(context['runtime']['decoding_preimage'])
        for binding in actual['model_bindings'].values():binding['decoding_sha256']=context['runtime']['decoding_sha256']
        context['scope_sha256']=d.digest(actual);context['plan']['scope_sha256']=context['scope_sha256']
        context['source']={'code_sha256':{'explicit_fixture.py':'a'*64}}
        identity={'schema':d.worker.IDENTITY,'pid':123,'creation_filetime_100ns':456,'boot_identity':'windows-boot-guid-explicit-fixture'}
        ref={'path':str(self.directory/'fixture.py'),'sha256':'b'*64,'bytes':1}
        sources={'python_ref':ref,'process_image_ref':ref,'request_ref':ref,'source_files':{'explicit_fixture.py':dict(ref,sha256='a'*64)},
            'entry':'explicit_fixture.py','entry_script_ref':None}
        worker_sha=d.digest(d.worker.worker_identity_preimage(identity,sources,context['request_sha256'],context['scope_sha256']))
        proof={'schema':d.worker.PROOF,'worker_identity_sha256':worker_sha,'os_identity':identity,'sources':sources,
            'request_sha256':context['request_sha256'],'scope_sha256':context['scope_sha256'],'native_qualified':False,
            'execution_verified':False,'fresh_process':False,'classification':'explicit synthetic metadata, no OS observation'}
        contract={'transformers':'4.57.1','max_new_tokens':128,'eos_token_ids':[151645,151643],'pad_token_id':151643,
            'use_model_defaults':False,'logits_processors':['_RawStepRecorder'],'stopping_criteria':['MaxLengthCriteria','EosTokenCriteria'],
            'greedy':{'do_sample':False,'num_beams':1,'batch_size':1,'use_cache':True,
                'raw_logits_source':'generate(output_logits=True) float32 step logits, cross-checked bitwise against the pass-through recorder'}}
        contract['contract_sha256']=d.digest(contract)
        self.body={'schema':d.runtime.READ,'status':'DEV_BODY_COMPLETE','read_id':'Supa-read-1','model_name':'Supa','read_index':1,
            'scope_sha256':context['scope_sha256'],'model_sha256':context['model']['Supa_sha256'],
            'deployed_state_fingerprint_sha256':context['model']['Supa_state_fingerprint_sha256'],
            'tokenizer_sha256':context['model']['tokenizer_receipt_ref']['sha256'],'decoding_sha256':context['runtime']['decoding_sha256'],
            'source':context['source'],'generation_contract':contract,'worker_proof':proof,'process_identity_sha256':worker_sha,
            'model_fingerprint_unchanged':True,'rows':1950,'greedy':d.runtime.GREEDY,'model_load_order':['base_FP32'],
            'os_terminal_verified':False,'native_qualified':False,'learning_claim':False,
            'token_journal':str(self.directory/'TOKEN_JOURNAL.jsonl'),'attempts_journal':str(self.directory/'ATTEMPTS.jsonl')}
        self.context=context;self.binding=d._binding(context,proof)
        self.completion={k:self.body[k] for k in ('read_id','model_name','read_index','scope_sha256','model_sha256',
            'tokenizer_sha256','decoding_sha256','greedy','source')}
        self.completion.update(completed_pair_verified=False,synthetic_pair_consistent=True,execution_reverified=False,
            verification_binding=self.binding,completion_ref=None,cost_seal_ref=None)
        self.completion['context_ref']=self.write('CONTEXT.json',context)
        self.completion['body_ref']=self.write('DEV_BODY_COMPLETE.json',self.body)
        self.completion['scope_receipt_ref']=self.write('DEV_RSP_SCOPE_RECEIPT.json',self.docs['DEV_RSP_SCOPE_RECEIPT.json'])
        _,bind=d._body(context,self.body,self.completion,True)
        self.rows=[];process={'process_identity_sha256':worker_sha,'os_identity':identity}
        decoder={'mode':'greedy','transformers':'4.57.1','attn_implementation':'eager','raw_logits_source':'output_logits',
            'argmax_replay_verified':True,'logits_processors':['_RawStepRecorder'],
            'stopping_criteria':['MaxLengthCriteria','EosTokenCriteria'],
            'effective_config':{'do_sample':False,'num_beams':1,'num_return_sequences':1,'max_new_tokens':128,'max_length':133,
                'eos_token_id':[151645,151643],'pad_token_id':151643,'use_cache':True,'output_logits':True,
                'output_scores':False,'return_dict_in_generate':True}}
        for name in d.runtime.BANKS:
            bank=actual['banks'][name]
            for rid in bank['ids']:
                row=dict(bind,id=rid,index=len(self.rows),bank=name,parent_id=bank['parent_by_id'][rid],
                    prompt_hash=bank['prompt_hash_by_id'][rid],prompt_ids_sha256='e'*64,render_sha256='f'*64,prompt_tokens=5,
                    completion_ids=[42,151645],generated_tokens=2,stop_reason='eos',response='explicit synthetic token fixture',
                    raw_step_logprobs=[-1.,-2.],body_special_ids=[],body_unknown_ids=[],wall_s=.001,process=process,
                    decoder=copy.deepcopy(decoder))
                self.rows.append(row)
        self.attempts=[dict(bind,event=worker_sha+':start',kind='start',worker_proof=proof,process=process,resume_index=0),
            dict(bind,event=worker_sha+':end',kind='end',outcome='COMPLETE',rows=1950,process=process)]
        self.save_journals()

    def write(self,name,value):
        blob=d.canonical(value)+b'\n';path=self.directory/name;path.write_bytes(blob)
        return {'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
    def journal(self,name,rows):
        blob=b''.join(d.canonical(dict(r,row_sha256=d.digest(r)))+b'\n' for r in rows)
        path=self.directory/name;path.write_bytes(blob)
        return {'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
    def save_journals(self):
        self.completion['token_journal_ref']=self.journal('TOKEN_JOURNAL.jsonl',self.rows)
        self.completion['attempts_journal_ref']=self.journal('ATTEMPTS.jsonl',self.attempts)
    def readers(self):
        bounds={key:10**13 for key in d.raw.BUDGET_FIELDS};events=[]
        operation=lambda *args,**fields:events.append((args,fields))
        root=Path(self.context['scope']['metadata_refs']['dataset_manifest']['path']).parent
        return d.raw.RawEvidenceReader(operation,bounds,self.directory),d.raw.RawEvidenceReader(operation,bounds,root),events
    def score(self):
        reader,banks,events=self.readers()
        return d.score_fixture_read(self.completion,reader=reader,bank_reader=banks,outer_preflight=lambda *args:None)

    def test_real1950_journal_four_unchanged_scoring_receipts_match_analyzer_contract(self):
        from experiments.skill_acquisition_native_v1 import ce_dev_analysis as analysis
        result=self.score();self.assertEqual(result['rows'],1950);self.assertEqual(list(result['reads']),list(d.runtime.BANKS))
        for name,read in result['reads'].items():
            self.assertEqual(len(analysis._read(read,self.context['scope'],name,'Supa')),len(self.context['scope']['banks'][name]['ids']))
        self.assertFalse(result['completion_pair_verified']);self.assertFalse(result['native_qualified'])
        self.assertFalse(result['execution_reverified']);self.assertTrue(result['synthetic_pair_consistent'])
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

    def test_default_closed_and_outer_preflight_precedes_any_read(self):
        with patch.object(Path,'open',side_effect=AssertionError('unpaid read')):
            with self.assertRaisesRegex(ValueError,'scorer closed'):
                d.score_completed_dev(None,reader=None,bank_reader=None,outer_preflight=None)
            reader,banks,_=self.readers()
            with self.assertRaisesRegex(ValueError,'denied external'):
                d.score_fixture_read(self.completion,reader=reader,bank_reader=banks,
                    outer_preflight=lambda *args:(_ for _ in ()).throw(ValueError('denied external')))

    def test_body_completion_flag_and_foreign_worker_cannot_score(self):
        self.completion['completed_pair_verified']=True
        with self.assertRaisesRegex(ValueError,'dependency required'):self.score()
        self.completion['completed_pair_verified']=False;self.rows[12]['process']['os_identity']=dict(self.rows[12]['process']['os_identity'],pid=999)
        self.save_journals()
        with self.assertRaisesRegex(ValueError,'original fresh complete|current worker'):self.score()

    def test_actual_supervisor_inner_dict_selection_delegates_source_identity_before_IO(self):
        from experiments.skill_acquisition_native_v1 import ce_dev_supervisor as supervisor
        selected={'read_id':'Supa-read-1','verification_binding':self.binding}
        source=MappingProxyType({'Supa-read-1':selected});reader,banks,_=self.readers()
        with patch.object(supervisor,'_SOURCE_COMPLETIONS',source), \
                patch.object(supervisor,'_SOURCE_COMPLETIONS_SHA256',d.digest(dict(source))), \
                patch.object(Path,'open',side_effect=AssertionError('unpaid read')):
            actual=supervisor.selected_completion('Supa-read-1')
            self.assertIs(actual,selected);self.assertIs(type(actual),dict)
            with self.assertRaisesRegex(ValueError,'actual source paid preflight'):
                d.score_completed_dev(actual,reader=reader,bank_reader=banks,
                    outer_preflight=lambda *args:(_ for _ in ()).throw(ValueError('actual source paid preflight')))
            with self.assertRaisesRegex(ValueError,'exact source-selected DEV completion'):
                d.score_completed_dev(dict(actual),reader=reader,bank_reader=banks,outer_preflight=lambda *args:None)

    def test_rehashed_out_of_order_wrong_parent_and_extra_rows_refuse(self):
        self.rows[0],self.rows[1]=self.rows[1],self.rows[0];self.save_journals()
        with self.assertRaisesRegex(ValueError,'sorted bank'):self.score()
        self.rows[0],self.rows[1]=self.rows[1],self.rows[0];self.rows[0]['parent_id']='foreign';self.save_journals()
        with self.assertRaisesRegex(ValueError,'parent'):self.score()

    def test_rehashed_decoder_sampling_EOS_shift_and_logprob_count_refuse(self):
        for change in (lambda r:r['decoder']['effective_config'].update(do_sample=True),
            lambda r:r['decoder']['effective_config'].update(max_length=132),
            lambda r:r.update(completion_ids=[151645,42]),lambda r:r.update(raw_step_logprobs=[-1.])):
            original=copy.deepcopy(self.rows[7]);change(self.rows[7]);self.save_journals()
            with self.assertRaises(ValueError):self.score()
            self.rows[7]=original

    def test_original_hash_changed_partial_tail_and_missing_end_cannot_score(self):
        ref=self.completion['token_journal_ref'];path=Path(ref['path']);blob=path.read_bytes();path.write_bytes(blob.replace(b'"wall_s":0.001',b'"wall_s":0.002',1))
        self.completion['token_journal_ref']=dict(ref,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError,'row hash mismatch'):self.score()
        self.save_journals();path.write_bytes(path.read_bytes()[:-1]);ref=self.completion['token_journal_ref']
        self.completion['token_journal_ref']=dict(ref,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError,'partial DEV journal tail'):self.score()
        self.attempts.pop();self.save_journals()
        with self.assertRaisesRegex(ValueError,'fresh complete DEV attempt'):self.score()


if __name__=='__main__':unittest.main()
