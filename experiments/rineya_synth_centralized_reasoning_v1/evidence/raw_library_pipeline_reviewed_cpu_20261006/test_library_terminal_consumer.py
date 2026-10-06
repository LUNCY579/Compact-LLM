"""Independent terminal consumer byte/source/anchor joins using CPU raw fixtures."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import hashlib,unittest
import test_library_failure_evidence as F
from test_full_native_cpu_workflow import compiled
HERE=Path(__file__).resolve().parent
class LibraryTerminalConsumerControls(unittest.TestCase):
    def setUp(self):
        self.f=F.ObservedLibraryFailureControls();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.A=compiled(HERE/'prefix_input_owned_auditor.py');self.E=F.E
        path=self.f.base.root/'prefix_library_evidence.py';path.write_bytes((HERE/path.name).read_bytes())
        self.sources=deepcopy(self.f.sources);self.sources[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'role':'python_source'}
        self.context={'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'expected_head':'a'*40,
            'config_path':str(self.f.base.root/'configs/CPU.json'),'anchors':{'expected_config_file':'c'*64},
            'config':{'source_files':self.sources},'runtime':{'library_boundary_plan':deepcopy(self.f.plan),'library_search_policy':{'enforced_boundary':deepcopy(self.f.policy)}}}
        worker=self.f.base.root/'prefix_input_owned_worker.py';worker.write_bytes((HERE/worker.name).read_bytes())
        self.sources[str(worker)]={'bytes':worker.stat().st_size,'sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'role':'python_source'}
        self.worker_source={'path':str(worker),'descriptor':self.sources[str(worker)]}
        self.owner={'worker_pid':self.f.identity['owned_birth']['pid']};self.birth=deepcopy(self.f.identity['owned_birth'])
    def envelope(self):
        packet=self.f.packet();packet['identity'].update(context_sha256=self.E.digest(self.context),producer_source=deepcopy(self.worker_source),owner_receipt_sha256=self.E.digest(self.owner))
        packet['sha256']=self.E.digest({k:v for k,v in packet.items() if k!='sha256'})
        env={'schema_version':1,'kind':'OWNED_LIBRARY_BOUNDARY_EVIDENCE_V1','source_commit':'a'*40,'original_scope':self.A.NATIVE_SCOPE,'context_sha256':self.E.digest(self.context),'external_anchors':self.context['anchors'],'policy_sha256':self.E.digest(self.f.policy),'provider_source':deepcopy(self.f.policy['provider_source']),'owner_receipt_sha256':self.E.digest(self.owner),'raw_packet':packet,'raw_packet_sha256':self.E.digest(packet),'main_errors':[],'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
        return env
    def run_consumer(self,env):
        env['sha256']=self.E.digest({k:v for k,v in env.items() if k!='sha256'})
        raw=(self.E.canonical(env)+'\n').encode('ascii');reader=self.A.Reader(self.f.base.root)
        result=self.A._audit_library_evidence_bytes(raw,selected_context=self.context,owner_receipt=self.owner,executing_birth=self.birth,reader=reader);reader.verify();return result
    def test_actual_selected_consumer_source_and_raw_replay_remain_unqualified(self):
        r=self.run_consumer(self.envelope());self.assertTrue(r['actual_packet_bytes_consumed']);self.assertTrue(r['conservation_complete'])
        self.assertFalse(r['native_PASS']);self.assertFalse(r['native_observer_qualified'])
    def test_resealed_context_owner_and_policy_substitution_refused(self):
        for key in ('context_sha256','owner_receipt_sha256','policy_sha256'):
            env=self.envelope();env[key]='0'*64
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_consumer(env)
    def test_resealed_nested_raw_digest_and_executing_birth_substitution_refused(self):
        env=self.envelope();env['raw_packet_sha256']='0'*64
        with self.assertRaises(ValueError):self.run_consumer(env)
        env=self.envelope();env['raw_packet']['identity']['owned_birth']['pid']+=1
        env['raw_packet']['sha256']=self.E.digest({k:v for k,v in env['raw_packet'].items() if k!='sha256'});env['raw_packet_sha256']=self.E.digest(env['raw_packet'])
        with self.assertRaises(ValueError):self.run_consumer(env)
    def test_independently_selected_plan_not_producer_claim_is_enforced(self):
        env=self.envelope();self.context['runtime']['library_boundary_plan'][1]['name']='extension:'+self.f.base.names['ext.pyd']+':exec'
        env['context_sha256']=self.E.digest(self.context);env['raw_packet']['identity']['context_sha256']=self.E.digest(self.context)
        env['raw_packet']['sha256']=self.E.digest({k:v for k,v in env['raw_packet'].items() if k!='sha256'});env['raw_packet_sha256']=self.E.digest(env['raw_packet'])
        with self.assertRaises(ValueError):self.run_consumer(env)
    def test_actual_consumer_bytes_drift_refused(self):
        env=self.envelope();p=Path(self.context['config_path']).parent.parent/'prefix_library_evidence.py';raw=p.read_bytes();p.write_bytes(b'#'+raw[1:])
        with self.assertRaises(ValueError):self.run_consumer(env)
    def test_clock_float_interval_conversion_uses_exact_integer_ratios(self):
        from fractions import Fraction
        import math
        for start,end in ((0.0,1.0),(1234567.123456789,1235167.123456789),(2**40+.125,2**40+.875)):
            low,high=self.A._library_clock_window_ns(start,end)
            exact_low=Fraction.from_float(math.nextafter(start,-math.inf))*10**9
            exact_high=Fraction.from_float(math.nextafter(end,math.inf))*10**9
            self.assertEqual(low,exact_low.numerator//exact_low.denominator)
            self.assertEqual(high,-((-exact_high.numerator)//exact_high.denominator))
            self.assertLessEqual(Fraction(low),exact_low)
            self.assertGreaterEqual(Fraction(high),exact_high)
    def test_main_error_prevents_full_library_completion_without_hiding_raw(self):
        env=self.envelope();env['main_errors']=[{'stage':'owned_return','error':{'type':'OSError','message':'inert writer failed','message_exact':True}}]
        r=self.run_consumer(env);self.assertFalse(r['conservation_complete']);self.assertTrue(r['replay']['conservation_complete']);self.assertEqual(r['main_errors'],env['main_errors'])

if __name__=='__main__':unittest.main()
