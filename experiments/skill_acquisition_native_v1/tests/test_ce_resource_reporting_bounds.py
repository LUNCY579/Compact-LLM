"""Scoped synthetic source-graph/stream controls, no hardware/model/native fit."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_native_authority as a
from experiments.skill_acquisition_native_v1 import ce_resource_evidence as r
from experiments.skill_acquisition_native_v1 import ce_resource_qualification as q
from experiments.skill_acquisition_native_v1.tests.test_ce_resource_qualification import isolated_writer

class GraphBoundsTests(unittest.TestCase):
    def test_exact_ascii_extent_unicode_escapes_and_float_representations(self):
        examples=[None,True,False,0,-17,1.0,-0.0,1e-30,
                  {'unicode':'\x7f雪😀','control':'"\\\n\b\t','empty':[]},
                  ['scalar',{'nested':[1,False,None]}]]
        for value in examples:
            with self.subTest(value=value):
                census=r.bounded_json_graph_census(value,maximum_json_bytes=4096)
                self.assertEqual(census['json_bytes'],len(a.canonical(value)))
                self.assertEqual(b''.join(q._ascii_json_chunks(value)),a.canonical(value))
    def test_no_encoding_or_deepcopy_in_census(self):
        with patch.object(json,'dumps',side_effect=AssertionError('census must not encode')),patch.object(copy,'deepcopy',side_effect=AssertionError('census must not copy')):
            self.assertGreater(r.bounded_json_graph_census({'data':['text']*100},maximum_json_bytes=4096)['nodes'],100)
    def test_arbitrary_objects_tuples_and_key_objects_deny(self):
        class Trap:
            def __getattribute__(self,name):raise AssertionError('object introspected')
        for value in (Trap(),(1,),{1:'value'}):
            with self.subTest(value=type(value)),self.assertRaises(ValueError):r.bounded_json_graph_census(value,maximum_json_bytes=4096)
    def test_cycles_depth_nodes_nan_and_huge_int_deny(self):
        cycle=[]; cycle.append(cycle)
        deep=[]; value=deep
        for _ in range(33):child=[];value.append(child);value=child
        for value in (cycle,deep,float('nan'),2**1025):
            with self.subTest(value=type(value)),self.assertRaises(ValueError):r.bounded_json_graph_census(value,maximum_json_bytes=4096)
        with self.assertRaises(ValueError):r.bounded_json_graph_census([0]*10,maximum_json_bytes=4096,maximum_nodes=5)
    def test_extent_refuses_before_copy(self):
        with patch.object(copy,'deepcopy',side_effect=AssertionError('copy before bound')):
            with self.assertRaises(ValueError):r.bounded_json_graph_census('x'*100,maximum_json_bytes=10)
    def test_repeated_references_conservatively_counted(self):
        one={'x':[]}; single=r.bounded_json_graph_census(one,maximum_json_bytes=4096)
        pair=r.bounded_json_graph_census([one,one],maximum_json_bytes=4096)
        self.assertEqual(pair['containers'],1+2*single['containers'])
        self.assertGreater(pair['copied_container_bytes_upper'],2*single['copied_container_bytes_upper'])

class ReportingOrderTests(unittest.TestCase):
    def test_cheap_and_512_gate_precede_full_copy_and_no_snapshot_clone(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp)); events=[]
            original=args[3].gate
            args[3].gate=lambda stage:(events.append(stage),original(stage))[1]
            args[3].gate.spec=original.spec
            args[4].snapshot=lambda:(_ for _ in ()).throw(AssertionError('duplicate full snapshot'))
            real=copy.deepcopy
            def observed_copy(value,*extra,**kw):
                self.assertIn('qualification_reducer_overhead',events)
                self.assertIn('resource_io_measurement',events)
                return real(value,*extra,**kw)
            with patch.object(a,'require_execution_permit',return_value=None),patch.object(copy,'deepcopy',side_effect=observed_copy):
                result=q.resource_adapter(*args)
            self.assertEqual(events,['qualification_reducer_overhead','resource_io_measurement','qualification_artifact'])
            self.assertEqual(json.loads(Path(result['raw_artifacts'][0]['path']).read_bytes())['payload']['raw_operations'],args[4].operations)
    def test_denied_cheap_gate_precedes_every_graph_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp))
            args[3].gate=lambda stage:(_ for _ in ()).throw(RuntimeError('cheap denial'))
            args[3].gate.spec={'host_transient_bytes':{'qualification_reducer_overhead':64*r.CHUNK,'resource_io_measurement':r.REPORT_TRANSIENT_BYTES,'qualification_artifact':r.REPORT_TRANSIENT_BYTES}}
            with patch.object(a,'require_execution_permit',return_value=None),patch.object(r,'bounded_json_graph_census',side_effect=AssertionError('scan before gate')):
                with self.assertRaisesRegex(RuntimeError,'cheap denial'):q.resource_adapter(*args)
            self.assertEqual(args[3].operation.events[-1][:2],('qualification_reducer_overhead','failed'))
    def test_denied_512_gate_precedes_copy_and_filesystem(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp))
            def gate(stage):
                if stage=='resource_io_measurement':raise RuntimeError('512 denial')
            args[3].gate=gate
            gate.spec={'host_transient_bytes':{'qualification_reducer_overhead':64*r.CHUNK,'resource_io_measurement':r.REPORT_TRANSIENT_BYTES,'qualification_artifact':r.REPORT_TRANSIENT_BYTES}}
            with patch.object(a,'require_execution_permit',return_value=None),patch.object(copy,'deepcopy',side_effect=AssertionError('copy before gate')),patch.object(Path,'mkdir',side_effect=AssertionError('IO before gate')):
                with self.assertRaisesRegex(RuntimeError,'512 denial'):q.resource_adapter(*args)
    def test_large_scalar_denies_before_graph_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp)); args[4].operations['unmeasured']={'kind':'other','phase':'failed','error':'x'*(r.CHUNK+1)}
            with patch.object(a,'require_execution_permit',return_value=None),patch.object(copy,'deepcopy',side_effect=AssertionError('copy before scalar bound')):
                with self.assertRaises(ValueError):q.resource_adapter(*args)
    def test_container_transient_denies_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp))
            with patch.object(a,'require_execution_permit',return_value=None),patch.object(r,'REPORT_TRANSIENT_BYTES',1),patch.object(copy,'deepcopy',side_effect=AssertionError('copy before graph RAM bound')):
                with self.assertRaisesRegex(ValueError,'transient'):q.resource_adapter(*args)
    def test_weakened_gate_transient_denies_before_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            args=isolated_writer(Path(temp));args[3].gate.spec['host_transient_bytes']['resource_io_measurement']=64*r.CHUNK
            with patch.object(a,'require_execution_permit',return_value=None),patch.object(r,'bounded_json_graph_census',side_effect=AssertionError('scan before declared gate bounds')):
                with self.assertRaisesRegex(ValueError,'transient gates'):q.resource_adapter(*args)
    def test_streaming_writer_keeps_canonical_bytes_without_whole_buffer(self):
        with tempfile.TemporaryDirectory() as temp:
            value={'keys':['x'*100]*500,'unicode':'雪😀'};expected=a.canonical(value)+b'\n';path=Path(temp)/'resource.json'
            with patch.object(a,'canonical',side_effect=AssertionError('whole JSON encoding')):
                size,sha=q._stream_resource_envelope(path,value,maximum=len(expected),expected_bytes=len(expected))
            self.assertEqual(path.read_bytes(),expected);self.assertEqual(size,len(expected));self.assertEqual(sha,hashlib.sha256(expected).hexdigest())
    def test_snapshot_view_matches_fixed_snapshot_source(self):
        with tempfile.TemporaryDirectory() as temp:
            owner=isolated_writer(Path(temp))[4]
            from experiments.skill_acquisition_native_v1.ce_qualification_accounting import QualificationAccounting
            self.assertEqual(q._snapshot_source_view(owner),QualificationAccounting.snapshot(owner))
            owner.rejection_policy={'fixture':True};owner.rejection_ticket={'reserved':1}
            owner.rejection_used=True;owner.rejection_record={'failure':'retained'}
            owner.rejection_unknown={'spending':'unknown'}
            self.assertEqual(q._snapshot_source_view(owner),QualificationAccounting.snapshot(owner))
    def test_no_native_framework_imports(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

if __name__=='__main__':unittest.main()
