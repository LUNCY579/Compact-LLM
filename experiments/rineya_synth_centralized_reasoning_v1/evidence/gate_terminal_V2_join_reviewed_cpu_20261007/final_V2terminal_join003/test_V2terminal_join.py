"""Full V2 record through actual preserved-body replay; CPU metadata only."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[2]
WORK=PACKAGE/'artifacts/tensor_observed_admission_work'
def load(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
B=load(WORK/'test_native_entry_auditor.py','CPU_V2_native_schema_fixture')
A=load(HERE/'prefix_input_owned_auditor.py','CPU_V2_selected_auditor');A.PACKAGE=PACKAGE;B.A=A;B.BASE.A=A
class Controls(B.Controls):
    def prepare(self):
        self.fixture();self.binding={'config_path':str(self.root/'CPU_V2_config.json'),'anchors':{'expected_config_file':'b'*64,'expected_config_body':'c'*64,'registration_path':str(self.root/'registration.json'),'expected_registration_file':'d'*64,'expected_registration_body':'e'*64,'runtime_path':str(self.root/'runtime.json'),'expected_runtime_file':'f'*64,'expected_runtime_body':'1'*64,'expected_head':'a'*40}};self.references={}
        for name in ('grant','owner_receipt'):
            document=A.seal({'kind':'EXPLICIT_CPU_V2_GATE_REFERENCE_DOCUMENT','fixture_only':True,'native_authority_issued':False,'role':name});path=self.root/(name+'.json');B.BASE.write(path,document);self.references[name]={'path':str(path),'file_sha256':B.BASE.descriptor(path)['sha256'],'body_sha256':document['sha256']}
        self.anchors={'schema_version':2,**deepcopy(self.binding),'grant_reference':deepcopy(self.references['grant']),'owner_receipt_reference':deepcopy(self.references['owner_receipt'])};self.rewrite_pair('OWNED_EXTERNAL_ANCHORS.json',self.anchors);self.resign()
    def resign(self):
        for p in self.output.iterdir():self.preservation['raw_files'][p.name]=B.BASE.descriptor(p)
        self.preservation=A.seal(self.preservation);B.BASE.write(self.preserved/'PRESERVATION.json',self.preservation)
    def audit(self):
        return A._native_preserved_body(self.preserved,original_directory=self.output,manifest=self.manifest,source_pins=self.manifest['source_files'],expected_preservation_file_sha256=B.BASE.descriptor(self.preserved/'PRESERVATION.json')['sha256'],expected_preservation_body_sha256=self.preservation['sha256'],expected_head='a'*40,anchors=self.anchors,_CPU_fixture_capability=A._mint_CPU_native_schema_fixture(),_expected_gate_references=self.references,_expected_external_binding=self.binding)
    def test_full_record_actual_native_preserved_body_CPU_replay(self):
        self.prepare();v=self.audit();self.assertEqual(v['status'],'CPU_NATIVE_SCHEMA_COMPLETE');self.assertFalse(v['native_PASS'])
    def test_missing_schema_and_rebound_reference_refuse_after_reseal(self):
        self.prepare();self.anchors.pop('schema_version');self.rewrite_pair('OWNED_EXTERNAL_ANCHORS.json',self.anchors);self.resign()
        # A V2 record missing its discriminator cannot silently become legacy.
        with self.assertRaises(ValueError):self.audit()
    def test_rebound_actual_gate_document_refused(self):
        self.prepare();self.anchors['grant_reference']=deepcopy(self.references['owner_receipt']);self.rewrite_pair('OWNED_EXTERNAL_ANCHORS.json',self.anchors);self.resign()
        with self.assertRaisesRegex(ValueError,'cannot rebind|separate original grant'):self.audit()
    def test_ninth_anchor_rebind_and_document_bytes_refused(self):
        self.prepare();self.anchors['anchors']['expected_runtime_body']='2'*64;self.rewrite_pair('OWNED_EXTERNAL_ANCHORS.json',self.anchors);self.resign()
        with self.assertRaisesRegex(ValueError,'nine-anchor'):self.audit()
        self.anchors=deepcopy({'schema_version':2,**self.binding,'grant_reference':self.references['grant'],'owner_receipt_reference':self.references['owner_receipt']});self.rewrite_pair('OWNED_EXTERNAL_ANCHORS.json',self.anchors);self.resign();Path(self.references['grant']['path']).write_bytes(b'{}\n')
        with self.assertRaises(ValueError):self.audit()
if __name__=='__main__':
    methods=('test_full_record_actual_native_preserved_body_CPU_replay','test_missing_schema_and_rebound_reference_refuse_after_reseal','test_rebound_actual_gate_document_refused','test_ninth_anchor_rebind_and_document_bytes_refused');result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(name) for name in methods));v={'kind':'CPU_GATE_V2_TERMINAL_RECORD_JOIN_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_native_preserved_body_called':True,'SUP_death_and_gate_documents_explicit_CPU_metadata':True,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('prefix_input_owned_auditor.py','test_V2terminal_join.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
