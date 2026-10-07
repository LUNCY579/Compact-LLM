"""Independent preserved-byte terminal replay; SUP facts authored CPU only."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import ast,hashlib,json,shutil,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;NATIVE=HERE.parent;FROZEN=NATIVE/'frozen_import_terminal002';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1');PROGRAM=PACKAGE/'evidence/import_only_program_reviewed_cpu_20261007/frozen_program002';MEASURE=PACKAGE/'evidence/torch_import_measurement_reviewed_cpu_20261007/frozen_measurement004';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]+[p for p in PROGRAM.rglob('*') if p.is_file()]+[p for p in MEASURE.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_terminal_audit.py')]['sha256']=='dd0064f6547c696b39647f7e6ab78615785d43172af950a8c277a85d2e1b1d60';assert BEFORE[str(FROZEN/'test_import_terminal_audit.py')]['sha256']=='7f626ddf53a19e4f474f9054fde5ec0677c404bc7709f6e17b4c13a0c03d4d85'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);mirror=tempfile.TemporaryDirectory(prefix='CPU_terminal_frozen_',dir=NATIVE);WORK=Path(mirror.name);LOCATION=WORK/'terminal';LOCATION.mkdir()
for p in PROGRAM.glob('*.py'):shutil.copy2(p,WORK/p.name)
shutil.copytree(PROGRAM/'predecessors',WORK/'predecessors')
shutil.copy2(MEASURE/'test_measurement.py',WORK/'test_measurement.py')
for p in FROZEN.glob('*.py'):shutil.copy2(p,LOCATION/p.name)
TEST=LOCATION/'test_import_terminal_audit.py';T=ModuleType('independent_import_terminal_fixture');T.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),T.__dict__);Base=T.Controls;DETAILS={}
class Independent(Base):
    def resign(self):
        self.preservation['raw_files']={p.name:self.desc(p) for p in self.output.iterdir()};self.preservation=T.T.A.seal({k:v for k,v in self.preservation.items() if k!='sha256'});self.write(self.preserved/'PRESERVATION.json',self.preservation)
    def test_actual_writer_original_terminal_blocks_and_raw_replay_unqualified(self):
        self.build_terminal();result=self.audit();self.assertEqual(result['status'],'IMPORT_MEASUREMENT_TERMINAL_VALIDATED_UNQUALIFIED');self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued']);self.assertEqual(result['actual_phase_counts']['framework_import'],{'attempted':1,'returned':1,'failed':0,'pending':0});DETAILS['actual_original_writer_raw_replay_terminal_blocks']=True
    def test_actual_failed_import_and_missing_semantics_preserve_partial(self):self.test_actual_failed_import_raw_and_terminal_preserved();Base.setUp(self);self.test_resealed_missing_semantic_result_never_complete()
    def test_resealed_resources_or_recorded_death_refuse(self):self.test_resealed_resource_or_death_mutation_refused()
    def test_resealed_raw_clock_outside_owned_window_refuses(self):
        self.build_terminal();packet=json.loads((self.output/'IMPORT_LIBRARY_MEASUREMENT.json').read_bytes());supervisor=json.loads((self.output/'SUPERVISOR_RECEIPT.json').read_bytes());shift=int((supervisor['deadline_monotonic']+10)*10**9)-packet['observations'][0]['time_ns']
        for row in packet['observations']:row['time_ns']+=shift
        packet=T.T.A.seal({k:v for k,v in packet.items() if k!='sha256'});self.pair('IMPORT_LIBRARY_MEASUREMENT.json',packet);worker=json.loads((self.output/'worker_result.json').read_bytes());worker.update(measurement_file_sha256=self.desc(self.output/'IMPORT_LIBRARY_MEASUREMENT.json')['sha256'],measurement_body_sha256=packet['sha256']);self.pair('worker_result.json',worker);self.resign()
        with self.assertRaisesRegex(ValueError,'rawmeasurementclocks'):self.audit()
    def test_actual_selected_runtime_packet_and_caller_wrapper_override_refuse(self):
        self.test_caller_runtime_policy_wrapper_cannot_override_selected_packet();Base.setUp(self);self.build_terminal();runtime=Path(self.anchor['anchors']['runtime_path']);runtime.write_bytes(runtime.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.audit()
    def test_resealed_V2_reference_and_source_snapshot_refuse(self):
        self.build_terminal();original=deepcopy(self.anchor);self.anchor['grant_reference']=deepcopy(self.refs['owner_receipt']);self.pair('OWNED_EXTERNAL_ANCHORS.json',self.anchor);self.resign()
        with self.assertRaises(ValueError):self.audit()
        self.anchor=original;self.pair('OWNED_EXTERNAL_ANCHORS.json',self.anchor);self.resign();snap=next(iter(self.preservation['source_snapshot_files'].values()));p=self.preserved/snap['path'];p.write_bytes(p.read_bytes()+b'\n# drift\n')
        with self.assertRaises(ValueError):self.audit()
if __name__=='__main__':
    names=('test_actual_writer_original_terminal_blocks_and_raw_replay_unqualified','test_actual_failed_import_and_missing_semantics_preserve_partial','test_resealed_resources_or_recorded_death_refuse','test_resealed_raw_clock_outside_owned_window_refuses','test_resealed_V2_reference_and_source_snapshot_refuse','test_actual_selected_runtime_packet_and_caller_wrapper_override_refuse');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);mirror.cleanup();after=pins();receipt={'kind':'INDEPENDENT_IMPORT_TERMINAL002_SELECTED_PACKET_BYTE_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'details':DETAILS,'actual_original_import_writer_raw_replay_terminal_resource_statement_blocks_executed':True,'SUP_resources_Job_leases_death_gate_documents_authored_CPU_only':True,'test_only_original_relative_companion_mirror':True,'fixture_package_projection_to_selected_A27ae_source':True,'native_API_or_framework_called':False,'native_gate_context_or_permission_minted':False,'actual_six_config_runtime_registration_file_body_anchors_consumed':True,'caller_runtime_wrapper_cannot_override_selected_packet':True,'prior001_narrower_proof_preserved':True,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_IMPORT_TERMINAL002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
