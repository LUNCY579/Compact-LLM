from pathlib import Path
from types import ModuleType
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;M=ModuleType('CPU_import_resource_guard');M.__file__=str(HERE/'import_resource_guard.py');exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
class Trap:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('torch','ctypes','psutil','native_worker'):raise AssertionError('native import '+name)
class Controls(unittest.TestCase):
    def setUp(self):self.now=10.;self.wall=100.;self.rss=1024**3;self.ram=2*1024**3;self.beat={'unix_seconds':100.,'supervisor_pid':2};self.calls=[];self.trap=Trap();sys.meta_path.insert(0,self.trap)
    def tearDown(self):sys.meta_path.remove(self.trap)
    def guard(self):return M.bind(HERE,{'job_name':'CPU_JOB','supervisor_pid':2},10.,owned_resources=lambda name:self.calls.append(name) or {'tree_rss_bytes':self.rss},heartbeat_reader=lambda path:self.beat,available=lambda:self.ram,monotonic=lambda:self.now,wall=lambda:self.wall)
    def test_original_source_guards_without_framework_view(self):self.guard()('before_import');self.guard()('after_import');self.assertEqual(self.calls,['CPU_JOB','CPU_JOB'])
    def test_original_deadline_RSS_and_RAM_limits(self):
        for kind in ('deadline','rss','ram'):
            self.now=10.;self.rss=1024**3;self.ram=2*1024**3
            if kind=='deadline':self.now=611.
            elif kind=='rss':self.rss=2*1024**3+1
            else:self.ram=int(1.25*1024**3)-1
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.guard()('guard')
    def test_original_heartbeat_identity_freshness(self):
        for beat in ({'unix_seconds':97.,'supervisor_pid':2},{'unix_seconds':101.,'supervisor_pid':2},{'unix_seconds':100.,'supervisor_pid':3}):
            self.beat=beat
            with self.assertRaises(ValueError):self.guard()('guard')
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
v={'kind':'ROOT_CPU_ORIGINAL_IMPORT_RESOURCE_GUARD_CONTROLS','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'actual_pinned_original_make_guard_executed':True,'inert_Job_RAM_heartbeat_suppliers':True,'Torch_view_permanently_None':True,'native_API_framework_called':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'import_resource_guard.py',Path(__file__),M.SOURCE)}}
(HERE/'ROOT_IMPORT_RESOURCE_GUARD_CONTROLS001.json').write_bytes((json.dumps(v,sort_keys=True,indent=2)+'\n').encode());raise SystemExit(0 if r.wasSuccessful() else 1)
