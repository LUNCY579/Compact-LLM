import runpy,pathlib,ast,copy,types,unittest,hashlib,json,subprocess
p=pathlib.Path('experiments/rineya_synth_centralized_reasoning_v1').resolve()
ns=runpy.run_path(str(p/'artifacts/observer_bootstrap_retry_preparation_work/test_bootstrap_v2.py'),run_name='independent_loader')
C,W,A=ns['C'],ns['W'],ns['A'];Base=ns['Controls']
class IndependentControls(Base):
 def test_location_flag_and_missing_path_refused(self):
  m=self.namespace();m.__spec__.has_location=True;self.assert_denied_measure(m)
  m=self.namespace();del m.__path__;self.assert_denied_measure(m)
  m=self.namespace();m.__spec__.submodule_search_locations=None;self.assert_denied_measure(m)
 def test_namespace_origin_metadata_refused(self):
  m=self.namespace();m.__spec__.origin=str(self.ns/'pythoncom311.dll');self.assert_denied_measure(m)
  m=self.namespace();m.__file__='';self.assert_denied_measure(m)
 def test_missing_dll_and_missing_pth_refused(self):
  pathlib.Path(next(iter(self.nsfiles))).unlink();self.assert_denied_measure()
  pathlib.Path(next(iter(self.hooks))).unlink()
  with self.assertRaises(ValueError):C._directory_roster(self.site,self.hooks,suffix='.pth')
 def test_finite_roles_refuse_extra_namespace_and_widened_directory(self):
  self.cache['allowed_namespace_packages']['other']=copy.deepcopy(self.entry)
  with self.assertRaises(ValueError):C._startup_admitted_roles(self.config,self.runtime)
  self.cache['allowed_namespace_packages'].pop('other');self.cache['startup_hook_sources']['directory']=str(self.base)
  with self.assertRaises(ValueError):C._startup_admitted_roles(self.config,self.runtime)
 def test_auditor_refuses_path_disagreement_duplicate_name_and_authority(self):
  good=self.measure()
  bad=copy.deepcopy(good);bad['modules'][0]['module_search_locations']=[str(self.base)]
  with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,self.runtime,fixture_only=True)
  bad=copy.deepcopy(good);bad['modules'].append(copy.deepcopy(bad['modules'][0]))
  with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,self.runtime,fixture_only=True)
  bad=copy.deepcopy(good);bad['namespace_execution_proven']=True
  with self.assertRaises(ValueError):A.replay_bootstrap_modules(bad,self.runtime,fixture_only=True)
 def test_frozen_constants_unchanged(self):
  old=subprocess.run(['git','show','HEAD:experiments/rineya_synth_centralized_reasoning_v1/observer_startup_measurement_v1/contract.py'],capture_output=True,check=True).stdout
  tree=ast.parse(old);constants={}
  for node in tree.body:
   if isinstance(node,ast.Assign):
    for target in node.targets:
     if isinstance(target,ast.Name) and target.id in {'SCOPE','PERMISSION','RESOURCES','OPS','ABI'}:constants[target.id]=ast.literal_eval(node.value)
  for name,value in constants.items():self.assertEqual(value,getattr(C,name))
methods=['test_location_flag_and_missing_path_refused','test_namespace_origin_metadata_refused','test_missing_dll_and_missing_pth_refused','test_finite_roles_refuse_extra_namespace_and_widened_directory','test_auditor_refuses_path_disagreement_duplicate_name_and_authority','test_frozen_constants_unchanged']
suite=unittest.TestSuite(IndependentControls(name) for name in methods);r=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'kind':'INDEPENDENT_CPU_OBSERVER_BOOTSTRAP_V2_FOCUSED_REVIEW','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'site_disabled':True,'native_run':False,'native_PASS':False,'namespace_execution_proven':False,'source_sha256':{name:hashlib.sha256((p/'observer_startup_measurement_v1'/name).read_bytes()).hexdigest() for name in ('contract.py','worker.py','audit.py')}},sort_keys=True))
raise SystemExit(0 if r.wasSuccessful() else 1)
