"""Check actual closed successor packets; no native gate or context call."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
import argparse, hashlib, json, unittest

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]

def module(path, name):
    value = ModuleType(name); value.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), value.__dict__)
    return value

R = module(HERE / 'refresh_final.py', 'CPU_final_recipe_builder')
Q = module(PACKAGE / 'import_runtime_qualification.py', 'CPU_final_recipe_qualifier')
D = module(PACKAGE / 'import_dispatch_postrun.py', 'CPU_final_postrun_source')

def validate(recipe, config, runtime, registration, selection):
    for value in (recipe, config, runtime, registration, selection):
        Q.require(value['sha256'] == R.digest({k: v for k, v in value.items() if k != 'sha256'}), 'exact closed successor packet seal')
    implementation = selection['implementation_commit']
    Q.require(type(implementation) is str and len(implementation) == 40 and all(c in '0123456789abcdef' for c in implementation), 'actual full implementation')
    Q.require(config['source_commit'] == runtime['source_commit'] == registration['source_commit'] == recipe['implementation_commit'] == implementation, 'exact implementation joins')
    Q.require(config['source_files'] == registration['source_files'] == selection['source_files'] and R.digest(config['source_files']) == selection['source_inventory_sha256'], 'selected source map joins')
    Q.require(config['runtime_files'] == registration['runtime_files'] == runtime['files'] == selection['runtime_files'] and R.digest(runtime['files']) == selection['runtime_inventory_sha256'], 'selected runtime map joins')
    Q.require(runtime['cache_policy']['separate_main_source'] == {'path': config['main_source'], 'descriptor': config['source_files'][config['main_source']]}, 'actual current main descriptor')
    Q.require(config['native_launch_authorized'] is registration['native_launch_authorized'] is recipe['native_launch_authorized'] is config['native_authority_issued'] is False and config['owner_approval'] is registration['owner_approval'] is recipe['owner_grant_reference'] is recipe['owner_receipt_reference'] is None, 'closed no permission or authority')
    Q.require(recipe['external_dispatch_head'] is recipe['complete_current_byte_review'] is registration['qualification_review_path'] is registration['qualification_review_file_sha256'] is registration['qualification_review_body_sha256'] is None, 'remaining witnesses explicit pending')
    Q.require(config['resources'] == recipe['resources'] == Q.B.RESOURCES and config['permission'] == recipe['permission'] == Q.PERMISSION and registration['profile_sha256'] == R.digest({'scope': Q.SCOPE, 'permission': Q.PERMISSION, 'resources': Q.B.RESOURCES}), 'unchanged purpose profile and resources')
    Q.require(set(config) == Q.CONFIG_FIELDS and set(runtime) == Q.RUNTIME_FIELDS and set(registration) == Q.REGISTRATION_FIELDS, 'exact approved schema fields with closed values')
    Q.require(runtime['metadata_references']['additional_preload'] == recipe['passive_extended_reference'], 'selected extended passive receipt')
    passive_ref = recipe['passive_extended_reference']
    passive = Q.B.sealed_json(passive_ref['path'], passive_ref['file_sha256'], passive_ref['body_sha256'])
    Q.require(Q.passive_preload_overlay(passive) == runtime['cache_policy']['additional_preload_sources'], 'exact V2 extended overlay')
    Q.require(len(runtime['files']) == 8725 and len(runtime['cache_policy']['preloaded_module_cache_pins']) == 59 and len(runtime['cache_policy']['additional_preload_sources']) == 11 and sum(r['cache_path'] is not None for r in runtime['cache_policy']['additional_preload_sources'].values()) == 8 and len(runtime['measurement_policy']['initial_loaded_libraries']) == 39, 'finite exact registered populations')
    Q.import_git_exception_policy(registration['git_output_exceptions'], config['source_files'], runtime['files'], runtime['library_event_observer'], Q.B)
    Q._source_roster({'config': config, 'runtime': runtime}, Q.B, Q.QualificationBudget())
    Q.require(recipe['exact_preservation_exception_count'] == len(registration['git_output_exceptions']) == len(config['source_files']) + 18, 'all source snapshots and18 exact future raw leaves')
    command = recipe['future_command']; anchors = command['nine_anchors']
    Q.require(set(anchors) == set(D.ANCHORS) == set(R.ANCHORS) and anchors['expected_head'] is None, 'exact9 anchor command pending dispatch HEAD')
    Q.require(command['argv_prefix'][3] == str(PACKAGE / 'import_dispatch_postrun.py') and command['separate_owner_references_required'] is True and len(command['required_owner_reference_arguments']) == 6, 'actual postrun CLI and6 outofband owner anchor arguments')
    return True

class Controls(unittest.TestCase):
    suffix = '005'
    def setUp(self):
        self.recipe = R.read_sealed(HERE / ('EXACT_IMPORT_RUN_RECIPE' + self.suffix + '.json'))
        self.config = R.read_sealed(HERE / ('IMPORT_OBSERVATION_CONFIG_' + self.suffix + '.json'))
        self.runtime = R.read_sealed(HERE / ('IMPORT_OBSERVATION_RUNTIME_' + self.suffix + '.json'))
        self.registration = R.read_sealed(HERE / ('IMPORT_OBSERVATION_REGISTRATION_' + self.suffix + '.json'))
        self.selection = R.read_sealed(HERE / ('ACTUAL_SOURCE_RUNTIME_SELECTION' + self.suffix + '.json'))
    def check(self):
        return validate(self.recipe, self.config, self.runtime, self.registration, self.selection)
    def reseal(self, *names):
        for name in names:
            value = getattr(self, name); setattr(self, name, R.seal({k: v for k, v in value.items() if k != 'sha256'}))
    def test_complete_actual_closed_packet_and_command(self):
        self.assertTrue(self.check())
    def test_stale_main_descriptor_refused(self):
        self.runtime['cache_policy']['separate_main_source']['descriptor']['sha256'] = '0' * 64; self.reseal('runtime')
        with self.assertRaisesRegex(ValueError, 'current main'): self.check()
    def test_resource_lowering_refused(self):
        self.config['resources']['GPU_free_at_admission_GiB'] = 1; self.recipe['resources'] = deepcopy(self.config['resources']); self.reseal('config', 'recipe')
        with self.assertRaisesRegex(ValueError, 'resources'): self.check()
    def test_fabricated_grant_or_dispatch_refused(self):
        self.recipe['owner_grant_reference'] = {'path': 'notapproved'}; self.reseal('recipe')
        with self.assertRaisesRegex(ValueError, 'no permission'): self.check()
        with self.assertRaises(ValueError): D.dispatch('unread', preserved_directory='uncreated', anchors={})
    def test_source_rebinding_refused(self):
        self.config['source_files'][str(PACKAGE / 'import_dispatch_postrun.py')]['sha256'] = '0' * 64; self.registration['source_files'] = self.config['source_files']; self.reseal('config', 'registration')
        with self.assertRaisesRegex(ValueError, 'source map'): self.check()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--suffix', default='005'); parser.add_argument('--receipt', type=Path)
    args = parser.parse_args(); Controls.suffix = args.suffix
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    receipt = R.seal({'kind': 'CPU_ACTUAL_IMPLEMENTATION_CLOSED_IMPORT_RECIPE_CONTROLS', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'native_context_or_gate_called': False, 'full_byte_review_or_native_readiness_claimed': False, 'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), HERE / 'refresh_final.py', PACKAGE / 'import_runtime_qualification.py')}})
    if args.receipt: R.write(args.receipt, receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)
