"""Source/JSON-only registration controls; never publishes or launches."""
from copy import deepcopy
from pathlib import Path
import contextlib,hashlib,io,json,sys,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import prepare_registration as prepare
class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope=prepare.scope();cls.template=prepare.pure.read_json(prepare.TEMPLATE)
    def test_exact_actual_auditor_admission_with_all_frozen_maps_and_dependencies(self):
        result=prepare.validate_template(self.template,self.scope)
        self.assertEqual('PASS_CPU_PHASE3_TEMPLATE_METADATA_ONLY',result['status'])
        self.assertFalse(result['native_launch_authorized']);self.assertFalse(result['launch_HEAD_frozen'])
        self.assertEqual({'source':131,'runtime':65,'decoder':17,'data':5,'model':7},result['counts'])
        self.assertEqual(6,len(self.template['auditor_dependency_pins']))
    def test_placeholder_denies_actual_public_native_prerequisites_before_paths(self):
        with self.assertRaisesRegex(ValueError,'exact launch HEAD'):
            prepare.auditor.prerequisites(terminal_signal=prepare.auditor.SIGNAL,inventory_path='UNUSED',inventory_sha='a'*64,
                profile_sha=prepare.PROFILE_SHA,config_sha=prepare.CONFIG_SHA,preservation_sha='b'*64,source_commit=self.template['source_commit'])
    def test_every_exact_section_omission_or_same_count_replacement_rejects(self):
        for name in prepare.SECTIONS:
            for replacement in (False,True):
                bad=deepcopy(self.template);entries=bad['inventories'][name];key=next(iter(entries));pin=entries.pop(key)
                if replacement:entries['CPU_REPLACEMENT_SAME_COUNT']=pin
                bad=prepare.seal(bad)
                with self.assertRaisesRegex(ValueError,'exactly reconstructed'):prepare.validate_template(bad,self.scope,verify_sources=False)
    def test_profile_core64_config_path_and_dependency_tampering_rejects(self):
        for field,value in (('profile_body_sha256','0'*64),('qualified_runtime64_sha256','0'*64),
            ('config_file_sha256','0'*64),('config_path','CPU_SUBSTITUTE_CONFIG'),('source_commit','1'*40)):
            bad=deepcopy(self.template);bad[field]=value;bad=prepare.seal(bad)
            with self.assertRaises(ValueError):prepare.validate_template(bad,self.scope,verify_sources=False)
        bad=deepcopy(self.template);bad['auditor_dependency_pins'].pop(next(iter(bad['auditor_dependency_pins'])));bad=prepare.seal(bad)
        with self.assertRaises(ValueError):prepare.validate_template(bad,self.scope,verify_sources=False)
    def test_unknown_fields_typed_counts_and_unsealed_templates_reject(self):
        for change in (lambda b:b.update(unknown='CPU_EXTRA'),lambda b:b['counts'].update(runtime=True),lambda b:b.update(sha256='0'*64)):
            bad=deepcopy(self.template);change(bad)
            with self.assertRaises(ValueError):prepare.validate_template(bad,self.scope,verify_sources=False)
    def test_source_only_final_body_uses_real_source_admission_but_mocked_git_and_no_write(self):
        with patch.object(prepare,'git_identity',return_value='1'*40) as git,patch.object(prepare,'write_final',side_effect=AssertionError('final publication')) as writer:
            result=prepare.final_body(self.template,self.scope,'1'*40)
            self.assertEqual('1'*40,result['source_commit']);self.assertNotEqual(self.template['sha256'],result['sha256'])
            self.assertEqual(self.template['inventories'],result['inventories']);self.assertEqual(self.template['auditor_dependency_pins'],result['auditor_dependency_pins'])
            self.assertEqual(2,git.call_count);writer.assert_not_called()
    def test_git_identity_rejects_placeholder_before_subprocess(self):
        with patch.object(prepare.subprocess,'run',side_effect=AssertionError('Git subprocess')):
            with self.assertRaisesRegex(ValueError,'explicit full final launch HEAD'):prepare.git_identity(prepare.HEAD_PLACEHOLDER)
    def test_default_cli_only_inspects_and_freeze_without_review_fails_without_writer(self):
        with patch.object(sys,'argv',['prepare_registration.py']),patch.object(prepare,'write_final',side_effect=AssertionError('final publication')) as writer:
            output=io.StringIO()
            with contextlib.redirect_stdout(output):prepare.main()
            result=json.loads(output.getvalue());self.assertFalse(result['native_launch_authorized']);writer.assert_not_called()
        with patch.object(sys,'argv',['prepare_registration.py','--freeze','--reviewed-template-file-sha256','0'*64,
            '--final-head','1'*40,'--output-name','SYNTH_SHORT_PHASE3_CPU_UNPUBLISHED.json']),patch.object(prepare,'write_final',side_effect=AssertionError('final publication')) as writer:
            with self.assertRaisesRegex(ValueError,'externally reviewed|independently reviewed'):prepare.main()
            writer.assert_not_called()
    def test_no_model_decoder_runtime_probes_or_imports(self):
        self.assertFalse(any(n in sys.modules for n in ('torch','transformers','tokenizers')))
        expected=prepare.dependency_pins(verify=False)
        self.assertEqual(expected,self.template['auditor_dependency_pins'])
        self.assertEqual(prepare.CORE64_SHA,self.template['qualified_runtime64_sha256'])
if __name__=='__main__':unittest.main()
