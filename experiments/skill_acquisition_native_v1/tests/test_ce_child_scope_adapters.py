"""Closed child source verifiers and retained singleton hooks; no native work."""
import copy
import sys
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_child_scope_adapters as a


class Trap:
    def __getattribute__(self, key): raise AssertionError('unadmitted source access')


class Tests(unittest.TestCase):
    def test_every_native_child_verifier_refuses_unregistered_or_foreign_issuer_before_io(self):
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        actions = [lambda: a.child_provider_service_verifier(Trap(), 'a' * 64, 'b' * 64),
            lambda: a.child_worker_metadata_verifier(Trap(), 'a' * 64, 'b' * 64, 'c' * 64),
            lambda: a.child_native_authority_verifier(Trap(), 'a' * 64, 'b' * 64, 'c' * 64, 'qualification', 'canonical'),
            lambda: a.child_bootstrap_native_issuer(Trap(), None, Trap()),
            lambda: a.child_bootstrap_prepayment_verifier(Trap(), Trap(), Trap(), Trap()),
            lambda: a.child_predecessor_verifier(Trap(), Trap(), Trap()),
            lambda: a.child_publication_issuer(Trap(), None, Trap()),
            lambda: a.child_launch_verifier(Trap(), 'a' * 64, 'b' * 64, 'c' * 64),
            lambda: a.child_launch_observation(Trap(), None, Trap(), Trap(), Trap()),
            lambda: a.cached_worker(Trap(), Trap(), Trap())]
        with patch.object(a.worker, 'verify_worker_handle', side_effect=AssertionError('unpriced OS')), patch.object(a.worker, '_WindowsOS', side_effect=AssertionError('unpriced OS')):
            for action in actions:
                with self.assertRaises(ValueError): action()
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})
        self.assertFalse(a.child._REGISTERED_LOCAL_ISSUERS); self.assertFalse(a.authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)
        self.assertIsNone(a.local._SOURCE_ENTRY_PROVIDER)
    def test_local_and_provider_begin_with_no_active_entry_or_observation_claim(self):
        issuer = a.local.LocalCheckpointIssuer(); provider = a.provider.CheckpointEntryProvider()
        self.assertIsNone(issuer._active_entry); self.assertIsNone(issuer._active_context); self.assertIsNone(issuer._active_proof)
        self.assertIsNone(issuer._active_request); self.assertIsNone(provider._active_configuration)
        self.assertEqual(provider._metadata_specs, {})
        with self.assertRaisesRegex(ValueError, 'unregistered'):
            a.child_native_authority_verifier(issuer, 'a' * 64, 'b' * 64, 'c' * 64, 'qualification', 'canonical')
    def test_json_flags_cannot_replace_local_source_singleton_or_worker_handle(self):
        context = {'native_services': {'local_checkpoint_issuer': {'approved': True, 'native_qualified': True}}}
        with self.assertRaisesRegex(ValueError, 'singleton'): a.child_predecessor_verifier(context, {}, {})
        with self.assertRaisesRegex(ValueError, 'sealed native'):
            a.cached_worker({'native': True, 'closed': False, 'worker_identity_sha256': 'a' * 64}, {}, object())
    def test_context_identity_hash_keeps_dynamic_scope_but_excludes_existing_live_service_objects(self):
        context = {'request_sha256': 'a' * 64, 'checkpoint_child_observed_lineage': {'child_request_sha256': 'b' * 64},
            'native_services': {'local_checkpoint_issuer': a.local.LocalCheckpointIssuer()}}
        hashed = a.digest(a.authority.context_preimage(context))
        context['native_services']['local_checkpoint_issuer'] = object()
        self.assertEqual(a.digest(a.authority.context_preimage(context)), hashed)
        context['checkpoint_child_observed_lineage']['child_request_sha256'] = 'c' * 64
        self.assertNotEqual(a.digest(a.authority.context_preimage(context)), hashed)
    def test_cached_authority_recheck_has_no_os_refresh_even_for_exact_unregistered_singleton(self):
        issuer = a.local.LocalCheckpointIssuer()
        with patch.object(a.worker, 'verify_worker_handle', side_effect=AssertionError('recursive priced refresh')):
            with self.assertRaises(ValueError):
                a.child_native_authority_verifier(issuer, 'a' * 64, 'b' * 64, 'c' * 64, 'qualification', 'canonical')
    def test_provider_has_no_config_or_authority_after_plain_construction(self):
        provider = a.provider.CheckpointEntryProvider()
        with patch.object(a.worker, 'open_worker', side_effect=AssertionError('OS worker')):
            with self.assertRaisesRegex(ValueError, 'not selected'): provider.open_prepaid_entry('unapproved.json')
        self.assertIsNone(provider._active_configuration); self.assertIsNone(provider._active_runtime)
        self.assertIsNone(provider._active_payment); self.assertEqual(provider._metadata_specs, {})

    def test_observed_child_scope_derives_actual_pid_creation_and_whole_raw_source_binding(self):
        from . import test_ce_checkpoint_processes as fixtures
        helper = fixtures.Tests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        spec = fixtures.p.child_launch_spec(helper.context, 'prefix', helper.marker, helper.proof)
        spec['observation_bounds'] = copy.deepcopy(spec['bounds'])
        parent = spec['parent_identity']
        parent['boot_identity'] = 'windows-boot-guid-fixture'
        identity = {'pid': parent['pid'] + 1, 'creation_filetime_100ns': parent['creation_filetime_100ns'] + 1,
            'boot_identity': parent['boot_identity'], 'image_path': spec['process_image_ref']['path'],
            'argv': [spec['process_image_ref']['path'], *spec['argv'][1:]]}
        observed = a.observed_child_spec(spec, identity, parent)
        self.assertEqual(observed['expected_pid'], identity['pid'])
        self.assertEqual(observed['expected_creation_filetime_100ns'], identity['creation_filetime_100ns'])
        self.assertEqual(observed['request_ref'], helper.marker)
        self.assertEqual(observed['source_files'], spec['source_files'])
        self.assertNotIn('native_execution_permit', observed)
        for change in (lambda values: values.update(pid=parent['pid']),
                       lambda values: values.update(creation_filetime_100ns=1),
                       lambda values: values.update(boot_identity='windows-boot-guid-foreign'),
                       lambda values: values.update(argv=['foreign'])):
            bad = copy.deepcopy(identity); change(bad)
            with self.assertRaises(ValueError): a.observed_child_spec(spec, bad, parent)


if __name__ == '__main__': unittest.main()
