"""Mechanical denial before targeted finalizer SHA work; no native services."""
from contextlib import contextmanager
from types import MappingProxyType,SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_finalizer as f
from experiments.skill_acquisition_native_v1 import ce_qualification_entry as entry


class DeniedReader:
    def __init__(self):self.profiles=[]
    @contextmanager
    def _overhead(self,reason,**profile):
        self.profiles.append((reason,profile));raise RuntimeError('denied before SHA')
        yield


class Tests(unittest.TestCase):
    def test_scope_owner_profile_starts_before_transitive_child_hashes(self):
        reader=DeniedReader();capsule={'schema':f.boot.SCHEMA,'snapshot':{},'header_ref':{},'journal_ref':{}}
        with patch.object(f,'_expected_bootstrap_owners',side_effect=AssertionError('unpaid expected owner SHA')):
            with self.assertRaisesRegex(RuntimeError,'denied before SHA'):
                f.replay_scope_bootstrap({},capsule,{},reader=reader,preflight=lambda *args:None,child=True)
        reason,profile=reader.profiles[0]
        self.assertEqual(profile['hash_attempts'],3);self.assertEqual(profile['hash_bytes'],3*f.raw.MAX_JSON)
    def test_child_linkage_profile_starts_before_context_or_constructor_read(self):
        reader=DeniedReader();owner=SimpleNamespace(reader=reader)
        with patch.object(f,'digest',side_effect=AssertionError('unpaid child SHA')):
            with self.assertRaisesRegex(RuntimeError,'denied before SHA'):
                f._paid_child_constructor_linkage(owner,{}, {'mode':'registered_scope'}, {},{}, {},{})
        self.assertEqual(reader.profiles[0][1]['hash_attempts'],3)
    def test_cached_configuration_profile_starts_before_entry_validation(self):
        reader=DeniedReader();owner=SimpleNamespace(reader=reader)
        with patch.object(entry,'_SOURCE_CONFIGURATION',MappingProxyType({})):
            with patch.object(entry,'_configuration',side_effect=AssertionError('unpaid configuration SHA')):
                with self.assertRaisesRegex(RuntimeError,'denied before SHA'):
                    f._paid_source_configuration(owner)
        self.assertEqual(reader.profiles[0][1]['hash_attempts'],1)
        self.assertEqual(reader.profiles[0][1]['hash_bytes'],8*1024**2)
    def test_cold_configuration_refuses_before_intake_or_reader_work(self):
        reader=DeniedReader();owner=SimpleNamespace(reader=reader)
        with patch.object(entry,'_SOURCE_CONFIGURATION',None):
            with patch.object(entry,'_configuration',side_effect=AssertionError('cold intake')):
                with self.assertRaisesRegex(ValueError,'already be cached'):
                    f._paid_source_configuration(owner)
        self.assertEqual(reader.profiles,[])


if __name__=='__main__':unittest.main()
