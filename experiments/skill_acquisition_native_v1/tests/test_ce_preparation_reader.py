"""Pure pre-Q service reader pricing/ownership; no frameworks or native work."""
import copy
import tempfile
from pathlib import Path
import unittest

from experiments.skill_acquisition_native_v1 import ce_native_services as s
from .test_ce_operation_contracts import manifest
from .test_ce_service_bootstrap_verifier import specification, Clock


def prep_spec(root):
    spec=specification(root);spec['operation_contracts']=manifest();spec['operation_profile_rule']=s.INPUT_RESERVATION_RULE
    spec['preparation_reader_policy']=s.PREPARATION_READER_RULE
    spec['explicit_reservation_kinds']=list(s.PREPARATION_EXPLICIT_KINDS)
    spec['caps'].update(hash_attempts=10**7,hash_bytes=1024**3,wall_seconds=10**9)
    for row in manifest()['contracts'].values():
        if 'bootstrap' in row['owners'] and row['kind'] in s.contracts.MAPPED_KINDS:
            profile=spec['operation_profiles'].setdefault(row['kind'],dict(read_bytes_upper=10**6,write_bytes_upper=10**6,
                identity_hash_attempts_upper=10000,identity_hash_bytes_upper=10**6,wall_s_upper=1))
            profile['wall_s_upper']=max(profile['wall_s_upper'],row['wall_s_upper'])
    spec['operation_profiles']['qualification_reducer_overhead']=dict(read_bytes_upper=0,write_bytes_upper=0,
        identity_hash_attempts_upper=1000000,identity_hash_bytes_upper=128*1024**3,wall_s_upper=60,
        reducer_work_bytes_upper=512*1024**3)
    spec['operation_profiles']['qualification_artifact_readback']=dict(read_bytes_upper=3*880932864+64*1024**2,
        write_bytes_upper=0,identity_hash_attempts_upper=8,identity_hash_bytes_upper=3*880932864+64*1024**2,
        wall_s_upper=60,reducer_work_bytes_upper=0)
    return spec


class PreparationReaderControls(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='prep-reader-pure-');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.spec=prep_spec(self.root)
    def test_missing_kind_and_version_marker_deny_before_preQ_work(self):
        bad=copy.deepcopy(self.spec);bad['operation_profiles'].pop('qualification_reducer_overhead')
        with self.assertRaisesRegex(ValueError,'profile missing'):s.validate_preparation_policy(bad)
        for change in (dict(preparation_reader_policy='unknown'),dict(preparation_reader_policy=None),dict(operation_contracts=None)):
            with self.subTest(change=change),self.assertRaises(ValueError):s.validate_preparation_policy(dict(self.spec,**change))
    def test_preQ_dispatcher_bootstrap_owner_actual_small_reader_extents(self):
        ledger=s.BootstrapLedger(self.spec,'a'*64,synthetic_fixture=True,clock=Clock())
        dispatch=s.StableDispatcher(ledger)
        before=copy.deepcopy(ledger.counters)
        fields=dict(io_read_bytes_upper=0,identity_hash_attempts_upper=200001,identity_hash_bytes_upper=31,
                    reducer_work_bytes_upper=64)
        dispatch('qualification_reducer_overhead','started',operation_id='chain',**fields)
        dispatch('qualification_reducer_overhead','returned',operation_id='chain',wall_s=1)
        self.assertIs(dispatch.owners['chain'][1],ledger)
        self.assertEqual(ledger.operations['chain']['profile']['identity_hash_bytes_upper'],31)
        self.assertEqual(ledger.counters['hash_attempts']-before['hash_attempts'],200003)
        dispatch('qualification_artifact_readback','started',operation_id='read',io_read_bytes_upper=18,
            identity_hash_attempts_upper=1,identity_hash_bytes_upper=18,reducer_work_bytes_upper=0)
        dispatch('qualification_artifact_readback','returned',operation_id='read',bytes_read=17,
            identity_hash_attempts_actual=1,identity_hash_bytes_actual=17,wall_s=1)
        self.assertEqual(ledger.operations['read']['profile']['read_bytes_upper'],18)
        row=ledger.events[ledger.operations['read']['started_seq']]['payload']
        self.assertEqual(s.reservation_profile(self.spec,'qualification_artifact_readback',row['fields'],contract_replay=True),row['reservation_profile'])
    def test_positive_coherent_work_and_readback_limits_are_hard(self):
        for fields in (dict(reducer_work_bytes_upper=0),dict(reducer_work_bytes_upper=True),
            dict(reducer_work_bytes_upper=512*1024**3+1),dict(reducer_work_bytes_upper=64,identity_hash_attempts_upper=1,identity_hash_bytes_upper=0)):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                s.reservation_profile(self.spec,'qualification_reducer_overhead',fields)
        for fields in (dict(read_bytes_upper=0),dict(read_bytes_upper=17,identity_hash_attempts_upper=1,identity_hash_bytes_upper=16),
            dict(read_bytes_upper=17,identity_hash_attempts_upper=1,identity_hash_bytes_upper=17,reducer_work_bytes_upper=1)):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                s.reservation_profile(self.spec,'qualification_artifact_readback',fields)
    def test_zero_hash_reducer_and_opaque_hash_fallback_keep_wall_no_refunds(self):
        zero=dict(read_bytes_upper=0,write_bytes_upper=0,identity_hash_attempts_upper=0,identity_hash_bytes_upper=0,reducer_work_bytes_upper=32)
        profile=s.reservation_profile(self.spec,'qualification_reducer_overhead',zero)
        self.assertEqual(profile['identity_hash_bytes_upper'],0);self.assertEqual(profile['wall_s_upper'],60)
        opaque=dict(zero,identity_hash_attempts_upper=1,identity_hash_bytes_upper=31,opaque_verification=True)
        self.assertEqual(s.reservation_profile(self.spec,'qualification_reducer_overhead',opaque)['identity_hash_bytes_upper'],128*1024**3)


if __name__=='__main__':unittest.main()
