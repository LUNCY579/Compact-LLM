"""Mechanical exact-offset and complete suffix controls; no native evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_finalizer_receipts as r
from experiments.skill_acquisition_native_v1 import ce_finalizer_accounting as f
from experiments.skill_acquisition_native_v1.tests.test_ce_finalizer_accounting import Clock, spec


class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='ce8-r4-exact-offset-mechanical-')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);self.clock=Clock()
        self.spec=spec(self.root);self.owner=f.fixture_owner(self.spec,self.root,clock=self.clock)
        self.rows=[{'seq':0,'offset':0,'bytes':5,'sha256':'1'*64},
                   {'seq':1,'offset':5,'bytes':7,'sha256':'2'*64}]
        self.journal={'path':str(self.root/'ORIGINAL.jsonl'),'sha256':'3'*64,'bytes':12}
    def test_maximum_encoded_row_and_table_envelope_are_exact(self):
        row={'seq':400007,'offset':268435455,'bytes':4194304,'sha256':'f'*64}
        self.assertEqual(len(r.canonical(row)+b'\n'),126)
        self.assertEqual(400008*126,50401008)
        self.assertLess(400008*126,r.OFFSET_BYTES)
        self.assertLess(r.FINAL_RECEIPT_BYTES+r.FINAL_SUFFIX_JOURNAL_BYTES+r.FINAL_SEAL_BYTES,4*1024**2)
        self.assertLess(4*1024**2+r.FINALIZER_JOURNAL_BYTES+r.FINAL_SEAL_BYTES+r.FINAL_RECEIPT_BYTES+4,128*1024**2)
    def test_plan_preserves_all_rows_exact_digest_and_raw_journal(self):
        plan=r.offset_plan(self.rows,self.journal);blob=b''.join(r.offset_blocks(self.rows))
        self.assertEqual(plan['records'],2);self.assertEqual(plan['bytes'],len(blob))
        self.assertEqual(plan['sha256'],hashlib.sha256(blob).hexdigest())
        self.assertEqual(plan['original_journal_ref'],self.journal)
    def test_wrong_order_coverage_hash_or_ref_refuses_before_paths(self):
        cases=[]
        rows=copy.deepcopy(self.rows);rows[1]['offset']=6;cases.append((rows,self.journal))
        rows=copy.deepcopy(self.rows);rows[1]['seq']=2;cases.append((rows,self.journal))
        rows=copy.deepcopy(self.rows);rows[1]['sha256']='x'*64;cases.append((rows,self.journal))
        cases.append((self.rows,dict(self.journal,bytes=13)))
        cases.append((self.rows,dict(self.journal,bytes=True)))
        with patch.object(Path,'open',side_effect=AssertionError('unpaid paths')):
            for rows,ref in cases:
                with self.assertRaises(ValueError):r.offset_plan(rows,ref)
    def test_paid_exact_companion_is_create_only_cached_and_readback_verified(self):
        before=dict(self.owner.counts)
        first=self.owner.externalize_offsets(self.rows,self.journal)
        ref=first['artifact_ref'];blob=Path(ref['path']).read_bytes()
        self.assertEqual(blob,b''.join(r.offset_blocks(self.rows)))
        self.assertEqual(hashlib.sha256(blob).hexdigest(),ref['sha256'])
        second=self.owner.externalize_offsets(copy.deepcopy(self.rows),copy.deepcopy(self.journal))
        self.assertEqual(first,second);self.assertEqual(len(self.owner._offset_tables),1)
        self.assertEqual(len(list(self.owner.directory.glob('ORIGINAL_OFFSETS_*.jsonl'))),1)
        self.assertGreaterEqual(self.owner.counts['io_write_bytes_upper']-before['io_write_bytes_upper'],r.OFFSET_BYTES)
        self.assertGreaterEqual(self.owner.counts['metadata_bytes_upper']-before['metadata_bytes_upper'],r.OFFSET_BYTES)
        self.assertFalse(self.owner.native)
    def test_same_journal_changed_offset_hash_cannot_reuse_cached_summary(self):
        self.owner.externalize_offsets(self.rows,self.journal)
        changed=copy.deepcopy(self.rows);changed[0]['sha256']='4'*64
        with self.assertRaisesRegex(ValueError,'table changed'):
            self.owner.externalize_offsets(changed,self.journal)
    def test_compaction_changes_only_offset_serialization_and_retains_proof(self):
        payload={'derived_totals':{'cost':19},'original_offsets':self.rows,
                 'original_journal_ref':self.journal,'native_qualified':False}
        original=copy.deepcopy(payload)
        result=self.owner.compact_replay_receipt({'one':payload,'duplicate':copy.deepcopy(payload)})
        self.assertEqual(payload,original)
        self.assertEqual(result['one']['derived_totals'],{'cost':19})
        self.assertEqual(result['one']['original_journal_ref'],self.journal)
        self.assertEqual(result['one']['original_offsets_ref'],result['duplicate']['original_offsets_ref'])
        self.assertNotIn('original_offsets',result['one'])
        self.assertFalse(result['one']['native_qualified'])
    def test_companion_after_suffix_refuses_before_its_path(self):
        self.owner.begin_finalization()
        with patch.object(Path,'open',side_effect=AssertionError('new companion after suffix')):
            with self.assertRaisesRegex(ValueError,'precede final suffix'):
                self.owner.externalize_offsets(self.rows,self.journal)
    def test_final_receipt_maximum_mib_closes_under_original_suffix_caps(self):
        overhead=len(r.canonical({'padding':''})+b'\n')
        value={'padding':'a'*(r.FINAL_RECEIPT_BYTES-overhead)}
        self.assertEqual(len(r.canonical(value)+b'\n'),r.FINAL_RECEIPT_BYTES)
        reference=self.owner.write_json('QUALIFICATION_RECEIPT.json',value)
        evidence=self.owner.close_and_seal()
        self.assertEqual(reference['bytes'],r.FINAL_RECEIPT_BYTES)
        self.assertTrue(evidence['owner_closed'])
        self.assertLess(evidence['seal_ref']['bytes'],r.FINAL_SEAL_BYTES)
        for key,upper in self.spec['finalization_bounds'].items():
            self.assertLessEqual(self.owner.counts[key]-self.owner._suffix_before[key],upper)
    def test_old_four_mib_receipt_is_refused_before_receipt_write(self):
        with self.assertRaisesRegex(ValueError,'compact final qualification receipt'):
            self.owner.write_json('QUALIFICATION_RECEIPT.json',{'padding':'a'*r.FINAL_RECEIPT_BYTES})
        self.assertFalse((self.owner.directory/'QUALIFICATION_RECEIPT.json').exists())
    def test_suffix_spec_cannot_omit_journal_or_seal_costs(self):
        wrong=copy.deepcopy(self.spec);wrong['finalization_bounds']['io_read_bytes_upper']=8*1024**2+2
        with self.assertRaisesRegex(ValueError,'complete compact'):
            f.validate_specification(wrong,self.root)
    def test_finalizer_producer_prechecks_record_and_suffix_limits(self):
        self.owner._known_size=r.FINALIZER_JOURNAL_BYTES
        with patch.object(Path,'open',side_effect=AssertionError('journal overrun append')):
            with self.assertRaisesRegex(ValueError,'producer'):
                self.owner._event('test',{})
    def test_nonoffset_copy_graph_is_bounded_and_permanently_paid(self):
        before=self.owner.counts['reducer_work_bytes_upper']
        self.owner.compact_replay_receipt({'small':19})
        self.assertGreaterEqual(self.owner.counts['reducer_work_bytes_upper']-before,512*1024**2)
        large={'chunks':['a'*1024**2]*5}
        with self.assertRaisesRegex(ValueError,'compact receipt graph'):
            self.owner.compact_replay_receipt(large)
    def test_offset_table_cannot_drop_or_add_a_row_without_exact_journal_coverage(self):
        with self.assertRaisesRegex(ValueError,'cover the exact'):
            r.offset_plan(self.rows[:1],self.journal)
        with self.assertRaisesRegex(ValueError,'offset order'):
            r.offset_plan(self.rows+[dict(self.rows[-1])],self.journal)
    def test_fixed_final_receipt_shape_has_a_proven_encoded_upper(self):
        self.assertEqual(r.final_receipt_shape({'leaf':'bounded'})['canonical_bytes_conservative_upper'],330752)
        with self.assertRaisesRegex(ValueError,'scalar extent'):
            r.final_receipt_shape({'oversized':'x'*1024})
        with self.assertRaisesRegex(ValueError,'scalar count'):
            r.final_receipt_shape({'too_many':[1]*257})
    def test_receipt_two_sha_preimages_are_paid_before_they_are_hashed(self):
        header=f.canonical(self.owner.header);original=hashlib.sha256;seen=[]
        def hashed(blob=b'',*args,**kwargs):
            is_body=b'"schema":"FIXTURE_RECEIPT"' in blob and b'"cost_owner_binding_sha256"' in blob
            if blob==header or is_body:
                pending=[value for value in self.owner.operations.values() if value['phase']=='started'
                    and value['kind']=='finalizer_metadata' and value['reservation']['identity_hash_attempts_upper']==2
                    and value['reservation']['identity_hash_bytes_upper']==f.MAX_JSON+r.FINAL_RECEIPT_BYTES]
                self.assertEqual(len(pending),1);seen.append(blob)
            return original(blob,*args,**kwargs)
        with patch.object(f.hashlib,'sha256',hashed):
            receipt=self.owner.make_final_receipt({'schema':'FIXTURE_RECEIPT','passed':False})
        self.assertEqual(len(seen),2)
        self.assertEqual(receipt['cost_owner_binding_sha256'],original(header).hexdigest())
        self.assertEqual(receipt['receipt_sha256'],f.digest({k:v for k,v in receipt.items() if k!='receipt_sha256'}))
    def test_receipt_digest_failure_keeps_both_reserved_hashes_and_failed_record(self):
        header=f.canonical(self.owner.header);original=hashlib.sha256;before=dict(self.owner.counts)
        def fail(blob=b'',*args,**kwargs):
            if blob==header:raise RuntimeError('paid receipt SHA failure')
            return original(blob,*args,**kwargs)
        with patch.object(f.hashlib,'sha256',fail):
            with self.assertRaisesRegex(RuntimeError,'receipt SHA failure'):
                self.owner.make_final_receipt({'schema':'FIXTURE_RECEIPT','passed':False})
        failed=[value for value in self.owner.operations.values() if value['phase']=='failed']
        self.assertEqual(len(failed),1);self.assertEqual(failed[0]['reservation']['identity_hash_attempts_upper'],2)
        self.assertGreaterEqual(self.owner.counts['identity_hash_attempts_upper']-before['identity_hash_attempts_upper'],2)
    def test_control_allowance_precedes_any_owner_canonical_or_copy(self):
        wrong=copy.deepcopy(self.spec);wrong['self_bounds']['reducer_work_bytes_upper']=0
        with patch.object(f,'canonical',side_effect=AssertionError('unpaid serialization')):
            with self.assertRaisesRegex(ValueError,'permanent finalizer allowance exhausted'):
                f.fixture_owner(wrong,self.root/'unpaid')
    def test_event_oversize_is_refused_before_payload_deepcopy(self):
        with patch.object(f.copy,'deepcopy',side_effect=AssertionError('oversize deepcopy')):
            with self.assertRaisesRegex(ValueError,'before encoding'):
                self.owner._event('oversize',{'value':'x'*r.FINALIZER_LINE_BYTES})


if __name__=='__main__':unittest.main()
