"""Actual observer-only preserver rejects duplicate or extra copied populations."""
from pathlib import Path
from unittest.mock import patch
import unittest
import test_audit as F

class PreserveControls(unittest.TestCase):
    def setUp(self):
        self.case=F.ObserverAuditControls();self.case.setUp();self.addCleanup(self.case.doCleanups)
    def test_duplicate_supervisor_partial_leaf_is_rejected(self):
        # Build actual worker output and preserve once, then exercise a fresh
        # target with a changed original SUP and matching lease terminal.
        self.case.build();c=self.case.C
        sup=c.parse((self.case.raw/'SUPERVISOR_RECEIPT.json').read_bytes())
        sup['partial_outputs'].append(dict(sup['partial_outputs'][0]))
        raw=(c.canonical(sup)+'\n').encode('ascii')
        (self.case.raw/'SUPERVISOR_RECEIPT.json').write_bytes(raw)
        (Path(sup['lease_path'])/'terminal.json').write_bytes(raw)
        P=F.source(Path(__file__).with_name('preserve.py'))
        with self.assertRaisesRegex(ValueError,'partial cardinality|duplicate'):
            P.preserve(self.case.raw,self.case.package/'evidence/newcopy',contract=c,context=self.case.context,
                revalidate=lambda:c.context_value(self.case.context),fresh_death=lambda _:False)
    def test_extra_snapshot_before_final_roster_check_is_rejected(self):
        self.case.build();c=self.case.C;P=F.source(Path(__file__).with_name('preserve.py'))
        target=self.case.package/'evidence/newcopy';original=c.file_descriptor;inserted=[False]
        def verify(path,descriptor,**kwargs):
            result=original(path,descriptor,**kwargs)
            if (target/'source_snapshots').exists() and len(list((target/'source_snapshots').iterdir()))==len(self.case.sources) and not inserted[0]:
                (target/'source_snapshots/extra.py').write_bytes(b'# late extra snapshot\n');inserted[0]=True
            return result
        with patch.object(c,'file_descriptor',side_effect=verify):
            with self.assertRaisesRegex(ValueError,'snapshot child roster'):
                P.preserve(self.case.raw,target,contract=c,context=self.case.context,
                    revalidate=lambda:c.context_value(self.case.context),fresh_death=lambda _:False)
        self.assertTrue(inserted[0])

if __name__=='__main__':unittest.main()
