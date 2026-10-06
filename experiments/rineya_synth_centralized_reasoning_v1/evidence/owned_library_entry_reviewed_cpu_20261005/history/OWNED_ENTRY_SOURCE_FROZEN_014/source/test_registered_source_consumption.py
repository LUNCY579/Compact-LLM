"""Inert selected-source consumption controls; no registered native authority."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,hashlib,io,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
def source_pipeline():
    p=HERE/'prefix_native_pipeline.py';module=ModuleType('CPU_registered_source_consume');module.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__);return module
class RegisteredSourceConsumption(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_entry_source_',dir=HERE);self.root=Path(self.temp.name).resolve();self.assertEqual(self.root.parent,HERE.resolve());self.addCleanup(self.temp.cleanup);self.P=source_pipeline()
    def selected(self,raw=b'value=7\n'):
        path=self.root/'selected.py';path.write_bytes(raw);return path,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}
    def test_positive_exact_actual_compiled_source_identity(self):
        path,pin=self.selected();module=self.P.source_module(path,pin);self.assertEqual(module.value,7);self.assertEqual(module.__compiled_source_descriptor__,pin);self.assertEqual(module.__file__,str(path))
    def test_bad_descriptor_and_size_fail_before_content_open(self):
        path,pin=self.selected()
        for changed in ({**pin,'bytes':pin['bytes']+1},{**pin,'bytes':True},{**pin,'role':'model_payload'},{**pin,'extra':True}):
            with patch.object(Path,'open',side_effect=AssertionError('read before descriptor admission')):
                with self.assertRaises(ValueError):self.P.source_module(path,changed)
    def test_source_hash_drift_rejects_before_compile(self):
        path,pin=self.selected(b'raise AssertionError("executed drift")\n');path.write_bytes(b'raise AssertionError("altered drift!")\n')
        with patch('builtins.compile',side_effect=AssertionError('must not compile drift')):
            with self.assertRaises(ValueError):self.P.source_module(path,pin)
    def test_changed_read_fingerprint_and_growth_reject_before_compile(self):
        path,pin=self.selected();original=Path.open
        class Changed(io.BytesIO):
            def read(self,size):
                self.requested=size;raw=super().read(size);path.write_bytes(path.read_bytes()+b'#');return raw
        stream=Changed(path.read_bytes())
        def opening(p,*args,**kwargs):return stream if p==path and args==('rb',) else original(p,*args,**kwargs)
        with patch.object(Path,'open',opening),patch('builtins.compile',side_effect=AssertionError('compile before stable read')):
            with self.assertRaises(ValueError):self.P.source_module(path,pin)
    def test_reparse_or_nonphysical_path_refuses_before_open(self):
        path,pin=self.selected();original=Path.lstat
        def metadata(p):
            real=original(p)
            return SimpleNamespace(st_file_attributes=0x400,st_mode=real.st_mode,st_size=real.st_size) if p==path else real
        with patch.object(Path,'lstat',metadata),patch.object(Path,'open',side_effect=AssertionError('alias read')):
            with self.assertRaises(ValueError):self.P.source_module(path,pin)
    def test_public_registered_entry_refuses_before_worker_consumption_or_context_remint(self):
        # Selected reader controls remain above; the old callback entry now
        # refuses before any worker source or new context can be consumed.
        callback=unittest.mock.Mock(side_effect=AssertionError('caller callback executed'))
        with patch.object(self.P.C,'stage_native_gate'), \
                patch.object(self.P.C,'validate_native_manifest',side_effect=AssertionError('public admission read')) as validate, \
                patch.object(self.P,'source_module',side_effect=AssertionError('unowned worker compile')) as compile_source:
            with self.assertRaisesRegex(ValueError,'registered pipeline requires actual owned main entry'):
                self.P.run_registered(object(),object(),object(),framework_boundary=callback,checker=callback)
        validate.assert_not_called();compile_source.assert_not_called();callback.assert_not_called()
if __name__=='__main__':unittest.main()
