"""Shared native framework binding enclosure with inert internal suppliers."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from outer_context_fixture import source

class FrameworkLibraryEnclosureControls(unittest.TestCase):
    def setUp(self):
        self.W=source('prefix_input_owned_worker.py');self.C=self.W.C
        self.context=object();self.value={'fixture_only':False,'config':{'source_files':{str(self.C.PACKAGE/'prefix_registered_loader.py'):{'bytes':1,'sha256':'0'*64,'role':'python_source'}}}}
    def test_native_binding_without_enforced_boundary_refuses_before_suppliers(self):
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value=self.value),patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('must not prepare suppliers')):
            with self.assertRaisesRegex(ValueError,'installed selected context-bound library finder'):
                self.W.real_torch_bindings(self.context,{},'unused')
    def test_actual_shared_supplier_import_is_inside_boundary(self):
        events=[];framework=object()
        @contextmanager
        def boundary():
            events.append('before');yield;events.append('after')
        def supplier():events.append('import');return framework
        def body(ctx,owner,out,value,suppliers,revalidate):
            return {'api':{'observed':suppliers['framework']()}}
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value=self.value),patch.object(self.W,'_installed_framework_boundary',return_value=boundary),patch.object(self.W,'_native_binding_suppliers',return_value={'framework':supplier}),patch.object(self.W,'_tensor_binding_body',side_effect=body):
            result=self.W.real_torch_bindings(self.context,{},'unused',framework_boundary=boundary)
        self.assertIs(result['observed'],framework);self.assertEqual(events,['before','import','after'])
    def test_boundary_refusal_prevents_framework_supplier(self):
        @contextmanager
        def boundary():
            raise ValueError('unqualified observed library scope');yield
        def body(ctx,owner,out,value,suppliers,revalidate):return {'api':suppliers['framework']()}
        supplier=unittest.mock.Mock(side_effect=AssertionError('must not import'))
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value=self.value),patch.object(self.W,'_installed_framework_boundary',return_value=boundary),patch.object(self.W,'_native_binding_suppliers',return_value={'framework':supplier}),patch.object(self.W,'_tensor_binding_body',side_effect=body):
            with self.assertRaisesRegex(ValueError,'unqualified observed library scope'):
                self.W.real_torch_bindings(self.context,{},'unused',framework_boundary=boundary)
        supplier.assert_not_called()
    def test_noop_callback_does_not_replace_actual_installed_library_scope(self):
        from contextlib import nullcontext
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value=self.value),patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('must not prepare suppliers')):
            with self.assertRaisesRegex(ValueError,'installed selected context-bound library finder'):
                self.W.real_torch_bindings(self.context,{},'unused',framework_boundary=nullcontext)
    def test_CPU_context_never_reaches_native_library_enclosure(self):
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value={'fixture_only':True}),patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('must not prepare suppliers')):
            with self.assertRaisesRegex(ValueError,'CPU context cannot'):
                self.W.real_torch_bindings(self.context,{},'unused',framework_boundary=lambda:None)

if __name__=='__main__':unittest.main()
