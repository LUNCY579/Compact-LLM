"""Legacy public factories refuse before suppliers; owned-route positives elsewhere.

Earlier callback-wrapper tests are preserved with predecessor source. Actual
captured-installer enclosure is exercised by test_owned_library_entry.py.
"""
from contextlib import contextmanager, nullcontext
import unittest
from unittest.mock import patch
from outer_context_fixture import source


class FrameworkLibraryEnclosureControls(unittest.TestCase):
    def setUp(self):
        self.W=source('prefix_input_owned_worker.py');self.C=self.W.C
        self.context=object();self.value={'fixture_only':False}
    def test_native_binding_without_owned_entry_refuses_before_suppliers(self):
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value=self.value), \
                patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('must not prepare suppliers')) as suppliers:
            with self.assertRaisesRegex(ValueError,'native bindings require actual owned main entry'):
                self.W.real_torch_bindings(self.context,{},'unused')
            suppliers.assert_not_called()
    def test_callback_cannot_restore_public_framework_binding(self):
        events=[]
        @contextmanager
        def boundary():
            events.append('callback');yield
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value=self.value), \
                patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('must not prepare suppliers')) as suppliers:
            with self.assertRaisesRegex(ValueError,'caller callbacks refused'):
                self.W.real_torch_bindings(self.context,{},'unused',framework_boundary=boundary)
        self.assertEqual(events,[]);suppliers.assert_not_called()
    def test_selector_refuses_without_searching_documentary_finder_metadata(self):
        with self.assertRaisesRegex(ValueError,'native library selector requires actual owned main entry'):
            self.W._installed_framework_boundary({},nullcontext)
    def test_pipeline_public_entry_refuses_checker_and_boundary_before_readers(self):
        pipeline=source('prefix_native_pipeline.py');callback=unittest.mock.Mock()
        with patch.object(pipeline.C,'stage_native_gate'), \
                patch.object(pipeline.C,'validate_native_manifest',side_effect=AssertionError('must not remint context')) as validate, \
                patch.object(pipeline,'source_module',side_effect=AssertionError('must not recompile worker')) as compile_source:
            with self.assertRaisesRegex(ValueError,'registered pipeline requires actual owned main entry'):
                pipeline.run_registered('unused',{},'unused',checker=callback,framework_boundary=callback)
        validate.assert_not_called();compile_source.assert_not_called();callback.assert_not_called()
    def test_CPU_context_never_reaches_public_native_binding(self):
        with patch.object(self.C,'stage_native_gate'),patch.object(self.C,'context_value',return_value={'fixture_only':True}), \
                patch.object(self.W,'_native_binding_suppliers',side_effect=AssertionError('must not prepare suppliers')) as suppliers:
            with self.assertRaisesRegex(ValueError,'CPU context cannot'):
                self.W.real_torch_bindings(self.context,{},'unused',framework_boundary=nullcontext)
        suppliers.assert_not_called()


if __name__=='__main__':unittest.main()
