"""Stdlib fake checkpoint controls, never a framework/model/native execution."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace,MethodType
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import checkpoint_successor_profile as profile


class Parameter:
    shape=(2,2);dtype="bfloat16";device="cpu";requires_grad=False
    def data_ptr(self):return id(self)
    def stride(self):return (2,1)
    def untyped_storage(self):return self
    def nbytes(self):return 8


class Layer:
    training=False;gradient_checkpointing=False;dropout=0.
    def __init__(self,index):self.index=index;self.calls=0;self.fail_assign=False;self.body_failure=False
    def __setattr__(self,name,value):
        object.__setattr__(self,name,value)
        if name=="forward" and getattr(self,"fail_assign",False):
            object.__setattr__(self,"fail_assign",False);raise RuntimeError("partial assignment")
    def forward(self,value,**kwargs):
        self.calls+=1
        if self.body_failure:raise RuntimeError("decoder failure")
        return value+self.index+1


class Model:
    training=False
    def __init__(self):
        self.model=SimpleNamespace(layers=[Layer(index) for index in range(28)])
        self.config=SimpleNamespace(attention_dropout=0.,use_cache=True)
        self.parameters=[Parameter() for _ in range(28)]
    def named_modules(self):return [("",self)]+[(str(index),layer) for index,layer in enumerate(self.model.layers)]
    def named_parameters(self,remove_duplicate=False):return [(str(index),value) for index,value in enumerate(self.parameters)]


class Checkpoint:
    def __init__(self):self.early=None;self.tape=[];self.fail=None;self.context_fail=False
    @contextmanager
    def early_context(self,value):
        if self.context_fail:raise RuntimeError("early context failure")
        self.early=value
        try:yield
        finally:self.early=None
    def __call__(self,function,*args,**kwargs):
        if self.fail=="before":raise RuntimeError("checkpoint before callback")
        assert self.early is False and kwargs.pop("use_reentrant") is False and kwargs.pop("preserve_rng_state") is True
        forward,recompute=kwargs.pop("context_fn")()
        with forward:value=function(*args,**kwargs)
        self.tape.append((function,args,kwargs,recompute))
        if self.fail=="after":raise RuntimeError("checkpoint after callback")
        return value
    def backward(self):
        for function,args,kwargs,recompute in reversed(self.tape):
            with recompute:function(*args,**kwargs)


class Controls(unittest.TestCase):
    def setUp(self):
        self.model=Model();self.checkpoint=Checkpoint();self.grad=True;self.ticks=0.;self.events=[]
        self.torch=SimpleNamespace(is_grad_enabled=lambda:self.grad)
    def clock(self):self.ticks+=.001;return self.ticks
    def journal(self,name,function):
        self.events.append((name,"attempt"));value=function();self.events.append((name,"returned"));return value
    def install(self,**kwargs):
        return profile.install(self.model,self.torch,self.checkpoint,early_stop_context=self.checkpoint.early_context,call=self.journal,clock=self.clock,**kwargs)
    def saved(self):return [(layer,"forward" in layer.__dict__,layer.__dict__.get("forward"),layer.forward.__func__) for layer in self.model.model.layers]
    def assert_restored(self,saved):
        for layer,had,value,function in saved:
            self.assertEqual(had,"forward" in layer.__dict__)
            self.assertIs(value,layer.__dict__.get("forward"));self.assertIs(function,layer.forward.__func__)
    def test_all28_full_forward_recompute_normal_returns_paired_costs_durations(self):
        saved=self.saved();installed=self.install();value=0
        for layer in self.model.model.layers:value=layer.forward(value,use_cache=False)
        self.assertEqual(406,value);self.checkpoint.backward()
        snapshot=installed["snapshot"]()
        self.assertEqual({"forward_regions":28,"recompute_regions":28,"no_grad_regions":0},snapshot["counters"])
        self.assertEqual(56,len(snapshot["regions"]));self.assertFalse(snapshot["failed_or_pending"])
        for phases in snapshot["layers"].values():
            for phase in ("forward","recompute"):
                self.assertEqual((1,1,0,0),tuple(phases[phase][key] for key in ("attempted","returned","failed","pending")))
                self.assertGreater(phases[phase]["elapsed_seconds"],0)
        pairs={}
        for entry in snapshot["regions"]:pairs.setdefault(entry["region_id"],set()).add(entry["phase"])
        self.assertTrue(all(phases=={"forward","recompute"} for phases in pairs.values()))
        installed["verify"]();installed["remove"]();self.assert_restored(saved)
    def test_no_grad_bypass_original_no_checkpoint_and_instance_attribute_restored(self):
        layer=self.model.model.layers[3];layer.forward=MethodType(lambda module,value,**kwargs:value+100,layer)
        saved=self.saved();installed=self.install();self.grad=False;value=0
        for layer in self.model.model.layers:value=layer.forward(value,use_cache=False)
        self.assertEqual(502,value);self.assertFalse(self.checkpoint.tape)
        self.assertEqual(28,installed["snapshot"]()["counters"]["no_grad_regions"])
        installed["remove"]();self.assert_restored(saved)
    def test_partial_install_and_install_journal_guard_failures_restore_exactly(self):
        saved=self.saved();self.model.model.layers[13].fail_assign=True
        with self.assertRaisesRegex(RuntimeError,"partial assignment"):self.install()
        self.assert_restored(saved)
        def after(name,function):
            value=function()
            if name=="checkpoint_install_attributes":raise RuntimeError("after assign journal")
            return value
        with self.assertRaisesRegex(RuntimeError,"after assign journal"):
            profile.install(self.model,self.torch,self.checkpoint,early_stop_context=self.checkpoint.early_context,call=after)
        self.assert_restored(saved)
        def guard(stage):
            if stage=="checkpoint_install_after":raise RuntimeError("install guard")
        with self.assertRaisesRegex(RuntimeError,"install guard"):self.install(guard=guard)
        self.assert_restored(saved)
    def test_checkpoint_before_after_and_early_context_failures_poison_stickily(self):
        for failure in ("before","after","context"):
            with self.subTest(failure=failure):
                self.model=Model();self.checkpoint=Checkpoint();saved=self.saved();installed=self.install()
                if failure=="context":self.checkpoint.context_fail=True
                else:self.checkpoint.fail=failure
                with self.assertRaises(RuntimeError):self.model.model.layers[0].forward(1,use_cache=False)
                self.assertTrue(installed["snapshot"]()["failed_or_pending"])
                self.checkpoint.fail=None;self.checkpoint.context_fail=False
                with self.assertRaisesRegex(ValueError,"unpoisoned"):self.model.model.layers[0].forward(1,use_cache=False)
                installed["remove"]();self.assert_restored(saved)
    def test_after_actual_return_failure_keeps_return_and_remove_survives_failed_guard(self):
        saved=self.saved()
        def call(name,function):
            value=function()
            if name.endswith("_forward"):raise RuntimeError("post returned journal")
            return value
        def guard(stage):
            if stage=="checkpoint_remove_before":raise RuntimeError("remove guard")
        installed=profile.install(self.model,self.torch,self.checkpoint,early_stop_context=self.checkpoint.early_context,call=call,guard=guard)
        with self.assertRaisesRegex(RuntimeError,"post returned"):self.model.model.layers[0].forward(1,use_cache=False)
        self.assertEqual((1,1,0,0),tuple(installed["snapshot"]()["layers"]["0"]["forward"][key] for key in ("attempted","returned","failed","pending")))
        with self.assertRaisesRegex(RuntimeError,"remove guard"):installed["remove"]()
        self.assert_restored(saved)
    def test_cache_dropout_mode_parameter_and_duplicate_install_rejected(self):
        installed=self.install()
        for kwargs in ({"use_cache":True},{"past_key_value":object()},{"past_key_values":object()},{"cache":object()}):
            with self.assertRaisesRegex(ValueError,"cache"):self.model.model.layers[0].forward(1,**kwargs)
        with self.assertRaisesRegex(ValueError,"uncheckpointed"):self.install()
        self.model.parameters[0].dtype="float32"
        with self.assertRaisesRegex(ValueError,"dtype"):installed["verify"]()
        self.model.parameters[0].dtype="bfloat16";installed["remove"]()
        self.model.model.layers[2].dropout=.1
        with self.assertRaisesRegex(ValueError,"dropout"):self.install()
        self.model.model.layers[2].dropout=0.;self.model.model.layers[2].training=True
        with self.assertRaisesRegex(ValueError,"eval"):self.install()
    def test_equal_aliased_parameter_replacement_is_not_same_parameter_object(self):
        installed=self.install();original=self.model.parameters[0]
        class Alias(Parameter):
            def data_ptr(self):return original.data_ptr()
            def untyped_storage(self):return original.untyped_storage()
            def __eq__(self,other):raise AssertionError("tensor equality must not be consulted")
        self.model.parameters[0]=Alias()
        with self.assertRaisesRegex(ValueError,"parameter object"):installed["verify"]()
        self.model.parameters[0]=original;installed["remove"]()


if __name__=="__main__":unittest.main()
