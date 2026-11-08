"""Stdlib injected-call controls. These traces are not native E0 measurements."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import (
    ACCOUNTING_VERSION, CALL_OPERATIONS, CallAccounting, record_resource_profile, verify_call_journal,
)


class Sink:
    def __init__(self):
        self.events = []

    def __call__(self, kind, **values):
        self.events.append(deepcopy({"event":kind, **values}))


class AccountingTests(unittest.TestCase):
    def test_empty_accounting_before_any_operation(self):
        sink=Sink();a=CallAccounting(sink)
        # A preload failure happens before any of the eight accounted operations.
        sink("failure",error="synthetic preload guard")
        state=a.snapshot()
        self.assertEqual(verify_call_journal(sink.events,state),state)
        self.assertEqual(state["accounting_version"],2)
        self.assertEqual(state["last_call_id"],0)
        self.assertEqual(state["pending_calls"],[])
        self.assertTrue(all(v==0 for item in state["operations"].values() for v in item.values()))

    def test_pre_call_marker_precedes_invocation_and_preserves_return_identity(self):
        sink=Sink();a=CallAccounting(sink);value=object()
        def operation():
            self.assertEqual(sink.events[-1]["event"],"operation_attempt")
            self.assertEqual(a.snapshot()["operations"]["forward"]["pending"],1)
            return value
        self.assertIs(a.call("forward",operation,stage="fixture",input_positions=82),value)
        state=a.snapshot()
        self.assertEqual(state["operations"]["forward"],{"attempted":1,"returned":1,"failed":0,"pending":0})
        self.assertEqual(state["forward_input_positions_attempted"],82) # Actual occupied input, including padding.
        self.assertEqual(state["forward_input_positions_returned"],82)
        self.assertEqual(state["training_forward_input_positions_attempted"],0)
        self.assertEqual(verify_call_journal(sink.events,state),state)

    def test_exception_after_intent_but_before_callable_leaves_pending(self):
        events=[];called=[]
        def interrupted_sink(kind,**values):
            events.append({"event":kind,**values})
            raise OSError("synthetic interruption after persisted intent")
        a=CallAccounting(interrupted_sink)
        with self.assertRaises(OSError):
            a.call("forward",lambda:called.append(True),stage="training_256",input_positions=256,training=True)
        self.assertEqual(called,[])
        state=a.snapshot()
        self.assertEqual(state["operations"]["forward"],{"attempted":1,"returned":0,"failed":0,"pending":1})
        self.assertEqual(state["training_forward_input_positions_attempted"],256)
        self.assertEqual(state["training_forward_input_positions_returned"],0)
        self.assertEqual(verify_call_journal(events,state),state)

    def test_exception_inside_operation_preserves_failed_attempt_and_positions(self):
        sink=Sink();a=CallAccounting(sink)
        def failing():raise RuntimeError("synthetic forward failure")
        with self.assertRaisesRegex(RuntimeError,"synthetic forward"):
            a.call("forward",failing,stage="training_512",input_positions=512,training=True)
        state=a.snapshot()
        self.assertEqual(state["operations"]["forward"],{"attempted":1,"returned":0,"failed":1,"pending":0})
        self.assertEqual(state["forward_input_positions_attempted"],512)
        self.assertEqual(state["training_forward_input_positions_attempted"],512)
        self.assertEqual(state["forward_input_positions_returned"],0)
        self.assertEqual(sink.events[-1]["event"],"operation_failed")
        self.assertEqual(sink.events[-1]["error_type"],"RuntimeError")
        self.assertEqual(verify_call_journal(sink.events,state),state)

    def test_post_return_hook_failure_is_not_failed_compute(self):
        sink=Sink();a=CallAccounting(sink);legacy={"optimizer_steps":0,"training_tokens":0}
        def bookkeeping():
            legacy["optimizer_steps"]+=1;legacy["training_tokens"]+=1024
            raise ValueError("synthetic post-return hook failure")
        with self.assertRaisesRegex(ValueError,"post-return"):
            a.call("optimizer_step",lambda:None,stage="resume_reloaded",on_return=bookkeeping)
        self.assertEqual(legacy,{"optimizer_steps":1,"training_tokens":1024})
        state=a.snapshot()
        self.assertEqual(state["operations"]["optimizer_step"],{"attempted":1,"returned":1,"failed":0,"pending":0})
        self.assertEqual(sink.events[-1]["event"],"operation_returned")
        self.assertEqual(verify_call_journal(sink.events,state),state)

    def test_later_numerical_check_cannot_erase_returned_step(self):
        sink=Sink();a=CallAccounting(sink);completed=[]
        try:
            a.call("optimizer_step",lambda:None,stage="resume_uninterrupted",on_return=lambda:completed.append(1024))
            raise RuntimeError("synthetic nonfinite parameter check")
        except RuntimeError:
            pass
        self.assertEqual(completed,[1024])
        state=a.snapshot()
        self.assertEqual(state["operations"]["optimizer_step"]["returned"],1)
        self.assertEqual(state["operations"]["optimizer_step"]["failed"],0)
        verify_call_journal(sink.events,state)

    def test_nested_checkpoint_span_with_failed_copy_is_preserved(self):
        sink=Sink();a=CallAccounting(sink)
        def load():
            a.call("tensor_copy",lambda:None,stage="tensor_copy:first")
            def failing():raise ArithmeticError("synthetic tensor-copy failure")
            a.call("tensor_copy",failing,stage="tensor_copy:second")
        with self.assertRaises(ArithmeticError):
            a.call("checkpoint_load",load,stage="checkpoint_load")
        state=a.snapshot()
        self.assertEqual(state["operations"]["checkpoint_load"],{"attempted":1,"returned":0,"failed":1,"pending":0})
        self.assertEqual(state["operations"]["tensor_copy"],{"attempted":2,"returned":1,"failed":1,"pending":0})
        self.assertEqual([r["call_id"] for r in sink.events],[1,2,2,3,3,1])
        verify_call_journal(sink.events,state)

    def test_hard_kill_prefix_reconstructs_multiple_pending_calls(self):
        sink=Sink();a=CallAccounting(sink)
        a.begin("checkpoint_load",stage="checkpoint_load")
        a.begin("tensor_copy",stage="tensor_copy:first")
        recovered=verify_call_journal(sink.events)
        self.assertEqual(recovered,a.snapshot())
        self.assertEqual([c["call_id"] for c in recovered["pending_calls"]],[1,2])
        self.assertEqual(recovered["operations"]["checkpoint_load"]["pending"],1)
        self.assertEqual(recovered["operations"]["tensor_copy"]["pending"],1)

    def test_journal_write_failure_after_return_cannot_forge_consistency(self):
        events=[]
        def fail_return_event(kind,**values):
            if kind=="operation_returned":raise OSError("synthetic disk failure")
            events.append({"event":kind,**values})
        a=CallAccounting(fail_return_event)
        with self.assertRaises(OSError):a.call("model_construct",lambda:None,stage="model_construct")
        self.assertEqual(a.snapshot()["operations"]["model_construct"]["returned"],1)
        recovered=verify_call_journal(events)
        self.assertEqual(recovered["operations"]["model_construct"]["pending"],1)
        with self.assertRaisesRegex(RuntimeError,"snapshot mismatch"):verify_call_journal(events,a.snapshot())

    def test_nested_journal_interruption_keeps_outer_and_inner_pending(self):
        events=[]
        def sink(kind,**values):
            events.append({"event":kind,**values})
            if kind=="operation_attempt" and values["operation"]=="tensor_copy":raise OSError("synthetic log interruption")
        a=CallAccounting(sink)
        with self.assertRaises(OSError):
            a.call("checkpoint_load",lambda:a.call("tensor_copy",lambda:None,stage="tensor_copy:a"),stage="checkpoint_load")
        self.assertEqual(len(a.snapshot()["pending_calls"]),2)
        verify_call_journal(events,a.snapshot())

    def test_optimizer_limit_prevents_dispatch_and_extra_attempt(self):
        sink=Sink();a=CallAccounting(sink);executed=[]
        for i in range(5):a.call("optimizer_step",lambda:executed.append(1),stage="step_"+str(i))
        before=a.snapshot();events=deepcopy(sink.events)
        with self.assertRaisesRegex(RuntimeError,"OPTIMIZER_CALL_BUDGET"):
            a.call("optimizer_step",lambda:executed.append(99),stage="sixth")
        self.assertEqual(executed,[1]*5)
        self.assertEqual(a.snapshot(),before)
        self.assertEqual(sink.events,events)
        verify_call_journal(events,before)

    def test_failed_optimizer_attempt_still_consumes_budget(self):
        sink=Sink();a=CallAccounting(sink,maximum_optimizer_calls=1)
        def failing():raise ValueError("synthetic optimizer failure")
        with self.assertRaises(ValueError):a.call("optimizer_step",failing,stage="first")
        with self.assertRaisesRegex(RuntimeError,"OPTIMIZER_CALL_BUDGET"):
            a.call("optimizer_step",lambda:None,stage="second")
        verify_call_journal(sink.events,a.snapshot(),maximum_optimizer_calls=1)

    def test_metadata_and_budget_types_are_strict_before_attempt(self):
        sink=Sink();a=CallAccounting(sink)
        for positions,training in ((True,False),(1.0,False),(0,False),(-1,False),(3,1)):
            with self.subTest(positions=positions,training=training),self.assertRaises(RuntimeError):
                a.call("forward",lambda:None,stage="bad",input_positions=positions,training=training)
        with self.assertRaises(RuntimeError):a.call("backward",lambda:None,stage="bad",input_positions=1)
        with self.assertRaises(RuntimeError):a.call("forward",lambda:None,stage=" ",input_positions=1)
        for maximum in (True,5.0,-1,6):
            with self.subTest(maximum=maximum),self.assertRaises(RuntimeError):CallAccounting(sink,maximum)
        self.assertEqual(a.snapshot()["last_call_id"],0)
        self.assertEqual(sink.events,[])

    def test_journal_tampering_nesting_and_snapshot_mismatch_rejected(self):
        sink=Sink();a=CallAccounting(sink)
        a.call("checkpoint_load",lambda:a.call("tensor_copy",lambda:None,stage="tensor_copy:a"),stage="checkpoint_load")
        original=deepcopy(sink.events)
        mutations=[]
        bad=deepcopy(original);bad[2],bad[3]=bad[3],bad[2];mutations.append(bad)
        bad=deepcopy(original);bad[1]["call_id"]=9;mutations.append(bad)
        bad=deepcopy(original);bad[2]["stage"]="different";mutations.append(bad)
        bad=deepcopy(original);bad[0]["accounting_version"]=True;mutations.append(bad)
        bad=deepcopy(original);bad[0]["event"]="operation_invented";mutations.append(bad)
        bad=deepcopy(original);bad[0]["input_positions"]=0.0;mutations.append(bad)
        bad=deepcopy(original);bad.append(deepcopy(bad[-1]));mutations.append(bad)
        bad=deepcopy(original);bad[0]["extra"]=0;mutations.append(bad)
        for index,events in enumerate(mutations):
            with self.subTest(index=index),self.assertRaises(RuntimeError):verify_call_journal(events)
        state=a.snapshot();state["operations"]["tensor_copy"]["returned"]=0
        with self.assertRaisesRegex(RuntimeError,"snapshot mismatch"):verify_call_journal(original,state)

    def test_completed_profiles_survive_later_call_failure(self):
        sink=Sink();a=CallAccounting(sink);result={"resource_profiles":[]}
        a.call("forward",lambda:None,stage="training_256",input_positions=256,training=True)
        measured={"context":256,"loss":1.5,"seconds":2.0}
        record_resource_profile(result["resource_profiles"],measured,sink)
        measured["loss"]=999
        def failing():raise RuntimeError("synthetic later forward failure")
        with self.assertRaises(RuntimeError):a.call("forward",failing,stage="training_512",input_positions=512,training=True)
        self.assertEqual(result["resource_profiles"],[{"context":256,"loss":1.5,"seconds":2.0}])
        self.assertEqual(sink.events[2],{"event":"resource_profile","profile":result["resource_profiles"][0]})
        verify_call_journal(sink.events,a.snapshot())

    def test_profile_is_retained_even_if_profile_event_sink_fails(self):
        profiles=[]
        def failing(*args,**kwargs):raise OSError("synthetic profile sink failure")
        with self.assertRaises(OSError):record_resource_profile(profiles,{"context":256},failing)
        self.assertEqual(profiles,[{"context":256}])

    def test_snapshot_copies_do_not_mutate_internal_counts(self):
        sink=Sink();a=CallAccounting(sink);a.begin("forward",stage="fixture",input_positions=9,training=True)
        snap=a.snapshot();snap["operations"]["forward"]["attempted"]=999;snap["pending_calls"][0]["stage"]="changed"
        self.assertEqual(a.snapshot()["operations"]["forward"]["attempted"],1)
        self.assertEqual(a.snapshot()["pending_calls"][0]["stage"],"fixture")

    def test_wrapper_does_not_use_rng(self):
        state=random.getstate()
        try:
            random.seed(12345);before=random.getstate()
            a=CallAccounting(Sink());a.call("forward",lambda:None,stage="fixture",input_positions=3,training=True)
            self.assertEqual(before,random.getstate())
        finally:random.setstate(state)

    def test_full_registered_trace_call_counts_and_position_formula(self):
        # Independently enumerate the source workload, using no tensors or model.
        for generated in (1,17,32):
            sink=Sink();a=CallAccounting(sink);legacy={"optimizer_steps":0,"training_tokens":0}
            def invoke(op,stage,positions=0,training=False,hook=None):
                return a.call(op,lambda:None,stage=stage,input_positions=positions,training=training,on_return=hook)
            invoke("model_construct","model_construct")
            a.call("checkpoint_load",lambda:[invoke("tensor_copy",f"tensor_copy:fixture_{i:03}") for i in range(310)],stage="checkpoint_load")
            contexts=(256,512,1024)
            for prefix in ("base_parity","zero_parity"):
                for i in range(32):invoke("forward",f"{prefix}_{i}",contexts[i%3])
            invoke("forward","padding_reference",80)
            for left,right in ((2,0),(0,2),(2,2)):invoke("forward",f"native_padding_{left}_{right}",80+left+right)
            invoke("forward","causality_perturbed",80)
            for i in range(generated):invoke("forward",f"sample_{i}",80+i)
            invoke("forward","sampled_logprobs_offline",80+generated)
            def update(length,stage):
                invoke("forward",stage,length,True);invoke("backward",stage)
                def complete():legacy["optimizer_steps"]+=1;legacy["training_tokens"]+=length
                invoke("optimizer_step",stage,hook=complete)
            for length in contexts:update(length,f"training_{length}")
            invoke("checkpoint_save","adapter_resume_save")
            update(1024,"resume_uninterrupted")
            invoke("checkpoint_reload","adapter_resume_reload")
            update(1024,"resume_reloaded")
            snapshot=a.snapshot()
            expected={"model_construct":1,"checkpoint_load":1,"tensor_copy":310,"forward":75+generated,
                      "backward":5,"optimizer_step":5,"checkpoint_save":1,"checkpoint_reload":1}
            with self.subTest(generated=generated):
                self.assertEqual({op:c["attempted"] for op,c in snapshot["operations"].items()},expected)
                self.assertTrue(all(c["attempted"]==c["returned"] and c["failed"]==c["pending"]==0 for c in snapshot["operations"].values()))
                self.assertEqual(snapshot["last_call_id"],399+generated)
                expanded=(2*sum(contexts[i%3] for i in range(32))+80+82+82+84+80
                          +sum(80+i for i in range(generated))+(80+generated)+sum(contexts)+1024+1024)
                self.assertEqual(expanded,41704+81*generated+generated*(generated-1)//2)
                self.assertEqual(snapshot["forward_input_positions_attempted"],expanded)
                self.assertEqual(snapshot["forward_input_positions_returned"],expanded)
                self.assertEqual(snapshot["training_forward_input_positions_attempted"],3840)
                self.assertEqual(snapshot["training_forward_input_positions_returned"],3840)
                self.assertEqual(legacy,{"optimizer_steps":5,"training_tokens":3840})
                verify_call_journal(sink.events,snapshot)

    def test_import_and_helper_use_no_torch_or_tokenizer(self):
        root=Path(__file__).resolve().parents[3]
        code=("import os,sys;before=dict(os.environ);"
              "from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting;"
              "a=CallAccounting(lambda *a,**k:None);a.call('forward',lambda:None,stage='fixture',input_positions=3);"
              "assert dict(os.environ)==before;assert not any(n.startswith(('torch','transformers','tokenizers')) for n in sys.modules)")
        result=subprocess.run([sys.executable,"-B","-c",code],cwd=root,env=dict(os.environ),capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__":
    unittest.main()
