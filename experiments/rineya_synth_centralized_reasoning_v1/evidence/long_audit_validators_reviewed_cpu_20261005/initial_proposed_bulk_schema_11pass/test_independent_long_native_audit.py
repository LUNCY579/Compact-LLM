"""Pure generated JSON/token fixtures only; cannot emit native audit authority."""
from copy import deepcopy
from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=="long_native_audit_work" and HERE.parent.name=="artifacts"
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(CODE))
import auditor as a
class Controls(unittest.TestCase):
    def fixture(self,p=992,role="trace1",length=32):
        row={"prompt_tokens":p,"prompt_ids":[1]*p};scope={"role":role,"row":row,"short_parity_row":{"prompt_ids":[1]*419},
            "recipe":{"prefix_stress_ids":[1]*31 if role=="prefix_stress" else None}}
        scope["dose"]=a.contract.work(row,role)
        trace=[{"token":2,"logprob":-.25,"prefix_length":p+i} for i in range(length)]
        if length<32:trace[-1]["token"]=151645
        capsule={"trace":trace,"greedy_EOS_respected":role!="prefix_stress","full32_greedy_coverage":role!="prefix_stress" and length==32,"stress_is_greedy":False}
        roster=[("long_native_dense_parity_forward",[1]*419),("long_native_zeroB_parity_forward",[1]*419)]
        if role=="bulk":roster.append(("long_native_bulk_forward",row["prompt_ids"]+[t["token"] for t in trace[:-1]]))
        elif role=="prefix_stress":roster.extend(("long_native_declared_prefix_stress_forward",row["prompt_ids"]+[1]*i) for i in range(32))
        else:roster.extend(("long_native_greedy_forward",row["prompt_ids"]+[t["token"] for t in trace[:i]]) for i in range(length))
        capsule["physical_dose"]={"attempted_calls":len(roster),"input_positions":sum(len(ids) for _,ids in roster),
            "records":[{"operation":name,"input_ids_sha256":a.digest(ids),"input_positions":len(ids)} for name,ids in roster]}
        return scope,capsule
    def test_all_registered_windows_complete_exact_physical_prefix_caps_fixture_only(self):
        for p in (992,2016,4064):
            for role in a.contract.ROLES:
                scope,cap=self.fixture(p,role)
                a.validate_trace(scope,cap);value=a.validate_physical(scope,cap)
                self.assertEqual(3 if role=="bulk" else 34,value["attempted_calls"]);self.assertLessEqual(value["input_positions"],134144)
                self.assertNotIn("status",value)
    def test_earlyEOS_denominator_truthful_and_hidden_earlyEOS_or_truncation_rejected(self):
        scope,cap=self.fixture(length=3);info=a.validate_trace(scope,cap);self.assertEqual("eos",info["finish"]);self.assertFalse(info["full32_greedy_coverage"])
        self.assertEqual(5,a.validate_physical(scope,cap)["attempted_calls"])
        cap["trace"][0]["token"]=151645
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
        scope,cap=self.fixture(length=3);cap["trace"][-1]["token"]=2
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
    def test_stress_not_laundered_as_greedy_or_earlyEOS_coverage(self):
        scope,cap=self.fixture(role="prefix_stress");cap["trace"][0]["token"]=151645
        self.assertEqual("declared_prefix_stress",a.validate_trace(scope,cap)["finish"]);a.validate_physical(scope,cap)
        cap["full32_greedy_coverage"]=True
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
        scope,cap=self.fixture(role="prefix_stress");scope["recipe"]["prefix_stress_ids"][0]=151645
        with self.assertRaises(ValueError):a.validate_trace(scope,cap)
    def test_any_extra_model_call_wrong_prefix_operation_or_positions_rejected(self):
        for mutate in (lambda c:c["physical_dose"]["records"].append(deepcopy(c["physical_dose"]["records"][0])),
            lambda c:c["physical_dose"]["records"][2].update(input_ids_sha256="a"*64),
            lambda c:c["physical_dose"]["records"][2].update(operation="unregistered_forward"),
            lambda c:c["physical_dose"].update(input_positions=1)):
            scope,cap=self.fixture();mutate(cap)
            with self.assertRaises(ValueError):a.validate_physical(scope,cap)
    def test_typed_token_LP_prefix_and_actual32_denominator(self):
        for key,value in (("token",True),("token",151936),("logprob",float("nan")),("logprob",.1),("prefix_length",True)):
            scope,cap=self.fixture();cap["trace"][0][key]=value
            with self.assertRaises(ValueError):a.validate_trace(scope,cap)
    def test_cumulative_operations_no_erased_prior_or_rollback_or_partial(self):
        prior={"load":{"attempted":1,"returned":1,"failed":0,"pending":0}}
        now={"load":{"attempted":2,"returned":2,"failed":0,"pending":0},"forward":{"attempted":34,"returned":34,"failed":0,"pending":0}}
        self.assertEqual(prior["load"],a.delta(now,prior)["load"])
        with self.assertRaises(ValueError):a.delta({"forward":now["forward"]},prior)
        with self.assertRaises(ValueError):a.counts({"bad":{"attempted":1,"returned":0,"failed":0,"pending":1}})
    def bulk(self):
        scope,cap=self.fixture(role="bulk",length=3)
        previous=[{"capsule_chain":{"capsule":{"trace":deepcopy(cap["trace"])}}} for _ in range(2)]
        records=[{"index":i,"position":scope["row"]["prompt_tokens"]-1+i,"reference_token":step["token"],"reference_logprob":step["logprob"],
            "actual_argmax_token":step["token"],"actual_selected_logprob":step["logprob"],"absolute_error":0.} for i,step in enumerate(cap["trace"])]
        cap["comparison"]={"records":records,"actual_compared_steps":3,"argmax_differences":[],"maximum_error":0.,"mean_error":0.}
        return scope,cap,previous
    def test_complete_actual_bulk_perstep_maxmean_and_argmax_fixtures_only(self):
        self.assertTrue(a.validate_bulk_records(*self.bulk()))
    def test_summary_only_current_plumbing_cannot_issue_complete_bulk_audit(self):
        scope,cap,previous=self.bulk();cap["comparison"].pop("records")
        with self.assertRaisesRegex(ValueError,"every-step"):a.validate_bulk_records(scope,cap,previous)
    def test_bulk_missing_step_wrongLP_wrongerror_or_repeatedtrace_mismatch_rejected(self):
        for kind in ("step","LP","error","repeat","token"):
            scope,cap,previous=self.bulk()
            if kind=="step":cap["comparison"]["records"].pop()
            elif kind=="LP":cap["comparison"]["records"][0]["actual_selected_logprob"]=-1.
            elif kind=="error":cap["comparison"]["records"][0]["absolute_error"]=1.
            elif kind=="token":cap["comparison"]["records"][0]["actual_argmax_token"]=3
            else:previous[1]["capsule_chain"]["capsule"]["trace"][0]["token"]=3
            with self.assertRaises(ValueError):a.validate_bulk_records(scope,cap,previous)
    def test_signal_and_complete_externalanchors_required_before_IO(self):
        with self.assertRaises(ValueError):a.validate_anchors(None,{})
        with self.assertRaises(ValueError):a.validate_anchors(a.SIGNAL,{})
    def test_no_fixture_nativePASS_or_framework_import(self):
        self.assertFalse(any(n in sys.modules for n in ("torch","transformers","tokenizers","jinja2")))
        scope,cap=self.fixture();self.assertNotIn("status",cap);self.assertNotIn("execution_scope",cap)
if __name__=="__main__":unittest.main()

