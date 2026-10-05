"""V2 metadata and whole-frame CPU fixtures. No actual native observations."""
from copy import deepcopy
from pathlib import Path
import hashlib,importlib.util,json,sys,unittest
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=="long_v2_audit_adaptation_work" and HERE.parent.name=="artifacts"
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(PACKAGE))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
d=module("explicit_long_terminal_driver_v2",CODE/"long_terminal_driver_v2.py")
a=d.v
legacy=module("explicit_unchanged_terminal_fixture_builder",PACKAGE/"tests/test_long_terminal_driver.py")
FIXTURE=HERE/"fixtures/V2_DECLARATION_METADATA_ONLY.json" if STAGED else PACKAGE/"evidence/long_v2_audit_adaptation_reviewed_cpu_20261005/fixtures/V2_DECLARATION_METADATA_ONLY.json"
assert Path(d.__file__).resolve()==(CODE/"long_terminal_driver_v2.py").resolve()
DECLARATION=json.loads(FIXTURE.read_text(encoding="utf-8"))
assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()=="f2a768b4fc062e8b86d5126d179e5a6a250008189b3fba64f22f877bd6f43fa4"
def resign(value):return d.seal({k:v for k,v in value.items() if k!="sha256"})
def rebuild(frame,anchors):
    """Resign only fixture envelopes: adversarial signatures are not authority."""
    scope=frame["scope"];reg=resign(scope["encoder_runtime_registration"]);scope["encoder_runtime_registration"]=reg
    scope["recipe"]["declared_execution_scope_registration_sha256"]=reg["sha256"]
    scope["recipe"]=resign(scope["recipe"]);sha=scope["recipe"]["sha256"];frame["scope"]=resign(scope)
    result=frame["result"];node=result["capsule_chain"];cap=node["capsule"]
    for proof in (cap["start_proof"],cap["end_proof"]):proof["identity_common"]["recipe_sha256"]=sha
    cap["recipe_sha256"]=sha;cap=resign(cap);node["capsule"]=cap;node["recipe_sha256"]=sha
    node["terminal"]["capsule_sha256"]=cap["sha256"];node["promotion"]["capsule_sha256"]=cap["sha256"]
    node["promotion"]["terminal_sha256"]=d.digest(node["terminal"]);result["capsule_chain"]=resign(node)
    result["recipe_sha256"]=sha;result["native_admission"]["binding"]["recipe_sha256"]=sha;frame["result"]=resign(result)
    anchors["recipe_sha256"]=sha;inventory=frame["inventory"];inventory["recipe_sha256"]=sha
    inventory["maps"]["runtime"]=deepcopy(scope["runtime_pins"]);inventory["counts"]["runtime"]=len(scope["runtime_pins"])
    frame["inventory"]=resign(inventory)
def frame():
    value,anchors=legacy.Controls().frame();scope=value["scope"];reg=deepcopy(DECLARATION)
    scope["runtime_pins"]=deepcopy(reg["runtime_pins"]);scope["encoder_runtime_registration"]=reg
    maps=value["result"]["capsule_chain"]["capsule"]["start_proof"]["actual_pin_maps"]
    maps["runtime"]=deepcopy(reg["runtime_pins"])
    std=reg["stdlib_inventory"]
    maps["runtime_observation"]={"policy":"ADMITTED_ENCODER45_FINITE_STDLIB_BYTE_OBSERVATION_V2","phase":"before_import",
        "actual_additional_runtime_pins":deepcopy(reg["additional_runtime_pins"]),"encoder_versions":deepcopy(reg["encoder_inventory"]["versions"]),
        "stdlib_roster":{"paths":deepcopy(std["paths"]),"approved_alias_binding":deepcopy(std["approved_alias_binding"]),"optional_stdlib_zip":deepcopy(std["optional_stdlib_zip"])},
        "native_qualification_established":False,"complete_machine_trust_claimed":False}
    cap=value["result"]["capsule_chain"]["capsule"];cap["start_proof"]["actual_pin_maps"]=maps
    cap["start_proof"]["identity_common"]["source_byte_maps_sha256"]=d.digest(maps)
    cap["end_proof"]["identity_common"]=deepcopy(cap["start_proof"]["identity_common"])
    cap["end_proof"]["actual_pin_maps_after"]=deepcopy(maps);cap["end_proof"]["actual_pin_maps_after"]["runtime_observation"]["phase"]="after_math"
    value["inventory"]["validator_file_sha256"]=d.VALIDATOR_SHA256;rebuild(value,anchors);return value,anchors
def proofs(value):
    cap=value["result"]["capsule_chain"]["capsule"]
    return cap["start_proof"]["actual_pin_maps"],cap["end_proof"]["actual_pin_maps_after"]
class Controls(unittest.TestCase):
    def reject(self,value,anchors):
        rebuild(value,anchors)
        with self.assertRaises((ValueError,KeyError)):d.validate_frame(value,anchors,CPU_fixture_mode=True)
    def test_complete_V2_whole_frame_exact887_only_CPU_status(self):
        value,anchors=frame();report=d.validate_frame(value,anchors,CPU_fixture_mode=True)
        self.assertEqual(d.CPU_KIND,report["kind"]);self.assertFalse(report["native_AUDIT_KIND_PASS_issued"]);self.assertFalse(report["native_ready"])
        self.assertFalse(report["raw_RNG_restore_or_tensor_reread_verified"]);self.assertEqual(887,len(value["scope"]["runtime_pins"]))
    def test_phase_only_normalization_preserves_original_before_digest(self):
        value,_=frame();before,after=proofs(value)
        self.assertNotEqual(before,after);self.assertEqual(a.normalize_runtime_phase_only(before),a.normalize_runtime_phase_only(after))
        self.assertEqual("before_import",before["runtime_observation"]["phase"]);self.assertEqual("after_math",after["runtime_observation"]["phase"])
        self.assertNotEqual(d.digest(before),d.digest(a.normalize_runtime_phase_only(before)))
    def test_missing_seventh_section_rejected_even_resigned(self):
        value,anchors=frame()
        for maps in proofs(value):maps.pop("runtime_observation")
        self.reject(value,anchors)
    def test_wrong_same_or_reversed_phase_rejected(self):
        for phase in ("before_import","unknown",None):
            value,anchors=frame();proofs(value)[1]["runtime_observation"]["phase"]=phase;self.reject(value,anchors)
        value,anchors=frame();before,after=proofs(value);before["runtime_observation"]["phase"]="after_math";after["runtime_observation"]["phase"]="before_import";self.reject(value,anchors)
    def test_no_other_observation_field_can_be_normalized(self):
        for key in ("encoder_versions","stdlib_roster","actual_additional_runtime_pins"):
            value,anchors=frame();proof=proofs(value)[1]["runtime_observation"];proof[key].pop(next(iter(proof[key])));self.reject(value,anchors)
    def test_observation_extra_field_or_native_flags_rejected(self):
        for key,new in (("extra",1),("native_qualification_established",True),("complete_machine_trust_claimed",True)):
            value,anchors=frame();proofs(value)[1]["runtime_observation"][key]=new;self.reject(value,anchors)
    def test_normalized_instead_of_actual_before_map_digest_rejected(self):
        value,anchors=frame();cap=value["result"]["capsule_chain"]["capsule"]
        for item in (cap["start_proof"],cap["end_proof"]):item["identity_common"]["source_byte_maps_sha256"]=d.digest(a.normalize_runtime_phase_only(proofs(value)[0]))
        self.reject(value,anchors)
    def test_V1_or_boolean_only_authority_rejected(self):
        for kind in ("SYNTH_LONG_ENCODER_RUNTIME_REGISTRATION_V1",None):
            value,anchors=frame();value["scope"]["encoder_runtime_registration"]["kind"]=kind;self.reject(value,anchors)
        value,anchors=frame();value["scope"]["encoder_runtime_registration"]["native_launch_authorized"]=True;self.reject(value,anchors)
    def test_exact_inherited65_not_just_count_union(self):
        value,anchors=frame();reg=value["scope"]["encoder_runtime_registration"];key=next(k for k in reg["inherited_runtime_pins"] if k not in reg["additional_runtime_pins"])
        reg["inherited_runtime_pins"][key]="f"*64;reg["runtime_pins"][key]="f"*64;value["scope"]["runtime_pins"][key]="f"*64
        self.reject(value,anchors)
    def test_missing_or_substituted_current_consumption_source_rejected(self):
        for change in ("missing","substitute"):
            value,anchors=frame();pins=value["scope"]["encoder_runtime_registration"]["consumption_source_pins"];key=next(iter(pins))
            if change=="missing":pins.pop(key)
            else:pins[key]="f"*64
            self.reject(value,anchors)
    def test_registered_union_conflict_or_omission_rejected(self):
        for change in ("conflict","omission"):
            value,anchors=frame();reg=value["scope"]["encoder_runtime_registration"]
            if change=="conflict":key=next(k for k in reg["inherited_runtime_pins"] if k in reg["additional_runtime_pins"]);reg["inherited_runtime_pins"][key]="f"*64
            else:reg["additional_runtime_pins"].pop(next(iter(reg["additional_runtime_pins"])))
            self.reject(value,anchors)
    def test_alias_reparse_optionalZIP_policy_mutation_rejected(self):
        for kind in ("alias","zip","category","version"):
            value,anchors=frame();std=value["scope"]["encoder_runtime_registration"]["stdlib_inventory"]
            if kind=="alias":std["approved_alias_binding"]["reparse_tag"]=0
            elif kind=="zip":std["optional_stdlib_zip"]["present"]=True
            elif kind=="category":std["paths"][next(iter(std["paths"]))]="arbitrary"
            else:std["python_version"]="3.12"
            self.reject(value,anchors)
    def test_encoder_version_or_roster_filehash_rejected(self):
        for kind in ("version","filehash","roster"):
            value,anchors=frame();enc=value["scope"]["encoder_runtime_registration"]["encoder_inventory"]
            if kind=="version":enc["versions"]["tokenizers"]="0.0"
            elif kind=="filehash":enc["files"][next(iter(enc["files"]))]["sha256"]="f"*64
            else:enc["packages"]["tokenizers"]["files"].pop()
            self.reject(value,anchors)
    def test_failed_terminal_or_wrong_ROLE_HEAD_expired_admission_rejected(self):
        for kind in ("failed","role","head","admission"):
            value,anchors=frame()
            if kind=="failed":value["supervisor"]["status"]="OWNED_WORKER_ABORTED"
            elif kind=="role":value["result"]["native_admission"]["binding"]["role"]="bulk"
            elif kind=="head":value["supervisor"]["preflight"]["source_commit"]="f"*40
            else:value["result"]["native_admission"]["expires_monotonic"]=10.3
            self.reject(value,anchors)
    def test_actual_runtime_observation_hash_drift_rejected(self):
        value,anchors=frame();maps=proofs(value)[1];key=next(iter(maps["runtime_observation"]["actual_additional_runtime_pins"]))
        maps["runtime_observation"]["actual_additional_runtime_pins"][key]="f"*64;self.reject(value,anchors)
    def test_staged_metadata_fixture_is_not_final_registration(self):
        import encoder_registration_v2 as r
        with self.assertRaises(ValueError):r.registration({"path":str(FIXTURE.resolve()),"sha256":hashlib.sha256(FIXTURE.read_bytes()).hexdigest()},a.contract.base_runtime_pins())
    def test_fixture_or_pending_cost_cannot_be_native(self):
        for kind in ("fixture","cost"):
            value,anchors=frame()
            if kind=="fixture":value["result"]["capsule_chain"]["capsule"]["CPU_fixture"]=True
            else:value["result"]["actual_costs"]["operations"]["model_construct"].update(returned=0,pending=1)
            self.reject(value,anchors)
    def test_native_entry_remains_denied_before_IO(self):
        value,anchors=frame()
        with self.assertRaisesRegex(ValueError,"PASS emission disabled"):d.audit_terminal("UNUSED","UNUSED","UNUSED","UNUSED",terminal_signal=a.SIGNAL,anchors=anchors)
        self.assertFalse(d.NATIVE_PASS_ENABLED);self.assertFalse(a.NATIVE_DRIVER_READY)
if __name__=="__main__":unittest.main()
