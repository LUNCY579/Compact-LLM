"""Tiny admission metadata controls only; no terminal/model/decoder audit."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import auditor


def check(value,message):
    if not value:raise ValueError(message)


def fixture():
    sources={str((auditor.package/name).resolve()):"1"*64 for name in
             (*auditor.contract.SOURCE_NAMES,*auditor.contract.HELPER_PINS)}
    core={"CPU_FIXTURE_RUNTIME_%02d"%i:"2"*64 for i in range(64)}
    extension={"CPU_FIXTURE/ast.py":"3"*64}
    profile={"id":auditor.PROFILE_ID,"runtime_extension_pins":extension,"helper_pins":{"CPU_FIXTURE_HELPER":"4"*64},
             "CPU_freshness_seconds":.2,"CPU_cadence_seconds":.1,"native_launch_authorized":False}
    inventories={"source":sources,"runtime":{**core,**extension},"decoder":{"CPU_DECODER_%02d"%i:"5"*64 for i in range(17)},
                 "data":{"CPU_DATA_%02d"%i:"6"*64 for i in range(5)},"model":{"CPU_NO_MODEL_%02d"%i:"7"*64 for i in range(7)}}
    registration={"id":auditor.REGISTRATION_ID,"schema_version":1,"contract":auditor.CONTRACT_VERSION,
        "source_commit":"a"*40,"profile_body_sha256":auditor.digest(profile),"inventories":inventories,
        "counts":{k:len(v) for k,v in inventories.items()},"qualified_runtime64_sha256":auditor.digest(core)}
    registration["sha256"]=auditor.digest(registration)
    scope={"contract":auditor.CONTRACT_VERSION,"short_RAM_monitor_profile":profile,
           **{key:deepcopy(inventories[section]) for section,key in auditor.SECTIONS.items()}}
    sup={"preflight":{"source_commit":"a"*40},"short_RAM_monitor_diagnostic":{"path":"SHORT_RAM_ABORT_DIAGNOSTIC.json",
         "attempted":False,"persisted":False,"write_error":None,"original_abort_reason_preserved":None}}
    return scope,sup,registration,auditor.digest(profile),deepcopy(profile),deepcopy(sources)


class AdmissionControls(unittest.TestCase):
    def test_original_full_raw_decode_numeric_cost_resource_base_joins_preserved(self):
        original=(HERE/'original_auditor_18f6.py.txt').read_text()
        current=(HERE/'auditor.py').read_text()
        start=" events=jsonl(leaf/'events.jsonl')";stop=" def compact(values):"
        self.assertEqual(original[original.index(start):original.index(stop)],current[current.index(start):current.index(stop)])

    def test_complete_tiny_registered_inventory_admitted_without_IO(self):
        with patch.object(auditor.contract.native,"filehash",side_effect=AssertionError("heavy hash")), \
             patch.object(auditor.contract.native,"read_json",side_effect=AssertionError("heavy read")):
            counts=auditor._registered_admission(*fixture(),check)
        self.assertEqual(counts,{"source":8,"runtime":65,"decoder":17,"data":5,"model":7})

    def test_unknown_old_profile_or_contract_rejected(self):
        for mutate in (lambda s,u,r:s["short_RAM_monitor_profile"].update(id="unknown"),
                       lambda s,u,r:s.update(contract="synth-short-M0-one-fixed-batch-raw-collection-1")):
            values=list(fixture());mutate(*values[:3])
            with self.assertRaises(ValueError):auditor._registered_admission(*values,check)

    def test_changed_helper_callback_policy_profile_hash_rejected(self):
        for key,value in (("helper_pins",{"CPU_FIXTURE_HELPER":"f"*64}),("CPU_freshness_seconds",.3),("CPU_cadence_seconds",.2)):
            values=list(fixture());values[0]["short_RAM_monitor_profile"][key]=value
            with self.assertRaisesRegex(ValueError,"exact trusted new-profile"):
                auditor._registered_admission(*values,check)

    def test_omitted_runtime_or_replaced_AST_is_not_runtime65(self):
        for omitted in ("CPU_FIXTURE/ast.py","CPU_FIXTURE_RUNTIME_00"):
            values=list(fixture());values[0]["runtime_pins"].pop(omitted)
            with self.assertRaisesRegex(ValueError,"every frozen path/hash retained runtime"):
                auditor._registered_admission(*values,check)
        values=list(fixture());values[0]["runtime_pins"]["CPU_FIXTURE/ast.py"]="f"*64
        with self.assertRaises(ValueError):auditor._registered_admission(*values,check)

    def test_omitted_or_changed_source_fails_exact_map_not_length(self):
        for mode in ("omit","replace_same_count"):
            values=list(fixture());source=values[0]["source_pins"];key=next(iter(source))
            source.pop(key)
            if mode=="replace_same_count":source["CPU_SUBSTITUTE_SOURCE"]="1"*64
            with self.assertRaisesRegex(ValueError,"every frozen path/hash retained source"):
                auditor._registered_admission(*values,check)

    def test_diagnostic_success_contradictions_rejected(self):
        for key,value in (("attempted",True),("persisted",True),("write_error","injected"),
                          ("original_abort_reason_preserved","stale_external_CPU_watchdog"),("path","other.json")):
            values=list(fixture());values[1]["short_RAM_monitor_diagnostic"][key]=value
            with self.assertRaisesRegex(ValueError,"diagnostic contradiction"):
                auditor._registered_admission(*values,check)

    def test_untrusted_registration_changes_require_external_pin_not_self_checksum(self):
        values=list(fixture());values[2]["source_commit"]="b"*40
        values[2]["sha256"]=auditor.digest({k:v for k,v in values[2].items() if k!="sha256"})
        with self.assertRaisesRegex(ValueError,"selected source commit"):
            auditor._registered_admission(*values,check)

    def test_no_signal_or_missing_registration_anchor_denies_before_all_IO(self):
        args={"expected_selector":("atomic_RS_DEV",4),"expected_config_file_sha256":"a"*64,
              "expected_preservation_body_sha256":"b"*64}
        with patch.object(auditor.contract.native,"filehash",side_effect=AssertionError("hash")) as hasher, \
             patch.object(auditor.contract.native,"read_json",side_effect=AssertionError("read")) as reader:
            with self.assertRaisesRegex(ValueError,"terminal and preservation signal"):
                auditor.audit_terminal("unused","unused","unused",**args)
            with self.assertRaisesRegex(ValueError,"inventory/profile handshake"):
                auditor.audit_terminal("unused","unused","unused",terminal_preservation_signal=auditor.SIGNAL,**args)
            reader.assert_not_called();hasher.assert_not_called()
        self.assertFalse(any(n in sys.modules for n in ("torch","transformers","tokenizers")))


if __name__=="__main__":unittest.main()
