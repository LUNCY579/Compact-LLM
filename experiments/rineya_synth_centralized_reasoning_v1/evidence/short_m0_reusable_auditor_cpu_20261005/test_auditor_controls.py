"""Light framework-free handshake/terminal controls; never a heavy audit."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import auditor


def check(value,message):
    if not value:raise ValueError(message)


def fixture():
    rows=[{"row_id":"%064x"%index,"prompt_sha256":"0"*64,"prompt_ids":[100],"prompt_tokens":1,"attention_mask":[1]} for index in range(64)]
    scope={"contract":auditor.CONTRACT_VERSION,"authorization":deepcopy(auditor.contract.AUTHORIZATION),
           "profile":"BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1","batch_selector":{"source_kind":"atomic_RS_DEV","ordinal":2},
           "rows":rows,"source_packet_sha256":auditor.digest(rows)}
    worker={"status":"COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED","error":None,"source_commit":"1"*40,
            "collection":{"rows":[{"row_id":row["row_id"]} for row in rows],"source_packet_sha256":scope["source_packet_sha256"]}}
    supervisor={"status":"SUPERVISED_WORKER_EXITED_ZERO","returncode":0,"partial_outputs_frozen":True,"preflight":{"source_commit":"1"*40}}
    return scope,worker,supervisor,{"source_commit":"1"*40}


class Controls(unittest.TestCase):
    def test_default_signal_denies_before_any_read_hash_or_write(self):
        with patch.object(auditor.contract.native,"read_json",side_effect=AssertionError("read")) as reader,patch.object(auditor.contract.native,"filehash",side_effect=AssertionError("hash")) as hasher:
            with self.assertRaisesRegex(ValueError,"terminal and preservation signal"):
                auditor.audit_terminal("unused","unused","unused",expected_selector=("atomic_RS_DEV",2),expected_config_file_sha256="0"*64,expected_preservation_body_sha256="1"*64)
            reader.assert_not_called();hasher.assert_not_called()
        self.assertFalse(any(name in sys.modules for name in ("torch","transformers","tokenizers")))
    def test_bad_independent_hash_and_selector_deny_before_paths(self):
        for config_hash,selector in (("bad",("atomic_RS_DEV",2)),("0"*64,("atomic_RS_DEV",7)),("0"*64,("atomic_RS_DEV",True)),("0"*64,("language",1))):
            with self.assertRaises(ValueError):
                auditor.audit_terminal("unused","unused","unused",expected_selector=selector,expected_config_file_sha256=config_hash,expected_preservation_body_sha256="1"*64,terminal_preservation_signal=auditor.SIGNAL)
    def test_complete_mock_header_retains_all64(self):
        auditor._early_terminal_checks(*fixture(),("atomic_RS_DEV",2),check)
    def test_incomplete_reordered_duplicate_or_unfrozen_terminal_denied(self):
        mutations=(lambda s,w,u,p:w["collection"]["rows"].pop(),lambda s,w,u,p:w["collection"]["rows"].reverse(),
                   lambda s,w,u,p:s["rows"][0].update(row_id=s["rows"][1]["row_id"]),lambda s,w,u,p:u.update(partial_outputs_frozen=False))
        for mutate in mutations:
            values=fixture();mutate(*values)
            with self.assertRaises(ValueError):auditor._early_terminal_checks(*values,("atomic_RS_DEV",2),check)
    def test_forged_return_status_error_or_commit_denied(self):
        mutations=(lambda s,w,u,p:u.update(returncode=137),lambda s,w,u,p:u.update(status="OWNED_WORKER_ABORTED"),
                   lambda s,w,u,p:w.update(status="SHORT_M0_RAW_COLLECTION_FAILED"),lambda s,w,u,p:w.update(error={"type":"Failure"}),
                   lambda s,w,u,p:p.update(source_commit="2"*40))
        for mutate in mutations:
            values=fixture();mutate(*values)
            with self.assertRaises(ValueError):auditor._early_terminal_checks(*values,("atomic_RS_DEV",2),check)
    def test_alternate_contract_profile_authority_or_selector_denied(self):
        mutations=(lambda s,w,u,p:s.update(contract="successor"),lambda s,w,u,p:s.update(profile="other"),
                   lambda s,w,u,p:s["authorization"].update(scoring=True),lambda s,w,u,p:s["batch_selector"].update(ordinal=3))
        for mutate in mutations:
            values=fixture();mutate(*values)
            with self.assertRaises(ValueError):auditor._early_terminal_checks(*values,("atomic_RS_DEV",2),check)
    def test_target_oracle_columns_rejected_even_with_resigned_packet(self):
        scope,worker,sup,pres=fixture();scope["rows"][0]["labels"]=[-100]
        scope["source_packet_sha256"]=worker["collection"]["source_packet_sha256"]=auditor.digest(scope["rows"])
        with self.assertRaisesRegex(ValueError,"prompt-only"):
            auditor._early_terminal_checks(scope,worker,sup,pres,("atomic_RS_DEV",2),check)
    def test_isolated_cli_help_and_missing_signal_denial(self):
        argv=[sys.executable,"-I","-B",str(HERE/"auditor.py")]
        result=subprocess.run(argv+["--help"],capture_output=True,text=True,timeout=30)
        self.assertEqual(0,result.returncode,result.stderr)
        denied=subprocess.run(argv+["--config","unused","--raw-directory","unused","--preserved-directory","unused",
                    "--kind","atomic_RS_DEV","--ordinal","2","--config-file-sha256","0"*64,"--preservation-body-sha256","1"*64],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(0,denied.returncode);self.assertIn("terminal and preservation signal",denied.stderr)


if __name__=="__main__":unittest.main()
