"""Root-only actual successful audit command. Not executed during preparation."""
from pathlib import Path
import argparse,json,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE));import preserve_terminal as preserve
sys.path.insert(0,str(PACKAGE/'artifacts/short_phase_audit_work'));import audit_adapter
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-terminal-and-preserved',action='store_true')
    parser.add_argument('--preservation-body-sha256',required=True)
    args=parser.parse_args();preserve.pure.require(args.root_terminal_and_preserved,'explicit Root successful terminal-and-preserved signal')
    leaf=PACKAGE/'artifacts'/preserve.LEAF_NAME;saved=PACKAGE/'evidence'/preserve.LEAF_NAME
    receipt=preserve.pure.read_json(saved/'preservation_receipt.json')
    preserve.pure.require(receipt['sha256']==args.preservation_body_sha256 and receipt['full_native_audit_eligible'] is True
        and receipt['status']=='SUCCESS_TERMINAL_BYTES_PRESERVED_FULL_AUDIT_PENDING','exact successful preservation anchor; failed/CPU evidence never audited as success')
    result=audit_adapter.audit_terminal(preserve.registration.CONFIG,leaf,saved,expected_selector=('atomic_RS_DEV',4),
        expected_config_file_sha256=preserve.registration.CONFIG_SHA,expected_preservation_body_sha256=args.preservation_body_sha256,
        inventory_registration_path=preserve.INVENTORY,expected_inventory_registration_file_sha256=preserve.INVENTORY_SHA,
        expected_profile_body_sha256=preserve.registration.PROFILE_SHA,expected_source_commit=preserve.HEAD,
        terminal_preservation_signal=audit_adapter.SIGNAL,
        output_path=PACKAGE/'evidence'/(preserve.LEAF_NAME+'_independent_audit_20261005.json'))
    print(json.dumps({'status':result['status'],'checks_passed':result['checks_passed'],'sha256':result['sha256']}))
if __name__=='__main__':main()
