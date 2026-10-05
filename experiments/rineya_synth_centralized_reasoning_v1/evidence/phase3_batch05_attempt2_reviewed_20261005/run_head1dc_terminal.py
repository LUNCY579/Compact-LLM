"""Root-only fixed05A2/06A1 terminal preservation/audit at exact HEAD1DC.

Reuse the reviewed29790d preserver's sole selector adaptation. No dispatch,
probe, policy change, retry or native work occurs on import or preparation.
"""
from pathlib import Path
import argparse,hashlib,json,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE));import preserve_fixed_remaining as original_adapter
ADAPTER_PIN='ad7d5d616e54444d4f2430298d9f3043b4cdd6dcd180e0ae1fd510f34145abdc'
HEAD='1dc787533d4e57257c5ae20b8e5324fb49894b70'
PROFILE='2d5547910aa57b0509a532972d899ceb5c913e41274bfa773d25536e6d8c9676'
SCOPES={
 5:{'config':'SYNTH_SHORT_M0_ATOMIC_BATCH05_PHASE3_A1.json','config_sha':'b8e2efe1e832ca865bc8a5de8f48b9d601246cd14335a2e6e09ecc4a1362ba00',
    'inventory':'SYNTH_SHORT_PHASE3_BATCH05_A2_HEAD1DC_AUDIT_INVENTORY_001.json','inventory_sha':'f2fad910e7d3dd9f82793db0864d633f4bbf1cf656af7e082fcd7ae83a6aab96',
    'leaf':'short_m0_atomic_batch05_phase3_attempt2_20261005'},
 6:{'config':'SYNTH_SHORT_M0_ATOMIC_BATCH06_PHASE3_A1.json','config_sha':'06677a414195afb137daba2f434edd8f4ba0a4f82b029c52e17582d8fb6e095c',
    'inventory':'SYNTH_SHORT_PHASE3_BATCH06_A1_HEAD1DC_AUDIT_INVENTORY_001.json','inventory_sha':'9dba239f5ccce2ab604f2f80e1e8c17e14610382a36eb0c9c9d5ad1a6a01f847',
    'leaf':'short_m0_atomic_batch06_phase3_attempt1_20261005'}}
def scope(ordinal):
    original_adapter.require(type(ordinal) is int and ordinal in SCOPES,'explicit fixed ordinal5 or6 before reads')
    item=SCOPES[ordinal]
    return {**item,'config':PACKAGE/'configs'/item['config'],'inventory':PACKAGE/'artifacts/short_m0_audit_registrations'/item['inventory'],
        'raw':PACKAGE/'artifacts'/item['leaf'],'saved':PACKAGE/'evidence'/item['leaf']}
def preserve_fixed(ordinal,*,sup_sha,terminal_signal=None,current_death_path=None,current_death_sha=None):
    item=scope(ordinal)
    original_adapter.require(terminal_signal==original_adapter.SIGNAL,'Root terminal-frozen copy signal required before reads')
    original_adapter.require(original_adapter.hashpin(sup_sha),'externally selected frozen supervisor SHA')
    original_adapter.require(hashlib.sha256(Path(original_adapter.__file__).read_bytes()).hexdigest()==ADAPTER_PIN,'exact reviewed one-site adapter bytes')
    body,_=original_adapter.adapted(ordinal)
    # The old wrapper's HEAD/inventory constants are historical. Its exact
    # adapted preservation body is reused with only this new frozen call frame.
    return body.preserve(item['config'],item['raw'],item['saved'],item['inventory'],config_sha=item['config_sha'],
        inventory_sha=item['inventory_sha'],sup_sha=sup_sha,commit=HEAD,terminal_signal=terminal_signal,
        current_death_path=current_death_path,current_death_sha=current_death_sha)
def audit_fixed(ordinal,*,preservation_sha,root_terminal_and_preserved=False):
    item=scope(ordinal)
    original_adapter.require(root_terminal_and_preserved is True and original_adapter.hashpin(preservation_sha),
        'Root successful-terminal preservation signal and exact body SHA before reads')
    sys.path.insert(0,str(PACKAGE/'artifacts/short_phase_audit_work'));import audit_adapter as auditor
    receipt=auditor.phase.read_json(item['saved']/'preservation_receipt.json')
    original_adapter.require(receipt.get('status')=='SUCCESS_TERMINAL_BYTES_PRESERVED_FULL_AUDIT_PENDING'
        and receipt.get('full_native_audit_eligible') is True and receipt.get('sha256')==preservation_sha,
        'exact successful preserved evidence only; failed/no-worker/CPU receipts never native audited')
    return auditor.audit_terminal(item['config'],item['raw'],item['saved'],expected_selector=('atomic_RS_DEV',ordinal),
        expected_config_file_sha256=item['config_sha'],expected_preservation_body_sha256=preservation_sha,
        inventory_registration_path=item['inventory'],expected_inventory_registration_file_sha256=item['inventory_sha'],
        expected_profile_body_sha256=PROFILE,expected_source_commit=HEAD,terminal_preservation_signal=auditor.SIGNAL,
        output_path=PACKAGE/'evidence'/(item['leaf']+'_independent_audit_20261005.json'))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('preserve','audit'));parser.add_argument('--ordinal',type=int,choices=(5,6),required=True)
    parser.add_argument('--root-terminal-frozen',action='store_true');parser.add_argument('--supervisor-file-sha256')
    parser.add_argument('--current-death-evidence');parser.add_argument('--current-death-file-sha256')
    parser.add_argument('--root-terminal-and-preserved',action='store_true');parser.add_argument('--preservation-body-sha256')
    args=parser.parse_args()
    if args.action=='preserve':
        result=preserve_fixed(args.ordinal,sup_sha=args.supervisor_file_sha256,terminal_signal=original_adapter.SIGNAL if args.root_terminal_frozen else None,
            current_death_path=args.current_death_evidence,current_death_sha=args.current_death_file_sha256)
        print(json.dumps({'status':result['status'],'preservation_body_sha256':result['sha256'],'full_native_audit_eligible':result['full_native_audit_eligible']}))
    else:
        result=audit_fixed(args.ordinal,preservation_sha=args.preservation_body_sha256,root_terminal_and_preserved=args.root_terminal_and_preserved)
        print(json.dumps({'status':result['status'],'checks_passed':result['checks_passed'],'sha256':result['sha256']}))
if __name__=='__main__':main()
