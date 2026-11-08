"""Same long-lived original metadata owner; target qualification is an output.

Invoke only after exact source review. Imports are metadata-only. Preparation
pins precede the fresh root; all target CLR startup is observed from birth.
"""
import sys
import admission as a
def main():
    a.require(len(sys.argv)==1,'fixed original same-owner entry')
    prep=a.preparation_owner()
    packet,_=a.fixed('PACKET.json',owner=prep)
    a.validate_packet(packet) # exact paths/scope/source identities before any role execution
    issuer=a.module(packet['source_refs']['issue_grant.py'],'ce8_prepared_issuer_entry',owner=prep)
    harness=a.module(packet['source_refs']['harness.py'],'ce8_prepared_harness_entry',owner=prep)
    permit=issuer.prepare_and_issue()
    primary=None
    try:return harness.main(permit)
    except BaseException as error:primary=error;raise
    finally:
        if not permit.custody._closed and not a._PIN_RELEASE_ATTEMPTED:
            # Only an entry refusal before harness failure handling may use
            # this original plan. Never retry already attempted failed pins.
            try:
                permit.owner.begin_final_tail()
                a.release_original_pins(permit.custody,permit.owner.pin_close)
            except BaseException as error:
                if primary is None:raise
                primary.add_note('last original pin cleanup: '+type(error).__name__)
        elif not permit.custody._closed:
            if primary is not None:primary.add_note('original pin plan already attempted; unresolved pins retained, no unselected retry')
            else:raise RuntimeError('unresolved original pins require external cleanup decision')
if __name__=='__main__':raise SystemExit(main())
