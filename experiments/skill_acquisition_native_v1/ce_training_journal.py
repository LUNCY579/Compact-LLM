"""Concrete native-journal role: delegate the existing backend before load."""
from . import ce_native_authority as authority
from .ce_native_backend import NativeCEBackend
from .ce_native_services import StableDispatcher


def native_journal(context,prepared,proof,resume,reason,operation,gate):
    permit=context.get('native_execution_permit') if type(context) is dict else None
    authority.require_execution_permit(permit,context,'journal_construct',operation=operation,gate=gate)
    authority.require(context['phase']=='training' and context.get('qualifiedreceipt_verified') is True
        and context.get('production_training_authorized') is True,'qualified separate production journal scope')
    backend=context.get('native_services',{}).get('training_backend')
    authority.require(type(backend) is NativeCEBackend and backend.context is context and backend.permit is permit
        and backend.operation is operation and backend.gate is gate and backend.model is backend.optimizer is None,
        'exact metadata production backend before journal/model allocation')
    authority.require(type(operation) is StableDispatcher and operation.ledger is None,'one native CE ledger attachment')
    return backend.open_journal(context,prepared,proof,resume,reason)
