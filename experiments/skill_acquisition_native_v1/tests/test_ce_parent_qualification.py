"""Pure concrete composition/control spies; no native permits, OS or models."""
import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_parent_qualification as p


class Trap:
    def __getattribute__(self,name): raise AssertionError('work before parent permission: '+name)


def metadata():
    permit=object(); operation=SimpleNamespace(ledger=None); gate=object()
    context={'phase':'qualification','native_execution_permit':permit,'native_services':{},
        'bindings':{'scope':{'qualifier_accounting':{'setup_bounds':{
            'read_bytes_upper':10,'write_bytes_upper':20,'identity_hash_attempts_upper':2,'identity_hash_bytes_upper':30,'wall_s_upper':1},
            'ingestion_bootstrap_ref':{'path':'C:/owned/INGESTION.json','sha256':'a'*64,'bytes':100},
            'outer_setup_reservation_ref':{'path':'C:/owned/PREPAYMENT.json','sha256':'b'*64,'bytes':100}}}}}
    service=SimpleNamespace(operation=operation,gate=gate,supervisor_lease=SimpleNamespace(backend=None),worker_handle=object())
    return context,permit,service


class ParentTests(unittest.TestCase):
    def test_all_public_native_entries_refuse_before_objects_or_constructor_work(self):
        for permit in (None,True,{'approved':True},object.__new__(p.authority.NativeExecutionPermit)):
            with self.assertRaises(ValueError): p.create_parent_qualification({'native_execution_permit':permit})
            with self.assertRaises(ValueError): p.reserve_parent_accounting(Trap(),permit,Trap(),Trap())
            with self.assertRaises(ValueError): p.parent_checkpoint_adapter(Trap(),permit,Trap(),Trap(),Trap(),Trap(),Trap())
            with self.assertRaises(ValueError): p.parent_resource_adapter(Trap(),permit,Trap(),Trap(),Trap(),Trap(),Trap())
        self.assertIsNone(p._BOOTSTRAP_ISSUER); self.assertIsNone(p._BOOTSTRAP_VERIFIER)
        self.assertIsNone(p._LAUNCH_ISSUER); self.assertIsNone(p._LAUNCH_VERIFIER)
        self.assertFalse(p.executor._REGISTERED_ACCOUNTING_FACTORIES)
        self.assertNotIn('torch',sys.modules); self.assertNotIn('transformers',sys.modules)

    def test_registered_configuration_denial_precedes_backend_or_os_owner(self):
        with patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_service_owner',side_effect=AssertionError('owner before config')):
            with self.assertRaisesRegex(ValueError,'unregistered'): p.create_parent_qualification({})

    def test_fixed_factory_constructs_actual_classes_and_fixed_functions(self):
        context,permit,service=metadata(); core=object.__new__(p.backend.NativeCEBackend)
        recorder=object.__new__(p.resource.NativeResourceRecorder)
        recorder.context=context; recorder.permit=permit; recorder.operation=service.operation
        service.bind_resource_recorder=lambda ctx,cap:recorder
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_registered_services',return_value=None),
              patch.object(p,'_service_owner',return_value=service),patch.object(p.backend,'NativeCEBackend',return_value=core) as backend,
              patch.object(p.executor,'NativeQualificationExecutor',return_value='pure-spy-result') as execute):
            result=p.create_parent_qualification(context)
        self.assertEqual(result,'pure-spy-result'); self.assertEqual(backend.call_args.args,(context,permit))
        functions=execute.call_args.args[3]
        self.assertIs(functions['accounting_factory'],p.reserve_parent_accounting)
        self.assertIs(functions['checkpoint_adapter'],p.parent_checkpoint_adapter)
        self.assertIs(functions['resource_adapter'],p.parent_resource_adapter)
        self.assertIs(context['native_services']['resource_recorder'],recorder)

    def test_metadata_executor_failure_closes_existing_backend_preserving_primary(self):
        context,permit,service=metadata(); calls=[]; failure=RuntimeError('executor construction failed')
        core=SimpleNamespace(cleanup=lambda:calls.append('cleanup'))
        recorder=object.__new__(p.resource.NativeResourceRecorder)
        recorder.context=context; recorder.permit=permit; recorder.operation=service.operation
        service.bind_resource_recorder=lambda ctx,cap:recorder
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_registered_services',return_value=None),
              patch.object(p,'_service_owner',return_value=service),patch.object(p.backend,'NativeCEBackend',return_value=core),
              patch.object(p.executor,'NativeQualificationExecutor',side_effect=failure)):
            with self.assertRaisesRegex(RuntimeError,'executor construction') as caught:p.create_parent_qualification(context)
        self.assertIs(caught.exception,failure);self.assertEqual(calls,['cleanup'])

    def test_actual_constructor_reservation_precedes_create_accounting_and_owner_is_exact(self):
        context,permit,service=metadata(); proof={'worker':'metadata-spy'}; receipt={'receipt_sha256':'c'*64}
        spec=context['bindings']['scope']['qualifier_accounting']
        context['qualifier_accounting_owner']={'accounting_spec_sha256':p.digest(spec),'prepared_receipt_sha256':receipt['receipt_sha256']}
        events=[]; token=object(); result=object.__new__(p.accounting.QualificationAccounting)
        result.context=context;result.bootstrap_handle=token
        outer=SimpleNamespace(reserve_setup=lambda owner,bounds:events.append(('reserve',copy.deepcopy(owner),bounds)) or token)
        def create(*args,**kwargs):
            events.append(('construct',kwargs)); return result
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_registered_services',return_value=None),
              patch.object(p,'_service_owner',return_value=service),patch.object(p.qualification,'validate_actual_ingestion',return_value=None),
              patch.object(p,'_bootstrap',return_value=outer),patch.object(p.accounting,'create_accounting',side_effect=create)):
            self.assertIs(p.reserve_parent_accounting(context,permit,proof,receipt),result)
        self.assertEqual([event[0] for event in events],['reserve','construct'])
        self.assertEqual(events[0][1],context['qualifier_accounting_owner'])
        self.assertEqual(events[0][2]['metadata_bytes_upper'],2*p.accounting.MAX_LINE)
        self.assertIs(events[1][1]['bootstrap_handle'],token)
        self.assertIs(context['native_services']['parent_qualification_accounting'],result)
        self.assertIsNone(service.operation.ledger)  # executor attaches, factory does not migrate costs

    def test_constructor_failure_marks_genuine_pending_ticket_failed_without_refund(self):
        context,permit,service=metadata(); proof={}; receipt={'receipt_sha256':'c'*64}
        context['qualifier_accounting_owner']={'accounting_spec_sha256':p.digest(context['bindings']['scope']['qualifier_accounting']),
                                               'prepared_receipt_sha256':receipt['receipt_sha256']}
        events=[]; token=SimpleNamespace(evidence={'status':'RESERVED'},fail=lambda error:events.append(error))
        outer=SimpleNamespace(reserve_setup=lambda owner,bounds:token); failure=RuntimeError('Q constructor failed')
        with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_registered_services',return_value=None),
              patch.object(p,'_service_owner',return_value=service),patch.object(p.qualification,'validate_actual_ingestion',return_value=None),
              patch.object(p,'_bootstrap',return_value=outer),patch.object(p.accounting,'create_accounting',side_effect=failure)):
            with self.assertRaisesRegex(RuntimeError,'Q constructor') as caught:p.reserve_parent_accounting(context,permit,proof,receipt)
        self.assertIs(caught.exception,failure);self.assertEqual(events,[failure])

    def test_missing_predecessor_refs_or_owner_reject_before_outer_constructor(self):
        for change in ('owner','ref'):
            context,permit,service=metadata(); receipt={'receipt_sha256':'c'*64}
            context['qualifier_accounting_owner']={'accounting_spec_sha256':p.digest(context['bindings']['scope']['qualifier_accounting']),
                                                   'prepared_receipt_sha256':receipt['receipt_sha256']}
            if change=='owner':context['qualifier_accounting_owner']['accounting_spec_sha256']='f'*64
            else:context['bindings']['scope']['qualifier_accounting'].pop('ingestion_bootstrap_ref')
            with (patch.object(p.authority,'require_execution_permit',return_value=None),patch.object(p,'_registered_services',return_value=None),
                  patch.object(p,'_service_owner',return_value=service),patch.object(p.qualification,'validate_actual_ingestion',return_value=None),
                  patch.object(p,'_bootstrap',side_effect=AssertionError('unpaid outer construction'))):
                with self.assertRaises(ValueError):p.reserve_parent_accounting(context,permit,{},receipt)


if __name__=='__main__':unittest.main()
