"""Pure service sequencing/source ingestion fixtures; no native controller permit."""
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_controller as c


def request(phase='qualification', arm='canonical'):
    refs = {key: {'path': '/' + key + '.json', 'sha256': c.digest(key)} for key in c.REFS}
    refs['qualifiedreceipt_ref'] = {'path': '/qualification.json', 'sha256': 'f' * 64}
    bounds = {key: 1000000 for key in c.BOUND_FIELDS}
    value = c.build_request(phase, arm, refs, bounds)
    value['mode'] = 'synthetic_controller_fixture'
    return value


class FixtureJournal:
    def __init__(self, C=0): self.p = {'C': C, 'U': C, 'generation': 0, 'snapshot': 'a' * 64}; self.used = False
    def progress_record(self): return copy.deepcopy(self.p)
    def commit_boundary(self, value):
        c.require(value['C'] == self.p['C'] and value['U_after'] == self.p['U'] + 1, 'fixture commit')
        self.p['C'] += 1; self.p['U'] += 1
    def install_restore(self, value):
        c.require(not self.used and value['generation'] > self.p['generation'], 'fresh fixture fence')
        self.p = copy.deepcopy(value); self.used = True


class Services:
    def __init__(self, req, *, denied=None, failure=None, resume_C=374):
        self.req, self.events, self.denied, self.failure, self.resume_C = req, [], denied, failure, resume_C
        self.authority = object(); self.sealed = False; self.journal = None
    def preflight_bootstrap(self, bounds, request_hash): self.events.append('bootstrap')
    def admit_request(self, req):
        self.events.append('admit')
        return {'schema': c.CONTEXT, 'phase': req['phase'], 'arm': req['arm'],
            'mode': 'synthetic_fixture', 'request_sha256': c.digest(c.request_preimage(req)),
            'scope_sha256': req['scope_ref']['sha256'], 'static_recipe_sha256': 'd' * 64,
            'admission_evidence_verified': True, 'qualifiedreceipt_verified': True,
            'production_training_authorized': True,
            'stage_plan': [{'stage': name, 'ordinal': i, 'caps': {key: 1 for key in c.BOUND_FIELDS}}
                           for i, name in enumerate(c.QUALIFICATION_STAGES)],
            'execution': {'boundaries': 375, 'checkpoint_boundaries': [0, 125, 250, 375]}}
    def observer(self, context):
        def observe(name, phase, **fields):
            if self.sealed: raise AssertionError('operation after terminal seal')
            self.events.append(name + ':' + phase)
            if name == self.denied and phase == 'started': raise c.AdmissionError('exhausted declared cap')
        return observe
    def make_outer_admission(self, req, context):
        def reserve(authority, stage, refs):
            self.events.append('outer:' + stage)
            if stage == self.denied: raise c.AdmissionError('exhausted outer cap')
        return c.OuterAdmission(req, context, self.authority, reserve,
            lambda handle, rh, sh, phase: handle is self.authority and rh == context['request_sha256']
                and sh == req['scope_ref']['sha256'] and phase == req['phase'])
    def verify_sources(self, req, context, outer):
        outer('ingestion_sources', {'preparation': req['preparation_ref'], 'plan': req['plan_ref'], 'arm': req['arm']})
        self.events.append('sources'); return {'verified': True}
    def verify_worker(self, context):
        self.events.append('worker')
        return {'fresh_process': True, 'boot_verified': True, 'boot_identity_sha256': 'e' * 64}
    @contextmanager
    def lease(self, context):
        self.events.append('lease_enter')
        try: yield
        finally:
            if self.sealed: raise AssertionError('lease exit after terminal seal')
            self.events.append('lease_exit')
            if self.failure == 'lease_exit': raise RuntimeError('physical lease release failure')
    def guard(self, context, stage): self.events.append('guard:' + stage)
    @contextmanager
    def watchdog(self, context):
        self.events.append('watchdog_enter')
        try: yield
        finally:
            if self.sealed: raise AssertionError('watchdog exit after terminal seal')
            self.events.append('watchdog_exit')
            if self.failure == 'watchdog_exit': raise RuntimeError('physical watchdog close failure')
    def configure_runtime(self, context): self.events.append('configure')
    def prepare_rows(self, verified, context, outer):
        outer('ingestion_tokenizer', {'preparation': self.req['preparation_ref'], 'plan': self.req['plan_ref'], 'arm': self.req['arm']})
        self.events.append('prepare')
        return SimpleNamespace(receipt={'schema': 'fixture_ingestion_not_native'},
            iter_boundary=lambda C: [{'id': 'b%d-r%d' % (C, i)} for i in range(8)])
    def open_journal(self, context, prepared, proof, resume, reason):
        self.events.append('journal'); self.journal = FixtureJournal(self.resume_C if resume else 0); return self.journal
    def checkpoints(self, context, backend, journal): self.events.append('checkpoints'); return object()
    def cleanup(self, context, backend):
        if self.sealed: raise AssertionError('cleanup after terminal seal')
        self.events.append('cleanup')
        if self.failure == 'cleanup': raise RuntimeError('physical model close failure')
    def publish_completion(self, context, artifacts, journal): self.events.append('publication'); return {'fixture': True}
    def seal(self, context, publication, journal): self.events.append('seal'); self.sealed = True; return {'fixture_seal': True}
    def interruption(self, context, error, journal):
        if self.sealed: raise AssertionError('interruption after terminal seal')
        self.events.append('interruption')


class Factory:
    synthetic_only = True
    def __init__(self, services): self.services = services
    def __call__(self, context): self.services.events.append('factory'); return Backend(self.services)


class Backend:
    def __init__(self, services): self.services = services
    def qualify_stage(self, name, prepared, artifacts, context):
        self.services.events.append('work:' + name)
        if name == self.services.failure: raise RuntimeError('physical fixture failure')
        return {'fixture_observation': name, 'passed': False}
    def load(self, context, prepared, observer): self.services.events.append('load')
    def save(self, checkpoints, journal): self.services.events.append('save:%d' % journal.p['C'])
    def restore(self, checkpoints, journal):
        self.services.events.append('restore'); value = journal.progress_record(); value['generation'] += 1; return value
    def refresh_policy(self, journal): self.services.events.append('actual_policy')
    def update(self, rows, journal):
        self.services.events.append('update:%d' % journal.p['C'])
        if self.services.failure == 'update': raise RuntimeError('partial Adam fixture')
        return {'C': journal.p['C'], 'U_after': journal.p['U'] + 1}
    def donor(self, context, journal): self.services.events.append('donor'); return {'fixture_donor': True}


class ControllerTests(unittest.TestCase):
    def test_explicit_checkpoint_schedule_includes_first_update_without_cadence_rounding(self):
        schedule = [0, 1, 75, 150, 225, 300, 375]
        self.assertEqual(c.checkpoint_boundaries({'boundaries': 375, 'checkpoint_boundaries': schedule}), tuple(schedule))
        for bad in (None, [], [0, 75, 75, 375], [0, True, 375], [0, 375, 1], [1, 375], [0, 374]):
            with self.subTest(schedule=bad), self.assertRaises(c.AdmissionError):
                c.checkpoint_boundaries({'boundaries': 375, 'checkpoint_boundaries': bad})

    def test_unapproved_and_forged_native_backend_refuse_before_any_service(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('service before admission')
        value = request(); value['mode'] = 'unapproved_proposal'
        with self.assertRaisesRegex(c.AdmissionError, 'before IO'): c.run_controller(value, Trap())
        value.update(mode='registered_scope', approved=True, qualification_authorized=True,
                     training_authorized=False, status='AUTHORIZED_QUALIFICATION_ONLY')
        forged = SimpleNamespace(registered_capabilities=set(c.ACTIVATION_GAPS))
        with self.assertRaises(c.MissingCapability): c.run_controller(value, Trap(), forged)

    def test_counts_are_strict_and_no_seed_price_or_worker_default(self):
        value = request(); value['bootstrap_bounds']['io_read_bytes_upper'] = True
        with self.assertRaisesRegex(c.AdmissionError, 'strict finite count'): c.validate_request(value, synthetic_fixture=True)
        value = request(); context = {'request_sha256': c.digest(c.request_preimage(value)), 'request_file_sha256': 'f' * 64}
        with self.assertRaisesRegex(c.AdmissionError, 'worker'):
            c.build_backend_command(value, context, {'path': '/request', 'sha256': 'f' * 64}, synthetic_fixture=True)
        self.assertNotIn('training_seed', value)

    def test_qualification_has_no_prior_PASS_dependency_and_cleanup_precedes_seal(self):
        value = request(); value['qualifiedreceipt_ref'] = None; services = Services(value)
        result = c.run_controller(value, services, Factory(services), synthetic_fixture=True)
        events = services.events
        self.assertLess(events.index('bootstrap'), events.index('sources'))
        self.assertLess(events.index('worker'), events.index('lease_enter'))
        self.assertLess(events.index('lease_enter'), events.index('configure'))
        self.assertLess(events.index('prepare'), events.index('factory'))
        self.assertLess(events.index('ingestion_tokenizer:returned'), events.index('runtime_anchor:started'))
        self.assertLess(events.index('cleanup'), events.index('publication'))
        self.assertLess(events.index('cleanup'), events.index('watchdog_exit'))
        self.assertLess(events.index('watchdog_exit'), events.index('lease_exit'))
        self.assertLess(events.index('lease_exit'), events.index('publication'))
        self.assertLess(events.index('completion_publication:returned'), events.index('seal'))
        self.assertFalse(result['native_qualified']); self.assertTrue(services.sealed)

    def test_cap_denial_precedes_backend_work_and_retains_failure_artifact(self):
        value = request(); services = Services(value, denied='runtime_anchor')
        with self.assertRaisesRegex(c.AdmissionError, 'exhausted'): c.run_controller(value, services, Factory(services), synthetic_fixture=True)
        self.assertNotIn('work:runtime_anchor', services.events); self.assertFalse(services.sealed)
        self.assertLess(services.events.index('cleanup'), services.events.index('interruption'))
        services = Services(value, failure='adam_clipping')
        with self.assertRaisesRegex(RuntimeError, 'physical'): c.run_controller(value, services, Factory(services), synthetic_fixture=True)
        self.assertIn('adam_clipping:failed', services.events); self.assertNotIn('publication', services.events)
        self.assertLess(services.events.index('cleanup'), services.events.index('watchdog_exit'))
        self.assertLess(services.events.index('watchdog_exit'), services.events.index('lease_exit'))

    def test_context_exit_or_cleanup_failure_cannot_publish_or_seal_success(self):
        for phase in ('qualification', 'training'):
            for failure in ('watchdog_exit', 'lease_exit', 'cleanup'):
                with self.subTest(phase=phase, failure=failure):
                    value = request(phase); services = Services(value, failure=failure)
                    with self.assertRaisesRegex(RuntimeError, 'physical'):
                        c.run_controller(value, services, Factory(services), synthetic_fixture=True)
                    self.assertFalse(services.sealed)
                    self.assertNotIn('publication', services.events)
                    self.assertNotIn('seal', services.events)
                    self.assertEqual(services.events.count('cleanup'), 1)
                    self.assertLess(services.events.index('cleanup'), services.events.index('watchdog_exit'))
                    self.assertLess(services.events.index('watchdog_exit'), services.events.index('lease_exit'))
                    self.assertLess(services.events.index('lease_exit'), services.events.index('interruption'))

    def test_context_cannot_suppress_failed_work_and_cleanup_preserves_primary(self):
        class SuppressingServices(Services):
            @contextmanager
            def watchdog(self, context):
                self.events.append('watchdog_enter')
                try: yield
                except RuntimeError: self.events.append('suppression_requested')
                finally: self.events.append('watchdog_exit')
        value = request(); services = SuppressingServices(value, failure='adam_clipping')
        with self.assertRaisesRegex(RuntimeError, 'physical fixture failure'):
            c.run_controller(value, services, Factory(services), synthetic_fixture=True)
        self.assertIn('suppression_requested', services.events)
        self.assertNotIn('publication', services.events); self.assertFalse(services.sealed)
        class FailedCleanupServices(Services):
            def cleanup(self, context, backend):
                super().cleanup(context, backend)
                raise RuntimeError('secondary close failure')
        services = FailedCleanupServices(value, failure='adam_clipping')
        with self.assertRaisesRegex(RuntimeError, 'physical fixture failure') as caught:
            c.run_controller(value, services, Factory(services), synthetic_fixture=True)
        self.assertEqual(caught.exception.cleanup_failure['message'], 'secondary close failure')
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertLess(services.events.index('cleanup'), services.events.index('watchdog_exit'))
        self.assertFalse(services.sealed)

    def test_training_checkpoint_zero_final_dedup_and_resume_fence_before_update(self):
        value = request('training'); services = Services(value)
        c.run_controller(value, services, Factory(services), synthetic_fixture=True)
        self.assertEqual([v for v in services.events if v.startswith('save:')], ['save:0', 'save:125', 'save:250', 'save:375'])
        self.assertEqual(services.journal.p['C'], 375)
        resumed = Services(value)
        c.run_controller(value, resumed, Factory(resumed), synthetic_fixture=True, resume=True, reason='owned checkpoint')
        self.assertLess(resumed.events.index('restore'), resumed.events.index('actual_policy'))
        self.assertLess(resumed.events.index('actual_policy'), resumed.events.index('update:374'))
        self.assertEqual([v for v in resumed.events if v.startswith('save:')], ['save:375'])
        failed = Services(value, failure='update')
        with self.assertRaisesRegex(RuntimeError, 'partial'): c.run_controller(value, failed, Factory(failed), synthetic_fixture=True)
        self.assertEqual(failed.journal.p['C'], 0); self.assertIn('interruption', failed.events); self.assertFalse(failed.sealed)

    def test_outer_capability_receipt_is_not_native_model_authority(self):
        value = request(); services = Services(value); context = services.admit_request(value)
        with self.assertRaisesRegex(c.AdmissionError, 'opaque'):
            c.OuterAdmission(value, context, {'approved': True}, lambda *a: None, lambda *a: True)
        outer = services.make_outer_admission(value, context)
        receipt = outer('ingestion_sources', {'preparation': value['preparation_ref'], 'arm': value['arm']})
        self.assertEqual(receipt['schema'], c.OUTER)
        context['mode'] = 'registered_scope'
        with self.assertRaisesRegex(c.AdmissionError, 'registered native outer authority'):
            services.make_outer_admission(value, context)
        refs = {'preparation': value['preparation_ref'], 'arm': value['arm']}
        with self.assertRaisesRegex(c.AdmissionError, 'model work'): outer.native_permit('model_load', refs)

    def test_opaque_object_and_forged_verifier_cannot_promote_fixture_to_native(self):
        value = request(); services = Services(value); context = services.admit_request(value)
        outer = services.make_outer_admission(value, context)
        refs = {'preparation': value['preparation_ref'], 'plan': value['plan_ref'], 'arm': value['arm']}
        before = list(services.events)
        outer.context['mode'] = 'registered_scope'
        for invoke in (lambda: outer('ingestion_sources', refs),
                       lambda: outer.native_permit('ingestion_sources', refs)):
            with self.assertRaisesRegex(c.AdmissionError, 'registered native outer authority'): invoke()
        self.assertEqual(services.events, before)
        value.update(mode='registered_scope', approved=True, qualification_authorized=True,
                     training_authorized=False, status='AUTHORIZED_QUALIFICATION_ONLY')
        context['mode'] = 'registered_scope'
        with self.assertRaisesRegex(c.AdmissionError, 'registered native outer authority'):
            c.OuterAdmission(value, context, object(), lambda *args: None, lambda *args: True)

    def test_concrete_adapter_drives_actual_pure_ingestion_both_policies(self):
        from experiments.skill_acquisition_native_v1 import ce_ingestion as ing
        from experiments.skill_acquisition_native_v1.tests import test_ce_ingestion as fixtures
        fixtures.PureTests.setUpClass()
        try:
            for arm in ('canonical', 'permuted'):
                value = request(arm=arm); value['preparation_ref'] = fixtures.PureTests.prep_ref
                value['plan_ref'] = fixtures.PureTests.plan_refs[arm]
                services = Services(value); context = services.admit_request(value)
                outer = services.make_outer_admission(value, context); observed = []
                operation = lambda kind, phase, **fields: observed.append((kind, phase))
                verified = ing.verify_sources(value['preparation_ref'], value['plan_ref'], arm, outer,
                                              operation=operation, gate=lambda stage: None)
                prepared = ing.PreparedRows(verified, 'fixture', outer_admission=outer, operation=operation,
                                           gate=lambda stage: None, renderer=fixtures.fake_render)
                self.assertEqual(len(prepared.iter_boundary(0)), 8)
                self.assertTrue(observed); self.assertIn('outer:ingestion_sources', services.events)
                self.assertIn('outer:ingestion_tokenizer', services.events)
                self.assertFalse(prepared.receipt['training_authorized'])
        finally: fixtures.PureTests.tearDownClass()


if __name__ == '__main__': unittest.main()
