"""Payload-free fixtures. Run under the hardware-policy offline watchdog only.
No native worker or Torch import, model launch, global-lock mutation, or claim.
"""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import parentq_launch_wmi as launcher
import parentq_safety as safety
import parentq_supervise as supervisor

HERE = Path(__file__).resolve().parent
FIXTURE_ROOT = Path('C:/Users/clclc/BusinessProject/triskill-geo001-parentq-offline-20261005')


class FakeLease:
    def __init__(self, lock):
        self.LOCK, self.holder, self.race = lock, None, False
    def foreign_gpu_holder(self):
        return self.holder
    def acquire(self, label, job, timeout_seconds=0):
        if self.race:
            self.LOCK.write_text('{"token":"foreign"}', encoding='utf-8')
            if self._try_reclaim():
                self.LOCK.unlink()
            raise RuntimeError('fixture acquire race')
        with self.LOCK.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'token': 'tok', 'label': label, 'worker_pids': []}, stream)
        return 'tok'
    def release(self, token):
        if self.LOCK.exists() and json.loads(self.LOCK.read_bytes()).get('token') == token:
            self.LOCK.unlink()


class FakeHardware:
    def __init__(self):
        self.closed, self.records = False, []
    def record(self, payload):
        self.records.append(payload)
    def close(self):
        self.closed = True


class FakeChild:
    def __init__(self, normal=True, dies=True):
        self.pid, self.done, self.normal, self.dies = 777, False, normal, dies
    def wait(self, timeout=None):
        if self.done or self.normal:
            self.done = True
            return 0
        raise subprocess.TimeoutExpired('fixture worker', timeout)
    def poll(self):
        return 0 if self.done else None
    def kill(self):
        self.done = self.dies


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='supervisor-', dir=FIXTURE_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo, self.runtime = self.base/'source', self.base/'runtime'
        patcher=patch.object(safety,'RUNTIME_ROOT',self.runtime)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.here = self.repo/'experiments/triskill_geo001_parentq_v1'
        self.here.mkdir(parents=True)
        geo = self.here.parent/'triskill_geometric_pilot_v1'
        geo.mkdir()
        shutil.copyfile(HERE.parent/'triskill_geometric_pilot_v1/FREEZE_SECTIONH.json', geo/'FREEZE_SECTIONH.json')
        self.geo = json.loads((geo/'FREEZE_SECTIONH.json').read_bytes())
        dep = self.geo['supervisor_dependencies']['gpu_lease']
        target = self.repo/dep['path']
        target.parent.mkdir(parents=True)
        shutil.copyfile(HERE.parents[1]/dep['path'], target)
        for name in supervisor.PINNED_LOCAL:
            (self.here/name).write_text('# synthetic fixture\n', encoding='utf-8')
        readiness = {'status':'READY_FOR_REVIEWED_PROPOSAL',
                     'decisions':{name:{'status':'RESOLVED'} for name in safety.REQUIRED_DECISIONS},
                     'independent_review':{'status':'PASS'}, 'native_execution_authorized':False, 'fixture_only':True}
        (self.here/'READINESS_PARENTQ.json').write_text(json.dumps(readiness), encoding='utf-8')
        files = {name:hashlib.sha256((self.here/name).read_bytes()).hexdigest() for name in supervisor.PINNED_LOCAL}
        self.freeze = self.here/supervisor.FREEZE_NAME
        self.freeze.write_text(json.dumps({'files':files,'runtime_root':str(self.runtime),
                                          'geo_freeze':{'sha256':supervisor.GEO_FREEZE_SHA256},
                                          'supervisor_dependencies':{'gpu_lease':dep}}), encoding='utf-8')
        self.sha = hashlib.sha256(self.freeze.read_bytes()).hexdigest()
        self.attempt = self.runtime/supervisor.ATTEMPTS_DIR/'a1'
        self.attempt.mkdir(parents=True)
        self.lock, self.central = self.base/'gpu.lock', self.base/'central.lock'
        self.hardware, self.launched, self.kills = FakeHardware(), [], []
        self.lease, self.child = FakeLease(self.lock), FakeChild()
        self.live_worker = False
        self.identity = lambda pid:float(pid)+0.5
        self.grant = self.make_grant()

    def make_grant(self, number=1, text='fixture explicit operator grant'):
        return json.dumps({'experiment':'TRISKILL-GEO-001-PARENTQ', 'attempt_number':number,
                           'freeze_sha256':self.sha, 'operator_grant':text, 'hardware_slot':'FIXTURE_ONLY'})

    def run_supervisor(self, **overrides):
        def popen(command, **kwargs):
            self.launched.append((command, kwargs))
            path=self.attempt/supervisor.WORKER_PID_NAME
            if not path.exists():
                path.write_text(json.dumps({'pid':self.child.pid,'create_time':self.identity(self.child.pid),
                    'lease_token':'tok','freeze_sha256':self.sha}),encoding='utf-8')
            return self.child
        def kill(pid, created):
            self.kills.append((pid, created))
            if pid == self.child.pid:
                self.child.kill()
            else:
                self.live_worker = False
            return {'returncode':0}
        options = dict(here=self.here, repo=self.repo, python='fixture-python', popen=popen,
            clock=lambda:0, lease_module=self.lease, kill=kill,
            alive=lambda pid:pid==9001 and self.live_worker, sleep=lambda _:None,
            attempt_dir=self.attempt, runtime_root=self.runtime, attempt_number=1,
            central_path=self.central, hardware_factory=lambda:self.hardware,
            crs_db=self.base/'fixture.sqlite3', crs_check=lambda _: {}, identity=self.identity)
        options.update(overrides)
        return supervisor.supervise(self.freeze, self.sha, self.grant, **options)

    def receipt(self):
        return json.loads((self.attempt/supervisor.RECEIPT_NAME).read_bytes())

    def test_normal_exit_dual_token_external_output(self):
        result = self.run_supervisor()
        self.assertEqual(result['status'], 'WORKER_EXITED')
        self.assertTrue(result['lease_released'] and result['central_lease_released'] and self.hardware.closed)
        self.assertFalse(self.lock.exists() or self.central.exists())
        command, kwargs = self.launched[0]
        self.assertIn('-B',command)
        self.assertEqual(command[command.index('--out')+1], str(self.runtime/supervisor.OUT_NAME))
        self.assertEqual(kwargs['env']['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertTrue(Path(kwargs['env']['TEMP']).is_relative_to(self.runtime))
        self.assertTrue(Path(kwargs['env']['HF_HOME']).is_relative_to(self.runtime))
        start = json.loads((self.attempt/supervisor.START_NAME).read_bytes())
        self.assertEqual(start['attempt_number'], 1)
        self.assertEqual(start['supervisor_create_time'], self.identity(os.getpid()))

    def test_hard_wall_identity_kill_bounded_wait(self):
        self.child = FakeChild(normal=False)
        ticks = iter((0, 0, 601, 602))
        result = self.run_supervisor(clock=lambda:next(ticks))
        self.assertEqual(result['status'], 'STOP_SUPERVISOR_HARD_WALL')
        self.assertEqual(self.kills, [(777, 777.5)])
        self.assertTrue(result['worker_exited'])

    def test_failure_after_popen_cleans_child_and_records(self):
        self.child = FakeChild(normal=False)
        with patch.object(supervisor, 'set_worker_pids', side_effect=RuntimeError('fixture rewrite failure')):
            with self.assertRaisesRegex(RuntimeError, 'fixture rewrite'):
                self.run_supervisor()
        self.assertTrue(self.child.done)
        self.assertIn('SUPERVISOR_ERROR', self.receipt()['status'])
        self.assertTrue(self.receipt()['lease_released'])

    def test_taskkill_error_receipt_and_unresolved_leases_retained(self):
        self.child = FakeChild(normal=False, dies=False)
        ticks = iter((0, 0, 601, 602))
        def fails(*_):
            raise subprocess.TimeoutExpired('taskkill', 60)
        result = self.run_supervisor(clock=lambda:next(ticks), kill=fails)
        self.assertEqual(result['kill']['status'], 'TASKKILL_ERROR')
        self.assertFalse(result['lease_released'])
        self.assertTrue(self.lock.exists() and self.central.exists())
        self.assertEqual(self.receipt()['status'], 'STOP_SUPERVISOR_HARD_WALL')

    def test_worker_pid_cleanup_bound(self):
        self.live_worker = True
        (self.attempt/supervisor.WORKER_PID_NAME).write_text(json.dumps({'pid':9001,'create_time':9001.5,
            'lease_token':'tok','freeze_sha256':self.sha}), encoding='utf-8')
        result = self.run_supervisor()
        self.assertIn((9001,9001.5), self.kills)
        self.assertTrue(result['worker_exited'])

    def test_missing_worker_identity_retains_unresolved_lease(self):
        result=self.run_supervisor(popen=lambda *_a,**_k:self.child)
        self.assertEqual(result['worker_pid_record_error'],'MISSING_WORKER_IDENTITY_RECORD')
        self.assertFalse(result['worker_exited'] or result['lease_released'])
        self.assertTrue(self.lock.exists() and self.central.exists())

    def test_markers_count_killed_crashed_legacy_once(self):
        for index in range(3):
            directory = self.runtime/supervisor.ATTEMPTS_DIR/f'load{index}'
            directory.mkdir()
            for name in (supervisor.LOAD_BEGUN_NAME, supervisor.INCOMPLETE_NAME):
                (directory/name).write_text('{}', encoding='utf-8')
        self.assertEqual(supervisor.loaded_attempts(self.runtime,self.here), 3)
        with self.assertRaisesRegex(RuntimeError, 'ATTEMPTS_EXHAUSTED'):
            self.run_supervisor()
        self.assertFalse(self.launched)

    def test_foreign_locks_and_holder_never_reclaimed(self):
        for path,message in ((self.lock,'STOP_LEASE_PRESENT'),(self.central,'STOP_CENTRAL_LEASE_PRESENT')):
            path.write_text('foreign nonce', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,message):
                self.run_supervisor()
            self.assertEqual(path.read_text(encoding='utf-8'),'foreign nonce')
            path.unlink()
        self.lease.holder = (99,3<<30)
        with self.assertRaisesRegex(RuntimeError,'FOREIGN_GPU_HOLDER'):
            self.run_supervisor()

    def test_exists_create_race_disables_reclaim(self):
        self.lease.race = True
        with self.assertRaisesRegex(RuntimeError,'acquire race'):
            self.run_supervisor()
        self.assertFalse(self.lease._try_reclaim())
        self.assertEqual(json.loads(self.lock.read_bytes())['token'],'foreign')
        self.assertTrue(self.hardware.closed)

    def test_unreadable_lease_is_not_release_evidence(self):
        self.lock.write_text('unreadable',encoding='utf-8')
        lease=types.SimpleNamespace(LOCK=self.lock,release=lambda _: None)
        self.assertFalse(supervisor.release_own_lease(lease,'tok',sleep=lambda _: None))
        self.assertTrue(self.lock.exists())

    def test_ownership_loss_during_wait_cleans_child(self):
        self.child = FakeChild(normal=False)
        calls = [0]
        def check(_):
            calls[0]+=1
            if calls[0]>=3:
                raise RuntimeError('STOP_CRS_OWNER_ACTIVE_OR_UNRECONCILED')
        with self.assertRaisesRegex(RuntimeError,'CRS_OWNER_ACTIVE'):
            self.run_supervisor(crs_check=check)
        self.assertTrue(self.child.done and self.receipt()['worker_exited'])

    def test_grant_freeze_attempt_slot_replay(self):
        for bad in ('','plain operator grant',json.dumps({'attempt_number':1})):
            self.grant=bad
            with self.assertRaisesRegex(ValueError,'ATTEMPT_BOUND_OPERATOR_GRANT'):
                self.run_supervisor()
        self.grant=self.make_grant()
        for field,bad in (('attempt_number',True),('operator_grant',12345678901234567),('hardware_slot',None)):
            value=json.loads(self.grant)
            value[field]=bad
            with self.assertRaisesRegex(ValueError,'ATTEMPT_BOUND_OPERATOR_GRANT'):
                supervisor.verify_grant(json.dumps(value),1,self.sha,self.runtime,self.here)
        self.run_supervisor()
        with self.assertRaisesRegex(RuntimeError,'GRANT_ALREADY_USED'):
            supervisor.verify_grant(self.make_grant(2),2,self.sha,self.runtime,self.here)

    def test_freeze_geo_lease_readiness_and_pin(self):
        supervisor.verify_freeze(self.freeze,self.sha,self.here,self.repo)
        module=supervisor.load_gpu_lease(json.loads(self.freeze.read_bytes()),self.repo)
        self.assertFalse(module._try_reclaim())
        value=json.loads(self.freeze.read_bytes())
        value['supervisor_dependencies']['gpu_lease']['sha256']='a'*64
        self.freeze.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'PIN_DIFFERS_FROM_GEO'):
            supervisor.verify_freeze(self.freeze,hashlib.sha256(self.freeze.read_bytes()).hexdigest(),self.here,self.repo)
        (self.here/'READINESS_PARENTQ.json').write_text('{"status":"UNFROZEN_DECISIONS_PENDING"}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'DECISIONS_OR_REVIEW_PENDING'):
            safety.validate_readiness(self.here)

    def test_pid_reuse_and_taskkill_timeout(self):
        calls=[]
        result=supervisor.kill_tree(77,4.0,identity=lambda _:5.0,run=lambda *_a,**_k:calls.append(1))
        self.assertEqual(result['status'],'PID_IDENTITY_CHANGED')
        self.assertFalse(calls)
        def failure(*_a,**_k):
            raise subprocess.TimeoutExpired('taskkill',60)
        self.assertEqual(supervisor.kill_tree(77,4.0,identity=lambda _:4.0,run=failure)['status'],'TASKKILL_ERROR')

    def test_argparse_refusal_external_receipts(self):
        with patch.object(safety,'RUNTIME_ROOT',self.runtime):
            with self.assertRaises(SystemExit): supervisor.main([])
            with self.assertRaises(SystemExit): launcher.main([])
        self.assertEqual(len(list((self.runtime/supervisor.ATTEMPTS_DIR).glob('SUPERVISOR_FAILURE_*.json'))),1)
        self.assertEqual(len(list((self.runtime/supervisor.ATTEMPTS_DIR).glob('LAUNCH_RECEIPT_*.json'))),1)

    def test_wmi_error_always_writes_distinct_launch_receipts(self):
        argv=['--freeze',str(self.freeze),'--freeze-sha256',self.sha,'--grant',self.grant,
              '--runtime-root',str(self.runtime),'--attempt-number','1','--launch']
        with patch.object(supervisor,'verify_freeze',return_value={'runtime_root':str(self.runtime)}), \
                patch.object(supervisor,'verify_grant'), \
                patch.object(launcher,'launch',side_effect=RuntimeError('fixture WMI error')):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError,'fixture WMI error'): launcher.main(argv)
        paths=list((self.runtime/supervisor.ATTEMPTS_DIR).glob('LAUNCH_RECEIPT_*.json'))
        self.assertEqual(len(paths),2)
        self.assertTrue(all(json.loads(p.read_bytes())['status']=='LAUNCH_ERROR' for p in paths))

    def test_runtime_directory_rotation_cannot_reset_accounting(self):
        with self.assertRaisesRegex(ValueError,'RUNTIME_ROOT_PIN_MISMATCH'):
            self.run_supervisor(runtime_root=self.base/'rotated-runtime')
        value=json.loads(self.freeze.read_bytes())
        value['runtime_root']=str(self.base/'rotated-runtime')
        self.freeze.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'NONCANONICAL_NATIVE_RUNTIME_ROOT'):
            supervisor.verify_freeze(self.freeze,hashlib.sha256(self.freeze.read_bytes()).hexdigest(),self.here,self.repo)

    def test_launch_confirmation_matches_supervisor_pid(self):
        folder=self.runtime/supervisor.ATTEMPTS_DIR
        foreign=folder/'SUPERVISOR_FAILURE_foreign.json'
        foreign.write_text('{"supervisor_pid":999,"error":"foreign"}',encoding='utf-8')
        result=launcher.confirm_started(self.runtime,0,timeout=0,supervisor_pid=123)
        self.assertEqual(result['supervisor'],'UNCONFIRMED')
        foreign.write_text('{"supervisor_pid":123,"error":"owned"}',encoding='utf-8')
        self.assertEqual(launcher.confirm_started(self.runtime,0,timeout=0,supervisor_pid=123)['supervisor'],'FAILED')

    def test_hardware_exclusive_handle_surviving_file_is_reusable(self):
        path=self.base/'private-hardware.lock'
        handle=safety.HardwareHandle(path)
        try:
            handle.record({'fixture_only':True,'pid':os.getpid()})
            with self.assertRaisesRegex(RuntimeError,'HARDWARE_LOCK_BUSY'):
                safety.HardwareHandle(path)
        finally:
            handle.close()
        self.assertTrue(path.exists())
        again=safety.HardwareHandle(path)
        again.close()

    def test_per_unit_token_and_supervisor_identity_check(self):
        self.run_supervisor()
        start=json.loads((self.attempt/supervisor.START_NAME).read_bytes())
        for path in (self.lock,self.central):
            path.write_text(json.dumps({'token':'tok','label':supervisor.LEASE_LABEL}),encoding='utf-8')
        safety.check_runtime_ownership(start,'tok',identity=self.identity,crs_check=lambda _: {})
        with self.assertRaisesRegex(RuntimeError,'SUPERVISOR_IDENTITY_LOST'):
            safety.check_runtime_ownership(start,'tok',identity=lambda _: 42.0,crs_check=lambda _: {})
        self.central.write_text('{"token":"foreign","label":"foreign"}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'LEASE_TOKEN_CHANGED'):
            safety.check_runtime_ownership(start,'tok',identity=self.identity,crs_check=lambda _: {})

    def test_outputs_refuse_source_codex_other_checkout(self):
        safety.validate_runtime_root(self.runtime,self.repo)
        for path in (self.here/'output',Path('C:/Users/clclc/.codex/output')):
            with self.assertRaises(ValueError): safety.validate_runtime_root(path,self.repo)
        other=self.base/'other_checkout'
        other.mkdir()
        (other/'.git').write_text('gitdir: unused',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'INSIDE_GIT_CHECKOUT'): safety.validate_runtime_root(other/'output',self.repo)

    def test_crs_live_stale_unknown_missing_fail_closed(self):
        path=self.base/'state.sqlite3'
        with self.assertRaisesRegex(RuntimeError,'STATE_UNAVAILABLE'): safety.check_crs_idle(path)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE leases(kind TEXT,status TEXT)')
            db.execute('CREATE TABLE jobs(key TEXT,state TEXT)')
            db.execute("INSERT INTO leases VALUES('model','RECONCILED')")
        self.assertEqual(safety.check_crs_idle(path)['active_jobs'],0)
        for status in ('LIVE','RECONCILIATION_REQUIRED','UNKNOWN'):
            with closing(sqlite3.connect(path)) as db, db: db.execute('UPDATE leases SET status=?',(status,))
            with self.assertRaisesRegex(RuntimeError,'OWNER_ACTIVE_OR_UNRECONCILED'): safety.check_crs_idle(path)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE leases SET status='RECONCILED'")
            db.execute("INSERT INTO jobs VALUES('fixture','OUTCOME_UNKNOWN')")
        with self.assertRaisesRegex(RuntimeError,'OWNER_ACTIVE_OR_UNRECONCILED'): safety.check_crs_idle(path)


class LauncherTests(unittest.TestCase):
    def test_command_hides_window_and_binds_runtime_attempt(self):
        command=launcher.build_command('C:/py dir/python.exe',Path('C:/a b/F.json'),'ab'*32,
            'grant "quoted" & piped | text',here=Path('C:/x'),runtime_root=FIXTURE_ROOT,attempt_number=2)
        self.assertIn('\\"quoted\\"',command)
        self.assertIn('--attempt-number 2',command)
        self.assertIn('-B -u',command)
        self.assertIn('ShowWindow=[uint16]0',launcher.POWERSHELL)
        def run(_args,**kwargs):
            self.assertEqual(kwargs['env']['PARENTQ_WMI_COMMAND'],command)
            return types.SimpleNamespace(returncode=0,stdout='{"ProcessId":4321,"ReturnValue":0}',stderr='')
        self.assertEqual(launcher.launch(command,Path('C:/x'),run=run)['ProcessId'],4321)


if __name__=='__main__': unittest.main()
