"""Owned-tree failure accounting and prelaunch refusal without a model process."""
import json
from pathlib import Path
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import probe_supervisor as parent


class Child:
    pid=987654
    returncode=0
    alive=False
    killed=False
    def poll(self):return None if self.alive else self.returncode
    def wait(self,timeout=None):self.alive=False;return self.returncode
    def kill(self):self.killed=True;self.alive=False;self.returncode=-9


def fixture_parent(monkeypatch,tmp_path,mode):
    campaign=tmp_path/'campaign';package=campaign/'v3';package.mkdir(parents=True)
    review=package/'review.json';review.write_text('{}',encoding='utf-8')
    admission_path=package/'admission.json';admission_path.write_text('{}',encoding='utf-8')
    admission={'study_id':'V3-PASSIVE-PROBE-001','lane':'GPU_FP32_PASSIVE_INPUT_FEATURE_PROBE',
        'head':'fixture','dispatch_hashes':{},'review_receipt':'v3/review.json','gpu_uuid':'u'}
    inventory={'uuid':'u','free_bytes':8*parent.GIB}
    monkeypatch.setattr(parent,'CAMPAIGN',campaign);monkeypatch.setattr(parent,'PACKAGE',package)
    monkeypatch.setattr(parent,'ROOT',tmp_path)
    monkeypatch.setattr(parent,'check_parent',lambda *a:(admission,inventory))
    child=Child();actions=[]
    class Job:
        def assign(self,c):
            actions.append('assign')
            if mode=='assignment_failure':raise RuntimeError('expected assignment refusal')
        def resume_suspended(self,c):actions.append('resume')
        def terminate(self):actions.append('terminate');child.alive=False;child.returncode=-1
        def close(self):actions.append('close')
    monkeypatch.setattr(parent,'WindowsJob',Job)
    def launch(command,**kwargs):
        assert command[3]=='experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.probe_entry'
        assert kwargs['creationflags']&0x4
        assert kwargs['env']['HF_HUB_OFFLINE']=='1'
        actions.append('popen')
        if mode=='popen_failure':raise OSError('expected no-child launch refusal')
        if mode=='deadline':child.alive=True
        return child
    monkeypatch.setattr(parent.subprocess,'Popen',launch)
    if mode=='deadline':
        times=iter((0.,1.,902.))
        monkeypatch.setattr(parent.time,'perf_counter',lambda:next(times,902.))
        monkeypatch.setattr(parent,'resource_sample',lambda c:{'rss_bytes':1234,'host_available_bytes':8*parent.GIB})
        monkeypatch.setattr(parent,'gpu_inventory',lambda:inventory)
    return campaign,package,admission_path,child,actions


@pytest.mark.parametrize('mode',['popen_failure','assignment_failure','fast_exit'])
def test_missing_resource_samples_remain_null(monkeypatch,tmp_path,mode):
    campaign,package,admission,child,actions=fixture_parent(monkeypatch,tmp_path,mode)
    result=parent.run(tmp_path/'source',package/'run',admission)
    terminal=json.loads((package/'run/terminal_receipt.json').read_text(encoding='utf-8'))
    assert terminal['peak_rss_bytes'] is None
    assert not (campaign/'native_worker.lock').exists()
    events=[json.loads(x) for x in (campaign/'BUDGET_LEDGER.jsonl').read_text().splitlines()]
    if mode=='popen_failure':
        assert result==2 and terminal['native_worker_seconds']==0 and 'worker_pid' not in terminal
        assert not any(e['event']=='native_started' for e in events)
    elif mode=='assignment_failure':
        assert result==2 and child.killed and 'resume' not in actions
        assert terminal['worker_pid']==child.pid
    else:
        assert result==0 and events[0]['event']=='native_started'
        assert actions[:3]==['popen','assign','resume']


def test_parent_hard_deadline_terminates_only_owned_job(monkeypatch,tmp_path):
    campaign,package,admission,child,actions=fixture_parent(monkeypatch,tmp_path,'deadline')
    assert parent.run(tmp_path/'source',package/'run',admission)==2
    terminal=json.loads((package/'run/terminal_receipt.json').read_text(encoding='utf-8'))
    assert terminal['status']=='SUPERVISOR_SAFETY_STOP' and terminal['stop_reason']=='HARD_DEADLINE'
    assert actions.count('terminate')==1 and not child.killed
    assert terminal['peak_rss_bytes']==1234 and terminal['native_worker_seconds']==901
    assert not (campaign/'native_worker.lock').exists()


def test_prelaunch_refusal_creates_no_output_process_or_ledger(monkeypatch,tmp_path):
    invoked=[]
    monkeypatch.setattr(parent,'check_parent',lambda *a:(_ for _ in ()).throw(PermissionError('expected')))
    monkeypatch.setattr(parent.subprocess,'Popen',lambda *a,**k:invoked.append(a))
    out=tmp_path/'exclusive'
    with pytest.raises(PermissionError):parent.run(tmp_path/'source',out,tmp_path/'admission')
    assert invoked==[] and not out.exists()


@pytest.mark.parametrize('fault',['source_hash','review_hash','gpu_floor','host_floor','output_exists','budget'])
def test_actual_prelaunch_guards_refuse_before_model(monkeypatch,tmp_path,fault):
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.tests.test_parent_authority import admission as scope_admission
    campaign=tmp_path/'c';package=campaign/'v3';package.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir();(source/'asset.json').write_text('{}')
    code=tmp_path/'code.py';code.write_text('# fixture\n')
    data={**scope_admission(),'source':str(source),'interpreter':parent.sys.executable,
        'dispatch_hashes':{'code.py':parent.sha(code)},'checkpoint_files':{'asset.json':parent.sha(source/'asset.json')},
        'gpu_uuid':'u','gpu_driver_version':'d','review_receipt':'v3/review.json'}
    if fault=='source_hash':data['dispatch_hashes']['code.py']='0'*64
    path=package/'admission.json';path.write_text(json.dumps(data),encoding='utf-8')
    review={'status':'PASS_PASSIVE_PROBE_FP32_DISPATCH','admission_sha256':parent.sha(path)}
    if fault=='review_hash':review['admission_sha256']='0'*64
    (package/'review.json').write_text(json.dumps(review))
    (campaign/'BUDGET_LEDGER.jsonl').write_text(json.dumps({'event':'native_completed','native_worker_seconds':20000 if fault=='budget' else 0})+'\n')
    monkeypatch.setattr(parent,'CAMPAIGN',campaign);monkeypatch.setattr(parent,'PACKAGE',package);monkeypatch.setattr(parent,'ROOT',tmp_path)
    # Scope itself is exercised against actual capsules by the authority suite.
    monkeypatch.setattr(parent,'validate_probe_unused',lambda d:None)
    monkeypatch.setattr(parent,'gpu_inventory',lambda:{'uuid':'u','driver_version':'d','free_bytes':5*parent.GIB if fault=='gpu_floor' else 8*parent.GIB})
    monkeypatch.setattr(parent.psutil,'virtual_memory',lambda:type('Memory',(),{'available':6*parent.GIB if fault=='host_floor' else 8*parent.GIB})())
    monkeypatch.setattr(parent.shutil,'disk_usage',lambda p:type('Disk',(),{'free':20*parent.GIB})())
    out=package/'out'
    if fault=='output_exists':out.mkdir()
    with pytest.raises((PermissionError,RuntimeError,MemoryError)):
        parent.check_parent(source,out,path)
    assert not (campaign/'native_worker.lock').exists()
