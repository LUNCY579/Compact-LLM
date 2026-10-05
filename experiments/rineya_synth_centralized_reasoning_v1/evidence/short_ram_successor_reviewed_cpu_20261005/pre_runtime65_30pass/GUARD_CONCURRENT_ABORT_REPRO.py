from pathlib import Path
import sys, threading, json
stage=Path.cwd()/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/short_ram_successor_work'
sys.path.insert(0,str(stage))
import short_ram_monitor as mod
entered=threading.Event(); release=threading.Event(); events=[]
class Job:
    def close(self): events.append('job_close')
class Worker:
    job=Job()
    def terminate(self):
        events.append('terminate_enter'); entered.set(); release.wait(2); events.append('terminate_return')
admitted={'sample_started_monotonic':9.9,'sample_completed_monotonic':9.91,'host_available_bytes':2*mod.shared.GIB,'disk_free_bytes':4*mod.shared.GIB,'worker_tree_rss_bytes':mod.shared.GIB,'gpu_sample_started_monotonic':9.9,'gpu_sample_completed_monotonic':9.91,'gpu_free_bytes':mod.shared.GIB,'gpu_total_bytes':8*mod.shared.GIB,'GPU_index':0,'gpu_observation_id':'FAKE','gpu_uuid':'FAKE'}
monitor=mod.ShortRamMonitor(Worker(),stage,600,admitted,clock=lambda:10.,phase_clock=lambda:10.,persist=lambda path,value:events.append('diagnostic_write'))
t=threading.Thread(target=lambda:monitor._abort('stale_external_CPU_watchdog'))
t.start(); assert entered.wait(2)
monitor._abort('stale_external_CPU_watchdog')
write_before_terminate_return='diagnostic_write' in events and 'terminate_return' not in events
release.set(); t.join(2)
print(json.dumps({'events':events,'diagnostic_before_containment_return':write_before_terminate_return,'thread_closed':not t.is_alive(),'reason':monitor.reason}))
