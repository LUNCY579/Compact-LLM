"""Synthetic clock/resource controls; no framework/model imports."""
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import post_import_admission as a

class Clock:
    def __init__(self):self.value=0.;self.sleeps=[]
    def now(self):return self.value
    def pause(self,seconds):self.sleeps.append(seconds);self.value+=seconds

def sample(host=3.,rss=.7,gpu=6.,disk=10.):
    return dict(host_available_gib=host,rss_gib=rss,gpu_free_gib=gpu,disk_free_gib=disk)

class AdmissionControls(unittest.TestCase):
    def run_wait(self,values):
        clock=Clock();records=[];index=[0]
        def read():
            value=values[min(index[0],len(values)-1)];index[0]+=1;return value.copy()
        result=a.wait_for_admission(read,clock=clock.now,pause=clock.pause,record=records.append)
        return result,records,clock
    def test_dynamic_growth_budget(self):
        measured=a.admission_measurement(sample(host=2.58,rss=.6682243347167969))
        self.assertAlmostEqual(measured['required_host_gib'],2.581775665283203)
        self.assertFalse(measured['admitted'])
    def test_two_current_consecutive_samples_required(self):
        result,records,clock=self.run_wait([sample(host=2.),sample(),sample(host=2.),sample(),sample()])
        self.assertEqual(result['status'],'ADMITTED_POST_IMPORT_RESOURCE_ONLY')
        self.assertEqual(result['observations'],5);self.assertEqual(len(records),5)
        self.assertEqual(clock.sleeps,[2.]*4)
    def test_timeout_does_not_pass_or_invoke_model(self):
        result,records,clock=self.run_wait([sample(host=2.)])
        self.assertEqual(result['status'],'TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL')
        self.assertEqual(result['elapsed_seconds'],60.)
        self.assertEqual(result['observations'],30);self.assertEqual(result['model_calls'],0)
    def test_gpu_admission_is_preserved_while_waiting(self):
        result,_,_=self.run_wait([sample(gpu=3.),sample(),sample()])
        self.assertEqual(result['observations'],3)
    def test_runtime_floors_abort_immediately(self):
        for value in [sample(host=1.24),sample(rss=2.01),sample(gpu=.49),sample(disk=3.99)]:
            with self.assertRaises(ValueError):self.run_wait([value])
    def test_type_and_nonfinite_values_rejected(self):
        for name in ('rss_gib','host_available_gib','gpu_free_gib','disk_free_gib'):
            for value in (True,float('nan'),float('inf'),-1.):
                row=sample();row[name]=value
                with self.assertRaises(ValueError):a.admission_measurement(row)
    def test_drifted_policy_rejected(self):
        for key,value in [('seconds',61),('poll_seconds',1.),('consecutive_samples',1)]:
            policy=a.POLICY.copy();policy[key]=value;clock=Clock()
            with self.assertRaises(ValueError):a.wait_for_admission(lambda:sample(),clock=clock.now,pause=clock.pause,record=lambda x:None,policy=policy)
    def test_sample_return_recorded_before_guard_failure(self):
        records=[];clock=Clock()
        with self.assertRaises(ValueError):a.wait_for_admission(lambda:sample(host=1.),clock=clock.now,pause=clock.pause,record=records.append)
        self.assertEqual(len(records),1);self.assertEqual(records[0]['sample']['host_available_gib'],1.)
    def test_slow_sample_cannot_pass_after_deadline(self):
        clock=Clock();records=[]
        def read():clock.value+=61;return sample()
        result=a.wait_for_admission(read,clock=clock.now,pause=clock.pause,record=records.append)
        self.assertEqual(result['status'],'TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL')
        self.assertEqual(result['observations'],1);self.assertEqual(len(records),1)
    def test_slow_journal_cannot_admit_past_deadline(self):
        clock=Clock();records=[]
        def record(value):
            records.append(value)
            if len(records)==2:clock.value+=59
        result=a.wait_for_admission(lambda:sample(),clock=clock.now,pause=clock.pause,record=record)
        self.assertEqual(result['status'],'TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL')
        self.assertEqual(result['elapsed_seconds'],61.)
        self.assertEqual(result['model_calls'],0)
    def test_final_clock_cannot_admit_past_deadline(self):
        values=iter([0.,0.,0.,0.,2.,2.,2.,61.])
        result=a.wait_for_admission(lambda:sample(),clock=lambda:next(values),pause=lambda seconds:None,record=lambda value:None)
        self.assertEqual(result['status'],'TIMED_OUT_POST_IMPORT_ADMISSION_NO_MODEL')
        self.assertEqual(result['elapsed_seconds'],61.)
    def test_frozen_helper_never_grants_qualification(self):
        result,_,_=self.run_wait([sample(),sample()])
        self.assertIs(result['native_qualification'],False);self.assertIs(result['E1_authorized'],False)
    def test_journal_failure_prevents_admission_return(self):
        clock=Clock()
        def record(value):raise OSError('receipt disk failure')
        with self.assertRaises(OSError):a.wait_for_admission(lambda:sample(),clock=clock.now,pause=clock.pause,record=record)
    def test_clock_regression_refused(self):
        values=iter([10.,9.])
        with self.assertRaises(ValueError):a.wait_for_admission(lambda:sample(),clock=lambda:next(values),pause=lambda x:None,record=lambda x:None)

if __name__=='__main__':unittest.main()
