"""One finite offline source correction; no candidate/native imports."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
p=ROOT/'harness.py';s=p.read_text();a=s.index('    def close_stream(self,index):');b=s.index('\n    def reader(self,index,fd,name):',a)
s=s[:a]+'''    def close_stream(self,index):
        stream=self.streams[index]
        if stream is None:return
        try:
            a.require(not self.close_attempted[index],'each original log stream close attempted at most once')
            # Admission and both guards belong to the SAME bounded secondary
            # diagnostic operation, not only the native stream.close call.
            def close():
                identity=self.log_identities[index]
                a.require(identity is not None,'initialized original log status before close')
                self.close_attempted[index]=True;identity.close_attempted=True
                try:return stream.close()
                except BaseException:identity.close_failed=True;raise
            self.perform(index,5,close,closing=True,returned=lambda value:self.retain_close(index,value))
        except BaseException as close_error:
            # One fixed diagnostic per reader. perform never runs its return
            # guard after an action raises, so actual-close/afterguard double
            # failure is not a reachable source path. Preserve first evidence.
            if self.secondary_close_errors[index] is None:
                name=type(close_error).__name__
                self.secondary_close_errors[index]=name if name.isascii() and len(name)<=128 else 'UNBOUNDED_CLOSE_ERROR_TYPE'
            cost.traceback.clear_frames(close_error.__traceback__)
            raise
''' +s[b:]
s=s.replace('len(block)<=1792','len(block)<=1536').replace('os.read(fd,1792)','os.read(fd,1536)')
s=s.replace('Two <=1792-byte payload slots +4608 logical state','Two <=1536-byte payload slots +5120 logical state')
p.write_text(s,newline='\n')
p=ROOT/'admission.py';s=p.read_text();a=s.index('JOB_CENSUS=');b=s.index('\n',a)
raw=(ROOT/'reusable_census.py').read_bytes();ref={'path':str(ROOT/'reusable_census.py'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
p.write_text(s[:a]+'JOB_CENSUS='+repr(ref)+s[b:],newline='\n')
for p in ROOT.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('Offline complete close-ledger syntax passed.')
