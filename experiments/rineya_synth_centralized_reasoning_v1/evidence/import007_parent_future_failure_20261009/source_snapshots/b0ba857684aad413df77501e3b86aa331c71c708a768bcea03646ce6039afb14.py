"""Separate finite import-effect measurement; never an exact-effect permit."""
from pathlib import Path
from types import ModuleType
from contextlib import contextmanager
import hashlib

PACKAGE=next(p for p in Path(__file__).resolve().parents if p.name=='rineya_synth_centralized_reasoning_v1')
ORIGINAL_SOURCE=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/history/WORKER_LIBRARY_PUBLICATION_SOURCE_FROZEN_017/source/prefix_library_policy.py'
ORIGINAL_PIN='2f8d5c30d85c066413accab77bfe80c2eb20cc184772cd44ec097acb8775b1c1'
raw=ORIGINAL_SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest()!=ORIGINAL_PIN:raise ValueError('exact original measurement helper source')
_O=ModuleType('measurement_original_library_policy');_O.__file__=str(ORIGINAL_SOURCE);exec(compile(raw,str(ORIGINAL_SOURCE),'exec'),_O.__dict__);_O.__compiled_source_descriptor__={'bytes':len(raw),'sha256':ORIGINAL_PIN,'role':'python_source'}
MEASUREMENT_SOURCE=Path(__file__).resolve();MEASUREMENT_DESCRIPTOR={'bytes':MEASUREMENT_SOURCE.stat().st_size,'sha256':hashlib.sha256(MEASUREMENT_SOURCE.read_bytes()).hexdigest(),'role':'python_source'}
KIND='TORCH_IMPORT_EFFECT_MEASUREMENT_RAW_V1'
SCOPE='OWNED_TORCH_IMPORT_LIBRARY_EFFECT_OBSERVATION_ONLY'
require=_O._require

def validate_grammar(grammar,policy):
    require(type(grammar) is dict and set(grammar)=={'schema_version','kind','required_extensions','optional_extensions','maximum_boundaries','maximum_depth','maximum_observations','model'},'exact finite measurement grammar')
    require(type(grammar['schema_version']) is int and grammar['schema_version']==1 and grammar['kind']=='FINITE_IMPORT_EFFECT_MEASUREMENT_GRAMMAR_V1' and grammar['model'] is False,'separate measurement grammar scope')
    required,optional=grammar['required_extensions'],grammar['optional_extensions']
    require(type(required) is list and required and type(optional) is list and all(type(p) is str for p in required+optional) and len(required)==len(set(required)) and len(optional)==len(set(optional)) and not set(required)&set(optional) and set(required)|set(optional)==set(policy['extension_transitions']),'explicit nonempty required and finite optional extension paths')
    require(type(grammar['maximum_boundaries']) is int and grammar['maximum_boundaries']==1+2*len(required+optional)<=2048 and type(grammar['maximum_depth']) is int and grammar['maximum_depth']==64 and type(grammar['maximum_observations']) is int and grammar['maximum_observations']==4096,'original topology/depth/observation caps')
    return grammar

def validate_measurement_policy(policy):
    require(type(policy) is dict and policy.get('framework_transition') is None and type(policy.get('extension_transitions')) is dict and all(type(row) is dict and set(row)=={'create','exec'} and row['create'] is row['exec'] is None for row in policy['extension_transitions'].values()),'unknown measurement effects remain explicit None')
    # Validate identity/directory/source schema only. No empty-list projection
    # is ever installed as a guard expectation or exported as observed facts.
    shape=_O._raw_copy(policy);shape['framework_transition']=[];shape['extension_transitions']={p:{'create':[],'exec':[]} for p in policy['extension_transitions']};_O.validate_policy(shape)
    return policy

class MeasurementGuard(_O.LibraryGuard):
    """Original source/scope/snapshot/event/cleanup checks, measured effects.

    A provider is supplied by a separately reviewed source-owned stage. This
    class has no native provider factory, permission factory or launch entry.
    """
    def __init__(self,policy,grammar,contract,provider,budget=None,*,clock_ns=None):
        self.grammar=contract.parse(contract.canonical(validate_grammar(grammar,policy)))
        self.effects=[];self._entered_ops=set();self._created=set();self._executed=set();self._framework=False;self._framework_returned=False
        validate_measurement_policy(policy)
        self.policy=contract.parse(contract.canonical(policy));self.contract=contract;self.provider=provider;self.budget=budget;self.fixture_only=provider.fixture_only;self.native_PASS=False;self.native_authority_issued=False;self.active=False;self.failed=False;self.closed=False;self.last=None;self.observed=[];self._depth=0;self._frames=[];self._directory_identities=None;self._event_log=[];self._event_state=None;self._raw=_O._RawObservations(self.policy,self.fixture_only,budget,clock_ns);self._boundary_counter=0;began=False
        try:
            self._raw.begin('initial');self._source_pin()
            for directory in self.policy['search_directories']:_O._physical(directory,directory=True)
            self._directories();began=True;provider.begin(tuple(self.policy['search_directories']),self.policy['system_directory']);self.active=True;self._scope();current=self._snapshot();require(set(current)==set(self.policy['initial_loaded_libraries']),'exact selected measurement initial DLL set');self.last=current;self._event_state=dict(current);self._events(allowed_new=set());self._raw.finish()
        except BaseException as failure:
            self.failed=True;self._raw.finish(failure)
            if began:self._cleanup_after_failure()
            raise
    def _source_pin(self):
        self._raw.component='source';source=self.policy['provider_source'];require(source['path']==str(ORIGINAL_SOURCE) and source['descriptor']==_O.__compiled_source_descriptor__,'exact original provider helper source selected for measurement');self.contract.file_descriptor(source['path'],source['descriptor'],budget=self.budget);self.contract.file_descriptor(str(MEASUREMENT_SOURCE),MEASUREMENT_DESCRIPTOR,budget=self.budget)
    def _check_active(self):
        require(self.active and not self.failed and not self.closed,'active nonfailed measurement boundary')
        self._source_pin();self._scope();self._directories();observed=self._snapshot()
        require(all(observed.get(p)==h for p,h in self.last.items()),'existing path/handle conservation before nested measurement')
        appeared=set(observed)-set(self.last)
        require(not appeared or self._frames,'no load outside active measurement boundary')
        self._events(allowed_new=appeared);require(self._event_state==observed,'measurement event/snapshot join before delegation');self.last=observed
        return observed
    def _enter_name(self,name):
        require(self._boundary_counter<self.grammar['maximum_boundaries'] and len(self._frames)<64,'bounded measurement entries/depth')
        if name=='framework_import':require(not self._framework and not self._frames and self._boundary_counter==0,'one first measurement framework root');self._framework=True
        else:
            require(self._framework and self._frames and type(name) is str and name.startswith('extension:'),'extension only in active framework grammar')
            p,sep,operation=name[len('extension:'):].rpartition(':');require(sep and p in self.policy['extension_transitions'] and operation in ('create','exec') and (p,operation) not in self._entered_ops,'finite single create/exec path')
            require(operation!='exec' or p in self._created,'exec after returned create');self._entered_ops.add((p,operation))
    @contextmanager
    def _boundary(self,name,expected_new=None):
        before=None;entered=False;boundary=None
        try:
            self._enter_name(name);self._boundary_counter+=1;boundary={'id':self._boundary_counter,'parent_id':None if not self._frames else self._frames[-1]['id'],'name':name}
            self._raw.begin('enter',boundary);before=self._check_active();self._depth+=1;self._frames.append({**boundary,'before':dict(before),'nested':set()});entered=True;self._raw.finish()
            yield
            self._raw.begin('exit',boundary);require(not self.failed,'poisoned measurement cannot return completion');self._source_pin();self._scope();self._directories();after=self._snapshot()
            require(all(after.get(p)==h for p,h in self.last.items()),'nested measured handle conservation');appeared=set(after)-set(self.last);self._events(allowed_new=appeared);require(self._event_state==after,'measurement exit event/snapshot join')
            frame=self._frames[-1];total=set(after)-set(before);own=total-frame['nested'];self.effects.append({'boundary':dict(boundary),'own_new_libraries':sorted(own),'nested_new_libraries':sorted(frame['nested']),'total_new_libraries':sorted(total)})
            self.last=after;self.observed.append({'boundary':name,'new_libraries':sorted(total),'fixture_only':self.fixture_only,'actual_loaded_identity_observed':True,'native_qualification_proven':False});self._raw.finish()
            if name=='framework_import':self._framework_returned=True
            else:
                p,_,operation=name[len('extension:'):].rpartition(':');(self._created if operation=='create' else self._executed).add(p)
        except BaseException as failure:
            self.failed=True
            if self._raw.current is not None:self._raw.finish(failure)
            if before is not None and boundary is not None:
                try:
                    self._raw.begin('boundary_error',boundary);self._raw.error(failure,'delegate_or_measurement');self._source_pin();self._scope();self._directories();after=self._snapshot();require(all(after.get(p)==h for p,h in self.last.items()),'failed measurement handle conservation');self._events(allowed_new=set(after)-set(self.last));require(self._event_state==after,'failed measurement event/snapshot join');self.last=after;self._raw.finish()
                except BaseException as observation_failure:self._raw.finish(observation_failure)
            raise
        finally:
            if entered:
                frame=self._frames.pop();self._depth-=1
                if self._frames and self.last is not None:self._frames[-1]['nested'].update(set(self.last)-set(frame['before']))
    def extension(self,path,operation):return self._boundary('extension:'+path+':'+operation)
    def framework_import(self):return self._boundary('framework_import')
    def close(self):
        if not self.failed:
            try:require(self._framework_returned and self._created==self._executed and set(self.grammar['required_extensions'])<=self._created,'required measured create/exec pairs returned')
            except BaseException as failure:self.failed=True;self._raw.error(failure,'required_measurement_pairs')
        original=super().close()
        return {'kind':'MEASUREMENT_RECORDED_UNQUALIFIED','fixture_only':self.fixture_only,'native_PASS':False,'native_authority_issued':False,'effects':list(self.effects),'original_cleanup':original}
    def export_raw_packet(self,identity=None):
        value=super().export_raw_packet(identity);value.pop('sha256');value.update(kind=KIND,measurement_scope=SCOPE,measurement_grammar=self.grammar,measured_boundary_effects=list(self.effects),exact_effects_predeclared=False)
        value['sha256']=self.contract.digest(value);require(len(self.contract.canonical(value).encode('ascii'))<=_O.MAX_RAW_BYTES,'bounded full measurement packet with effect metadata');return value
