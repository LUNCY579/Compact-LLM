"""Authored causal trace controls; no original INFQ payload reads."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys,time,unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import infq_projection as P

def fixture():
    rows=[];controls=[];stages={};events=[];expected=[]
    for i,(length,generated) in enumerate(zip((419,186,419,186),(10,6,15,8)),1):
        row={'row_id':P.digest(['AUTHORED',i]),'prompt_sha256':P.digest(['TEXT',i]),'prompt_ids':[100+i]*length,
            'prompt_tokens':length,'attention_mask':[1]*length};rows.append(row)
    config={'rows':rows,'source_packet_sha256':P.digest(rows)}
    for operation in ('zero_parity_base_forward','zero_parity_adapter_forward'):expected.append((operation,len(rows[0]['prompt_ids'])))
    for i,(row,generated) in enumerate(zip(rows,(10,6,15,8)),1):
        trace=[{'token':500+t if t<generated-1 else 151645,'logprob':-.25,'prefix_length':row['prompt_tokens']+t} for t in range(generated)]
        control={'row_id':row['row_id'],'source_packet_sha256':config['source_packet_sha256'],
            'prompt_ids_sha256':P.digest(row['prompt_ids']),'traces':[deepcopy(trace),deepcopy(trace)],'repeat_exact':True};controls.append(control)
        for repeat in (0,1):
            for step,value in enumerate(trace):
                stages['stage_inference_q%04d_r%d_t%02d.json'%(i,repeat,step)]={'row_id':row['row_id'],'repeat':repeat,'step':step,
                    'prompt_ids_sha256':P.digest(row['prompt_ids']),'input_row_sha256':P.digest(row),
                    'source_packet_sha256':config['source_packet_sha256'],**value}
                expected.append(('inference_qualification_forward',row['prompt_tokens']+step))
        expected.append(('inference_qualification_bulk_forward',row['prompt_tokens']+generated-1))
    operations={}
    for ordinal,(operation,positions) in enumerate(expected,1):
        identity=ordinal*3;events.extend([{'event':'attempt','id':identity,'operation':operation},{'event':'returned','id':identity,'operation':operation}])
        item=operations.setdefault(operation,{'attempted_calls':0,'input_positions':0});item['attempted_calls']+=1;item['input_positions']+=positions
    worker={'forward_dose':{'attempted_calls':84,'input_positions':28627,'operations':operations}}
    audit={'accounting':{'actual_forward_calls':84,'actual_input_positions':28627}}
    parity={'zero_adapter_output_equal':True,'forward_dose':{'attempted_calls':2}}
    return config,controls,events,stages,parity,worker,audit

class ProjectionControls(unittest.TestCase):
    def test_actual_original_recipe84_and28627_totals_from_authored_traces(self):
        value=P.reconstruct(*fixture())
        self.assertEqual(84,value['total_calls']);self.assertEqual(28627,value['total_input_positions'])
        self.assertEqual(78,value['operation_counts']['inference_qualification_forward'])
        self.assertEqual(26544,value['operation_input_positions']['inference_qualification_forward'])
        self.assertEqual(1245,value['operation_input_positions']['inference_qualification_bulk_forward'])
        self.assertEqual(838,sum(v for k,v in value['operation_input_positions'].items() if k.startswith('zero_parity')))
        self.assertFalse(value['input_values_are_independently_recorded_per_call']);self.assertFalse(value['global_exposure_coverage'])
    def test_missing_reordered_or_duplicate_journal_calls_refuse(self):
        for change in (lambda events:events.pop(),lambda events:events.reverse(),lambda events:events.append(deepcopy(events[0]))):
            data=list(fixture());change(data[2])
            with self.subTest(change=change),self.assertRaises(ValueError):P.reconstruct(*data)
    def test_wrong_prefix_token_stage_or_repeat_mismatch_refuse(self):
        for what in ('prefix','stage','repeat'):
            data=list(fixture())
            if what=='prefix':data[1][0]['traces'][0][0]['prefix_length']+=1
            elif what=='stage':data[3]['stage_inference_q0001_r0_t00.json']['token']+=1
            else:data[1][0]['traces'][1][0]['token']+=1
            with self.subTest(what=what),self.assertRaises(ValueError):P.reconstruct(*data)
    def test_changed_prompt_dose_and_nonfinite_JSON_refuse(self):
        data=list(fixture());data[0]['rows'][0]['prompt_ids'][0]=999
        with self.assertRaises(ValueError):P.reconstruct(*data)
        data=list(fixture());data[5]['forward_dose']['input_positions']-=1
        with self.assertRaises(ValueError):P.reconstruct(*data)
        for raw in (b'{"x":NaN}',b'{"x":1e999}',b'{"x":0,"x":1}'):
            with self.assertRaises(ValueError):P.parse(raw)
    def test_candidate_identity_projection_joins640and20_without_prompt_or_target_emission(self):
        candidates=[];enrollment={'parents':['AUTHORED_PARENT_%02d'%i for i in range(20)],'record_ids':{'M1':[],'M2':[]}}
        for arm in ('M1','M2'):
            for i in range(320):
                rid=P.digest([arm,i]);enrollment['record_ids'][arm].append(rid)
                candidates.append({'row_id':rid,'parent_id':enrollment['parents'][i//16],'component_id':'COMPONENT_%d'%(i//16),
                    'source_hash':P.digest(['SOURCE',i//16]),'arm':arm,'skill':'R' if i%2 else 'S','split':'train',
                    'messages':'AUTHORED_IGNORED_PROMPT','target':'AUTHORED_IGNORED_TARGET','spec':'AUTHORED_IGNORED_SPEC'})
        raw=''.join(P.canonical(row)+'\n' for row in candidates).encode()
        projected=P.parent_membership(raw,enrollment)
        self.assertEqual(640,len(projected));self.assertEqual({'row_id','parent_id','component_id','source_hash','arm','skill','split'},set(next(iter(projected.values()))))
        self.assertNotIn('AUTHORED_IGNORED_TARGET',P.canonical(projected))
        bad=deepcopy(enrollment);bad['record_ids']['M1'].pop()
        with self.assertRaises(ValueError):P.parent_membership(raw,bad)
        with self.assertRaises(ValueError):P.parent_membership(raw.splitlines(keepends=True)[0]+raw,enrollment)

if __name__=='__main__':
    sources=('infq_projection.py','test_infq_projection.py','READ_SCOPE002.json')
    before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in sources};start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectionControls))
    receipt={'kind':'INFQ_SOURCE_RECIPE_PROJECTION_AUTHORED_CONTROLS001','tests':result.testsRun,'success':result.wasSuccessful(),
        'errors':len(result.errors),'failures':len(result.failures),'seconds':time.monotonic()-start,'source_before':before,
        'source_after':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in sources},
        'original_INFQ_payloads_read':False,'model_tokenizer_GPU_API_or_scoring_performed':False,'global_exposure_coverage':False}
    receipt['sha256']=P.digest(receipt);(HERE/'CPU_CONTROLS001.json').write_bytes((P.canonical(receipt)+'\n').encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
