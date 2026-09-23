import json,sys
sys.dont_write_bytecode=True
AB='C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/'
raw=open(AB+'data/monitor_probe.jsonl','rb').read()
print('CRLF count',raw.count(b'\r\n'),'LF',raw.count(b'\n'))
rows=[json.loads(l) for l in raw.decode('utf-8').splitlines() if l.strip()]
print(len(rows)); print(json.dumps(rows[0],indent=1)[:1500])
a=json.load(open(AB+'evidence/retention-design-v1/PROBE_LABEL_AUDIT.json',encoding='utf-8'))
print([k for k in a if k!='rows'])
print(json.dumps(a['rows'][0],indent=1)[:2000])
f=json.load(open(AB+'evidence/retention-design-v1/ARM_DOSE_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json',encoding='utf-8'))
pr=f['checkpoints'][-1]['retention']['per_row']; k=next(iter(pr)); print(k,pr[k])
print(json.dumps(f['checkpoints'][-1]['retention']['permissive']['per_class']['time_duration'])[:800])
s=json.load(open(AB+'outputs/retention-a329eb7b/B28/retention.json',encoding='utf-8'))
print(json.dumps(s['rows'][0],indent=1))
