"""One selected primary-source Microsoft documentary pass, no tool launch."""
import hashlib,json,time,urllib.request,urllib.parse
from pathlib import Path
START=time.monotonic()
OUT=Path(__file__).resolve().parent
URLS=[('windows-apisets','https://learn.microsoft.com/en-us/windows/win32/apiindex/windows-apisets'),
      ('dll-search-order','https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order'),
      ('WaitForSingleObjectEx','https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobjectex')]
receipt={'status':'IN_PROGRESS','responses':0,'bytes':0,'events':[],'whole_stage_monotonic_start':START,
         'scope_sha256':'7baa3d54ccca624d3638c641f36d246440357e061bc8036da0a5f6dec41c5e7a',
         'selected_contract':'api-ms-win-core-synch-l1-2-0.dll; no physical system graph expansion',
         'UV_or_target_execution':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,url):
        if urllib.parse.urlparse(url).hostname!='learn.microsoft.com':raise RuntimeError('non-Microsoft redirect')
        return super().redirect_request(req,fp,code,msg,headers,url)
try:
    for name,url in URLS:
        if time.monotonic()-START>=120 or receipt['responses']>=6:raise RuntimeError('documentary cap')
        event={'url':url,'attempt':1};receipt['events'].append(event);began=time.monotonic()
        with urllib.request.build_opener(Redirect()).open(url,timeout=min(30,120-(time.monotonic()-START))) as response:
            receipt['responses']+=1
            data=response.read(2097153-receipt['bytes']);receipt['bytes']+=len(data)
            if receipt['bytes']>2097152:raise RuntimeError('documentary bytes cap')
            destination=OUT/(name+'.html')
            with destination.open('xb') as f:f.write(data)
            event.update(status=response.status,final_url=response.url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),file=destination.name)
        event['elapsed_seconds']=time.monotonic()-began
    receipt['status']='SELECTED_MICROSOFT_DOCUMENTS_RECEIVED_NOT_LOCAL_RUNTIME_PROOF'
except Exception as error:
    receipt['status']='DOCUMENTARY_UNKNOWN_STOP_NO_RETRY';receipt['error']=type(error).__name__+': '+str(error)
finally:
    receipt['whole_stage_elapsed_before_receipt_seconds']=time.monotonic()-START
    receipt['whole_stage_monotonic_after_source_reads']=time.monotonic()
    with (OUT/'DOCUMENTARY_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
    files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.iterdir()) if p.is_file()]
    freeze={'status':'DOCUMENTARY_FROZEN','files':files,'whole_stage_elapsed_through_receipt_hashing_seconds':time.monotonic()-START,
            'final_freeze_write_not_in_this_internal_timer':True,'clock_start':START}
    with (OUT/'DOCUMENTARY_FREEZE.json').open('x',encoding='utf-8') as f:json.dump(freeze,f,indent=2);f.write('\n')
    print(json.dumps({'status':receipt['status'],'responses':receipt['responses'],'bytes':receipt['bytes'],
                      'whole_stage_elapsed_through_freeze_seconds':time.monotonic()-START}))
