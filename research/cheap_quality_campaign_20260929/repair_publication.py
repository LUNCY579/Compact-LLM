"""Preserve first publication and expose the measured finalization cost link."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json

OUT=Path(__file__).resolve().parent/'runs/final-001'
report=OUT/'FINAL_REPORT.html'
before=report.read_bytes()
with (OUT/'FINAL_REPORT_PUBLICATION1.html').open('xb') as f:f.write(before)
text=before.decode('utf-8')
note='<p>Final synthesis verification adds 1.375 supervised seconds, for 47.329 seconds across 19 admitted process invocations including final synthesis. See <a href="FINALIZATION_COSTS.json">finalization cost receipt</a>. Operator/agent/tool overhead and the two separately reported Codex-assisted reviews remain outside this supervisor total.</p>'
assert 'finalization cost receipt' not in text
text=text.replace('</html>',note+'</html>')
report.write_text(text,encoding='utf-8')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')
parser=Links();parser.feed(text)
assert len(parser.links)==12
assert all((OUT/link).resolve().is_file() for link in parser.links)
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
       for p in OUT.iterdir() if p.is_file()}
manifest={'status':'PASS_FINAL_PUBLISHED_ARTIFACTS','supersedes':'FINAL_PUBLISHED_MANIFEST.json initial publication snapshot',
    'links_checked':len(parser.links),'files':files,'scientific_outputs_changed':False,
    'publication_correction':'Insert finalization supplement before closing html tag; prior report versions preserved.'}
with (OUT/'FINAL_PUBLISHED_MANIFEST_CURRENT.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print(json.dumps({'status':manifest['status'],'links_checked':12,'finalization_receipt_linked':True},indent=2))
