"""Publish final links/cost supplement; scientific study artifacts stay unchanged."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import unquote

BASE=Path(__file__).resolve().parent
OUT=BASE/'runs/final-001'
started=time.monotonic()
report=OUT/'FINAL_REPORT.html'
before=report.read_bytes()
with (OUT/'FINAL_REPORT_PRE_PUBLICATION.html').open('xb') as f:f.write(before)
text=before.decode('utf-8').replace('September29,2026','September 29, 2026').replace('All56 latest','All 56 latest')
footer='<p>Final synthesis verification adds 1.375 supervised seconds, for 47.329 seconds across 19 admitted process invocations including final synthesis. See <a href="FINALIZATION_COSTS.json">finalization cost receipt</a>. This total excludes unmetered operator/agent/tool overhead and the two separately reported Codex-assisted reviews.</p>'
text=text.replace('</html>',footer+'</html>')
report.write_text(text,encoding='utf-8')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            self.links.extend(value for key,value in attrs if key=='href')
parser=Links();parser.feed(text)
links=[]
for link in parser.links:
    path=(OUT/unquote(link)).resolve()
    assert path.is_file(), 'Broken local report link: '+link
    links.append({'href':link,'resolved_path':str(path)})
summary=json.loads((OUT/'FINAL_SUMMARY.json').read_text())
assert summary['status']=='ALL_SIX_CPU_STUDIES_VERIFIED_COMPLETE'
assert len(summary['completion_audit'])==6 and summary['latest_scoped_tests_passed']==56
old_partial=BASE/'runs/PARTIAL_REPORT.html'
with (BASE/'runs/PARTIAL_REPORT_PRE_COMPLETION.html').open('xb') as f:f.write(old_partial.read_bytes())
old_partial.write_text('<!doctype html><meta charset="utf-8"><title>Rineya campaign completed</title><h1>All six studies completed</h1><p>The earlier partial state is preserved separately. <a href="final-001/FINAL_REPORT.html">Read the final report, verified results and full cost scope.</a></p>',encoding='utf-8')
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file()}
with (OUT/'FINAL_PUBLISHED_MANIFEST.json').open('x',encoding='utf-8') as f:
    json.dump({'status':'PASS_FINAL_LINKS_AND_ARTIFACTS','links':links,'files':files,
        'scientific_outputs_changed':False,'original_final_report_preserved':True,
        'metadata_publication_seconds_through_receipt':time.monotonic()-started,
        'initial_manifest_scope':'FINAL_ARTIFACTS.json preserves pre-publication synthesis snapshot; this file binds published report and cost supplement.'},f,indent=2)
print(json.dumps({'status':'PASS_FINAL_LINKS_AND_ARTIFACTS','links_checked':len(links),'files_hashed':len(files),'studies':6,'tests':56},indent=2))
