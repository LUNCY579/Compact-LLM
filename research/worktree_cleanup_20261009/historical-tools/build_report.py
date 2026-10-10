import datetime,html,json,re,subprocess
from pathlib import Path
from urllib.parse import quote

OUT=Path(r'C:\Users\clclc\BusinessProject\Rineya-Worktree-Audit-20261009')
ROOT=OUT.parent
data=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
cleanup=json.loads((OUT/'cleanup-receipt.json').read_text())
tags=json.loads((OUT/'remote-preservation.json').read_text())['tags']
release=json.loads((OUT/'supa-hybrid-remote-verification.json').read_text())
phase2_releases=[]
if (OUT/'phase2-cleanup-receipt.json').exists():
    extra=json.loads((OUT/'phase2-cleanup-receipt.json').read_text())
    for w in extra['worktrees']:
        cleanup['worktrees'].append({'path':w['path'],'repository':'Chat-Bubble','head':w['head'],'verified_remote_tag':w['verified_source_tag'],'logical_bytes_removed':w['logical_checkout_bytes_removed'],'scientific_artifact_release':w['verified_private_release'],'phase':2,'material_files_preserved':w['material_files_preserved']})
    cleanup['logical_bytes_removed']+=extra['logical_bytes_removed']
    tags.extend({'repository':'Chat-Bubble','tag':t['tag'],'commit':t['head'],'verified_remote_tag_object':t['remote_tag_object'],'phase':2} for t in json.loads((OUT/'phase2-verified-tags.json').read_text()))
    phase2_releases=json.loads((OUT/'phase2-remote-release-verification.json').read_text())
chats=json.loads((OUT/'local-chat-goals.json').read_text())
remote_notes=json.loads((OUT/'remote-research-status.json').read_text())['branches']
removed={str(Path(w['path']).resolve()).casefold():w for w in cleanup['worktrees']}
goals={}
for t in chats:
    if not t['archived'] and t.get('goal') and t['goal']['status']!='complete':
        cwd=t['cwd'].replace('\\\\?\\','')
        goals.setdefault(str(Path(cwd).resolve()).casefold(),[]).append({'thread_id':t['id'],'status':t['goal']['status']})

dependent={
 'ChatBubble-comp-rs-s1-dev-20260930':'CE8 production ce_qualification_adoption.py:15 hard-codes this checkout; current registrations bind its paths. Five previously unpushed commits are now tagged remotely.',
 'ChatBubble-titan-clare-sparse-v1':'WorldModel RCWM-T1 env.py:22, runtime.py:38, and p3_make_bank.py:82–86 depend on its data banks and shared lease directory. Keep until consumers are deliberately rebound.',
 'SupaExperiment-matched-mlp-shipping-v1':'M2 readiness still depends on the original M1 shipping path and guard; contains an initialized submodule.',
}
special={
 'ChatBubble-rvs001':'Ignored frozen study receipts and bank state remain locally. A paused watch references this checkout. Preserve full bank evidence before future retirement.',
 'ChatBubble-rineya-reasoning-posttraining-v1':'Source status is ACTIVE_NOT_COMPLETE; current successor lanes and source dependencies remain. Keep the open research track.',
 'ChatBubble-rineya-quality-sft-v1':'Open quality-SFT study; four previously unpushed commits are now remotely tagged. Source preservation does not complete the native study.',
 'SupaExperiment-rineya-quality-v1':'Open RINQ native study; seven previously unpushed commits are now remotely tagged. Keep the owner checkout.',
 'ChatBubble-rineya-centralized-reasoning-system-v1':'CRS mission and hourly supervisor remain current. GitHub lacks some committed history; separate reviewed private HF custody and excluded raw provenance need reconciliation before any retirement.',
}

worktrees=[]
for repo in data['repositories']:
    for w in repo.get('worktrees',[]):
        p=Path(w['worktree']); key=str(p.resolve()).casefold(); name=p.name
        row={'repository':Path(repo['path']).name,'path':str(p),'branch':w.get('branch','detached'),'head':w.get('HEAD',''),'size_bytes':w.get('total_bytes',0),'tracked_change_count':len(w.get('tracked_changes',[])),'untracked_file_count':len(w.get('untracked',[])),'ignored_file_count':len(w.get('ignored',[])),'owners':goals.get(key,[])}
        if repo.get('bare'):
            row.update(action='Keep recovery repository',reason='Small local bare recovery repository, not a populated worktree. Its DSD commit is also preserved by the ChatBubble remote.')
        elif key in removed:
            row.update(action='Removed',reason=('Verified exact working-state preservation and scientific snapshot release before retiring the dormant checkout; source tag verified. Research branches and open questions remain.' if removed[key].get('phase')==2 else 'Verified clean source or disposable-cache-only checkout and remote source tag before removal; no additional scientific archive was required. Research branches and open questions remain.'),archive_tag=removed[key]['verified_remote_tag'])
            if 'hybrid-memory' in name: row['reason']='Completed fixed negative-result study; all 145 scientific ignored files preserved in a private release, downloaded and hash-verified before removal.'
            if name=='ChatBubble-small-vocab-depth': row['reason']='Seventeen unpublished LCACC files archived; one consumer lookup moved to the byte-identical retained CED parser. Original consumer and historical receipts preserved privately and unchanged.'
        elif not w.get('exists'):
            row.update(action='Pruned stale registration',reason='Physical folder was already absent. Only stale Git worktree metadata removed; branch and history retained.')
        elif not w.get('inside_requested_directory'):
            row.update(action='Keep outside requested directory',reason='External owner checkout inspected for dependencies and source coverage. Keep its unique source, current bindings, and local evidence.')
        elif w.get('primary'):
            row.update(action='Keep primary repository',reason='Primary repository or research preservation clone. Retain source history and local data.')
        elif name in dependent:
            row.update(action='Keep path dependency',reason=dependent[name])
        elif name in special:
            row.update(action='Keep open research / evidence',reason=special[name])
        elif goals.get(key):
            row.update(action='Keep owner research',reason='A nonarchived research goal is active, blocked, paused, or usage-limited. Its work and open questions remain owned.')
        elif row['tracked_change_count'] or row['untracked_file_count']:
            row.update(action='Keep unpublished work',reason='Checkout contains unpublished source, prepared proposals, receipts, or records. Git tags do not preserve these files.')
        else:
            row.update(action='Keep local evidence / open research',reason='Ignored evidence or an ongoing research dependency remains. No verified redundant checkout established for this audit.')
        worktrees.append(row)

branches=[]
canonical_remote_branches=set()
chat_remote=next(r['final_live_remote_refs'] for r in data['repositories'] if Path(r['path']).name=='Chat-Bubble')
for repo in data['repositories']:
    name=Path(repo['path']).name
    remote_url=next((l.split()[1] for l in repo.get('remotes','').splitlines() if l.startswith('origin') and '(fetch)' in l),'')
    for ref,oid in repo['final_live_remote_refs'].items():
        if ref.startswith('refs/heads/'): canonical_remote_branches.add((remote_url,ref))
    for b in repo['branches']:
        remaining=b['final_live_uncovered_commits']
        cross=[]
        if repo.get('bare'): cross=[n for n,o in chat_remote.items() if o==b['oid']]
        checkout=next((w for w in worktrees if w['repository']==name and w['branch']==b['ref']),None)
        disposition=checkout['action'] if checkout else 'Keep ref; no populated checkout'
        if remaining and not cross: disposition='Preserve committed history before future cleanup'
        branches.append({'repository':name,'kind':'local','branch':b['ref'][11:],'head':b.get('final_oid',b['oid']),'subject':b.get('final_subject',b['subject']),'live_remote_preserved':remaining==0 or bool(cross),'uncovered_commits_in_same_remote':remaining,'exact_remote_refs':b['final_exact_remote_refs'],'cross_repository_remote_refs':cross,'disposition':disposition,'concurrent_owner_advance':b.get('concurrent_owner_advance')})
    local={b['ref'] for b in repo['branches']}
    for ref,oid in repo['final_live_remote_refs'].items():
        if ref.startswith('refs/heads/') and ref not in local:
            short=ref[11:]
            note=next((n['analysis']['state'] for n in remote_notes if n['branch']==short),'Preserved remote research/source history; no local checkout')
            branches.append({'repository':name,'kind':'remote only','branch':short,'head':oid,'subject':note,'live_remote_preserved':True,'uncovered_commits_in_same_remote':0,'exact_remote_refs':[ref],'cross_repository_remote_refs':[],'disposition':'Keep remote history; no checkout to clean'})

nonworktree=[]
for p in ROOT.iterdir():
    if not p.is_dir() or p==OUT or (p/'.git').exists(): continue
    if re.search(r'rineya|^(ce8-|dc1-|m2-|mlp-|rdd|rinq|rsd|mq-|gd001|comp-rs|clare-|titan-|dsd|profiler-|observer-|s0pre|synth-|triskill|taps|supa-run)|^WorldModel$',p.name,re.I):
        nonworktree.append({'path':str(p),'name':p.name,'classification':'Evidence, records, fixtures, dataset, or output directory; no top-level Git worktree marker. Retained.'})
(OUT/'non-worktree-research-directories.json').write_text(json.dumps(nonworktree,indent=2),encoding='utf-8')

before_live=sum(1 for w in worktrees if w['action']!='Pruned stale registration' and w['action']!='Keep recovery repository')
summary={'audit_date_client':'2026-10-09','timezone':'America/Chicago','repository_locations':len(data['repositories']),'distinct_remote_repositories':len({x[0] for x in canonical_remote_branches}),'local_branch_refs_examined':sum(len(r['branches']) for r in data['repositories']),'live_remote_branch_refs_examined':len(canonical_remote_branches),'populated_worktrees_before':before_live,'populated_worktrees_after':before_live-len(removed),'worktrees_removed':len(removed),'stale_registrations_pruned':3,'logical_bytes_removed':cleanup['logical_bytes_removed'],'remote_tags_verified':len(tags),'private_artifact_releases_verified':1+len(phase2_releases),'research_branches_deleted':0,'local_only_committed_refs_retained':[b for b in branches if b['kind']=='local' and not b['live_remote_preserved']],'nonworktree_research_directories_retained':len(nonworktree)}
report={'summary':summary,'worktrees':worktrees,'branches':branches,'tags':tags,'release':release,'phase2_releases':phase2_releases,'method':'Git worktree inventory, all local/remote heads and tags, live remote object identities, ancestry and unique commit coverage; tracked/untracked/ignored file inventories; scientific source status; read-only owner goals and automations; executable path dependency search; selected release member hashing and independent download verification. No native model training/evaluation was launched. Phase2 archives streamed at BelowNormal priority with one compression worker.','source_reports':['research-status.json','remote-research-status.json','auxiliary-audit.json','auxiliary-dependency-notes.json','supa-audit.json','phase2-research-retirement.json','phase2-auxiliary-retirement.json','phase2-storage-review.json'],'internal_ref_repair':'Two 41-byte internal Codex ref files consisted entirely of null bytes and had no valid object identity. Moved outside Git to quarantined-invalid-ref-0.bin / -1.bin; ordinary fetch succeeded afterwards.'}
if (OUT/'publication-current-state.json').exists():
    report['publication_state_review']=json.loads((OUT/'publication-current-state.json').read_text())
    summary['interpretation']='Timestamped cleanup endpoint: 14 removal events and 29 populated worktrees at completion. Later restores/new owner work are recorded separately; logical removed file lengths are not current net freed disk allocation.'
if (OUT/'phase2-hardware-final.json').exists():
    report['hardware_before_phase2']=json.loads((OUT/'phase2-hardware-baseline.json').read_text(encoding='utf-8-sig'))
    report['hardware_after_phase2']=json.loads((OUT/'phase2-hardware-final.json').read_text(encoding='utf-8-sig'))
    report['parser_dependency_handoff']=json.loads((OUT/'phase2-parser-migration-receipt.json').read_text())
    report['cache_cleanup_rejection']=json.loads((OUT/'phase2-cache-cleanup-receipt.json').read_text())
(OUT/'audit-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def e(v): return html.escape(str(v))
def size(n): return f'{n/1024**3:.2f} GiB' if n>=1024**3 else f'{n/1024**2:.1f} MiB'
def table(headers,rows): return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+c+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
parts=['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rineya research worktree audit · October 9, 2026</title><style>body{margin:0;background:#f3f5f8;color:#142033;font:15px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:32px auto;padding:0 24px}h1{font-size:32px;letter-spacing:-.7px;margin-bottom:6px}h2{font-size:22px;margin-top:34px}p{max-width:1080px}.meta{color:#596579}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}.card{background:white;padding:20px;border:1px solid #dce2eb;border-radius:12px}.card strong{display:block;font-size:29px}.scroll{overflow:auto;border:1px solid #dce2eb;background:white;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:11px 13px;border-bottom:1px solid #edf0f5;vertical-align:top}th{background:#e8edf5;color:#304057;white-space:nowrap}td:first-child{min-width:170px}code{font:12px ui-monospace,monospace;overflow-wrap:anywhere}a{color:#155bb5}input{border:1px solid #bcc8d8;border-radius:8px;padding:12px;font:inherit;width:min(600px,90%);margin:12px 0}details{margin:14px 0;padding:14px;background:white;border:1px solid #dce2eb;border-radius:10px}.good{color:#19633e}.hold{color:#9a4d05}.small{font-size:12px;color:#667389}ul{padding-left:24px}@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:0 12px}h1{font-size:26px}}</style><main><h1>Rineya research worktree audit</h1><div class="meta">October 9, 2026 · America/Chicago · C:\\Users\\clclc\\BusinessProject</div>''']
parts.append(f'<p>Removed {len(removed)} redundant, completed, or dormant research checkouts after remote preservation. Kept active owners and live dependencies. Unpublished source and scientific evidence in retired checkouts were preserved in verified private snapshots. No research branch was deleted, and no research hypothesis was declared obsolete merely because its checkout was old or blocked.</p>')
parts.append('<div class="cards">'+''.join(f'<div class="card"><strong>{e(a)}</strong>{e(b)}</div>' for a,b in [(str(len(removed)),'historical checkout removal events'),(f'{cleanup["logical_bytes_removed"]/1024**3:.1f} GiB','logical files removed at cleanup'),(str(len(tags)),'remote archive tags verified'),('3','stale registrations pruned')])+'</div>')
parts.append(f'<p>Examined <strong>{summary["local_branch_refs_examined"]} local branch refs</strong> and <strong>{summary["live_remote_branch_refs_examined"]} live remote branch refs</strong> across five remote repositories and seven local repository locations. At cleanup completion, the populated-worktree count fell from {before_live} to {before_live-len(removed)}; four retained checkouts were outside the requested directory. {len(nonworktree)} related directories without a top-level Git worktree marker were retained as research data, receipts, fixtures, or outputs.</p>')
if 'publication_state_review' in report:
    current=report['publication_state_review']
    parts.append(f'<p><strong>Later publication review ({e(current["captured_utc"])}):</strong> the current inventory contains {current["populated_worktrees_now"]} populated worktrees. {len(current["later_restored_paths"])} formerly removed paths have since been restored by other research owners. The 29-worktree figure above is the cleanup endpoint; the 14 deletion receipts remain historical evidence. Current owner work is retained.</p>')
parts.append('<h2>Removed checkouts and recovery</h2>')
parts.append(table(['Checkout','Preserved commit','Remote tag / artifacts','Removed files'],[[e(Path(w['path']).name),'<code>'+e(w['head'][:12])+'</code>',f'<a href="https://github.com/LUNCY579/{"ChatBubble" if w["repository"]=="Chat-Bubble" else w["repository"]}/tree/{quote(w["verified_remote_tag"],safe="/")}">{e(w["verified_remote_tag"])}</a>'+('<br><a href="'+e(w['scientific_artifact_release'])+'">Verified private snapshot release</a>' if w.get('scientific_artifact_release') else ''),size(w['logical_bytes_removed'])] for w in cleanup['worktrees']]))
parts.append('<p>The hybrid-memory release preserves 145 scientific files, including trained custom sidecar tensors, immutable synthetic banks, raw evaluation and recovery records, and ledgers. Every member matched its original length and SHA256. All four release assets were independently downloaded and matched the local hashes. Disposable runtime packages and caches were excluded.</p>')
if phase2_releases:
    hw=report['hardware_after_phase2']
    parts.append(f'<p>The second pass archived seven additional dormant checkouts, preserving 167 modified, untracked, or scientifically material ignored files plus their exact working-state manifests and tracked-difference patches. All seven releases were independently downloaded and hash-verified. There are {1+len(phase2_releases)} verified private preservation releases in total.</p><h2>Hardware and dependency handoff</h2><p>Archive compression ran serially at BelowNormal priority, using one worker and 1 MiB streaming buffers. No Torch, model, or GPU workload was started. Complete CUDA environments and active runtime donors remain. Final hardware snapshot: {hw["FreeRAMBytes"]/1024**3:.1f} GiB free RAM and {hw["FreeDiskBytes"]/1024**3:.1f} GiB free on C:; other owner jobs continued independently.</p><p>One pruning recount lookup now resolves to the identical retained CED parser. The asserted parser SHA256 is unchanged. The original script and historical receipts are archived privately, and old reports retain their original source hash/path. No experiment was rerun and no selection, threshold, outcome, or scientific claim changed.</p><p>Automatic approval review rejected removal of 17 small source-backed bytecode cache files with “blocked by policy” and provided no further reason. Those candidates remain untouched (about 232 KB); no complete runtime environment was removed.</p>')
parts.append('<details><summary>Restore a removed checkout when needed</summary><p>Every original research branch remains. Use the retained primary repository to add a worktree at its original path from the recorded branch or archive tag. For hybrid memory, also download the artifact release and extract the archive at that checkout root. Original receipt bytes and historical paths are preserved.</p><pre><code>git -C &lt;primary-repository&gt; worktree add --detach &lt;original-checkout-path&gt; &lt;archive-tag&gt;</code></pre></details>')
parts.append('<h2>Why other checkouts stay</h2><p>Age and a clean Git status do not establish redundancy. CLARE supplies WorldModel data and shared leases; COMP-RS S1 supplies CE8 qualification code; shipping remains an M2 dependency. Blocked studies retain their owners and questions. Tags preserve committed history but exclude untracked proposals, run receipts, ignored checkpoints, and local datasets.</p><input id="filter" placeholder="Filter worktrees and branch tables by name, repository, or action" aria-label="Filter audit tables">')
parts.append(table(['Repository / checkout','Decision','Tracked / new / ignored files','Source / owner reason'],[[e(w['repository'])+'<br><code>'+e(Path(w['path']).name)+'</code><div class="small">'+e(w['branch'].removeprefix('refs/heads/'))+'</div>','<strong class="'+('good' if w['action'].startswith(('Removed','Pruned')) else 'hold')+'">'+e(w['action'])+'</strong>',e(f'{w["tracked_change_count"]} / {w["untracked_file_count"]} / {w["ignored_file_count"]}')+'<div class="small">'+size(w['size_bytes'])+'</div>',e(w['reason'])+('<div class="small">Goal states: '+e(', '.join(sorted({g['status'] for g in w['owners']})))+'</div>' if w['owners'] else '')] for w in worktrees]))
parts.append(f'<h2>Committed history still local</h2><p>Previously local-only quality-SFT, COMP-RS S1, Supa quality, GEN2, and TGSA MTP tips now have verified remote source tags. The following {len(summary["local_only_committed_refs_retained"])} active or custody-related ChatBubble tips remain retained locally. They require an appropriately reviewed remote preservation payload before future source/ref retirement. The small bare DSD recovery repository has no own remote, but its exact commit is already preserved in ChatBubble.</p>')
parts.append(table(['Repository','Local branch','Tip','Commits absent from this GitHub remote'],[[e(b['repository']),e(b['branch']),'<code>'+e(b['head'][:12])+'</code>',e(b['uncovered_commits_in_same_remote'])] for b in summary['local_only_committed_refs_retained']]))
parts.append('<h2>All branch dispositions</h2><p>All local branches and remote branches without local branch refs are listed. Existing archival branches and tags were verified through live ref identities and ancestry. Remote-only branches consume no checkout space. Fifteen remote-only ChatBubble research branches also received source-status review: seven newly fetched Rineya tracks contain distinct CPU/reference implementations; ID balancing exactly aliases origin/main and has no dedicated package. Remote research history remains intact.</p>')
parts.append(table(['Repository / kind','Branch','Tip / source status','Remote coverage','Disposition'],[[e(b['repository'])+'<div class="small">'+e(b['kind'])+'</div>','<code>'+e(b['branch'])+'</code>','<code>'+e(b['head'][:12])+'</code><div class="small">'+e(b['subject'])+'</div>','<strong class="'+('good' if b['live_remote_preserved'] else 'hold')+'">'+('Preserved' if b['live_remote_preserved'] else str(b['uncovered_commits_in_same_remote'])+' commits local only')+'</strong>',e(b['disposition'])] for b in sorted(branches,key=lambda x:(x['repository'],x['branch'],x['kind']))]))
parts.append('<h2>Audit evidence</h2><p>Snapshots and decisions reflect the scan window. Other owner chats continued independently, so retained checkouts can acquire newer files. The only retained-source edit was the documented single parser-location literal; original source and receipts are preserved. No retained work was reset, stashed, staged, or committed by this cleanup. Two corrupt internal Codex refs containing only null bytes were quarantined locally so normal Git fetching could succeed.</p><ul>')
for f,label in [('audit-report.json','Complete decisions and branch inventory'),('inventory.json','Detailed Git and tracked/untracked/ignored file inventories'),('cleanup-receipt.json','Exact deletion receipt'),('remote-preservation.json','Ten verified remote annotated tags'),('supa-hybrid-remote-verification.json','Downloaded release asset verification'),('research-status.json','Nine historical research checkout reviews'),('remote-research-status.json','Fifteen remote-only research branch source reviews'),('auxiliary-dependency-notes.json','Executable path dependency findings'),('auxiliary-audit.json','Auxiliary checkout and evidence inventory'),('supa-audit.json','Supa, architecture and retention-repository review'),('non-worktree-research-directories.json','Retained non-worktree research directories'),('pruned-stale-registrations.json','Three stale registrations removed')]:
    parts.append('<li><a href="'+e(f)+'">'+e(label)+'</a></li>')
for f,label in [('phase2-cleanup-receipt.json','Seven additional deletion receipts'),('phase2-remote-release-verification.json','Seven verified private snapshots and downloaded assets'),('phase2-verified-tags.json','Nine additional verified source tags'),('phase2-parser-migration-receipt.json','Exact-byte dependency handoff'),('phase2-cache-cleanup-receipt.json','Blocked cache action; no deletions performed'),('phase2-research-retirement.json','Dormant research checkout assessment'),('phase2-auxiliary-retirement.json','Dormant auxiliary checkout assessment'),('phase2-storage-review.json','Runtime and cache storage assessment'),('phase2-hardware-final.json','Final hardware snapshot'),('audit-report-phase1.html','First-pass report retained for history')]:
    if (OUT/f).exists():parts.append('<li><a href="'+e(f)+'">'+e(label)+'</a></li>')
if (OUT/'publication-current-state.json').exists():parts.append('<li><a href="publication-current-state.json">Later owner restorations and current publication inventory</a></li>')
parts.append('</ul><div class="small">File size totals are logical file lengths, not a measured disk-allocation change. Shared primary Git object databases remain.</div></main><script>document.getElementById("filter").addEventListener("input",function(){let q=this.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>')
(OUT/'audit-report.html').write_text(''.join(parts),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('HTML report:',OUT/'audit-report.html')
