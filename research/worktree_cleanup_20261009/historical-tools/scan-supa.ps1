$ErrorActionPreference='Stop'
$base='C:\Users\clclc\BusinessProject'
$output=Join-Path $base 'Rineya-Worktree-Audit-20261009\supa-audit.json'
$roots=@('SupaExperiment','SupaExperiment-hybrid-memory-v1','SupaExperiment-matched-mlp-m2-readiness-v1','SupaExperiment-matched-mlp-shipping-v1','SupaExperiment-nonrecurrent-adaptation-v1','SupaExperiment-rineya-quality-v1','RineyaArchitecture','rineya-dataset-retention-records')
function GitLines($p,[string[]]$argsList) { $out=@(& git -c core.longpaths=true -C $p @argsList 2>&1 | ForEach-Object {"$_"}); return $out }
function FileFacts($p,$paths) {
 $facts=@(foreach($rel in $paths) { $fp=Join-Path $p $rel; if(Test-Path -LiteralPath $fp -PathType Leaf) { $i=Get-Item -LiteralPath $fp -Force; [pscustomobject]@{path=$rel;bytes=$i.Length;modifiedUtc=$i.LastWriteTimeUtc.ToString('o');attributes="$($i.Attributes)"} } else { [pscustomobject]@{path=$rel;bytes=0;missing=$true} } })
 return $facts
}
$reports=@(foreach($name in $roots) {
 $p=Join-Path $base $name
 $status=@(GitLines $p @('--no-optional-locks','status','--porcelain=v1','--untracked-files=all'))
 $ignoredPaths=@(GitLines $p @('ls-files','--others','--ignored','--exclude-standard'))
 $untrackedPaths=@(GitLines $p @('ls-files','--others','--exclude-standard'))
 $ignored=@(FileFacts $p $ignoredPaths)
 $untracked=@(FileFacts $p $untrackedPaths)
 $refs=@(foreach($line in (GitLines $p @('for-each-ref','--format=%(refname)|%(objectname)|%(*objectname)|%(upstream)|%(upstream:track)|%(committerdate:iso-strict)|%(subject)','refs/heads','refs/remotes','refs/tags'))) { $parts=$line -split '\|',7; [pscustomobject]@{ref=$parts[0];object=$parts[1];peeledObject=$parts[2];upstream=$parts[3];upstreamTrack=$parts[4];committerDate=$parts[5];subject=$parts[6]} })
 $branches=@(foreach($ref in $refs | Where-Object {$_.ref.StartsWith('refs/heads/')}) {
  $remoteCoverage=@(GitLines $p @('for-each-ref',"--contains=$($ref.object)",'--format=%(refname)','refs/remotes'))
  $tagCoverage=@(GitLines $p @('for-each-ref',"--contains=$($ref.object)",'--format=%(refname)','refs/tags'))
  $exactTags=@($refs | Where-Object {$_.ref.StartsWith('refs/tags/') -and ($_.object -eq $ref.object -or $_.peeledObject -eq $ref.object)} | ForEach-Object {$_.ref})
  $uniqueNotRemote=@(GitLines $p @('rev-list','--count',$ref.object,'--not','--remotes'))
  [pscustomobject]@{ref=$ref.ref;head=$ref.object;remoteCoverage=$remoteCoverage;tagCoverage=$tagCoverage;exactTags=$exactTags;commitsNotReachableFromRemoteRefs=$uniqueNotRemote[0]}
 })
 $head=@(GitLines $p @('rev-parse','HEAD'))[0]
 $branch=@(GitLines $p @('symbolic-ref','--quiet','--short','HEAD'))[0]
 $submodules=@(GitLines $p @('submodule','status','--recursive'))
 $nested=@(foreach($child in (Get-ChildItem -LiteralPath $p -Directory -Force)) { if(Test-Path -LiteralPath (Join-Path $child.FullName '.git')) { [pscustomobject]@{path=$child.FullName;head=@(GitLines $child.FullName @('rev-parse','HEAD'))[0];branch=@(GitLines $child.FullName @('symbolic-ref','--quiet','--short','HEAD'));status=@(GitLines $child.FullName @('--no-optional-locks','status','--porcelain=v1','--untracked-files=all'));ignoredPaths=@(GitLines $child.FullName @('ls-files','--others','--ignored','--exclude-standard'));remotes=@(GitLines $child.FullName @('remote','-v'));worktrees=@(GitLines $child.FullName @('worktree','list','--porcelain'))} } })
 [pscustomobject]@{
  name=$name;path=$p;head=$head;branch=$branch;status=$status;clean=($status.Count -eq 0);
  remotes=@(GitLines $p @('remote','-v'));worktrees=@(GitLines $p @('worktree','list','--porcelain'));
  refs=$refs;branches=$branches;submoduleStatus=$submodules;nestedRepositories=$nested;
  ignoredCount=$ignored.Count;ignoredBytes=($ignored | Measure-Object bytes -Sum).Sum;ignored=$ignored;
  untrackedCount=$untracked.Count;untrackedBytes=($untracked | Measure-Object bytes -Sum).Sum;untracked=$untracked;
  ignoredTopLevel=@($ignored | Group-Object {($_.path -split '/')[0]} | ForEach-Object {[pscustomobject]@{name=$_.Name;count=$_.Count;bytes=($_.Group | Measure-Object bytes -Sum).Sum}} | Sort-Object bytes -Descending);
  largestIgnored=@($ignored | Sort-Object bytes -Descending | Select-Object -First 25)
 }
})
[pscustomobject]@{auditDateClient='2026-10-09';capturedUtc=[DateTime]::UtcNow.ToString('o');scope='Read-only repository scan: SupaExperiment registered worktrees, RineyaArchitecture, and rineya-dataset-retention-records';repositories=$reports} | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $output -Encoding UTF8
$reports | Select-Object name,head,branch,clean,untrackedCount,untrackedBytes,ignoredCount,ignoredBytes | Format-Table -AutoSize
Write-Output "Saved $output"
