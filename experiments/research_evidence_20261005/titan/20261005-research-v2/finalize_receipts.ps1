param([ValidateSet('cpu','index')][string]$Stage)
$ErrorActionPreference = 'Stop'
$qualityRoot = 'C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1/20261005-research-v2'
$qualityRepo = 'C:/Users/clclc/BusinessProject/ChatBubble-triskill-geo001-sectionh'
$qualitySource = Join-Path $qualityRepo 'experiments/titan_quality_repair_v1'
function Read-QualityJson([string]$Path) { Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json -AsHashtable }
function Get-QualityHash([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }
function Get-QualityDescriptor([string]$Path) { @{path=$Path; sha256=(Get-QualityHash $Path)} }
function Write-QualityJson([string]$Path, $Value) {
    $qualityBytes = [System.Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 80) + "`n")
    $qualityStream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
    try { $qualityStream.Write($qualityBytes); $qualityStream.Flush($true) } finally { $qualityStream.Dispose() }
}
function Assert-QualityMap($Expected, $Actual) {
    if ($Expected.Count -ne $Actual.Count) { throw 'Source map membership changed' }
    foreach ($qualityKey in $Expected.Keys) {
        if (-not $Actual.ContainsKey($qualityKey) -or $Actual[$qualityKey] -cne $Expected[$qualityKey]) { throw "Source drift: $qualityKey" }
    }
}
$qualityProposalPath = Join-Path $qualityRoot 'spec-002/NATIVE_QUALIFICATION_V2.json'
$qualityProposal = Read-QualityJson $qualityProposalPath
$qualityManifest = @{}
foreach ($qualityFile in Get-ChildItem -LiteralPath $qualitySource -File) { $qualityManifest[$qualityFile.Name] = Get-QualityHash $qualityFile.FullName }
Assert-QualityMap $qualityProposal.source_manifest $qualityManifest
$qualityFixturesPath = Join-Path $qualityRoot 'fixtures-002/OFFLINE_SUPERVISOR_RECEIPT.json'
$qualityContainmentPath = Join-Path $qualityRoot 'containment-001/OFFLINE_SUPERVISOR_RECEIPT.json'
$qualityFixtureResultPath = Join-Path $qualityRoot 'fixtures-002/CPU_FIXTURES.json'
$qualityFixtures = Read-QualityJson $qualityFixturesPath
$qualityContainment = Read-QualityJson $qualityContainmentPath
$qualityFixtureResult = Read-QualityJson $qualityFixtureResultPath
foreach ($qualityReceipt in @($qualityFixtures, $qualityContainment)) {
    if ($qualityReceipt.status -cne 'PASS' -or $qualityReceipt.source_unchanged_during_worker -ne $true -or $qualityReceipt.job_assigned_before_fixture_imports -ne $true -or $qualityReceipt.native_model_loads -ne 0 -or $qualityReceipt.native_model_forwards -ne 0) { throw 'CPU receipt is not genuine current-source PASS evidence' }
    Assert-QualityMap $qualityManifest $qualityReceipt.source_admission_sha256
    Assert-QualityMap $qualityManifest $qualityReceipt.source_completion_sha256
}
if ($qualityContainment.exclusive_handle_refuses_competing_open -ne $true) { throw 'Measured exclusive hardware handle proof missing' }
if ($qualityFixtureResult.passed -ne $true -or $qualityFixtureResult.failures -ne 0 -or $qualityFixtureResult.errors -ne 0 -or $qualityFixtureResult.model_loads -ne 0 -or $qualityFixtureResult.model_forwards -ne 0) { throw 'CPU fixture failures or native activity' }
foreach ($qualityKey in @('payload_in_job','grandchild_in_job','grandchild_exited')) {
    if ($qualityContainment.containment_fixture[$qualityKey] -ne $true) { throw "Containment proof missing: $qualityKey" }
}
$qualityCpuPath = Join-Path $qualityRoot 'CPU_SOURCE_QUALIFICATION.json'
if ($Stage -eq 'cpu') {
    $qualityCpu = @{
        experiment='TITAN-QUALITY-REPAIR-V1'; status='PASS'; scope='MEASURED_CPU_FIXTURES_AND_TINY_PROCESS_TREE_ONLY';
        source_manifest_sha256=$qualityProposal.source_manifest_sha256;
        source_unchanged_during_worker=$true; job_assigned_before_fixture_imports=$true;
        native_model_loads=0; native_model_forwards=0; native_generations=0; training_updates=0;
        actual_native_runtime_qualified=$false; capability_evidence=$false; native_execution_authorized=$false;
        tests_run=$qualityFixtureResult.tests_run;
        actual_receipts=@((Get-QualityDescriptor $qualityFixturesPath),(Get-QualityDescriptor $qualityFixtureResultPath),(Get-QualityDescriptor $qualityContainmentPath),(Get-QualityDescriptor (Join-Path $qualityRoot 'containment-001/CONTAINMENT_PAYLOAD.json')),(Get-QualityDescriptor (Join-Path $qualityRoot 'containment-001/GRANDCHILD.json')));
        derivation='Raw measured PASS receipts, admitted/completed maps, exact Job descendant exit and zero-model counters verified; this file is an evidence index, not a native or operator permit.';
        utc=[DateTime]::UtcNow.ToString('o')
    }
    Write-QualityJson $qualityCpuPath $qualityCpu
    $qualityCpu | Select-Object status,tests_run,native_model_loads,source_manifest_sha256 | ConvertTo-Json
    exit
}
$qualityReviewPath = Join-Path $qualityRoot 'INDEPENDENT_REVIEW.json'
$qualityReview = Read-QualityJson $qualityReviewPath
if ($qualityReview.status -cne 'PASS_FOR_BOUNDED_NATIVE_QUALIFICATION_SOURCE' -or $qualityReview.scope -cne 'STATIC_CODE_AND_CPU_MECHANICS_ONLY' -or $qualityReview.source_manifest_sha256 -cne $qualityProposal.source_manifest_sha256 -or $qualityReview.proposal_payload_sha256 -cne $qualityProposal.proposal_payload_sha256 -or $qualityReview.native_execution_authorized -ne $false) { throw 'Independent review missing or stale' }
$qualityProposal.review_descriptor = Get-QualityDescriptor $qualityReviewPath
$qualityProposal.cpu_qualification_descriptor = Get-QualityDescriptor $qualityCpuPath
$qualityReviewedProposalPath = Join-Path $qualityRoot 'NATIVE_QUALIFICATION_REVIEWED.json'
Write-QualityJson $qualityReviewedProposalPath $qualityProposal
$qualityInitialPath = Join-Path (Split-Path $qualityRoot -Parent) '20261005-offline/INITIAL_IDENTITY.json'
$qualityInitial = Read-QualityJson $qualityInitialPath
foreach ($qualityKey in $qualityInitial.parentq_and_geo_hashes.Keys) {
    if ((Get-QualityHash (Join-Path $qualityRepo $qualityKey)) -cne $qualityInitial.parentq_and_geo_hashes[$qualityKey]) { throw "Preservation drift: $qualityKey" }
}
$qualityOldIndexPath = Join-Path (Split-Path $qualityRoot -Parent) '20261005-offline/VERIFICATION_INDEX.json'
$qualityOldIndex = Read-QualityJson $qualityOldIndexPath
$qualitySnapshot = @{}
foreach ($qualityFile in Get-ChildItem -LiteralPath (Join-Path $qualityRoot 'original-offline-source') -File) { $qualitySnapshot[$qualityFile.Name] = Get-QualityHash $qualityFile.FullName }
Assert-QualityMap $qualityOldIndex.package_source_sha256 $qualitySnapshot
foreach ($qualityScientificFile in @('registry.py','generator.py','oracles.py','schema.py','scoring.py','selection.py','statistics.py','token_audit.py','packet.py','exclusion.py','replay.py','budget.py')) {
    if ($qualityManifest[$qualityScientificFile] -cne $qualitySnapshot[$qualityScientificFile]) { throw "Original scientific implementation changed: $qualityScientificFile" }
}
$qualityReports = @{}
foreach ($qualityStageName in @('fixtures-001','input-patterns-001','spec-001','fixtures-002','containment-001','spec-002','input-patterns-002')) {
    $qualityReceiptPath = Join-Path $qualityRoot "$qualityStageName/OFFLINE_SUPERVISOR_RECEIPT.json"
    $qualityStageReceipt = Read-QualityJson $qualityReceiptPath
    $qualityReports[$qualityStageName] = @{status=$qualityStageReceipt.status; elapsed_seconds=$qualityStageReceipt.elapsed_seconds; peak_tree_rss_bytes=$qualityStageReceipt.peak_tree_rss_bytes; peak_output_bytes=$qualityStageReceipt.peak_output_bytes; receipt=(Get-QualityDescriptor $qualityReceiptPath)}
}
$qualityInputPath = Join-Path $qualityRoot 'input-patterns-002/INPUT_PATTERN_AUDIT.json'
$qualityInput = Read-QualityJson $qualityInputPath
Assert-QualityMap $qualityManifest $qualityInput.source_manifest
$qualityIndex = @{
    experiment='TITAN-QUALITY-REPAIR-V1'; utc=[DateTime]::UtcNow.ToString('o');
    status='OFFLINE_PACKET_AND_QUALIFICATION_SOURCE_REVIEWED_NATIVE_UNRUN'; goal_completed=$false;
    branch=(& git -C $qualityRepo branch --show-current).Trim(); head=(& git -C $qualityRepo rev-parse HEAD).Trim();
    source_manifest_sha256=$qualityProposal.source_manifest_sha256; source_manifest=$qualityManifest;
    native_qualification_proposal=(Get-QualityDescriptor $qualityReviewedProposalPath);
    qualification_payload_sha256=$qualityProposal.proposal_payload_sha256;
    independent_review=(Get-QualityDescriptor $qualityReviewPath); cpu_source_qualification=(Get-QualityDescriptor $qualityCpuPath);
    original_offline_packet=(Get-QualityDescriptor $qualityOldIndexPath); original_source_snapshot_files=$qualitySnapshot.Count;
    original_scientific_implementation_files_unchanged=$true;
    original_parentq_geo_file_hashes_verified=$qualityInitial.parentq_and_geo_hashes.Count;
    parentq_and_geo_preserved=$true; geometry_composition='DEFERRED'; published_core_modified=$false;
    source_metadata_and_sidecars_verified=$true; fresh_weight_digest_verified=$false;
    native_model_loads=0; native_model_forwards=0; training_updates=0; verified_model_quality_gains=0;
    operator_permit_created=$false; native_execution_authorized=$false; training_sweep_authorized=$false; final_evaluation_authorized=$false; promotion_authorized=$false;
    protected_bank_accessed=$false; final_model_results_opened=$false;
    tests_run=$qualityFixtureResult.tests_run; stage_receipts=$qualityReports;
    input_patterns=(Get-QualityDescriptor $qualityInputPath);
    opposite_directions=(Get-QualityDescriptor (Join-Path $qualityRoot 'spec-002/OPPOSITE_DIRECTIONS.json'));
    decision_rule_audit=(Get-QualityDescriptor (Join-Path $qualityRoot 'spec-002/OPERATING_CHARACTERISTICS.json'));
    observed_input_pattern='In all held-factor/both-seed pairs, 20% replay changes missing-information examples from 500/2000 to 400/2000. Fixed-budget retention and answerability-dose effects are coupled.';
    prospective_rule_pattern='At 48-endpoint Bonferroni alpha .05, conditional damage upper bound .10 requires at least 66 baseline-correct cases even with zero losses. No observed final outcome was used; thresholds unchanged.';
    model_direction_status='UNMEASURED';
    limitations=@('No trained candidate or verified quality gain exists.', 'Actual native CUDA/HF/optimizer/serialization/padding behavior and full-program throughput remain unqualified.', 'Full checkpoint bytes have not been hashed or loaded offline; native read custody is prospective.', 'CPU process-tree checks do not qualify native GPU/optimizer containment.', 'The fixed final confirms selected recipe performance, not universal deterministic factorial effects.', 'Published source backup remote receipt remains unverified; no network/source writes occurred.', 'Original raw-output packet failure and first continuation cleanup fixture failure are retained.');
    next_concrete_decision='Allocate a named exclusive GPU0 slot and explicitly authorize this exact reviewed bounded qualification only. It charges 720 seconds with 540/600/120 limits, eight fixed DEV readouts, two disposable updates and one dense reload; no sweep/final/promotion grant.'
}
if ($qualityIndex.branch -cne 'research/triskill-geo001-parentq' -or $qualityIndex.head -cne $qualityInitial.head) { throw 'Branch or HEAD changed' }
Write-QualityJson (Join-Path $qualityRoot 'VERIFICATION_INDEX.json') $qualityIndex
$qualityIndex | Select-Object status,tests_run,branch,native_model_loads,verified_model_quality_gains,goal_completed | ConvertTo-Json
