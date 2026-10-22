[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'Qualification',
        'Synthetic',
        'PhaseCPrepare',
        'PhaseCCache',
        'PhaseCPreregister',
        'PhaseCPilot',
        'PhaseCQualification',
        'PhaseCGate',
        'PhaseDGuard'
    )]
    [string] $Stage,

    [string] $OriginalRepoRoot = '',
    [string] $LabRoot = '',
    [string] $RunRoot = '',
    [string] $BaseModel = $env:PCSM_BASE_MODEL,
    [string] $CorpusRoot = $env:PCSM_CORPUS_ROOT,

    [ValidateSet('cpu', 'cuda')]
    [string] $Device = 'cuda',

    [int] $MaximumTokens = 40960,
    [double] $CudaMinimumFreeGiB = 6.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ExistingDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,
        [Parameter(Mandatory = $true)]
        [string] $Label
    )

    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label directory does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Require-File {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,
        [Parameter(Mandatory = $true)]
        [string] $Label
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $PythonArguments,
        [string] $Interpreter = $script:RuntimePython
    )

    & $Interpreter @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($PythonArguments -join ' ')"
    }
}

function Read-JsonObject {
    param([Parameter(Mandatory = $true)][string] $Path)
    Require-File -Path $Path -Label 'JSON artifact'
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Assert-CommittedFile {
    param([Parameter(Mandatory = $true)][string] $Path)

    Require-File -Path $Path -Label 'Committed root-of-trust file'
    $GitRoot = (& git -C $script:LabRootForGit rev-parse --show-toplevel).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($GitRoot)) {
        throw 'Unable to resolve the Git root for preregistration authentication.'
    }
    $Resolved = (Resolve-Path -LiteralPath $Path).Path
    $Relative = [System.IO.Path]::GetRelativePath($GitRoot, $Resolved).Replace('\', '/')
    & git -C $GitRoot ls-files --error-unmatch -- $Relative *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Preregistration is not tracked at HEAD: $Relative"
    }
    & git -C $GitRoot diff --quiet HEAD -- $Relative
    if ($LASTEXITCODE -ne 0) {
        throw "Preregistration differs from committed HEAD bytes: $Relative"
    }
}

if ([string]::IsNullOrWhiteSpace($LabRoot)) {
    $LabRoot = Join-Path $PSScriptRoot '..\..'
}
$LabRoot = Resolve-ExistingDirectory -Path $LabRoot -Label 'PCSM lab'
$script:LabRootForGit = $LabRoot

if ([string]::IsNullOrWhiteSpace($OriginalRepoRoot)) {
    $WorktreeRoot = Split-Path -Parent $LabRoot
    $WorkspaceParent = Split-Path -Parent $WorktreeRoot
    $OriginalRepoRoot = Join-Path $WorkspaceParent 'Chat-Bubble'
}
$OriginalRepoRoot = Resolve-ExistingDirectory -Path $OriginalRepoRoot -Label 'Original repository'

$script:RuntimePython = Join-Path $OriginalRepoRoot '.venv\Scripts\python.exe'
$script:DevPython = Join-Path $LabRoot '.venv\Scripts\python.exe'
Require-File -Path $script:RuntimePython -Label 'Original repository runtime Python interpreter'
Require-File -Path $script:DevPython -Label 'PCSM development Python interpreter'

$NeedsRunRoot = $Stage -like 'PhaseC*' -or $Stage -eq 'PhaseDGuard'
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    if ($NeedsRunRoot) {
        throw 'Phase-C and Phase-D stages require an explicit -RunRoot so every stage uses the same frozen artifacts.'
    }
    $Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $RunRoot = Join-Path $LabRoot "results\reproduction\$Stamp"
}
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

$env:PYTHONPATH = Join-Path $LabRoot 'src'
$PreviousLocation = Get-Location
Set-Location -LiteralPath $LabRoot

try {
    Write-Host "PCSM stage: $Stage"
    Write-Host "Lab root: $LabRoot"
    Write-Host "CUDA/runtime interpreter: $script:RuntimePython"
    Write-Host "Qualification interpreter: $script:DevPython"
    Write-Host "Run root: $RunRoot"

    if ($Stage -eq 'Qualification') {
        Invoke-CheckedPython -Interpreter $script:DevPython -PythonArguments @(
            '-m', 'pytest', '-q'
        )
        Invoke-CheckedPython -Interpreter $script:DevPython -PythonArguments @(
            '-m', 'ruff', 'check', 'src', 'tests', 'scripts'
        )
        Invoke-CheckedPython -Interpreter $script:DevPython -PythonArguments @(
            '-m', 'mypy', '--strict', 'src\rineya_pcsm'
        )
        return
    }

    if ($Stage -eq 'Synthetic') {
        $SyntheticRoot = Join-Path $RunRoot 'synthetic'
        New-Item -ItemType Directory -Path $SyntheticRoot -Force | Out-Null
        Invoke-CheckedPython -Interpreter $script:DevPython -PythonArguments @(
            'scripts\run_synthetic.py',
            '--config', 'configs\synthetic.yaml',
            '--output-dir', $SyntheticRoot,
            '--bootstrap-resamples', '10000',
            '--statistics-seed', '20260826'
        )
        return
    }

    $BaseModel = Resolve-ExistingDirectory -Path $BaseModel -Label 'PCSM base model'
    $CorpusRoot = Resolve-ExistingDirectory -Path $CorpusRoot -Label 'PCSM corpus root'
    $env:PCSM_BASE_MODEL = $BaseModel
    $env:PCSM_CORPUS_ROOT = $CorpusRoot

    Require-File -Path (Join-Path $BaseModel 'model.safetensors') -Label 'Base model weights'
    Require-File -Path (Join-Path $BaseModel 'config.json') -Label 'Base model config'
    Require-File -Path (Join-Path $BaseModel 'tokenizer.json') -Label 'Base tokenizer'
    Require-File -Path (Join-Path $CorpusRoot 'reports\main_lineage\train.jsonl') -Label 'Train lineage'
    Require-File -Path (Join-Path $CorpusRoot 'reports\main_lineage\validation.jsonl') -Label 'Validation lineage'

    $PhaseRoot = Join-Path $RunRoot 'phase_c'
    $SplitManifest = Join-Path $PhaseRoot 'split_manifest.json'
    $SampleManifest = Join-Path $PhaseRoot 'sample_manifest.json'
    $ResourcePreflight = Join-Path $PhaseRoot 'resource_preflight.json'
    $BudgetManifest = Join-Path $PhaseRoot 'budget_manifest.json'
    $GradientRoot = Join-Path $RunRoot 'cache\phase_c\gradients'
    $GradientSummary = Join-Path $PhaseRoot 'gradient_cache_summary.json'
    $Preregistration = Join-Path $PhaseRoot 'PREREGISTRATION.json'
    $PlanRoot = Join-Path $RunRoot 'plans'
    $PilotPlans = Join-Path $PhaseRoot 'pilot_plans.json'
    $PilotRun = Join-Path $PhaseRoot 'pilot_run.json'
    $PilotDecision = Join-Path $PhaseRoot 'pilot_decision.json'
    $QualificationPlans = Join-Path $PhaseRoot 'qualification_plans.json'
    $QualificationRun = Join-Path $PhaseRoot 'qualification_run.json'
    $QualificationDecision = Join-Path $PhaseRoot 'qualification_decision.json'
    $Gate = Join-Path $PhaseRoot 'ADVANCEMENT_GATE.json'
    New-Item -ItemType Directory -Path $PhaseRoot -Force | Out-Null

    if ($Stage -eq 'PhaseCPrepare') {
        Invoke-CheckedPython -PythonArguments @(
            'scripts\build_phase_c_splits.py',
            '--corpus-root', $CorpusRoot,
            '--output', $SplitManifest,
            '--meta-components', '20',
            '--mask-components', '5',
            '--dev-components', '1',
            '--salt', 'pcsm-phase-c-v1-frozen'
        )
        Invoke-CheckedPython -PythonArguments @(
            'scripts\freeze_phase_c_samples.py',
            '--split-manifest', $SplitManifest,
            '--corpus-root', $CorpusRoot,
            '--model', $BaseModel,
            '--output', $SampleManifest,
            '--per-split', '5',
            '--maximum-tokens', [string] $MaximumTokens,
            '--salt', 'pcsm-phase-c-v1-sample'
        )
        Invoke-CheckedPython -PythonArguments @(
            'scripts\check_phase_c_resources.py',
            '--output', $ResourcePreflight,
            '--cuda-minimum-free-gib', [string] $CudaMinimumFreeGiB
        )
        Invoke-CheckedPython -PythonArguments @(
            'scripts\freeze_phase_c_budget.py',
            '--samples', $SampleManifest,
            '--model', $BaseModel,
            '--output', $BudgetManifest
        )
        return
    }

    Require-File -Path $BudgetManifest -Label 'Frozen Phase-C budget'

    if ($Stage -eq 'PhaseCCache') {
        New-Item -ItemType Directory -Path $GradientRoot -Force | Out-Null
        Invoke-CheckedPython -PythonArguments @(
            'scripts\cache_phase_c_gradients.py',
            '--budget-manifest', $BudgetManifest,
            '--corpus-root', $CorpusRoot,
            '--model', $BaseModel,
            '--cache', $GradientRoot,
            '--result', $GradientSummary,
            '--device', $Device,
            '--maximum-tokens', [string] $MaximumTokens
        )
        return
    }

    Require-File -Path $GradientSummary -Label 'Gradient-cache summary'
    $GradientStatus = Read-JsonObject -Path $GradientSummary
    if ($GradientStatus.status -ne 'complete') {
        throw "Gradient cache is not complete (status=$($GradientStatus.status))."
    }

    if ($Stage -eq 'PhaseCPreregister') {
        Invoke-CheckedPython -PythonArguments @(
            'scripts\freeze_phase_c_preregistration.py',
            '--budget', $BudgetManifest,
            '--sample-manifest', $SampleManifest,
            '--split-manifest', $SplitManifest,
            '--gradient-summary', $GradientSummary,
            '--gradient-cache', $GradientRoot,
            '--model', $BaseModel,
            '--corpus-root', $CorpusRoot,
            '--maximum-tokens', [string] $MaximumTokens,
            '--output', $Preregistration
        )
        $Frozen = Read-JsonObject -Path $Preregistration
        if (
            $Frozen.schema_version -ne '2.0' -or
            $Frozen.expected_run_cell_count -ne 200 -or
            $Frozen.expected_plan_cell_count -ne 175 -or
            @($Frozen.encoded_one_step_dev_by_stage.pilot).Count -ne 1 -or
            @($Frozen.encoded_one_step_dev_by_stage.qualification).Count -ne 4 -or
            $Frozen.maximum_tokens -ne $MaximumTokens -or
            @($Frozen.source_files.PSObject.Properties).Count -lt 1 -or
            @($Frozen.access_log).Count -ne 1 -or
            $Frozen.sealed_test_read -ne $false
        ) {
            throw 'Preregistration did not freeze the canonical 200-run/175-plan schema-v2 grid.'
        }
        Write-Host 'Root of trust generated. For an official run, review and commit PREREGISTRATION.json before building any qualification plan.'
        return
    }

    Require-File -Path $Preregistration -Label 'Committed schema-v2 preregistration root of trust'
    Assert-CommittedFile -Path $Preregistration

    if ($Stage -eq 'PhaseCPilot') {
        Invoke-CheckedPython -PythonArguments @(
            'scripts\build_phase_c_plans.py',
            '--budget-manifest', $BudgetManifest,
            '--preregistration', $Preregistration,
            '--gradient-cache', $GradientRoot,
            '--model', $BaseModel,
            '--plan-root', $PlanRoot,
            '--result', $PilotPlans,
            '--stage', 'pilot'
        )
        Invoke-CheckedPython -PythonArguments @(
            'scripts\run_phase_c.py',
            '--budget-manifest', $BudgetManifest,
            '--preregistration', $Preregistration,
            '--sample-manifest', $SampleManifest,
            '--split-manifest', $SplitManifest,
            '--gradient-summary', $GradientSummary,
            '--plan-result', $PilotPlans,
            '--gradient-cache', $GradientRoot,
            '--corpus-root', $CorpusRoot,
            '--model', $BaseModel,
            '--output', $PilotRun,
            '--decision', $PilotDecision,
            '--stage', 'pilot',
            '--device', $Device,
            '--maximum-tokens', [string] $MaximumTokens
        )
        return
    }

    if ($Stage -eq 'PhaseCQualification') {
        $Pilot = Read-JsonObject -Path $PilotDecision
        if ($Pilot.status -ne 'VALID_RESOURCE_PILOT') {
            throw "Qualification blocked: pilot status is $($Pilot.status)."
        }
        Invoke-CheckedPython -PythonArguments @(
            'scripts\build_phase_c_plans.py',
            '--budget-manifest', $BudgetManifest,
            '--preregistration', $Preregistration,
            '--gradient-cache', $GradientRoot,
            '--model', $BaseModel,
            '--plan-root', $PlanRoot,
            '--result', $QualificationPlans,
            '--stage', 'qualification'
        )
        $PlanEvidence = Read-JsonObject -Path $QualificationPlans
        if (
            $PlanEvidence.schema_version -ne '2.0' -or
            $PlanEvidence.status -ne 'complete' -or
            @($PlanEvidence.records).Count -ne 175
        ) {
            throw 'Qualification planner did not produce exactly 175 complete schema-v2 plan cells.'
        }
        Invoke-CheckedPython -PythonArguments @(
            'scripts\run_phase_c.py',
            '--budget-manifest', $BudgetManifest,
            '--preregistration', $Preregistration,
            '--sample-manifest', $SampleManifest,
            '--split-manifest', $SplitManifest,
            '--gradient-summary', $GradientSummary,
            '--plan-result', $QualificationPlans,
            '--gradient-cache', $GradientRoot,
            '--corpus-root', $CorpusRoot,
            '--model', $BaseModel,
            '--output', $QualificationRun,
            '--decision', $QualificationDecision,
            '--stage', 'qualification',
            '--device', $Device,
            '--maximum-tokens', [string] $MaximumTokens
        )
        $RunEvidence = Read-JsonObject -Path $QualificationRun
        if (
            $RunEvidence.schema_version -ne '2.0' -or
            $RunEvidence.status -ne 'complete' -or
            @($RunEvidence.records).Count -ne 200
        ) {
            throw 'Qualification runner did not produce exactly 200 complete schema-v2 run cells.'
        }
        return
    }

    if ($Stage -eq 'PhaseCGate') {
        Require-File -Path $QualificationRun -Label 'Complete qualification run'
        Require-File -Path $QualificationPlans -Label 'Complete qualification plans'
        & $script:RuntimePython @(
            'scripts\audit_phase_c_gate.py',
            '--run', $QualificationRun,
            '--plans', $QualificationPlans,
            '--preregistration', $Preregistration,
            '--budget', $BudgetManifest,
            '--sample-manifest', $SampleManifest,
            '--split-manifest', $SplitManifest,
            '--gradient-summary', $GradientSummary,
            '--gradient-cache', $GradientRoot,
            '--output', $Gate
        )
        if ($LASTEXITCODE -notin @(0, 2)) {
            throw "Gate audit failed operationally with exit code $LASTEXITCODE."
        }
        if ($LASTEXITCODE -eq 2) {
            Write-Host 'Scientific gate completed with STOP_NO_PHASE_D.'
        }
        return
    }

    if ($Stage -eq 'PhaseDGuard') {
        Require-File -Path $Gate -Label 'Phase-C advancement gate'
        Invoke-CheckedPython -PythonArguments @(
            'scripts\train_phase_d.py',
            '--gate', $Gate,
            '--run', $QualificationRun,
            '--plans', $QualificationPlans,
            '--preregistration', $Preregistration,
            '--budget', $BudgetManifest,
            '--sample-manifest', $SampleManifest,
            '--split-manifest', $SplitManifest,
            '--gradient-summary', $GradientSummary,
            '--gradient-cache', $GradientRoot
        )
        return
    }

    throw "Unhandled stage: $Stage"
}
finally {
    Set-Location -LiteralPath $PreviousLocation
}
