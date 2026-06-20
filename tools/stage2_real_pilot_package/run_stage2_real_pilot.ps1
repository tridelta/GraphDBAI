param(
    [string]$RunDir = "runs",
    [string]$Prefix = "stage2_pilot_real_s2e_",
    [int]$Seed = 2501,
    [int]$Episodes = 5,
    [int]$MaxSteps = 12,
    [int]$Window = 2,
    [double]$MaxBudgetRmb = 50,
    [string]$Difficulty = "medium",
    [string]$CaseSchedule = "shuffled_cycle",
    [string]$Model = "deepseek-v4-flash",
    [int]$LlmMaxTokens = 4096,
    [int]$LlmRetryMaxTokens = 8192,
    [int]$LlmRetries = 1,
    [switch]$Parallel,
    [int]$MaxWorkers = 3,
    [switch]$AnalyzeOnly,
    [switch]$SkipValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$env:PYTHONPATH = "src"
$env:EXPERIENCE_GRAPH_LLM_PROVIDER = "deepseek"
$env:DEEPSEEK_MODEL = $Model
$env:EG_LLM_MAX_TOKENS = "$LlmMaxTokens"
$env:EG_LLM_RETRY_MAX_TOKENS = "$LlmRetryMaxTokens"
$env:EG_LLM_RETRIES = "$LlmRetries"

if (-not $AnalyzeOnly) {
    if (-not $env:DEEPSEEK_API_KEY) {
        throw "DEEPSEEK_API_KEY is not set. Set it in this terminal before running the real pilot."
    }
}

$Conditions = @(
    @{ Label = "react"; Agent = "react"; Variant = "full" },
    @{ Label = "reflexion"; Agent = "reflexion"; Variant = "full" },
    @{ Label = "vector_trajectory"; Agent = "vector_trajectory"; Variant = "full" },
    @{ Label = "skill_library"; Agent = "skill_library"; Variant = "full" },
    @{ Label = "graph_full"; Agent = "graph"; Variant = "full" },
    @{ Label = "graph_no_graph_context"; Agent = "graph"; Variant = "no_graph_context" }
)

function New-Stage2Args {
    param([hashtable]$Condition)

    $Label = $Condition.Label
    $Agent = $Condition.Agent
    $Variant = $Condition.Variant
    $RunId = "${Prefix}${Label}_${Difficulty}_seed${Seed}"
    $RunPath = Join-Path $RunDir $RunId
    $Args = @(
        "-B", "-m", "experience_graph.scripts.run_experiment",
        "--agent", $Agent,
        "--variant", $Variant,
        "--difficulty", $Difficulty,
        "--episodes", "$Episodes",
        "--max-steps", "$MaxSteps",
        "--seed", "$Seed",
        "--case-schedule", $CaseSchedule,
        "--llm-provider", "deepseek",
        "--llm-model", $Model,
        "--llm-max-tokens", "$LlmMaxTokens",
        "--llm-retry-max-tokens", "$LlmRetryMaxTokens",
        "--llm-retries", "$LlmRetries",
        "--max-budget-rmb", "$MaxBudgetRmb",
        "--run-dir", $RunDir,
        "--run-id", $RunId
    )

    if (Test-Path (Join-Path $RunPath "config.yaml")) {
        Write-Host "Resuming existing run: $RunId"
        $Args += "--resume"
    } else {
        Write-Host "Starting new run: $RunId"
    }
    return $Args
}

function Invoke-Stage2Run {
    param([hashtable]$Condition)
    $Args = New-Stage2Args -Condition $Condition
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for condition=$($Condition.Label) with exit code $LASTEXITCODE"
    }
}

function Wait-ForProcessSlot {
    param([array]$Records, [int]$Limit)
    while (@($Records | Where-Object { -not $_.Process.HasExited }).Count -ge $Limit) {
        Start-Sleep -Seconds 1
    }
}

function Write-ProcessLog {
    param([hashtable]$Record)
    if (Test-Path $Record.Stdout) {
        Write-Host "--- stdout: $($Record.Label) ---"
        Get-Content -Path $Record.Stdout
    }
    if (Test-Path $Record.Stderr) {
        $ErrText = Get-Content -Path $Record.Stderr
        if ($ErrText) {
            Write-Host "--- stderr: $($Record.Label) ---"
            $ErrText
        }
    }
}

if (-not $AnalyzeOnly) {
    if ($Parallel) {
        $ProcessLogDir = Join-Path $RunDir "${Prefix}process_logs"
        New-Item -ItemType Directory -Force -Path $ProcessLogDir | Out-Null
        $Records = @()
        foreach ($Condition in $Conditions) {
            Wait-ForProcessSlot -Records $Records -Limit $MaxWorkers
            $Args = New-Stage2Args -Condition $Condition
            $Stdout = Join-Path $ProcessLogDir "$($Condition.Label).out.log"
            $Stderr = Join-Path $ProcessLogDir "$($Condition.Label).err.log"
            Write-Host "Launching process: $($Condition.Label)"
            $Process = Start-Process -FilePath "python" -ArgumentList $Args -WorkingDirectory $RepoRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
            $Records += @{ Label = $Condition.Label; Process = $Process; Stdout = $Stdout; Stderr = $Stderr }
        }
        $Failed = $false
        foreach ($Record in $Records) {
            Wait-Process -Id $Record.Process.Id
            $ExitedProcess = Get-Process -Id $Record.Process.Id -ErrorAction SilentlyContinue
            if ($null -eq $ExitedProcess) {
                $Record.Process.Refresh()
            }
            Write-ProcessLog -Record $Record
            $ExitCode = $Record.Process.ExitCode
            if ($null -eq $ExitCode) {
                $ExitCode = 0
            }
            if ($ExitCode -ne 0) {
                $Failed = $true
                Write-Error "Process $($Record.Label) exited with code $ExitCode"
            }
        }
        if ($Failed) {
            throw "One or more parallel Stage 2 runs failed. See $ProcessLogDir for logs."
        }
    } else {
        foreach ($Condition in $Conditions) {
            Invoke-Stage2Run -Condition $Condition
        }
    }
}

$AnalysisDir = Join-Path $RunDir "${Prefix}analysis"
& python -B tools\analyze_stage2.py --run-dir $RunDir --prefix $Prefix --output-dir $AnalysisDir --window $Window
if ($LASTEXITCODE -ne 0) {
    throw "Analysis failed with exit code $LASTEXITCODE"
}

$ReportPath = Join-Path $AnalysisDir "stage2_pilot_report.md"
& python -B tools\build_stage2_report.py --analysis-dir $AnalysisDir --output $ReportPath --viewer tools\episode_log_viewer.html
if ($LASTEXITCODE -ne 0) {
    throw "Report generation failed with exit code $LASTEXITCODE"
}

if (-not $SkipValidation) {
    $ValidationCode = @"
import json
from pathlib import Path

run_dir = Path(r'$RunDir')
prefix = '$Prefix'
episodes = $Episodes
seed = $Seed
difficulty = '$Difficulty'
conditions = [('react', 'react', 'full'), ('reflexion', 'reflexion', 'full'), ('vector_trajectory', 'vector_trajectory', 'full'), ('skill_library', 'skill_library', 'full'), ('graph_full', 'graph', 'full'), ('graph_no_graph_context', 'graph', 'no_graph_context')]
required = [
    'config.yaml',
    'metrics.jsonl',
    'steps.jsonl',
    'experience_views.jsonl',
    'graph_nodes.jsonl',
    'graph_edges.jsonl',
    'path_records.jsonl',
]
errors = []
for label, agent, variant in conditions:
    run = run_dir / f'{prefix}{label}_{difficulty}_seed{seed}'
    missing = [name for name in required if not (run / name).exists()]
    if missing:
        errors.append(f'{run.name}: missing {missing}')
        continue
    config_text = (run / 'config.yaml').read_text(encoding='utf-8')
    for needle in ['llm_max_tokens: $LlmMaxTokens', 'llm_retry_max_tokens: $LlmRetryMaxTokens', 'llm_retries: $LlmRetries', f'agent: {agent}', f'variant: {variant}']:
        if needle not in config_text:
            errors.append(f'{run.name}: config missing {needle}')
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    steps = [json.loads(line) for line in (run / 'steps.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    hidden = sum(1 for step in steps if (step.get('prompt_diagnostics') or {}).get('prompt_hidden_facts'))
    if len(metrics) != episodes:
        errors.append(f'{run.name}: expected {episodes} metrics rows, got {len(metrics)}')
    if hidden:
        errors.append(f'{run.name}: prompt_hidden_facts count is {hidden}')

fig_dir = run_dir / f'{prefix}analysis' / 'figures'
for name in ['learning_curve_medium.png', 'graph_growth.png', 'token_cost.png']:
    path = fig_dir / name
    if not path.exists():
        errors.append(f'missing figure {path}')
        continue
    data = path.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        errors.append(f'not a PNG file: {path}')

if errors:
    print('VALIDATION FAILED')
    for error in errors:
        print('-', error)
    raise SystemExit(1)
print('VALIDATION OK')
"@
    $ValidationCode | python -B -
    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Stage 2 real pilot package completed."
Write-Host "Analysis directory: $AnalysisDir"
Write-Host "Report: $ReportPath"
