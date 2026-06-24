param(
    [string]$Task = "smoke",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "py"
}

function Invoke-PythonModule {
    param(
        [string]$Module,
        [string[]]$ModuleArgs = @()
    )

    if ($Python -eq "py") {
        & py -3.13 -m $Module @ModuleArgs
    } else {
        & $Python -m $Module @ModuleArgs
    }
}

function Invoke-Checked {
    param(
        [string]$Module,
        [string[]]$ModuleArgs = @()
    )

    Invoke-PythonModule $Module $ModuleArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Show-Help {
    Write-Output "rtk.ps1 - Charter repo toolkit"
    Write-Output ""
    Write-Output "Usage:"
    Write-Output "  .\rtk.ps1 <task> [args]"
    Write-Output "  .\rtk.ps1 -Task <task> [args]"
    Write-Output ""
    Write-Output "Tasks:"
    Write-Output "  help           Show this help"
    Write-Output "  setup          Create .venv and install -e .[dev]"
    Write-Output "  smoke          Run minimal smoke tests"
    Write-Output "  quick          Run fast pytest subset (-m 'not anyio and not slow')"
    Write-Output "  tui            Run Textual test slice"
    Write-Output "  test [args]    Run pytest (args forwarded)"
    Write-Output "  lint [args]    Run ruff check"
    Write-Output "  types [args]   Run mypy (default: src)"
    Write-Output "  check          Run ruff check + mypy src"
    Write-Output "  all [args]     Run pytest + ruff check + mypy src"
    Write-Output "  preflight      Agent preflight report"
    Write-Output "  scout [args]   Preflight scout bundle (--scout)"
    Write-Output "  boundaries     Fail if engine packages import game.ui"
    Write-Output "  review-packet  Print lightweight review packet (Markdown)"
    Write-Output "  doctor         Read-only handoff freshness checks"
    Write-Output ""
    Write-Output "Examples:"
    Write-Output '  .\rtk.ps1 preflight'
    Write-Output '  .\rtk.ps1 scout --task "One-line task"'
    Write-Output '  .\rtk.ps1 boundaries'
    Write-Output '  .\rtk.ps1 review-packet'
    Write-Output '  .\rtk.ps1 doctor'
}

switch ($Task) {
    "help" {
        Show-Help
    }
    "setup" {
        if (-not (Test-Path $VenvPython)) {
            & py -3.13 -m venv (Join-Path $ProjectRoot ".venv")
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            $Python = $VenvPython
        }
        Invoke-Checked "pip" @("install", "-e", ".[dev]")
    }
    "smoke" {
        Invoke-Checked "pytest" @("tests/test_main.py")
    }
    "quick" {
        Invoke-Checked "pytest" (@("-m", "not anyio and not slow") + $Rest)
    }
    "tui" {
        Invoke-Checked "pytest" (@("tests/test_tui.py") + $Rest)
    }
    "test" {
        Invoke-Checked "pytest" $Rest
    }
    "lint" {
        Invoke-Checked "ruff" (@("check") + $Rest)
    }
    "types" {
        if ($Rest.Count -eq 0) {
            Invoke-Checked "mypy" @("src")
        } else {
            Invoke-Checked "mypy" $Rest
        }
    }
    "check" {
        Invoke-Checked "ruff" @("check")
        Invoke-Checked "mypy" @("src")
    }
    "all" {
        Invoke-Checked "pytest" $Rest
        Invoke-Checked "ruff" @("check")
        Invoke-Checked "mypy" @("src")
    }
    "preflight" {
        Invoke-PythonModule "game.dev.agent_preflight" $Rest
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "boundaries" {
        Invoke-PythonModule "game.dev.check_engine_boundaries" $Rest
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "scout" {
        Invoke-PythonModule "game.dev.agent_preflight" (@("--scout") + $Rest)
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "review-packet" {
        Invoke-PythonModule "game.dev.agent_review_packet" $Rest
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "doctor" {
        Invoke-PythonModule "game.dev.agent_doctor" $Rest
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    default {
        Write-Error "Unknown task: $Task"
        Show-Help
        exit 2
    }
}
