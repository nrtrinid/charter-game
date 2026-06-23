param(
    [ValidateSet("smoke", "quick", "tui", "test", "lint", "types", "check", "all", "setup", "preflight", "boundaries", "scout")]
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

switch ($Task) {
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
}
