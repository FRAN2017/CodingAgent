[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task,

    [string]$Workspace = ".",

    [ValidateRange(1, 100)]
    [int]$MaxSteps = 10,

    [ValidateSet("deepseek", "qianwen")]
    [string]$Provider = "deepseek",

    [string]$Session,

    [string]$EnvFile = (Join-Path $PSScriptRoot ".env"),

    [switch]$CheckConfig
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-DotEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Environment file does not exist: $Path"
    }

    foreach ($rawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $rawLine.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith("#")) {
            continue
        }

        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).TrimStart()
        }

        $separator = $line.IndexOf("=")
        if ($separator -le 0) {
            throw "Invalid .env line (expected NAME=VALUE)."
        }

        $name = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()

        if ($name -notmatch "^[A-Za-z_][A-Za-z0-9_]*$") {
            throw "Invalid environment variable name in .env: $name"
        }

        $hasDoubleQuotes = $value.Length -ge 2 -and
            $value.StartsWith('"') -and $value.EndsWith('"')
        $hasSingleQuotes = $value.Length -ge 2 -and
            $value.StartsWith("'") -and $value.EndsWith("'")
        if ($hasDoubleQuotes -or $hasSingleQuotes) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

Import-DotEnv -Path $EnvFile

$requiredKeyName = if ($Provider -eq "qianwen") {
    "QIANWEN_API_KEY"
}
else {
    "DEEPSEEK_API_KEY"
}
$requiredKeyValue = [Environment]::GetEnvironmentVariable(
    $requiredKeyName,
    "Process"
)
if ([string]::IsNullOrWhiteSpace($requiredKeyValue)) {
    throw "$requiredKeyName is missing or empty in $EnvFile"
}

if ($CheckConfig) {
    Write-Host "$Provider configuration loaded successfully." -ForegroundColor Green
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Task)) {
    throw "Task is required. Example: .\run-agent.ps1 'Summarize this project'"
}

$virtualEnvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $virtualEnvPython -PathType Leaf) {
    $python = $virtualEnvPython
}
else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "Python was not found. Create .venv or add Python to PATH."
    }
    $python = $pythonCommand.Source
}

$agentArguments = @(
    "-m",
    "coding_agent",
    "run",
    $Task,
    "--workspace",
    $Workspace,
    "--max-steps",
    $MaxSteps,
    "--provider",
    $Provider
)
if (-not [string]::IsNullOrWhiteSpace($Session)) {
    $agentArguments += @("--session", $Session)
}

& $python @agentArguments
exit $LASTEXITCODE
