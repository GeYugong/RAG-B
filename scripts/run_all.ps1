param(
    [Parameter(Mandatory = $true)]
    [string]$DeepseekApiKey,

    [Parameter(Mandatory = $true)]
    [string]$EmbeddingApiKey,

    [string]$EmbeddingBaseUrl = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    [string]$EmbeddingModel = "text-embedding-v4",
    [string]$GenerateBaseUrl = "https://api.deepseek.com/v1",
    [string]$GenerateModel = "deepseek-chat",
    [int]$EmbeddingBatchSize = 10,
    [switch]$SkipVenv,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not $SkipVenv) {
    if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
        python -m venv .venv
    }
    . .\\.venv\\Scripts\\Activate.ps1
}

if (-not $SkipInstall) {
    python -m pip install -r requirements.txt
}

$env:DEEPSEEK_API_KEY = $DeepseekApiKey
$env:GENERATE_BASE_URL = $GenerateBaseUrl
$env:GENERATE_MODEL = $GenerateModel

$env:EMBEDDING_API_KEY = $EmbeddingApiKey
$env:EMBEDDING_BASE_URL = $EmbeddingBaseUrl
$env:EMBEDDING_MODEL = $EmbeddingModel
$env:EMBEDDING_BATCH_SIZE = "$EmbeddingBatchSize"

Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue

Write-Host "[1/3] healthcheck..."
python src/healthcheck.py

Write-Host "[2/3] ingest..."
python src/ingest.py

Write-Host "[3/3] rag..."
python src/rag.py
