$ErrorActionPreference = 'Stop'
if (Get-Command nasm -ErrorAction SilentlyContinue) { exit 0 }
choco install nasm --yes --no-progress
if ($LASTEXITCODE -ne 0) { throw 'NASM installation failed' }
$nasm = Join-Path $env:ProgramFiles 'NASM'
if (-not (Test-Path (Join-Path $nasm 'nasm.exe'))) { throw 'NASM executable is absent' }
$nasm | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
