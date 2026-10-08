$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Deps = Join-Path $Root ".deps"
$Source = Join-Path $Deps "ninfer-v3"
$Repo = "https://github.com/Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco.git"
$Commit = "b06908ba3caa4f73269274fc7984b96f16d4295c"

New-Item -ItemType Directory -Force $Deps | Out-Null
if (-not (Test-Path (Join-Path $Source ".git"))) {
    git clone --filter=blob:none --no-checkout $Repo $Source
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone Ryan-gsq NInfer v3 source." }
}
git -C $Source cat-file -e "$Commit^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $Source fetch origin $Commit --depth 1
    if ($LASTEXITCODE -ne 0) { throw "Failed to fetch pinned NInfer v3 commit." }
}
$currentCommit = git -C $Source rev-parse --verify HEAD 2>$null
if ($currentCommit -ne $Commit) {
    $localChanges = git -C $Source status --porcelain
    if ($localChanges) { throw "NInfer source has local changes at another revision; preserve them before switching to $Commit." }
    git -C $Source checkout --detach $Commit
    if ($LASTEXITCODE -ne 0) { throw "Failed to checkout pinned NInfer v3 commit." }
}

if (-not (Test-Path (Join-Path $Source "tools\convert\gguf_blocks.py"))) {
    throw "Pinned source does not contain the GSQ/RCO block-preserving converter."
}
Write-Host "NInfer v3 converter source ready:"
Write-Host "  $Source"
Write-Host "  commit $Commit"
