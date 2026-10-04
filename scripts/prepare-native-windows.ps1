$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Engine = Join-Path $Root "engine"
$StubSource = Join-Path $Root "overlays\windows\nvfp4_tma_windows_stub.cpp"
$StubDest = Join-Path $Engine "src\ops\linear\nvfp4\nvfp4_tma_windows_stub.cpp"

if (-not (Test-Path (Join-Path $Engine "CMakeLists.txt"))) {
    & (Join-Path $Root "scripts\bootstrap-source.ps1")
}

if (-not (Test-Path $StubSource)) { throw "Missing Windows stub: $StubSource" }
Copy-Item $StubSource $StubDest -Force

# Strengthen the existing MSVC flags for CUDA-generated host translation units.
$RootCMake = Join-Path $Engine "CMakeLists.txt"
$Text = Get-Content $RootCMake -Raw
$Old = @'
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
'@ -join "`n"
$New = @'
  foreach(_flag /utf-8 /Zc:__cplusplus /Zc:preprocessor /bigobj)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:${_flag}>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>)
  endforeach()
'@ -join "`n"
if ($Text.Contains($Old)) {
    $Text = $Text.Replace($Old, $New)
}
Set-Content -Path $RootCMake -Value $Text -NoNewline

# Replace only the Blackwell non-RDC NVFP4 source selection. The prompt kernel
# remains real; the two CUtensorMap-by-value kernels become an MSVC-only stub.
$SrcCMake = Join-Path $Engine "src\CMakeLists.txt"
$Text = Get-Content $SrcCMake -Raw
$Pattern = '(?s)if\(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a"\)\s+add_library\(ninfer_nvfp4_non_rdc STATIC\s+ops/linear/nvfp4/nvfp4_w4a4_tma\.cu\s+ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma\.cu\s+ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc\.cu\)'
$Replacement = @'
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  if(MSVC)
    add_library(ninfer_nvfp4_non_rdc STATIC
      ops/linear/nvfp4/nvfp4_tma_windows_stub.cpp
      ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc.cu)
  else()
    add_library(ninfer_nvfp4_non_rdc STATIC
      ops/linear/nvfp4/nvfp4_w4a4_tma.cu
      ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu
      ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc.cu)
  endif()
'@ -join "`n"
if ($Text -match $Pattern) {
    $Text = [regex]::Replace($Text, $Pattern, $Replacement, 1)
} elseif (-not $Text.Contains("nvfp4_tma_windows_stub.cpp")) {
    throw "Could not patch src/CMakeLists.txt for the native Windows NVFP4 stub."
}
Set-Content -Path $SrcCMake -Value $Text -NoNewline

Write-Host "Native Windows source patches are applied."
