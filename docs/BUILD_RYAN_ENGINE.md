# Build Ryan-gsq NInfer v3 natively on Windows

This is the preferred way to avoid the Quark download and avoid Docker/WSL.

## One command

```powershell
git pull
.\scripts\build-ryan-engine.ps1
```

The script:

1. checks out the pinned Ryan-gsq source at `b06908ba3caa4f73269274fc7984b96f16d4295c`;
2. discovers Visual Studio 2022 and imports the x64 developer environment;
3. prefers MSVC 14.44 when installed, otherwise uses the current VS2022 v143 toolset;
4. uses CUDA 13.x, preferring `v13.4`;
5. bootstraps a private vcpkg checkout under `.deps\vcpkg`;
6. configures a Release-only `sm_120a` / `NINFER_SM120_NATIVE=ON` build;
7. builds `ninfer_ops`;
8. runs `nvprune -arch sm_120a` before final linking to avoid the Windows PE-size problem;
9. links `ninfer-serve.exe`;
10. packages all required vcpkg, CUDA and VC-runtime DLLs under `runtime-v3\engine`;
11. strips the development PATH and runs `ninfer-serve.exe --help` to verify the packaged directory is standalone;
12. writes `runtime-v3\engine\build-manifest.json` including the executable SHA-256.

## Requirements

Only the one-time build needs:

- Visual Studio 2022 Build Tools/Community with **Desktop development with C++**
- CMake + Ninja (the Visual Studio CMake component is sufficient)
- Git
- CUDA Toolkit 13.x with `nvcc.exe` and `nvprune.exe`

Ryan validated:

- MSVC 14.44
- CUDA 13.4.2 / nvcc 13.4.92
- Native SM120a Release

The script will also attempt your existing CUDA 13.0 installation. If the Ryan source uses something that specifically requires 13.4, install CUDA 13.4.2 side-by-side and select it with:

```powershell
$env:NINFER_CUDA_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
.\scripts\build-ryan-engine.ps1 -Fresh
```

No existing CUDA installation needs to be removed.

## Memory-conscious build settings

The default is four parallel jobs:

```powershell
.\scripts\build-ryan-engine.ps1 -Jobs 4
```

This matches Ryan's documented 32 GB RAM build setup. If compilation memory pressure is high:

```powershell
.\scripts\build-ryan-engine.ps1 -Jobs 2
```

To discard the existing CMake tree and start clean:

```powershell
.\scripts\build-ryan-engine.ps1 -Fresh
```

## Output

```text
runtime-v3\
└── engine\
    ├── ninfer-serve.exe
    ├── cublas64_13.dll
    ├── cublasLt64_13.dll
    ├── cudart64_13.dll
    ├── FFmpeg/curl/vcpkg DLLs...
    ├── VC runtime DLLs...
    └── build-manifest.json
```

After this completes, Visual Studio, CUDA Toolkit, CMake and vcpkg are not needed for inference. The standalone engine loads its runtime DLLs from `runtime-v3\engine`.

## Normal use

```powershell
.\scripts\run-rentednoodle.ps1
```

If the runtime is missing, the v3 runner now invokes `build-ryan-engine.ps1` automatically.

The Quark package importer is retained only as an optional alternative:

```powershell
.\scripts\install-prebuilt-v3-runtime.ps1 -PackageRoot "C:\path\to\package"
```
