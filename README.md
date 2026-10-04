# NInfer RTX 5080 · RentedNoodle GSQ/RCO v3

> **Current default:** Ryan-gsq NInfer v3 preserved-block pipeline.  
> The older roofkid Q3/Q4/Q5 requantization path is retained only as a legacy fallback.

This project now targets:

- **RentedNoodle OrcaRouter Qwen3.8-27B IQ3_XXS v2.1**
- original mixed GSQ/RCO GGUF blocks preserved byte-for-byte where supported
- embedded RentedNoodle MTP head preserved
- **RentedNoodle BF16 mmproj Vision** imported through NInfer's verified Qwen3.8 vision mapping
- NInfer v3 proposal head
- Ryan-gsq **Native SM120a Windows runtime**
- RTX 5080 / 5070 Ti / 5090
- no Docker or WSL in the default path

## Recommended workflow

### 1. Update this repository

```powershell
git pull
```

### 2. Download and convert RentedNoodle

```powershell
.\scripts\download-rentednoodle-v3.ps1
.\scripts\convert-rentednoodle-v3.ps1
```

The converter is pinned to Ryan-gsq NInfer commit
`b06908ba3caa4f73269274fc7984b96f16d4295c` and uses:

```text
recipe:      qwen3_8_27b_gguf
components:  text,vision,mtp
text source: RentedNoodle GSQ/RCO GGUF
vision:      RentedNoodle BF16 mmproj
proposal:    enabled
device:      CPU
```

The language/MTP GSQ/RCO blocks are imported rather than dequantized and re-quantized.

### 3. Build the native Windows engine once

Quark is no longer part of the recommended path:

```powershell
.\scripts\build-ryan-engine.ps1
```

This builds Ryan-gsq NInfer v3 as Release `sm_120a`, prunes redundant PTX before linking,
and assembles a standalone `runtime-v3\engine` directory containing `ninfer-serve.exe`
and all required runtime DLLs.

The script prefers Ryan's validated CUDA 13.4 setup but will attempt your existing CUDA 13.0
installation first. CUDA 13.4.2 can be installed side-by-side if 13.0 proves insufficient.

See [docs/BUILD_RYAN_ENGINE.md](docs/BUILD_RYAN_ENGINE.md).

The old Quark package importer remains optional if you ever obtain the package:

```powershell
.\scripts\install-prebuilt-v3-runtime.ps1 -PackageRoot "C:\path\to\extracted\qwen27b"
```

### 4. Run

```powershell
.\scripts\run-rentednoodle.ps1
```

The normal command now selects the v3 pipeline automatically. If `runtime-v3\engine\ninfer-serve.exe`
does not exist yet, it automatically invokes `build-ryan-engine.ps1`.

Defaults:

```text
Vision:          ON
Context:         98,304 tokens
KV:              rk8v4
Prefill chunk:   1024
MTP:             adaptive, max 4 drafts
Host cache:      1024 MiB
Concurrency:     1
API:             http://127.0.0.1:8080/v1
```

For text-only:

```powershell
$env:NINFER_VISION="0"
.\scripts\run-rentednoodle.ps1
```

Text-only defaults to 131,072 context and the engine's strict dedicated-VRAM policy.

Useful overrides:

```powershell
$env:NINFER_CONTEXT="102400"
$env:NINFER_KV_DTYPE="rk4v4-e8"
$env:NINFER_HOST_CACHE_MIB="512"
$env:NINFER_PORT="8100"
.\scripts\run-rentednoodle.ps1
```

## Legacy pipeline

The previous custom converter remains available for comparison only:

```powershell
$env:NINFER_PIPELINE="legacy"
.\scripts\run-rentednoodle.ps1
```

Docker/WSL legacy fallback:

```powershell
$env:NINFER_PIPELINE="legacy-docker"
.\scripts\run-rentednoodle.ps1
```

See [docs/V3_PIVOT.md](docs/V3_PIVOT.md) for architecture and provenance details.

---

## Legacy development history


Blackwell (`sm_120a`) bring-up for **Qwen3.8-27B GSQ3 on 16 GB RTX 5080 / RTX 5070 Ti**.

This project ports the GSQ3 work from [`roofkid/ninfer-4080`](https://github.com/roofkid/ninfer-4080) to Blackwell. The upstream 4080 fork already contains the important pieces: `Q3G128_F16S`, Q3 decode/prefill kernels, `rk4v4-e8` KV, MTP3, DFlash2 K=7, long-context serving, and the OpenAI/Anthropic-compatible server. Its published build is hard-gated to Ada `sm_89`; this repo retargets that gate to `sm_120a`.

## Target configuration

- GPU: RTX 5080 Laptop 16 GB / RTX 5080 16 GB / RTX 5070 Ti 16 GB
- CUDA architecture: `sm_120a`
- Model: Qwen3.8-27B
- Text-body quant: GSQ3, `Q3G128_F16S`, 3.125 bpw
- Artifact: `qwen3_8_27b_gsq3.ninfer`, 12.41 GiB
- KV: `rk4v4-e8`
- Speculation: MTP3 or DFlash2 K=7
- Reference context target: ~100K

## Quick start on Windows 11

Requirements: Git, Docker Desktop with WSL2 GPU support, and a recent NVIDIA driver.

```powershell
git clone https://github.com/raghunandanprk/ninfer-rtx5080.git
cd ninfer-rtx5080

.\scripts\bootstrap-source.ps1
.\scripts\download-model.ps1
.\scripts\build-docker.ps1
.\scripts\run-mtp.ps1
```

The OpenAI-compatible API is then available at:

```text
http://127.0.0.1:8080/v1
```

For the faster DFlash2 path:

```powershell
.\scripts\run-dflash2.ps1
```

Useful overrides:

```powershell
$env:NINFER_CONTEXT="65536"
$env:NINFER_PORT="8100"
$env:NINFER_VISION="0"   # MTP script defaults vision on; set 0 for text-only
.\scripts\run-mtp.ps1
```

For DFlash2, vision is off by default. Set `$env:NINFER_VISION="1"` to enable it; the script then defaults to 65,536 context.

## Model artifact

Source:

`roofkid/Qwen3.8-27B-GSQ3-NInfer`

Expected SHA-256:

```text
c6f27073393e5bcc629489420470d71f52a27553bfc5c360fef07a25b3b550d7
```

`download-model.ps1` verifies the hash before use.

## Upstream reference numbers

The RTX 4080 fork reports up to roughly 2.7K tok/s prefill and 262 tok/s DFlash2 decode on its benchmark sweep, with about 100K context on a 16 GB card. Those are **reference measurements only**. This repo will not call them RTX 5080 results until the same tests are run on actual Blackwell hardware.

See [`PORTING_STATUS.md`](PORTING_STATUS.md) for the validation checklist.

## Source strategy

`bootstrap-source.ps1` clones the pinned upstream GSQ3 runtime into `engine/` and patches the top-level CUDA architecture gate from `sm_89` to `sm_120a`. `engine/` is intentionally git-ignored so the source provenance stays explicit and refreshing the pinned upstream remains deterministic.

The optional workflow under `.github/workflows/import-upstream.yml` can be used later if the full upstream tree should be vendored into this repository.

## Credits

- `roofkid/ninfer-4080` — GSQ3, Q3 kernel work, DFlash2 profile and 4080 tuning
- `YukinoKaorisuna/ninfer-5070ti` / `toddballinger/ninfer-5080` — 16 GB Blackwell bring-up references
- `Neroued/ninfer` — original NInfer engine
- `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` — GSQ3 source checkpoint

Apache-2.0 upstream licensing applies to the imported runtime and model components as documented by their respective projects.

## Native Windows (recommended on 16 GB + 32 GB RAM)

Docker/WSL is no longer required for the RentedNoodle path. Native Windows is the default:

```powershell
git pull
.\scripts\build-native-windows.ps1
.\scripts\run-rentednoodle.ps1
```

The native runner caps Host-KV at 512 MiB instead of NInfer's 8192 MiB server default and avoids
the Docker Desktop / WSL2 VM memory overhead. Set `NINFER_RUNTIME=docker` only when you explicitly
want the Docker fallback.

See [docs/NATIVE_WINDOWS.md](docs/NATIVE_WINDOWS.md) for prerequisites, RAM-conscious settings,
Vision/text context profiles and overrides.

## RentedNoodle uncensored GSQ/RCO build

The repo now includes a dedicated converter for:

`RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored`

Pinned source:

- revision: `main`
- file: `Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf`
- source size: about 9.75 GiB / 3.06 bpw
- SHA-256: `41ad7dfb3f4397d626408a96e88a46c5964e88bd6c4240c191e1131af92ea8cd`

The converter uses the RentedNoodle GGUF for all ported trunk matrices and its embedded
`blk.64` MTP head, and uses RentedNoodle's own `mmproj/mmproj-Qwen3.8-27B-BF16.gguf` for all
333 NInfer Vision objects. The NInfer GSQ3 artifact remains a donor only for components not
shipped by RentedNoodle, principally DFlash2 and the draft shortlist IDs. The optimized NInfer
draft head is regenerated from the RentedNoodle output head.

Build it with:

```powershell
.\scripts\bootstrap-source.ps1
.\scripts\download-rentednoodle.ps1
.\scripts\download-model.ps1
.\scripts\convert-rentednoodle.ps1
```

Then run:

```powershell
.\scripts\run-rentednoodle.ps1
```

The conversion emits both the `.ninfer` artifact and a `.conversion.json` provenance report.

Important: DFlash2 is copied from the aligned GSQ3 donor and is therefore not assumed to be
lossless against the OrcaRouter-modified trunk until acceptance is benchmarked. MTP is the
preferred first validation path because the RentedNoodle v2.1 GGUF carries its own custom MTP head.
