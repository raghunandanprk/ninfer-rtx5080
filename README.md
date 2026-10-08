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

### 2. Obtain the model artifacts

#### Option A: Download verified quants directly (Fastest)

Download pre-converted, verified SM120a compatible artifacts from Hugging Face ([`raghualgt/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer`](https://huggingface.co/raghualgt/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-NInfer)):

```powershell
# Downloads all verified artifacts (Text MTP, DFlash2, and Vision)
.\scripts\download-ryan-compatible-models.ps1

# Or download only text models (no vision)
.\scripts\download-ryan-compatible-models.ps1 -Variant text
```

Artifacts downloaded into `models\rentednoodle-native-v3\`:
- `qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer` (10.06 GiB · Coding/Research/Long-context)
- `qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer` (12.13 GiB · Fast Chat ~117 tok/s)
- `qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer` (10.33 GiB · Image/Video Multimodal)

#### Option B: Download and convert from GGUF

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

### 3. Get the native Windows engine

> [!NOTE]
> **Why `runtime-v3/` is not tracked directly in git:**  
> The compiled Blackwell engine directory `runtime-v3\engine\` is ~1.62 GB uncompressed (`ninfer-serve.exe` is ~1.05 GB; `cublasLt64_13.dll` is ~493 MB), which exceeds GitHub's 100 MB per-file push limit. To avoid repository bloat and bandwidth constraints, the standalone engine is packaged and hosted as a public release asset.

#### Option A: Install prebuilt standalone engine (Recommended · No compiler needed)

The prebuilt Blackwell `sm_120a` engine package is available publicly for everyone:

- **Direct Download Link:** [`ninfer-sm120a-engine.zip`](https://github.com/raghunandanprk/ninfer-rtx5080/releases/download/v0.3.0/ninfer-sm120a-engine.zip) (~943 MB compressed, 1.62 GB extracted)
- **Release Information:** [GitHub Release v0.3.0](https://github.com/raghunandanprk/ninfer-rtx5080/releases/tag/v0.3.0)
- **SHA-256 Checksum:** `beb24ea5b627328708309d32490a6c1d745c6dc9d1dec7bf8481fed7ab9adb4e`
- **Included in package:** Standalone compiled `ninfer-serve.exe` (SM120a / Blackwell optimized) and full dynamic runtime dependencies (`cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll`, `msvcp140.dll`, `vcruntime140.dll`, etc.). No CUDA toolkit or Visual Studio C++ installation is required.

**Automated 1-command install:**

```powershell
.\scripts\install-prebuilt-v3-runtime.ps1
```

*(This automatically downloads `ninfer-sm120a-engine.zip` from Release v0.3.0, verifies it, and extracts it directly into `runtime-v3\engine\`.)*

**Manual install:**
1. Download [`ninfer-sm120a-engine.zip`](https://github.com/raghunandanprk/ninfer-rtx5080/releases/download/v0.3.0/ninfer-sm120a-engine.zip).
2. Extract the archive contents into `runtime-v3\engine\`.
3. Confirm that `runtime-v3\engine\ninfer-serve.exe` is present.

#### Option B: Build from source locally

If you have Visual Studio 2022 C++ and CUDA Toolkit 13.x installed and wish to compile locally:

```powershell
.\scripts\build-ryan-engine.ps1
```

This builds Ryan-gsq NInfer v3 as Release `sm_120a`, prunes redundant PTX before linking,
and assembles a standalone `runtime-v3\engine` directory containing `ninfer-serve.exe`
and all required runtime DLLs.

See [docs/BUILD_RYAN_ENGINE.md](docs/BUILD_RYAN_ENGINE.md).

### 4. Run

```powershell
.\scripts\run-rentednoodle.ps1
```

The normal command now selects the v3 pipeline automatically. If `runtime-v3\engine\ninfer-serve.exe`
does not exist yet, run `.\scripts\install-prebuilt-v3-runtime.ps1`.

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

### 5. Workload-specific launchers (`inf-cmd/`)

Pre-tuned launchers calibrated on the 16 GB RTX 5080 Laptop for maximum context and decode throughput:

| Workload | Command | Port | Model / Speculation | Context | Benchmark Speed |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Chat** | `.\inf-cmd\chat.ps1` | 8091 | DFlash2 K=5 | 121,856 | ~117 tok/s (short) |
| **Coding** | `.\inf-cmd\coding.ps1` | 8092 | MTP3 + n-gram | 229,376 | ~100 tok/s |
| **Research** | `.\inf-cmd\research.ps1` | 8093 | MTP3 | 231,424 | 80 tok/s @ 99K depth |
| **Image** | `.\inf-cmd\image.ps1` | 8094 | Vision (VRAM-resident) | 147,456 | ~1.4s encode |
| **Video** | `.\inf-cmd\video.ps1` | 8095 | Vision (RAM overlay) | 169,984 | 4 fps downsample |

Each launcher manages port binding, memory policies, spec draft depths, and reasoning parameters automatically. See [inf-cmd/user-guide.md](inf-cmd/user-guide.md) for full documentation.

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
