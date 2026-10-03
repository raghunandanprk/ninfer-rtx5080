# NInfer RTX 5080 GSQ3

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
