# V3 pivot: preserved GSQ/RCO blocks

## Why the project changed

The original prototype converted the RentedNoodle GGUF by dequantizing mixed GGUF tensors and
re-encoding them into NInfer's older Q3/Q4/Q5 row-split formats. That worked as a portability
strategy, but it unnecessarily changed the quantization.

Ryan-gsq's NInfer v3 line already contains a Qwen3.8 GGUF-block runtime and converter:

`qwen3_8_27b_gguf`

For text and embedded MTP tensors, it imports the GGUF block encoding rather than requantizing it.
This makes it a better match for RentedNoodle's OrcaRouter-native GSQ/RCO quant.

## Pinned converter source

Repository:

`Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco`

Pinned commit:

`b06908ba3caa4f73269274fc7984b96f16d4295c`

The runtime package itself is not stored in this repository. Ryan-gsq distributes its precompiled
Windows SM120a engine externally; `install-prebuilt-v3-runtime.ps1` imports the engine directory
from an extracted package.

## RentedNoodle input

Default model:

`Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.1.gguf`

Expected SHA-256:

`ab955b5083d9cdf0bf55c37acdcae359b78756c4544d97960c23d8fca98feb9b`

Vision:

`mmproj/mmproj-Qwen3.8-27B-BF16.gguf`

The converter receives the RentedNoodle Froggeric template, tokenizer/config metadata and image
preprocessor. Qwen's official `generation_config.json` and `video_preprocessor_config.json`
fill the two frontend-resource roles that are not distributed in the RentedNoodle repository.

## Conversion command represented by the script

```powershell
python -u -m tools.convert \
  --model <rentednoodle-metadata> \
  --recipe qwen3_8_27b_gguf \
  --source "gguf=<RentedNoodle-v2.1.gguf>" \
  --source "vision=<RentedNoodle-BF16-mmproj.gguf>" \
  --components text,vision,mtp \
  --proposal \
  --device cpu \
  --rows-per-chunk 512 \
  --name qwen3.8-27b-rentednoodle-orcarouter-v2.1 \
  --out <artifact.ninfer>
```

The script uses PowerShell line continuation rather than the shell syntax shown above.

## What is preserved

### Text and MTP

The converter validates the Qwen3.8-27B dense GGUF geometry and copies each supported quantized
row in its existing ggml block representation. Tensor-specific RCO precision choices therefore
survive conversion.

### Vision

Vision is different. The RentedNoodle BF16 mmproj is mapped through NInfer's verified Qwen3.8
vision-name map, including reconstruction of the split temporal patch embedding. Vision tensors
are then stored in NInfer's official Vision formats. This is intentional; the preserved-block
path applies to text/MTP GGUF blocks, while Vision uses the supported companion route.

### Proposal head

`--proposal` adds NInfer's indexed proposal head for optional `--lm-head-draft` experiments.
The default Ryan-style MTP profile does not require `--lm-head-draft`.

## Runtime policy

Default v3 launch:

- native Windows precompiled SM120a runtime
- no Docker
- no WSL
- Vision enabled
- 98,304 context
- rk8v4 KV
- adaptive MTP with up to 4 drafts
- 1 GiB host cache
- one request at a time

The lower Host cache is deliberate for 32 GB system-RAM machines. Ryan's packaged profiles use a
larger Host cache for long-running session reuse; increase `NINFER_HOST_CACHE_MIB` if that
feature matters more than RAM pressure.

Text-only switches to 131,072 context and strict dedicated-VRAM policy by default. These values are
starting profiles, not validated RTX 5080 Laptop limits. Measure actual VRAM headroom before raising
them.

## Legacy

The old custom row-split conversion remains under the older scripts so artifacts and experiments
can still be reproduced. It is no longer the recommended path.
