# RTX 5080 Laptop tuning: max context and speed

**Date:** 2026-10-05 · **Data:** [tuning.json](tuning.json) (text), [vision/vision.json](vision/vision.json) (vision) ·
**Scripts:** [scripts/tune-ryan-engine.py](../../scripts/tune-ryan-engine.py), [scripts/tune-vision.py](../../scripts/tune-vision.py)

This run looked for the largest context window and the fastest decode one RTX 5080 Laptop GPU
can hold for a single stream of RentedNoodle Qwen3.8-27B on the native NInfer v3 engine, first for
text and then for images and video ([Vision](#vision-images-and-video)).
Earlier runs used only `bf16` KV. The largest context that worked there was 16K, and 100K failed
to start ([previous results](../2026-10-05-rtx5080/benchmark.json)).

**Bottom line, text:** MTP speculation with 3 drafts and `rk4v4` KV holds **231,424 tokens (226K)**.
At that size it retrieved all three planted codes from a 229,375-token prompt. Decode is about
100 tok/s on short prompts, 80 tok/s at 99K depth and 60 tok/s at full depth.

**Bottom line, vision:** the same MTP3 setup with the Vision tower kept in VRAM answers about an
image in 1.4–1.5 s at 93 tok/s, with a 147,456-token window. Keeping the tower in system RAM and
encoding it on the GPU per request (`overlay`) costs about 1 s on a video and raises the window to
169,984. Encoding on the CPU is not viable: 56 s per image at full resolution.

Per-use-case launchers built from these results live in [inf-cmd/](../../inf-cmd/): `chat.ps1`,
`coding.ps1`, `research.ps1`, `image.ps1` and `video.ps1`, plus `prepare-video.ps1`.

## Recommendation

| Use | Configuration | Context | Why |
|---|---|---:|---|
| **Default: max context** | MTP3, `rk4v4`, no proposal head | **231,424** | Largest window; only 4–5% slower than with the head on short prompts |
| Slightly faster, still long | MTP3 + `--lm-head-draft`, `rk4v4` | 212,992 | 4–5% faster on short prompts, about 2% at depth; 18K fewer tokens |
| Short chats only | DFlash2 K=5, `rk4v4`, no proposal head | 121,856 | 10–14% faster than MTP on short prompts; 15% slower by 99K |

Recommended command, verified to start and answer with these exact flags:

```powershell
$env:NINFER_PREFILL_ALIGN="0"
.\runtime-v3\engine\ninfer-serve.exe "E:\llm\RentedNoodle-NInfer-v3\qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer" `
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b-rentednoodle-orcarouter `
  --max-context 231424 --kv-capacity 231424 --max-concurrency 1 --default-max-tokens 0 `
  --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 1024 `
  --cuda-memory-policy strict --host-cache-mib 5120 --spec mtp --draft-tokens 3 `
  --default-reasoning-effort xhigh --preserve-thinking --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

To switch to another configuration:

- **Slightly faster variant:** add `--lm-head-draft` and set both context values to `212992`.
- **DFlash2:** use `qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer`, `--spec dflash2 --draft-tokens 5` and `121856`.

`scripts/run-rentednoodle-v3.ps1` cannot express these configurations yet. It always adds
`--lm-head-draft`, fixes DFlash2 at K=4, defaults to `bf16`, and has no `--gdn-state-fp16`.

## Test setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16,303 MiB, driver 616.92, 175 W limit; display on the integrated GPU, so the dGPU was idle (0 MiB used) before each run |
| Host | Intel Core Ultra 9 275HX, 32 GB RAM, Windows 11 Home 10.0.26200, Balanced power plan, on AC power |
| Engine | Ryan-gsq NInfer v3 `b06908ba`, Release `sm_120a`, CUDA 13.4.2 (`runtime-v3/engine`) |
| MTP model | `qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer` (text + MTP, sha256 `947d2c51…`) |
| DFlash2 model | `qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer` (text + MTP + DFlash2, sha256 `a2cf5282…`) |
| Vision models | `…-vision-mtp.ninfer` and `…-vision-mtp-dflash2.ninfer`, built for this run ([Vision](#vision-images-and-video)) |
| Flags on every run | `--max-concurrency 1 --cuda-memory-policy strict --host-cache-mib 5120 --gdn-state-fp16 --prefill-chunk 1024`, prefix reuse on (vision runs use `--cuda-memory-policy default`) |
| Benchmark-only flags | `--no-thinking --greedy --presence-penalty 0 --frequency-penalty 0` |

The two downloaded models contain no vision component. Everything up to [Vision](#vision-images-and-video)
is text-only.

**How the metrics were measured:**

- **Decode** is (completion tokens − 1) ÷ server decode time. "Short" decode is weighted over two
  512-token generations (prose and code).
- **TTFT** (time to first token) runs from HTTP submission to the first streamed text.
- **Prefill** rates come from the server's own timings.

## Method

1. **Speed sweep.** Each speculative setting ran at a 32K context with `rk4v4`, on three short
   chats and two 512-token generations. Each setting ran once.
2. **Capacity search.** Every profile started with an explicit `--max-context` equal to
   `--kv-capacity` under the strict dedicated-VRAM policy.
   - Two deliberately oversized starts, at 262,144 and 524,288, make the engine report its exact
     reservation, which gives the true bytes per token.
   - The search then backed off and bisected to 1,024-token resolution. This took 40 engine starts.
3. **Full-window validation.** Each finalist started at its ceiling.
   - The prompt filled the window to the ceiling minus 2,048 tokens, using the engine's docs and
     source as text. Three codes were planted at 25%, 55% and 85% depth, and the model was asked
     to list them.
   - A follow-up turn then asked for a 400+ word summary, which gives 512 tokens of decode at
     full depth on top of the cached context.
4. **Equal-depth comparison.** The same 99,328-token prompt was replayed on every finalist, each at
   its own ceiling. This compares backends and KV formats at the same depth.

## Results

### Full-window validation

Every configuration below retrieved all three planted codes.

| Configuration | Max context | Short decode | Decode @ 99K | Decode @ full | Full-window prefill | TTFT, full window | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MTP3, `rk4v4`, no head** | **231,424** | 100.5 | 80.2 | 60.5 @ 229K | 711 | 323 s | 15,706 MiB |
| MTP3 + head, `rk4v4` | 212,992 | 104.3 | 82.0 | 63.5 @ 211K | 714 | 296 s | 15,714 MiB |
| MTP3 + head, `rk8v4` | 145,408 | 103.8 | 83.7 | 70.6 @ 143K | 855 | 168 s | 15,694 MiB |
| DFlash2 K5, `rk4v4`, no head | 121,856 | 114.2 | 68.4 | 70.1 @ 120K | 883 | 136 s | 15,556 MiB |
| DFlash2 K5 + head, `rk4v4` | 101,376 | **116.4** | 69.1 | 69.1 @ 99K | 938 | 106 s | 15,546 MiB |
| DFlash2 K5 + head, `rk8v4` | 69,632 | — | — | — | — | — | not validated |

All speeds are in tok/s.

- **Follow-up turns are fast.** They reached first token in 0.38–0.68 s, because prefix reuse
  restored the whole conversation; the follow-up reused 99,319 to 229,366 cached tokens.
- **Prefill is paid once.** A full window costs 2–5 minutes the first time only.
- **Speed at depth by backend.**
  - MTP draft acceptance stayed at 0.50–0.59.
  - DFlash2 acceptance fell to 0.35–0.38 at depth, which is why its short-prompt lead disappears.

### Speculative decoding sweep

32K context, `rk4v4`, short prompts.

| Setting | Decode tok/s | Draft acceptance | Peak VRAM |
|---|---:|---:|---:|
| MTP, 3 drafts | 100.9 | 0.584 | 12,034 MiB |
| **MTP, 3 drafts + `--lm-head-draft`** | **106.4** | 0.573 | 12,376 MiB |
| MTP, adaptive up to 4 | 97.3 | 0.590 | 12,040 MiB |
| MTP, adaptive up to 4 + head | 103.6 | 0.578 | 12,448 MiB |
| MTP, adaptive up to 5 + head | 99.2 | 0.554 | 12,390 MiB |
| DFlash2 K=4 + head | 116.2 | 0.597 | 14,298 MiB |
| **DFlash2 K=5 + head** | **118.9** | 0.545 | 14,360 MiB |
| DFlash2 K=6 + head | 114.1 | 0.459 | 14,300 MiB |
| DFlash2 K=7 + head | 102.1 | 0.394 | 14,300 MiB |
| DFlash2 K=5, no head | 116.7 | 0.538 | 13,958 MiB |

The best DFlash2 width here is K=5. Upstream's RTX 3090 measurements peaked at K=7.

### Prefill route

24K-token prompt, MTP adaptive up to 4 + head.

| Route | Prefill tok/s | TTFT | Peak VRAM |
|---|---:|---:|---:|
| Chunk 1024 (default kernels) | 1,383 | 17.4 s | 12,448 MiB |
| `--prefill-cublas`, chunk 2048 | 1,401 | 17.2 s | 13,444 MiB |
| `--prefill-cublas`, chunk 4096 | 1,374 | 17.5 s | 15,436 MiB |

On these GGUF-quantized weights, cuBLAS prefill is within about 1% of the default route but needs
1–3 GiB more VRAM. It is not worth enabling. Upstream reports 1.6–1.8× on RTX 3090 artifacts.

### Maximum context

Strict policy; each value is the largest that started.

| Profile | Bytes/token | Runtime budget | Planner bound | Actual max | Planner error |
|---|---:|---:|---:|---:|---:|
| MTP3 + head, `rk8v4` | 27,744 | 5.79 GiB | 158,599 | 145,408 | −13,191 |
| MTP3 + head, `rk4v4` | 19,040 | 5.79 GiB | 231,102 | 212,992 | −18,110 |
| MTP3, `rk4v4`, no head | 19,040 | 6.12 GiB | 249,854 | 231,424 | −18,430 |
| DFlash2 K5 + head, `rk8v4` | 26,112 | 4.04 GiB | 70,069 | 69,632 | −437 |
| DFlash2 K5 + head, `rk4v4` | 17,920 | 4.04 GiB | 102,101 | 101,376 | −725 |
| DFlash2 K5, `rk4v4`, no head | 17,920 | 4.37 GiB | 122,025 | 121,856 | −169 |

How to read this table:

- **Bytes/token.** MTP pays 1.1–1.6 KB per token more than the KV format alone, for the MTP
  layer's own attention cache.
- **Runtime budget.** DFlash2's draft model adds 1.8 GiB of weights, which is why its budget is
  smaller.
- **Proposal head cost.** `--lm-head-draft` costs about 340 MiB: 18K tokens with MTP, 20K with
  DFlash2.
- **Planner bound.** This is the limit implied by the engine's "requires X bytes, but only Y"
  report. For MTP it is 13–18K tokens too optimistic: the planner accepts those sizes, but strict
  admission then fails on a contiguous device allocation. Only measured ceilings can be trusted.
- **No spill.** At every ceiling, strict admission placed 15,239–15,413 MiB of device allocations
  in dedicated VRAM with zero growth in Windows Shared memory, so nothing fell back to system RAM.

## Findings

- **`rk4v4` is the main gain.** It gives 46% more context than `rk8v4`. Deep decode was 2% lower
  (82.0 vs 83.7 tok/s at 99K), which is within run-to-run variance.
- **MTP beats DFlash2 for long context.** At 99K, MTP decoded at 80–84 tok/s against 68–69 for
  DFlash2. MTP also fits about twice the context. DFlash2 only wins on short prompts, by 10–14%.
- **The proposal head is marginal.** For MTP it adds about 4–5% on short prompts and about 2% at
  depth, for 18K tokens. For DFlash2 it adds about 2% on short prompts and nothing at depth, for
  20K tokens.
- **Fixed MTP3 is the best MTP setting.** Adaptive widths up to 4 or 5 were 3–7% slower.

## Flags that don't apply to these models

- **`--embedding-q4` and `--lm-head-q6`** are refused at startup. They require the token embedding
  and output head to be stored as row-split Q8_G32, and these GGUF imports keep both in GGUF quants.
- **`--prefill-cublas`** gives no meaningful gain here and costs VRAM (see the prefill table above).

## Vision (images and video)

### Recommendation

| Use | Configuration | Context | Why |
|---|---|---:|---|
| **Image to prompt** | MTP3 + head, `rk4v4`, Vision tower `resident` | 147,456 | Fastest: first token in 1.4–1.5 s for a 1.6 MP image |
| **Video to prompt, YouTube summaries** | MTP3 + head, `rk4v4`, Vision tower `overlay` | 169,984 | 22.5K more context for transcripts and follow-ups, at about 1 s more per video |

Launchers: [inf-cmd/image.ps1](../../inf-cmd/image.ps1) and [inf-cmd/video.ps1](../../inf-cmd/video.ps1).
Prepare videos with [inf-cmd/prepare-video.ps1](../../inf-cmd/prepare-video.ps1) first (see
[engine media limits](#engine-media-limits)).

### Vision model files

Neither downloaded model file has a Vision component, so two were built with the pinned converter
(`ninfer-v3` `b06908ba`, recipe `qwen3_8_27b_gguf`, `--proposal`):

| File | Components | Size | Built from |
|---|---|---:|---|
| `qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer` | text, vision, MTP | 11.09 GB | RentedNoodle v2.0 GGUF + `mmproj-Qwen3.8-27B-BF16.gguf` |
| `qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp-dflash2.ninfer` | text, vision, MTP, DFlash2 | 13.32 GB | the same + `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c` |

- **Same text weights.** All 981 text, MTP and proposal-head tensors of the first file are
  byte-identical to the MTP-only model above. All 1,072 tensors of the second are byte-identical
  to the DFlash2 model above. The text results therefore apply unchanged.
- **Same frontend.** The tokenizer and Froggeric chat template were copied from the original
  files. `config.json` and both Vision preprocessor configs come from `Qwen/Qwen3.8-27B`.
- **Validated.** Both files pass `scripts/inspect-ryan-model.py` (schema, encodings, layouts,
  geometry). The recipe stores the Vision tower in 4- and 5-bit groups, so it adds only ~0.3 GB.

### Memory policy and residency

- **Vision cannot use the strict policy.** Strict accepts only text (the engine refuses `--vision`
  with it), so every vision run uses the `default` policy, which does not guarantee dedicated-VRAM
  residency. Instead, each run sampled the engine process's Windows GPU counters
  (`\GPU Process Memory\Dedicated|Shared Usage`, about every 2–3 s). Shared usage stayed flat
  throughout every media run, so nothing spilled to system RAM under load.
- **Three places to run the Vision tower ("mmproj"):**
  - `resident`: the tower and a 1.2 GB encode workspace stay in VRAM.
  - `overlay`: the tower stays in pinned system RAM and is streamed through the GPU per request,
    into memory borrowed from free KV pages or from temporarily evicted text weights.
  - `cpu`: the tower is decoded to system RAM and encoded on CPU threads. By default this caps
    each image or video at 256 merged tokens. The `cpu, full resolution` row lifts that cap
    (`--vision-max-merged 16384`) to price CPU encoding at equal detail.

### Maximum context with Vision

`rk4v4`, chunk 1024, default policy; each value is the largest that started.

| Profile | Max context | VRAM free at ceiling | Dedicated VRAM in use |
|---|---:|---:|---:|
| MTP3 + head, `resident` | 147,456 | 417 MiB | 14,894 MiB |
| **MTP3 + head, `overlay`** | **169,984** | 423 MiB | 14,891 MiB |
| MTP3 + head, `cpu` | 171,008 | 421 MiB | 14,822–14,890 MiB |
| DFlash2 K5 + head, `resident` | 26,624 | 723 MiB | 14,591 MiB |
| DFlash2 K5 + head, `overlay` | 39,936 | 899 MiB | 14,418 MiB |

For comparison, the same MTP3 + head text setup reaches 212,992 under the strict policy. The
default policy's smaller runtime budget and the Vision workspace account for the gap.

- **`overlay` is nearly free in context.** It costs 1,024 tokens against `cpu`, which keeps no
  Vision memory on the GPU at all.
- **DFlash2 barely fits.** Its draft model leaves 26–40K tokens. With the tower `resident`, it
  cannot even hold one prompt at the 32,768-token Vision limit (see below).
- **No RAM cache at the ceiling.** With about 420 MiB of VRAM free, the engine cannot pin its
  system-RAM KV cache (it reports `host … 0 B KV`). This only matters for conversations evicted
  from VRAM; follow-up turns in the active conversation still reuse the GPU cache.
- **`overlay` pins ~560 MiB extra RAM** at the ceiling. This is most likely the pinned copy of
  text weights that overlay may evict for an encode window: the engine's docs describe one, but
  the startup log doesn't itemize it. Shared memory stayed flat through the media runs, so it is
  not a spill. With MTP, every encode window borrowed free KV pages and none evicted weights.

### Images: image to prompt

Three PNGs (941×1672, 941×1672, 1024×1536); a 512-token-limit image-to-prompt request each, plus
one follow-up turn on the first image.

| Profile | Image tokens | Vision encode | Time to first token | Decode | Follow-up first token |
|---|---:|---:|---:|---:|---:|
| **MTP3 + head, `resident`** | 1,563 | 0.19 s | **1.41–1.49 s** | **93.4** | 0.38 s |
| MTP3 + head, `overlay` | 1,563 | 0.21 s | 1.48–1.55 s | 92.5 | 0.40 s |
| DFlash2 K5 + head, `resident` | 1,563 | 0.19 s | 1.45–1.52 s | 86.0 | 0.39 s |
| DFlash2 K5 + head, `overlay` | 1,563 | 0.20 s | 1.49–1.55 s | 85.6 | 0.40 s |
| MTP3 + head, `cpu` (256-token cap) | 296 | 6.81 s | 7.22–7.39 s | 99.3 | 0.13 s |
| MTP3 + head, `cpu`, full resolution | 1,563 | 55.72 s | 56.05–58.39 s | 91.8 | — |

Decode is in tok/s, weighted over the three images.

### Video: video to prompt

Three clips of 10–14 s each (720p landscape, 4K vertical, 720p vertical), a 512-token-limit
describe-then-prompt request each. The 4K clip exceeds an engine limit and was sent at 1904×3384
(see [engine media limits](#engine-media-limits)).

| Profile | Video tokens | Frame decoding | Vision encode | Time to first token | Decode |
|---|---:|---:|---:|---:|---:|
| **MTP3 + head, `resident`** | 12,246 | 2.84 s | 1.34 s | **11.3–15.6 s** | **103.0** |
| MTP3 + head, `overlay` | 12,246 | 2.97 s | 1.42 s | 12.2–16.6 s | 102.1 |
| DFlash2 K5 + head, `resident` | 12,246 | 2.92 s | 1.37 s | 12.0–16.3 s | 93.2 |
| DFlash2 K5 + head, `overlay` | 12,246 | 3.07 s | 1.41 s | 12.6–16.8 s | 92.7 |
| MTP3 + head, `cpu` (256-token cap) | 349 | 2.55 s | 5.06 s | 6.7–10.4 s | 102.7 |

Encode and decoding times are means; the 4K clip takes the longest to decode (5.7–5.9 s).

Two harder requests:

- **150 s video:** the three clips scaled to 360×640 and repeated four times, so 303 frames are
  sampled.
- **Near the limit:** two videos and three images in one request, 28,755 tokens.

| Profile | 150 s video: tokens / first token / decode | Near the limit: tokens / first token / decode |
|---|---|---|
| **MTP3 + head, `resident`** | 12,399 / 13.1 s / 104.9 | 28,755 / 27.5 s / 106.3 |
| MTP3 + head, `overlay` | 12,399 / 13.8 s / 103.8 | 28,755 / 29.1 s / 104.6 |
| DFlash2 K5 + head, `resident` | 12,399 / 13.8 s / 101.6 | rejected: over its 26,624-token window |
| DFlash2 K5 + head, `overlay` | 12,399 / 14.1 s / 100.5 | 28,755 / 29.8 s / 88.1 |
| MTP3 + head, `cpu` (256-token cap) | 1,607 / 7.5 s / 97.0 | 1,372 / 29.8 s / 112.0 |

**Launcher checks.** These used the launchers' own sampling, not the benchmark's greedy settings.

- **`image.ps1`:** first token in 1.7 s, decoding at 88.8 tok/s.
- **`video.ps1`, 10-minute video:** prepared to 590 s and 768 frames (14,546 tokens). First token
  in 15.2 s, 92.3 tok/s with medium reasoning.
- **`video.ps1`, 19 s YouTube video plus its transcript:** first token in 1.5 s, 104.3 tok/s.

Every description checked by eye matched its media: one image, frames from two clips, and the
YouTube video.

### Engine media limits

These limits are compiled into the engine (`src/media/decode/decode.h`,
`src/models/qwen3_5/frontend/`); no server flag changes them.

| Limit | Value | Effect |
|---|---|---|
| Video sampling | 2 fps, 4 to 768 frames, spread evenly | Longer videos get sparser frames, not more of them |
| Decoded pixels per video | 128 Mi across the sampled frames | 4K is rejected above ~8 s, 1080p above ~32 s, 720p above ~72 s |
| Video length | 600 s | Longer videos are rejected |
| Media bytes | 256 MiB per prompt | |
| Pixels per video | 25,165,824 in total (Qwen's video config) | One video uses at most about 12,300 Vision tokens, whatever its length |
| Image size | 64 Mi pixels (about 67 megapixels) | |
| Vision tokens | 16,384 per item, 32,768 per prompt | Caps how much media one request carries, whatever the window |

`prepare-video.ps1` handles these automatically, at no loss in what the model sees:

- It drops the audio, caps the frame rate at 4 fps and scales the video to the largest size the
  engine accepts.
- It speeds a video over 590 s up to 590 s. The engine never samples more than 768 frames anyway,
  so a long video keeps the same frame coverage, but times the model reports are compressed by
  the speed-up factor.
- For a YouTube or other `yt-dlp` URL, it also saves the English subtitles as a plain-text
  transcript. **Qwen3.8 cannot hear:** a summary covers anything said aloud only if the transcript
  is in the prompt.

### Vision findings

- **MTP beats DFlash2 for vision.** DFlash2's drafts are accepted less often on descriptions:
  0.35–0.41 against MTP's 0.49–0.60. As a result DFlash2 decodes 8% slower on images, 10% slower
  on videos and 17% slower near the limit, and it fits 4–6× less context. The vision model file
  with DFlash2 is not needed, and was deleted after this run (rebuild it with the command under
  Reproduce).
- **`overlay` costs about 1 s on a video and almost nothing elsewhere.**
  - Decode stays within 1% of `resident`.
  - Image encoding takes 0.02 s longer, and first token is 0.07 s later.
  - Video first token is 0.9–1.0 s later, and the near-limit request 1.5 s later.
  - In return it gives 22,528 more tokens of context.
- **CPU encoding is not viable.** At equal detail it is about 290× slower than the GPU (55.7 s
  against 0.19 s per image). At its default 256-token cap it still encodes 4–36× slower than the
  GPU while showing the model 5–35× fewer tokens. Only its tiny prompts make some of its video
  first-token times shorter.
- **Long videos cost about the same as short ones.** The 150 s video used 12,399 tokens, the same
  as the 14 s clips, because Qwen's video config caps total pixels. Frame resolution drops instead.
- **Prefill dominates video latency.** About 8–10 s of the 11–16 s to first token is prefill. Frame
  decoding adds 1.3–5.9 s depending on source resolution, and encoding adds 0.8–1.5 s.
- **Vision context needs are bounded.** One prompt carries at most 32,768 vision tokens. Context
  beyond that serves transcripts and follow-up turns, which reuse the cached media (0.4 s to first
  token).

### Vision caveats

- **Default policy.** Ceilings were measured with an idle dGPU. Unlike strict, the default policy
  will not refuse to start if another application holds VRAM, so lower the context if the GPU is
  shared. The engine leaves about 420 MiB free at these MTP ceilings.
- **Single runs, greedy, thinking off,** as for text. The launchers use different settings:
  `image.ps1` uses Qwen's non-thinking preset at temperature 0.7, and `video.ps1` uses medium
  reasoning at temperature 1.0.
- **Quality was spot-checked, not evaluated.** Description accuracy was checked by eye on four
  items.
- **No audio.** The model reads frames only. Speech must come in as a transcript.

## Text caveats

- **Single runs.** Each configuration ran once. Identical configurations varied by up to about 2%
  between runs; for example, MTP3 + head measured 106.4 tok/s in the sweep and 104.3 in
  validation. Treat differences under about 2% as noise.
- **Sampling.** Speeds are greedy with thinking off. Normal sampling at temperature 1.0 lowers
  draft acceptance, so real-use decode will be somewhat lower.
- **Idle dGPU required.** Ceilings were measured with nothing else on the dGPU and only the 64 MiB
  strict reserve left free. If another application is using the GPU, strict startup at these
  sizes fails rather than spilling. Leave about 8K tokens of margin if that is common.
- **Thermals.** During long runs the GPU sat at its power limit: 167–175 W peak, 88–92 °C peak,
  and a median SM clock of 2.0–2.13 GHz, all on the Balanced power plan. These are sustained laptop
  figures, not burst figures.
- **Quality.** Quality was checked only by needle retrieval and coherent long outputs. No
  perplexity was measured. Upstream reports `rk4v4` at about +0.21% and `rk8v4` at about +0.08%
  perplexity against `int8` KV.
- **When to re-measure.** Re-probe the ceilings after any driver, engine build, model or
  desktop-load change.

## Reproduce

```powershell
python scripts/tune-ryan-engine.py --output benchmark-results/<new-dir> --phase speed
python scripts/tune-ryan-engine.py --output benchmark-results/<new-dir> --phase probe
python scripts/tune-ryan-engine.py --output benchmark-results/<new-dir> --phase long `
  --profiles mtp-rk4v4-nohead,mtp-rk4v4,dflash2-rk4v4-nohead,dflash2-rk4v4,mtp-rk8v4
python scripts/tune-ryan-engine.py --output benchmark-results/<new-dir> --phase matched
```

Vision (the model files first, then two phases):

```powershell
# from .deps
infer-v3, with the .deps\convert-v3 venv (numpy, gguf, CPU torch); the metadata folder
# holds the original files' tokenizer, template and generation config plus Qwen/Qwen3.8-27B's
# config.json, preprocessor_config.json and video_preprocessor_config.json
..\convert-v3\Scripts\python.exe -m tools.convert --model <metadata> --recipe qwen3_8_27b_gguf `
  --source gguf=<Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf> `
  --source vision=<mmproj-Qwen3.8-27B-BF16.gguf> `
  --components text,vision,mtp --proposal --device cpu --rows-per-chunk 512 `
  --out E:\llm\RentedNoodle-NInfer-v3\qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer
# DFlash2 variant: add --source dflash2=<z-lab/Qwen3.8-27B-DFlash2> and components ...,dflash2

python scripts/tune-vision.py --output benchmark-results/<new-dir>/vision --phase probe
python scripts/tune-vision.py --output benchmark-results/<new-dir>/vision --phase media
```

Each phase saves after every step and skips completed steps when rerun, so use a new output
directory for a fresh measurement. The four text phases take about an hour in total, and the two
vision phases about 25 minutes. Set `PYTHONIOENCODING=utf-8` for the vision script, because one
test video's filename contains emoji.

## Files in this directory

| Path | Contents |
|---|---|
| `tuning.json` | Every measurement and its exact command line |
| `sweep-*/` | Speed-sweep runs |
| `probe/*/` | Capacity-search startups |
| `long-*/` | Full-window validation runs |
| `matched-*/` | Equal-depth runs |
| `workload-*.json` | Generated needle prompts, keyed by token count |
| `vision/vision.json` | Every vision measurement, including per-request timings and GPU memory samples |
| `vision/probe/*/` | Vision capacity-search startups |
| `vision/media-*/` | Vision media runs, one per profile |
| `vision/media/` | The 4K test clip fitted to the engine's limit, and the 150 s test video |

Each run directory contains `command.json`, `stdout.log`, `stderr.log`, `requests.jsonl`, `gpu.csv`
(1 s GPU telemetry) and one JSON file per request.
