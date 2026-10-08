# Inference launchers: user guide

One launcher per use case, each tuned on the RTX 5080 Laptop GPU (16 GB) for the longest context
and fastest decode that use case can get. The measurements behind every setting are in
[benchmark-results/2026-10-05-rtx5080-tuning/summary.md](../benchmark-results/2026-10-05-rtx5080-tuning/summary.md).

| Use case | Launcher | Port | Model ID | Context (tokens) | Setup |
|---|---|---:|---|---:|---|
| Chat | `chat.ps1` | 8091 | `qwen3.8-27b-chat` | 121,856 | DFlash2 speculation, reasoning medium |
| Coding harnesses | `coding.ps1` | 8092 | `qwen3.8-27b-coding` | 229,376 | MTP speculation, n-gram copying, reasoning high |
| Research | `research.ps1` | 8093 | `qwen3.8-27b-research` | 231,424 | MTP speculation, reasoning high |
| Image analysis | `image.ps1` | 8094 | `qwen3.8-27b-image` | 147,456 | Vision on the GPU, thinking off |
| Video analysis | `video.ps1` | 8095 | `qwen3.8-27b-video` | 169,984 | Vision streamed from RAM, reasoning medium |

`prepare-video.ps1` readies a video file or a YouTube link for `video.ps1`.

## Before you start

- **One server at a time.** The GPU holds one model. Each launcher refuses to start while another
  `ninfer-serve` is running.
- **Keep the GPU free.** The context sizes fill the GPU almost completely and were measured with
  nothing else on it. On this laptop the display runs on the integrated GPU, so that is the normal
  state. A game or GPU-accelerated app on the NVIDIA GPU can stop a launcher from starting; see
  [Troubleshooting](#troubleshooting).
- **Required files:**
  - the engine in `runtime-v3\engine`
  - the model files in `E:\llm\RentedNoodle-NInfer-v3`: `…-mtp-only.ninfer` for coding and
    research, `…-mtp-dflash2.ninfer` for chat, and `…-vision-mtp.ninfer` for image and video.
    The vision file was built locally; see the summary for how.
  - `ffmpeg`, plus `yt-dlp` for YouTube links, for `prepare-video.ps1`.

## Start and stop

Run a launcher from the repository root in PowerShell:

```powershell
.\inf-cmd\coding.ps1
```

It prints its address, loads for about 15–20 s, and is ready when the log shows `listening on`.

- **Stop with Ctrl+C**, not by closing the window. `coding.ps1` and `research.ps1` save their
  conversation cache on Ctrl+C and restore it on the next start, so earlier sessions resume without
  re-reading their prompts. Closing the window can lose that save.
- **Change the port** with `-Port`, for example `.\inf-cmd\chat.ps1 -Port 9000`.
- **Add engine flags** by appending them, for example `.\inf-cmd\research.ps1 --structured-output`.

There is no API key: the servers listen on `127.0.0.1` only. Clients that insist on a key accept any
value.

## Chat

```powershell
.\inf-cmd\chat.ps1
```

Point any OpenAI-compatible client at `http://127.0.0.1:8091/v1` with model `qwen3.8-27b-chat`.

- Fastest launcher for short and medium conversations: about 117 tok/s on short prompts.
- It slows down as a conversation grows, to about 68 tok/s at 99K tokens. For a very long
  conversation, use `research.ps1` instead.
- Replies think at medium effort by default. For quick small talk, send `"reasoning_effort": "none"`
  in the request.

## Coding with Claude Code, opencode or Pi

```powershell
.\inf-cmd\coding.ps1
```

| Setting | Value |
|---|---|
| Anthropic API base (Claude Code) | `http://127.0.0.1:8092` |
| OpenAI API base (opencode, Pi) | `http://127.0.0.1:8092/v1` |
| Model | `qwen3.8-27b-coding` |
| Context window to configure in the harness | about 223,000 tokens |

**Claude Code**, in the PowerShell window you start it from:

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8092"
$env:ANTHROPIC_AUTH_TOKEN = "local"
$env:ANTHROPIC_MODEL = "qwen3.8-27b-coding"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "qwen3.8-27b-coding"
claude
```

The server window is 229,376 tokens, above the 200K window Claude Code compacts against.

**opencode and Pi:** add an OpenAI-compatible provider with the base URL and model above, and set
the model's context window to about 223,000 tokens. That makes the harness compact its history
before the server runs out of room.

Good to know:

- **Copying from files:** n-gram drafting copies text from files and tool output, which speeds up
  edits that repeat existing code.
- **Rewritten history:** when a harness compacts or rewrites history, the server resumes from the
  nearest cached point instead of re-reading everything.
- **Settings:** the defaults are high reasoning effort, earlier reasoning kept in the conversation,
  and temperature 0.6, Qwen's coding preset. A harness that sends its own thinking or sampling
  settings overrides them.
- **Tested so far:** the Anthropic Messages API, including tool calls and thinking. A full Claude
  Code, opencode or Pi session has not been run against it yet.

## Research

```powershell
.\inf-cmd\research.ps1
```

Point your research engine at `http://127.0.0.1:8093/v1` with model `qwen3.8-27b-research`.

Reading a long prompt the first time takes a while:

| Prompt size | First response |
|---:|---:|
| 25K tokens | ~17 s |
| 100K tokens | ~1.7 min |
| 229K tokens | ~5.4 min |

Follow-up questions on the same documents start in under a second, because the server keeps them
cached.

- **Order every prompt the same way:** fixed instructions first, then the documents, then the
  question. The server finds any shared beginning by itself and reads only what changed, so no
  cache markers are needed.
- **JSON output:** if you need JSON-schema responses, start with
  `.\inf-cmd\research.ps1 --structured-output`.
- **Effort:** reasoning is high by default. For the hardest comparisons across many documents, send
  `"reasoning_effort": "xhigh"`.

## Image analysis (image to prompt)

```powershell
.\inf-cmd\image.ps1
```

Send images as base64 data URLs in the `image_url` part of a Chat Completions message; see
[Sending images and video](#sending-images-and-video). A 1.6-megapixel image reaches its first
token in about 1.5 s, then decodes at about 90 tok/s.

- **Thinking is off by default**, which suits description and prompt writing. To reason about an
  image, such as reading a chart or comparing two images, send `"reasoning_effort": "medium"`.
- **Follow-ups are fast.** Asking a follow-up about the same image starts in about 0.4 s, since the
  image is already encoded.
- **Limits:** an image may be up to about 67 megapixels and uses at most 16,384 tokens. One request
  can carry up to 32,768 image and video tokens in total.

## Video analysis and YouTube summaries

The engine rejects many videos as they come: anything over 10 minutes, a 4K video over about
8 s, a 720p video over about 72 s. Prepare each video first:

```powershell
.\inf-cmd\prepare-video.ps1 "C:\videos\clip.mp4"
.\inf-cmd\prepare-video.ps1 "https://www.youtube.com/watch?v=VIDEO_ID"
.\inf-cmd\prepare-video.ps1 "C:\videos\clip.mp4" -OutDir "D:\prepared"
```

It prints the file to send, plus a transcript for a YouTube link. By default the output goes to
`%TEMP%\ninfer-video`.

- **The model sees exactly the same thing.** The engine shrinks every video to the same budget
  anyway, so preparing it costs no quality. A local MP4 that already fits is used as is.
- **Audio is dropped.** **Qwen3.8 cannot hear.** For anything said aloud, paste the transcript into
  the prompt. For a YouTube link, the English subtitles are saved as `<id>.transcript.txt`.
- **Videos over about 10 minutes are sped up** to fit. The model sees the same number of frames, but
  times it mentions are compressed by the printed factor.

Then start the server and send the prepared file:

```powershell
.\inf-cmd\video.ps1
```

| Video | First response | Decode |
|---|---:|---:|
| 10–14 s clip | 12–17 s | ~102 tok/s |
| 10-minute video, prepared | ~15 s | ~92 tok/s |
| 19 s YouTube video plus its transcript | ~1.5 s | ~104 tok/s |

### Longer videos lose detail

Every video gets the same budget of about 12,300 tokens. A longer video is therefore seen in
smaller frames, and past 6.4 minutes the frames are also further apart:

| Video length | Frame size the model sees (16:9) | Time between frames |
|---|---|---:|
| 14 s | ~1264×711 | 0.5 s |
| 1 min | ~611×344 | 0.5 s |
| 2.5 min | ~386×217 | 0.5 s |
| 10 min | ~241×136 | 0.8 s |
| 30 min | ~241×136 | 2.3 s |

People, scenes, colours and large movements survive. Small on-screen text, distant faces and brief
moments do not. When detail matters:

- **Split the video** into 1–2 minute pieces with `ffmpeg`, ask about each piece separately, then ask
  for a summary of the answers. Each piece gets the full budget.
- **Send key moments as images.** A single frame keeps its full resolution.
- **Lean on the transcript** for YouTube videos, where most of the content is usually in the speech.

One request can carry two prepared videos (32,768 image and video tokens in total).

## Sending images and video

Both vision servers take the standard OpenAI Chat Completions format, with each image or video as a
base64 data URL. This Python example sends one image; for video, pass a prepared `.mp4` and use port
8095 and model `qwen3.8-27b-video`.

```python
import base64, json, mimetypes, urllib.request
from pathlib import Path

def media(path):
    """A Chat Completions content part for an image or an .mp4 video, as a base64 data URL."""
    mime = mimetypes.guess_type(path)[0]
    kind = "video" if mime.startswith("video/") else "image"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": f"{kind}_url", f"{kind}_url": {"url": f"data:{mime};base64,{data}"}}

body = {
    "model": "qwen3.8-27b-image",
    "messages": [{"role": "user", "content": [
        media(r"C:\ninfer-rtx5080\vision-test\1.png"),
        {"type": "text", "text": "Write a detailed text-to-image prompt that would recreate this image."},
    ]}],
    "max_tokens": 1024,
}
request = urllib.request.Request("http://127.0.0.1:8094/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # bypass any system proxy
reply = json.load(opener.open(request, timeout=600))
print(reply["choices"][0]["message"]["content"])
```

For a YouTube summary, add the transcript as text next to the video:

```python
transcript = Path(r"C:\Users\you\AppData\Local\Temp\ninfer-video\VIDEO_ID.transcript.txt").read_text(encoding="utf-8")
content = [media(r"C:\Users\you\AppData\Local\Temp\ninfer-video\VIDEO_ID-fit-640x360.mp4"),
           {"type": "text", "text": "Transcript:\n" + transcript + "\n\nSummarize this video."}]
```

## Thinking and reasoning effort

Each launcher sets a default, and any request can override it with `reasoning_effort`: one of
`none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. The reasoning comes back separately,
in `reasoning_content`.

| Launcher | Default | When to change it |
|---|---|---|
| `chat.ps1` | medium | `none` for quick small talk |
| `coding.ps1` | high, earlier reasoning kept | Usually leave it to the harness |
| `research.ps1` | high | `xhigh` for the hardest comparisons across documents |
| `image.ps1` | off | `medium` to reason about an image rather than describe it |
| `video.ps1` | medium | `low` for a quick video-to-prompt |

Thinking helps multi-step work (code, tool use, analysis) and adds little to plain description or
casual chat. It doesn't change decode speed, but every thinking token comes before the answer:
1,000 of them add about 10 s.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Another ninfer-serve is running` | Stop the other server with Ctrl+C in its window, or end `ninfer-serve.exe` in Task Manager |
| `cannot bind 127.0.0.1:<port>` | Something else uses the port; start with `-Port <other>`. Port 8081 belongs to Apache on this machine |
| `strict could not admit max_context=…` (chat, coding, research) | Something else is using the NVIDIA GPU. Close it, or lower both `--max-context` and `--kv-capacity` in the launcher by 8,192 |
| A vision launcher fails to start or runs slowly | Same cause: free the GPU, or lower its two context values by 8,192 |
| HTTP 400 `media_budget_exceeded` | The video was not prepared: run `prepare-video.ps1`. Or the request carries more than 32,768 image and video tokens |
| HTTP 400 `context_length_exceeded` | The prompt plus the reply budget is over the launcher's context; trim the prompt or start a new conversation |
| A long prompt seems stuck | It is reading the prompt: about 1.7 min at 100K tokens and 5.4 min at 229K. Only the first request pays this |
| A previous coding or research session reads its whole prompt again | The server was closed instead of stopped with Ctrl+C, so its cache was not saved |
