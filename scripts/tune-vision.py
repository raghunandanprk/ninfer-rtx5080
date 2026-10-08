"""Benchmark Vision on images and video for each Vision residency, and find each profile's ceiling.

The strict memory policy accepts only text, so every profile here runs under the default policy,
which does not stop the driver from spilling device data into Shared (system) memory. Windows'
per-process GPU counters are therefore sampled: a profile's Shared usage at its ceiling is
compared with the same profile at 8K, where nothing can spill.

probe: context ceiling per profile (search shared with tune-ryan-engine.py), plus an 8K reference.
media: start each profile at its ceiling and time, per image, an image-to-prompt request and a warm
       follow-up; per video, a video-to-prompt request; then a 150 s video assembled from the test
       clips and one request filling most of the 32,768-token Vision envelope. A video over the
       engine's decoded-pixel limit is first scaled to the largest size it accepts.

Every step is saved as it finishes; rerunning skips completed steps.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tune", ROOT / "scripts/tune-ryan-engine.py")
tune = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tune)
bench = tune.bench

MODEL_DIR = Path("E:/llm/RentedNoodle-NInfer-v3")
bench.MODELS["vision-mtp"] = MODEL_DIR / "qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp.ninfer"
bench.MODELS["vision-dflash2"] = MODEL_DIR / "qwen3.8-27b-orcarouter-iq3-xxs-vision-mtp-dflash2.ninfer"
MEDIA = ROOT / "vision-test"
COMMON = [flag if flag != "strict" else "default" for flag in tune.COMMON]
SPEC = {"vision-mtp": ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"],
        "vision-dflash2": ["--spec", "dflash2", "--draft-tokens", "5", "--lm-head-draft"]}
IMAGE_TASK = ("Write a detailed text-to-image prompt that would recreate this image: subject, pose, "
              "clothing, setting, lighting, camera and style. Output only the prompt.")
FOLLOW_TASK = "Now condense that into a single prompt under 40 words."
VIDEO_TASK = ("Describe this video in detail: subjects, actions in order, setting, camera movement and "
              "mood. Then write a text-to-video prompt that would recreate it.")


def profile(model: str, residency: str, max_merged: int | None = None) -> dict:
    flags = ["--kv-dtype", "rk4v4", "--prefill-chunk", "1024", "--gdn-state-fp16", *SPEC[model],
             "--vision", "--vision-residency", residency]
    if max_merged:
        flags += ["--vision-max-merged", str(max_merged)]
    return {"model": model, "kv": "rk4v4", "residency": residency, "common": COMMON, "flags": flags}


PROFILES = {
    "mtp-resident": profile("vision-mtp", "resident"),
    "mtp-overlay": profile("vision-mtp", "overlay"),
    "mtp-cpu": profile("vision-mtp", "cpu"),
    "dflash2-resident": profile("vision-dflash2", "resident"),
    "dflash2-overlay": profile("vision-dflash2", "overlay"),
}
# CPU residency caps an item at 256 merged tokens unless told otherwise; this profile prices
# full-resolution CPU encoding on the images only.
CPU_FULL = profile("vision-mtp", "cpu", 16384)


class GpuMemorySampler:
    """Samples one process's Windows GPU Dedicated and Shared usage (MiB) in the background."""

    SCRIPT = ("$c = Get-Counter -Counter '\\GPU Process Memory(pid_PID_*)\\Dedicated Usage',"
              "'\\GPU Process Memory(pid_PID_*)\\Shared Usage' -ErrorAction SilentlyContinue; "
              "$d = ($c.CounterSamples | ? Path -like '*dedicated usage' | Measure-Object CookedValue -Sum).Sum; "
              "$s = ($c.CounterSamples | ? Path -like '*shared usage' | Measure-Object CookedValue -Sum).Sum; "
              "'{0:F1},{1:F1}' -f ($d/1MB), ($s/1MB)")

    def __init__(self, pid: int):
        self.script, self.samples, self.stop = self.SCRIPT.replace("PID", str(pid)), [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self) -> tuple[float, float] | None:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", self.script], capture_output=True,
                             text=True, creationflags=bench.NO_WINDOW).stdout.strip()
        try:
            dedicated, shared = map(float, out.split(","))
            return dedicated, shared
        except ValueError:
            return None

    def run(self) -> None:
        while not self.stop.is_set():
            if value := self.sample():
                self.samples.append(value)
            self.stop.wait(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        return {"samples": len(self.samples), "first_dedicated_mib": self.samples[0][0],
                "first_shared_mib": self.samples[0][1],
                "peak_dedicated_mib": max(s[0] for s in self.samples),
                "peak_shared_mib": max(s[1] for s in self.samples)}


def media_part(path: Path) -> dict:
    kind = "video" if path.suffix == ".mp4" else "image"
    mime = "video/mp4" if kind == "video" else "image/png"
    url = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return {"type": f"{kind}_url", f"{kind}_url": {"url": url}}


# The engine samples video at 2 fps (4..768 frames) and rejects a video whose sampled source
# frames exceed 128 Mi decoded pixels (src/media/decode/decode.h, max_decoded_video_pixels);
# no server flag changes either.
MAX_DECODED_VIDEO_PIXELS = 128 * 1024 * 1024
VIDEO_FPS, VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES = 2.0, 4, 768


def video_geometry(path: Path) -> tuple[int, int, int, float]:
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout)
    stream = probe["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    frames = int(stream.get("nb_frames") or round(float(probe["format"]["duration"]) * fps))
    return stream["width"], stream["height"], frames, fps


def sampled_frames(frames: int, fps: float) -> int:
    count = int(frames / fps * VIDEO_FPS)
    return min(max(count, VIDEO_MIN_FRAMES), VIDEO_MAX_FRAMES, frames)


def fit_video(path: Path, output: Path) -> Path:
    """The video itself when the engine accepts it, else a copy scaled to the largest size it accepts.

    The engine downsamples every video to its merged-token budget anyway, so this loses nothing
    the model would see."""
    width, height, frames, fps = video_geometry(path)
    pixels = sampled_frames(frames, fps) * width * height
    if pixels <= MAX_DECODED_VIDEO_PIXELS:
        return path
    scale = (MAX_DECODED_VIDEO_PIXELS / pixels) ** 0.5 * 0.98
    size = (int(width * scale) // 2 * 2, int(height * scale) // 2 * 2)
    fitted = output / "media" / f"{path.stem[:40]}-fit-{size[0]}x{size[1]}.mp4"
    if not fitted.exists():
        fitted.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", f"scale={size[0]}:{size[1]}",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(fitted)], check=True)
    return fitted


def long_video(output: Path) -> Path:
    """150 s of the three test clips, each scaled and padded to 360x640 and repeated four times.

    360x640 keeps the 300 sampled frames inside the engine's decoded-pixel limit."""
    path = output / "media" / "long-150s-360x640.mp4"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    clips = sorted(MEDIA.glob("*.mp4"))
    inputs = [arg for _ in range(4) for clip in clips for arg in ("-i", str(clip))]
    count = 4 * len(clips)
    scale = "".join(f"[{i}:v]fps=24,scale=360:640:force_original_aspect_ratio=decrease,"
                    f"pad=360:640:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];" for i in range(count))
    graph = scale + "".join(f"[v{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(path)], check=True)
    return path


def last_done(directory: Path) -> dict:
    lines = (directory / "requests.jsonl").read_text(encoding="utf8").splitlines()
    return next(json.loads(line) for line in reversed(lines) if '"request_done"' in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results/2026-10-05-rtx5080-tuning/vision")
    parser.add_argument("--phase", choices=["probe", "media", "all"], default="all")
    parser.add_argument("--profiles", help="comma-separated profiles (default: all whose artifact exists)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "vision.json"
    report = json.loads(report_path.read_text(encoding="utf8")) if report_path.exists() else {
        "started_utc": datetime.now(timezone.utc).isoformat(), "power": bench.power_status(),
        "common_flags": COMMON, "probes": {}, "max_context": {}, "media": {}}
    names = args.profiles.split(",") if args.profiles else [
        name for name, item in PROFILES.items() if bench.MODELS[item["model"]].exists()]
    profiles = {**PROFILES, "mtp-cpu-fullres": CPU_FULL}

    def save() -> None:
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        bench.save_json(report_path, report)

    def probe(name: str, context: int) -> dict:
        key = f"{name}@{context}"
        if key not in report["probes"]:
            directory = tune.fresh(output / "probe" / f"{name}-{context}")
            item = {"profile": name, "context": context}
            try:
                with tune.TunedServer(directory, profiles[name], context) as server:
                    sampler = GpuMemorySampler(server.process.pid)
                    item.update(ok=True, startup_seconds=server.startup_seconds, gpu_process=dict(
                        zip(("dedicated_mib", "shared_mib"), sampler.sample() or (None, None))))
            except Exception as exc:
                item.update(ok=False, error=str(exc))
            item.update(tune.startup_facts(directory), memory=bench.memory_summary(directory))
            report["probes"][key] = item
            save()
            print(f"PROBE {key}: {'ok' if item['ok'] else 'fail'} {item.get('gpu_process', '')}"
                  f"{item.get('fatal', '')[:110]}", flush=True)
        return report["probes"][key]

    if args.phase in ("probe", "all"):
        for name in names:
            if name in report["max_context"]:
                continue
            reference = probe(name, 8192)
            fits, fails, slope = tune.search_max_context(lambda context: probe(name, context), "rk4v4")
            top = report["probes"].get(f"{name}@{fits}", {})
            spill = None
            if reference.get("ok") and top.get("ok"):
                spill = top["gpu_process"]["shared_mib"] - reference["gpu_process"]["shared_mib"]
            report["max_context"][name] = {
                "context": fits, "first_failing": fails if fails <= tune.NATIVE_CONTEXT else None,
                "bytes_per_token_observed": slope, "shared_growth_vs_8k_mib": spill, "profile": profiles[name]}
            save()
            print(f"MAX {name}: {fits} tokens, Shared growth vs 8K: {spill} MiB", flush=True)

    if args.phase in ("media", "all"):
        images = sorted(MEDIA.glob("*.png"))
        videos = [fit_video(path, output) for path in sorted(MEDIA.glob("*.mp4"))]
        report["media_inputs"] = {
            path.name: dict(zip(("width", "height", "frames", "fps"), video_geometry(path)),
                            sampled_frames=sampled_frames(*video_geometry(path)[2:]))
            for path in [*videos, long_video(output)]}
        runs = [(name, report["max_context"][name]["context"], False) for name in names]
        if "mtp-cpu" in names:
            # Full-resolution CPU encoding needs no extra device memory: run it at the CPU ceiling.
            runs.append(("mtp-cpu-fullres", report["max_context"]["mtp-cpu"]["context"], True))
        for name, context, images_only in runs:
            if report["media"].get(name, {}).get("passed"):
                continue
            directory = tune.fresh(output / f"media-{name}")
            result = {"profile": profiles[name], "context": context, "passed": False, "requests": []}

            def ask(server, label: str, messages: list, limit: int) -> dict | None:
                # A rejected or timed-out request is recorded and the remaining requests still run.
                try:
                    row = server.chat((label, messages, limit))
                except Exception as exc:
                    result["requests"].append({"label": label, "error": str(exc)})
                    print(f"  {label}: FAILED {str(exc)[:200]}", flush=True)
                    return None
                done = last_done(directory)
                record = {"label": label, "prompt_tokens": row["prompt_tokens"],
                          "completion_tokens": row["completion_tokens"], "cached_tokens": row["cached_tokens"],
                          "client_ttft_ms": row["ttft_ms"], "decode_tps": row["decode_tps"],
                          "drafted": row["drafted"], "accepted": row["accepted"],
                          "timings_seconds": done.get("timings_seconds"),
                          "vision_overlay": done.get("vision_overlay"), "content": row["content"]}
                result["requests"].append(record)
                t = record["timings_seconds"] or {}
                print(f"  {label}: in={record['prompt_tokens']} cached={record['cached_tokens']} "
                      f"prepare={t.get('prepare', 0):.2f}s vision={t.get('vision', 0):.2f}s "
                      f"prefill={t.get('prefill', 0):.2f}s TTFT={t.get('ttft', 0):.2f}s "
                      f"decode={record['decode_tps']:.1f} tok/s", flush=True)
                return record

            try:
                with tune.TunedServer(directory, profiles[name], context) as server, \
                        GpuMemorySampler(server.process.pid) as sampler:
                    for image in images:
                        first = [{"role": "user", "content": [media_part(image), {"type": "text", "text": IMAGE_TASK}]}]
                        answer = ask(server, f"image:{image.name[:24]}", first, 512)
                        if answer and image == images[0] and not images_only:
                            ask(server, f"follow-up:{image.name[:24]}", first + [
                                {"role": "assistant", "content": answer["content"]},
                                {"role": "user", "content": FOLLOW_TASK}], 128)
                    if not images_only:
                        for video in videos:
                            ask(server, f"video:{video.name[:24]}", [{"role": "user", "content": [
                                media_part(video), {"type": "text", "text": VIDEO_TASK}]}], 512)
                        ask(server, "video:long-150s", [{"role": "user", "content": [
                            media_part(long_video(output)), {"type": "text", "text": VIDEO_TASK}]}], 512)
                        ask(server, "envelope:2-videos+3-images", [{"role": "user", "content": [
                            *(media_part(path) for path in videos[:2]), *(media_part(path) for path in images),
                            {"type": "text", "text": "Describe each of these media items in one sentence, in order."}]}], 512)
                    result.update(passed=True, startup_seconds=server.startup_seconds, command=server.command)
                result["gpu_process"] = sampler.summary()
            except Exception as exc:
                result["error"] = str(exc)
                print(f"FAILED media-{name}: {exc}", flush=True)
            result.update(directory=str(directory), startup=tune.startup_facts(directory),
                          memory=bench.memory_summary(directory) if (directory / "stderr.log").exists() else None)
            report["media"][name] = result
            save()
            print(f"MEDIA {name} @ {context}: passed={result['passed']} gpu={result.get('gpu_process')}", flush=True)
    save()
    print(f"Saved vision report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
