"""Benchmark the packaged Windows engine through its real streaming chat API.

Uses the same decode-rate boundary as Ryan's tools/bench/refbench.py:
(completion_tokens - 1) / server decode seconds. Cold prefixes, warm-up exclusion,
reversed comparison order, and client-visible first-text latency are explicit.
Model files are opened only by the native engine and are never converted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "runtime-v3/engine/ninfer-serve.exe"
MODEL_DIR = Path("E:/llm/RentedNoodle-NInfer-v3")
MODELS = {
    "none": MODEL_DIR / "qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer",
    "mtp": MODEL_DIR / "qwen3.8-27b-orcarouter-iq3-xxs-mtp-only.ninfer",
    "dflash2": MODEL_DIR / "qwen3.8-27b-orcarouter-iq3-xxs-mtp-dflash2.ninfer",
}
LATENCY = [
    ("chat-explain", "Explain what a GPU does in two short sentences.", 96),
    ("chat-recall", "Remember the code BLUE-5080. Reply with only the code.", 64),
    ("chat-advice", "Give three brief practical tips for reducing laptop GPU heat.", 96),
]
GENERATION = [
    ("prose", "Write a detailed practical guide of at least 1,200 words explaining "
     "how local GPU inference works. Cover model loading, prompt processing, "
     "token generation, GPU memory, and latency. Use specific examples and "
     "paragraphs, and continue until the guide is complete.", 512),
    ("code", "Implement a complete Python LRU cache from scratch using a doubly "
     "linked list and a dictionary. Include get, put, deletion, capacity "
     "handling, and a comprehensive unittest suite. Explain the invariants "
     "and complexity after the code. Provide the full implementation.", 512),
]
NEEDLES = ["ORCA-V3-5080", "DRIFT-MTP-27B", "RUNTIME-BF16-120A"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SMI = shutil.which("nvidia-smi.exe") or str(Path(os.environ["SystemRoot"]) / "System32/nvidia-smi.exe")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def power_status() -> dict:
    class Power(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte), ("system_flag", ctypes.c_ubyte),
            ("seconds", ctypes.c_ulong), ("full_seconds", ctypes.c_ulong),
        ]
    status = Power()
    ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    return {"read_ok": bool(ok), "ac_power": status.ac, "battery_percent": status.percent}


class Server:
    def __init__(self, output: Path, backend: str, draft: int, context: int, concurrency: int,
                 ngram: int | None = None, kv_dtype: str = "bf16", prefix_cache: bool = False):
        self.output, self.backend, self.draft = output, backend, draft
        self.context, self.concurrency = context, concurrency
        self.ngram = ngram
        self.kv_dtype, self.prefix_cache = kv_dtype, prefix_cache
        self.process = self.monitor = None
        self.handles = []
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def request(self, path: str, body: dict | None = None) -> dict:
        raw = None if body is None else json.dumps(body).encode("utf8")
        req = urllib.request.Request(self.base + path, raw, {"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=600 if body else 2) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if path == "/health":
                raise
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf8', 'replace')}") from exc

    def build_command(self) -> list[str]:
        command = [
            str(ENGINE), str(MODELS[self.backend]), "--host", "127.0.0.1",
            "--port", str(self.port), "--device", "0",
            "--max-context", str(self.context), "--kv-capacity", str(self.context * self.concurrency),
            "--max-concurrency", str(self.concurrency), "--prefill-chunk", "256",
            "--kv-dtype", self.kv_dtype, "--no-thinking", "--greedy",
            "--presence-penalty", "0", "--frequency-penalty", "0",
            "--request-log-jsonl", str(self.output / "requests.jsonl"),
            "--log-colours", "off",
        ]
        if self.prefix_cache:
            command += ["--host-cache-mib", "6144" if self.backend == "dflash2" else "5120"]
        else:
            command += ["--no-prefix-reuse"]
        if self.backend != "none":
            command += ["--spec", self.backend, "--draft-tokens", str(self.draft), "--lm-head-draft"]
        if self.ngram is not None:
            command += ["--ngram-draft-tokens", str(self.ngram)]
        return command

    def __enter__(self):
        self.output.mkdir(parents=True)
        self.command = self.build_command()
        env = dict(os.environ)
        windows_root = os.environ["SystemRoot"]
        env["PATH"] = os.pathsep.join([str(Path(windows_root) / "System32"), windows_root])
        env["NINFER_PREFILL_ALIGN"] = "0"
        for name in ("CUDA_LAUNCH_BLOCKING", "NINFER_PROMPT_FAST", "NINFER_LM_HEAD_Q4"):
            env.pop(name, None)
        save_json(self.output / "command.json", {"arguments": self.command, "path": env["PATH"]})
        so = (self.output / "stdout.log").open("w", encoding="utf8")
        se = (self.output / "stderr.log").open("w", encoding="utf8")
        self.handles.extend([so, se])
        started = time.perf_counter()
        try:
            telemetry = (self.output / "gpu.csv").open("w", encoding="utf8")
            self.handles.append(telemetry)
            self.monitor = subprocess.Popen(
                [SMI, "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,"
                 "clocks.sm,power.draw,utilization.gpu", "--format=csv,noheader,nounits", "--loop-ms=1000"],
                stdout=telemetry, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
            self.process = subprocess.Popen(
                self.command, cwd=ENGINE.parent, env=env, stdout=so, stderr=se, creationflags=NO_WINDOW)
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {self.process.returncode}")
                try:
                    if self.request("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.perf_counter() - started > 240:
                    raise TimeoutError("server startup exceeded 240 seconds")
                time.sleep(0.25)
            self.startup_seconds = time.perf_counter() - started
            self.model_id = self.request("/v1/models")["data"][0]["id"]
            save_json(self.output / "props.json", self.request("/props"))
            self.chat(("warmup", "Explain GPU parallelism in a short paragraph.", 96))
            print(f"READY {self.output.name}: {self.startup_seconds:.1f}s", flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for process in (self.process, self.monitor):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
        for handle in self.handles:
            handle.close()
        self.handles.clear()

    def chat(self, case: tuple[str, str, int]) -> dict:
        name, prompt, limit = case
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body = {
            "model": self.model_id, "messages": messages,
            "temperature": 0, "seed": 5080, "max_tokens": limit,
            "enable_thinking": False, "reasoning_effort": "none",
            "presence_penalty": 0, "frequency_penalty": 0,
            "stream": True, "stream_options": {"include_usage": True, "include_obfuscation": False},
        }
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     json.dumps(body).encode("utf8"), {"Content-Type": "application/json"})
        start = time.perf_counter()
        text, usage, timings, finish, first = [], None, None, None, None
        try:
            with self.opener.open(req, timeout=600) as response:
                for raw in response:
                    if not raw.startswith(b"data: "):
                        continue
                    data = raw[6:].strip()
                    if data == b"[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("timings"):
                        timings = chunk["timings"]
                    for choice in chunk.get("choices", []):
                        content = choice.get("delta", {}).get("content") or ""
                        if content:
                            if first is None:
                                first = time.perf_counter()
                            text.append(content)
                        finish = choice.get("finish_reason") or finish
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf8', 'replace')}") from exc
        end = time.perf_counter()
        content = "".join(text)
        if not usage or not timings or not content.strip() or first is None:
            raise RuntimeError(f"{name}: incomplete streaming response")
        if "\ufffd" in content or usage["completion_tokens"] <= 1:
            raise RuntimeError(f"{name}: empty, corrupt, or too short output")
        if not self.prefix_cache and timings.get("cache_n", 0) != 0:
            raise RuntimeError(f"{name}: prefix reuse contaminated measurement")
        generated = usage["completion_tokens"]
        row = {
            "case": name, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": generated,
            "cached_tokens": timings.get("cache_n", 0),
            "output_limit": limit, "ttft_ms": 1000 * (first - start), "wall_seconds": end - start,
            "prefill_ms": timings["prompt_ms"], "decode_ms": timings["predicted_ms"],
            "prefill_tps": timings["prompt_per_second"], "decode_tps": timings["predicted_per_second"],
            "drafted": timings.get("draft_n", 0), "accepted": timings.get("draft_n_accepted", 0),
            "finish_reason": finish, "content": content,
            "output_sha256": hashlib.sha256(content.encode("utf8")).hexdigest(),
        }
        if name == "long-context":
            row["needles_found"] = [needle in content for needle in NEEDLES]
        save_json(self.output / f"{name}.json", row)
        if name != "warmup":
            print(f"  {name}: in={row['prompt_tokens']} out={generated} "
                  f"TTFT={row['ttft_ms']:.0f}ms decode={row['decode_tps']:.1f} tok/s", flush=True)
        return row

    def long_case(self, target_tokens: int = 6000) -> tuple[str, str, int]:
        corpus = (ROOT / ".deps/ninfer-v3/docs/maintainer/engine-architecture.md").read_text(encoding="utf8")
        corpus += "\n\n" + (ROOT / ".deps/ninfer-v3/README.en.md").read_text(encoding="utf8")
        corpus += "\n\n" + (ROOT / ".deps/ninfer-v3/docs/serving.md").read_text(encoding="utf8")
        if target_tokens > 12000:
            for path in sorted((ROOT / ".deps/ninfer-v3/docs").rglob("*.md")):
                corpus += "\n\n" + path.read_text(encoding="utf8")
                if len(corpus) >= target_tokens * 8:
                    break
        def prompt_for(length: int) -> str:
            text = corpus[:length]
            points = [int(len(text) * fraction) for fraction in (0.25, 0.55, 0.85)]
            for point, needle in reversed(list(zip(points, NEEDLES))):
                text = text[:point] + f"\nBenchmark checkpoint code: {needle}.\n" + text[point:]
            return ("Read the following technical document. It contains three benchmark checkpoint codes.\n\n"
                    + text + "\n\nList only the three benchmark checkpoint codes in the order they appear.")
        low, high = 1000, len(corpus)
        chosen = prompt_for(low)
        while low <= high:
            middle = (low + high) // 2
            candidate = prompt_for(middle)
            count = self.request("/v1/messages/count_tokens", {
                "model": self.model_id, "messages": [{"role": "user", "content": candidate}],
                "thinking": {"type": "disabled"},
            })["input_tokens"]
            if count <= target_tokens:
                chosen, low = candidate, middle + 1
            else:
                high = middle - 1
        return "long-context", chosen, 96


def summarize_rows(rows: list[dict]) -> dict:
    decode_ms = sum(row["decode_ms"] for row in rows)
    prefill_ms = sum(row["prefill_ms"] for row in rows)
    drafted, accepted = sum(row["drafted"] for row in rows), sum(row["accepted"] for row in rows)
    return {
        "samples": len(rows), "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "weighted_decode_tps": 1000 * sum(row["completion_tokens"] - 1 for row in rows) / decode_ms,
        "weighted_prefill_tps": 1000 * sum(row["prompt_tokens"] for row in rows) / prefill_ms,
        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in rows),
        "draft_acceptance": accepted / drafted if drafted else None,
    }


def memory_summary(output: Path) -> dict:
    text = (output / "stderr.log").read_text(encoding="utf8", errors="replace")
    capacity = next((line for line in text.splitlines() if "capacity |" in line), "")
    weights = re.search(r"weights ready \| ([\d.]+ GiB)", text)
    result = {"cuda_capacity_log": capacity, "weights": weights.group(1) if weights else None}
    gpu_rows = list(csv.reader((output / "gpu.csv").read_text().splitlines())) if (output / "gpu.csv").exists() else []
    def numbers(index: int) -> list[float]:
        values = []
        for row in gpu_rows:
            try:
                values.append(float(row[index].strip()))
            except (IndexError, ValueError):
                pass
        return values
    temperatures, clocks, watts = numbers(3), numbers(4), numbers(5)
    if temperatures:
        result["peak_temperature_c"] = max(temperatures)
    if clocks:
        result["median_sm_clock_mhz"] = statistics.median(clocks)
    if watts:
        result["peak_power_w"] = max(watts)
    used = numbers(1)
    if used and max(used) > 0:
        result["peak_device_used_mib"] = max(used)
    else:
        result["memory_used_counter_available"] = False
    return result


def refresh_summary(report: dict) -> None:
    compared = [profile for profile in report["profiles"] if profile["phase"] == "compare" and profile["passed"]]
    reference_profile = next((profile for profile in compared if profile["backend"] == "none"), None)
    reference = {row["case"]: row["output_sha256"] for row in reference_profile["cases"]} if reference_profile else {}
    summary = []
    for backend in ("none", "mtp", "dflash2"):
        profiles = [profile for profile in compared if profile["backend"] == backend]
        if not profiles:
            continue
        rows = [row for profile in profiles for row in profile["cases"]]
        generation = [row for row in rows if row["case"] in ("prose", "code")]
        latency = [row for row in rows if row["case"].startswith("chat-")]
        long_context = [row for row in rows if row["case"] == "long-context"]
        first = {row["case"]: row["output_sha256"] for row in profiles[0]["cases"]}
        summary.append({
            "backend": backend, "repetitions": len(profiles),
            "generation": summarize_rows(generation),
            "short_chat_median_ttft_ms": statistics.median(row["ttft_ms"] for row in latency),
            "short_chat_samples": len(latency),
            "long_prompt_tokens": [row["prompt_tokens"] for row in long_context],
            "long_median_ttft_ms": statistics.median(row["ttft_ms"] for row in long_context),
            "long_prefill_tps": 1000 * sum(row["prompt_tokens"] for row in long_context)
                                  / sum(row["prefill_ms"] for row in long_context),
            "long_retrieval_all_pass": all(all(row["needles_found"]) for row in long_context),
            "repeat_outputs_identical_within_mode": all(
                row["output_sha256"] == first[row["case"]] for row in rows),
            "cases_differing_from_baseline": sorted({
                row["case"] for row in rows if row["output_sha256"] != reference.get(row["case"])}),
            "peak_gpu_used_mib": max(profile["memory"].get("peak_device_used_mib", 0) for profile in profiles),
            "peak_temperature_c": max(profile["memory"].get("peak_temperature_c", 0) for profile in profiles),
        })
    report["comparison_summary"] = summary
    report["quality_limit"] = (
        "Performance measurements and retrieval checks; no broad model-quality evaluation. "
        "Greedy long-generation outputs are repeatable within each comparison mode but "
        "are not byte-identical across baseline/MTP/DFlash2.")
    report["concurrency_summary"] = [
        {key: profile.get(key) for key in ("label", "backend", "draft_tokens", "ngram_draft_tokens", "concurrency",
                                         "passed", "aggregate_output_tps", "error", "memory")}
        for profile in report["profiles"] if profile["phase"] in ("concurrency", "scale", "confirmation")]
    report["long_context_summary"] = [
        {"backend": profile["backend"], "context": profile["context"], "passed": profile["passed"],
         "error": profile.get("error"), "memory": profile.get("memory"),
         "requests": [{key: row[key] for key in
                       ("prompt_tokens", "completion_tokens", "ttft_ms", "prefill_tps", "decode_tps", "needles_found")}
                      for row in profile.get("cases", [])]}
        for profile in report["profiles"] if profile["phase"] in ("long", "stock-100k")]
    stock = []
    for profile in report["profiles"]:
        if profile["phase"] != "stock-100k":
            continue
        item = {"backend": profile["backend"], "context": profile["context"],
                "kv_dtype": profile.get("kv_dtype", "bf16"), "passed": profile["passed"],
                "memory": profile.get("memory")}
        match = re.search(r"reservation requires (\d+) bytes, but only (\d+) bytes",
                          profile.get("stderr_tail", ""))
        if match:
            required, available = map(int, match.groups())
            item.update(runtime_required_gib=required / 2**30, runtime_available_gib=available / 2**30,
                        runtime_shortfall_gib=(required - available) / 2**30)
            weights = (profile.get("memory") or {}).get("weights")
            if weights:
                item["estimated_weights_plus_runtime_gib"] = float(weights.split()[0]) + required / 2**30
            item["estimate_scope"] = (
                "Rounded logged weights plus exact requested runtime reservation; excludes other "
                "CUDA/driver overhead. Startup failed before inference or full runtime allocation.")
        stock.append(item)
    report["stock_100k_summary"] = stock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--phase", choices=["compare", "tune", "concurrency", "scale", "long", "stock-100k", "all"], default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("Reports must be written inside the workspace.")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf8")) if report_path.exists() else {
        "started_utc": datetime.now(timezone.utc).isoformat(), "profiles": [],
        "engine": str(ENGINE), "models": {key: str(value) for key, value in MODELS.items()},
        "power": power_status(),
        "settings": {"kv_dtype": "bf16", "prefill_chunk": 256, "temperature": 0,
                     "thinking": False, "prefix_reuse": False, "ngram_draft": "engine default",
                     "ttft_boundary": "HTTP submission to first nonempty streamed text",
                     "decode_boundary": "(completion_tokens - 1) / server decode seconds",
                     "comparison_context": 8192, "throughput_context": 2048},
    }
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in set(MODELS.values())}
    for path in set(MODELS.values()):
        with path.open("rb") as handle:
            if handle.read(8) != b"NINFER\0\3":
                raise SystemExit(f"Not a v3 artifact: {path}")
    gpu_info = subprocess.run([SMI, "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True, check=True,
                              creationflags=NO_WINDOW).stdout.strip()
    report["gpu"] = gpu_info

    def run(label: str, phase: str, backend: str, draft: int, context: int, concurrency: int,
            cases: list[tuple[str, str, int]] | None, rep: int, batches: int = 1,
            ngram: int | None = None, kv_dtype: str = "bf16", prefix_cache: bool = False) -> dict:
        previous = next((item for item in report["profiles"] if item["label"] == label and item.get("passed")), None)
        if previous:
            return previous
        result = {"label": label, "phase": phase, "backend": backend, "draft_tokens": draft,
                  "context": context, "concurrency": concurrency, "repetition": rep, "passed": False,
                  "ngram_draft_tokens": ngram, "kv_dtype": kv_dtype, "prefix_cache": prefix_cache}
        directory = output / label
        if directory.exists():
            directory = output / f"{label}-retry-{time.time_ns()}"
        try:
            with Server(directory, backend, draft, context, concurrency, ngram, kv_dtype, prefix_cache) as server:
                if cases is None:
                    target_tokens = 99000 if phase == "stock-100k" else 12000 if phase == "long" else 6000
                    workload_path = output / ("workloads-99000.json" if phase == "stock-100k" else
                                              "workloads-12000.json" if phase == "long" else "workloads.json")
                    if workload_path.exists():
                        long_case = tuple(json.loads(workload_path.read_text(encoding="utf8"))["long_context"])
                    else:
                        long_case = server.long_case(target_tokens)
                        save_json(workload_path, {"latency": LATENCY, "generation": GENERATION,
                                                 "long_context": long_case})
                    cases = [long_case] if phase in ("long", "stock-100k") else LATENCY + GENERATION + [long_case]
                rows, batch_wall = [], []
                for batch in range(batches):
                    active = [(f"{name}-batch{batch}" if batches > 1 else name, prompt, limit)
                              for name, prompt, limit in cases]
                    start = time.perf_counter()
                    if concurrency == 1:
                        rows.extend(server.chat(case) for case in active)
                    else:
                        with ThreadPoolExecutor(concurrency) as pool:
                            rows.extend(pool.map(server.chat, active))
                    batch_wall.append(time.perf_counter() - start)
                result.update(passed=True, startup_seconds=server.startup_seconds, cases=rows,
                              batch_wall_seconds=batch_wall,
                              aggregate_output_tps=sum(row["completion_tokens"] for row in rows) / sum(batch_wall),
                              summary=summarize_rows(rows), command=server.command)
                save_json(directory / "load.json", server.request("/v1/load"))
            result["memory"] = memory_summary(directory)
        except Exception as exc:
            result["error"] = str(exc)
            if (directory / "stderr.log").exists():
                result["stderr_tail"] = (directory / "stderr.log").read_text(encoding="utf8", errors="replace")[-3500:]
                result["memory"] = memory_summary(directory)
            print(f"FAILED {label}: {exc}", flush=True)
        result["directory"] = str(directory)
        report["profiles"] = [item for item in report["profiles"] if item["label"] != label] + [result]
        save_json(report_path, report)
        if result["passed"]:
            print(f"RESULT {label}: decode={result['summary']['weighted_decode_tps']:.1f} tok/s "
                  f"output={result['aggregate_output_tps']:.1f} tok/s", flush=True)
        return result

    try:
        if args.phase in ("compare", "all"):
            modes = [("none", 0), ("mtp", 3), ("dflash2", 4)]
            for rep in range(args.repetitions):
                for backend, draft in modes if rep % 2 == 0 else reversed(modes):
                    result = run(f"compare-r{rep}-{backend}{draft}", "compare", backend, draft, 8192, 1, None, rep)
                    if not result["passed"]:
                        raise RuntimeError(f"Comparison failed: {result['label']}; inspect its logs.")
        if args.phase in ("tune", "all"):
            for backend, draft in [("mtp", 2), ("dflash2", 6), ("mtp", 5),
                                   ("dflash2", 2), ("mtp", 3), ("dflash2", 4)]:
                run(f"tune-{backend}{draft}", "tune", backend, draft, 2048, 1, GENERATION, 0)
        if args.phase in ("concurrency", "all"):
            candidates = [profile for profile in report["profiles"] if profile["phase"] == "tune" and profile["passed"]]
            if not candidates:
                raise RuntimeError("Run the tuning phase before the concurrency phase.")
            selected = [{"backend": "none", "draft_tokens": 0}]
            for backend in ("mtp", "dflash2"):
                best = max((profile for profile in candidates if profile["backend"] == backend),
                           key=lambda profile: profile["aggregate_output_tps"])
                selected.append({key: best[key] for key in ("backend", "draft_tokens")})
            report["selected_throughput_configurations"] = selected
            concurrent_cases = [(f"{name}-{index}", prompt, limit)
                                for index in range(2) for name, prompt, limit in GENERATION]
            for concurrency in (1, 2, 4):
                for selected_mode in selected if concurrency != 2 else reversed(selected):
                    backend, draft = selected_mode["backend"], selected_mode["draft_tokens"]
                    run(f"throughput-c{concurrency}-{backend}{draft}", "concurrency",
                        backend, draft, 2048, concurrency, concurrent_cases, 0)
            tested = [profile for profile in report["profiles"] if profile["phase"] == "concurrency" and profile["passed"]]
            best = max(tested, key=lambda profile: profile["aggregate_output_tps"])
            report["best_throughput_configuration"] = {
                key: best[key] for key in ("backend", "draft_tokens", "concurrency", "aggregate_output_tps")}
            run(f"confirmation-c{best['concurrency']}-{best['backend']}{best['draft_tokens']}", "confirmation",
                best["backend"], best["draft_tokens"], 2048, best["concurrency"], concurrent_cases, 1, batches=2)
        if args.phase in ("scale", "all"):
            eight_cases = [(f"{name}-{index}", prompt, limit)
                           for index in range(4) for name, prompt, limit in GENERATION]
            for backend, draft in (("none", 0), ("mtp", 3)):
                result = run(f"throughput-c8-{backend}{draft}", "scale", backend, draft, 2048, 8, eight_cases, 0)
                if backend == "mtp" and not result["passed"]:
                    run("throughput-c8-mtp3-no-ngram", "scale", "mtp", 3, 2048, 8, eight_cases, 0, ngram=0)
            dflash_four = next((profile for profile in report["profiles"]
                               if profile["label"] == "throughput-c4-dflash24"), None)
            if dflash_four and not dflash_four["passed"]:
                run("throughput-c4-dflash24-no-ngram", "scale", "dflash2", 4, 2048, 4,
                    eight_cases[:4], 0, ngram=0)
            tested = [profile for profile in report["profiles"]
                      if profile["phase"] in ("concurrency", "scale") and profile["passed"]]
            best = max(tested, key=lambda profile: profile["aggregate_output_tps"])
            report["best_throughput_configuration"] = {
                key: best[key] for key in ("backend", "draft_tokens", "concurrency", "aggregate_output_tps")}
            report["best_throughput_configuration"]["ngram_draft_tokens"] = best.get("ngram_draft_tokens")
            suffix = "-no-ngram" if best.get("ngram_draft_tokens") == 0 else ""
            confirmation_cases = eight_cases if best["concurrency"] == 8 else eight_cases[:4]
            run(f"confirmation-c{best['concurrency']}-{best['backend']}{best['draft_tokens']}{suffix}", "confirmation",
                best["backend"], best["draft_tokens"], 2048, best["concurrency"], confirmation_cases, 1,
                ngram=best.get("ngram_draft_tokens"))
        if args.phase in ("long", "all"):
            for backend, draft in (("none", 0), ("mtp", 3), ("dflash2", 4)):
                run(f"long-16k-{backend}{draft}", "long", backend, draft, 16384, 1, None, 0,
                    prefix_cache=True)
        if args.phase == "stock-100k":
            for backend, draft in (("mtp", 3), ("dflash2", 4)):
                run(f"stock-100k-{backend}{draft}-bf16", "stock-100k", backend, draft, 100000, 1,
                    None, 0, prefix_cache=True)
    finally:
        report["models_unchanged_size_and_mtime"] = all(
            (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == original for path, original in before.items())
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        refresh_summary(report)
        save_json(report_path, report)
        csv_rows = []
        for profile in report["profiles"]:
            for row in profile.get("cases", []):
                csv_rows.append({**{key: profile[key] for key in
                                    ("label", "phase", "backend", "draft_tokens", "context", "concurrency")},
                                 "kv_dtype": profile.get("kv_dtype", "bf16"),
                                 "prefix_cache": profile.get("prefix_cache", False),
                                 "ngram_draft_tokens": profile.get("ngram_draft_tokens"),
                                 **{key: row[key] for key in row if key not in ("content", "needles_found")}})
        if csv_rows:
            with (output / "requests.csv").open("w", newline="", encoding="utf8") as handle:
                fields = list(dict.fromkeys(key for row in csv_rows for key in row))
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(csv_rows)
    print(f"Saved benchmark report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
