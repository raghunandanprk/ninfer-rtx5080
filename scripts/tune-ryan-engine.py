"""Find the longest context and fastest speculative settings this GPU holds for one stream.

speed: short-prompt sweep of speculative backends, draft widths and prefill routes at 32K.
probe: per memory profile, secant search for the largest --max-context (= --kv-capacity)
       that starts under the strict dedicated-VRAM policy, driven by the engine's own
       "reservation requires X bytes, but only Y" report.
long:  start each profile at its measured maximum, fill the window to within 2K tokens
       with a three-needle prompt, then decode 512 tokens at that depth.
matched: replay the smallest validated window's prompt on every profile for equal-depth decode.

Every step is saved as it finishes; rerunning skips completed steps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("bench", ROOT / "scripts/benchmark-ryan-engine.py")
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

NATIVE_CONTEXT = 262144
STEP = 1024
SWEEP_CONTEXT = 32768
LONG_MARGIN = 2048
KV_BYTES = {"bf16": 65536, "rk8v4": 26112, "rk4v4": 17920}
COMMON = ["--max-concurrency", "1", "--cuda-memory-policy", "strict", "--host-cache-mib", "5120",
          "--no-thinking", "--greedy", "--presence-penalty", "0", "--frequency-penalty", "0",
          "--log-colours", "off"]
# --embedding-q4 and --lm-head-q6 need Q8_G32 row-split tensors; these GGUF imports keep
# their embedding and output head in GGUF quants, so the engine refuses both.
LEAN = ["--gdn-state-fp16"]
PREFILL = {"chunk1024": ["--prefill-chunk", "1024"],
           "cublas2048": ["--prefill-cublas", "--prefill-chunk", "2048"],
           "cublas4096": ["--prefill-cublas", "--prefill-chunk", "4096"]}
SPEC = {
    "mtp3": ["--spec", "mtp", "--draft-tokens", "3"],
    "mtp3-head": ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"],
    "mtp4-adaptive": ["--spec", "mtp", "--draft-tokens", "4", "--adaptive-mtp"],
    "mtp4-adaptive-head": ["--spec", "mtp", "--draft-tokens", "4", "--adaptive-mtp", "--lm-head-draft"],
    "mtp5-adaptive-head": ["--spec", "mtp", "--draft-tokens", "5", "--adaptive-mtp", "--lm-head-draft"],
    "dflash2-k4": ["--spec", "dflash2", "--draft-tokens", "4", "--lm-head-draft"],
    "dflash2-k5": ["--spec", "dflash2", "--draft-tokens", "5", "--lm-head-draft"],
    "dflash2-k6": ["--spec", "dflash2", "--draft-tokens", "6", "--lm-head-draft"],
    "dflash2-k7": ["--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft"],
    "dflash2-k5-nohead": ["--spec", "dflash2", "--draft-tokens", "5"],
}


def model_of(spec: str) -> str:
    return "dflash2" if spec.startswith("dflash2") else "mtp"


def variant(spec: str, kv: str = "rk4v4", prefill: str = "chunk1024") -> dict:
    return {"model": model_of(spec), "spec": spec, "kv": kv, "prefill": prefill,
            "flags": ["--kv-dtype", kv, *PREFILL[prefill], *LEAN, *SPEC[spec]]}


class TunedServer(bench.Server):
    def __init__(self, output: Path, profile: dict, context: int):
        super().__init__(output, profile["model"], 0, context, 1, prefix_cache=True)
        self.profile = profile

    def build_command(self) -> list[str]:
        return [str(bench.ENGINE), str(bench.MODELS[self.profile["model"]]), "--host", "127.0.0.1",
                "--port", str(self.port), "--device", "0",
                "--max-context", str(self.context), "--kv-capacity", str(self.context),
                "--request-log-jsonl", str(self.output / "requests.jsonl"),
                *self.profile.get("common", COMMON), *self.profile["flags"]]


def fresh(directory: Path) -> Path:
    return directory if not directory.exists() else directory.with_name(f"{directory.name}-{time.time_ns()}")


def startup_facts(directory: Path) -> dict:
    path = directory / "stderr.log"
    text = path.read_text(encoding="utf8", errors="replace") if path.exists() else ""
    facts = {}
    if match := re.search(r"capacity \| KV ([\d,]+) tokens.*?runtime ([\d.]+) GiB \| free ([\d.]+) (B|KiB|MiB|GiB)", text):
        scale = {"B": 2**-30, "KiB": 2**-20, "MiB": 2**-10, "GiB": 1}[match[4]]
        facts.update(kv_tokens=int(match[1].replace(",", "")), runtime_gib=float(match[2]),
                     free_gib=round(float(match[3]) * scale, 3))
    if match := re.search(r"strict: admitted \d+ KV tokens, ([\d.]+) MiB device allocations, "
                          r"Shared ([\d.]+) / baseline ([\d.]+) MiB.*?(\d+) retries", text):
        facts.update(strict_device_mib=float(match[1]), shared_growth_mib=float(match[2]) - float(match[3]),
                     strict_retries=int(match[4]))
    if match := re.search(r"requires (\d+) bytes, but only (\d+) bytes", text):
        facts.update(required_bytes=int(match[1]), available_bytes=int(match[2]))
    if match := re.search(r"weights ready \| ([\d.]+) GiB", text):
        facts["weights_gib"] = float(match[1])
    if match := re.search(r"FATAL (.*)", text):
        facts["fatal"] = match[1][:400]
    return facts


def search_max_context(probe, kv: str) -> tuple[int, int, float]:
    """Largest context that starts, the smallest that fails, and the observed bytes per token.

    Two refused reservations (at 1x and 2x the native window, both rejected before any
    allocation) give the exact per-token slope and the planner's upper bound. The planner
    is optimistic: strict admission can still fail the real contiguous allocations, and
    those failures carry no byte counts, so below the bound back off exponentially until
    a start succeeds, then bisect. probe(context) returns a startup_facts dict plus "ok".
    """
    fits, fails, context = 0, NATIVE_CONTEXT + STEP, NATIVE_CONTEXT
    failures, slope, backoff = {}, KV_BYTES[kv], 4 * STEP
    for _ in range(16):
        item = probe(context)
        if item["ok"]:
            fits = max(fits, context)
        else:
            if context <= NATIVE_CONTEXT:
                fails = min(fails, context)
            if "required_bytes" in item:
                failures[context] = (item["required_bytes"], item["available_bytes"])
        if fails - fits <= STEP:
            break
        if context == NATIVE_CONTEXT and context in failures:
            context = 2 * NATIVE_CONTEXT
            continue
        if len(failures) >= 2:
            (c1, (r1, _)), (c2, (r2, _)) = sorted(failures.items())[:2]
            slope = (r2 - r1) / (c2 - c1) if r2 > r1 else slope
        bound = fails
        if failures:
            c, (required, available) = min(failures.items())
            bound = min(bound, (c - math.ceil((required - available) / slope)) // STEP * STEP + STEP)
        if bound < fails:
            guess = bound - STEP
        elif fits:
            guess = (fits + fails) // 2 // STEP * STEP
        else:
            guess, backoff = fails - backoff, backoff * 2
        context = min(max(guess, fits + STEP), fails - STEP)
    return fits, fails, slope


def corpus(chars: int) -> str:
    deps = ROOT / ".deps/ninfer-v3"
    sources = sorted((deps / "docs").rglob("*.md")) + sorted((deps / "src").rglob("*.cpp"))
    parts, total = [], 0
    for path in sources:
        parts.append(path.read_text(encoding="utf8", errors="replace"))
        total += len(parts[-1])
        if total >= chars:
            break
    return "\n\n".join(parts)


def needle_prompt(server: TunedServer, target: int, cache: Path) -> str:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf8"))["prompt"]
    text = corpus(target * 6)

    def build(length: int) -> str:
        body = text[:length]
        points = [int(len(body) * fraction) for fraction in (0.25, 0.55, 0.85)]
        for point, needle in reversed(list(zip(points, bench.NEEDLES))):
            body = body[:point] + f"\nBenchmark checkpoint code: {needle}.\n" + body[point:]
        return ("Read the following technical document. It contains three benchmark checkpoint codes.\n\n"
                + body + "\n\nList only the three benchmark checkpoint codes in the order they appear.")

    low, high, chosen, tokens = 1000, len(text), build(1000), 0
    while low <= high:
        middle = (low + high) // 2
        candidate = build(middle)
        count = server.request("/v1/messages/count_tokens", {
            "model": server.model_id, "messages": [{"role": "user", "content": candidate}],
            "thinking": {"type": "disabled"}})["input_tokens"]
        if count <= target:
            chosen, tokens, low = candidate, count, middle + 1
        else:
            high = middle - 1
    if tokens < target - 4096:
        raise RuntimeError(f"corpus too small: {tokens} tokens for a {target}-token target")
    bench.save_json(cache, {"target": target, "tokens": tokens, "prompt": chosen})
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results/2026-10-05-rtx5080-tuning")
    parser.add_argument("--phase", choices=["speed", "probe", "long", "matched", "all"], default="all")
    parser.add_argument("--profiles", help="comma-separated probe profiles for the long phase (default: all)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "tuning.json"
    report = json.loads(report_path.read_text(encoding="utf8")) if report_path.exists() else {
        "started_utc": datetime.now(timezone.utc).isoformat(), "power": bench.power_status(),
        "common_flags": COMMON, "sweep": {}, "probes": {}, "max_context": {}, "long": {}}

    def save() -> None:
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        bench.save_json(report_path, report)

    def serve_cases(label: str, profile: dict, context: int, cases: list, prefill_target: int = 0) -> dict:
        directory = fresh(output / label)
        result = {"profile": profile, "context": context, "passed": False}
        try:
            with TunedServer(directory, profile, context) as server:
                rows = [server.chat(case) for case in cases]
                generation = [row for row in rows if row["case"] in ("prose", "code")]
                result.update(cases=rows, summary=bench.summarize_rows(generation))
                if prefill_target:
                    prompt = needle_prompt(server, prefill_target, output / f"workload-{prefill_target}.json")
                    row = server.chat(("long-context", prompt, 96))
                    result["prefill"] = {key: row[key] for key in
                                         ("prompt_tokens", "ttft_ms", "prefill_tps", "decode_tps", "needles_found")}
                result.update(passed=True, startup_seconds=server.startup_seconds, command=server.command)
        except Exception as exc:
            result["error"] = str(exc)
            print(f"FAILED {label}: {exc}", flush=True)
        result.update(directory=str(directory), startup=startup_facts(directory),
                      memory=bench.memory_summary(directory) if (directory / "stderr.log").exists() else None)
        return result

    if args.phase in ("speed", "all"):
        sweep = [(name, variant(name)) for name in SPEC]
        sweep += [(f"mtp4-adaptive-head-{route}", variant("mtp4-adaptive-head", prefill=route))
                  for route in ("cublas2048", "cublas4096")]
        for name, profile in sweep:
            if report["sweep"].get(name, {}).get("passed"):
                continue
            # The three prefill routes share one cached 24K needle prompt so they time identical input.
            prefill_target = 24000 if name.startswith("mtp4-adaptive-head") else 0
            result = serve_cases(f"sweep-{name}", profile, SWEEP_CONTEXT,
                                 bench.LATENCY + bench.GENERATION, prefill_target)
            report["sweep"][name] = result
            save()
            if result["passed"]:
                s = result["summary"]
                print(f"SWEEP {name}: decode={s['weighted_decode_tps']:.1f} tok/s "
                      f"accept={s['draft_acceptance'] or 0:.3f}", flush=True)
        passed = {name: item for name, item in report["sweep"].items() if item["passed"] and name in SPEC}
        report["chosen"] = {
            backend: max((name for name in passed if model_of(name) == backend),
                         key=lambda name: passed[name]["summary"]["weighted_decode_tps"])
            for backend in ("mtp", "dflash2") if any(model_of(name) == backend for name in passed)}
        save()
        print(f"CHOSEN {report['chosen']}", flush=True)

    # cuBLAS prefill measured +1% at 24K for ~1 GiB more peak VRAM on these GGUF-quant weights,
    # so the capacity search instead prices the proposal head against context.
    profiles = {}
    for backend, spec in report.get("chosen", {}).items():
        for kv in ("rk8v4", "rk4v4"):
            profiles[f"{backend}-{kv}"] = variant(spec, kv)
        if "--lm-head-draft" in SPEC[spec]:
            lean = variant(spec, "rk4v4")
            lean["flags"] = [flag for flag in lean["flags"] if flag != "--lm-head-draft"]
            profiles[f"{backend}-rk4v4-nohead"] = lean

    if args.phase in ("probe", "all"):
        if not profiles:
            raise SystemExit("Run the speed phase first; it chooses the speculative settings to probe.")
        for name, profile in profiles.items():
            if name in report["max_context"]:
                continue

            def probe(context: int) -> dict:
                key = f"{name}@{context}"
                if key not in report["probes"]:
                    directory = fresh(output / "probe" / f"{name}-{context}")
                    item = {"profile": name, "context": context}
                    try:
                        with TunedServer(directory, profile, context) as server:
                            item.update(ok=True, startup_seconds=server.startup_seconds)
                    except Exception as exc:
                        item.update(ok=False, error=str(exc))
                    item.update(startup_facts(directory), memory=bench.memory_summary(directory))
                    report["probes"][key] = item
                    save()
                    print(f"PROBE {key}: {'ok' if item['ok'] else 'fail'} "
                          f"{item.get('free_gib', '')}{item.get('fatal', '')[:120]}", flush=True)
                return report["probes"][key]

            fits, fails, slope = search_max_context(probe, profile["kv"])
            report["max_context"][name] = {"context": fits, "first_failing": fails if fails <= NATIVE_CONTEXT else None,
                                           "bytes_per_token_observed": slope, "profile": profile}
            save()
            print(f"MAX {name}: {fits} tokens", flush=True)

    def depth_run(server: TunedServer, prompt: str) -> tuple[dict, dict]:
        needle = server.chat(("long-context", prompt, 96))
        follow = server.chat(("deep-decode", [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": needle["content"]},
            {"role": "user", "content": "Now write a detailed summary of the main topics in "
                                        "that document, at least 400 words."}], 512))
        return needle, follow

    def depth_profile(phase: str, name: str, target: int, short: bool) -> None:
        entry = report["max_context"][name]
        context, profile = entry["context"], entry["profile"]
        directory = fresh(output / f"{phase}-{name}")
        result = {"profile": profile, "context": context, "passed": False}
        try:
            with TunedServer(directory, profile, context) as server:
                if short:
                    rows = [server.chat(case) for case in bench.LATENCY + bench.GENERATION]
                    result.update(short=bench.summarize_rows([row for row in rows if row["case"] in ("prose", "code")]),
                                  short_cases=rows)
                prompt = needle_prompt(server, target, output / f"workload-{target}.json")
                needle, follow = depth_run(server, prompt)
                result.update(passed=True, startup_seconds=server.startup_seconds, long_context=needle,
                              deep_decode=follow, needles_all_found=all(needle["needles_found"]),
                              command=server.command)
        except Exception as exc:
            result["error"] = str(exc)
            print(f"FAILED {phase}-{name}: {exc}", flush=True)
        result.update(directory=str(directory), startup=startup_facts(directory),
                      memory=bench.memory_summary(directory) if (directory / "stderr.log").exists() else None)
        report.setdefault(phase, {})[name] = result
        save()
        if result["passed"]:
            print(f"{phase.upper()} {name} @ {context}: in={needle['prompt_tokens']} "
                  f"prefill={needle['prefill_tps']:.0f} tok/s TTFT={needle['ttft_ms'] / 1000:.1f}s "
                  f"needles={needle['needles_found']} deep decode={follow['decode_tps']:.1f} tok/s "
                  f"cached={follow['cached_tokens']}", flush=True)

    if args.phase in ("long", "all"):
        names = args.profiles.split(",") if args.profiles else list(report["max_context"])
        for name in names:
            if not report["long"].get(name, {}).get("passed"):
                depth_profile("long", name, report["max_context"][name]["context"] - LONG_MARGIN, short=True)

    if args.phase in ("matched", "all"):
        # Replay the smallest validated window's prompt on every profile, so deep decode
        # compares backends and KV formats at one depth rather than at each one's ceiling.
        validated = [name for name, item in report["long"].items() if item["passed"]]
        target = min(report["max_context"][name]["context"] for name in validated) - LONG_MARGIN
        report["matched_target_tokens"] = target
        for name in validated:
            if report.get("matched", {}).get(name, {}).get("passed"):
                continue
            if report["max_context"][name]["context"] - LONG_MARGIN == target:
                report.setdefault("matched", {})[name] = report["long"][name]
                save()
            else:
                depth_profile("matched", name, target, short=False)
    save()
    print(f"Saved tuning report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
