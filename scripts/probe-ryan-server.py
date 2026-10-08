"""Exercise a local engine through its real chat API; use only the standard library."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-seconds", type=int, default=300)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path: str, body: dict | None = None) -> dict:
        raw = None if body is None else json.dumps(body).encode("utf8")
        req = urllib.request.Request(base + path, data=raw, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=120 if body else 2) as response:
            return json.load(response)

    deadline = time.monotonic() + args.startup_seconds
    process = None
    if args.pid:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        process = kernel.OpenProcess(0x1000, False, args.pid)
        if not process:
            raise RuntimeError("engine exited before its startup could be checked")
    while True:
        if process:
            exit_code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(process, ctypes.byref(exit_code)) or exit_code.value != 259:
                kernel.CloseHandle(process)
                raise RuntimeError(f"engine exited during startup: {exit_code.value}")
        try:
            health = request("/health")
            if health.get("status") == "ok":
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            if process:
                kernel.CloseHandle(process)
            raise RuntimeError("engine did not become ready; inspect its startup logs")
        time.sleep(0.5)
    if process:
        kernel.CloseHandle(process)
    models = request("/v1/models")
    model_id = models["data"][0]["id"]
    cases = [
        ("arithmetic", "What is 17 + 25? Answer with only the number.", "42"),
        ("recall", "Remember this code: BLUE-5080. What is the code? Reply with only the code.", "BLUE-5080"),
        ("generation", "Explain what a GPU does in one short sentence.", None),
    ]
    results = []
    for name, prompt, expected in cases:
        start = time.monotonic()
        response = request("/v1/chat/completions", {
            "model": model_id, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 5080, "max_tokens": 80,
            "enable_thinking": False, "stream": False,
        })
        content = response["choices"][0]["message"].get("content", "")
        if not content.strip() or "\ufffd" in content:
            raise RuntimeError(f"{name}: empty or corrupt response: {content!r}")
        if expected is not None and expected not in content:
            raise RuntimeError(f"{name}: expected {expected!r}, got {content!r}")
        if response.get("usage", {}).get("completion_tokens", 0) <= 0:
            raise RuntimeError(f"{name}: no generated tokens recorded")
        results.append({"case": name, "seconds": round(time.monotonic() - start, 3), "response": response})
        print(f"{name}: {content!r}", flush=True)
    result = {"passed": True, "health": health, "model_id": model_id,
              "props": request("/props"), "load": request("/v1/load"), "cases": results}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
