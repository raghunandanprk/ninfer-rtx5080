"""Check a model against the pinned v3 source before compiling or serving it."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="local artifact path or HTTPS resolve URL")
    parser.add_argument("--source", type=Path, default=root / ".deps/ninfer-v3")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from tools.artifact.framing import HEADER, MAGIC, PAYLOAD_ALIGNMENT
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import TensorObject, decode_directory, validate_encoding

    remote = args.artifact.startswith("https://")
    sha256 = None
    if remote:
        filename = args.artifact.rsplit("/", 1)[1]

        def prefix(count: int) -> tuple[bytes, int]:
            separator = "&" if "?" in args.artifact else "?"
            request = urllib.request.Request(
                args.artifact + f"{separator}preflight={count}",
                headers={"Range": f"bytes=0-{count - 1}"},
            )
            with urllib.request.urlopen(request, timeout=40) as response:
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                raw = response.read(count + 1)
            if (
                status != 206
                or len(raw) != count
                or not content_range.startswith(f"bytes 0-{count - 1}/")
            ):
                raise ValueError("server did not honor the bounded header request")
            return raw, int(content_range.rsplit("/", 1)[1])

        header, file_bytes = prefix(HEADER.size)
        magic, count, artifact_id = HEADER.unpack(header)
        if magic != MAGIC or not 0 < count < 8 * 1024 * 1024:
            raise ValueError("expected a bounded NInfer v3 directory")
        raw, file_bytes = prefix(HEADER.size + count)
        directory = decode_directory(raw[HEADER.size:], entry_name=filename)
        payload_offset = (
            (HEADER.size + count + PAYLOAD_ALIGNMENT - 1) // PAYLOAD_ALIGNMENT
        ) * PAYLOAD_ALIGNMENT
        if file_bytes != payload_offset + directory.files[0].payload_bytes:
            raise ValueError("published length disagrees with the v3 framing")
        if len(directory.files) != 1:
            raise ValueError("preflight needs the continuation-file inventory")
    else:
        path = Path(args.artifact).resolve()
        filename = path.name
        with Artifact(path) as artifact:
            directory = artifact.directory
            artifact_id = artifact.artifact_id
            file_bytes = artifact.file_bytes
            for index in range(1, len(directory.files)):
                artifact._file(index)  # Verify continuation IDs and exact lengths.
            for role, object_id in directory.components["text"].get("resources", {}).items():
                raw = artifact.read_object(object_id)
                if role.endswith(".json"):
                    json.loads(raw)
                elif role == "chat_template.jinja" and not raw.strip():
                    raise ValueError("empty chat template")
        if args.sha256:
            sha256 = digest(path)
            if sha256 != args.sha256.lower():
                raise ValueError(f"SHA-256 mismatch: {sha256}")

    for obj in directory.objects:
        validate_encoding(obj)
    cpp = (args.source / "src/artifact/formats.cpp").read_text(encoding="utf8")
    supported = set(re.findall(r'std::string_view\{"([^"}]+)"\}', cpp))
    tensors = [obj for obj in directory.objects if isinstance(obj, TensorObject)]
    for obj in tensors:
        if obj.format not in supported or obj.layout not in supported:
            raise ValueError(f"C++ runtime does not support {obj.format}/{obj.layout}")
    config = directory.components["text"]["config"]
    expected = {"hidden_size": 5120, "num_hidden_layers": 64, "vocab_size": 248320,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256}
    if config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError("expected the supported Qwen3.8 dense architecture")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unexpected text geometry: {key}={config.get(key)!r}")
    resources = directory.components["text"].get("resources", {})
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if name not in resources:
            raise ValueError(f"missing frontend resource: {name}")
    result = {
        "artifact": args.artifact if remote else str(path),
        "filename": filename, "version": 3, "artifact_id": artifact_id.hex(),
        "bytes": file_bytes, "sha256": sha256, "remote_preflight": remote,
        "pinned_source": str(args.source.resolve()), "schema_and_encodings_pass": True,
        "cpp_formats_and_layouts_pass": True, "qwen_dense_geometry_pass": True,
        "components": list(directory.components), "objects": len(directory.objects),
        "bindings": len(directory.bindings), "uses": len(directory.uses),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "name": directory.metadata.get("name"),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
