"""Verify an AMD64 package's direct and delay-loaded DLL dependency closure."""
import argparse
import json
import os
import pathlib
import re
import struct


def pe_imports(path):
    with path.open("rb") as f:
        def read(offset, size):
            f.seek(offset)
            data = f.read(size)
            if len(data) != size:
                raise ValueError(f"Truncated PE file: {path}")
            return data

        pe = struct.unpack("<I", read(0x3C, 4))[0]
        if read(pe, 4) != b"PE\0\0":
            raise ValueError(f"Invalid PE signature: {path}")
        machine, sections = struct.unpack("<HH", read(pe + 4, 4))
        if machine != 0x8664:
            raise ValueError(f"Expected Windows x64 PE: {path} ({machine:#x})")
        optional_size = struct.unpack("<H", read(pe + 20, 2))[0]
        optional = read(pe + 24, optional_size)
        if struct.unpack_from("<H", optional)[0] != 0x20B:
            raise ValueError(f"Expected PE32+: {path}")
        image_base = struct.unpack_from("<Q", optional, 24)[0]
        header_size = struct.unpack_from("<I", optional, 60)[0]
        ranges = []
        for i in range(sections):
            section = read(pe + 24 + optional_size + i * 40, 40)
            virtual_size, rva, raw_size, offset = struct.unpack_from("<IIII", section, 8)
            ranges.append((rva, max(virtual_size, raw_size), offset))

        def rva_offset(rva):
            if rva < header_size:
                return rva
            for start, size, offset in ranges:
                if start <= rva < start + size:
                    return offset + rva - start
            raise ValueError(f"Unmapped PE RVA {rva:#x} in {path}")

        def name(rva):
            f.seek(rva_offset(rva))
            return f.read(512).split(b"\0", 1)[0].decode("ascii").lower()

        imports = set()
        for directory, descriptor_size, name_offset in ((1, 20, 12), (13, 32, 4)):
            rva, size = struct.unpack_from("<II", optional, 112 + directory * 8)
            if not rva:
                continue
            offset = rva_offset(rva)
            for i in range(min(size // descriptor_size + 1, 4096)):
                entry = read(offset + i * descriptor_size, descriptor_size)
                if not any(entry):
                    break
                name_rva = struct.unpack_from("<I", entry, name_offset)[0]
                if directory == 13 and not (struct.unpack_from("<I", entry)[0] & 1):
                    name_rva -= image_base
                imports.add(name(name_rva))
        return sorted(imports)


def inspect(root):
    files = {p.name.lower(): p for p in root.iterdir() if p.suffix.lower() in (".exe", ".dll")}
    if "ninfer-serve.exe" not in files:
        raise ValueError("ninfer-serve.exe is missing")
    system = pathlib.Path(os.environ["SystemRoot"]) / "System32"
    dependencies, missing, external = {}, [], set()
    for filename, path in sorted(files.items()):
        dependencies[filename] = pe_imports(path)
        for imported in dependencies[filename]:
            if imported in files:
                continue
            if re.match(r"(?:vcruntime|msvcp|concrt)\d", imported):
                missing.append({"file": filename, "dll": imported})
            elif imported.startswith(("api-ms-", "ext-ms-")) or (system / imported).is_file():
                external.add(imported)
            else:
                missing.append({"file": filename, "dll": imported})
    return {"passed": not missing, "platform": "Windows x64", "packaged_dlls": len(files) - 1,
            "missing": missing, "system_dependencies": sorted(external), "imports": dependencies}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = inspect(args.runtime.resolve())
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "platform", "packaged_dlls", "missing")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
