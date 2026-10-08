"""Assemble Microsoft's downloaded VS2022 packages for a private x64 build."""
import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYOUT = ROOT / ".deps/vs2022-layout"
DEST = ROOT / ".deps/vs2022"
MANIFEST = ROOT / ".deps/toolchain-downloads/vs2022-manifest.json"
IDS = {
    "Microsoft.VC.14.44.17.14.Tools.HostX64.TargetX64.base",
    "Microsoft.VC.14.44.17.14.Tools.HostX64.TargetX64.Res.base",
    "Microsoft.VC.14.44.17.14.CRT.Headers.base",
    "Microsoft.VC.14.44.17.14.CRT.x64.Desktop.base",
    "Microsoft.VC.14.44.17.14.CRT.x64.Store.base",
    "Microsoft.VC.14.44.17.14.CRT.Redist.X64.base",
    "Microsoft.VisualStudio.VC.CMake",
}
packages = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["packages"]
selected = [p for p in packages if p["id"] in IDS
            and p.get("language") in (None, "en-US")
            and p.get("productArch") in (None, "x86")]
if {p["id"] for p in selected} != IDS:
    raise RuntimeError("Required VS2022 packages missing from Microsoft's manifest")
DEST.mkdir(parents=True, exist_ok=True)
records = []
for package in selected:
    dirs = list(LAYOUT.glob(f'{package["id"]},version={package["version"]}*'))
    archives = [p / "payload.vsix" for p in dirs if (p / "payload.vsix").is_file()]
    if len(archives) != 1 or len(package["payloads"]) != 1:
        raise RuntimeError(f'Expected one VSIX for {package["id"]}')
    archive = archives[0]
    payload = package["payloads"][0]
    actual = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual.lower() != payload["sha256"].lower():
        raise RuntimeError(f"SHA-256 mismatch: {archive}")
    print(f'Extracting {package["id"]} {package["version"]}', flush=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not entry.filename.startswith("Contents/") or entry.is_dir():
                continue
            target = (DEST / entry.filename[len("Contents/"):]).resolve()
            if not target.is_relative_to(DEST.resolve()):
                raise RuntimeError("Unexpected archive path")
            target.parent.mkdir(parents=True, exist_ok=True)
            content = z.read(entry)
            if not target.is_file() or target.read_bytes() != content:
                target.write_bytes(content)
    records.append({"id": package["id"], "version": package["version"], "sha256": actual, "url": payload["url"]})

toolsets = list((DEST / "VC/Tools/MSVC").iterdir())
redists = list((DEST / "VC/Redist/MSVC").iterdir())
if len(toolsets) != 1 or len(redists) != 1:
    raise RuntimeError("Expected a single private v143 toolset and redist")
tools, redist = toolsets[0], redists[0]
sdk = pathlib.Path(r"C:\Program Files (x86)\Windows Kits\10")
sdk_version = "10.0.26100.0"
if not (sdk / f"Lib/{sdk_version}/um/x64/kernel32.lib").is_file():
    raise RuntimeError("The installed Windows 11 SDK 10.0.26100.0 is required")
variables = {
    "VSINSTALLDIR": str(DEST) + "\\",
    "VCINSTALLDIR": str(DEST / "VC") + "\\",
    "VCToolsInstallDir": str(tools) + "\\",
    "VCToolsRedistDir": str(redist) + "\\",
    "VCToolsVersion": tools.name,
    "VisualStudioVersion": "17.0",
    "WindowsSDKDir": str(sdk) + "\\",
    "WindowsSDKVersion": sdk_version + "\\",
    "WindowsSDKLibVersion": sdk_version + "\\",
    "UniversalCRTSdkDir": str(sdk) + "\\",
    "UCRTVersion": sdk_version,
    "VSCMD_ARG_HOST_ARCH": "x64",
    "VSCMD_ARG_TGT_ARCH": "x64",
    "INCLUDE": ";".join(str(p) for p in [tools / "include"] + [sdk / f"Include/{sdk_version}/{d}" for d in ("ucrt", "shared", "um", "winrt", "cppwinrt")]),
    "LIB": ";".join(str(p) for p in [tools / "lib/x64", sdk / f"Lib/{sdk_version}/ucrt/x64", sdk / f"Lib/{sdk_version}/um/x64"]),
    "LIBPATH": str(tools / "lib/x64"),
    "PATH": f'{tools / "bin/Hostx64/x64"};{sdk / f"bin/{sdk_version}/x64"};%PATH%',
}
batch = DEST / "VC/Auxiliary/Build/vcvars64.bat"
batch.parent.mkdir(parents=True, exist_ok=True)
batch.write_text("@echo off\n" + "\n".join(f'set "{k}={v}"' for k, v in variables.items()) + "\nexit /b 0\n", encoding="ascii")
# nvcc also checks this standard entry point while identifying the MSVC host.
(batch.parent / "vcvarsall.bat").write_bytes(batch.read_bytes())
(DEST / "toolchain-manifest.json").write_text(json.dumps({"visual_studio": "2022", "platform_toolset": "v143", "packages": records, "sdk": sdk_version}, indent=2), encoding="utf-8")
print(f"Private VS2022 v143 toolchain: {DEST}")
