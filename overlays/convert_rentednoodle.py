"""Convert RentedNoodle's OrcaRouter GSQ/RCO IQ3_XXS GGUF to NInfer.

The converter deliberately preserves the RentedNoodle trunk and embedded MTP
values as the value source.  The RentedNoodle BF16 mmproj is the authoritative Vision source. The existing
NInfer GSQ3 artifact is only a donor for components absent from the RentedNoodle
release (principally DFlash2), the draft shortlist IDs, and fallback frontend resources.

Target source:
  RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored
  Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf

The RCO allocation is mapped through NInfer's existing gsqrco inventory:
IQ3/IQ2-family trunk tensors become the registered Q3G128 grid; wider source
tensors are promoted to Q4/Q5/Q6 as required by the fused NInfer objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_8_27b import inventory_gsq3
from tools.convert.qwen3_8_27b import inventory_gsqrco as inventory
from tools.convert.qwen3_8_27b.gsqrco_quantize import quantize_and_encode
from tools.convert.qwen3_8_27b.gsqrco_source import GgufSource
from tools.convert.qwen3_8_27b.recipe_gsq3 import attention_indices

RECIPE_ID = "qwen3_8_27b_rentednoodle_orcarouter_gsqrco_iq3xxs-v1"
SOURCE_REPOSITORY = "RentedNoodle/Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-Uncensored"
SOURCE_REVISION = "main"
SOURCE_FILE = "Qwen3.8-27B-OrcaRouter-GSQ-RCO-IQ3_XXS-v2.0.gguf"
SOURCE_SHA256 = "41ad7dfb3f4397d626408a96e88a46c5964e88bd6c4240c191e1131af92ea8cd"

MTP_SOURCE_NAMES = {
    "mtp/input_projection": "blk.64.nextn.eh_proj.weight",
    "mtp/embedding_norm": "blk.64.nextn.enorm.weight",
    "mtp/hidden_norm": "blk.64.nextn.hnorm.weight",
    "mtp/layer/input_norm": "blk.64.nextn.attn_norm.weight",
    "mtp/layer/attention/query_norm": "blk.64.nextn.attn_q_norm.weight",
    "mtp/layer/attention/key_norm": "blk.64.nextn.attn_k_norm.weight",
    "mtp/layer/attention/output": "blk.64.nextn.attn_output.weight",
    "mtp/layer/post_attention_norm": "blk.64.nextn.ffn_norm.weight",
    "mtp/layer/mlp/down": "blk.64.nextn.ffn_down.weight",
    "mtp/final_norm": "blk.64.nextn.shared_head_norm.weight",
}

MTP_MATRIX_SOURCES = (
    "blk.64.nextn.eh_proj.weight",
    "blk.64.nextn.attn_q.weight",
    "blk.64.nextn.attn_k.weight",
    "blk.64.nextn.attn_v.weight",
    "blk.64.nextn.attn_output.weight",
    "blk.64.nextn.ffn_gate.weight",
    "blk.64.nextn.ffn_up.weight",
    "blk.64.nextn.ffn_down.weight",
)

MTP_VECTOR_SOURCES = (
    "blk.64.nextn.enorm.weight",
    "blk.64.nextn.hnorm.weight",
    "blk.64.nextn.attn_norm.weight",
    "blk.64.nextn.attn_q_norm.weight",
    "blk.64.nextn.attn_k_norm.weight",
    "blk.64.nextn.ffn_norm.weight",
    "blk.64.nextn.shared_head_norm.weight",
)


class MmprojSource:
    """Read the RentedNoodle Qwen3.8 BF16 mmproj into NInfer Vision objects."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader = GGUFReader(str(path))
        self._tensors = {tensor.name: tensor for tensor in self._reader.tensors}
        if len(self._tensors) != 334:
            raise ValueError(
                f"{self.path}: expected 334 Qwen3.8 mmproj tensors, "
                f"found {len(self._tensors)}"
            )

    def tensor(self, name: str) -> torch.Tensor:
        try:
            tensor = self._tensors[name]
        except KeyError as exc:
            raise ValueError(f"RentedNoodle mmproj is missing {name!r}") from exc
        values = dequantize(np.asarray(tensor.data), tensor.tensor_type)
        shape = tuple(reversed(tuple(int(dim) for dim in tensor.shape)))
        array = np.asarray(values)
        if array.size != int(np.prod(shape)):
            raise ValueError(
                f"{name}: decoded element count {array.size} does not match {shape}"
            )
        return torch.from_numpy(np.ascontiguousarray(array.reshape(shape)))

    def object_tensor(self, object_name: str) -> torch.Tensor:
        if object_name == "vision/patch_embedding":
            first = self.tensor("v.patch_embd.weight")
            second = self.tensor("v.patch_embd.weight.1")
            if first.shape != (1152, 3, 16, 16) or second.shape != first.shape:
                raise ValueError(
                    "RentedNoodle mmproj patch halves do not match Qwen3.8 geometry"
                )
            # llama.cpp stores the temporal conv3d kernel as two conv2d halves.
            # Reconstruct [out, channel, temporal, y, x], then flatten exactly
            # like NInfer's safetensors converter does for patch_embed.proj.weight.
            return torch.stack((first, second), dim=2).reshape(1152, 1536)
        if object_name == "vision/patch_embedding_bias":
            return self.tensor("v.patch_embd.bias")
        if object_name == "vision/position_embedding":
            return self.tensor("v.position_embd.weight")

        if object_name.startswith("vision/layers/"):
            parts = object_name.split("/")
            layer = int(parts[2])
            suffix = "/".join(parts[3:])
            suffix_map = {
                "attention/qkv": "attn_qkv.weight",
                "attention/qkv_bias": "attn_qkv.bias",
                "attention/output": "attn_out.weight",
                "attention/output_bias": "attn_out.bias",
                "mlp/fc1": "ffn_up.weight",
                "mlp/fc1_bias": "ffn_up.bias",
                "mlp/fc2": "ffn_down.weight",
                "mlp/fc2_bias": "ffn_down.bias",
                "norm1/weight": "ln1.weight",
                "norm1/bias": "ln1.bias",
                "norm2/weight": "ln2.weight",
                "norm2/bias": "ln2.bias",
            }
            try:
                gguf_suffix = suffix_map[suffix]
            except KeyError as exc:
                raise ValueError(f"unsupported NInfer Vision object {object_name}") from exc
            return self.tensor(f"v.blk.{layer}.{gguf_suffix}")

        merger_map = {
            "vision/merger/fc1": "mm.0.weight",
            "vision/merger/fc1_bias": "mm.0.bias",
            "vision/merger/fc2": "mm.2.weight",
            "vision/merger/fc2_bias": "mm.2.bias",
            "vision/merger/norm/weight": "v.post_ln.weight",
            "vision/merger/norm/bias": "v.post_ln.bias",
        }
        try:
            return self.tensor(merger_map[object_name])
        except KeyError as exc:
            raise ValueError(f"unsupported NInfer Vision object {object_name}") from exc




def _raw_tensor(source: GgufSource, name: str) -> torch.Tensor:
    try:
        tensor = source._tensors[name]
    except KeyError as exc:
        raise ValueError(f"RentedNoodle GGUF is missing required tensor {name!r}") from exc
    values = dequantize(np.asarray(tensor.data), tensor.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(values))


def _mtp_tensor(source: GgufSource, object_name: str) -> torch.Tensor:
    if object_name == "mtp/layer/attention/query_key_gate_value":
        query_rows, gate_rows = attention_indices()
        q = source.matrix("blk.64.nextn.attn_q.weight")
        k = source.matrix("blk.64.nextn.attn_k.weight")
        v = source.matrix("blk.64.nextn.attn_v.weight")
        return torch.cat((q[list(query_rows)], k, q[list(gate_rows)], v), dim=0)
    if object_name == "mtp/layer/mlp/gate_up":
        return torch.cat(
            (
                source.matrix("blk.64.nextn.ffn_gate.weight"),
                source.matrix("blk.64.nextn.ffn_up.weight"),
            ),
            dim=0,
        )
    try:
        source_name = MTP_SOURCE_NAMES[object_name]
    except KeyError as exc:
        raise ValueError(f"no RentedNoodle MTP mapping for {object_name}") from exc
    return _raw_tensor(source, source_name)


def _frontend_payloads(
    donor: Artifact,
    frontend_dir: Path | None,
) -> dict[str, bytes]:
    resources: dict[str, bytes] = {}
    for spec in inventory.RESOURCE_SPECS:
        payload: bytes | None = None
        if frontend_dir is not None:
            filename = spec.name.removeprefix("frontend/")
            candidates = [frontend_dir / filename]
            if filename == "chat_template.jinja":
                candidates.insert(0, frontend_dir / "froggeric-qwen3.8-tool-use.jinja")
            for candidate in candidates:
                if candidate.is_file():
                    payload = candidate.read_bytes()
                    break
        if payload is None:
            payload = bytes(donor.payload(donor.find(spec.name)))
        if not payload:
            raise ValueError(f"frontend resource {spec.name} is empty")
        resources[spec.name] = payload
    return resources


def _draft_token_ids(donor: Artifact) -> torch.Tensor:
    payload = bytes(donor.payload(donor.find("text/draft_head_token_ids")))
    token_ids = np.frombuffer(payload, dtype="<i4").copy()
    if token_ids.size != 131072:
        raise ValueError(
            f"donor draft shortlist has {token_ids.size} rows; expected 131072"
        )
    return torch.from_numpy(token_ids.astype(np.int64, copy=False))


def _validate_source(source: GgufSource) -> None:
    names = set(source.tensor_types)
    missing = [
        name
        for name in (*MTP_MATRIX_SOURCES, *MTP_VECTOR_SOURCES)
        if name not in names
    ]
    if missing:
        raise ValueError(
            "RentedNoodle GGUF does not contain the expected embedded MTP block: "
            + ", ".join(missing)
        )

    for object_name in inventory.ported_object_names():
        for source_name in inventory.object_source_tensors(object_name) or ():
            if source_name not in names:
                raise ValueError(
                    f"RentedNoodle GGUF is missing trunk tensor {source_name!r}"
                )


def convert(
    gguf_path: str | Path,
    donor_artifact_path: str | Path,
    mmproj_path: str | Path,
    out_path: str | Path,
    *,
    frontend_dir: str | Path | None = None,
    device: str = "cuda",
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    source = GgufSource(gguf_path)
    _validate_source(source)
    vision = MmprojSource(mmproj_path)

    source_types = source.tensor_types
    specs = inventory.build_object_specs(source_types)
    ported = inventory.ported_object_names()
    spec_by_name = {
        spec.name: spec
        for spec in inventory.build_tensor_specs(source_types)
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    encoded_trunk = 0
    encoded_mtp = 0
    encoded_vision = 0
    copied = 0
    draft_payload: bytes | None = None

    with Artifact.open(donor_artifact_path) as donor:
        resources = _frontend_payloads(
            donor,
            Path(frontend_dir) if frontend_dir is not None else None,
        )
        plan = family_conversion.build_object_plan(specs, resources)
        draft_ids = _draft_token_ids(donor)

        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            plan.specs,
        ) as writer:
            if writer.objects != plan.objects:
                raise RuntimeError("writer object plan differs from preflight")

            total = len(writer.objects)
            for index, obj in enumerate(writer.objects, start=1):
                if obj.kind == "resource":
                    payload = resources[obj.name]

                elif obj.name in ported:
                    matrix = source.object_matrix(obj.name)
                    payload = quantize_and_encode(
                        matrix, obj.format, device=resolved_device
                    )
                    encoded_trunk += 1

                    if obj.name == "text/output_head":
                        draft_matrix = matrix.index_select(0, draft_ids)
                        draft_spec = spec_by_name["text/draft_head"]
                        draft_payload = quantize_and_encode(
                            draft_matrix,
                            draft_spec.format,
                            device=resolved_device,
                        )
                        del draft_matrix
                    del matrix

                elif obj.name == "text/draft_head":
                    if draft_payload is None:
                        raise RuntimeError(
                            "draft head reached before RentedNoodle output head was encoded"
                        )
                    payload = draft_payload
                    draft_payload = None

                elif obj.name == "text/draft_head_token_ids":
                    payload = bytes(donor.payload(donor.find(obj.name)))
                    copied += 1

                elif obj.name.startswith("mtp/"):
                    tensor = _mtp_tensor(source, obj.name)
                    spec = spec_by_name[obj.name]
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{obj.name}: RentedNoodle MTP shape "
                            f"{tuple(tensor.shape)} != expected {spec.shape}"
                        )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                    encoded_mtp += 1

                elif obj.name.startswith("vision/"):
                    tensor = vision.object_tensor(obj.name)
                    spec = spec_by_name[obj.name]
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{obj.name}: RentedNoodle mmproj shape "
                            f"{tuple(tensor.shape)} != expected {spec.shape}"
                        )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                    encoded_vision += 1

                else:
                    # DFlash2 is not distributed by RentedNoodle. Text-side
                    # non-matrix values are retained from the validated donor
                    # where the existing RCO converter does not port them.
                    payload = bytes(donor.payload(donor.find(obj.name)))
                    copied += 1

                writer.write(obj.name, payload)
                del payload
                if index % 64 == 0 or index == total:
                    print(f"[{index}/{total}] {obj.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = {
        "recipe_id": RECIPE_ID,
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "file": SOURCE_FILE,
            "sha256": SOURCE_SHA256,
            "gguf": str(Path(gguf_path).resolve()),
        },
        "donor_artifact": str(Path(donor_artifact_path).resolve()),
        "vision": {
            "mmproj": str(Path(mmproj_path).resolve()),
            "source": "RentedNoodle mmproj/mmproj-Qwen3.8-27B-BF16.gguf",
        },
        "frontend_dir": (
            str(Path(frontend_dir).resolve()) if frontend_dir is not None else None
        ),
        "device": str(resolved_device),
        "encoded_trunk_objects": encoded_trunk,
        "encoded_mtp_objects": encoded_mtp,
        "encoded_vision_objects": encoded_vision,
        "copied_objects": copied,
        "final_bytes": final_bytes,
        "elapsed_seconds": round(elapsed, 3),
        "notes": [
            "trunk matrices are sourced from the RentedNoodle v2.0 GGUF",
            "embedded blk.64 MTP values are sourced from the RentedNoodle v2.0 GGUF",
            "optimized NInfer draft_head is regenerated from the RentedNoodle output head",
            "all 333 NInfer Vision tensors are rebuilt from the RentedNoodle BF16 mmproj",
            "the mmproj temporal patch halves are reconstructed into the NInfer conv3d layout",
            "DFlash2 is copied from the validated GSQ3 donor artifact",
            "DFlash2 is not OrcaRouter-native and requires separate acceptance validation",
        ],
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; "
        f"trunk={encoded_trunk} mtp={encoded_mtp} vision={encoded_vision} "
        f"copied={copied}; "
        f"report={report_path}",
        flush=True,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--donor-artifact", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--frontend-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    convert(
        args.gguf,
        args.donor_artifact,
        args.mmproj,
        args.out,
        frontend_dir=args.frontend_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
