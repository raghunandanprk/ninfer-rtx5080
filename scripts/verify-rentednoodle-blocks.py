"""Compare every text/MTP quantized GGUF row with its corresponding v3 stored row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent / ".deps/ninfer-v3"
    sys.path.insert(0, str(source))
    import numpy as np
    from tools.artifact.formats import GGUF_FORMATS_BY_TYPE
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import binding_parts
    from tools.convert.sources.gguf import GGUFFile

    def sequential(begin, end, base=0):
        return np.arange(begin, end, dtype=np.int64) + base

    def query(begin, end, gate=False):
        r = sequential(begin, end)
        return (r // 256) * 512 + (256 if gate else 0) + r % 256

    def grouped(begin, end, base=0):
        r = sequential(begin, end)
        head = r // 128
        return base + ((head % 3) * 16 + head // 3) * 128 + r % 128

    mappings = [("text/token_embedding", "token_embd.weight", sequential),
                ("text/output_head", "output.weight", sequential)]
    for layer in range(65):
        p = f"text/layers/{layer}/" if layer < 64 else "mtp/layers/0/"
        g = f"blk.{layer}."
        for role in ("gate", "up", "down"):
            mappings.append((p + "mlp/" + role, g + f"ffn_{role}.weight", sequential))
        if layer % 4 == 3 or layer == 64:
            a = p + "attention/"
            mappings.extend([
                (a + "query", g + "attn_q.weight", query),
                (a + "gate", g + "attn_q.weight", lambda b, e: query(b, e, True)),
                (a + "key", g + "attn_k.weight", sequential),
                (a + "value", g + "attn_v.weight", sequential),
                (a + "output", g + "attn_output.weight", sequential),
            ])
        else:
            a = p + "gdn/"
            mappings.extend([
                (a + "query", g + "attn_qkv.weight", sequential),
                (a + "key", g + "attn_qkv.weight", lambda b, e: sequential(b, e, 2048)),
                (a + "value", g + "attn_qkv.weight", lambda b, e: grouped(b, e, 4096)),
                (a + "z", g + "attn_gate.weight", grouped),
                (a + "output", g + "ssm_out.weight", sequential),
            ])
    mappings.append(("mtp/input_projection", "blk.64.nextn.eh_proj.weight", sequential))
    checked_bytes = 0
    reports = []
    with GGUFFile(args.gguf) as gguf, Artifact(args.artifact) as artifact:
        coverage = {}
        for parameter, name, select in mappings:
            info = gguf.info(name)
            fmt = GGUF_FORMATS_BY_TYPE[info.type_id]
            columns = info.shape[1]
            row_bytes = columns // fmt.block_elements * fmt.block_bytes
            coverage.setdefault(name, np.zeros(info.shape[0], dtype=np.uint8))
            destination_row = 0
            parts = binding_parts(artifact.directory.bindings[parameter], artifact.by_id, parameter)
            for object_id, begin, end in parts:
                obj = artifact.object(object_id)
                if obj.format != fmt.name or obj.layout != "gguf_blocks_v1" or obj.shape[1] != columns:
                    raise ValueError(f"{parameter}: stored type or width differs from the GGUF")
                if begin % columns or end % columns:
                    raise ValueError(f"{parameter}: binding splits a stored row")
                rows = (end - begin) // columns
                for first in range(0, rows, 512):
                    last = min(first + 512, rows)
                    indices = select(destination_row + first, destination_row + last)
                    low, high = int(indices.min()), int(indices.max()) + 1
                    expected = gguf.read_blocks(name, low, high)[indices - low].tobytes()
                    offset = obj.offset + (begin // columns + first) * row_bytes
                    actual = artifact.read_range(offset, (last - first) * row_bytes)
                    if actual != expected:
                        raise ValueError(f"{parameter}: encoded bytes differ at rows {first}:{last}")
                    coverage[name][indices] += 1
                    checked_bytes += len(actual)
                destination_row += rows
            reports.append({"parameter": parameter, "source": name, "rows": destination_row})
        expected_names = {name for name, info in gguf.tensors.items() if info.type_id in GGUF_FORMATS_BY_TYPE}
        if coverage.keys() != expected_names:
            raise ValueError(f"source quantized tensor coverage differs: {expected_names - coverage.keys()}")
        for name, rows in coverage.items():
            if not np.all(rows == 1):
                raise ValueError(f"{name}: not every source row was verified exactly once")
    result = {"passed": True, "source": str(args.gguf.resolve()), "artifact": str(args.artifact.resolve()),
              "quantized_source_tensors": len(coverage), "logical_matrices": len(reports),
              "encoded_bytes_compared": checked_bytes, "all_source_quantized_rows_identical": True,
              "scope": "text and MTP stored quantized blocks; generated proposal head and direct-value convention transforms excluded"}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
