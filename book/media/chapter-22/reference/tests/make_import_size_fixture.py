#!/usr/bin/env python3
"""Create ZERO weights at SmolLM2 dimensions for isolated conversion/load sizing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_smollm2 as exporter
from import_fixture import raw_config


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 tests/make_import_size_fixture.py NEW_DIRECTORY")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    raw = raw_config()
    raw.update(vocab_size=49152, num_hidden_layers=30, hidden_size=576,
               num_attention_heads=9, num_key_value_heads=3, intermediate_size=1536,
               max_position_embeddings=8192, initializer_range=1/24, rope_theta=100000)
    c = exporter.configuration(raw)
    header = {"__metadata__": {"format": "pt", "fixture": "synthetic zeros, NOT pretrained"}}
    cursor = 0
    for source, _, shape, rows, cols in exporter.tensor_map(c):
        size = 2 * rows * cols
        header[source] = dict(dtype="BF16", shape=shape, data_offsets=[cursor, cursor+size])
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (out / "config.json").write_text(json.dumps(raw))
    (out / "README.txt").write_text("SYNTHETIC ZERO WEIGHTS. Storage test only. NOT pretrained.\n")
    with (out / "model.safetensors").open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        file.flush()
        # Extending a regular file supplies zero bytes; the filesystem may store holes.
        os.ftruncate(file.fileno(), 8 + len(encoded) + cursor)
    print(f"synthetic BF16 payload={cursor} bytes; no model-quality meaning")


if __name__ == "__main__":
    main()
