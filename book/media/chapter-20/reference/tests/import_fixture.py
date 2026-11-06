"""Small synthetic assets, never pretrained values. No network or model execution."""
from __future__ import annotations

import json
from pathlib import Path
import struct

def raw_config():
    return {
        "architectures": ["LlamaForCausalLM"], "attention_bias": False,
        "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 0,
        "hidden_act": "silu", "hidden_size": 8, "initializer_range": 0.125,
        "intermediate_size": 12, "is_llama_config": True,
        "max_position_embeddings": 8, "model_type": "llama",
        "num_attention_heads": 4, "num_hidden_layers": 2,
        "num_key_value_heads": 2, "pretraining_tp": 1, "rms_norm_eps": 1e-5,
        "rope_interleaved": False, "rope_scaling": None, "rope_theta": 10000,
        "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "transformers_version": "fixture", "use_cache": True, "vocab_size": 16
    }


def fixture_records():
    records = [("model.embed_tokens.weight", [16, 8])]
    suffixes = (
        ("input_layernorm.weight", [8]), ("self_attn.q_proj.weight", [8, 8]),
        ("self_attn.k_proj.weight", [4, 8]), ("self_attn.v_proj.weight", [4, 8]),
        ("self_attn.o_proj.weight", [8, 8]), ("post_attention_layernorm.weight", [8]),
        ("mlp.gate_proj.weight", [12, 8]), ("mlp.up_proj.weight", [12, 8]),
        ("mlp.down_proj.weight", [8, 12])
    )
    for layer in range(2):
        records.extend((f"model.layers.{layer}.{name}", shape) for name, shape in suffixes)
    return records + [("model.norm.weight", [8])]

def fixture_source(dtype="BF16"):
    records = fixture_records()
    # Deliberately reverse physical and JSON order: converter must use names.
    payload = bytearray()
    header = {"__metadata__": {"format": "pt"}}
    for p in range(len(records)-1, -1, -1):
        name, shape = records[p]
        count = shape[0] * (shape[1] if len(shape) == 2 else 1)
        begin = len(payload)
        for i in range(count):
            bf = 0x3d00 + (97 * p + i) % 256
            if (p + i) % 3 == 0:
                bf |= 0x8000
            payload.extend(struct.pack("<H", bf) if dtype == "BF16"
                           else struct.pack("<I", bf << 16))
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[begin, len(payload)])
    return header, bytes(payload)


def write_source(directory, header=None, payload=None, config=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if header is None:
        header, payload = fixture_source()
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (directory / "config.json").write_text(json.dumps(raw_config() if config is None else config))
    return directory

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 tests/import_fixture.py NEW_DIRECTORY")
    if Path(sys.argv[1]).exists():
        raise SystemExit("refusing to overwrite an existing fixture directory")
    write_source(sys.argv[1])
