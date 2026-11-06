#!/usr/bin/env python3
"""Asset-only, bounded SmolLM2 conversion. No model computation or pickle."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import urllib.request

MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
HEADER_LIMIT = 1024 * 1024
SOURCE_LIMIT = 300 * 1024 * 1024
OUTPUT_LIMIT = 600 * 1024 * 1024
CHUNK_VALUES = 65536

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated input")
    return data


def json_object(data):
    value = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def integer(value, name, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"invalid {name}: {value!r}")
    return value

def configuration(raw, *, fixture=False):
    required = {
        "architectures", "attention_bias", "attention_dropout", "bos_token_id",
        "eos_token_id", "hidden_act", "hidden_size", "initializer_range",
        "intermediate_size", "is_llama_config", "max_position_embeddings",
        "model_type", "num_attention_heads", "num_hidden_layers",
        "num_key_value_heads", "pretraining_tp", "rms_norm_eps", "rope_interleaved",
        "rope_scaling", "rope_theta", "tie_word_embeddings", "torch_dtype",
        "transformers_version", "use_cache", "vocab_size"
    }
    if not required <= raw.keys() or raw.keys() - required - {"mlp_bias", "head_dim"}:
        raise ValueError("missing or unsupported configuration fields")
    exact = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_act": "silu", "attention_bias": False, "attention_dropout": 0.0,
        "is_llama_config": True, "pretraining_tp": 1, "rope_interleaved": False,
        "rope_scaling": None, "tie_word_embeddings": True,
        "torch_dtype": "bfloat16", "use_cache": True
    }
    for key, value in exact.items():
        if raw[key] != value or (type(value) is bool and type(raw[key]) is not bool):
            raise ValueError(f"unsupported {key}")
    if raw.get("mlp_bias", False) is not False:
        raise ValueError("MLP biases are unsupported")
    keys = ("vocab_size", "num_hidden_layers", "hidden_size", "num_attention_heads",
            "num_key_value_heads", "intermediate_size")
    dims = tuple(integer(raw[k], k, 100000) for k in keys)
    v, layers, width, hq, hk, hidden = dims
    if v < 2 or layers > 128 or width % hq or hq % hk or (width // hq) % 2:
        raise ValueError("incompatible dimensions")
    if "head_dim" in raw and raw["head_dim"] != width // hq:
        raise ValueError("custom head dimension is unsupported")
    positions = integer(raw["max_position_embeddings"], "positions", 8192)
    epsilon, init, theta = (float(raw[k]) for k in
                           ("rms_norm_eps", "initializer_range", "rope_theta"))
    if not all(math.isfinite(x) and x > 0 for x in (epsilon, init, theta)) or theta <= 1:
        raise ValueError("invalid normalization/rotary parameters")
    # Require that the positive scalars survive storage in binary32.
    epsilon, init = struct.unpack("<ff", struct.pack("<ff", epsilon, init))
    if not all(math.isfinite(x) and x > 0 for x in (epsilon, init)):
        raise ValueError("scalar cannot be represented in binary32")
    for key in ("bos_token_id", "eos_token_id"):
        if type(raw[key]) is not int or not 0 <= raw[key] < v:
            raise ValueError(f"invalid {key}")
    if not fixture:
        if dims != (49152, 30, 576, 9, 3, 1536) or positions != 8192:
            raise ValueError("not the selected SmolLM2-135M configuration")
        if (epsilon != struct.unpack("<f", struct.pack("<f", 1e-5))[0] or
                init != struct.unpack("<f", struct.pack("<f", 1 / 24))[0] or
                theta != 100000 or raw["bos_token_id"] != 0 or raw["eos_token_id"] != 0):
            raise ValueError("selected model scalar/token settings changed")
    return dict(dims=dims, positions=positions, epsilon=epsilon, init=init,
                theta=theta, seed=42 if fixture else 0)

def tensor_map(c):
    v, layers, d, hq, hk, f = c["dims"]
    k = hk * (d // hq)
    records = [("model.embed_tokens.weight", "token_embedding", [v, d], v, d)]
    slots = (
        ("input_layernorm.weight", "attn_norm", [d], 1, d),
        ("self_attn.q_proj.weight", "query", [d, d], d, d),
        ("self_attn.k_proj.weight", "key", [k, d], k, d),
        ("self_attn.v_proj.weight", "value", [k, d], k, d),
        ("self_attn.o_proj.weight", "output", [d, d], d, d),
        ("post_attention_layernorm.weight", "ffn_norm", [d], 1, d),
        ("mlp.gate_proj.weight", "gate", [f, d], f, d),
        ("mlp.up_proj.weight", "up", [f, d], f, d),
        ("mlp.down_proj.weight", "down", [d, f], d, f)
    )
    for layer in range(layers):
        for suffix, local, shape, rows, cols in slots:
            records.append((f"model.layers.{layer}.{suffix}",
                            f"blocks.{layer}.{local}", shape, rows, cols))
    records.append(("model.norm.weight", "final_norm", [d], 1, d))
    return records

def inspect_source(file, records):
    size = os.fstat(file.fileno()).st_size
    if not 8 <= size <= SOURCE_LIMIT:
        raise ValueError("source size exceeds limit")
    length = struct.unpack("<Q", read_exact(file, 8))[0]
    if not 2 <= length <= HEADER_LIMIT or 8 + length > size:
        raise ValueError("invalid safetensors header length")
    header = json_object(read_exact(file, length))
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("invalid safetensors metadata")
    expected = {source: shape for source, _, shape, _, _ in records}
    # The selected artifact drops the tied lm_head alias. Extra tensors are not ignored.
    if set(header) != set(expected):
        raise ValueError(f"tensor inventory mismatch: missing={set(expected)-set(header)}, "
                         f"extra={set(header)-set(expected)}")
    spans = []
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid tensor record: {name}")
        shape = entry["shape"]
        if (not isinstance(shape, list) or any(type(x) is not int for x in shape) or
                shape != expected[name]):
            raise ValueError(f"wrong shape: {name}")
        dtype = entry["dtype"]
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"unsupported dtype: {dtype}")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int or x < 0 for x in offsets)):
            raise ValueError(f"invalid offsets: {name}")
        begin, end = offsets
        if end - begin != math.prod(shape) * (2 if dtype == "BF16" else 4):
            raise ValueError(f"wrong byte count: {name}")
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError("overlapping tensors or a hole in source payload")
        cursor = end
    if 8 + length + cursor != size:
        raise ValueError("source payload size mismatch")
    return header, 8 + length

def convert_chunk(raw, dtype):
    if dtype == "BF16":
        output = bytearray(2 * len(raw))
        for index, (bits,) in enumerate(struct.iter_unpack("<H", raw)):
            if bits & 0x7f80 == 0x7f80:
                raise ValueError("nonfinite BF16 weight")
            struct.pack_into("<I", output, 4 * index, bits << 16)
        return output
    for (value,) in struct.iter_unpack("<f", raw):
        if not math.isfinite(value):
            raise ValueError("nonfinite F32 weight")
    return raw

def write_weights(source, target, c, records, header, data_start):
    count = sum(rows * cols for _, _, _, rows, cols in records)
    payload = 128 + 96 * len(records)
    total = payload + 4 * count + 8
    if total > OUTPUT_LIMIT:
        raise ValueError("converted file exceeds output budget")
    fixed = (1, 1, *c["dims"], len(records), count, c["positions"])
    target.write(b"DLWGHT01" + struct.pack("<11Q", *fixed))
    target.write(struct.pack("<ffdQQ", c["epsilon"], c["init"], c["theta"],
                             c["seed"], payload))
    cursor = payload
    for _, name, _, rows, cols in records:
        encoded = name.encode("ascii")
        if len(encoded) >= 64:
            raise ValueError("parameter name too long")
        target.write(struct.pack("<64s4Q", encoded, rows, cols, cursor, rows * cols))
        cursor += rows * cols * 4
    for name, _, _, rows, cols in records:
        entry = header[name]
        source.seek(data_start + entry["data_offsets"][0])
        remaining = rows * cols
        width = 2 if entry["dtype"] == "BF16" else 4
        while remaining:
            n = min(remaining, CHUNK_VALUES)
            raw = read_exact(source, n * width)
            converted = convert_chunk(raw, entry["dtype"])
            target.write(converted)
            del raw, converted  # No previous chunk survives the next read.
            remaining -= n
    target.write(b"ENDWGT01")
    if target.tell() != total:
        raise ValueError("internal output-size mismatch")
    return count, total


def config_text(c):
    keys = ("vocab", "layers", "width", "query_heads", "kv_heads", "ffn_width")
    text = "".join(f"{key}={value}\n" for key, value in zip(keys, c["dims"]))
    return (text + f"batch=1\ntime={min(16, c['positions'])}\n" + f"epsilon={c['epsilon']:.9g}\n"
            f"rope_theta={c['theta']:.17g}\ninit_std={c['init']:.9g}\nseed={c['seed']}\n")

def export(source_dir, destination, *, fixture=False, downloaded=False):
    source_dir, destination = Path(source_dir), Path(destination)
    # Single writer, ordinary local paths; never replace an existing result directory.
    if destination.exists():
        raise FileExistsError(destination)
    config_path = source_dir / "config.json"
    if config_path.stat().st_size > 65536:
        raise ValueError("configuration exceeds limit")
    raw = json_object(config_path.read_bytes())
    c = configuration(raw, fixture=fixture)
    records = tensor_map(c)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".import-", dir=destination.parent))
    try:
        with (source_dir / "model.safetensors").open("rb") as source:
            header, start = inspect_source(source, records)
            with (stage / "weights.dlw").open("wb") as target:
                count, total = write_weights(source, target, c, records, header, start)
                target.flush()
                os.fsync(target.fileno())
        (stage / "model.cfg").write_text(config_text(c), encoding="ascii")
        provenance = dict(model_profile="synthetic fixture" if fixture else MODEL,
            expected_revision=None if fixture else REVISION,
            acquisition="pinned download" if downloaded else "caller-supplied local files",
            tensors=len(records), values=count, weight_bytes=4 * count,
            file_bytes=total, position_limit=c["positions"],
            dtypes=sorted({entry["dtype"] for entry in header.values()}))
        (stage / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return provenance

def download(destination):
    """Fetch only two pinned assets, streaming with explicit bounds; no retries."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".download-", dir=destination.parent))
    try:
        for name, limit in (("config.json", 65536), ("model.safetensors", SOURCE_LIMIT)):
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "decoder-lab-assets"})
            with urllib.request.urlopen(request, timeout=60) as response:
                with (stage / name).open("wb") as file:
                    count = 0
                    while True:
                        chunk = response.read(min(65536, limit - count + 1))
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > limit:
                            raise ValueError(f"download too large: {name}")
                        file.write(chunk)
            if name == "config.json":
                configuration(json_object((stage / name).read_bytes()))
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path, help="existing config.json/model.safetensors directory")
    sources.add_argument("--download", type=Path, help="new directory for pinned source assets")
    parser.add_argument("--output", type=Path, required=True, help="new converted-asset directory")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.download is not None:
            download(args.download)
        report = export(args.source or args.download, args.output,
                        downloaded=args.download is not None)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, OverflowError, TypeError, struct.error) as error:
        print(f"conversion failed: {error}", file=__import__("sys").stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
