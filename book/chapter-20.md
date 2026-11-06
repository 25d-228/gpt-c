# Chapter 20 — Import and validate pretrained weights

> **Learning objectives.** Map a published decoder's named tensors into the existing C parameter directory, distinguish stored precision from computation precision, build a bounded conversion/load boundary, and reject incompatible files without publishing a partial weight owner.

This is one **4.00-hour implementation chapter**, presented in four connected parts. Allocate **2.50 hours** to the mapping, conversion, and loader, and **1.50 hours** to fixtures, failure diagnosis, and load-memory checks. Reading and unattended downloads/conversion are separate. No learner timing has yet replaced this estimate.

**Prerequisites:** the complete [Chapter 19 source state](https://25d-228.github.io/gpt-c/chapter-19.html), especially `DlWeights` and inference from Chapter 16, the cache from Chapter 17, and the checkpoint helpers from Chapters 15 and 19. Start from its complete 73-file companion directory. This chapter changes seven existing files and adds seven files; the [complete new snapshot](https://25d-228.github.io/gpt-c/media/chapter-20/reference/Makefile) contains 80 files. Earlier training and generation paths remain available.

Part 1 implements the asset converter. [Part 2](https://25d-228.github.io/gpt-c/chapter-20-part-2.html) extends the C loader and position-limit contracts. [Part 3](https://25d-228.github.io/gpt-c/chapter-20-part-3.html) provides the complete C fixtures and inspection application. [Part 4](https://25d-228.github.io/gpt-c/chapter-20-part-4.html) tests the Python/C boundary, runs the import, and separates allocation counts from measured process memory.

## Loading numbers is not yet loading their meaning

A successful read of 134 million finite numbers proves very little. Swapping the gate and value matrices preserves all dimensions. Transposing a square projection preserves its shape. Using adjacent rotary pairs rather than split halves preserves vector norms. All three can produce a runnable but different model.

We therefore make two separate checks. This chapter checks **representation**: the architecture profile, source tensor names/shapes, scalar formats, byte offsets, parameter order, ownership, and execution through the existing C operators. Chapter 21 checks **numerical agreement** against an independently executed pretrained reference, with explicit token IDs and matching precision. An argmax token or a finite logit is not a substitute for that comparison.

Python performs asset conversion here. It does not construct a transformer, run attention, produce hidden states, or replace C inference. The converter needs only the Python standard library. It will not load a pickle or execute downloaded model code.

## Fix the model and its interpretation

Use the base model `HuggingFaceTB/SmolLM2-135M`, not the similarly named instruct model or the first SmolLM release. The import commands pin revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The [official Hub metadata](https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M) identifies that revision and reports **134,515,008 BF16 values**. The [published configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json), read alongside it, supplies the following settings:

| Symbol | Meaning | Value |
|---|---|---:|
| $V$ | Vocabulary rows | 49,152 |
| $L$ | Decoder blocks | 30 |
| $D$ | Hidden width | 576 |
| $H_q$ | Query heads | 9 |
| $H_{kv}$ | Key/value heads | 3 |
| $F$ | Feed-forward hidden width | 1,536 |
| $d=D/H_q$ | Coordinates per head | 64 |
| $K=H_{kv}d$ | Combined key/value width | 192 |

Its RMSNorm epsilon is $10^{-5}$, rotary base is $100000$, and position limit is 8,192. Projections are bias-free, activation is SiLU, embeddings are tied, and RoPE is unscaled and non-interleaved. The import path does not implement alternative scaling, sliding windows, quantized tensors, extra normalization layers, adapters, or an untied output table.

These are interpretation constraints, not suggestions to silently coerce another checkpoint. The converter rejects unknown configuration fields and extra tensors. A newer library's additional harmless metadata may require a deliberate parser update; rejecting it is safer than accidentally ignoring an architectural change. The configuration's historical `transformers_version` is descriptive metadata, not an installation requirement for this converter.

### Count what the engine should own

Each block owns two $D\times D$ matrices, two $K\times D$ matrices, three $F\times D$ matrices, and two length-$D$ scales. The embedding table is shared with the output projection; the final scale is separate. Consequently the unique parameter count $P$ is

$$
P=VD+L(2D^2+2KD+3FD+2D)+D.
$$

For the published dimensions, the per-block count is $3,540,096$, the shared table contains $28,311,552$ values, and the final scale contains 576. Substitution gives $P=134,515,008$, agreeing with the aggregate source metadata. The directory contains $2+9L=272$ records. These counts are derived from the architecture; the converter must still check the actual downloaded file's complete directory. An aggregate count cannot establish that every tensor name is correct.

We retain **one** shared table. Safetensors does not represent arbitrary shared tensor aliases in its ordinary tensor dictionary; serialization can omit a redundant name. The [shared-tensor documentation](https://huggingface.co/docs/safetensors/main/en/torch_shared_tensors) explains this distinction. Our selected import profile requires the embedding tensor and no separate `lm_head.weight`. It does not silently discard a second head or assume an arbitrary extra tensor is an alias.

## Map names, not enumeration order

For block number `i`, prefix each source suffix below with `model.layers.i.`. Every destination name uses `blocks.i.`. The table gives **stored** dimensions; normalization vectors become one-row matrix views without changing their element order.

| Source suffix or full name | Engine name | Source shape | Engine shape |
|---|---|---|---|
| `model.embed_tokens.weight` | `token_embedding` | $[V,D]$ | $[V,D]$ |
| `input_layernorm.weight` | `attn_norm` | $[D]$ | $[1,D]$ |
| `self_attn.q_proj.weight` | `query` | $[D,D]$ | unchanged |
| `self_attn.k_proj.weight` | `key` | $[K,D]$ | unchanged |
| `self_attn.v_proj.weight` | `value` | $[K,D]$ | unchanged |
| `self_attn.o_proj.weight` | `output` | $[D,D]$ | unchanged |
| `post_attention_layernorm.weight` | `ffn_norm` | $[D]$ | $[1,D]$ |
| `mlp.gate_proj.weight` | `gate` | $[F,D]$ | unchanged |
| `mlp.up_proj.weight` | `up` | $[F,D]$ | unchanged |
| `mlp.down_proj.weight` | `down` | $[D,F]$ | unchanged |
| `model.norm.weight` | `final_norm` | $[D]$ | $[1,D]$ |

This is the complete mapping the converter enforces. Source dictionary order and physical byte order need not match either block order or this table. For example, lexicographic ordering can place block 10 before block 2. We look up each exact name and its validated byte span, then write in engine order. Part 3's synthetic source deliberately reverses both source orders.

The [PyTorch linear-layer contract](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html) stores weights as `[output,input]` and applies $Y=XW^T$. That is already our convention. With $x=(2,0,-1)$ and stored rows $W_0=(1,2,3)$, $W_1=(4,5,6)$, the output is $(-1,2)$. The flat bytes `[1,2,3,4,5,6]` are copied, not transposed. The matrix kernel's transpose flag describes access during multiplication; it does not request rewriting the asset.

Similarly, do not apply an older Llama converter's query/key permutation here. The profile requires split-half rotation within each 64-coordinate head: coordinate $j$ pairs with $j+32$, not $2j+1$. Query heads 0–2 read key/value head 0, heads 3–5 read head 1, and heads 6–8 read head 2. The existing engine implements these conventions; the converter must not compensate for an imagined mismatch.

## A small, strict converter

Create `scripts/export_smollm2.py`. All its Python blocks in this part, in order, form the complete file. The private `fixture=True` function argument admits small synthetic dimensions for tests; the command line does not offer that bypass. File sizes and chunk sizes are explicit constants, not inferred from available RAM.

The `unique_object` callback sees every JSON object before ordinary dictionaries discard duplicate keys. Without it, a repeated tensor name could silently replace an earlier entry. `read_exact` treats a short file as an error rather than accepting partially initialized data. `integer` deliberately rejects Python booleans, which otherwise behave like integers.

```python title="scripts/export_smollm2.py"
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
```


```python title="scripts/export_smollm2.py"
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
```

### Accept one architecture rather than guessing defaults

The next function separates field presence, supported mechanisms, dimensional compatibility, scalar representability, and the selected model's exact profile. The tuple order is vocabulary, layers, hidden width, query heads, key/value heads, and feed-forward width. `*c["dims"]` will later expand those six values into a fixed wire header.

Optional `mlp_bias` must be false. Optional `head_dim` must equal $D/H_q$. Every other unknown key is rejected. In the published configuration those two optional keys are absent; the profile supplies the already-understood bias-free and inferred-head-width interpretation. Unknown tensor names independently catch an unexpected bias or normalization parameter.

The pack/unpack round trip converts epsilon and initialization metadata into exactly the binary32 values the C configuration will contain. A positive Python number that underflows to zero or overflows to infinity is not accepted. `init_std` and `seed` remain configuration metadata for compatibility; loading weights does **not** call random initialization.

```python title="scripts/export_smollm2.py"
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
```

### Construct the canonical directory

`tensor_map` describes logical meaning independently of the source file's offsets. Each record holds the external name, internal name, expected external shape, and internal matrix rows/columns. Shape lists contain only positive dimensions validated above. The source and destination shapes have equal element counts even when a vector becomes `[1,D]`.

```python title="scripts/export_smollm2.py"
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
```

## Read Safetensors without a whole-model temporary

The source begins with an eight-byte little-endian unsigned header length, followed by a JSON header and the tensor payload. Each tensor record specifies its dtype, shape, and a half-open byte interval relative to the **start of the payload**. The [official metadata-parsing guide](https://huggingface.co/docs/safetensors/main/en/metadata_parsing) demonstrates reading that header separately from weights.

Let $N$ be the header length and $[a,b)$ a tensor's relative span. Its first byte is at absolute file offset $8+N+a$, and its payload length is $b-a$. A `[3,4]` BF16 tensor needs $3\cdot4\cdot2=24$ bytes. If its span is `[40,64)`, seeking to `40` is wrong; seek to `8 + N + 40`.

We inspect all metadata before writing any values. The expected name set is exact, shapes are exact, and dtypes are limited to BF16/F32. Sorted spans must cover the payload without overlap or holes, and the final span must end at exact file length. A wrong `end` cannot make us read a neighboring tensor. Shards, compressed arrays, F16, and quantized formats are outside this converter; they require another explicit version or adapter.

`os.fstat` measures the open file rather than a second path lookup. This is still an ordinary-local-file contract: no concurrent writer may modify an input while it is being inspected and converted. The one-MiB header limit bounds input text, not the Python allocator's exact object overhead.

```python title="scripts/export_smollm2.py"
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
```

## BF16 expansion is a bit transformation, not an integer cast

BF16 stores a sign bit, eight exponent bits, and seven fraction bits. Binary32 uses the same sign/exponent layout above 23 fraction bits. Expanding a finite BF16 value places its 16 bits in the high half of a binary32 word and fills the low half with zeros. This preserves the represented value exactly; it does not recover precision discarded before the checkpoint was saved. The [Google format description](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) gives the fields, and its [conversion documentation](https://docs.cloud.google.com/tpu/docs/bfloat16) describes zero-padding on expansion.

For a normal BF16 bit pattern, let $s$ be the sign bit, $e$ the stored exponent, and $f$ the seven-bit fraction interpreted as an integer. Its value is

$$
(-1)^s 2^{e-127}\left(1+\frac{f}{128}\right),\qquad 1\le e\le254.
$$

For $e=0$, the value is $(-1)^s f\,2^{-133}$. For $e=255$, the encoding is nonfinite and our weight loader rejects it. These formulas also give an independent test oracle that does not repeat the converter's bit shift.

For example, the source bytes `c0 3f` encode BF16 `0x3fc0`, or 1.5. Shifting gives binary32 `0x3fc00000`, stored as `00 00 c0 3f`. A numerical cast of the integer `0x3fc0` would instead produce 16,320. The words are bit patterns, not integer-valued weights.

The exponent mask `0x7f80` ignores sign and fraction, rejecting both infinities and every NaN encoding. Signed zero and finite subnormals are preserved as representations. This is a file conversion, not a claim that every later hardware arithmetic mode preserves subnormals.

```python title="scripts/export_smollm2.py"
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
```

## Define a weight-only wire format

Do not disguise this asset as a resumable training checkpoint. It has no moments, successful-update count, schedule, data RNG, or embedded dataset. Its eight-byte magic is `DLWGHT01`, independent of `DLCKPT01`/`DLCKPT02`. The existing training writer remains unchanged.

All integers are unsigned little-endian 64-bit words unless stated otherwise. Floats are explicit IEEE binary32 or binary64. The [Python `struct` documentation](https://docs.python.org/3/library/struct.html) defines `<` as little-endian with standard sizes and no alignment padding. A C structure's compiler-dependent padding is never written to disk.

| Header byte offset | Field |
|---:|---|
| 0 | Eight-byte magic `DLWGHT01` |
| 8 | Format version, 1 |
| 16 | Architecture tag, 1: this tied, bias-free RMSNorm/SwiGLU/split-half-RoPE decoder |
| 24, 32, 40, 48, 56, 64 | $V,L,D,H_q,H_{kv},F$ |
| 72, 80, 88 | Tensor count, value count, maximum prefix length |
| 96, 100 | Binary32 epsilon and initialization metadata |
| 104 | Binary64 rotary base |
| 112, 120 | Seed metadata and absolute payload offset |

The header is exactly 128 bytes. Each 96-byte directory entry contains a 64-byte NUL-padded ASCII name, then rows, columns, absolute payload offset, and element count. There is no arbitrary ordering: the file directory follows the canonical engine order. Payload values are consecutive binary32 numbers in that same order. The eight-byte trailer is `ENDWGT01`, followed immediately by EOF.

For $J$ records and $P$ values, the payload begins at $128+96J$, and the file length is

$$
128+96J+4P+8.
$$

With $J=272$ and the derived $P$, the converted file contains **538,086,280 bytes**, of which **538,060,032** are weights. The directory's offset fields are redundancy for validation, not permission for the C loader to seek into arbitrary file regions.

### Stream each tensor in canonical order

The converter validates the output budget before writing the header. It keeps at most 65,536 source values and their expanded chunk at once. At BF16 input precision that is 131,072 source bytes plus 262,144 expanded bytes: **393,216 bytes** of numeric chunk storage. JSON objects, buffered I/O, Python runtime state, and filesystem caches are separate. F32 chunks are checked and written without an additional conversion buffer.

`del raw, converted` prevents the previous chunk from remaining live during the next `read`. This is not required for correctness, but makes the live-buffer accounting match the stated bound. The output writer never assembles a giant Python list of weights.

```python title="scripts/export_smollm2.py"
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
```

## Expose a complete result directory only on success

`export` creates a temporary sibling directory. It writes the validated weight file, a C configuration, and a small provenance description there. Only then does it rename the directory to the requested destination. An existing destination is an error; it is not overwritten. Any earlier exception removes the staging directory. The source directory remains available for diagnosis.

This contract assumes one writer and caller-controlled local paths, not adversarial pathname replacement. The rename boundary avoids exposing a partially converted directory during ordinary operation. It is not a claim of power-loss durability for the whole directory; only the weight file receives an explicit flush/sync here.

For caller-supplied files, `expected_revision` identifies the intended source, not authenticated origin. A structurally valid file of finite zeros can pass. We will deliberately use such a file for a size experiment. Structural validation is not an integrity signature or proof of model identity.

```python title="scripts/export_smollm2.py"
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
```

### Optional bounded download

The downloader requests exactly `config.json` and `model.safetensors` at the pinned revision. It checks the configuration before downloading weights and limits response bytes while streaming, even when no trustworthy content length is supplied. It makes no automatic retries, requests no credentials, and does not execute repository code. A failed download removes its staging directory. A successful download is still only acquisition: tensor validation occurs during conversion.

A functioning network connection to the public model host is required for this option. An existing local source directory uses the same validation path without networking. Standard-library HTTPS/JSON/binary-file features are sufficient; no exact Python release or model-library version is a gate.

```python title="scripts/export_smollm2.py"
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
```


```python title="scripts/export_smollm2.py"
if __name__ == "__main__":
    raise SystemExit(main())
```

## The imported configuration used by C

Add the following `configs/smollm2-135m.cfg`. It matches the converter's generated `model.cfg`. The displayed decimal epsilon and initialization metadata round-trip to the intended binary32 numbers. `batch` and `time` are required by the existing configuration parser; they do not allocate an inference workspace or override the imported 8,192-position limit.

```ini title="configs/smollm2-135m.cfg"
vocab=49152
layers=30
width=576
query_heads=9
kv_heads=3
ffn_width=1536
batch=1
time=16
epsilon=9.99999975e-06
rope_theta=100000
init_std=0.0416666679
seed=0
```

The conversion program is complete, but do not start by trusting a large output. First implement the C loading boundary in [Part 2](https://25d-228.github.io/gpt-c/chapter-20-part-2.html), then run the tiny fixtures. They check exactly the file representation above without requiring a model download.
