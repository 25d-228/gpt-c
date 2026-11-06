# Chapter 21 — Tokenizer boundaries and pretrained reference agreement

> **Learning objectives**
> Preserve the exact token sequence supplied to C; make beginning/end-token policy visible; compare every logit and localize numerical disagreement; generate in C and replay the same prefixes through an independent reference.

Start from the complete [Chapter 20 implementation](https://25d-228.github.io/gpt-c/chapter-20.html), including its weight-only loader, and retain [Chapter 17's cache](https://25d-228.github.io/gpt-c/chapter-17.html) and Chapter 18's tiled inference. The latest complete source is the [Chapter 21 companion](https://25d-228.github.io/gpt-c/media/chapter-21/reference/Makefile). Its 90 files include the prerequisite implementation; they are not a replacement for the explanations below.

This is one **3.50-hour implementation chapter** in four linked parts. Allocate approximately 2.00 hours to token boundaries, C observation and reference plumbing, and 1.50 hours to comparisons and diagnosis. Reading and unattended execution are separate. Part 1 defines the data boundary; [Part 2](https://25d-228.github.io/gpt-c/chapter-21-part-2.html) observes and runs C; [Part 3](https://25d-228.github.io/gpt-c/chapter-21-part-3.html) builds the independent equation reference; [Part 4](https://25d-228.github.io/gpt-c/chapter-21-part-4.html) supplies complete tests and the real-asset commands.

## The same sentence is not necessarily the same input

Our byte model used one vocabulary row for each possible byte. The imported model does not. Let the UTF-8 input contain $B$ bytes and let its tokenizer return $T$ integers $i_0,\ldots,i_{T-1}$. Here $T$ need not equal $B$, and every integer is an **embedding-row index**, not a Unicode character or a byte to print.

For example, UTF-8 represents `é` by bytes `c3 a9`. That tells us two byte-model inputs, 195 and 169. It does **not** tell us the pretrained model's IDs for `é`. Those depend on the selected tokenizer assets and surrounding text. We will obtain them from the tokenizer, not invent a vocabulary mapping. An ID of 256 is valid for a 49,152-row table but cannot be stored in an unsigned byte without losing information.

The selected base model remains `HuggingFaceTB/SmolLM2-135M`, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Its [published tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/tokenizer_config.json) names `GPT2Tokenizer`, uses `<|endoftext|>` as both beginning and end token, and disables cleanup of tokenization spaces. The [model configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json) assigns both beginning and end IDs to zero. This does not tell an application to insert or stop on that ID automatically.

Our boundary has these explicit choices:

| Operation | This chapter's policy |
|---|---|
| Input text | Read bytes, strictly decode UTF-8, pass the resulting text unchanged |
| Automatic additions | No BOS, EOS, padding, truncation or chat template |
| Optional beginning token | `--bos` prepends exactly one ID 0 |
| Literal special markers | Reject recognized special IDs unless `--allow-special` is given |
| Output text | Decode the complete ID sequence, with cleanup disabled; retain special tokens unless `--skip-special` is given |
| Generation stopping | Stop only when a **newly selected** ID equals the explicitly supplied EOS ID |

The script does not strip whitespace, change CRLF to LF, normalize Unicode, or split text into words. The tokenizer still performs the processing encoded by its own assets. Setting `add_special_tokens=False` disables automatic additions; it does not necessarily prevent a literal special-token spelling from being recognized. These are separate controls in the [tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer).

A beginning ID already present in the prompt must not immediately stop generation merely because beginning and end tokens share the number zero. Likewise, adding a beginning token shifts every following absolute position by one. A reference with it and C without it are calculating different functions, even when the visible sentence is identical.

## A small format for explicit IDs

Create `scripts/probe_io.py`. It uses only the Python standard library. The complete file is supplied in three consecutive listings in this part; concatenate them in order.

The token file has an eight-byte magic, two little-endian unsigned 64-bit integers for vocabulary size $V$ and count $N$, exactly $N$ unsigned 32-bit IDs, and an eight-byte trailer. Its length is

$$
8+8+8+4N+8=32+4N\quad\text{bytes}.
$$

`DLIDS001` and `ENDIDS01` are literal eight-byte strings. For $N=4$, the file is 48 bytes. We bound $N$ at 256 and $V$ at 100,000, so every accepted ID fits in 32 bits. C independently checks the bounds and requires the file's vocabulary size to equal its loaded model's vocabulary.

The `<` prefix in a Python `struct` format means little-endian with standard widths and no native padding. `Q` stores an unsigned 64-bit value, and `I` an unsigned 32-bit value. We never dump a native Python or C object layout. The values `[0,255,256,49151]` consequently survive the file boundary unchanged.

`exclusive_file` is a context manager: `with` opens a new file, supplies its handle to the body, and closes it on either exit path. Opening with `x` refuses an existing destination. An exception—including a close failure—removes only the new output created by this call. These are ordinary, single-writer local paths, not a concurrent publication or power-loss durability protocol.

```python title="scripts/probe_io.py"
"""Small explicit token/trace formats shared by asset and comparison scripts."""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import json
import math
from pathlib import Path
import struct

MAX_IDS = 256
TRACE_LIMIT = 256 * 1024 * 1024
LINE_LIMIT = 64 * 1024 * 1024


@contextmanager
def exclusive_file(path, mode="w"):
    """Ordinary single-writer paths; no overwrite, remove a failed new output."""
    path = Path(path)
    stream = path.open("xb" if mode == "wb" else "x",
                       **({} if mode == "wb" else {"encoding": "utf-8", "newline": "\n"}))
    try:
        with stream:
            yield stream
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def validate_ids(vocab, ids):
    if type(vocab) is not int or not 2 <= vocab <= 100000:
        raise ValueError("invalid vocabulary size")
    if not 1 <= len(ids) <= MAX_IDS:
        raise ValueError("token count must be in 1..256")
    if any(type(x) is not int or not 0 <= x < vocab for x in ids):
        raise ValueError("token outside vocabulary")


def write_ids(path, vocab, ids):
    validate_ids(vocab, ids)
    with exclusive_file(path, "wb") as file:
        file.write(struct.pack("<8sQQ", b"DLIDS001", vocab, len(ids)))
        file.write(struct.pack(f"<{len(ids)}I", *ids))
        file.write(b"ENDIDS01")


def read_ids(path):
    with Path(path).open("rb") as file:
        data = file.read(33 + 4 * MAX_IDS)
    if len(data) < 36 or len(data) > 32 + 4 * MAX_IDS:
        raise ValueError("invalid token-file length")
    magic, vocab, count = struct.unpack_from("<8sQQ", data)
    if magic != b"DLIDS001" or not 1 <= count <= MAX_IDS or len(data) != 32 + 4 * count:
        raise ValueError("invalid token header/count")
    if data[-8:] != b"ENDIDS01":
        raise ValueError("invalid token trailer")
    ids = list(struct.unpack_from(f"<{count}I", data, 24))
    validate_ids(vocab, ids)
    return vocab, ids


```

`read_ids` reads at most one byte beyond the largest permitted file. It does not trust a count before checking the total byte length. A truncated header, an extra byte after the trailer, or an ID outside the declared vocabulary is a failed input, not a partly usable prompt. `type(x) is int` also rejects Python booleans, which otherwise behave like integers.

A small boundary check needs no tokenizer or neural-network library:

```sh
python3 - <<'PYIDS'
import sys
sys.path.insert(0, 'scripts')
from probe_io import write_ids, read_ids
write_ids('example.ids', 49152, [0, 255, 256, 49151])
print(read_ids('example.ids'))
PYIDS
```

It prints `(49152, [0, 255, 256, 49151])`. Running it again fails rather than overwriting `example.ids`. Remove that disposable file explicitly before repeating the exercise.

## A trace describes both the input and the numerical experiment

The comparison format is newline-delimited JSON. A header identifies the exact IDs, geometry, normalization epsilon, rotary base, and evaluation mode. Geometry is `[vocabulary, layers, width, query_heads, kv_heads, ffn_width]`. Settings record which engine and numerical choices produced the file. Settings are deliberately not required to be identical: comparing C with Python is the purpose of the experiment.

Each tensor record names a stage, a stage index, its two-dimensional shape, and flattened row-major values. A final record declares how many tensors were written. Without it, a crash after ten apparently valid tensors could be mistaken for a complete ten-tensor comparison. Duplicate JSON keys and nonfinite numeric constants are rejected rather than resolved ambiguously.

```python title="scripts/probe_io.py"
def unique_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def trace_header(mode, ids, geometry, start, settings, *, epsilon=1e-5, theta=10000.0):
    return dict(kind="header", version=1, mode=mode, ids=ids, geometry=list(geometry),
                start=start, epsilon=epsilon, theta=theta, settings=settings)


def write_json(file, record):
    file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


```

The reader below is a generator. It holds one tensor record at a time, not the entire experiment. A caller **must exhaust it** to establish that the counted trailer and end of file are valid. Part 3's comparator does so even after discovering an early numerical disagreement.

There is a representation subtlety in a text trace. C emits each stored `float` with nine significant decimal digits. Those digits recover the original binary32 value when converted **back to binary32**. Merely parsing them into Python's binary64 `float` retains tiny decimal-rounding differences that were never present in either engine's tensor. `array("f", values)` performs the required rounding; its four-byte element size and finite results are checked. This does not reduce an FP64 reference to conceal error: both engines explicitly declare FP32 tensor stores in this experiment.

Bounds apply to file size, individual lines, record count and token count. The parser accepts only the specified fields, positive shapes, a matching flattened length and finite representable values. It does not infer missing dimensions from whatever values happen to remain.

```python title="scripts/probe_io.py"
def trace_records(path):
    """Yield validated records without retaining an entire trace in memory."""
    if Path(path).stat().st_size > TRACE_LIMIT:
        raise ValueError("trace exceeds 256 MiB limit")
    count, ended = 0, False
    with Path(path).open("rb") as file:
        line = file.readline(LINE_LIMIT + 1)
        if len(line) > LINE_LIMIT:
            raise ValueError("trace header too large")
        header = unique_json(line)
        if set(header) != {"kind", "version", "mode", "ids", "geometry", "start", "epsilon", "theta", "settings"}:
            raise ValueError("invalid trace header fields")
        g, ids, start = header["geometry"], header["ids"], header["start"]
        if header["kind"] != "header" or type(header["version"]) is not int or header["version"] != 1:
            raise ValueError("unsupported trace format")
        if not isinstance(g, list) or len(g) != 6 or any(type(x) is not int or x <= 0 for x in g):
            raise ValueError("invalid trace geometry")
        if not isinstance(ids, list):
            raise ValueError("invalid token list")
        validate_ids(g[0], ids)
        if header["mode"] not in ("trace", "prefix") or type(start) is not int or not 1 <= start <= len(ids):
            raise ValueError("invalid trace mode/start")
        if header["mode"] == "trace" and start != len(ids):
            raise ValueError("trace must describe the complete supplied prefix")
        if not isinstance(header["settings"], dict):
            raise ValueError("invalid settings")
        if any(type(header[k]) not in (float, int) or not math.isfinite(header[k]) or
               header[k] <= 0 for k in ("epsilon", "theta")):
            raise ValueError("invalid model numerical settings")
        yield header
        while True:
            line = file.readline(LINE_LIMIT + 1)
            if not line:
                break
            if len(line) > LINE_LIMIT or ended:
                raise ValueError("oversized record or data after trailer")
            row = unique_json(line)
            if row.get("kind") == "end":
                if set(row) != {"kind", "records"} or type(row["records"]) is not int or row["records"] != count:
                    raise ValueError("invalid trace trailer")
                ended = True
                continue
            if set(row) != {"kind", "name", "index", "rows", "cols", "values"} or row["kind"] != "tensor":
                raise ValueError("invalid tensor fields")
            if not isinstance(row["name"], str) or not row["name"] or len(row["name"]) > 32:
                raise ValueError("invalid tensor name")
            if type(row["index"]) is not int or row["index"] < 0:
                raise ValueError("invalid tensor index")
            if any(type(row[k]) is not int or row[k] <= 0 for k in ("rows", "cols")):
                raise ValueError("invalid tensor shape")
            values = row["values"]
            if not isinstance(values, list) or len(values) != row["rows"] * row["cols"] or not values:
                raise ValueError("tensor shape/payload mismatch")
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4028235e38 for v in values):
                raise ValueError("invalid finite FP32 tensor value")
            # Nine significant decimal digits round-trip a binary32 value only
            # when the parsed decimal is rounded BACK to binary32, not kept as double.
            rounded = array("f", values)
            if rounded.itemsize != 4 or any(not math.isfinite(v) for v in rounded):
                raise ValueError("unsupported or overflowing binary32 representation")
            row["values"] = rounded
            count += 1
            if count > 4096:
                raise ValueError("too many trace records")
            yield row
    if not ended or count == 0:
        raise ValueError("incomplete or empty trace")
```

## Tokenizer assets are data, not executable model code

Create `scripts/tokenizer_boundary.py`. Its complete file follows in three groups. Fetching requires network access, but encoding and decoding require only the downloaded local directory. The model and revision constants come from Chapter 20's `scripts/export_smollm2.py`; there is one selected revision, not an independent moving tokenizer revision.

The fetcher reads three bounded assets from that revision, writes them into a new temporary directory beside the destination, and exposes the directory only after every file succeeds. It does not retry blindly or overwrite existing assets. `source.json` records where this script requested the data; it is provenance, not proof that a user-provided local directory is authentic.

```python title="scripts/tokenizer_boundary.py"
#!/usr/bin/env python3
"""Tokenizer asset acquisition and explicit encode/decode policies."""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import importlib.metadata
import json
import shutil
import tempfile
import urllib.request
from export_smollm2 import MODEL, REVISION
from probe_io import exclusive_file, read_ids, validate_ids, write_ids

def fetch(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tokenizer-", dir=directory.parent))
    try:
        for name, limit in (("tokenizer.json", 4*1024*1024),
                            ("tokenizer_config.json", 65536),
                            ("special_tokens_map.json", 65536)):
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            request = urllib.request.Request(url, headers={"User-Agent": "decoder-lab-assets"})
            with urllib.request.urlopen(request, timeout=60) as response, (stage/name).open("wb") as file:
                count = 0
                while True:
                    data = response.read(min(65536, limit-count+1))
                    if not data:
                        break
                    count += len(data)
                    if count > limit:
                        raise ValueError(f"oversized tokenizer asset: {name}")
                    file.write(data)
        (stage/"source.json").write_text(json.dumps(dict(model=MODEL, revision=REVISION,
            acquisition="pinned public download"), indent=2)+"\n", encoding="utf-8")
        if directory.exists():
            raise FileExistsError(directory)
        stage.rename(directory)
    except BaseException:
        shutil.rmtree(stage)
        raise

```

The local loader uses `trust_remote_code=False` and `local_files_only=True`. It requires the fast tokenizer interface, the expected vocabulary size, and the expected beginning/end IDs. Those checks catch obvious mismatches but do not prove that two different tokenizers with equal vocabulary sizes assign every row identically. Use the pinned assets and keep the emitted ID sequence as the actual model input.

`encode_text` reads no model weights. It rejects malformed UTF-8, checks the explicit special-token policy, optionally prepends the beginning ID, and then applies the shared count/range validation. An empty prompt without a beginning ID is rejected because our decoder needs a position from which to predict. With `--bos`, an empty text has the explicit one-ID prompt `[0]`.

`decode_ids` decodes the **whole sequence**. A token boundary need not coincide with a UTF-8 character boundary, so concatenating independently decoded token strings can introduce replacement characters. Even whole-sequence decoding may display a replacement character if arbitrary generated tokens end with an incomplete UTF-8 sequence; this interface returns tokenizer text, not a promise of byte-perfect output for every arbitrary ID list.

```python title="scripts/tokenizer_boundary.py"
def load_tokenizer(directory):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
        trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast or len(tokenizer) != 49152 or \
       tokenizer.bos_token_id != 0 or tokenizer.eos_token_id != 0:
        raise ValueError("tokenizer is not the selected SmolLM2 profile")
    return tokenizer

def encode_text(tokenizer, raw, *, bos=False, allow_special=False):
    if len(raw) > 65536:
        raise ValueError("prompt exceeds 65536 UTF-8 bytes")
    text = raw.decode("utf-8", errors="strict")
    # Do not strip, normalize line endings, apply a chat template, or truncate.
    ids = list(tokenizer.encode(text, add_special_tokens=False,
                                padding=False, truncation=False))
    if not allow_special and any(x in tokenizer.all_special_ids for x in ids):
        raise ValueError("literal special token in prompt; use --allow-special deliberately")
    if bos:
        ids.insert(0, tokenizer.bos_token_id)
    validate_ids(len(tokenizer), ids)
    return ids

def decode_ids(tokenizer, vocab, ids, *, skip_special=False):
    validate_ids(vocab, ids)
    if vocab != len(tokenizer):
        raise ValueError("tokenizer vocabulary does not match token file")
    return tokenizer.decode(ids, skip_special_tokens=skip_special,
                            clean_up_tokenization_spaces=False).encode("utf-8")

```

The command-line layer keeps these choices visible. `inspect` can read IDs without importing Transformers. Encoding reports the actual library versions, flags and token list rather than claiming that the source model's old `transformers_version` field is a current installation requirement. Decoding writes no extra newline: adding one would change the requested text output.

```python title="scripts/tokenizer_boundary.py"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetcher = commands.add_parser("fetch"); fetcher.add_argument("directory", type=Path)
    inspect = commands.add_parser("inspect"); inspect.add_argument("ids", type=Path)
    encoder = commands.add_parser("encode")
    encoder.add_argument("tokenizer", type=Path); encoder.add_argument("input", type=Path)
    encoder.add_argument("output", type=Path); encoder.add_argument("--bos", action="store_true")
    encoder.add_argument("--allow-special", action="store_true")
    decoder = commands.add_parser("decode")
    decoder.add_argument("tokenizer", type=Path); decoder.add_argument("input", type=Path)
    decoder.add_argument("output", type=Path); decoder.add_argument("--skip-special", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch(args.directory)
        elif args.command == "inspect":
            vocab, ids = read_ids(args.ids)
            print(json.dumps(dict(vocab=vocab, count=len(ids), ids=ids)))
        else:
            tokenizer = load_tokenizer(args.tokenizer)
            if args.command == "encode":
                with args.input.open("rb") as file:
                    raw = file.read(65537)
                ids = encode_text(tokenizer, raw, bos=args.bos, allow_special=args.allow_special)
                write_ids(args.output, len(tokenizer), ids)
                report = dict(expected_model=MODEL, expected_revision=REVISION,
                    local_assets=str(args.tokenizer), provenance="local files, not authenticated",
                    transformers=importlib.metadata.version("transformers"),
                    tokenizers=importlib.metadata.version("tokenizers"), utf8_bytes=len(raw),
                    bos="explicit prepend" if args.bos else "none", eos_append=False,
                    literal_specials=args.allow_special, chat_template=False,
                    padding=False, truncation=False, ids=ids)
                print(json.dumps(report, indent=2))
            else:
                vocab, ids = read_ids(args.input)
                raw = decode_ids(tokenizer, vocab, ids, skip_special=args.skip_special)
                with exclusive_file(args.output, "wb") as file:
                    file.write(raw)
        return 0
    except (OSError, ValueError, ImportError, TypeError) as error:
        print(f"token boundary failed: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
```

## The entry-point name and Python's own `tokenize` module

The requested entry point is `scripts/tokenize.py`. Python itself also has a standard-library module named `tokenize`, which can be imported indirectly by numerical packages. When the `scripts` directory is first on the module search path, a naive file of this name can accidentally replace the standard library and create a circular import.

The small wrapper below runs our boundary only when executed as the main script. When imported as `tokenize`, it loads the installed Python standard-library file under a private module name and exposes that module's public interface. `sysconfig.get_path("stdlib")` identifies the local interpreter's standard library; this is not remote code or a download. Registering the private module before executing it also allows classes defined there to find their defining module.

```python title="scripts/tokenize.py"
#!/usr/bin/env python3
"""Course CLI when executed; preserve Python's stdlib tokenize when imported."""
import importlib.util
from pathlib import Path
import sys
import sysconfig

if __name__ == "__main__":
    from tokenizer_boundary import main
    raise SystemExit(main())
else:
    # Other Python libraries import stdlib tokenize. This neighboring filename
    # must not accidentally replace its generate_tokens/open/TokenInfo API.
    path = Path(sysconfig.get_path("stdlib")) / "tokenize.py"
    spec = importlib.util.spec_from_file_location("_dl_stdlib_tokenize", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if not name.startswith("__") or name == "__all__":
            globals()[name] = value
```

Run `python3 scripts/tokenize.py --help` to see the four operations. No decoder function is called in this part. The result of encoding is a checked sequence of integers; the result of decoding is a rendering of integers already selected elsewhere.

Continue to [Part 2 — Observing C execution and selecting vocabulary IDs](https://25d-228.github.io/gpt-c/chapter-21-part-2.html). It will consume this file without converting the IDs to bytes.
