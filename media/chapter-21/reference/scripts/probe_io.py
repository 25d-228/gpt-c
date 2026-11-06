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
