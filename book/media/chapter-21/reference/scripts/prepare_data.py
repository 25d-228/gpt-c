#!/usr/bin/env python3
"""Prepare bounded byte streams; never compute model outputs or gradients."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any

DATASET = "roneneldan/TinyStories"
SPLITS = ("train", "validation")
CAPS = {"train": 16 * 1024 * 1024, "validation": 1024 * 1024}
SEPARATOR = b"\n\n"
RECORD_CAP = 64 * 1024

def write_split(path: Path, rows: Iterable[Mapping[str, Any]], *,
                byte_limit: int, record_limit: int, min_bytes: int) -> dict:
    """Write a prefix of WHOLE records. A non-fitting record ends the prefix."""
    if not (0 < min_bytes <= byte_limit and record_limit > 0):
        raise ValueError("invalid byte/record limits")
    written = inspected = total = 0
    stop = "record_limit"
    iterator = iter(rows)
    with path.open("xb") as stream:
        for _ in range(record_limit):
            try:
                row = next(iterator)
            except StopIteration:
                stop = "source_exhausted"
                break
            inspected += 1
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("record must have a string text field")
            # Bound encoding expansion, too; the source library already owns text.
            if len(text) > RECORD_CAP - len(SEPARATOR):
                raise ValueError("record exceeds the character precheck")
            payload = text.encode("utf-8", errors="strict") + SEPARATOR
            if len(payload) > RECORD_CAP:
                raise ValueError("record exceeds the UTF-8 byte cap")
            if len(payload) > byte_limit - total:
                stop = "byte_limit"
                break
            if stream.write(payload) != len(payload):
                raise OSError("short output write")
            total += len(payload)
            written += 1
            if total == byte_limit:
                stop = "byte_limit"
                break
    if total < min_bytes:
        raise ValueError(f"{path.name}: {total} bytes, need at least {min_bytes}")
    return {"bytes": total, "records": written, "inspected_records": inspected,
            "byte_limit": byte_limit, "record_limit": record_limit, "stop": stop}

def prepare(output: Path, streams: Mapping[str, Iterable[Mapping[str, Any]]],
            source: dict, byte_limits: Mapping[str, int],
            record_limits: Mapping[str, int], min_bytes: int) -> dict:
    """Single-writer publication to a NEW local directory; no overwrite/resume."""
    if set(streams) != set(SPLITS):
        raise ValueError("require separate train and validation streams")
    for split in SPLITS:
        if not (0 < min_bytes <= byte_limits[split] <= CAPS[split]):
            raise ValueError(f"invalid {split} byte limit")
        if record_limits[split] <= 0:
            raise ValueError(f"invalid {split} record limit")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing destination: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staging = Path(tmp) / "prepared"
        staging.mkdir()
        result = {"format": "decoder-lab-byte-v1", "source": source,
                  "encoding": "utf-8", "separator_hex": SEPARATOR.hex(),
                  "record_byte_cap": RECORD_CAP, "min_bytes": min_bytes,
                  "transform": "text encoded strictly, unchanged, then two LF bytes",
                  "splits": {}}
        for split in SPLITS:
            result["splits"][split] = write_split(
                staging / f"{split}.bin", streams[split],
                byte_limit=byte_limits[split], record_limit=record_limits[split],
                min_bytes=min_bytes)
        with (staging / "metadata.json").open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
            f.write("\n")
        # Files have closed successfully. Same-parent rename exposes the set.
        # Single writer: this is not a concurrent no-replace or fsync protocol.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"destination appeared: {output}")
        staging.rename(output)
    return result

def hub_source(revision: str) -> tuple[dict, dict]:
    # Optional imports: --fixture and unit tests need only the standard library.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi(token=False).dataset_info(DATASET, revision=revision)
    resolved = info.sha
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError("Hub did not return an exact commit revision")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    license_id = card.get("license")
    if not isinstance(license_id, str) or not license_id:
        raise ValueError("dataset license metadata is missing; inspect its card")
    files = {split: sorted(s.rfilename for s in (info.siblings or [])
                           if s.rfilename.startswith(f"data/{split}-") and
                           s.rfilename.endswith(".parquet")) for split in SPLITS}
    if not all(files.values()):
        raise ValueError("expected distinct train/validation Parquet files")
    streams = {split: load_dataset(DATASET, data_files=files, split=split,
                                    revision=resolved, streaming=True, token=False)
               for split in SPLITS}
    versions = {"python": sys.version.split()[0]}
    for name in ("datasets", "huggingface_hub", "pyarrow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    source = {"kind": "huggingface-dataset", "dataset": DATASET,
              "requested_revision": revision, "resolved_revision": resolved,
              "url": f"https://huggingface.co/datasets/{DATASET}/tree/{resolved}",
              "license": license_id, "files": files, "versions": versions}
    return source, streams

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--train-bytes", type=int, default=CAPS["train"])
    parser.add_argument("--validation-bytes", type=int, default=CAPS["validation"])
    parser.add_argument("--train-records", type=int, default=50000)
    parser.add_argument("--validation-records", type=int, default=5000)
    parser.add_argument("--min-bytes", type=int, default=129)
    args = parser.parse_args(argv)
    limits = {"train": args.train_bytes, "validation": args.validation_bytes}
    counts = {"train": args.train_records, "validation": args.validation_records}
    # Reject obvious mistakes before contacting the Hub.
    for split in SPLITS:
        if not (0 < args.min_bytes <= limits[split] <= CAPS[split] and counts[split] > 0):
            parser.error(f"invalid {split} limits")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new directory")
    if args.fixture:
        source = {"kind": "original-fixture", "revision": "fixture-v1",
                  "license": "CC0-1.0", "description": "Original four ASCII strings"}
        streams = {"train": [{"text": "ABCD"}, {"text": "EFGH"}],
                   "validation": [{"text": "wxyz"}, {"text": "mnop"}]}
    else:
        source, streams = hub_source(args.revision)
    result = prepare(args.output, streams, source, limits, counts, args.min_bytes)
    for split in SPLITS:
        info = result["splits"][split]
        print(f"{split}: {info['bytes']} bytes, {info['records']} records, {info['stop']}")
    print(f"metadata: {args.output / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"prepare_data: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
