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
