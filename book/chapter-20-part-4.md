# Chapter 20 — Cross-language validation and separate memory measurements

This concludes the chapter begun with [pretrained tensor mapping](https://25d-228.github.io/gpt-c/chapter-20.html). The remaining work is to validate the converter independently, run the actual acquisition/import commands locally, and distinguish numerical buffer accounting from whole-process memory.

## Test the representation, not only a friendly example

Create `tests/test_export_smollm2.py`. The imports and setup below locate the asset module without installing it as a package. Optional executable paths enable the C boundary tests. `TemporaryDirectory` gives each test independent source/output paths, and registered cleanup runs even when an assertion fails.

The `rejects` helper requires the final directory to remain absent and the temporary import directory to be removed. This is stronger than merely catching an exception: a failed conversion must not leave a plausible completed asset behind.

```python title="tests/test_export_smollm2.py"
#!/usr/bin/env python3
"""Offline asset tests, optionally followed by the independent C loader checks."""
from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_smollm2 as exporter
from import_fixture import fixture_source, raw_config, write_source

C_TEST = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
C_APP = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None


class ImportAssets(unittest.TestCase):
```


```python title="tests/test_export_smollm2.py"
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = write_source(self.root / "source")
        self.output = self.root / "converted"

    def convert(self):
        return exporter.export(self.source, self.output, fixture=True)

    def rejects(self):
        with self.assertRaises((ValueError, OverflowError, struct.error)):
            self.convert()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".import-*")))
```

All following method blocks remain indented inside `ImportAssets`; concatenate them in order. The class header appears only in the first block. The final `if __name__` block is unindented.

### Two source precisions, one exact output

The first test converts reversed-order BF16 and F32 files with identical values and requires identical engine files. It verifies payload accounting and the synthetic provenance label, then invokes the independent C exact-value test and inspection application when executable paths are supplied.

The second test visits **all 65,536 BF16 encodings**. For the 65,280 finite encodings, it calculates the reference from sign/exponent/fraction arithmetic using `ldexp`, rather than the converter's `bits << 16`. It also compares the sign of zero. Every one of the 256 nonfinite encodings must be rejected.

```python title="tests/test_export_smollm2.py"
    def test_cross_language_and_f32_identity(self):
        report = self.convert()
        self.assertEqual((report["tensors"], report["values"], report["file_bytes"]),
                         (20, 1128, 6568))
        self.assertEqual(report["weight_bytes"], 4512)
        self.assertEqual(report["model_profile"], "synthetic fixture")
        original = (self.output / "weights.dlw").read_bytes()
        header, payload = fixture_source("F32")
        other_source = write_source(self.root / "f32", header, payload)
        other = self.root / "converted-f32"
        exporter.export(other_source, other, fixture=True)
        self.assertEqual(original, (other / "weights.dlw").read_bytes())
        if C_TEST is not None:
            subprocess.run([str(C_TEST), str(self.output / "weights.dlw")], check=True)
        if C_APP is not None:
            args = [str(C_APP), str(self.output / "model.cfg"), str(self.output / "weights.dlw")]
            result = subprocess.run(args + ["3"], check=True, capture_output=True, text=True)
            self.assertIn("weight_payload_bytes=4512", result.stdout)
            self.assertIn("input_id=3", result.stdout)
            for token in ("16", "-1", "3x", "", " 3"):
                self.assertNotEqual(subprocess.run(args + [token], capture_output=True).returncode, 0)

    def test_all_bf16_encodings(self):
        # An independent arithmetic reference, not the same bit-shift conversion.
        finite = [b for b in range(65536) if b & 0x7f80 != 0x7f80]
        converted = exporter.convert_chunk(struct.pack(f"<{len(finite)}H", *finite), "BF16")
        for bits, (value,) in zip(finite, struct.iter_unpack("<f", converted)):
            exponent, fraction = (bits >> 7) & 255, bits & 127
            magnitude = math.ldexp(float(fraction), -133) if exponent == 0 else \
                        math.ldexp(1 + fraction / 128, exponent - 127)
            expected = -magnitude if bits & 0x8000 else magnitude
            self.assertEqual(value, expected)
            self.assertEqual(math.copysign(1, value), math.copysign(1, expected))
        for bits in range(65536):
            if bits & 0x7f80 == 0x7f80:
                with self.assertRaises(ValueError):
                    exporter.convert_chunk(struct.pack("<H", bits), "BF16")
```

### Reject the wrong inventory before decoding payload

These methods cover missing and extra tensors, an unexpected tied-head record, transposed non-square shapes, rank changes, unsupported dtype, noninteger shape fields, malformed record fields, overlaps, holes, wrong byte counts, trailing bytes, and duplicate JSON keys. The valid fixture's reversed source order remains accepted; rejecting all unusual orders would be the wrong fix.

```python title="tests/test_export_smollm2.py"
    def test_tensor_inventory_and_shapes(self):
        original, payload = fixture_source()
        name = "model.layers.0.self_attn.k_proj.weight"
        cases = []
        h = copy.deepcopy(original); del h[name]; cases.append(h)
        h = copy.deepcopy(original); h["lm_head.weight"] = h[name]; cases.append(h)
        h = copy.deepcopy(original); h[name]["shape"] = [8, 4]; cases.append(h)
        h = copy.deepcopy(original); h[name]["shape"] = [32]; cases.append(h)
        h = copy.deepcopy(original); h[name]["dtype"] = "F16"; cases.append(h)
        h = copy.deepcopy(original); h[name]["shape"] = [4.0, 8]; cases.append(h)
        h = copy.deepcopy(original); h[name]["unexpected"] = 0; cases.append(h)
        for header in cases:
            with self.subTest(header=header[name] if name in header else "missing"):
                write_source(self.source, header, payload)
                self.rejects()

    def test_offsets_and_file_length(self):
        original, payload = fixture_source()
        name = "model.layers.0.self_attn.k_proj.weight"
        for offsets in ([0, 64], [1, 65], [-1, 63], [20, 21], [1], [False, 64]):
            header = copy.deepcopy(original)
            header[name]["data_offsets"] = offsets
            write_source(self.source, header, payload)
            self.rejects()
        for tail in (payload[:-1], payload + b"X"):
            write_source(self.source, original, tail)
            self.rejects()

    def test_duplicate_json_and_headers(self):
        path = self.source / "model.safetensors"
        for data in (b"", b"1234567", struct.pack("<Q", 2**63),
                     struct.pack("<Q", 16) + b"{}",
                     struct.pack("<Q", 15) + b'{"x":1,"x":2}  '):
            path.write_bytes(data)
            self.rejects()
        write_source(self.source)
        (self.source / "config.json").write_text('{"vocab_size":16,"vocab_size":32}')
        self.rejects()
```

A shape-compatible architecture change can still be wrong. The next method rejects rotary layout/scaling changes, untied embeddings, different activation, biases, custom head width, invalid sharing, bad scalars, excessive context, and unknown features. Tiny synthetic geometry is accepted only through the test function argument, never by the production CLI profile.

```python title="tests/test_export_smollm2.py"
    def test_architecture_rejection(self):
        fields = {"rope_interleaved": True, "rope_scaling": {"factor": 2},
                  "tie_word_embeddings": False, "hidden_act": "gelu",
                  "attention_bias": True, "mlp_bias": True, "head_dim": 4,
                  "num_key_value_heads": 3, "rms_norm_eps": 0,
                  "rope_theta": 1, "max_position_embeddings": 8193,
                  "hidden_size": True, "sliding_window": 128}
        for key, value in fields.items():
            with self.subTest(key=key):
                raw = raw_config(); raw[key] = value
                (self.source / "config.json").write_text(json.dumps(raw))
                self.rejects()
        with self.assertRaises(ValueError):
            exporter.configuration(raw_config())  # Tiny data is not the real profile.

    def test_nonfinite_payload_and_cleanup(self):
        for dtype, bits, fmt in (("BF16", 0x7fc0, "<H"), ("BF16", 0x7f80, "<H"),
                                 ("F32", 0x7fc00000, "<I")):
            header, data = fixture_source(dtype)
            payload = bytearray(data)
            # Embedding is last in the source, FIRST in output. Also test the final tensor.
            for name in ("model.embed_tokens.weight", "model.norm.weight"):
                broken = bytearray(payload)
                offset = header[name]["data_offsets"][0]
                struct.pack_into(fmt, broken, offset, bits)
                write_source(self.source, header, broken)
                self.rejects()
```

### Verify resource and publication boundaries

For the chunk check, reduce `CHUNK_VALUES` to seven and record every conversion input length. The sum must cover exactly the source payload and no chunk may exceed fourteen BF16 bytes. This tests bounded calls without timing a machine-dependent allocator.

The next method preserves an existing result byte-for-byte and injects a writer exception in a fresh conversion. Output/source/header budgets are tested separately. An injected write failure is **not** a claim that the operating system actually ran out of disk space; it is a deterministic cleanup test.

```python title="tests/test_export_smollm2.py"
    def test_bounded_chunk_calls(self):
        seen = []
        original = exporter.convert_chunk
        def record(raw, dtype):
            seen.append(len(raw))
            return original(raw, dtype)
        with patch.object(exporter, "CHUNK_VALUES", 7), \
             patch.object(exporter, "convert_chunk", side_effect=record):
            self.convert()
        self.assertGreater(len(seen), 100)
        self.assertLessEqual(max(seen), 14)
        self.assertEqual(sum(seen), 2 * 1128)

    def test_existing_destination_and_write_failure(self):
        self.convert()
        before = (self.output / "weights.dlw").read_bytes()
        with self.assertRaises(FileExistsError):
            self.convert()
        self.assertEqual(before, (self.output / "weights.dlw").read_bytes())
        self.output = self.root / "failed"
        with patch.object(exporter, "write_weights", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.convert()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".import-*")))

    def test_source_and_output_budgets(self):
        with patch.object(exporter, "SOURCE_LIMIT", 16):
            self.rejects()
        with patch.object(exporter, "HEADER_LIMIT", 2):
            self.rejects()
        with patch.object(exporter, "OUTPUT_LIMIT", 6567):
            self.rejects()
        with patch.object(exporter, "OUTPUT_LIMIT", 6568):
            self.convert()
```

The downloader test substitutes an in-memory HTTP response and records the requested URLs. It verifies the pinned revision and cleanup without contacting a server. Its fake weight payload is deliberately not a valid model. This test establishes request construction and failure behavior, not successful live acquisition.

```python title="tests/test_export_smollm2.py"
    def test_download_boundary_without_network(self):
        config = raw_config()
        config.update(vocab_size=49152, num_hidden_layers=30, hidden_size=576,
                      num_attention_heads=9, num_key_value_heads=3, intermediate_size=1536,
                      max_position_embeddings=8192, initializer_range=1/24, rope_theta=100000)
        config_bytes = json.dumps(config).encode()
        requests = []
        def fake(request, timeout):
            requests.append(request.full_url)
            self.assertEqual(timeout, 60)
            return io.BytesIO(config_bytes if request.full_url.endswith("config.json") else b"fake weights")
        target = self.root / "download"
        with patch.object(exporter.urllib.request, "urlopen", side_effect=fake):
            exporter.download(target)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(exporter.REVISION in url for url in requests))
        self.assertEqual((target / "model.safetensors").read_bytes(), b"fake weights")
        # Download success is not tensor validation: export would reject this fake payload.
        failed = self.root / "failed-download"
        with patch.object(exporter.urllib.request, "urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError): exporter.download(failed)
        self.assertFalse(failed.exists())
        self.assertFalse(list(self.root.glob(".download-*")))
```


```python title="tests/test_export_smollm2.py"
if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
```

Run the complete offline checks in any build mode:

```sh
python3 tests/test_export_smollm2.py build/debug/test_import build/debug/inspect_weights
python3 tests/test_export_smollm2.py build/sanitize/test_import build/sanitize/inspect_weights
python3 tests/test_export_smollm2.py build/release/test_import build/release/inspect_weights
make test-assets
```

Expected Python result: **11 tests pass** in the import test file. Its subcases cover more than eleven inputs, including the complete BF16 encoding space. `test-assets` retains Chapter 13's nine preparation tests. Tests in this chapter require no external Python packages or dataset.

## Run the real pinned conversion locally

Keep model files outside the book and documentation branches. From the implementation root:

```sh
python3 scripts/export_smollm2.py \
  --download assets/smollm2-source \
  --output assets/smollm2-fp32

./build/release/inspect_weights \
  assets/smollm2-fp32/model.cfg \
  assets/smollm2-fp32/weights.dlw

./build/release/inspect_weights \
  assets/smollm2-fp32/model.cfg \
  assets/smollm2-fp32/weights.dlw 0
```

These commands create directories that must not already exist. A failed conversion retains a successfully downloaded source directory; retry conversion with `--source assets/smollm2-source` and a new output destination after diagnosing the failure. Do not blindly delete an existing successful asset or repeatedly retry an unchanged error.

For already acquired files, supply a directory containing `config.json` and `model.safetensors`:

```sh
python3 scripts/export_smollm2.py \
  --source /path/to/local/source \
  --output assets/smollm2-fp32
```

Local inputs receive the same schema and value validation, but their provenance is explicitly caller-supplied. The expected revision field does not prove that someone has not substituted another finite tensor file.

Expected load-only ownership counts are:

```text
tensors=272 values=134515008 weight_payload_bytes=538060032 position_limit=8192
```

The directory-byte count depends on the native ABI's `sizeof(DlWeight)` and is printed rather than hard-coded. Gradients and moments are absent. The optional `0` is a literal vocabulary ID for a smoke call, not an encoded natural-language prompt. No expected argmax is supplied here: a meaningful reference result belongs to the next chapter's fixed-token comparison.

The byte-model generator from earlier chapters deliberately requires vocabulary 256. Do not bypass that check and feed raw text bytes to a 49,152-row pretrained table. Token IDs are vocabulary-specific indices; integer 97 is not automatically the pretrained token for the character `a`.

## Separate disk, owned arrays, and resident memory

The unique weight payload contains $P=134,515,008$ values. Its source BF16 payload is $2P=269,030,016$ bytes; the C float payload is $4P=538,060,032$ bytes, or approximately **513.134 MiB**, where one MiB is $2^{20}$ bytes. Keeping both files uses disk space for both representations plus their headers. It does not require both whole representations in the converter's address space.

| Stage | Explicit numeric storage | Other memory to account for |
|---|---:|---|
| BF16 conversion | At most 393,216 live chunk bytes | JSON text/objects, Python runtime, buffered I/O, allocator overhead |
| C weight loading | $4P$ bytes plus 4,096-byte decode chunk | Directory, stack, C runtime, buffered I/O |
| C inference | Weights plus chosen workspace/cache/logits | Metadata, frequencies, positions, kernel stack |

The loader's candidate arena becomes the final arena; there is no extra model-sized commit copy. Initial zeroing still touches it before decoding. A synthetic zero source can be sparse on disk, but C still checks every decoded float and writes every destination value; sparse disk allocation is not an excuse to report zero runtime memory.

Native M4 capacity planning should therefore begin with the approximately 513-MiB weight payload, then add the workspace actually selected. Do not extrapolate an eight-thousand-token full-prefix score matrix merely because the model permits that context. Tiled attention controls temporary score storage; cached inference stores persistent key/value history. Their existing constructors still enforce their separate budgets. This chapter makes no M4 timing claim.

### Build a large geometry fixture without pretending it is pretrained

Add `tests/make_import_size_fixture.py`. It uses the published dimensions but supplies **only zero weights**, with an explicit source metadata note. `ftruncate` extends a regular file; newly extended bytes read as zero, and the filesystem may represent them as holes. The program deliberately refuses an existing directory. It is a test-data generator, not an alternative model downloader.

```python title="tests/make_import_size_fixture.py"
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
```

Run conversion and loading in separate processes. This prevents the converter's runtime or retained buffers from contaminating the loader observation. On macOS, `/usr/bin/time -l` reports the resource statistics; keep its units as printed rather than importing Linux conventions. The exact command below runs on the target native shell:

```sh
python3 tests/make_import_size_fixture.py assets/zeros-source

/usr/bin/time -l python3 scripts/export_smollm2.py \
  --source assets/zeros-source --output assets/zeros-fp32

/usr/bin/time -l ./build/release/inspect_weights \
  assets/zeros-fp32/model.cfg assets/zeros-fp32/weights.dlw
```

On Linux the corresponding resource flag is `-v`, not `-l`. Leave out the optional token ID when measuring loading alone. Running all three inference strategies in the same process would measure their combined retained memory, not the cost of one strategy.

### Actual executed checks

The chapter's code was executed using Clang 17 on x86_64 Linux, not on an M4. The cumulative **20 C test programs** passed in debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized configurations, without compiler warnings. New import tests passed all 1,128 exact loaded-value checks and 512 full/tiled/cached logit comparisons per C run. The 39 malformed/truncated cases and two exact-budget failures left outputs uncommitted. Leak-enabled sanitizer runs reported no leaks. All 11 import-asset Python tests and the inherited nine preparation tests passed in each mode.

The large synthetic fixture also completed conversion and a separate C load. These are observations for **zero data at the real dimensions**, not downloaded pretrained weights:

| Separate Linux process | Wall-clock observation | Peak resident set reported by `time -v` |
|---|---:|---:|
| Standard-library BF16 conversion | 22.12 seconds | 97,164 KiB |
| Optimized C load-only inspection | 0.36 seconds | 526,384 KiB |

The C program reported 538,060,032 weight bytes and 32,640 directory bytes on this ABI. These payload counts and the resident-set measurements answer different questions. The Python process includes its interpreter/imports and environment; its observed peak is not a measurement of only the 393,216-byte numeric chunk. Disk cache state, filesystem sparsity, and machine conditions also affect elapsed time. These two observations are not a speed comparison or a performance promise.

Public configuration and aggregate tensor metadata were accessible during this chapter's preparation. The actual source `model.safetensors` payload and full per-tensor header were **not acquired**: container DNS failed and the separate browser retrieval path did not provide the binary. Therefore no live pretrained conversion/load, pretrained continuation, or independent pretrained-logit agreement is claimed. The supplied converter performs the full tensor inventory check when run on the actual pinned file. Synthetic successes do not remove that remaining execution check.

## Diagnose a mismatch by boundary

A `tensor inventory mismatch` means the source profile is wrong, a required tensor is missing, or the artifact contains an unsupported parameter; do not solve it by ignoring extra keys. A non-square shape reversal usually means a mistaken transpose. Correct shapes but wrong exact fixture values suggest wrong physical source order, wrong relative-to-absolute offset conversion, or BF16 bit conversion.

A load `DL_SHAPE` before payload decoding points to header/directory incompatibility. A late `DL_NUMERIC` points to a nonfinite payload and still must leave the destination empty. A successful load followed by disagreement only at nonzero positions suggests rotary base/pairing/position handling rather than file I/O. A plausible continuation is not enough to decide among those causes.

Three useful deliberate variations expose these boundaries. Replace BF16 bit expansion with a numeric integer-to-float conversion: the independent encoding test must fail. Skip the C directory-offset comparison while continuing to read sequentially: ordinary valid loads still work, but the corrupted-offset test must fail because it is now incorrectly accepted. Remove the cache's imported-position check while retaining the full-prefix checks: the exact-capacity constructor test must fail. All three variations were executed: each failed the intended assertion, and the two C variants produced no sanitizer diagnostic. Restore each change before proceeding; memory sanitizers alone cannot detect these semantic defects.

Before leaving this chapter, calculate one tensor's source byte span, its engine directory offset, and its borrowed arena offset in their respective units. Explain why one tied table is enough, why no projection transpose is performed, and why a clean load does not yet prove pretrained numerical agreement. Record actual implementation time separately from reading and unattended work. Chapter 21 adds revision-matched tokenization and independent expected logits while C remains responsible for every model operation.
