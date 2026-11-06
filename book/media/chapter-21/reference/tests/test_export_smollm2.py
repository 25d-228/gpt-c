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

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
