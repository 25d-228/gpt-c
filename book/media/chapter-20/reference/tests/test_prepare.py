#!/usr/bin/env python3
"""Offline preparation checks; all text fixtures are original."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import CAPS, RECORD_CAP, hub_source, main, prepare, write_split


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = {"kind": "original-fixture", "revision": "test-v1"}

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, streams, output=None, **changes):
        options = {"byte_limits": {"train": 40, "validation": 40},
                   "record_limits": {"train": 5, "validation": 5}, "min_bytes": 2}
        options.update(changes)
        return prepare(output or self.root / "data", streams, self.source, **options)

    def test_fixture_and_no_overwrite(self):
        out = self.root / "fixture"
        self.assertEqual(main(["--fixture", "--output", str(out), "--min-bytes", "5"]), 0)
        self.assertEqual((out / "train.bin").read_bytes(), b"ABCD\n\nEFGH\n\n")
        self.assertEqual((out / "validation.bin").read_bytes(), b"wxyz\n\nmnop\n\n")
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["source"]["kind"], "original-fixture")
        self.assertEqual(meta["splits"]["train"]["bytes"], 12)
        with self.assertRaises(FileExistsError):
            self.run_prepare({"train": [], "validation": []}, output=out)
        self.assertEqual((out / "train.bin").read_bytes(), b"ABCD\n\nEFGH\n\n")

    def test_raw_encoding_and_separator(self):
        text = "A\x00\r\né中"
        data = [{"text": text}]
        self.run_prepare({"train": data, "validation": [{"text": "V"}]})
        self.assertEqual((self.root / "data/train.bin").read_bytes(),
                         b"A\x00\r\n\xc3\xa9\xe4\xb8\xad\n\n")

    def test_whole_records_not_character_counts(self):
        seen = []
        def rows():
            for text in ("é", "éé", "A"):
                seen.append(text)
                yield {"text": text}
        path = self.root / "bounded.bin"
        result = write_split(path, rows(), byte_limit=8, record_limit=10, min_bytes=2)
        self.assertEqual(path.read_bytes(), b"\xc3\xa9\n\n")
        self.assertEqual(seen, ["é", "éé"])
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["inspected_records"], 2)
        self.assertEqual(result["stop"], "byte_limit")

    def test_record_count_does_not_fetch_one_more(self):
        def rows():
            yield {"text": "A"}
            raise AssertionError("must not fetch another record")
        result = write_split(self.root / "one.bin", rows(), byte_limit=100,
                             record_limit=1, min_bytes=2)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["stop"], "record_limit")

    def test_exact_byte_limit_does_not_fetch_one_more(self):
        def rows():
            yield {"text": "A"}
            raise AssertionError("must stop at exact byte cap")
        result = write_split(self.root / "exact.bin", rows(), byte_limit=3,
                             record_limit=100, min_bytes=3)
        self.assertEqual(result["bytes"], 3)
        self.assertEqual(result["stop"], "byte_limit")

    def test_failure_does_not_publish_either_split(self):
        def broken():
            yield {"text": "V"}
            raise OSError("injected input failure")
        with self.assertRaises(OSError):
            self.run_prepare({"train": [{"text": "TRAIN"}], "validation": broken()})
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_short_and_invalid_records(self):
        for rows in ([], [{"text": "A"}], [{"text": None}],
                     [{"text": "\ud800"}], [{"text": "A" * RECORD_CAP}],
                     [{"text": "中" * (RECORD_CAP // 2)}]):
            with self.subTest(kind=str(rows)[:30]):
                with self.assertRaises((ValueError, UnicodeError)):
                    self.run_prepare({"train": rows, "validation": [{"text": "V"}]},
                                     min_bytes=5)
                self.assertFalse((self.root / "data").exists())

    def test_split_and_cap_validation(self):
        with self.assertRaises(ValueError):
            self.run_prepare({"train": [{"text": "A"}]})
        streams = {"train": [{"text": "A"}], "validation": [{"text": "V"}]}
        for limits in ({"train": CAPS["train"] + 1, "validation": 40},
                       {"train": 40, "validation": CAPS["validation"] + 1},
                       {"train": 0, "validation": 40}):
            with self.assertRaises(ValueError):
                self.run_prepare(streams, byte_limits=limits)
        self.assertFalse((self.root / "data").exists())

    def test_hub_adapter_pins_both_splits(self):
        # Interface-level test, not evidence of a network transfer or Parquet decode.
        revision = "a" * 40
        calls = []
        info = types.SimpleNamespace(
            sha=revision,
            card_data=types.SimpleNamespace(to_dict=lambda: {"license": "cdla-sharing-1.0"}),
            siblings=[types.SimpleNamespace(rfilename=s) for s in
                      ("data/train-b.parquet", "data/validation-a.parquet",
                       "data/train-a.parquet", "TinyStories-train.txt")])
        class Api:
            def __init__(self, **kwargs):
                self.options = kwargs
            def dataset_info(self, name, **kwargs):
                calls.append(("info", name, kwargs, self.options))
                return info
        def load(name, **kwargs):
            calls.append(("load", name, kwargs))
            return [{"text": kwargs["split"]}]
        modules = {"datasets": types.SimpleNamespace(load_dataset=load),
                   "huggingface_hub": types.SimpleNamespace(HfApi=Api)}
        with patch.dict(sys.modules, modules):
            source, streams = hub_source("main")
        self.assertEqual(source["resolved_revision"], revision)
        self.assertEqual(source["license"], "cdla-sharing-1.0")
        self.assertEqual(set(streams), {"train", "validation"})
        self.assertEqual(calls[0][2]["revision"], "main")
        self.assertFalse(calls[0][3]["token"])
        self.assertEqual([c[2]["split"] for c in calls[1:]], ["train", "validation"])
        for _, _, options in calls[1:]:
            self.assertEqual(options["revision"], revision)
            self.assertTrue(options["streaming"])
            self.assertFalse(options["token"])
            self.assertEqual(options["data_files"]["train"],
                             ["data/train-a.parquet", "data/train-b.parquet"])
            self.assertEqual(options["data_files"]["validation"],
                             ["data/validation-a.parquet"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
