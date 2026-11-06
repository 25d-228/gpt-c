#!/usr/bin/env python3
"""Offline CLI checks; no Python model computation and no performance thresholds."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from import_fixture import raw_config, write_source
from export_smollm2 import export

APP = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if APP is None or not APP.is_file():
            raise RuntimeError("supply an existing compare executable")
        cls.temp = tempfile.TemporaryDirectory(prefix="dl-measure-")
        cls.root = Path(cls.temp.name)
        config = raw_config()
        config["max_position_embeddings"] = 1024
        write_source(cls.root / "source", config=config)
        export(cls.root / "source", cls.root / "converted", fixture=True)
        cls.config = cls.root / "converted/model.cfg"
        cls.weights = cls.root / "converted/weights.dlw"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def call(self, args, fail=False, **kwargs):
        command = [str(APP), *map(str, args)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, **kwargs)
        self.assertEqual(result.returncode != 0, fail, msg=f"{command}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    def model(self, action="check", mode="cache", backend="blocked", prompt=3, steps=3,
              reps=1, qt=3, kt=5):
        return ["model", action, self.config, self.weights, mode, backend,
                prompt, steps, reps, qt, kt]

    def training(self, action="check", mode="tiled", backend="blocked", time=5, reps=1):
        return ["training", action, self.config, mode, backend, time, reps, 3, 5]

    def test_all_inference_choices(self):
        for mode in ("full", "tiled", "cache"):
            for backend in ("ref", "blocked"):
                output = self.call(self.model(mode=mode, backend=backend))
                self.assertIn("checked_values=64", output)
                self.assertNotIn("peak_rss", output)
                self.assertNotIn("phase=", output)

    def test_single_token_prefill_and_tail_tiles(self):
        for mode in ("full", "tiled", "cache"):
            output = self.call(self.model(mode=mode, prompt=1, steps=6, qt=2, kt=3))
            self.assertIn("checked_values=112", output)

    def test_measure_replays_and_accounting(self):
        for mode in ("full", "tiled", "cache"):
            output = self.call(self.model("measure", mode, reps=2))
            self.assertIn("repeat_final_logits=exact", output)
            self.assertIn("peak_rss_native=", output)
            self.assertNotIn("checked_values", output)
            parts = re.search(r"payload weights=(\d+) scratch=(\d+) history=(\d+) "
                              r"other_heap=(\d+) total_heap=(\d+)", output)
            self.assertIsNotNone(parts)
            weights, scratch, history, other, total = map(int, parts.groups())
            self.assertEqual(weights, 1128 * 4)
            self.assertEqual(total, weights + scratch + history + other)
            self.assertEqual(history, 2 * 2 * 6 * 4 * 4 if mode == "cache" else 0)
            for phase in ("prefill", "decode"):
                values = re.findall(rf"sample phase={phase} index=\d batch_seconds=(\S+)", output)
                self.assertEqual(len(values), 5)
                self.assertTrue(all(math.isfinite(float(x)) and float(x) > 0 for x in values))
                line = next(x for x in output.splitlines() if x.startswith(f"phase={phase} "))
                fields = dict(x.split("=", 1) for x in line.split())
                self.assertEqual(int(fields["repetitions"]), 2)
                median = sorted(map(float, values))[2] / 2
                self.assertAlmostEqual(float(fields["median_seconds"]), median, delta=median*2e-8)
                self.assertAlmostEqual(float(fields["seconds_per_item"]), median/3, delta=median*2e-8)

    def test_gradients_all_backends_and_strategies(self):
        for mode in ("full", "tiled"):
            for backend in ("ref", "blocked"):
                for time in (1, 5):
                    output = self.call(self.training(mode=mode, backend=backend, time=time))
                    self.assertIn("checked_gradients=1128", output)
                    self.assertNotIn("peak_rss", output)

    def test_training_measure_has_no_optimizer_update(self):
        for mode in ("full", "tiled"):
            output = self.call(self.training("measure", mode, reps=2))
            self.assertIn("no_optimizer_update=1", output)
            self.assertIn("items_per_replay=5", output)
            parts = re.search(r"payload parameters_and_gradients=(\d+) tape=(\d+) "
                              r"other_heap=(\d+) total_heap=(\d+)", output)
            self.assertIsNotNone(parts)
            parameters, tape, other, total = map(int, parts.groups())
            self.assertEqual(parameters, 1128*8)
            self.assertEqual(total, parameters + tape + other)

    def test_training_tape_scaling(self):
        counts = {}
        for mode in ("full", "tiled"):
            for time in (16, 32, 64):
                output = self.call(self.training("measure", mode, time=time))
                counts[mode, time] = int(re.search(r" tape=(\d+)", output).group(1))
        self.assertEqual(counts["tiled", 64] - counts["tiled", 32],
                         2 * (counts["tiled", 32] - counts["tiled", 16]))
        self.assertGreater(counts["full", 64], counts["tiled", 64])
        self.assertGreater(counts["full", 64] - counts["full", 32],
                            2 * (counts["full", 32] - counts["full", 16]))

    def test_invalid_commands(self):
        cases = [["model"], ["training"], self.model(reps=0), self.model(prompt=0),
                 self.model(steps=0), self.model(qt=257), self.model(kt="-1"),
                 self.model(prompt="9999999999999999999999999999999"),
                 self.model(prompt=4096, steps=1), self.model(mode="unknown"),
                 self.model(backend="both"), self.training(time=0),
                 self.training(mode="cache"), self.training(reps=1001)]
        for args in cases:
            with self.subTest(args=args):
                output = self.call(args, fail=True)
                self.assertNotIn("repeat_final_logits=exact", output)

    def test_imported_position_bound(self):
        output = self.call(self.model(prompt=1024, steps=1), fail=True)
        self.assertNotIn("phase=", output)

    def test_bad_weights_and_missing_config(self):
        bad = self.root / "bad.dlw"
        original = self.weights.read_bytes()
        bad.write_bytes(original[:-1])
        args = self.model()
        args[3] = bad
        self.call(args, fail=True)
        self.assertEqual(bad.read_bytes(), original[:-1])
        args = self.model()
        args[2] = self.root / "missing.cfg"
        self.call(args, fail=True)
        self.assertEqual(self.weights.read_bytes(), original)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires an OS failing-output device")
    def test_output_failure(self):
        with open("/dev/full", "wb") as sink:
            result = subprocess.run([str(APP), *map(str, self.model("measure"))],
                                    stdout=sink, stderr=subprocess.PIPE, timeout=90)
        self.assertNotEqual(result.returncode, 0)

    def test_pretrained_driver_rejects_fixture_provenance(self):
        tokenizer = self.root / "fake-tokenizer"
        tokenizer.mkdir()
        (tokenizer / "source.json").write_text(json.dumps({"model": "fixture", "revision": "none"}))
        output = self.root / "rejected-pretrained"
        result = subprocess.run(["sh", str(ROOT/"scripts/run_pretrained_demo.sh"),
                                 str(self.root/"source"), str(self.root/"converted"),
                                 str(tokenizer), str(output)], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a fixture", result.stderr)
        self.assertFalse((output/"completed.txt").exists())

    def test_demo_refuses_existing_directory(self):
        keep = self.root / "existing"
        keep.mkdir()
        (keep / "keep.txt").write_text("unrelated work")
        result = subprocess.run(["sh", str(ROOT/"scripts/run_demo.sh"), str(keep)],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((keep / "keep.txt").read_text(), "unrelated work")
        self.assertEqual(sorted(x.name for x in keep.iterdir()), ["keep.txt"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
