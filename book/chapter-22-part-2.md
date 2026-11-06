# Chapter 22, Part 2 — Measure forward/backward pairs and test the harness

Continue `app/measure.c` from [Part 1](https://25d-228.github.io/gpt-c/chapter-22.html). This part measures the complete model forward/backward path with fixed parameters. It does not report that interval as an AdamW update or an end-to-end training step: it excludes data sampling, gradient accumulation groups, clipping, moment updates, validation, and checkpoints. The real training application will measure complete updates in Part 3.

## Keep the training workload identical

The config supplies batch size $B$; the command supplies sequence length $T$. There are $M=BT$ input/target pairs. For batch item $b$ and token position $t$, generate a bounded deterministic input ID and the next position's target. Reusing the same weights and targets on every replay makes any change in loss or gradients a state bug rather than a training effect.

The existing forward call stores the tape revision and makes its activations available. Backward consumes that forward and writes every unique parameter gradient using `DL_SET`. A subsequent replay must run forward again. There is no AdamW owner and no parameter update. This deliberately includes the actual forward/backward validation, projections, normalization, loss, and shared-gradient accumulation—not only a favorable attention inner loop.

Create the following three helpers after Part 1's final function. `gradient_sum` inspects every gradient for finiteness after the timed batch. The squared sum detects some replay mistakes cheaply, but it is not an equality proof: two different vectors can have equal norm. The `check` action below compares each gradient coordinate separately.

```c title="app/measure.c"
static DlStatus tape_create(DlTrainTape *tape, const DlModel *model,
                             size_t batch, size_t time, int tiled, size_t qt, size_t kt)
{
    if (tiled) return dl_train_tape_create_tiled(tape, model, batch, time, qt, kt, 512*MIB);
    return dl_train_tape_create(tape, model, batch, time, 512*MIB);
}

static DlStatus forward_backward(DlModel *model, DlTrainTape *tape, DlBackend backend,
                                  const size_t *ids, const size_t *targets, double *loss)
{
    DlStatus s = dl_model_forward(model, tape, backend, ids, targets, tape->rows, loss);
    if (s != DL_OK) return s;
    return dl_model_backward(model, tape, DL_SET);
}

static int gradient_sum(const DlModel *model, double *sum)
{
    double x = 0.0;
    if (!model->gradients_valid) return 0;
    for (size_t p = 0; p < model->param_count; ++p)
        for (size_t i = 0; i < model->params[p].grad.count; ++i) {
            double v = (double)model->params[p].grad.data[i];
            if (!isfinite(v)) return 0;
            x += v*v;
        }
    if (!isfinite(x)) return 0;
    *sum = x;
    return 1;
}

```

## Derive the storage distinction before measuring RSS

Let $P_{\theta}$ be the number of unique trainable float values. Parameters and their gradients need $8P_{\theta}$ bytes, because each occupies four bytes. Tied embeddings have one value/gradient owner; they are not counted twice. An optimizer is absent from this particular process.

For a materialized tape, let $S=BH_qT^2$, and keep the meanings of $L,D,K,F,V,M$ from Part 1. The Chapter 12 layout, retained for the materialized path, contains

$$
A_{\mathrm{full}}=(L+1)\left[6MD+3M(D+K)+2S+3MF\right]+2MV
$$

float cells. The $L$ saved blocks and the shared reverse workspace explain the $L+1$ factor. In tiled mode the $2S$ terms disappear. There are instead per-block forward double workspaces and one shared reverse double workspace:

$$
A_{\mathrm{linear}}=(L+1)\left[6MD+3M(D+K)+3MF\right]+2MV,
$$

$$
A_{\mathrm{forward,double}}=L\left[3BH_qT+Q_bK_b+Q_bd+2Q_b\right],
$$

$$
A_{\mathrm{backward,double}}=2Q_bK_b+Q_bd+2K_bd+Q_b.
$$

Thus numerical tiled-tape bytes are $4A_{\mathrm{linear}}+8(A_{\mathrm{forward,double}}+A_{\mathrm{backward,double}})$. Every term is at most linear in $T$ with fixed architecture, batch and tiles. The computation is still dense causal attention and remains quadratic in $T$.

For the pilot at $B=2,T=256$, $M=512$, $S=524{,}288$, and $P_{\theta}=29{,}856$. Materialization needs 17,170,432 tape bytes; 16-by-32 tiling needs 4,710,016. The numerical tape reduction is approximately 3.65-fold. Parameter/gradient bytes, metadata, allocator/runtime pages, and timings are not multiplied by this ratio.

Use the existing `dl_train_tape_payload_bytes` implementation to count the numerical tape. The remaining heap count includes the parameter directory, hidden/block view records, five rows' worth of index storage per token (three in the tape and two external input/target arrays), frequencies, and the tiled owner records. It does not include the stack-embedded model/tape structures.

```c title="app/measure.c"
static int training_storage(const DlModel *model, const DlTrainTape *tape)
{
    size_t parameters = 0, payload = 0, other = 0;
    size_t layers = model->config.layers, rows = tape->rows;
    if (dl_train_tape_payload_bytes(tape, &payload) != DL_OK ||
        !add_bytes(&parameters, model->value_count, 2*sizeof(float)) ||
        !add_bytes(&other, model->param_count, sizeof(DlParameter)) ||
        !add_bytes(&other, layers+1, sizeof(DlMatrix)) ||
        !add_bytes(&other, layers, sizeof(DlBlockTape)) ||
        !add_bytes(&other, rows, 5*sizeof(size_t)) ||
        !add_bytes(&other, model->config.width/model->config.query_heads/2, sizeof(double)) ||
        (tape->tiled != NULL && !add_bytes(&other, layers, sizeof(DlTiledWorkspace)))) return 0;
    size_t total = parameters;
    if (!add_bytes(&total, payload, 1) || !add_bytes(&total, other, 1)) return 0;
    printf("payload parameters_and_gradients=%zu tape=%zu other_heap=%zu total_heap=%zu\n",
           parameters, payload, other, total);
    return !ferror(stdout);
}

```

## Complete the training entry point

The parameter budget bounds both values and gradients at 128 MiB. That admits the 2.9-million-parameter learner model but intentionally rejects training the 135-million-parameter inference configuration with this diagnostic command. The tape budget is independently 512 MiB, and `B*T` is capped at 65,536 targets. All products used for allocations are validated before arithmetic or allocation.

In `check`, two separately initialized models use the same explicit seed and configuration. A materialized scalar forward/backward supplies a loss and every parameter-gradient element. The candidate uses the requested backend and attention policy. Both parameter sets remain fixed. In `measure`, the reference model/tape are never constructed. One initial forward/backward is the warmup; each of five timed batches then repeats the pair. Finiteness, loss stability and gradient-norm stability are checked outside the interval.

```c title="app/measure.c"
int dl_compare_training(int argc, char **argv)
{
    size_t time = 0, reps = 0, qt = 0, kt = 0;
    DlBackend backend = DL_BACKEND_REF;
    if (argc != 10 || (strcmp(argv[2], "check") && strcmp(argv[2], "measure")) ||
        (strcmp(argv[4], "full") && strcmp(argv[4], "tiled")) ||
        !backend_name(argv[5], &backend) || !positive(argv[6], &time) ||
        !positive(argv[7], &reps) || !positive(argv[8], &qt) || !positive(argv[9], &kt) ||
        time > MAX_TOKENS || reps > MAX_REPETITIONS || qt > 256 || kt > 256) {
        fprintf(stderr, "usage: %s training check|measure CONFIG full|tiled "
                "ref|blocked TIME REPS QUERY_TILE KEY_TILE\n", argv[0]);
        return EXIT_FAILURE;
    }
    int result = EXIT_FAILURE, check = strcmp(argv[2], "check") == 0;
    DlModelConfig config = {0};
    DlModel model = {0}, reference = {0};
    DlTrainTape tape = {0}, reference_tape = {0};
    size_t batch = 0, ignored_t = 0, param_count = 0, values = 0, bytes = 0, rows = 0;
    size_t *ids = NULL, *targets = NULL;
    TRY(dl_model_config_load(argv[3], &config, &batch, &ignored_t));
    TRY(dl_model_config_validate(config, &param_count, &values));
    TRY(dl_size_mul(values, 2*sizeof(float), &bytes));
    TRY(dl_size_mul(batch, time, &rows));
    REQUIRE(bytes <= 128*MIB && rows <= 65536);
    TRY(dl_model_create(&model, config));
    TRY(tape_create(&tape, &model, batch, time, strcmp(argv[4], "tiled") == 0, qt, kt));
    ids = malloc(rows*sizeof(*ids));
    targets = malloc(rows*sizeof(*targets));
    REQUIRE(ids != NULL && targets != NULL);
    for (size_t b = 0; b < batch; ++b)
        for (size_t t = 0; t < time; ++t) {
            ids[b*time+t] = (13*(b%97)+37*(t%97)+11) % config.vocab;
            targets[b*time+t] = (13*(b%97)+37*((t+1)%97)+11) % config.vocab;
        }
    printf("training action=%s mode=%s backend=%s batch=%zu time=%zu "
           "query_tile=%zu key_tile=%zu parameters=%zu\n", argv[2], argv[4], argv[5],
           batch, time, qt, kt, values);
    REQUIRE(environment());
    double loss = 0.0;
    TRY(forward_backward(&model, &tape, backend, ids, targets, &loss));
    if (check) {
        TRY(dl_model_create(&reference, config));
        TRY(tape_create(&reference_tape, &reference, batch, time, 0, qt, kt));
        double reference_loss = 0.0, largest = 0.0;
        TRY(forward_backward(&reference, &reference_tape, DL_BACKEND_REF,
                             ids, targets, &reference_loss));
        REQUIRE(isfinite(loss) && isfinite(reference_loss) && fabs(loss-reference_loss) <= 3e-5);
        for (size_t p = 0; p < param_count; ++p)
            REQUIRE(close_values(reference.params[p].grad.data, model.params[p].grad.data,
                                 model.params[p].grad.count, 3e-5, 3e-5, &largest));
        printf("checked_gradients=%zu max_abs=%.9g loss_abs=%.9g atol=3e-5 rtol=3e-5\n",
               values, largest, fabs(loss-reference_loss));
        puts("check_only=no_timing_or_rss_reference_buffers_are_live");
    } else {
        REQUIRE(training_storage(&model, &tape));
        double initial_loss = loss, initial_sum = 0.0, samples[SAMPLE_COUNT] = {0};
        REQUIRE(gradient_sum(&model, &initial_sum));
        for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
            struct timespec a = {0}, b = {0};
            REQUIRE(tick(&a));
            for (size_t repeat = 0; repeat < reps; ++repeat)
                TRY(forward_backward(&model, &tape, backend, ids, targets, &loss));
            REQUIRE(tick(&b));
            samples[sample] = elapsed(a, b);
            double sum = 0.0;
            REQUIRE(gradient_sum(&model, &sum) && loss == initial_loss && sum == initial_sum);
        }
        REQUIRE(report("forward_backward", samples, reps, rows));
        REQUIRE(resident_peak());
        printf("loss=%.9g gradient_squared_sum=%.12g no_optimizer_update=1\n", loss, initial_sum);
    }
    REQUIRE(fflush(stdout) == 0);
    result = EXIT_SUCCESS;
done:
    free(targets);
    free(ids);
    dl_train_tape_destroy(&reference_tape);
    dl_model_destroy(&reference);
    dl_train_tape_destroy(&tape);
    dl_model_destroy(&model);
    return result;
}

#undef TRY
#undef REQUIRE
```

## Connect the command without changing old semantics

Add `#include "measure.h"` immediately after `#include "pretrained.h"` in the Chapter 21 `app/compare.c`:

```c title="app/compare.c: include insertion"
#include "measure.h"
```

At the start of its existing `main`, before the existing `pretrained` dispatch, insert these complete conditional statements. Keep the rest of `main`, including the kernel CLI, unchanged:

```c title="app/compare.c: dispatch insertion"
    if (argc > 1 && strcmp(argv[1], "model") == 0)
        return dl_compare_model(argc, argv);
    if (argc > 1 && strcmp(argv[1], "training") == 0)
        return dl_compare_training(argc, argv);
```

Replace the three existing `build/<mode>/compare` rules in the Makefile with the following. The helper sources are dependencies as well as compilation inputs: changing the timer must rebuild its executable. The other applications, common modules, compiler flags, and all earlier targets remain unchanged. Tabs before recipes are literal Makefile tabs.

```makefile title="Makefile: compare rules"
build/debug/compare: app/compare.c app/pretrained.c app/pretrained.h app/measure.c app/measure.h $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) app/pretrained.c app/measure.c $< -o $@ $(LDLIBS)

build/sanitize/compare: app/compare.c app/pretrained.c app/pretrained.h app/measure.c app/measure.h $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) app/pretrained.c app/measure.c $< -o $@ $(LDLIBS)

build/release/compare: app/compare.c app/pretrained.c app/pretrained.h app/measure.c app/measure.h $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) app/pretrained.c app/measure.c $< -o $@ $(LDLIBS)
```

Append the new optional integration target:

```makefile title="Makefile: measurement target"
.PHONY: test-measure
test-measure: build/release/compare
	python3 tests/test_measure.py build/release/compare
```

## Test arithmetic, state, and the reporting boundary

Create `tests/test_measure.py` using these two consecutive listings. It imports the complete `tests/import_fixture.py` and `scripts/export_smollm2.py` from Chapter 20. The source weights are explicitly synthetic; the model/tokenizer is not downloaded. Python creates bytes and launches C, but performs no model computation.

The first half tests six inference choices, one-token prefill, partial tiles, and exact accounting. It independently calculates the median/divisors from the five raw samples rather than merely searching for the word “passed.” It checks for the absence of RSS/timing fields in correctness output. A cache-history count is calculated from its shape, independently of the C reporter.

```python title="tests/test_measure.py"
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

```

The remaining tests compare all unique gradients, verify that the tiled tape grows linearly at three lengths, reject malformed/overflowing commands, enforce imported positions, and exercise an actual failing output device when available. Neither time nor RSS is required to beat another implementation. The last two tests use the scripts supplied in Parts 3 and 4, so run this suite after assembling the complete chapter.

The imported-position failure is distinct from the application's 4,096-ID limit: a 1,025-ID capacity fits the latter but exceeds this fixture's imported 1,024-position limit. Failure must occur without a timing report. The checkpoint fixture is also truncated deliberately, ensuring the benchmark cannot time a partially loaded model.

```python title="tests/test_measure.py"
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
```

After adding the scripts from the next parts, run:

```sh
make test
python3 tests/test_measure.py build/debug/compare
python3 tests/test_measure.py build/sanitize/compare
python3 tests/test_measure.py build/release/compare
make test-assets test-import-assets test-reference test-training-cli test-generation-cli test-tiled-cli
```

Use `CC=clang` with `make` on a compatible non-Apple host instead of pretending `xcrun` exists there. The measurement tests must receive an actual executable; an absent path is an error, not a skipped cross-language run. There are twelve new tests. `/dev/full` is an optional host facility and its test explicitly skips when absent. Such a skip is not evidence that output-failure handling ran on that host.

## Two diagnostic changes that should fail

In a disposable copy, replace cached decode's `ids[count-1]` with `ids[0]` in `prefix`. The memory access remains valid, but later prefixes no longer match the reference. The all-prefix test must fail; a sanitizer need not complain. Testing only prefill would miss this bug.

Restore the ID selection. Now remove the reset before each cached replay. The warmup fills the cache; the next prefill requires an empty cache and fails `DL_STATE`. A partially printed header is not a measurement result. Do not change the cache API to admit the call merely to obtain a number.

A third useful mutation changes `median/(double)items` to `median`. All model outputs can remain correct while the reported per-item number becomes wrong. The independent division test is what detects that error. These are changes to copied sources, never reasons to weaken the correct comparator.

Part 3 puts training, restart, generation, correctness checks, and independent measurement processes into one ordinary shell driver.
