# Chapter 22, Part 3 — A repeatable training, restart, and generation run

Parts 1 and 2 added measurement entry points; this part connects the already implemented model to explicit inputs and retained outputs. The driver does not invent a new learning loop, call a framework model, or hide a tokenizer inside C. It invokes the existing applications and stops when an invocation fails.

The main demonstration uses the 29,856-parameter pilot to keep repeated integration checks inexpensive. The selected 2,885,888-parameter byte model uses the same operators, applications, and config format; its separate bounded run appears in Part 4. Genuine pretrained weights are a third input state for the same inference engine, not a substitute for training the byte model. Their external-asset workflow is kept separate so an unavailable network cannot silently replace the model with a fixture.

## Own a run directory, not a global results folder

Create `scripts/run_demo.sh` from the following consecutive listings. Invoke it with `sh`; its execute permission is not a requirement. It accepts exactly one new output directory. `mkdir` is the no-overwrite boundary: existing directories, regular files, and symlinks cause failure. It does not delete unrelated work, delete a failed run, or retry a command. A partial output directory is diagnostic material, not proof that the demonstration completed.

`set -e` makes a failing standalone command stop this driver. `set -u` catches undefined variables. Neither setting is a universal shell error-handling system: a pipeline can hide an earlier failure, and conditions have special rules. The driver therefore keeps critical C calls and `cmp` as standalone commands rather than piping them into `tee` and trusting the last pipeline status.

The first section resolves the source root independently of the current directory, preserves a caller-relative output path, bounds integer settings before shell arithmetic, and records machine/build inputs. No macOS version or chip model is hard-coded. On Linux, host-memory and cgroup records are printed separately because a container's visible machine is not necessarily its resource allocation. The `build.log` contains the actual compiler invocations and flags; `-B` rebuilds the selected applications even when stale binaries exist.

```sh title="scripts/run_demo.sh"
#!/bin/sh
# All model computation is in the C executables. No downloads or hidden retries.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: sh scripts/run_demo.sh NEW_OUTPUT_DIRECTORY" >&2
    exit 1
fi
root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
case "$1" in /*) out=$1 ;; *) out=$PWD/$1 ;; esac
# mkdir is the no-overwrite boundary, including existing files and symlinks.
mkdir "$out"
cd "$root"
config=${DEMO_CONFIG:-configs/byte-pilot.conf}
updates=${DEMO_UPDATES:-160}
reps=${DEMO_REPS:-5}
for value in "$updates" "$reps"; do
    case "$value" in ''|0*|*[!0-9]*) echo "counts must be decimal integers" >&2; exit 1 ;; esac
    # Bound length before shell arithmetic; avoid platform integer overflow.
    if [ "${#value}" -gt 5 ]; then echo "count is too large" >&2; exit 1; fi
done
if [ "$updates" -lt 2 ] || [ "$updates" -gt 10000 ] || [ "$reps" -lt 1 ] || [ "$reps" -gt 1000 ]; then
    echo "need updates 2..10000 and repetitions 1..1000" >&2
    exit 1
fi
split=$((updates / 2))
{
    date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
    uname -a
    printf 'CC=%s\nCONFIG=%s\nUPDATES=%s\nREPS=%s\n' "${CC:-xcrun clang}" "$config" "$updates" "$reps"
    if [ "$(uname -s)" = Darwin ]; then
        sw_vers
        sysctl -n machdep.cpu.brand_string hw.memsize
        xcrun clang --version
    else
        getconf LONG_BIT
        if [ -r /proc/cpuinfo ]; then sed -n '/model name/{p;q;}' /proc/cpuinfo; fi
        if [ -r /proc/meminfo ]; then sed -n '1p' /proc/meminfo; fi
        # Container host totals are not a promise of available/assigned memory.
        if [ -r /sys/fs/cgroup/memory.max ]; then cat /sys/fs/cgroup/memory.max; fi
        if [ -r /sys/fs/cgroup/cpu.max ]; then cat /sys/fs/cgroup/cpu.max; fi
    fi
} > "$out/machine.txt"
cp Makefile "$out/build-settings.txt"
cp "$config" "$out/model.conf"
# Force rebuilding the two applications and their helper modules, not stale binaries.
make -B CC="${CC:-xcrun clang}" build/release/compare build/release/train \
    build/release/generate > "$out/build.log" 2>&1
python3 scripts/make_training_fixture.py "$out/text" > "$out/preparation.txt"
train=./build/release/train
compare=./build/release/compare
generate=./build/release/generate
```

## Replay the same schedule, not just the same weights

For each materialized/tiled training strategy, run one uninterrupted trajectory and one split trajectory. Both start with the same explicit model/data random states and use the **same total schedule length**. Only `STOP` differs at the first boundary. Resuming a checkpoint under a newly shortened schedule would be a different experiment.

`cmp` checks final checkpoint bytes within each strategy. It does not demand byte-identical full-versus-tiled training checkpoints: their arithmetic and saved attention policies differ. The v1/v2 formats, optimizer moments, update count, data random state, and exact dataset bytes are already covered by Chapter 15/19's continuation checks.

With the default config, $B=2,T=16$, and one microbatch, each successful update processes 32 targets. At 160 updates, each trajectory has processed 5,120 target occurrences, sampled with replacement from the small diagnostic text. That is not 5,120 distinct bytes of new data. The extra split run repeats the same trajectory for verification; it does not double the model's effective training count.

The generation calls all load the **same** full-training checkpoint. They compare full-prefix, tiled-prefix, and cached execution for greedy selection and for categorical sampling with the same seed/temperature. Each transcript contains its ten-byte prompt plus forty selected bytes; only 49 inputs need to be consumed. Byte equality is useful for this deterministic diagnostic fixture, but it is not a universal requirement across platforms when near-tied logits differ. A mismatch should lead to all-logit and margin diagnosis, not an automatic tolerance increase or a hidden changed seed.

```sh title="scripts/run_demo.sh"
for strategy in full tiled; do
    "$train" new "$config" "$out/text/train.bin" "$out/text/valid.bin" \
        "$out/$strategy-whole.ckpt" "$updates" "$updates" 1 random blocked "$strategy" \
        > "$out/$strategy-whole.log"
    "$train" new "$config" "$out/text/train.bin" "$out/text/valid.bin" \
        "$out/$strategy-split.ckpt" "$updates" "$split" 1 random blocked "$strategy" \
        > "$out/$strategy-first.log"
    "$train" resume "$config" "$out/text/train.bin" "$out/text/valid.bin" \
        "$out/$strategy-split.ckpt" "$updates" "$updates" 1 random blocked "$strategy" \
        > "$out/$strategy-resume.log"
    cmp "$out/$strategy-whole.ckpt" "$out/$strategy-split.ckpt"
done
printf 'Once upon ' > "$out/prompt.bin"
for sampling in greedy sample; do
    for strategy in full tiled cache; do
        "$generate" "$config" "$out/full-whole.ckpt" "$out/prompt.bin" \
            "$out/$sampling-$strategy.bin" 64 40 "$sampling" 0.8 2026 blocked "$strategy" \
            > "$out/$sampling-$strategy.log"
    done
    cmp "$out/$sampling-full.bin" "$out/$sampling-cache.bin"
    cmp "$out/$sampling-full.bin" "$out/$sampling-tiled.bin"
done
```

## Put a process boundary between checking and measuring

The remaining section runs three matrix shapes, including a single-row product and incomplete tiles. Its matrix harness is the unchanged Chapter 6 implementation: both backends are checked, warmed, and alternated across samples. A one-row product exercises a different packing/loop balance from a larger batch.

For every inference length/backend/strategy, the driver runs `check` first and `measure` second as separate executables. The check receives repetition count one because it does not time replays. It still compares all nine prediction rows for the prompt plus eight consumed IDs. The measurement uses the requested repetition count. A correctness process is deliberately not wrapped in a memory report.

The training sweep fixes weights and measures forward/backward pairs at lengths 32, 128, and 256. It is separate from the actual AdamW training above. The final tile sweep uses length 65 so 4-, 16-, and 32-wide tiles all encounter boundaries; it holds weights, inputs, backend, and sequence length fixed. This is the chosen deeper-C experiment instead of adding a platform BLAS adapter.

```sh title="scripts/run_demo.sh"
# Matmul shapes include a single row and awkward tails, not only square products.
"$compare" kernel both 1 32 32 "$reps" nt > "$out/kernel-1-32-32.txt"
"$compare" kernel both 32 80 32 "$reps" nt > "$out/kernel-32-80-32.txt"
"$compare" kernel both 17 65 33 "$reps" nn > "$out/kernel-17-65-33.txt"
for length in 16 64 128; do
    for backend in ref blocked; do
        for strategy in full tiled cache; do
            stem=model-$length-$backend-$strategy
            "$compare" model check "$config" "$out/full-whole.ckpt" "$strategy" "$backend" \
                "$length" 8 1 16 32 > "$out/$stem-check.txt"
            # A NEW process: no reference arrays were ever allocated in this run.
            "$compare" model measure "$config" "$out/full-whole.ckpt" "$strategy" "$backend" \
                "$length" 8 "$reps" 16 32 > "$out/$stem-measure.txt"
        done
    done
done
for length in 32 128 256; do
    for strategy in full tiled; do
        stem=training-$length-$strategy
        "$compare" training check "$config" "$strategy" blocked "$length" 1 16 32 \
            > "$out/$stem-check.txt"
        "$compare" training measure "$config" "$strategy" blocked "$length" "$reps" 16 32 \
            > "$out/$stem-measure.txt"
    done
done
# A tile sweep is the optional deeper-C experiment; no Accelerate adapter is required.
for tile in 4 16 32; do
    "$compare" training check "$config" tiled blocked 65 1 "$tile" "$tile" \
        > "$out/tile-$tile-check.txt"
    "$compare" training measure "$config" tiled blocked 65 "$reps" "$tile" "$tile" \
        > "$out/tile-$tile-measure.txt"
done
python3 scripts/summarize_demo.py "$out" > "$out/summary.txt"
printf '%s\n' 'pretrained=not_run_by_this_offline_byte_model_driver' >> "$out/summary.txt"
printf 'Completed byte-model demonstration: %s\n' "$out"
```

## Summarize only values that were produced

Create `scripts/summarize_demo.py`. It reads C logs and bytes; there is no model evaluation in this script. Checkpoint equality is verified again in bounded chunks rather than inferred from filenames or a printed success marker. The loss file's x-coordinate is the cumulative **processed target count**, not the number of updates. This remains meaningful when another config or accumulation count changes the work per update.

Its `open("x")` creates each `*-loss.tsv` exclusively. Choose a fresh directory for a new run. To deliberately regenerate a summary in an existing run, first preserve or explicitly remove only its old generated TSV files; the script will not overwrite them silently. An interrupted summary can leave a partial new TSV, and must not be reported as a complete result.

Escaped Python byte representations make zero bytes, non-UTF-8 values, and trailing spaces visible without pretending a byte-model transcript is always clean Unicode. Completed comparison logs must include their successful end marker and a numerical bound. This is a small reader of ordinary result files, not an alternate pass/fail system for model correctness.

```python title="scripts/summarize_demo.py"
#!/usr/bin/env python3
"""Summarize C logs and escaped bytes; never execute a model or infer a missing pass."""
from __future__ import annotations

from pathlib import Path
import re
import sys


def same_files(a: Path, b: Path) -> bool:
    with a.open("rb") as left, b.open("rb") as right:
        while True:
            x, y = left.read(65536), right.read(65536)
            if x != y:
                return False
            if not x:
                return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 scripts/summarize_demo.py DEMO_DIRECTORY")
    root = Path(sys.argv[1])
    for strategy in ("full", "tiled"):
        if not same_files(root / f"{strategy}-whole.ckpt", root / f"{strategy}-split.ckpt"):
            raise ValueError(f"{strategy}: final checkpoints differ")
        lines = (root / f"{strategy}-whole.log").read_text().splitlines()
        before = next(x for x in lines if x.startswith("validation before"))
        after = [x for x in lines if x.startswith("validation update=")][-1]
        print(strategy, before)
        print(strategy, after)
        seen = 0
        with (root / f"{strategy}-loss.tsv").open("x", encoding="ascii") as output:
            output.write("update\tprocessed_targets\ttrain_loss\tlr\n")
            for line in lines:
                if not line.startswith("update="):
                    continue
                fields = dict(x.split("=", 1) for x in line.split())
                seen += int(fields["targets"])
                output.write(f'{fields["update"]}\t{seen}\t{fields["train_loss"]}\t{fields["lr"]}\n')
        print(f"{strategy} processed_targets={seen} checkpoint_resume=byte_equal")
    for sampling in ("greedy", "sample"):
        paths = [root / f"{sampling}-{mode}.bin" for mode in ("full", "cache", "tiled")]
        payloads = [p.read_bytes() for p in paths]
        if len(set(payloads)) != 1:
            raise ValueError("generation strategies differ: inspect logits and greedy margins")
        print(f"{sampling} transcript={payloads[0]!r}")
    for path in sorted(root.glob("*-measure.txt")):
        print(path.name)
        for line in path.read_text().splitlines():
            if line.startswith(("payload ", "phase=", "peak_rss_", "note=")):
                print("  " + line)
    checks = sorted(root.glob("*-check.txt"))
    if not checks:
        raise ValueError("no comparison files")
    for path in checks:
        lines = path.read_text().splitlines()
        if "check_only=no_timing_or_rss_reference_buffers_are_live" not in lines:
            raise ValueError(f"incomplete check: {path}")
        errors = [float(x) for line in lines for x in re.findall(r"max_abs=([\deE+.-]+)", line)]
        if not errors:
            raise ValueError(f"no error bound: {path}")
    print(f"completed_comparison_processes={len(checks)}")


if __name__ == "__main__":
    main()
```

## Run it locally

After creating the remaining scripts from Part 4, start in the learner implementation root:

```sh
make test
make test-assets test-import-assets test-reference test-measure
mkdir -p runs
sh scripts/run_demo.sh runs/pilot-01
```

The default compiler is `xcrun clang` on the Mac. On the Linux host used for the executed examples, the driver was invoked with `CC=clang`. A two-update, one-repetition smoke run exercises orchestration without claiming useful learning or precise timing:

```sh
DEMO_UPDATES=2 DEMO_REPS=1 sh scripts/run_demo.sh runs/smoke-01
```

A larger repetition count is appropriate when a phase reports short batches, but it increases compute cost for every sweep. To repeat one fast phase with enough work, use its `compare ... measure` command directly rather than rerunning every training trajectory. For example:

```sh
./build/release/compare model check configs/byte-pilot.conf \
  runs/pilot-01/full-whole.ckpt cache blocked 128 8 1 16 32
./build/release/compare model measure configs/byte-pilot.conf \
  runs/pilot-01/full-whole.ckpt cache blocked 128 8 100 16 32
```

All commands use actual saved paths, not an assumed global checkpoint. Keep the selected config with its checkpoint. A generated table without its weights, shape, selected mode, backend, tile sizes, and build settings is difficult to interpret.

## Predict a changed experiment before running it

Double `PROMPT` while keeping eight subsequent inputs. Cache history capacity increases; each decode attends to more keys, but old rows' projections are not recomputed. Full-prefix work grows in both projection rows and causal pairs. Tiled scratch grows approximately linearly, but tiled full-prefix arithmetic still grows with the number of causal pairs.

Next double `REPS` while holding everything else fixed. Batch intervals should roughly double under similar conditions, while seconds per replay and payload bytes should not. An exactly doubled seconds-per-item report indicates a missing divisor; a changed cache result indicates an incorrect reset/position boundary. Large timing scatter without numerical changes suggests scheduling, background load, or insufficiently long intervals rather than a derivative bug.

Part 4 gives the separate real-asset driver, the concise local run guide, actual observations, and their limits.
