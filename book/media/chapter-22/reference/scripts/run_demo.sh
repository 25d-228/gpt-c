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
