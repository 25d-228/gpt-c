# Local runs

Run from the learner source root. All output directories must be new. Keep runs,
assets, weights, traces and binaries on the learner branch/local disk, not on the
book branch. The scripts do not download assets, retry commands, or delete failed
runs. A failed process leaves its log for diagnosis; it is not a completed result.

```sh
make test
make test-assets test-import-assets test-reference test-measure
mkdir -p runs
sh scripts/run_demo.sh runs/pilot-01
```

The compiler defaults to `xcrun clang` on the Mac. An equivalent compiler can be
selected with `CC=clang`. The demo defaults to the 29,856-parameter pilot, 160
updates, one microbatch/update and five repetitions per timing sample. It records
actual build commands, machine information, full/tiled losses by processed targets,
split/uninterrupted checkpoint equality, escaped continuations, all-logit/gradient
checks, and separate-process timings and peak resident memory.

For a fast smoke run use `DEMO_UPDATES=2 DEMO_REPS=1`. For the 2,885,888-parameter
configuration use `DEMO_CONFIG=configs/byte-small.conf`; start with
`DEMO_UPDATES=8 DEMO_REPS=1` and inspect actual CPU cost before scaling. The full demo
performs duplicate training for restart checks and many prefix/gradient comparisons;
it is more work than one training run. Reading and unattended computation are not
implementation hours. Repetition counts are positive decimals without leading zeros.

`model ... check` and `training ... check` intentionally do not report timing/RSS:
reference arrays coexist. The following `measure` invocation starts another process
and constructs only the selected implementation. Keep capacities, IDs, backends,
tiles, build flags and repetitions aligned between each check/measurement pair.

```sh
./build/release/compare model check configs/byte-pilot.conf \
  runs/pilot-01/full-whole.ckpt cache blocked 128 8 1 16 32
./build/release/compare model measure configs/byte-pilot.conf \
  runs/pilot-01/full-whole.ckpt cache blocked 128 8 100 16 32
./build/release/compare training check configs/byte-pilot.conf tiled blocked 256 1 16 32
./build/release/compare training measure configs/byte-pilot.conf tiled blocked 256 5 16 32
```

Prefill seconds cover one prompt. Decode seconds/item are an average over consumed
contexts `PROMPT+1` through `PROMPT+STEPS`, not a percentile or a fixed-context
latency. Timed regions exclude loading, allocation, sampling and text I/O.
Training measurements are fixed-weight forward/backward pairs, not AdamW updates.
Do not rank tiny timing differences when min/max intervals overlap or batches are
flagged short. Rerun without competing builds and vary order between whole runs.

Array payload and peak RSS answer different questions. The code prints native RSS
units (Darwin bytes, Linux KiB) and converted bytes; confirm the local `getrusage(2)`
manual on other systems. RSS includes startup/loading, stack, allocator and runtime,
and can retain earlier peaks. It is not a sum of the reported live array payloads.

For already prepared genuine SmolLM2 assets, activate the optional Python reference
and tokenizer environment and run:

```sh
sh scripts/run_pretrained_demo.sh assets/smollm2-source assets/smollm2-f32 \
  assets/smollm2-tokenizer runs/pretrained-01
```

This requires Chapter 20/21's pinned asset preparation; fixture metadata is rejected.
It compares C traces to the independent CPU FP32 equations, selects IDs in C, decodes
only afterward, compares identical generated prefixes, and measures three strategies
in separate processes. The supplied tolerances are diagnostic targets, not guaranteed
bounds. Provenance declarations do not authenticate arbitrary local file contents.

`summary.txt`, `*-loss.tsv`, small `*-check.txt` and `*-measure.txt` files are enough
for the compact result table. Checkpoints and detailed traces remain local. To rerun,
choose another output directory rather than silently overwriting an earlier result.
