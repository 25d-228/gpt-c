# Chapter 22, Part 4 — Interpret the measurements and complete the demonstration

The byte-model demonstration in [Part 3](https://25d-228.github.io/gpt-c/chapter-22-part-3.html) runs without a remote service. The pretrained demonstration has a different boundary: it requires actual pinned model and tokenizer assets. Keep those two results distinct. Synthetic files test representation and equations; they cannot establish that an external model's complete inventory or tokenizer matches our interpretation.

## Run the already prepared genuine assets

Use the local source, converted, and tokenizer directories produced by [Chapter 20](https://25d-228.github.io/gpt-c/chapter-20.html) and [Chapter 21](https://25d-228.github.io/gpt-c/chapter-21.html). Do not substitute fixture files for a missing download. Activate the compatible Python reference/tokenizer environment described there. The script below performs no download and makes no network retry.

The three input directories have different jobs. `SOURCE` contains original `config.json` and `model.safetensors` for the independent CPU FP32 equations. `CONVERTED` contains `model.cfg`, `weights.dlw`, and the converter's `source.json` for C. `TOKENIZER` contains the explicitly acquired tokenizer and its own provenance record. Matching declared revisions detects an accidental fixture or mixed acquisition; it does not authenticate arbitrary local bytes.

The driver first round-trips an ordinary UTF-8 prompt through the actual tokenizer with the earlier no-automatic-special-token policy. It then compares all intermediate values for full and tiled C execution. C selects four greedy continuation IDs, stopping on newly selected EOS ID zero. Python only decodes those selected IDs. Finally, all three C strategies and the independent equations replay the same generated prefixes. The reference is the explicit Chapter 21 eager implementation, not an invocation of an official transformer model class.

The [selected model configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json) declares vocabulary size 49,152, 30 blocks, width 576, nine query heads, three key/value heads, feed-forward width 1,536, and EOS ID zero. These are the converter's supported geometry, not values inferred from a model's name. Keep the pinned revision from the previous chapters; do not silently replace it with a moving default branch.

Create `scripts/run_pretrained_demo.sh`:

```sh title="scripts/run_pretrained_demo.sh"
#!/bin/sh
# Existing pinned assets only; model execution and token selection stay in C.
set -eu
if [ "$#" -ne 4 ]; then
    echo 'usage: sh scripts/run_pretrained_demo.sh SOURCE CONVERTED TOKENIZER NEW_OUTPUT' >&2
    exit 1
fi
root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
source=$(CDPATH= cd "$1" && pwd)
converted=$(CDPATH= cd "$2" && pwd)
tokenizer=$(CDPATH= cd "$3" && pwd)
case "$4" in /*) out=$4 ;; *) out=$PWD/$4 ;; esac
mkdir "$out"
cd "$root"
python3 - "$converted" "$tokenizer" > "$out/assets.txt" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from export_smollm2 import MODEL, REVISION
weights = json.loads((Path(sys.argv[1])/'source.json').read_text())
tokens = json.loads((Path(sys.argv[2])/'source.json').read_text())
if (weights.get('model_profile') != MODEL or weights.get('expected_revision') != REVISION
        or tokens.get('model') != MODEL or tokens.get('revision') != REVISION):
    raise SystemExit('asset provenance must name the selected model/revision, not a fixture')
print(json.dumps(dict(weights=weights, tokenizer=tokens), indent=2))
print('Provenance records are declarations, not authentication of the file contents.')
PY
make -B CC="${CC:-xcrun clang}" build/release/compare > "$out/build.log" 2>&1
{
    date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
    uname -a
    python3 -c 'import sys, torch, safetensors, transformers, tokenizers; print(sys.version); print("torch",torch.__version__,"safetensors",safetensors.__version__,"transformers",transformers.__version__,"tokenizers",tokenizers.__version__)'
} > "$out/environment.txt"
printf 'Once upon a time,' > "$out/prompt.utf8"
python3 scripts/tokenize.py encode "$tokenizer" "$out/prompt.utf8" "$out/prompt.ids" \
    > "$out/tokenization.json"
python3 scripts/tokenize.py decode "$tokenizer" "$out/prompt.ids" "$out/roundtrip.utf8"
cmp "$out/prompt.utf8" "$out/roundtrip.utf8"
P=$(python3 -c 'import sys; sys.path.insert(0,"scripts"); from probe_io import read_ids; p=len(read_ids(sys.argv[1])[1]); assert 0<p<=16, "trace prompt must contain 1..16 IDs"; print(p)' "$out/prompt.ids")
compare=./build/release/compare
python3 scripts/reference_smollm2.py "$source" "$out/prompt.ids" "$out/reference.jsonl"
for mode in full tiled; do
    "$compare" pretrained trace "$converted/model.cfg" "$converted/weights.dlw" \
        "$out/prompt.ids" "$out/c-$mode.jsonl" "$mode" blocked
    python3 scripts/compare_traces.py "$out/c-$mode.jsonl" "$out/reference.jsonl" \
        --atol 1e-4 --rtol 2e-4 > "$out/trace-$mode.txt"
done
"$compare" pretrained generate "$converted/model.cfg" "$converted/weights.dlw" \
    "$out/prompt.ids" "$out/transcript.ids" cache blocked 4 0
python3 scripts/tokenize.py decode "$tokenizer" "$out/transcript.ids" "$out/transcript.utf8"
python3 scripts/reference_smollm2.py "$source" "$out/transcript.ids" \
    "$out/reference-prefix.jsonl" --mode prefix --start "$P"
for mode in full tiled cache; do
    "$compare" pretrained prefix "$converted/model.cfg" "$converted/weights.dlw" \
        "$out/transcript.ids" "$out/prefix-$mode.jsonl" "$mode" blocked "$P"
    python3 scripts/compare_traces.py "$out/prefix-$mode.jsonl" "$out/reference-prefix.jsonl" \
        --atol 1e-4 --rtol 2e-4 > "$out/prefix-$mode.txt"
    # Deterministic synthetic IDs for timing, not a claim about generated text.
    "$compare" model check "$converted/model.cfg" "$converted/weights.dlw" \
        "$mode" blocked 4 2 1 16 32 > "$out/measure-$mode-check.txt"
    "$compare" model measure "$converted/model.cfg" "$converted/weights.dlw" \
        "$mode" blocked 4 2 1 16 32 > "$out/measure-$mode.txt"
done
printf '%s\n' 'All requested real-asset commands exited successfully; inspect errors, margins and output.' \
    > "$out/completed.txt"
```

The first failed command stops the script. A new output directory and its partial logs remain for diagnosis; only the last successful command creates `completed.txt`. A missing dependency, failed trace comparison, or real-data incompatibility is not permission to enlarge tolerances until it passes. Read the first divergent stage, check tokenizer IDs and configuration, then inspect the corresponding C coordinates.

A fixed `1e-4 + 2e-4|reference|` bound is a diagnostic target, not a proven upper bound for this complete real model. Maximum error, argmax agreement, and the gap between the two largest logits answer different questions. Near ties can change the selected ID inside the numerical bound. Matching tokens alone can hide a wrong logit vector. The fixed-prefix comparison deliberately does not let those different choices change its input.

The last measurements use deterministic IDs with four prompt positions and two subsequent consumed inputs. They are small computational workloads, not generated prose. Every check and measurement is a separate process, and Python reference execution has finished before the corresponding C timing starts. This limits co-resident storage and avoids assigning a Python process's peak to the C engine.

**Executed status for this course:** real pretrained weights and tokenizer files could not be acquired in the available runtime because DNS resolution failed; Transformers/tokenizers were not installed. This driver was syntax-checked, and its rejection of fixture provenance was executed. Its genuine tokenization, full-model reference agreement, continuation, and timing commands were **not executed**. Chapters 20–21's synthetic import and independent-equation checks did execute. Nothing in the tables below is a genuine-pretrained result.

## Keep a concise run guide beside the implementation

Create `runs/README.md`. This is documentation, not a generated result directory. The shell drivers create each result in a fresh child directory. Model weights and result files stay on the learner's local disk/implementation branch, rather than becoming book assets.

````markdown title="runs/README.md"
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
````

## Observed byte-model learning and continuation

The following runs executed the C engine on **x86_64 Linux, Clang 17**, not the M4 Mac. The host identified an Intel Xeon Platinum 8573C; the container reported a 4 GiB memory limit and a four-CPU scheduling quota. The C kernels are single-threaded. These settings do not imply exclusive access to the host or fixed processor frequency. The optimized flags were `-O2 -fno-lto -fno-fast-math -ffp-contract=off`; the blocked backend retained its double reductions and float stores.

The full 160-update pilot used the original diagnostic text produced by `make_demo_text.py`: a training file and a separate validation file. The data is intentionally tiny and repeatedly sampled. This is a test of learning mechanics, not a representative held-out language benchmark. Both strategies used the same seed, schedule, batch of two sequences of length 16, and **one** microbatch per update. The processed target count is $160\times2\times16=5{,}120$, including repeated targets.

| Configuration and attention | Updates | Processed targets | Validation targets scored | Initial loss | Final loss |
|---|---:|---:|---:|---:|---:|
| 29,856 parameters, full | 160 | 5,120 | 128 | 5.725389095 | 2.398719681 |
| 29,856 parameters, tiled | 160 | 5,120 | 128 | 5.725389094 | 2.398719537 |
| 2,885,888 parameters, tiled | 8 | 2,048 | 256 | 5.474827494 | 3.021004048 |

The larger configuration used two sequences of length 128 and one microbatch per update. It was run separately with the existing application, not inferred by scaling the pilot's loss. The two validation traversals leave 264 and 136 trailing bytes unscored respectively, as reported by the bounded iterator. They therefore do not evaluate identical sets of target positions; do not rank their final loss as a controlled model-size comparison.

Earlier chapters used other microbatch schedules, including two microbatches per update. Equal update counts under those settings do not mean equal training exposure. The target-count TSV from Part 3 prevents that mismatch from being hidden by an update-only horizontal axis. None of these runs establishes broad language quality.

For each pilot attention strategy, a split run resumed from its saved checkpoint and produced **byte-identical final checkpoint contents** to the uninterrupted run. Equality is required within a strategy, not between full and tiled checkpoint formats. The three generation strategies then used the same full-trained checkpoint. Greedy output and sampled output each matched byte-for-byte across full-prefix, tiled, and cached execution:

```text
Greedy: b'Once upon the t the t the the the the the the the '
Sample: b'Once upon the thed blt che d rnd aindrrnd ind thit'
```

Each transcript contains ten prompt bytes and forty selected bytes. The cached application consumes 49 inputs, because the last selected byte is returned without another forward pass. The small-model eight-byte greedy continuation was `b'Once upon         '`: it also matched across the three strategies, and is plainly not useful prose. A falling loss and correct state transitions do not make a model linguistically capable.

The long pilot command sequence completed its training, equality, generation, gradient, and timing steps; an external command time limit interrupted its final measurement sweep, whose remaining commands were then executed directly. A fresh two-update smoke invocation subsequently completed the **entire unmodified driver** successfully. The selected timing tables below were rerun separately after compilation and correctness tests had stopped. This separates successful orchestration from a claim that one uninterrupted long process produced every observation.

To repeat the larger eight-update run without duplicating the entire demonstration sweep, use a newly named checkpoint and the prepared diagnostic files:

```sh
./build/release/train new configs/byte-small.conf \
  runs/pilot-01/text/train.bin runs/pilot-01/text/valid.bin \
  runs/small-eight.ckpt 8 8 1 random blocked tiled
./build/release/generate configs/byte-small.conf runs/small-eight.ckpt \
  runs/pilot-01/prompt.bin runs/small-eight.bin 32 8 greedy 0.8 2026 blocked cache
```

The training positional arguments are the total schedule length, stopping update, and microbatch count. The generation argument order is the previously implemented `CAPACITY NEW_TOKENS MODE TEMPERATURE SEED BACKEND STRATEGY`; greedy mode records but does not consume the sampling RNG. Use the application's usage message rather than swapping capacity and output-count arguments.

## Matrix blocking: report the complete call

The Chapter 6 kernel harness includes packing, dispatch, validation, arithmetic, and stores. All three shapes below passed its numerical comparison with maximum absolute error zero in this run. The operation count is a conventional dense multiply/add count, not an instrumented CPU instruction count. Each cell is **minimum / median / maximum microseconds per complete call** across five batches.

| Logical shape $(M,N,K)$ / orientation | Repetitions per batch | Scalar reference, microseconds | Blocked C, microseconds |
|---|---:|---:|---:|
| (1, 32, 32) / `nt` | 100,000 | 0.903 / 1.079 / 1.513 | 0.770 / 0.833 / 0.945 |
| (32, 80, 32) / `nt` | 1,000 | 78.922 / 88.182 / 121.707 | 36.496 / 41.759 / 48.620 |
| (17, 65, 33) / `nn` | 2,000 | 44.063 / 45.649 / 46.354 | 12.036 / 15.859 / 25.986 |

The one-row result does not predict the many-row result. Packing cost, reuse, and tail handling interact with shape. The incomplete $17\times65\times33$ case exercises both output and reduction tails. No Accelerate adapter is introduced here: the final 0.75-hour extension is the C shape/tile sweep. A future library adapter must preserve orientation, leading dimensions, SET/ADD behavior, and tests before its time can be compared.

## Prefill and subsequent inputs are different phases

These measurements use the trained pilot, the blocked matrix backend, 128 prompt IDs, eight further consumed IDs, capacity 136, and query/key tiles $16\times32$ in tiled mode. Each of five samples contains **30 complete transcript replays**, after an untimed warmup. The correctness processes compared 2,304 logit values per strategy against scalar full-prefix execution: full and cached maximum error was zero; tiled maximum error was $5.96046448\times10^{-7}$.

| Strategy | Prefill milliseconds, min / median / max | Eight-input decode group milliseconds, min / median / max | Median milliseconds per subsequent input |
|---|---:|---:|---:|
| full | 4.247 / 4.398 / 5.051 | 36.616 / 40.125 / 47.386 | 5.016 |
| tiled | 3.349 / 3.677 / 4.326 | 29.460 / 30.653 / 35.792 | 3.832 |
| cache | 5.948 / 6.172 / 6.372 | 0.858 / 1.100 / 1.187 | 0.138 |

The cache improves the subsequent-input group here, but has a slower measured prefill. That is a useful counterexample to treating “cached” as one universal speed multiplier. It avoids old-row projections after prefill; its prefill still uses the implementation's own attention traversal and validation. Tiled full-prefix inference saves workspace but continues to recompute old rows. These are observations for the specified code, sizes, flags, and host—not M4 latency estimates.

The decode rate averages contexts 129 through 136. It is neither a percentile of eight individual token latencies nor a fixed-context benchmark. The sampled times exclude checkpoint loading, allocation, resetting cache length, token selection, and I/O. Separate-process peak RSS includes startup/loading as well as computation:

| Strategy | Weight bytes | Scratch bytes | History bytes | Other counted heap bytes | Total counted heap bytes | Peak resident bytes |
|---|---:|---:|---:|---:|---:|---:|
| full | 119,424 | 870,400 | 0 | 6,656 | 996,480 | 2,097,152 |
| tiled | 119,424 | 296,960 | 0 | 6,656 | 423,040 | 1,572,864 |
| cache | 119,424 | 262,208 | 34,816 | 6,656 | 423,104 | 1,499,136 |

The weight arena is identical in every row. Cache history is not zero just because the inference scratch shrinks. Tiled and cached heap totals are nearly equal at this particular capacity for different structural reasons. Their small RSS difference is not evidence that the smaller-looking one always has the better memory design: RSS includes effects not represented by these live arrays and is only one process observation.

## Tiled backward: a memory change can cost time

The next sweep measures a complete fixed-weight forward/backward pair, not an optimizer update. Batch size is two; the pilot geometry is unchanged; tiled mode uses $16\times32$ tiles; both modes use blocked matrices. Five batches of **ten** pairs follow the warmup. The check action compares all 29,856 unique gradient values before each corresponding measurement; no reference tape exists in the measured process.

| Sequence length | Strategy | Pair milliseconds, min / median / max | Tape bytes | Total counted heap bytes | Peak resident bytes |
|---:|---|---:|---:|---:|---:|
| 32 | full | 3.988 / 4.145 / 4.506 | 770,048 | 1,018,128 | 1,998,848 |
| 32 | tiled | 3.798 / 4.081 / 5.074 | 609,920 | 858,096 | 1,966,080 |
| 128 | full | 25.856 / 26.010 / 30.952 | 5,439,488 | 5,695,248 | 6,815,744 |
| 128 | tiled | 24.838 / 25.613 / 25.911 | 2,367,104 | 2,622,960 | 3,801,088 |
| 256 | full | 66.925 / 68.524 / 73.932 | 17,170,432 | 17,436,432 | 18,612,224 |
| 256 | tiled | 71.346 / 76.109 / 90.138 | 4,710,016 | 4,976,112 | 6,078,464 |

At length 256, tiled tape storage is about 3.65 times smaller, while the median pair time is **higher** (76.109 rather than 68.524 milliseconds). Recomputed scores and probabilities, additional passes, and different traversal all do work. The result demonstrates the intended memory–recomputation tradeoff without asserting that recomputation is always slower or always faster. At the smaller lengths, timing intervals overlap substantially; a narrow median difference is not a stable ranking.

Parameter/gradient storage stays at 238,848 bytes in every row. Tape storage changes with $T$. Process resident memory also includes stack, allocator/runtime pages, and earlier model-construction peaks, so neither a constant overhead nor a constant ratio should be assumed. The materialized-minus-tiled storage difference is predictable from the square attention terms derived in Part 2; the runtime difference must be measured.

A tail-heavy tile sweep uses sequence length 65, batch two, and ten repetitions per sample. All three choices pass the gradient comparison. None divides the sequence length exactly:

| Query/key tiles | Pair milliseconds, min / median / max | Tape bytes |
|---|---:|---:|
| 4 × 4 | 9.267 / 9.860 / 11.289 | 1,191,712 |
| 16 × 16 | 10.469 / 13.435 / 20.967 | 1,203,712 |
| 32 × 32 | 9.773 / 10.079 / 10.332 | 1,234,048 |

The 16-by-16 run has conspicuous spread, including a much slower maximum. There was no concurrent course compilation during these selected reruns, but that does not eliminate shared-host scheduling or background activity. Repeat the whole sweep in a different order before treating tile size as the cause. The difference between 4-by-4 and 32-by-32 medians is too small relative to the observed ranges to justify an “optimal tile” claim.

## Run realistically on the 16 GB M4 target

Start with debug/sanitizer correctness, then perform timings with the release executable. Run the pilot before the larger byte model. Record the actual `sw_vers`, compiler version, and hardware supplied by the driver; do not copy the Linux environment table into a Mac result. Close unrelated compute-heavy workloads, keep power/thermal conditions reasonably consistent, and repeat a changed experiment without simultaneous compilation. A monotonic clock's advertised resolution does not remove scheduling variation.

Use the four-byte weight counts to budget ownership, not to predict resident memory. The selected pretrained geometry has 134,515,008 unique values, or 538,060,032 FP32 weight bytes. A 1,024-position cache needs $4\times2\times30\times1024\times192=47{,}185{,}920$ history bytes, in addition to workspace, result rows, metadata, and runtime. These are calculations from the [pinned configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json) and the implemented layouts, not measured M4 process memory. Avoid loading the Python reference and the C model simultaneously merely to time C; the drivers intentionally execute them sequentially.

Memory fitting does not imply acceptable CPU time. The genuine driver deliberately begins with a short prompt and four selected IDs. The full tiled backward path is educational C with recomputation and explicit checks, not an Apple GPU/Neural Engine kernel. No macOS version, Accelerate backend, GPU speedup, or measured M4 throughput is assumed. Increase data exposure, context, and repetitions separately, using actual elapsed time and resident-memory observations to choose the next experiment.

## Diagnose the demonstration, not only the model

Three deliberate harness mistakes were executed. Replacing `ids[count-1]` with `ids[0]` in cached decode failed the all-logit check (the first observed reference/actual pair was approximately −0.588776 and −0.195684). Removing the reset failed the next prefill's cache-state contract. Omitting division by item count failed the timing-arithmetic test even though the model calculations were still correct. Sanitizers reported no memory error in these three cases: arithmetic and state tests were the detecting tools.

The correct implementation passed all 21 cumulative C test programs in debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized builds on the stated Linux host. The 12 new measurement integration tests passed against each build, including a real `/dev/full` output failure. The inherited independent-reference, import, preparation, and training/generation checks also passed. These results establish the exercised paths, not untested genuine assets or M4 behavior.

A rerun should end with four concrete artifacts: an interpretable loss-versus-processed-targets trace, an actual continuation, all-coordinate comparisons for the selected implementations, and a small timing/memory table with settings. A command that failed or never ran must remain labeled that way. A checkpoint restart that silently changes data RNG, schedule, or accumulation is not a continuation; a cache comparison on different prefixes is not an equivalence check; an RSS sample taken after allocating both alternatives does not measure either alternative in isolation.

## What the completed engine exposes

A byte or pretrained vocabulary ID indexes one explicitly owned embedding table. The model traverses pre-normalized residual blocks with grouped-query attention, rotary coordinates, and a two-branch SwiGLU transformation. Stable loss produces the reverse signal; manual backward routines accumulate shared heads, residual paths, and the tied output/embedding contributions. The optimizer consumes a correctly weighted and clipped aggregate, while checkpoints preserve continuation state rather than only parameter values.

The same weights then support full-prefix inference, reusable workspaces, categorical selection, and a cache whose committed length means all blocks have consumed the same prefix. Tiled attention replaces quadratic saved intermediates with bounded tiles and recomputed probabilities while preserving the modeled function within explicit numerical tolerances. Each of those mechanisms is inspectable as loops, memory owners, invariants, and executable checks.

That is the completed demonstration: a C implementation that can learn a bounded text problem, resume coherently, select subsequent IDs, and expose where its computation and storage go. The trained byte-model results above are observed. The genuine pretrained commands are implemented but remain an explicit local validation task, not a fabricated result. The mathematical model, its engineering state, and the evidence for each claim are kept separate.

## Primary references used in this chapter

The [Linux `clock_gettime` manual](https://man7.org/linux/man-pages/man3/clock_gettime.3.html) documents monotonic clock behavior and resolution queries. The [Linux `getrusage` manual](https://man7.org/linux/man-pages/man2/getrusage.2.html) and [macOS `getrusage` manual reproduction](https://man.freebsd.org/cgi/man.cgi?query=getrusage&sektion=2&manpath=macOS+10.13.6) specify the process statistic and platform units. The [FlashAttention paper, Section 3.1](https://arxiv.org/html/2205.14135v2#S3.SS1), explains the bounded-intermediate/recomputation strategy; its GPU measurements are not used as predictions here. The selected model's [pinned configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json) supplies the external geometry and token conventions. The numerical result tables in this chapter are direct executions of the supplied C programs, not values taken from those sources.
