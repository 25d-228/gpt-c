# Chapter 6 — Timing complete kernel calls

**Part 2 of Chapter 6.** Start from the implementation and tests in [Part 1 — Blocked C kernels and meaningful timing](https://25d-228.github.io/gpt-c/chapter-06.html). Keep the Chapter 5 modules and the Part 1 header/kernel changes. This part creates `app/compare.c` and replaces the root Makefile. Concatenate the titled `app/compare.c` code blocks in order; all helpers and the complete `main` function are supplied.

The two parts share Chapter 6's **2.00-hour implementation estimate**. This is the timing and experiment portion, not an additional chapter. The full fifteen-file companion state is under `media/chapter-06/reference/`; Part 1 links its new kernel and tests, and this part supplies the [comparison source](https://25d-228.github.io/gpt-c/media/chapter-06/reference/app/compare.c) and [Makefile](https://25d-228.github.io/gpt-c/media/chapter-06/reference/Makefile).

> **Learning objectives**  Define the measured region; use a monotonic elapsed-time interface; keep setup and validation scans outside timing while charging per-call work; report shape, precision, repetitions, error, and variability; and distinguish a kernel observation from a model-throughput claim.

## Decide what the clock is measuring

We will measure elapsed time for repeated **complete, checked, synchronous kernel calls** on already allocated inputs. Allocation, deterministic input filling, the reference result, comparison scans, sorting, and printing stay outside the measured interval. Dispatch, geometry/alias validation, scratch initialization, panel packing, arithmetic, output stores, numeric checks, repetition-loop control, and the per-call status branch remain inside it. The two clock reads delimit the batch; their overhead is not subtracted.

This is neither a bare arithmetic loop nor an end-to-end model benchmark. Data is deliberately reused and each batch has an untimed warm-up, so describe the result as a warmed, repeated-input measurement. It does not establish cold-cache behavior, token throughput, or process memory usage. All computation is single-threaded C.

The application uses `clock_gettime(CLOCK_MONOTONIC, ...)` and `clock_getres`, with `struct timespec` from `<time.h>`. A timestamp has a seconds field `tv_sec` and a nanoseconds field `tv_nsec`. We need a difference, not a calendar date. The [OpenBSD system manual](https://man.openbsd.org/clock_gettime.2) documents these POSIX interfaces and distinguishes monotonic elapsed time from process CPU time. We use the portable interface, not its BSD-only helper functions or extra clock IDs.

On the native Mac, build with the installed SDK and inspect `man 3 clock_gettime` for its platform details. The actual requirements are these clock functions and a compatible C17 compiler, not an invented macOS version or compiler LTS label. The [Gnulib portability notes](https://www.gnu.org/software/gnulib/manual/html_node/clock_005fgettime.html) list older platforms lacking the interface; ISO C17 alone does not promise this POSIX timer. There is no Mach tick conversion, x86 cycle counter, or Linux-only clock in the code. A compiler or runtime failure is reported rather than replaced silently by a different measurement.

For start $(s_0,n_0)$ and end $(s_1,n_1)$, calculate seconds as:

$$
\Delta t=(s_1-s_0)+10^{-9}(n_1-n_0).
$$

Subtract the integer fields **before** converting them to double. For `(12,900000000)` to `(13,100000000)`, the result is $1-0.8=0.2$ seconds. A negative nanosecond difference is normal; it does not mean the clock ran backwards. This avoids subtracting two large, already-rounded floating-point absolute timestamps. The field differences over these short runs fit the target platform's types.

Reported clock resolution is not call overhead, measurement accuracy, or protection from scheduling delays. Elapsed time includes interruption by other work. Keep the machine awake, avoid unrelated heavy jobs, and repeat questionable observations. Five samples help reveal variability; they do not create a statistical guarantee.

## Give the comparison application a narrow interface

Create `app/compare.c`. Its command line is:

```text
compare kernel ref|blocked|both M N K repetitions nn|nt|tn|tt
```

`M`, `N`, and `K` always describe the logical matrix product. Orientation controls the stored shapes. The `kernel` subcommand leaves room for later comparisons without requiring any unimplemented model mode now. `ref` or `blocked` selects one measured backend; the reference result is still computed once outside timing for validation. `both` alternates the first backend across the five samples to reduce a fixed ordering bias.

All benchmarked calls use SET, so repetitions do not accumulate a growing result. ADD is already exercised in the correctness tests. Timing ADD would require deciding whether it represents deliberate repeated accumulation or repeated independent calls with restoration; quietly adding a reset into one backend's timing would compare different workloads.

The first source block declares the timer feature request before any headers. `_POSIX_C_SOURCE` is a feature-test macro used by headers that gate POSIX declarations; it does not change the C language standard. The `Bench` struct merely borrows matrices for the synchronous sample call. Its owner objects stay in `main` until cleanup.

```c title="app/compare.c"
#define _POSIX_C_SOURCE 200809L
#include "dl/kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SAMPLES = 5 };

typedef struct {
    DlConstMatrix a, b;
    DlMatrix out;
    DlTranspose ta, tb;
    size_t repetitions;
} Bench;

```

The argument parser accepts only nonempty decimal digit strings with a positive result. Before `value*10 + digit`, it checks `value <= (SIZE_MAX-digit)/10`, so an overflowing request is rejected rather than wrapping into a small allocation. A leading sign, whitespace, or trailing junk is not accepted.

The four matrix owners contain $MK+KN+2MN$ floats: two inputs, an immutable reference output, and a measured output. `payload_bytes` uses the Chapter 2 checked arithmetic helpers all the way to bytes. The application caps that matrix payload at 256 MiB; this is a practical bound for this command, not a new limit in the general kernel. It also caps repetitions per batch at one million. These bounds prevent accidental oversized buffers, not arbitrarily long CPU runs: begin with small shapes and repetitions.

```c title="app/compare.c"
static int positive_size(const char *text, size_t *out)
{
    if (*text == '\0') return 0;
    size_t value = 0;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return 0;
        const size_t digit = (size_t)(*p - '0');
        if (value > (SIZE_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    if (value == 0) return 0;
    *out = value;
    return 1;
}

static int payload_bytes(size_t m, size_t n, size_t k, size_t *out)
{
    size_t mk = 0, kn = 0, mn = 0, total = 0;
    DlStatus s = dl_size_mul(m, k, &mk);
    if (s == DL_OK) s = dl_size_mul(k, n, &kn);
    if (s == DL_OK) s = dl_size_mul(m, n, &mn);
    if (s == DL_OK) s = dl_size_add(mk, kn, &total);
    if (s == DL_OK) s = dl_size_add(total, mn, &total);
    if (s == DL_OK) s = dl_size_add(total, mn, &total);
    if (s == DL_OK) s = dl_size_mul(total, sizeof(float), out);
    return s == DL_OK;
}

static void fill_input(DlTensor *owner, unsigned seed)
{
    for (size_t i = 0; i < owner->count; ++i) {
        const int q = (int)((i % 101 * 37 + seed) % 101) - 50;
        owner->data[i] = (float)q / 53.0f;
    }
}

static int compare_output(DlConstMatrix expected, DlMatrix actual,
                           double *maximum_error, double *checksum)
{
    double largest = 0.0, total = 0.0;
    for (size_t i = 0; i < actual.rows; ++i) {
        for (size_t j = 0; j < actual.cols; ++j) {
            const double r = (double)expected.base[expected.offset + i*expected.ld + j];
            const double x = (double)actual.base[actual.offset + i*actual.ld + j];
            const double error = fabs(x - r);
            if (!isfinite(r) || !isfinite(x) || error > 1e-6 + 2e-6*fabs(r)) {
                fprintf(stderr, "comparison failed at [%zu,%zu]: %.9g vs %.9g\n",
                        i, j, x, r);
                return 0;
            }
            if (error > largest) largest = error;
            total += x;
        }
    }
    *maximum_error = largest;
    *checksum = total;
    return 1;
}

```

`fill_input` initializes the entire dense inputs before any read. The deterministic pattern varies with the physical offset; changing orientation therefore changes which values appear at logical coordinates. Backend comparisons within a run always share the same arrays. Timing different orientations probes layout and shape, not an algebraic claim that their scalar outputs must coincide.

`compare_output` visits every cell and rejects non-finite values before accepting the error threshold. Its checksum is a simple signed diagnostic sum, **not** a correctness substitute, cryptographic digest, or artifact-validation procedure. Different errors can cancel in a sum; elementwise checks are authoritative. The scan happens after each batch and before any throughput summary is printed.

### Warm up, measure, and reject failed samples

`run_sample` first calls the selected backend once outside timing. It then places only the repetition loop between the two timestamp calls. A kernel error stops the loop, the ending timestamp is still obtained, and the sample is discarded. A failing clock call uses `perror` to show the system error and also discards the sample. No partial batch is divided by the originally requested repetition count.

```c title="app/compare.c"
static int run_sample(const Bench *job, DlBackend backend, double *seconds)
{
    /* One untimed warm-up per sample, including every output store. */
    DlStatus s = dl_matmul(backend, job->a, job->ta, job->b, job->tb,
                           job->out, DL_SET);
    if (s != DL_OK) {
        fprintf(stderr, "warm-up failed: %d\n", (int)s);
        return 0;
    }
    struct timespec begin = {0}, end = {0};
    if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) {
        perror("clock_gettime begin");
        return 0;
    }
    for (size_t r = 0; r < job->repetitions; ++r) {
        s = dl_matmul(backend, job->a, job->ta, job->b, job->tb,
                      job->out, DL_SET);
        if (s != DL_OK) break;
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        perror("clock_gettime end");
        return 0;
    }
    if (s != DL_OK) {
        fprintf(stderr, "timed product failed: %d; sample discarded\n", (int)s);
        return 0;
    }
    /* Subtract integer fields before conversion; nanoseconds may be negative. */
    const double elapsed = (double)(end.tv_sec - begin.tv_sec)
                         + 1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
    if (!isfinite(elapsed) || elapsed <= 0.0) {
        fputs("nonpositive clock interval; increase repetitions\n", stderr);
        return 0;
    }
    *seconds = elapsed;
    return 1;
}

```

The output is read after the timed region, and the build uses separate C translation units without link-time optimization. The compiler of `compare.c` cannot replace an externally defined, status-returning kernel with a guessed constant result or erase its repeated side effects. Do not add `volatile` to every matrix element: that would change the memory operations being measured. The optimized build explicitly retains `-fno-lto`; an aggressively changed whole-program benchmark would need its optimization boundaries reconsidered.

### Summarize samples without hiding variability

For $R_b$ repetitions in a batch, define nominal arithmetic work $F=2MNK$ per call: one multiply and one accumulator add for each reduction term. This convention counts arithmetic, not actual CPU instructions, and ignores packing, validation, indexing, and conversion operations. Both kernels use double reductions over float storage; do not label the number pure FP32 hardware throughput.

$$
\text{seconds per call}=\frac{\Delta t_{\mathrm{median}}}{R_b},\qquad
\text{equivalent GFLOP/s}
=\frac{2MNK R_b}{10^9\Delta t_{\mathrm{median}}}.
$$

For the $[9,33,65]$ test, nominal work is $2\cdot9\cdot33\cdot65=38610$. If a hypothetical batch of 100 calls took 0.02 seconds, this convention would report 0.19305 equivalent GFLOP/s. Those are illustrative numbers, not a measured result.

`report` copies five elapsed values and sorts that copy using insertion sort. The middle entry is the median; it also prints the smallest and largest values. The original samples remain unchanged. A batch shorter than 0.02 seconds gets a prompt to increase repetitions. That threshold is a practical warning, not a timer-accuracy theorem and not an automatic request to make a small operation large.

```c title="app/compare.c"
static void report(const char *name, const double samples[SAMPLES],
                    size_t repetitions, double operations,
                    double maximum_error, double checksum)
{
    double sorted[SAMPLES];
    for (size_t i = 0; i < SAMPLES; ++i) sorted[i] = samples[i];
    for (size_t i = 1; i < SAMPLES; ++i) {
        const double key = sorted[i];
        size_t j = i;
        while (j > 0 && sorted[j - 1] > key) {
            sorted[j] = sorted[j - 1];
            --j;
        }
        sorted[j] = key;
    }
    const double median = sorted[SAMPLES / 2];
    printf("backend=%s batches=%d repetitions=%zu\n", name, SAMPLES, repetitions);
    printf("  batch_seconds min=%.9g median=%.9g max=%.9g\n",
           sorted[0], median, sorted[SAMPLES - 1]);
    printf("  seconds_per_call=%.9g equivalent_GFLOP_s=%.6g\n",
           median / (double)repetitions,
           operations * (double)repetitions / median / 1e9);
    printf("  max_abs=%.9g checksum=%.12g\n", maximum_error, checksum);
    if (median < 0.02) {
        puts("  note: short batch; increase repetitions for a steadier measurement");
    }
}

```

### Parse and allocate before timing begins

The following three code blocks concatenate into the complete `main` function. The first validates arguments before using them. C's left-to-right short-circuit evaluation of `||` ensures the argument-count check protects later `argv` indexing. Similarly, an orientation string's length is checked before reading its two letters. `strcmp` returns zero for equality, so each explicit `== 0` is a successful match.

```c title="app/compare.c"
int main(int argc, char **argv)
{
    size_t m = 0, n = 0, k = 0, repetitions = 0, bytes = 0;
    int selection = -1; /* -1 means both; otherwise an explicit backend value. */
    if (argc != 8 || strcmp(argv[1], "kernel") != 0 ||
        !positive_size(argv[3], &m) || !positive_size(argv[4], &n) ||
        !positive_size(argv[5], &k) || !positive_size(argv[6], &repetitions)) {
        fprintf(stderr, "usage: %s kernel ref|blocked|both M N K repetitions nn|nt|tn|tt\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[2], "ref") == 0) selection = DL_BACKEND_REF;
    else if (strcmp(argv[2], "blocked") == 0) selection = DL_BACKEND_BLOCKED;
    else if (strcmp(argv[2], "both") != 0) {
        fputs("backend must be ref, blocked, or both\n", stderr);
        return EXIT_FAILURE;
    }
    if (strlen(argv[7]) != 2 ||
        (argv[7][0] != 'n' && argv[7][0] != 't') ||
        (argv[7][1] != 'n' && argv[7][1] != 't')) {
        fputs("orientation must be nn, nt, tn, or tt\n", stderr);
        return EXIT_FAILURE;
    }
    if (repetitions > 1000000 || !payload_bytes(m, n, k, &bytes) ||
        bytes > (size_t)256 * 1024 * 1024) {
        fputs("limit: 256 MiB matrix payload, 1000000 repetitions per batch\n", stderr);
        return EXIT_FAILURE;
    }
    const DlTranspose ta = argv[7][0] == 't' ? DL_TRANS : DL_NOTRANS;
    const DlTranspose tb = argv[7][1] == 't' ? DL_TRANS : DL_NOTRANS;
    const size_t shapes[4][2] = {
        {ta == DL_TRANS ? k : m, ta == DL_TRANS ? m : k},
        {tb == DL_TRANS ? n : k, tb == DL_TRANS ? k : n},
        {m, n}, {m, n}
    };
```

The next block allocates the four dense owners and obtains borrowed views. Every owner starts empty. Any failure reaches the one cleanup path at the function's end; destroying a still-empty owner is safe. The reference owner is populated once, then exposed only through a const view. The measured output is not initialized here because SET must not read it.

Before any sample, each requested backend is executed and compared with the reference. A program that produces the wrong matrix does not get a plausible-looking successful timing summary. The timer's resolution is queried outside the samples, and the dimensions, storage payload, scratch-array bytes, block dimensions, and nominal work are printed before timing begins.

```c title="app/compare.c"
    DlTensor owners[4] = {{0}};
    DlMatrix views[4] = {{0}};
    int result = EXIT_FAILURE;
    for (size_t i = 0; i < 4; ++i) {
        DlStatus s = dl_tensor_create(&owners[i], 2, shapes[i]);
        if (s == DL_OK) s = dl_tensor_rows(&owners[i], &views[i]);
        if (s != DL_OK) {
            fprintf(stderr, "allocation/view failed: %d\n", (int)s);
            goto cleanup;
        }
    }
    fill_input(&owners[0], 1);
    fill_input(&owners[1], 7);
    const Bench job = {.a = dl_matrix_read(views[0]),
        .b = dl_matrix_read(views[1]), .out = views[3],
        .ta = ta, .tb = tb, .repetitions = repetitions};
    DlStatus s = dl_matmul_ref(job.a, ta, job.b, tb, views[2], DL_SET);
    if (s != DL_OK) {
        fprintf(stderr, "reference failed: %d\n", (int)s);
        goto cleanup;
    }
    const DlConstMatrix expected = dl_matrix_read(views[2]);
    /* Check every selected backend before entering any timed sample. */
    for (int backend = 0; backend < 2; ++backend) {
        if (selection >= 0 && backend != selection) continue;
        double error = 0.0, checksum = 0.0;
        s = dl_matmul((DlBackend)backend, job.a, ta, job.b, tb, job.out, DL_SET);
        if (s != DL_OK || !compare_output(expected, job.out, &error, &checksum)) {
            fprintf(stderr, "preflight failed: status %d\n", (int)s);
            goto cleanup;
        }
    }
    struct timespec resolution = {0};
    if (clock_getres(CLOCK_MONOTONIC, &resolution) != 0) {
        perror("clock_getres");
        goto cleanup;
    }
    const double operations = 2.0 * (double)m * (double)n * (double)k;
    const size_t scratch = sizeof(double) * DL_BLOCK_M * DL_BLOCK_N
                         + sizeof(float) * DL_BLOCK_K * DL_BLOCK_N;
    printf("kernel M=%zu N=%zu K=%zu orientation=%s write=SET\n", m,n,k,argv[7]);
    printf("stored A=[%zu,%zu] B=[%zu,%zu] C=[%zu,%zu]\n",
           shapes[0][0],shapes[0][1],shapes[1][0],shapes[1][1],m,n);
    printf("matrix_payload_bytes=%zu blocked_array_scratch_bytes=%zu\n", bytes, scratch);
    printf("blocks=%d,%d,%d nominal_operations_per_call=%.0f\n",
           DL_BLOCK_M, DL_BLOCK_N, DL_BLOCK_K, operations);
    printf("clock=CLOCK_MONOTONIC reported_resolution_seconds=%.9g\n",
           (double)resolution.tv_sec + 1e-9 * (double)resolution.tv_nsec);
```

The final block records five samples per selected backend. For both backends, the sequence starts reference–blocked, then blocked–reference, and alternates thereafter. Each sample has its own immediate warm-up. No printing or error scan occurs inside the timed repetition loop; comparisons happen after the clock stops.

The sample arrays are fixed-size local values. `errors` keeps the largest elementwise discrepancy observed after any measured batch; `sums` keeps the most recent diagnostic sum for each backend. Only selected backends are reported, so zero-initialized unused timing slots never enter a median. All error paths destroy all four owners.

```c title="app/compare.c"
    double times[2][SAMPLES] = {{0}}, errors[2] = {0}, sums[2] = {0};
    const char *names[2] = {"ref", "blocked"};
    const size_t count = selection < 0 ? 2 : 1;
    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        for (size_t slot = 0; slot < count; ++slot) {
            const int backend = selection < 0 ? (int)((sample + slot) % 2) : selection;
            double error = 0.0;
            if (!run_sample(&job, (DlBackend)backend, &times[backend][sample]) ||
                !compare_output(expected, job.out, &error, &sums[backend])) goto cleanup;
            if (error > errors[backend]) errors[backend] = error;
        }
    }
    for (int backend = 0; backend < 2; ++backend) {
        if (selection < 0 || backend == selection) {
            report(names[backend], times[backend], repetitions, operations,
                   errors[backend], sums[backend]);
        }
    }
    result = EXIT_SUCCESS;
cleanup:
    for (size_t i = 0; i < 4; ++i) dl_tensor_destroy(&owners[i]);
    return result;
}
```

## Keep correctness builds separate from measurement builds

Replace the root Makefile with the following. The three inherited programs continue to run. `test_blocked` is added to the test set; `compare` is built from `app/compare.c` through explicit rules rather than the pattern for `tests/`.

`make test` also runs a small comparison in each configuration to exercise allocation, parsing of a valid command, timer calls, dispatch, and cleanup. Its debug and sanitizer timing numbers are **smoke-test diagnostics**, not performance results. For measurement use only `build/release/compare`, with the same compiler and flags for both backends. The only new optimized flag is `-fno-lto`, making the intended cross-translation-unit boundary explicit.

```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2 -fno-lto
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked

.PHONY: all debug sanitize release test clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare

build/debug/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

clean:
	rm -rf build
```

On the native Mac, run from the learner workspace root:

```sh
make clean
make test
./build/release/test_blocked
./build/release/compare kernel both 96 256 256 20 nt
```

The default compiler is `xcrun clang`. With an otherwise compatible non-Apple Clang toolchain, use `make CC=clang test`. Neither form requires a GPU backend, Accelerate, a particular compiler patch version, or an external matrix library.

The blocked test's expected successful ending is:

```text
blocked: 56 padded orientation/mode cases passed; max_abs=0
blocked: reduction precision and failure checks passed
```

The zero discrepancy was observed in the checked Clang 17 x86_64 Linux debug, sanitizer, and optimized builds. The threshold in the test, rather than an exact diagnostic decimal, is the acceptance rule. The inherited kernel checks, loss checks, and 104 finite-difference comparisons also pass in those builds. Those executions are not M4/macOS observations.

A valid comparison prints the logical and stored shapes, selected backend, five batch samples summarized as minimum/median/maximum elapsed seconds, repetitions, nominal work, equivalent rate, numerical discrepancy, and a diagnostic sum. The `both` mode prints one summary per backend. `blocked` alone still computes a scalar reference outside the timed region. A one-repetition smoke test may legitimately print the short-batch warning.

Run a small set that separates layout from output-row reuse:

```sh
./build/release/compare kernel both 96 256 256 20 nn
./build/release/compare kernel both 96 256 256 20 nt
./build/release/compare kernel both 96 256 256 20 tn
./build/release/compare kernel both 1 256 256 1000 nt
./build/release/compare kernel both 17 35 129 1000 tt
```

The single-row case is a matrix-vector-like workload; it is not a complete decoding benchmark. For a layer with $R=96,D=256,O=384$, use forward `96 384 256 nt`, input gradient `96 256 384 nn`, and weight gradient `384 256 96 tn` as the logical dimensions/orientations. The command's independently generated arrays do not form one actual forward/backward graph, so describe this as matching its product shapes.

For a native measurement, record `sw_vers`, `uname -m`, and `xcrun clang --version`, together with the command and the actual compile flags shown by `make`. `uname -m` should identify the architecture you intend to measure; do not label an x86_64/Rosetta run as native arm64. This records the environment rather than requiring a new installation. Run the commands sequentially, not with several timing processes competing in parallel.

One executed example on **x86_64 Linux with Clang 17, `-O2`, no fast math, contraction off, and no LTO** used $[96,256,256]$, NT, and 20 repetitions per batch. The scalar median batch was approximately 0.11920 seconds and the blocked median 0.037755 seconds, with zero maximum absolute difference. That is about 5.960 ms versus 1.888 ms per call in that run. The corresponding min/max ranges were approximately 0.10907–0.12349 seconds and 0.033430–0.039784 seconds. These container observations demonstrate a functioning measurement path, not predicted M4 performance, a stable benchmark ranking, or a model-training speedup.

The reported matrix payload for that case is 557056 bytes: four bytes times $96\cdot256+256\cdot256+2\cdot96\cdot256$. The blocked scratch arrays add 10240 bytes on the target float/double sizes. Neither value includes allocator metadata, runtime code, shared libraries, or other process memory. No resident-memory measurement is implied.

## Break one invariant at a time

Use a disposable source copy, rebuild, and restore each variation before the next. Run the test under sanitizers too, but distinguish an arithmetic failure from an invalid memory access.

### Reset at the wrong lifetime boundary

Inside `dl_matmul_blocked`, insert the following immediately after the opening of the `pp` loop:

```c
for (size_t i = 0; i < im; ++i) {
    for (size_t j = 0; j < jn; ++j) {
        acc[i][j] = 0.0;
    }
}
```

This repeats initialization once per reduction tile. Shapes with only one tile can still pass. The first just-over-boundary shape, $[9,33,65]$, loses the first 64 terms. In the checked fixture, its first NN/SET cell becomes approximately `0.04152249` rather than `0.07612458`. The failure is a numerical mismatch, not an AddressSanitizer finding. Explain why a single small golden product was insufficient.

### Round before the result is ready

Restore initialization. Immediately before `pp += pk`, insert:

```c
for (size_t i = 0; i < im; ++i) {
    for (size_t j = 0; j < jn; ++j) {
        if (fabs(acc[i][j]) <= (double)FLT_MAX) {
            acc[i][j] = (double)(float)acc[i][j];
        }
    }
}
```

This looks harmless because the public destination is float, but it narrows an **intermediate** that the reference keeps wide. The guard keeps this experiment's out-of-float-range partials in double; removing it would add an invalid narrowing conversion rather than isolate rounding. The $2^{24}+1-2^{24}$ fixture now finishes at 5 rather than 6 after ADD and reports `partial sum was rounded to float`. Ordinary moderate-valued products can remain within tolerance, which is why this focused check matters. Restore the original code; do not enlarge the tolerance to accept a changed precision contract.

### Skip rather than clip the tail

Restore the reduction. Replace the column-tile loop condition `jj < q.n` with `q.n - jj >= DL_BLOCK_N`, leaving its other code unchanged. This silently ignores a partial final tile. The one-by-one SET fixture now leaves its NaN output untouched and fails immediately. It performs no illegal access at all. The correct loop visits every remaining column and uses `jn` to clip the active length.

For a useful non-fault variation, change `DL_BLOCK_K` from 64 to 32, rebuild everything, and run all tests before timing again. The default failure-order fixture depends on the output blocks, not the reduction block, so it remains applicable. Keeping the same per-cell reduction order should preserve accepted results; performance can improve, worsen, or stay indistinguishable from noise. Do not change the arithmetic flags at the same time: that would confound a layout experiment with a numerical one.

## Diagnose the discrepancy before the rate

| Observation | First mechanism to inspect |
|---|---|
| Only dimensions just above a block boundary fail | Active lengths and whether the final tile is visited |
| One reduction tile works but two do not | Scratch lifetime; initialization must be outside the `pp` loop |
| SET succeeds but ADD is multiplied or offset | Old output added more than once, or read before the complete reduction |
| NT works but NN or TN fails | Stored shapes and the four logical element strides, not a new weight convention |
| Padding changes without a crash | Output uses a dense width instead of its physical `ld`, or commits inactive cells |
| Small matrices agree but the wide-partial check fails | Float intermediates or premature range checks |
| A numeric error leaves different valid-looking cells | Different traversal; discard the whole result, do not consume a prefix |
| Tiny calls show erratic rates | Timer/dispatch cost and interruption dominate; batch more work |
| Blocked is slower for a shape | Packing and scratch overhead may exceed reuse benefits; validate before tuning |
| A suspiciously large speedup appears after changing flags | Check fast math, contraction, LTO, output verification, and measured boundaries |

The backend is now an explicit execution choice for the same matrix operator. The remaining model components can use it without changing shapes, ownership, or gradient accumulation rules. [Chapter 7 in the roadmap](https://25d-228.github.io/gpt-c/index.html#chapter-7-rmsnorm-and-residual-gradient-paths) adds RMSNorm and residual paths; the kernel optimization does not replace their derivations or their gradient checks.

## References

The algorithm, traces, fixtures, and code above are self-contained. The following references support the named external mechanisms rather than substitute for their implementation:

- R. Clint Whaley, ATLAS developer documentation: [Speeding up the Level 3 BLAS](https://netlib.org/atlas/developer/atlas_contrib/node10.html) and [Building the General Matrix Multiply](https://netlib.org/atlas/developer/atlas_contrib/node11.html), for packing, reuse, incomplete blocks, and copy-overhead tradeoffs. The tile constants here are not taken as an optimum from that implementation.
- Clang: [Floating-point behavior](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior) and [`-ffp-contract` argument](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-ffp-contract), for the numerical compiler controls retained by the Makefile.
- OpenBSD's primary [clock_gettime/clock_getres manual](https://man.openbsd.org/clock_gettime.2), for the POSIX interface, return status, timespec fields, and elapsed-versus-CPU distinction; [GNU Gnulib's portability note](https://www.gnu.org/software/gnulib/manual/html_node/clock_005fgettime.html), for the fact that this timer is not universally supplied by a C implementation. Consult the installed Mac manual for its local implementation, not BSD-specific clock extensions.

