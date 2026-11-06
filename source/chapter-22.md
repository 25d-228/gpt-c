# Chapter 22 — End-to-end demonstration and measured tradeoffs

A model implementation is now present all the way from bytes to gradients and from a saved weight table to a selected next token. The remaining engineering problem is to make its demonstration repeatable and its measurements interpretable. A faster number is not useful when it measures a different prefix, excludes an expensive packing step, or reports memory after a reference allocation has already raised the process peak.

> **Learning objectives.** Build a C measurement path that separates prefill from subsequent inputs, verifies identical prefixes before timing, measures one storage strategy per process, and distinguishes live arrays from resident memory. Connect actual byte-model training, restart, generation, tiled backward, and the existing pretrained-reference workflow without introducing a second model implementation.

Start from the complete [Chapter 21 source state](https://25d-228.github.io/gpt-c/media/chapter-21/reference/Makefile), including `app/compare.c`, `app/pretrained.c`, the checkpoint loader, and all tests. The completed [Chapter 22 source state](https://25d-228.github.io/gpt-c/media/chapter-22/reference/Makefile) is a companion, not a replacement for the explanations. Chapters 15–19 supply the trainer, immutable inference owners, cache, and tiled tape. Chapters 20–21 supply conversion, explicit vocabulary IDs, observation, and independent reference equations. No new neural operator is hidden in this chapter.

Four parts form one **5.00-hour implementation chapter**: this part develops inference measurement; [Part 2](https://25d-228.github.io/gpt-c/chapter-22-part-2.html) measures complete forward/backward pairs and tests the harness; [Part 3](https://25d-228.github.io/gpt-c/chapter-22-part-3.html) connects the reproducible byte-model run; [Part 4](https://25d-228.github.io/gpt-c/chapter-22-part-4.html) interprets actual results and supplies the genuine-asset driver. Allocate 2.00 hours to integration, 1.50 to useful measurements, 0.75 to diagnosis and reruns, and 0.75 to the C tile/shape sweep. Reading and unattended computation are separate. No exact speedup is required.

## Decide what one invocation means

There are two actions, `check` and `measure`. A check may own both a candidate and a materialized scalar reference. It compares every output element, then exits **without publishing timing or resident-memory numbers**. A measurement loads the same weights in a new process and constructs only the selected candidate. Deallocating a reference inside the measurement process would not repair its peak-memory statistic.

There are also two independent implementation choices. `ref` and `blocked` choose matrix multiplication. `full`, `tiled`, and `cache` choose how inference handles attention/history. Tiled full-prefix execution still recomputes the entire prefix for the next input; it is not a persistent cache. The current cache uses the Chapter 17 row-wise attention path; it is not a fused cached/tiled backend.

The two new command forms are:

```text
compare model check|measure CONFIG WEIGHTS full|tiled|cache ref|blocked PROMPT STEPS REPS QUERY_TILE KEY_TILE
compare training check|measure CONFIG full|tiled ref|blocked TIME REPS QUERY_TILE KEY_TILE
```

`WEIGHTS` can be an existing byte-model checkpoint or a Chapter 20 weight-only file. `CONFIG` is the strict, ordered configuration file already implemented. The model command ignores its training batch/time values: inference is one sequence, with capacity `PROMPT+STEPS`. All integer arguments are positive. `REPS` is at most 1,000, capacity/time at most 4,096, and each tile dimension at most 256. Tiles affect only `tiled` mode; the other modes report but do not use them. Imported position limits still apply. The bounds make accidental huge experiments fail before a long computation; they are application limits, not transformer laws.

## Count inputs, not vaguely named generated tokens

Let $P$ be the prompt length, $G$ the number of subsequent **consumed input IDs**, and $C=P+G$ the workspace capacity. The first call produces logits after consuming $P$ IDs. Then the experiment consumes one deterministic ID at each of the contexts $P+1,\ldots,P+G$ and obtains another logit row. It computes $G+1$ predictions in total. These IDs are chosen in advance, not sampled from the predictions.

For $P=3$ and $G=2$, the cache length progresses $0\rightarrow3\rightarrow4\rightarrow5$. Full-prefix execution instead evaluates prefixes of lengths 3, 4, and 5. The reference and candidate receive exactly the same five IDs. In particular, this differs from the generation application: selecting two output IDs after a three-ID prompt normally needs only four consumed inputs. A selected final ID need not be consumed to return the transcript.

This distinction prevents a misleading cache comparison in which one engine samples different tokens, then appears numerically different because it is answering a different question.

## Derive the cost that the clocks will observe

Let $L$ be the number of decoder blocks, $D$ the model width, $F$ the feed-forward width, $H_q$ the query-head count, $H_{kv}$ the key/value-head count, $d=D/H_q$ the head width, $K=H_{kv}d$ the shared key/value width, and $V$ the vocabulary size. One block's learned matrix values, excluding its normalization scales, total

$$
W=2D^2+2KD+3FD.
$$

At a prefix of length $t$, dense projections therefore require nominally $2tW$ scalar multiply/add operations per block. The attention query/key and probability/value products together require approximately $4H_qd\,t(t+1)/2$ operations when only causal pairs are counted. This does not count exponentials, normalization, validation, or indexing. Full-prefix generation repeats this work for every $t=P+1,\ldots,P+G$.

A cached step computes new projections for one row, so its projection term is approximately $2W$ per block rather than $2tW$. It still attends to $t$ stored keys and values: that term grows approximately as $4H_qdt$. The tied output projection also costs about $2DV$ for each requested last-row prediction. These formulas explain why a large vocabulary or repeated validation can obscure an attention-only speedup. They do not predict wall-clock time.

Tiling changes storage and traversal, not the set of dense causal pairs. The original [FlashAttention analysis, Section 3.1](https://arxiv.org/html/2205.14135v2#S3.SS1) motivates avoiding quadratic intermediates and recomputing them during backward. This C laboratory is not that paper's GPU kernel, and GPU speedup figures do not transfer to its scalar loops.

## A clock interval, a sample, and a reported item

For inference, one replay starts from an empty cache, or a fresh full-prefix call, and executes the same transcript. Allocation, loading, cache reset, output checking, and printing are outside the timed regions. The first region contains prefill. The second contains the $G$ subsequent input calls and their loop/dispatch overhead. Sampling and tokenization are absent.

For sample $s$ and repetition count $R$, let $a_{s,r}$ be the prefill interval and $b_{s,r}$ the interval containing all $G$ subsequent calls. Report

$$
t_{\mathrm{prefill}}=\operatorname{median}_{s=1}^{5}
\left(\frac{1}{R}\sum_{r=1}^{R}a_{s,r}\right),
$$

$$
t_{\mathrm{decode/item}}=\frac{1}{G}\operatorname{median}_{s=1}^{5}
\left(\frac{1}{R}\sum_{r=1}^{R}b_{s,r}\right).
$$

The second quantity is a mean latency across a specified growing-context interval, then a median across five samples. It is **not** a percentile of individual token latencies. For three repetitions, two consumed IDs per replay, and a median decode batch of 0.012 seconds, the reported time is $0.012/(3\times2)=0.002$ seconds per consumed ID. Omitting either divisor is a measurement bug.

`CLOCK_MONOTONIC` measures elapsed time rather than CPU time. It includes descheduling. The [clock interface documentation](https://man7.org/linux/man-pages/man3/clock_gettime.3.html) explains why the monotonic clock is preferable to calendar time for these intervals; its reported resolution is not a claim that every measurement is accurate to that resolution. Do not subtract an independently estimated timer cost from very short samples and present the remainder as precise. Increase the workload or repetition count instead.

## Create the measurement module

Create `app/measure.h`:

```c title="app/measure.h"
#ifndef DL_MEASURE_H
#define DL_MEASURE_H

/* CLI entry points. Each invocation owns and destroys all of its state. */
int dl_compare_model(int argc, char **argv);
int dl_compare_training(int argc, char **argv);

#endif
```

Create `app/measure.c` by concatenating its listings in order across this part and Part 2. The feature-test macro precedes every header so POSIX clock declarations are visible. `TRY` propagates an existing decoder status to one cleanup path; `REQUIRE` handles host/API failures that do not use `DlStatus`. Neither macro continues after a failed timed call.

The numeric parser admits decimal digits only and checks before multiplying. Unlike `atoi`, it cannot silently turn an invalid token count into zero or wrap a large one. The backend parser keeps a valid enum out of the rest of the computation.

```c title="app/measure.c"
#define _POSIX_C_SOURCE 200809L
#include "measure.h"
#include "dl/cache.h"
#include "dl/checkpoint.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

enum { SAMPLE_COUNT = 5, MAX_TOKENS = 4096, MAX_REPETITIONS = 1000 };
static const size_t MIB = (size_t)1024 * 1024;

#define TRY(call) do { DlStatus status_ = (call); if (status_ != DL_OK) { \
    fprintf(stderr, "%s: %s\n", #call, dl_status_name(status_)); goto done; \
} } while (0)
#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "failed: %s\n", #condition); goto done; \
} } while (0)

static int positive(const char *text, size_t *out)
{
    size_t n = 0;
    if (*text == '\0') return 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9') return 0;
        size_t digit = (size_t)(*text - '0');
        if (n > (SIZE_MAX - digit) / 10) return 0;
        n = n * 10 + digit;
    }
    if (n == 0) return 0;
    *out = n;
    return 1;
}

static int backend_name(const char *text, DlBackend *out)
{
    if (strcmp(text, "ref") == 0) *out = DL_BACKEND_REF;
    else if (strcmp(text, "blocked") == 0) *out = DL_BACKEND_BLOCKED;
    else return 0;
    return 1;
}

```

### Measure elapsed time without rounding the clock's origin

Subtract the integer `timespec` fields first, then convert their small difference to double. Converting a very large absolute clock origin to a float would needlessly lose small elapsed intervals. The nanosecond difference can be negative when the second counter crosses a boundary; adding both differences handles that normally.

`close_values` explicitly rejects non-finite values. A comparison such as `fabs(NaN) > tolerance` alone would be false and could admit a broken run. Its error rule is coordinatewise: $|x-r|\le a+b|r|$. Independent strategy checks use $a=b=3\times10^{-5}$ as a local diagnostic target; repeated identical execution uses zero tolerance. The target is not a guarantee for every imported model or platform.

`report` preserves all five raw batch intervals, sorts a copy, divides by repetitions exactly once, then divides by items only for the item-level rate. Five samples expose some variability; they do not form a confidence interval. The 0.02-second warning is a prompt to investigate timer/scheduling noise, not an acceptance threshold for model correctness.

```c title="app/measure.c"
static int tick(struct timespec *out)
{
    if (clock_gettime(CLOCK_MONOTONIC, out) == 0) return 1;
    perror("clock_gettime");
    return 0;
}

static double elapsed(struct timespec a, struct timespec b)
{
    return (double)(b.tv_sec - a.tv_sec) +
           1e-9 * (double)(b.tv_nsec - a.tv_nsec);
}

static int close_values(const float *expected, const float *actual, size_t count,
                        double atol, double rtol, double *largest)
{
    for (size_t i = 0; i < count; ++i) {
        double r = (double)expected[i], x = (double)actual[i];
        double error = fabs(x - r);
        if (!isfinite(r) || !isfinite(x) || error > atol + rtol * fabs(r)) {
            fprintf(stderr, "difference[%zu]: expected %.9g actual %.9g\n", i, r, x);
            return 0;
        }
        if (error > *largest) *largest = error;
    }
    return 1;
}

static int report(const char *phase, const double seconds[SAMPLE_COUNT],
                  size_t repetitions, size_t items)
{
    double ordered[SAMPLE_COUNT];
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        if (!isfinite(seconds[i]) || seconds[i] <= 0.0) return 0;
        ordered[i] = seconds[i] / (double)repetitions;
        printf("sample phase=%s index=%zu batch_seconds=%.9g\n", phase, i, seconds[i]);
    }
    for (size_t i = 1; i < SAMPLE_COUNT; ++i) {
        double x = ordered[i];
        size_t j = i;
        while (j > 0 && ordered[j-1] > x) {
            ordered[j] = ordered[j-1];
            --j;
        }
        ordered[j] = x;
    }
    double median = ordered[SAMPLE_COUNT/2];
    printf("phase=%s repetitions=%zu items_per_replay=%zu min_seconds=%.9g "
           "median_seconds=%.9g max_seconds=%.9g seconds_per_item=%.9g items_per_second=%.9g\n",
           phase, repetitions, items, ordered[0], median, ordered[SAMPLE_COUNT-1],
           median/(double)items, (double)items/median);
    if (median * (double)repetitions < 0.02)
        puts("note=short_sample_increase_repetitions_or_workload");
    return !ferror(stdout);
}

```

### Resident memory is a process statistic, not an arena length

The live array count answers “what storage does this ownership design request?” Peak resident set size answers “what maximum resident memory did this process reach?” They differ because of executable/runtime pages, stack, allocator behavior, lazily touched pages, and earlier loading allocations. A process peak is not reset by `free`.

`getrusage(RUSAGE_SELF)` measures this C process rather than a Python reference or a sum of children. Linux documents `ru_maxrss` in KiB in its [system-call manual](https://man7.org/linux/man-pages/man2/getrusage.2.html). The [macOS manual reproduced by FreeBSD](https://man.freebsd.org/cgi/man.cgi?query=getrusage&sektion=2&manpath=macOS+10.13.6) specifies bytes. The native value/unit is printed alongside the conversion. Check the installed `man 2 getrusage` on another target; some very old archived Apple pages describe different units. This is not an Activity Monitor “memory footprint” implementation.

The preprocessor branches choose the known platform conversion without pretending it is universal. An unknown platform gets an explicitly unconverted native number. Stack locals, including the known 10,240-byte blocked-kernel array scratch, are not added to a heap count. Kernel/runtime stacks contribute to process memory instead.

```c title="app/measure.c"
static int environment(void)
{
    struct timespec resolution = {0};
    if (clock_getres(CLOCK_MONOTONIC, &resolution) != 0) return 0;
    printf("clock=CLOCK_MONOTONIC resolution_seconds=%.9g samples=%d\n",
           (double)resolution.tv_sec + 1e-9*(double)resolution.tv_nsec, SAMPLE_COUNT);
#ifdef __OPTIMIZE__
    puts("compiler_optimization=enabled");
#else
    puts("compiler_optimization=disabled_not_a_performance_result");
#endif
    return !ferror(stdout);
}

static int resident_peak(void)
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__) && defined(__MACH__)
    printf("peak_rss_native=%ld native_unit=bytes peak_rss_bytes=%.0f\n",
           usage.ru_maxrss, (double)usage.ru_maxrss);
#elif defined(__linux__)
    printf("peak_rss_native=%ld native_unit=KiB peak_rss_bytes=%.0f\n",
           usage.ru_maxrss, 1024.0*(double)usage.ru_maxrss);
#else
    printf("peak_rss_native=%ld native_unit=consult_local_getrusage_manual\n",
           usage.ru_maxrss);
#endif
    return !ferror(stdout);
}

/* Accounting uses checked addition even though constructors have already bounded
 * each individual allocation. The totals exclude allocator overhead and stacks. */
static int add_bytes(size_t *total, size_t count, size_t element)
{
    size_t bytes = 0, next = 0;
    if (dl_size_mul(count, element, &bytes) != DL_OK ||
        dl_size_add(*total, bytes, &next) != DL_OK) return 0;
    *total = next;
    return 1;
}

```

### One owner for each selected execution strategy

A `Runner` embeds empty candidate owners but constructs only one of them. It borrows an immutable `DlWeights` object, which must outlive it. Destruction always releases workspace/cache storage before the weights. The 512 MiB workspace budget is checked by the existing constructors; it is not a process-memory cap.

`prefix` is only dispatch glue. The forward functions remain the complete C implementations from earlier chapters. For a cached first call, the caller must have an empty cache; later calls pass exactly `ids[count-1]`. For a full/tiled call, `count-1` selects the last hidden row for the final vocabulary projection while the block stack still evaluates all prefix rows.

```c title="app/measure.c"
typedef struct {
    DlInference full;
    DlKvCache cache;
    int cached;
} Runner;

static DlStatus runner_create(Runner *r, const DlWeights *weights, const char *mode,
                               size_t capacity, size_t qt, size_t kt)
{
    r->cached = strcmp(mode, "cache") == 0;
    if (r->cached) return dl_cache_create(&r->cache, weights, capacity, 512*MIB);
    if (strcmp(mode, "tiled") == 0)
        return dl_inference_create_tiled(&r->full, weights, capacity, qt, kt, 512*MIB);
    return dl_inference_create(&r->full, weights, capacity, 512*MIB);
}

static void runner_destroy(Runner *r)
{
    dl_cache_destroy(&r->cache);
    dl_inference_destroy(&r->full);
    r->cached = 0;
}

static DlStatus prefix(Runner *r, DlBackend backend, const size_t *ids,
                        size_t count, int first, DlMatrix output)
{
    if (!r->cached)
        return dl_inference_forward(&r->full, backend, ids, count, count-1, output);
    if (first) return dl_cache_prefill(&r->cache, backend, ids, count, output);
    return dl_cache_decode(&r->cache, backend, ids[count-1], output);
}

```

### Account for the actual owners

Let $C$ be capacity, and $Q_b,K_b$ the query/key tile dimensions. Inference reuses one block's scratch rather than retaining every block's activations. For this implementation, define $A(C)=C(7D+3K+3F)$. The materialized scratch contains

$$
4\left[A(C)+2H_qC^2\right]\quad\text{bytes}.
$$

The tiled variant contains $4A(C)$ float bytes plus

$$
8\left[3CH_q+Q_bK_b+Q_bd+2Q_b\right]
$$

double bytes. The three row-statistic planes retain maximum, scaled sum, and log-sum-exp. They must not be omitted from an “attention memory” count.

The cache's history occupies $4(2LCK)$ bytes; its scratch occupies $4C(7D+K+3F+2)$ bytes. The extra two cells per capacity position are reusable score/probability rows, not a square matrix. For the pilot, $L=2,D=32,K=16,F=80,H_q=4$ and $C=136$. History is $4\times2\times2\times136\times16=34{,}816$ bytes. Cache scratch is 262,208 bytes. Those numbers describe reserved capacity, not just the initial 128 consumed IDs.

`model_storage` reads constructed owner lengths instead of duplicating those layout formulas. It also counts the weight directory, input IDs, position IDs, rotary frequencies, and two vocabulary rows. One row is the live result and one stores the warmup's final result. The owner structures themselves live on the stack. The whole-process measurement additionally includes loading and startup peaks, even though those operations are outside the timer.

```c title="app/measure.c"
static int model_storage(const DlWeights *w, const Runner *r, size_t capacity)
{
    size_t weights = 0, scratch = 0, history = 0, other = 0;
    if (!add_bytes(&weights, w->arena.count, sizeof(float)) ||
        !add_bytes(&scratch, r->full.arena.count, sizeof(float)) ||
        !add_bytes(&scratch, r->full.tiled.count, sizeof(double)) ||
        !add_bytes(&scratch, r->cache.arena.count, sizeof(float)) ||
        !add_bytes(&history, r->cache.history.count, sizeof(float)) ||
        !add_bytes(&other, w->param_count, sizeof(DlWeight)) ||
        !add_bytes(&other, capacity, 2*sizeof(size_t)) ||
        !add_bytes(&other, w->config.width/w->config.query_heads/2, sizeof(double)) ||
        !add_bytes(&other, w->config.vocab, 2*sizeof(float))) return 0;
    size_t total = weights;
    if (!add_bytes(&total, scratch, 1) || !add_bytes(&total, history, 1) ||
        !add_bytes(&total, other, 1)) return 0;
    printf("payload weights=%zu scratch=%zu history=%zu other_heap=%zu total_heap=%zu\n",
           weights, scratch, history, other, total);
    return !ferror(stdout);
}

```

## Complete inference measurement and cleanup

The following function assembles these pieces. All heap pointers and owners start empty so any failure can take the same cleanup path. Only `check` constructs `reference`. Its every-prefix comparison uses scalar, full-prefix execution as the baseline. This is an internal consistency check, not an independent mathematical proof; Chapter 21's eager equations and the earlier hand/gradient checks remain necessary.

For `measure`, sample zero is one untimed warmup transcript. Subsequent samples replay that transcript `REPS` times. Resetting cache length before each replay ensures that the next prefill starts at absolute position zero, not at the end of the previous repetition. Every timed replay must end with exactly the same finite last-row logits as the warmup. That replay check catches state drift but cannot prove that both executions implemented the correct model.

The deterministic input expression stays bounded before modulo, and its IDs lie in `[0,V)`. They are workload fixtures, not a natural-language prompt or a tokenizer substitute. Loading accepts at most 1 GiB of FP32 weights and a 2 GiB file. Its genuine-import configuration must already match the supplied config file.

```c title="app/measure.c"
int dl_compare_model(int argc, char **argv)
{
    size_t prompt = 0, steps = 0, reps = 0, qt = 0, kt = 0, capacity = 0;
    DlBackend backend = DL_BACKEND_REF;
    if (argc != 12 || (strcmp(argv[2], "check") && strcmp(argv[2], "measure")) ||
        (strcmp(argv[5], "full") && strcmp(argv[5], "tiled") && strcmp(argv[5], "cache")) ||
        !backend_name(argv[6], &backend) || !positive(argv[7], &prompt) ||
        !positive(argv[8], &steps) || !positive(argv[9], &reps) ||
        !positive(argv[10], &qt) || !positive(argv[11], &kt) ||
        dl_size_add(prompt, steps, &capacity) != DL_OK || capacity > MAX_TOKENS ||
        reps > MAX_REPETITIONS || qt > 256 || kt > 256) {
        fprintf(stderr, "usage: %s model check|measure CONFIG WEIGHTS full|tiled|cache "
                "ref|blocked PROMPT STEPS REPS QUERY_TILE KEY_TILE\n", argv[0]);
        return EXIT_FAILURE;
    }
    int result = EXIT_FAILURE, check = strcmp(argv[2], "check") == 0;
    DlModelConfig config = {0};
    DlWeights weights = {0};
    Runner candidate = {0}, reference = {0};
    size_t ignored_b = 0, ignored_t = 0;
    size_t *ids = NULL;
    float *rows = NULL;
    TRY(dl_model_config_load(argv[3], &config, &ignored_b, &ignored_t));
    REQUIRE(config.vocab <= 65536);
    TRY(dl_checkpoint_load_weights(argv[4], &weights, config,
                                    UINT64_C(2)*1024*1024*1024, 1024*MIB));
    TRY(runner_create(&candidate, &weights, argv[5], capacity, qt, kt));
    ids = malloc(capacity * sizeof(*ids));
    rows = malloc(2 * config.vocab * sizeof(*rows));
    REQUIRE(ids != NULL && rows != NULL);
    for (size_t i = 0; i < capacity; ++i) ids[i] = (37*(i%97)+11) % config.vocab;
    DlMatrix actual = {rows, 2*config.vocab, 0, 1, config.vocab, config.vocab};
    DlMatrix expected = actual;
    expected.offset = config.vocab;
    printf("model action=%s mode=%s backend=%s prompt=%zu steps=%zu capacity=%zu "
           "query_tile=%zu key_tile=%zu\n", argv[2], argv[5], argv[6], prompt, steps,
           capacity, qt, kt);
    REQUIRE(environment());
    if (check) {
        TRY(runner_create(&reference, &weights, "full", capacity, qt, kt));
        double largest = 0.0;
        for (size_t step = 0; step <= steps; ++step) {
            size_t n = prompt + step;
            TRY(prefix(&reference, DL_BACKEND_REF, ids, n, 1, expected));
            TRY(prefix(&candidate, backend, ids, n, step == 0, actual));
            REQUIRE(close_values(rows+config.vocab, rows, config.vocab,
                                 3e-5, 3e-5, &largest));
        }
        printf("checked_values=%zu max_abs=%.9g atol=3e-5 rtol=3e-5\n",
               (steps+1)*config.vocab, largest);
        puts("check_only=no_timing_or_rss_reference_buffers_are_live");
    } else {
        /* Exactly one selected workspace; no materialized reference ever exists. */
        REQUIRE(model_storage(&weights, &candidate, capacity));
        double pre[SAMPLE_COUNT] = {0}, decode[SAMPLE_COUNT] = {0};
        for (size_t sample = 0; sample <= SAMPLE_COUNT; ++sample) {
            size_t repeats = sample == 0 ? 1 : reps;
            for (size_t repeat = 0; repeat < repeats; ++repeat) {
                if (candidate.cached) TRY(dl_cache_reset(&candidate.cache));
                struct timespec a = {0}, b = {0}, c = {0};
                REQUIRE(tick(&a));
                TRY(prefix(&candidate, backend, ids, prompt, 1, actual));
                REQUIRE(tick(&b));
                for (size_t step = 1; step <= steps; ++step)
                    TRY(prefix(&candidate, backend, ids, prompt+step, 0, actual));
                REQUIRE(tick(&c));
                if (sample == 0) {
                    for (size_t j = 0; j < config.vocab; ++j) {
                        REQUIRE(isfinite(rows[j]));
                        rows[config.vocab+j] = rows[j];
                    }
                } else {
                    double largest = 0.0;
                    REQUIRE(close_values(rows+config.vocab, rows, config.vocab, 0, 0, &largest));
                    pre[sample-1] += elapsed(a, b);
                    decode[sample-1] += elapsed(b, c);
                }
            }
        }
        REQUIRE(report("prefill", pre, reps, prompt));
        REQUIRE(report("decode", decode, reps, steps));
        REQUIRE(resident_peak());
        puts("repeat_final_logits=exact no_sampling_no_io_in_timed_region=1");
    }
    REQUIRE(fflush(stdout) == 0);
    result = EXIT_SUCCESS;
done:
    free(rows);
    free(ids);
    runner_destroy(&reference);
    runner_destroy(&candidate);
    dl_weights_destroy(&weights);
    return result;
}

```

Notice the lifetime order at `done`: free caller-owned rows and IDs, destroy reference and candidate workspaces, and only then destroy the shared immutable weights. No output row borrows those arenas. A later numeric failure invalidates its computation; the harness exits rather than timing a retry as though it were useful work. `fflush` is checked so an output-device failure cannot be reported as a successful measurement.

The next part appends the training measurement functions, connects these entry points to `compare`, and checks the harness itself. Do not compile a half-written `measure.c`: it deliberately continues there.
