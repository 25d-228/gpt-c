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
