#define _POSIX_C_SOURCE 200809L
#include "dl/kernels.h"
#include "pretrained.h"

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

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "pretrained") == 0)
        return dl_compare_pretrained(argc, argv);
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
