#include "buffer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t attempts;
    size_t fail_at;
} AllocationProbe;

static void *probe_allocate(size_t bytes, void *context)
{
    AllocationProbe *probe = context;
    ++probe->attempts;
    if (probe->fail_at != 0 && probe->attempts == probe->fail_at) {
        return NULL;
    }
    return malloc(bytes);
}

static int expect(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

static int same_view(FloatView a, FloatView b)
{
    return a.data == b.data && a.count == b.count && a.stride == b.stride;
}

static int test_views(void)
{
    int ok = 0;
    FloatBuffer matrix = {.data = NULL, .count = 0};
    FloatView row = {.data = NULL, .count = 0, .stride = 1};
    FloatView column = {.data = NULL, .count = 0, .stride = 1};
    FloatView empty = {.data = NULL, .count = 0, .stride = 1};
    AllocationProbe probe = {.attempts = 0, .fail_at = 0};
    size_t count = 0;

    if (!expect(size_mul(2, 3, &count) == STORAGE_OK, "matrix element count")) {
        goto cleanup;
    }
    StorageStatus status = buffer_create(&matrix, count, probe_allocate, &probe);
    if (!expect(status == STORAGE_OK, storage_status_name(status))) {
        goto cleanup;
    }
    for (size_t i = 0; i < matrix.count; ++i) {
        matrix.data[i] = (float)(i + 1);
    }

    if (!expect(buffer_view(&matrix, 3, 3, 1, &row) == STORAGE_OK,
                "borrow row 1") ||
        !expect(buffer_view(&matrix, 1, 2, 3, &column) == STORAGE_OK,
                "borrow column 1")) {
        goto cleanup;
    }
    const float row_total = view_sum(row);
    const float column_total = view_sum(column);
    printf("row 1 sum = %.0f\n", (double)row_total);
    printf("column 1 sum = %.0f\n", (double)column_total);
    if (!expect(row_total == 15.0f && column_total == 7.0f, "strided sums")) {
        goto cleanup;
    }

    FloatView copy = row;
    matrix.data[3] = 40.0f;
    if (!expect(view_sum(copy) == 51.0f && view_sum(row) == 51.0f,
                "view copies alias the original floats")) {
        goto cleanup;
    }
    matrix.data[3] = 4.0f;

    FloatView unchanged = row;
    if (!expect(buffer_view(&matrix, 3, 2, 3, &unchanged) == STORAGE_RANGE &&
                same_view(unchanged, row), "invalid last element preserves view") ||
        !expect(buffer_view(&matrix, 1, 2, SIZE_MAX, &unchanged) == STORAGE_RANGE &&
                same_view(unchanged, row), "huge stride rejected safely") ||
        !expect(buffer_view(&matrix, 0, 1, 0, &unchanged) == STORAGE_ARGUMENT &&
                same_view(unchanged, row), "zero stride rejected") ||
        !expect(buffer_view(&matrix, SIZE_MAX, 0, 1, &unchanged) == STORAGE_RANGE &&
                same_view(unchanged, row), "invalid empty offset rejected") ||
        !expect(buffer_view(&matrix, matrix.count, 0, 1, &empty) == STORAGE_OK &&
                empty.data == NULL && empty.count == 0 && empty.stride == 1 &&
                view_sum(empty) == 0.0f, "canonical empty view")) {
        goto cleanup;
    }

    float *original = matrix.data;
    const size_t attempts = probe.attempts;
    status = buffer_create(&matrix, 2, probe_allocate, &probe);
    if (!expect(status == STORAGE_STATE && matrix.data == original &&
                matrix.count == 6 && matrix.data[3] == 4.0f &&
                probe.attempts == attempts, "live owner cannot be overwritten")) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    /* No view, saved pointer, or owner copy is inspected after destruction. */
    buffer_destroy(&matrix);
    buffer_destroy(&matrix);
    if (!expect(matrix.data == NULL && matrix.count == 0, "destroy resets owner")) {
        ok = 0;
    }
    return ok;
}

static int test_sizes_and_failures(void)
{
    int ok = 0;
    size_t product = 77;
    FloatBuffer candidate = {.data = NULL, .count = 0};
    FloatView empty = {.data = NULL, .count = 0, .stride = 1};
    AllocationProbe probe = {.attempts = 0, .fail_at = 1};

    if (!expect(size_mul(SIZE_MAX, 2, &product) == STORAGE_OVERFLOW &&
                product == 77, "overflow leaves scalar output unchanged") ||
        !expect(size_mul(0, SIZE_MAX, &product) == STORAGE_OK && product == 0,
                "zero product") ||
        !expect(size_mul(1, SIZE_MAX, &product) == STORAGE_OK && product == SIZE_MAX,
                "maximum representable product") ||
        !expect(size_mul(2, 3, NULL) == STORAGE_ARGUMENT, "null size output")) {
        goto cleanup;
    }
    if (!expect(sizeof(float) > 1, "byte-overflow fixture needs sizeof(float) > 1")) {
        goto cleanup;
    }
    StorageStatus status = buffer_create(&candidate, SIZE_MAX, probe_allocate, &probe);
    if (!expect(status == STORAGE_OVERFLOW && probe.attempts == 0 &&
                candidate.data == NULL && candidate.count == 0,
                "byte overflow rejected before allocator")) {
        goto cleanup;
    }
    puts("overflow rejected before allocation");

    status = buffer_create(&candidate, 0, probe_allocate, &probe);
    if (!expect(status == STORAGE_OK && probe.attempts == 0 &&
                candidate.data == NULL && candidate.count == 0, "empty owner") ||
        !expect(buffer_view(&candidate, 0, 0, 1, &empty) == STORAGE_OK &&
                view_sum(empty) == 0.0f, "empty owner view without pointer arithmetic")) {
        goto cleanup;
    }
    status = buffer_create(&candidate, 6, probe_allocate, &probe);
    if (!expect(status == STORAGE_ALLOC && probe.attempts == 1 &&
                candidate.data == NULL && candidate.count == 0,
                "injected constructor failure preserves empty owner")) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    buffer_destroy(&candidate);
    buffer_destroy(NULL);
    return ok;
}

static StorageStatus matrix_sums(float *out, AllocateFn allocate, void *context)
{
    FloatBuffer matrix = {.data = NULL, .count = 0};
    FloatBuffer scratch = {.data = NULL, .count = 0};
    FloatView row = {.data = NULL, .count = 0, .stride = 1};
    FloatView column = {.data = NULL, .count = 0, .stride = 1};
    StorageStatus status = STORAGE_ARGUMENT;

    if (out == NULL || allocate == NULL) {
        goto cleanup;
    }
    status = buffer_create(&matrix, 6, allocate, context);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    status = buffer_create(&scratch, 2, allocate, context);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    for (size_t i = 0; i < matrix.count; ++i) {
        matrix.data[i] = (float)(i + 1);
    }
    status = buffer_view(&matrix, 3, 3, 1, &row);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    status = buffer_view(&matrix, 1, 2, 3, &column);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    scratch.data[0] = view_sum(row);
    scratch.data[1] = view_sum(column);
    out[0] = scratch.data[0];
    out[1] = scratch.data[1];

cleanup:
    buffer_destroy(&scratch);
    buffer_destroy(&matrix);
    return status;
}

static int test_partial_cleanup(void)
{
    for (size_t fail_at = 1; fail_at <= 2; ++fail_at) {
        AllocationProbe probe = {.attempts = 0, .fail_at = fail_at};
        float out[2] = {-7.0f, -9.0f};
        StorageStatus status = matrix_sums(out, probe_allocate, &probe);
        if (!expect(status == STORAGE_ALLOC && probe.attempts == fail_at &&
                    out[0] == -7.0f && out[1] == -9.0f,
                    "partial construction preserves caller result")) {
            return 0;
        }
        printf("allocation %zu failure: caller result unchanged\n", fail_at);
    }
    float out[2] = {-7.0f, -9.0f};
    StorageStatus status = matrix_sums(out, system_allocate, NULL);
    if (!expect(status == STORAGE_OK && out[0] == 15.0f && out[1] == 7.0f,
                "success after injected failures")) {
        return 0;
    }
    printf("success after failure: [%.0f, %.0f]\n", (double)out[0], (double)out[1]);
    return 1;
}

/* out[0..n) must not overlap either input. Read-only inputs may overlap. */
static void add_disjoint(float *restrict out, const float *restrict a,
                         const float *restrict b, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

int main(void)
{
    if (!test_views() || !test_sizes_and_failures() || !test_partial_cleanup()) {
        return EXIT_FAILURE;
    }
    const float input[3] = {1.0f, 2.0f, 3.0f};
    float doubled[3] = {0.0f, 0.0f, 0.0f};
    add_disjoint(doubled, input, input, 3);
    if (!expect(doubled[0] == 2.0f && doubled[1] == 4.0f && doubled[2] == 6.0f,
                "restricted output with shared read-only inputs")) {
        return EXIT_FAILURE;
    }
    puts("storage checks passed");
    return EXIT_SUCCESS;
}
