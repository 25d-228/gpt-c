#include "dl/kernels.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int require_status(DlStatus got, DlStatus expected, const char *where)
{
    if (got != expected) {
        fprintf(stderr, "%s: status %d, expected %d\n",
                where, (int)got, (int)expected);
        return 0;
    }
    return 1;
}

/* Fixed small test shapes, but construction still checks the size arithmetic. */
static int padded(DlTensor *owner, DlMatrix *view, size_t rows, size_t cols)
{
    size_t ld = 0, count = 0;
    DlStatus status = dl_size_add(cols, 3, &ld);
    if (status == DL_OK) status = dl_size_mul(rows, ld, &count);
    if (status == DL_OK) status = dl_size_add(count, 4, &count);
    if (status == DL_OK) status = dl_tensor_create(owner, 1, &count);
    if (status == DL_OK) {
        status = dl_matrix_view(owner, 2, rows, cols, ld, view);
    }
    if (!require_status(status, DL_OK, "padded allocation")) return 0;
    for (size_t i = 0; i < owner->count; ++i) owner->data[i] = -12345.0f;
    return 1;
}

static float value_at(size_t i, size_t j, unsigned seed)
{
    const int q = (int)((i % 23 * 7 + j % 23 * 11 + seed) % 23) - 11;
    return (float)q / 17.0f;
}

static void fill(DlMatrix v, unsigned seed, int nan_cells)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            v.base[v.offset + i * v.ld + j] = nan_cells
                ? NAN : value_at(i, j, seed);
        }
    }
}

static int guards_ok(DlMatrix v)
{
    for (size_t x = 0; x < v.count; ++x) {
        int logical = 0;
        if (x >= v.offset) {
            const size_t d = x - v.offset;
            logical = d / v.ld < v.rows && d % v.ld < v.cols;
        }
        if (!logical && v.base[x] != -12345.0f) {
            fprintf(stderr, "guard changed at allocation element %zu\n", x);
            return 0;
        }
    }
    return 1;
}

static int inputs_ok(DlMatrix v, unsigned seed)
{
    if (!guards_ok(v)) return 0;
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (v.base[v.offset + i * v.ld + j] != value_at(i, j, seed)) {
                fputs("input modified\n", stderr);
                return 0;
            }
        }
    }
    return 1;
}

static int one_case(size_t m, size_t n, size_t k,
                    DlTranspose ta, DlTranspose tb, DlWriteMode mode,
                    double *maximum_error)
{
    DlTensor owners[4] = {{0}};
    DlMatrix v[4] = {{0}};
    int ok = 0;
    if (!padded(&owners[0], &v[0], ta == DL_TRANS ? k : m,
                                      ta == DL_TRANS ? m : k) ||
        !padded(&owners[1], &v[1], tb == DL_TRANS ? n : k,
                                      tb == DL_TRANS ? k : n) ||
        !padded(&owners[2], &v[2], m, n) ||
        !padded(&owners[3], &v[3], m, n)) goto cleanup;
    fill(v[0], 1, 0);
    fill(v[1], 3, 0);
    fill(v[2], 5, mode == DL_SET);
    fill(v[3], 5, mode == DL_SET);
    for (int backend = 0; backend < 2; ++backend) {
        if (!require_status(dl_matmul((DlBackend)backend,
                dl_matrix_read(v[0]), ta, dl_matrix_read(v[1]), tb,
                v[2 + backend], mode), DL_OK, "product")) goto cleanup;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double expected = (double)v[2].base[v[2].offset + i*v[2].ld + j];
            const double actual = (double)v[3].base[v[3].offset + i*v[3].ld + j];
            const double error = fabs(actual - expected);
            const double tolerance = 1e-6 + 2e-6 * fabs(expected);
            if (!isfinite(expected) || !isfinite(actual) || error > tolerance) {
                fprintf(stderr, "M=%zu N=%zu K=%zu ta=%d tb=%d mode=%d "
                        "at [%zu,%zu]: %.9g vs %.9g\n", m, n, k,
                        (int)ta, (int)tb, (int)mode, i, j, actual, expected);
                goto cleanup;
            }
            if (error > *maximum_error) *maximum_error = error;
        }
    }
    ok = inputs_ok(v[0], 1) && inputs_ok(v[1], 3)
        && guards_ok(v[2]) && guards_ok(v[3]);
cleanup:
    for (size_t i = 0; i < 4; ++i) dl_tensor_destroy(&owners[i]);
    return ok;
}

/* A double partial sum may exceed FLT_MAX and cancel before the final store. */
static int reduction_checks(void)
{
    float a[130] = {0}, b[130] = {0}, out = 0.0f;
    const DlConstMatrix av = {a, 130, 0, 1, 130, 130};
    const DlConstMatrix bv = {b, 130, 0, 130, 1, 1};
    const DlMatrix cv = {&out, 1, 0, 1, 1, 1};
    for (size_t p = 0; p < 130; ++p) b[p] = 2.0f;
    a[0] = FLT_MAX;
    a[64] = -FLT_MAX;
    a[129] = 3.0f;
    for (int backend = 0; backend < 2; ++backend) {
        out = 5.0f;
        if (!require_status(dl_matmul((DlBackend)backend, av, DL_NOTRANS,
                bv, DL_NOTRANS, cv, DL_ADD), DL_OK, "wide partial") ||
            out != 11.0f) {
            fputs("double partial or ADD-once check failed\n", stderr);
            return 0;
        }
    }
    for (size_t p = 0; p < 130; ++p) {
        a[p] = 0.0f;
        b[p] = 1.0f;
    }
    a[0] = 16777216.0f;
    a[1] = 1.0f;
    a[64] = -16777216.0f;
    for (int backend = 0; backend < 2; ++backend) {
        out = 5.0f;
        if (!require_status(dl_matmul((DlBackend)backend, av, DL_NOTRANS,
                bv, DL_NOTRANS, cv, DL_ADD), DL_OK, "round once") ||
            out != 6.0f) {
            fputs("partial sum was rounded to float\n", stderr);
            return 0;
        }
    }
    return 1;
}

static int failure_checks(void)
{
    float a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1};
    float c[4] = {9, 9, 9, 9};
    const DlConstMatrix av = {a, 4, 0, 2, 2, 2};
    const DlConstMatrix bv = {b, 4, 0, 2, 2, 2};
    DlMatrix cv = {c, 4, 0, 2, 2, 2};
    if (!require_status(dl_matmul((DlBackend)99, av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_SET), DL_ARGUMENT, "backend")) return 0;
    if (!require_status(dl_matmul_blocked(av, (DlTranspose)99, bv,
            DL_NOTRANS, cv, DL_SET), DL_ARGUMENT, "transpose")) return 0;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, (DlWriteMode)99), DL_ARGUMENT, "mode")) return 0;
    DlMatrix bad = cv;
    bad.rows = 1;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, bad, DL_SET), DL_SHAPE, "shape")) return 0;
    bad = cv;
    bad.count = 3;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, bad, DL_SET), DL_RANGE, "capacity")) return 0;
    bad = cv;
    bad.base = a;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, bad, DL_SET), DL_ALIAS, "alias")) return 0;
    for (size_t i = 0; i < 4; ++i) {
        if (c[i] != 9.0f || a[i] != (float)(i + 1)) {
            fputs("structural error wrote output\n", stderr);
            return 0;
        }
    }
    /* A and B are allowed to be the same input view. */
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, av,
            DL_NOTRANS, cv, DL_SET), DL_OK, "shared inputs")) return 0;
    if (c[0] != 7 || c[1] != 10 || c[2] != 15 || c[3] != 22) return 0;
    c[0] = NAN;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_ADD), DL_NUMERIC, "nonfinite old C")) return 0;
    a[0] = INFINITY;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_SET), DL_NUMERIC, "nonfinite input")) return 0;
    a[0] = FLT_MAX;
    b[0] = 2;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_SET), DL_NUMERIC, "final float overflow")) return 0;
    return 1;
}

/* Do not depend on a row-major prefix after a numerical failure. */
static int failure_order_check(void)
{
    float a[2] = {1.0f, FLT_MAX}, b[33], scalar[66], blocked[66];
    for (size_t j = 0; j < 33; ++j) b[j] = j == 0 ? 2.0f : 1.0f;
    for (size_t x = 0; x < 66; ++x) scalar[x] = blocked[x] = -17.0f;
    const DlConstMatrix av = {a, 2, 0, 2, 1, 1};
    const DlConstMatrix bv = {b, 33, 0, 1, 33, 33};
    const DlMatrix sr = {scalar, 66, 0, 2, 33, 33};
    const DlMatrix br = {blocked, 66, 0, 2, 33, 33};
    if (!require_status(dl_matmul_ref(av, DL_NOTRANS, bv, DL_NOTRANS,
            sr, DL_SET), DL_NUMERIC, "scalar partial result") ||
        !require_status(dl_matmul_blocked(av, DL_NOTRANS, bv, DL_NOTRANS,
            br, DL_SET), DL_NUMERIC, "blocked partial result")) return 0;
    /* With 8-by-32 output tiles, row 0's last cell has not been committed. */
    if (scalar[32] != 1.0f || blocked[32] != -17.0f) {
        fputs("unexpected failure traversal for the default tile sizes\n", stderr);
        return 0;
    }
    return 1;
}

int main(void)
{
    const size_t shapes[][3] = {
        {1, 1, 1}, {7, 31, 63}, {8, 32, 64}, {9, 33, 65},
        {17, 35, 129}, {1, 257, 129}, {129, 33, 1}
    };
    size_t cases = 0;
    double maximum_error = 0.0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s) {
        for (int ta = 0; ta < 2; ++ta) {
            for (int tb = 0; tb < 2; ++tb) {
                for (int mode = 0; mode < 2; ++mode) {
                    if (!one_case(shapes[s][0], shapes[s][1], shapes[s][2],
                            (DlTranspose)ta, (DlTranspose)tb, (DlWriteMode)mode,
                            &maximum_error)) return EXIT_FAILURE;
                    ++cases;
                }
            }
        }
    }
    if (!reduction_checks() || !failure_checks() || !failure_order_check()) {
        return EXIT_FAILURE;
    }
    printf("blocked: %zu padded orientation/mode cases passed; max_abs=%.9g\n",
           cases, maximum_error);
    puts("blocked: reduction precision and failure checks passed");
    return EXIT_SUCCESS;
}
