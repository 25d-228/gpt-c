#include "dl/kernels.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int expect(int condition, const char *name)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        return 0;
    }
    return 1;
}

static int status_is(DlStatus actual, DlStatus wanted, const char *name)
{
    if (actual != wanted) {
        fprintf(stderr, "FAIL: %s: got %s, wanted %s\n", name,
                dl_status_name(actual), dl_status_name(wanted));
        return 0;
    }
    return 1;
}

static int close_finite(double actual, double expected)
{
    if (!isfinite(actual) || !isfinite(expected)) {
        return 0;
    }
    const double error = fabs(actual - expected);
    const double bound = 1e-6 + 1e-6 * fmax(fabs(actual), fabs(expected));
    return isfinite(error) && isfinite(bound) && error <= bound;
}

/* Test helpers borrow valid views and sufficiently long fixture arrays. */
static void fill_owner(DlTensor *owner, float value)
{
    for (size_t i = 0; i < owner->count; ++i) {
        owner->data[i] = value;
    }
}

static void write_matrix(DlMatrix matrix, const float *values)
{
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            matrix.base[matrix.offset + i * matrix.ld + j] =
                values[i * matrix.cols + j];
        }
    }
}

static int matrix_is(DlMatrix matrix, const float *values, const char *name)
{
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            const float actual = matrix.base[matrix.offset + i * matrix.ld + j];
            const float wanted = values[i * matrix.cols + j];
            if (!close_finite((double)actual, (double)wanted)) {
                fprintf(stderr, "FAIL: %s [%zu,%zu]: %.9g != %.9g\n",
                        name, i, j, (double)actual, (double)wanted);
                return 0;
            }
        }
    }
    return 1;
}

static int check_storage(void)
{
    DlTensor tensor = {0};
    DlMatrix rows = {0};
    size_t result = 17;
    int ok = 0;
    const size_t shape[3] = {2, 3, 4};
    const size_t bad_shape[2] = {2, 0};
    const size_t huge_shape[2] = {SIZE_MAX, 2};
    const size_t huge_bytes[1] = {SIZE_MAX / sizeof(float) + 1};
    if (!status_is(dl_size_mul(SIZE_MAX, 2, &result), DL_OVERFLOW,
                   "element multiplication overflow") ||
        !expect(result == 17, "failed multiply keeps result") ||
        !status_is(dl_size_add(SIZE_MAX, 1, &result), DL_OVERFLOW,
                   "element addition overflow") ||
        !expect(result == 17, "failed add keeps result") ||
        !status_is(dl_size_mul(0, SIZE_MAX, &result), DL_OK, "zero product") ||
        !expect(result == 0, "zero product value") ||
        !status_is(dl_tensor_create(&tensor, 0, shape), DL_ARGUMENT, "zero rank") ||
        !status_is(dl_tensor_create(&tensor, 2, bad_shape), DL_ARGUMENT,
                   "zero extent rejected") ||
        !status_is(dl_tensor_create(&tensor, 2, huge_shape), DL_OVERFLOW,
                   "shape product overflow") ||
        !status_is(dl_tensor_create(&tensor, 1, huge_bytes), DL_OVERFLOW,
                   "byte product overflow") ||
        !expect(tensor.data == NULL && tensor.count == 0 && tensor.rank == 0,
                "failed constructors keep empty owner") ||
        !status_is(dl_tensor_create(&tensor, 3, shape), DL_OK, "dense tensor")) {
        goto done;
    }
    for (size_t i = 0; i < tensor.count; ++i) {
        tensor.data[i] = (float)i;
    }
    float *const original = tensor.data;
    if (!expect(tensor.count == 24 && tensor.stride[0] == 12 &&
                tensor.stride[1] == 4 && tensor.stride[2] == 1,
                "dense strides") ||
        !status_is(dl_tensor_create(&tensor, 3, shape), DL_STATE,
                   "cannot reconstruct a live owner") ||
        !expect(tensor.data == original, "live owner preserved") ||
        !status_is(dl_tensor_rows(&tensor, &rows), DL_OK, "flatten rows") ||
        !expect(rows.base == tensor.data && rows.rows == 6 &&
                rows.cols == 4 && rows.ld == 4, "flatten does not copy") ||
        !expect(rows.base[rows.offset + 5 * rows.ld + 3] == 23.0f,
                "[1,2,3] and row [5,3] share offset 23")) {
        goto done;
    }
    printf("layout: [2,3,4] strides=[12,4,1]; row[5,3]=23\n");
    ok = 1;
done:
    dl_tensor_destroy(&tensor);
    dl_tensor_destroy(&tensor);
    return ok && expect(tensor.data == NULL && tensor.count == 0 &&
                        tensor.rank == 0, "destroy resets owner");
}

static int check_projection(void)
{
    DlTensor x_owner = {0}, w_owner = {0}, y_owner = {0};
    DlTensor g_owner = {0}, dx_owner = {0}, dw_owner = {0};
    DlMatrix x = {0}, w = {0}, y = {0}, g = {0}, dx = {0}, dw = {0};
    const size_t shape23[2] = {2, 3};
    const size_t shape22[2] = {2, 2};
    const size_t guard_shape[1] = {9};
    const float xv[6] = {1, 2, 3, 4, 5, 6};
    const float wv[6] = {1, 0, -1, 2, 1, 0};
    const float yv[4] = {-2, 4, -2, 13};
    const float twice_y[4] = {-4, 8, -4, 26};
    const float gv[4] = {1, 2, 3, 4};
    const float dxv[6] = {5, 2, -1, 11, 4, -3};
    const float dwv[6] = {13, 17, 21, 18, 24, 30};
    int ok = 0;
    if (!status_is(dl_tensor_create(&x_owner, 2, shape23), DL_OK, "X owner") ||
        !status_is(dl_tensor_create(&w_owner, 2, shape23), DL_OK, "W owner") ||
        !status_is(dl_tensor_create(&y_owner, 1, guard_shape), DL_OK, "Y guards") ||
        !status_is(dl_tensor_create(&g_owner, 2, shape22), DL_OK, "G owner") ||
        !status_is(dl_tensor_create(&dx_owner, 2, shape23), DL_OK, "dX owner") ||
        !status_is(dl_tensor_create(&dw_owner, 2, shape23), DL_OK, "dW owner") ||
        !status_is(dl_tensor_rows(&x_owner, &x), DL_OK, "X view") ||
        !status_is(dl_tensor_rows(&w_owner, &w), DL_OK, "W view") ||
        !status_is(dl_matrix_view(&y_owner, 1, 2, 2, 4, &y), DL_OK, "Y view") ||
        !status_is(dl_tensor_rows(&g_owner, &g), DL_OK, "G view") ||
        !status_is(dl_tensor_rows(&dx_owner, &dx), DL_OK, "dX view") ||
        !status_is(dl_tensor_rows(&dw_owner, &dw), DL_OK, "dW view")) {
        goto done;
    }
    write_matrix(x, xv);
    write_matrix(w, wv);
    write_matrix(g, gv);
    fill_owner(&y_owner, 777.0f);
    /* SET must ignore old logical output even when it contains NaNs. */
    const float poison[4] = {NAN, NAN, NAN, NAN};
    write_matrix(y, poison);
    if (!status_is(dl_matmul_ref(dl_matrix_read(x), DL_NOTRANS,
                                 dl_matrix_read(w), DL_TRANS, y, DL_SET),
                   DL_OK, "X W^T") || !matrix_is(y, yv, "projection") ||
        !status_is(dl_matmul_ref(dl_matrix_read(x), DL_NOTRANS,
                                 dl_matrix_read(w), DL_TRANS, y, DL_ADD),
                   DL_OK, "accumulate projection") ||
        !matrix_is(y, twice_y, "two contributions") ||
        !status_is(dl_matmul_ref(dl_matrix_read(x), DL_NOTRANS,
                                 dl_matrix_read(w), DL_TRANS, y, DL_SET),
                   DL_OK, "overwrite again") || !matrix_is(y, yv, "reset") ||
        !status_is(dl_matmul_ref(dl_matrix_read(g), DL_NOTRANS,
                                 dl_matrix_read(w), DL_NOTRANS, dx, DL_SET),
                   DL_OK, "G W") || !matrix_is(dx, dxv, "input-side product") ||
        !status_is(dl_matmul_ref(dl_matrix_read(g), DL_TRANS,
                                 dl_matrix_read(x), DL_NOTRANS, dw, DL_SET),
                   DL_OK, "G^T X") || !matrix_is(dw, dwv, "weight-side product")) {
        goto done;
    }
    const size_t guards[5] = {0, 3, 4, 7, 8};
    for (size_t i = 0; i < 5; ++i) {
        if (!expect(y_owner.data[guards[i]] == 777.0f, "output guard unchanged")) {
            goto done;
        }
    }
    printf("projection: [-2,4; -2,13]; SET/ADD and guards pass\n");
    printf("products: G W=[5,2,-1; 11,4,-3]; G^T X=[13,17,21; 18,24,30]\n");
    ok = 1;
done:
    dl_tensor_destroy(&dw_owner);
    dl_tensor_destroy(&dx_owner);
    dl_tensor_destroy(&g_owner);
    dl_tensor_destroy(&y_owner);
    dl_tensor_destroy(&w_owner);
    dl_tensor_destroy(&x_owner);
    return ok;
}

static int check_transposes(void)
{
    DlTensor arena = {0};
    const size_t shape[1] = {128};
    const float av[6] = {1, 2, 3, 4, 5, 6};
    const float bv[12] = {1, 0, -1, 2, 0, 1, 2, -1, 2, 1, 0, 1};
    const float wanted[8] = {7, 5, 3, 3, 16, 11, 6, 9};
    int ok = 0;
    if (!status_is(dl_tensor_create(&arena, 1, shape), DL_OK, "shared storage")) {
        goto done;
    }
    const DlTranspose modes[2] = {DL_NOTRANS, DL_TRANS};
    for (size_t ia = 0; ia < 2; ++ia) {
        for (size_t ib = 0; ib < 2; ++ib) {
            DlMatrix a = {0}, b = {0}, c = {0};
            const size_t ar = ia == 0 ? 2 : 3, ac = ia == 0 ? 3 : 2;
            const size_t br = ib == 0 ? 3 : 4, bc = ib == 0 ? 4 : 3;
            fill_owner(&arena, 777.0f);
            if (!status_is(dl_matrix_view(&arena, 1, ar, ac, ac + 2, &a),
                           DL_OK, "padded A") ||
                !status_is(dl_matrix_view(&arena, 33, br, bc, bc + 2, &b),
                           DL_OK, "padded B") ||
                !status_is(dl_matrix_view(&arena, 81, 2, 4, 6, &c),
                           DL_OK, "padded C")) {
                goto done;
            }
            for (size_t i = 0; i < 2; ++i) {
                for (size_t p = 0; p < 3; ++p) {
                    const size_t index = ia == 0 ? i * a.ld + p : p * a.ld + i;
                    a.base[a.offset + index] = av[i * 3 + p];
                }
            }
            for (size_t p = 0; p < 3; ++p) {
                for (size_t j = 0; j < 4; ++j) {
                    const size_t index = ib == 0 ? p * b.ld + j : j * b.ld + p;
                    b.base[b.offset + index] = bv[p * 4 + j];
                }
            }
            float before[128];
            for (size_t i = 0; i < 128; ++i) {
                before[i] = arena.data[i];
            }
            if (!status_is(dl_matmul_ref(dl_matrix_read(a), modes[ia],
                                         dl_matrix_read(b), modes[ib], c, DL_SET),
                           DL_OK, "transpose combination") ||
                !matrix_is(c, wanted, "non-square product")) {
                goto done;
            }
            for (size_t i = 0; i < 128; ++i) {
                const int is_output = (i >= 81 && i < 85) || (i >= 87 && i < 91);
                if (!is_output && !expect(arena.data[i] == before[i],
                                          "inputs and every gap preserved")) {
                    goto done;
                }
            }
        }
    }
    /* Both read-only operands may be the same view. */
    DlMatrix a = {0}, c = {0};
    const float gram[4] = {14, 32, 32, 77};
    if (!status_is(dl_matrix_view(&arena, 1, 2, 3, 5, &a), DL_OK, "shared input") ||
        !status_is(dl_matrix_view(&arena, 81, 2, 2, 6, &c), DL_OK, "Gram output")) {
        goto done;
    }
    write_matrix(a, av);
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(a), DL_TRANS, c, DL_SET),
                   DL_OK, "overlapping inputs are legal") ||
        !matrix_is(c, gram, "A A^T")) {
        goto done;
    }
    puts("transpose: all four padded non-square cases and shared inputs pass");
    ok = 1;
done:
    dl_tensor_destroy(&arena);
    return ok;
}

static int check_rejections(void)
{
    DlTensor a_owner = {0}, b_owner = {0}, c_owner = {0};
    DlMatrix a = {0}, b = {0}, c = {0};
    const size_t shape23[2] = {2, 3}, shape22[2] = {2, 2};
    int ok = 0;
    if (!status_is(dl_tensor_create(&a_owner, 2, shape23), DL_OK, "reject A") ||
        !status_is(dl_tensor_create(&b_owner, 2, shape23), DL_OK, "reject B") ||
        !status_is(dl_tensor_create(&c_owner, 2, shape22), DL_OK, "reject C") ||
        !status_is(dl_tensor_rows(&a_owner, &a), DL_OK, "reject A view") ||
        !status_is(dl_tensor_rows(&b_owner, &b), DL_OK, "reject B view") ||
        !status_is(dl_tensor_rows(&c_owner, &c), DL_OK, "reject C view")) {
        goto done;
    }
    fill_owner(&a_owner, 1.0f);
    fill_owner(&b_owner, 2.0f);
    fill_owner(&c_owner, 19.0f);
    DlMatrix saved = c;
    if (!status_is(dl_matrix_view(&a_owner, 0, 2, 3, 2, &c), DL_ARGUMENT,
                   "leading dimension too small") ||
        !status_is(dl_matrix_view(&a_owner, 0, 0, 3, 3, &c), DL_ARGUMENT,
                   "empty matrix rejected") ||
        !status_is(dl_matrix_view(&a_owner, 1, 2, 3, 3, &c), DL_RANGE,
                   "last cell exceeds allocation") ||
        !status_is(dl_matrix_view(&a_owner, SIZE_MAX, 1, 1, 1, &c), DL_RANGE,
                   "invalid starting offset") ||
        !status_is(dl_matrix_view(&a_owner, 0, SIZE_MAX, 1, 3, &c), DL_OVERFLOW,
                   "span multiplication overflow") ||
        !expect(c.base == saved.base && c.count == saved.count &&
                c.offset == saved.offset && c.rows == saved.rows &&
                c.cols == saved.cols && c.ld == saved.ld,
                "failed view leaves every field unchanged") ||
        !status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_SHAPE, "wrong inner dimension") ||
        !status_is(dl_matmul_ref(dl_matrix_read(a), (DlTranspose)7,
                                 dl_matrix_read(b), DL_TRANS, c, DL_SET),
                   DL_ARGUMENT, "invalid transpose flag")) {
        goto done;
    }
    const float nineteen[4] = {19, 19, 19, 19};
    if (!matrix_is(c, nineteen, "structural failures do not write")) {
        goto done;
    }
    DlMatrix alias = {0};
    if (!status_is(dl_matrix_view(&a_owner, 0, 2, 2, 3, &alias), DL_OK,
                   "overlapping output view") ||
        !status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_TRANS, alias, DL_SET),
                   DL_ALIAS, "reject destructive output overlap")) {
        goto done;
    }
    /* A uses offsets 0,3; C uses 1,4. Their enclosing spans still intersect. */
    DlMatrix narrow = {0}, scalar = {0}, gap_output = {0};
    if (!status_is(dl_matrix_view(&a_owner, 0, 2, 1, 3, &narrow), DL_OK, "narrow") ||
        !status_is(dl_matrix_view(&b_owner, 0, 1, 1, 1, &scalar), DL_OK, "scalar") ||
        !status_is(dl_matrix_view(&a_owner, 1, 2, 1, 3, &gap_output), DL_OK, "gap") ||
        !status_is(dl_matmul_ref(dl_matrix_read(narrow), DL_NOTRANS,
                                 dl_matrix_read(scalar), DL_NOTRANS,
                                 gap_output, DL_SET), DL_ALIAS,
                   "conservative overlap includes row gaps")) {
        goto done;
    }
    for (size_t i = 0; i < a_owner.count; ++i) {
        if (!expect(a_owner.data[i] == 1.0f, "alias rejection keeps input")) {
            goto done;
        }
    }
    puts("rejections: invalid sizes, shapes, flags, and output overlap pass");
    ok = 1;
done:
    dl_tensor_destroy(&c_owner);
    dl_tensor_destroy(&b_owner);
    dl_tensor_destroy(&a_owner);
    return ok;
}

static int check_numerics(void)
{
    DlTensor a_owner = {0}, b_owner = {0}, c_owner = {0};
    DlMatrix a = {0}, b = {0}, c = {0};
    const size_t shape[1] = {3};
    int ok = 0;
    if (!status_is(dl_tensor_create(&a_owner, 1, shape), DL_OK, "numeric A") ||
        !status_is(dl_tensor_create(&b_owner, 1, shape), DL_OK, "numeric B") ||
        !status_is(dl_tensor_create(&c_owner, 1, shape), DL_OK, "numeric C") ||
        !status_is(dl_matrix_view(&a_owner, 0, 1, 1, 1, &a), DL_OK, "scalar A") ||
        !status_is(dl_matrix_view(&b_owner, 0, 1, 2, 2, &b), DL_OK, "two B cells") ||
        !status_is(dl_matrix_view(&c_owner, 0, 1, 2, 2, &c), DL_OK, "two C cells")) {
        goto done;
    }
    a_owner.data[0] = FLT_MAX;
    b_owner.data[0] = 0.0f;
    b_owner.data[1] = 2.0f;
    fill_owner(&c_owner, -9.0f);
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_NUMERIC, "finite double result too large for float") ||
        !expect(c_owner.data[0] == 0.0f && c_owner.data[1] == -9.0f &&
                c_owner.data[2] == -9.0f, "numeric error may leave a written prefix")) {
        goto done;
    }
    a_owner.data[0] = 1.0f;
    b_owner.data[0] = NAN;
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_NUMERIC, "non-finite operand")) {
        goto done;
    }
    if (!status_is(dl_matrix_view(&a_owner, 0, 1, 3, 3, &a), DL_OK, "wide dot A") ||
        !status_is(dl_matrix_view(&b_owner, 0, 3, 1, 1, &b), DL_OK, "wide dot B") ||
        !status_is(dl_matrix_view(&c_owner, 0, 1, 1, 1, &c), DL_OK, "wide dot C")) {
        goto done;
    }
    a_owner.data[0] = 16777216.0f;
    a_owner.data[1] = 1.0f;
    a_owner.data[2] = -16777216.0f;
    fill_owner(&b_owner, 1.0f);
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_OK, "double dot accumulation") ||
        !expect(c_owner.data[0] == 1.0f, "small term survives the double reduction")) {
        goto done;
    }
    c_owner.data[0] = NAN;
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_ADD),
                   DL_NUMERIC, "ADD cannot use a non-finite initial output")) {
        goto done;
    }
    puts("numerics: wide reduction, non-finite rejection, and partial failure pass");
    ok = 1;
done:
    dl_tensor_destroy(&c_owner);
    dl_tensor_destroy(&b_owner);
    dl_tensor_destroy(&a_owner);
    return ok;
}

int main(void)
{
    if (!expect(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53,
                "binary32/binary64 fixture precision") ||
        !check_storage() || !check_projection() || !check_transposes() ||
        !check_rejections() || !check_numerics()) {
        return EXIT_FAILURE;
    }
    puts("all kernel checks passed");
    return EXIT_SUCCESS;
}
