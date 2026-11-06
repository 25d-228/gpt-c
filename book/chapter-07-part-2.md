# Chapter 7, Part 2 — Checking normalization and residual graphs

This is the second half of [Chapter 7 — RMSNorm and residual gradient paths](https://25d-228.github.io/gpt-c/chapter-07.html). The first half derived and implemented the operators in `include/dl/norm.h` and `src/norm.c`. This half supplies every function in `tests/test_ops.c`, the updated Makefile, a complete learned residual branch, and numerical fault experiments.

**Prerequisites:** the complete Chapter 6 workspace plus Part 1's two new files. Keep all previous files. The code blocks named `tests/test_ops.c` concatenate in order into that single new executable; the Makefile block replaces the root file. No other source changes are required. The matching [test source](https://25d-228.github.io/gpt-c/media/chapter-07/reference/tests/test_ops.c) and [Makefile](https://25d-228.github.io/gpt-c/media/chapter-07/reference/Makefile) are included in the cumulative companion state.

**Implementation estimate:** shared with Part 1: 3.00 hours total, including 1.00 for traces and checks. There is no separate allocation for this page. The tests use the C operators directly and do not depend on a future decoder, optimizer, or external model library.

## Build executable checks with visible fixtures

Create `tests/test_ops.c`. It is a complete executable, not a collection of snippets that require a future model driver. The small test helpers below do not allocate memory. Their direct matrix construction is restricted to the fixed tiny arrays used in this file; production allocations still use the checked owner/view constructors.

`matrix` describes dense storage. `padded` describes a fixed 40-float array with an initial offset and a leading dimension larger than the logical width. Every unused element begins at `-7777`. `guards` checks that operators leave those elements unchanged.

```c title="tests/test_ops.c"
#include "dl/norm.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got != want) {
        fprintf(stderr, "%s: %s; expected %s\n", where,
                dl_status_name(got), dl_status_name(want));
        return 0;
    }
    return 1;
}

static int near_value(double got, double want, double tolerance, const char *where)
{
    if (!isfinite(got) || !isfinite(want) || fabs(got - want) > tolerance) {
        fprintf(stderr, "%s: %.12g; expected %.12g, tolerance %.3g\n",
                where, got, want, tolerance);
        return 0;
    }
    return 1;
}

/* Test-only: caller supplies exactly rows*cols live floats; shapes are tiny. */
static DlMatrix matrix(float *p, size_t rows, size_t cols)
{
    return (DlMatrix){p, rows * cols, 0, rows, cols, cols};
}

static void fill(DlMatrix v, float value)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            v.base[v.offset + i * v.ld + j] = value;
        }
    }
}

static void load(DlMatrix v, const float *values)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            v.base[v.offset + i * v.ld + j] = values[i * v.cols + j];
        }
    }
}

static int expect(DlMatrix v, const float *values, const char *where)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (!near_value(v.base[v.offset + i * v.ld + j],
                    values[i * v.cols + j], 2e-6, where)) return 0;
        }
    }
    return 1;
}

/* Fixed fixtures below have rows <= 2 and cols <= 3, well inside 40 floats. */
static DlMatrix padded(float *p, size_t rows, size_t cols)
{
    for (size_t i = 0; i < 40; ++i) p[i] = -7777.0f;
    return (DlMatrix){p, 40, 2, rows, cols, cols + 2};
}

static int guards(DlMatrix v)
{
    for (size_t i = 0; i < v.count; ++i) {
        const int active = i >= v.offset && (i - v.offset) / v.ld < v.rows
                         && (i - v.offset) % v.ld < v.cols;
        if (!active && v.base[i] != -7777.0f) {
            fprintf(stderr, "changed padding at %zu\n", i);
            return 0;
        }
    }
    return 1;
}

```

`expect` compares against a dense expected array even when the actual view is padded. It first rejects non-finite values through `near_value`; a NaN cannot accidentally pass because a comparison with its error was false. The very small absolute tolerance is appropriate to the hand fixtures, whose expected values are binary fractions.

The array-size multiplication inside `matrix` is safe for these literal small test dimensions. It does not replace `dl_size_mul` in an API accepting external dimensions. Likewise, `load` receives a fixture with exactly the stated number of logical cells; it is not a general file loader.

### Check row sums, all mode combinations, and shared residual edges

Append `hand_and_modes`. The seven separate arrays hold input, scale, incoming gradient, normalized output, input gradient, scale gradient, and residual output. Scale arrays have one logical row; activations have two.

```c title="tests/test_ops.c"
static int hand_and_modes(void)
{
    float storage[7][40];
    DlMatrix v[7];
    for (size_t i = 0; i < 7; ++i) {
        v[i] = padded(storage[i], (i == 1 || i == 5) ? 1 : 2, 2);
    }
    const DlMatrix x = v[0], scale = v[1], dy = v[2], y = v[3];
    const DlMatrix dx = v[4], ds = v[5], residual = v[6];
    const float xv[4] = {3, 4, -3, 4}, sv[2] = {2, -1};
    const float gv[4] = {1, -2, 0.5f, 1};
    const float yv[4] = {1.5f, -1, -1.5f, -1};
    const float dxv[4] = {11.0f/64, 1.0f/16, 11.0f/128, -1.0f/32};
    const float dsv[2] = {0.375f, -1};
    load(x, xv); load(scale, sv); load(dy, gv); fill(y, NAN);
    if (!status_is(dl_rmsnorm_forward(dl_matrix_read(x), dl_matrix_read(scale),
            3.5f, y), DL_OK, "hand forward") || !expect(y, yv, "hand y")) return 0;
    for (int xm = 0; xm < 2; ++xm) {
        for (int sm = 0; sm < 2; ++sm) {
            const DlWriteMode xmode = xm ? DL_ADD : DL_SET;
            const DlWriteMode smode = sm ? DL_ADD : DL_SET;
            fill(dx, xm ? 3.0f : NAN); fill(ds, sm ? 5.0f : NAN);
            if (!status_is(dl_rmsnorm_backward(dl_matrix_read(x),
                    dl_matrix_read(scale), dl_matrix_read(dy), 3.5f,
                    dx, xmode, ds, smode), DL_OK, "independent modes")) return 0;
            float wantx[4], wants[2];
            for (size_t j = 0; j < 4; ++j) wantx[j] = dxv[j] + (xm ? 3.0f : 0.0f);
            for (size_t j = 0; j < 2; ++j) wants[j] = dsv[j] + (sm ? 5.0f : 0.0f);
            if (!expect(dx, wantx, "dx mode") || !expect(ds, wants, "scale mode")) return 0;
        }
    }
    if (!status_is(dl_rmsnorm_backward(dl_matrix_read(x), dl_matrix_read(scale),
            dl_matrix_read(dy), 3.5f, dx, DL_ADD, ds, DL_ADD), DL_OK,
            "second contributor")) return 0;
    float twice_x[4], twice_s[2];
    for (size_t j = 0; j < 4; ++j) twice_x[j] = 3.0f + 2.0f * dxv[j];
    for (size_t j = 0; j < 2; ++j) twice_s[j] = 5.0f + 2.0f * dsv[j];
    if (!expect(dx, twice_x, "two dx contributions") ||
        !expect(ds, twice_s, "two scale contributions")) return 0;
    const float rv[4] = {4.5f, 3, -4.5f, 3};
    if (!status_is(dl_residual_add(dl_matrix_read(x), dl_matrix_read(y), residual),
            DL_OK, "residual forward") || !expect(residual, rv, "residual y")) return 0;
    if (!status_is(dl_residual_backward(dl_matrix_read(dy), dx, DL_SET),
            DL_OK, "identity path") ||
        !status_is(dl_rmsnorm_backward(dl_matrix_read(x), dl_matrix_read(scale),
            dl_matrix_read(dy), 3.5f, dx, DL_ADD, ds, DL_SET),
            DL_OK, "learned path")) return 0;
    float merged[4];
    for (size_t j = 0; j < 4; ++j) merged[j] = gv[j] + dxv[j];
    if (!expect(dx, merged, "residual gradient sum")) return 0;
    printf("hand row: y=[%.6f, %.6f], norm dx=[%.6f, %.6f]\n",
            (double)yv[0], (double)yv[1], (double)dxv[0], (double)dxv[1]);
    printf("residual row: dx=[%.6f, %.6f]\n", (double)merged[0], (double)merged[1]);
    /* Two operand edges can lead to one logical variable. */
    if (!status_is(dl_residual_add(dl_matrix_read(x), dl_matrix_read(x), residual),
            DL_OK, "shared residual inputs") ||
        !status_is(dl_residual_backward(dl_matrix_read(dy), dx, DL_SET),
            DL_OK, "first edge") ||
        !status_is(dl_residual_backward(dl_matrix_read(dy), dx, DL_ADD),
            DL_OK, "second edge")) return 0;
    for (size_t j = 0; j < 4; ++j) merged[j] = 2.0f * gv[j];
    if (!expect(dx, merged, "x+x derivative") || !expect(x, xv, "unchanged x") ||
        !expect(scale, sv, "unchanged scale") || !expect(dy, gv, "unchanged dy")) return 0;
    for (size_t i = 0; i < 7; ++i) if (!guards(v[i])) return 0;
    return 1;
}

```

The two nested mode loops cover all four combinations, not only SET/SET and ADD/ADD. SET destinations begin as NaNs; ADD destinations begin at three for `dx` and five for `dscale`. The expected arrays distinguish old contributions from new ones.

After the four combinations, both destinations contain their nonzero starting contribution plus one derivative. Calling backward again with ADD/ADD must add a second derivative, not reset either destination. This checks two callers contributing to the same gradient buffers without introducing a training loop.

The residual sequence then copies the identity contribution to `dx` and calls RMSNorm backward with `dx=ADD`, `dscale=SET`. The final `x+x` case reads the same activation view twice but writes a separate result; its two reverse edges add to the same `dx`. Guard and unchanged-input checks cover every padded array at the end.

### A zero output does not imply a zero input gradient

The next function checks zero rows, zero scales, an input whose float square would overflow, and width one. With epsilon $10^{-4}$, the zero-row input gradient in this fixture is approximately $(200,200)$; the slight allowance reflects that the decimal epsilon is stored as float.

```c title="tests/test_ops.c"
static int boundary_values(void)
{
    float x[2] = {0, 0}, scale[2] = {2, -1}, g[2] = {1, -2};
    float y[2] = {NAN, NAN}, dx[2] = {NAN, NAN}, ds[2] = {NAN, NAN};
    const DlConstMatrix xv = dl_matrix_read(matrix(x, 1, 2));
    const DlConstMatrix sv = dl_matrix_read(matrix(scale, 1, 2));
    const DlConstMatrix gv = dl_matrix_read(matrix(g, 1, 2));
    if (!status_is(dl_rmsnorm_forward(xv, sv, 1e-4f, matrix(y, 1, 2)),
            DL_OK, "zero forward") ||
        !status_is(dl_rmsnorm_backward(xv, sv, gv, 1e-4f, matrix(dx, 1, 2),
            DL_SET, matrix(ds, 1, 2), DL_SET), DL_OK, "zero backward")) return 0;
    for (size_t j = 0; j < 2; ++j) {
        if (!near_value(y[j], 0, 0, "zero y") || !near_value(ds[j], 0, 0, "zero ds") ||
            !near_value(dx[j], 200, 3e-5, "zero dx is not zero")) return 0;
    }
    x[0] = 3; x[1] = 4; scale[0] = 0; scale[1] = 0;
    if (!status_is(dl_rmsnorm_backward(xv, sv, gv, 3.5f, matrix(dx, 1, 2),
            DL_SET, matrix(ds, 1, 2), DL_SET), DL_OK, "zero scale")) return 0;
    if (!near_value(dx[0], 0, 0, "zero scale dx") ||
        !near_value(dx[1], 0, 0, "zero scale dx") ||
        !near_value(ds[0], 0.75, 0, "zero scale ds") ||
        !near_value(ds[1], -2, 0, "zero scale ds")) return 0;
    /* Squaring in float would overflow before the double accumulation. */
    x[0] = FLT_MAX; x[1] = 0; scale[0] = 1; scale[1] = 1;
    if (!status_is(dl_rmsnorm_forward(xv, sv, 1e-5f, matrix(y, 1, 2)),
            DL_OK, "large finite input") ||
        !near_value(y[0], sqrt(2.0), 2e-7, "wide square") || y[1] != 0) return 0;
    float one_x = 2, one_scale = 3, one_g = 4, one_dx = NAN, one_ds = NAN;
    if (!status_is(dl_rmsnorm_backward(dl_matrix_read(matrix(&one_x, 1, 1)),
            dl_matrix_read(matrix(&one_scale, 1, 1)), dl_matrix_read(matrix(&one_g, 1, 1)),
            0.5f, matrix(&one_dx, 1, 1), DL_SET, matrix(&one_ds, 1, 1), DL_SET),
            DL_OK, "width one")) return 0;
    if (!near_value(one_dx, 6.0 / (4.5 * sqrt(4.5)), 1e-7, "epsilon radial term") ||
        !near_value(one_ds, 8.0 / sqrt(4.5), 2e-7, "width one scale")) return 0;
    puts("zero rows, zero scales, large inputs, and width-one checks passed");
    return 1;
}

```

The zero-scale case is different from the zero-input case. With zero scales, the input derivative vanishes here because the output has no dependence on the input at those scale values. The scale derivative still sees the nonzero normalized input. This is also a concrete reason not to reconstruct normalized coordinates by dividing output by scale.

The width-one check retains epsilon's radial derivative. It uses moderate values and a substantial epsilon, so cancellation is not severe enough to justify treating the result as numerically zero.

### Separate preflight rejection from arithmetic failure

The next function deliberately supplies malformed metadata and non-finite logical values while keeping the backing arrays live. The status checks run before the unchanged-output comparison. The final arena example proves that shared ownership is not itself an error: disjoint input/output spans in one allocation are permitted, and unused NaN padding is ignored.

```c title="tests/test_ops.c"
static int rejected_calls(void)
{
    float x[4] = {1, 2, 3, 4}, s[2] = {1, 1}, g[4] = {1, 2, 3, 4};
    float y[4] = {41, 41, 41, 41}, dx[4] = {43, 43, 43, 43}, ds[2] = {47, 47};
    const float ys[4] = {41, 41, 41, 41}, xs[4] = {43, 43, 43, 43}, ss[2] = {47, 47};
    const DlConstMatrix xv = dl_matrix_read(matrix(x, 2, 2));
    const DlConstMatrix sv = dl_matrix_read(matrix(s, 1, 2));
    const DlConstMatrix gv = dl_matrix_read(matrix(g, 2, 2));
    const DlMatrix yv = matrix(y, 2, 2), dv = matrix(dx, 2, 2), dsview = matrix(ds, 1, 2);
    const float bad_eps[4] = {0, -1, NAN, INFINITY};
    for (size_t i = 0; i < 4; ++i) {
        if (!status_is(dl_rmsnorm_forward(xv, sv, bad_eps[i], yv),
                DL_ARGUMENT, "bad epsilon")) return 0;
    }
    if (!status_is(dl_rmsnorm_backward(xv, sv, gv, 0.1f, dv, (DlWriteMode)99,
            dsview, DL_SET), DL_ARGUMENT, "bad dx mode") ||
        !status_is(dl_rmsnorm_backward(xv, sv, gv, 0.1f, dv, DL_SET,
            dsview, (DlWriteMode)99), DL_ARGUMENT, "bad scale mode")) return 0;
    DlMatrix bad = yv; bad.rows = 1;
    if (!status_is(dl_rmsnorm_forward(xv, sv, 0.1f, bad), DL_SHAPE, "bad shape")) return 0;
    bad = yv; bad.count = 3;
    if (!status_is(dl_rmsnorm_forward(xv, sv, 0.1f, bad), DL_RANGE, "short view")) return 0;
    bad = yv; bad.rows = SIZE_MAX;
    if (!status_is(dl_rmsnorm_forward(xv, sv, 0.1f, bad), DL_OVERFLOW, "size overflow")) return 0;
    if (!status_is(dl_rmsnorm_forward(xv, sv, 0.1f, matrix(x, 2, 2)),
            DL_ALIAS, "in-place norm") ||
        !status_is(dl_rmsnorm_backward(xv, sv, gv, 0.1f, dv, DL_SET,
            (DlMatrix){dx, 4, 0, 1, 2, 2}, DL_SET), DL_ALIAS, "two outputs overlap") ||
        !status_is(dl_residual_add(xv, gv, matrix(x, 2, 2)), DL_ALIAS, "in-place residual") ||
        !status_is(dl_residual_backward(gv, matrix(g, 2, 2), DL_SET),
            DL_ALIAS, "in-place reverse edge")) return 0;
    g[3] = NAN;
    if (!status_is(dl_rmsnorm_backward(xv, sv, gv, 0.1f, dv, DL_SET,
            dsview, DL_SET), DL_NUMERIC, "late nonfinite incoming") ||
        !status_is(dl_residual_add(xv, gv, yv), DL_NUMERIC, "late nonfinite residual")) return 0;
    g[3] = 4;
    s[1] = INFINITY;
    if (!status_is(dl_rmsnorm_forward(xv, sv, 0.1f, yv), DL_NUMERIC, "nonfinite scale")) return 0;
    s[1] = 1;
    if (memcmp(y, ys, sizeof y) || memcmp(dx, xs, sizeof dx) || memcmp(ds, ss, sizeof ds)) {
        fputs("preflight changed an output\n", stderr); return 0;
    }
    dx[3] = NAN;
    if (!status_is(dl_rmsnorm_backward(xv, sv, gv, 0.1f, dv, DL_ADD,
            dsview, DL_SET), DL_NUMERIC, "invalid old dx") ||
        dx[0] != 43 || !isnan(dx[3]) || memcmp(ds, ss, sizeof ds)) return 0;
    dx[3] = 43; ds[1] = NAN;
    if (!status_is(dl_rmsnorm_backward(xv, sv, gv, 0.1f, dv, DL_SET,
            dsview, DL_ADD), DL_NUMERIC, "invalid old scale") ||
        memcmp(dx, xs, sizeof dx) || ds[0] != 47 || !isnan(ds[1])) return 0;
    /* Same canonical allocation, but disjoint spans are permitted. */
    float arena[8] = {3, 4, NAN, NAN, 0, 0, NAN, NAN};
    const DlConstMatrix ax = {arena, 8, 0, 1, 2, 2};
    const DlMatrix ay = {arena, 8, 4, 1, 2, 2};
    if (!status_is(dl_rmsnorm_forward(ax, sv, 3.5f, ay), DL_OK, "disjoint arena") ||
        arena[4] != 0.75f || arena[5] != 1.0f || !isnan(arena[2])) return 0;
    puts("shape, alias, epsilon, and nonfinite preflight checks passed");
    return 1;
}

```

`memcmp` compares the exact bytes of initialized finite sentinel arrays; it is used here to verify that a rejected call made no writes. For intentionally introduced NaNs, the test checks `isnan` instead of demanding a particular NaN representation. No test depends on the content of uninitialized storage.

A late non-finite incoming gradient is caught during preflight, before the SET scale destination is cleared. An invalid old ADD destination is likewise rejected before the other output changes. These tests would catch moving those checks inside the writing loop.

Now append an arithmetic-failure fixture. All declared input values are finite, so preflight succeeds. A later result nevertheless exceeds the representable float range.

```c title="tests/test_ops.c"
static int partial_failures(void)
{
    float x[2] = {0, 2}, s[2] = {1, FLT_MAX}, g[2] = {1, 2};
    float y[2] = {9, 9}, dx[2] = {9, 9}, ds[2] = {9, 9};
    const DlConstMatrix xv = dl_matrix_read(matrix(x, 1, 2));
    const DlConstMatrix sv = dl_matrix_read(matrix(s, 1, 2));
    const DlConstMatrix gv = dl_matrix_read(matrix(g, 1, 2));
    if (!status_is(dl_rmsnorm_forward(xv, sv, 0.25f, matrix(y, 1, 2)),
            DL_NUMERIC, "forward range failure") || y[0] != 0 || y[1] != 9) return 0;
    x[1] = 0;
    if (!status_is(dl_rmsnorm_backward(xv, sv, gv, 0.25f, matrix(dx, 1, 2),
            DL_SET, matrix(ds, 1, 2), DL_SET), DL_NUMERIC, "backward range failure") ||
        dx[0] != 2 || dx[1] != 9 || ds[0] != 0 || ds[1] != 0) return 0;
    x[0] = 0; x[1] = FLT_MAX; g[0] = 1; g[1] = FLT_MAX;
    y[0] = 9; y[1] = 9;
    if (!status_is(dl_residual_add(xv, gv, matrix(y, 1, 2)), DL_NUMERIC,
            "residual range failure") || y[0] != 1 || y[1] != 9) return 0;
    puts("partial numerical failures detected; no damaged result reused");
    return 1;
}

```

For the backward failure, the zero input row and epsilon `0.25` give `r=2`. The first input gradient is two and is stored; the second uses the huge scale and cannot be represented. The scale destination has already been cleared by SET. The test inspects this state only to demonstrate the contract; it never feeds the damaged results to another operation.

These routines are not transactional updates of an entire tensor. Adding a full staging buffer merely to roll back every numerical error would create a different memory contract. For now, the caller discards the failed result and restarts from valid inputs and properly initialized destinations.

## Compose a complete learned residual branch

The next object owns all arrays for the `[2,3]` graph introduced earlier. It contains no owning pointers. Assigning a compound literal initializes its values, and copying the object copies its inline arrays rather than creating two owners of one heap allocation. Borrowed views are constructed only after the object has its final address.

The incoming gradient is fixed by the diagnostic objective

$$
L=\sum_{i,j}G_{ij}Y_{ij},
$$

where `upstream` stores $G$. Therefore $\partial L/\partial Y=G$. This scalar objective tests the operator composition without needing another cross-entropy graph. In normalization-only mode, substitute $N$ for $Y$.

```c title="tests/test_ops.c"
/* Owns inline arrays, not pointers or heap allocations. Views are temporary. */
typedef struct {
    float x[6], scale[3], w[9], upstream[6];
    float n[6], branch[6], y[6], db[6], dn[6], dx[6], ds[3], dw[9];
    float epsilon;
} NormGraph;

static void graph_init(NormGraph *g, int variant)
{
    *g = (NormGraph){
        .x = {0.75f, -0.5f, 1.25f, -1.0f, 0.3f, 0.8f},
        .scale = {1.25f, -0.75f, 0.5f},
        .w = {0.4f, -0.3f, 0.2f, -0.1f, 0.5f, -0.6f, 0.7f, 0.1f, -0.2f},
        .upstream = {0.3f, -0.4f, 0.7f, -0.6f, 0.2f, 0.5f},
        .epsilon = 0.125f};
    if (variant == 1) {
        for (size_t j = 0; j < 6; ++j) g->x[j] = 0.0f;
        g->epsilon = 0.25f;
    } else if (variant == 2) {
        g->epsilon = 1e-5f;
    }
}

```

Variant zero uses ordinary nonzero rows and epsilon `0.125`. Variant one makes both rows zero with epsilon `0.25`; it still has a meaningful input derivative. Variant two restores the ordinary rows and uses the decoder epsilon `1e-5`. The learned scales include a negative value, so dropping scales from the denominator's dot product cannot hide behind equal unit gains.

Append the full forward call. It executes RMSNorm, the selected matrix-product backend, and residual addition in C. The final double sum is only the small diagnostic scalar; it is not a replacement model implementation in another language.

```c title="tests/test_ops.c"
static DlStatus graph_forward(NormGraph *g, DlBackend backend,
                               int residual, double *objective)
{
    const DlConstMatrix x = dl_matrix_read(matrix(g->x, 2, 3));
    const DlMatrix n = matrix(g->n, 2, 3);
    DlStatus s = dl_rmsnorm_forward(x, dl_matrix_read(matrix(g->scale, 1, 3)),
                                   g->epsilon, n);
    if (s == DL_OK && residual) {
        s = dl_matmul(backend, dl_matrix_read(n), DL_NOTRANS,
            dl_matrix_read(matrix(g->w, 3, 3)), DL_TRANS,
            matrix(g->branch, 2, 3), DL_SET);
        if (s == DL_OK) s = dl_residual_add(x,
            dl_matrix_read(matrix(g->branch, 2, 3)), matrix(g->y, 2, 3));
    }
    if (s != DL_OK) return s;
    const float *values = residual ? g->y : g->n;
    double sum = 0.0;
    for (size_t j = 0; j < 6; ++j) sum += (double)values[j] * (double)g->upstream[j];
    if (!isfinite(sum)) return DL_NUMERIC;
    *objective = sum;
    return DL_OK;
}

```

The status chain stops at the first failure. The scalar objective is committed only after every required output is valid. Intermediate arrays may have been changed before an error, so the graph result as a whole must still be discarded on failure.

Append backward. The two residual edges write different intermediate gradients, and the final normalization call adds its input contribution to the identity contribution already in `dx`.

```c title="tests/test_ops.c"
static DlStatus graph_backward(NormGraph *g, DlBackend backend, int residual)
{
    const DlConstMatrix upstream = dl_matrix_read(matrix(g->upstream, 2, 3));
    const DlMatrix dx = matrix(g->dx, 2, 3);
    DlConstMatrix incoming = upstream;
    DlStatus s = DL_OK;
    if (residual) {
        s = dl_residual_backward(upstream, dx, DL_SET);
        if (s == DL_OK) s = dl_residual_backward(upstream, matrix(g->db, 2, 3), DL_SET);
        if (s == DL_OK) s = dl_matmul(backend,
            dl_matrix_read(matrix(g->db, 2, 3)), DL_NOTRANS,
            dl_matrix_read(matrix(g->w, 3, 3)), DL_NOTRANS,
            matrix(g->dn, 2, 3), DL_SET);
        if (s == DL_OK) s = dl_matmul(backend,
            dl_matrix_read(matrix(g->db, 2, 3)), DL_TRANS,
            dl_matrix_read(matrix(g->n, 2, 3)), DL_NOTRANS,
            matrix(g->dw, 3, 3), DL_SET);
        incoming = dl_matrix_read(matrix(g->dn, 2, 3));
    }
    if (s == DL_OK) s = dl_rmsnorm_backward(dl_matrix_read(matrix(g->x, 2, 3)),
        dl_matrix_read(matrix(g->scale, 1, 3)), incoming, g->epsilon,
        dx, residual ? DL_ADD : DL_SET, matrix(g->ds, 1, 3), DL_SET);
    return s;
}

```

Trace the destinations rather than only reading the formulas:

| Operation | Destination and mode | Meaning after success |
|---|---|---|
| Reverse identity edge | `dx`, SET | $G$ |
| Reverse branch edge | `db`, SET | $G$ for the matrix branch |
| Matrix product, NN | `dn`, SET | $GW$ |
| Matrix product, TN | `dw`, SET | $G^{\mathsf T}N$ |
| RMSNorm backward | `dx`, ADD; `ds`, SET | Identity plus branch input gradient; first scale gradient |

The explicit `db` copy makes the two residual edges observable. A later liveness-aware implementation could borrow `upstream` directly for the matrix branch, but that optimization is not required to make this graph correct. No input pointer is secretly reused as a writable destination.

In normalization-only mode, `incoming` remains the original upstream view and `dx` uses SET because there is no identity contribution. In the residual graph, the generic matrix dispatcher supplies both products, so the graph gradient checks exercise the scalar and blocked paths. Chapter 5's convenience wrappers remain unchanged.

## Differentiate the actual C forward evaluations numerically

The checker below calls forward and backward once to obtain the analytic gradients. It then perturbs every input and scale coordinate; for the residual graph it also perturbs every weight coordinate. Each objective evaluation uses the real C operators, not the same closed-form derivative rewritten as an alleged reference.

`parameters` is an array of three pointers, each pointing to one already-owned inline array. `derivatives` similarly points to the three gradient arrays. These pointers do not allocate, move, or extend the lifetime of anything. The `counts` and `names` arrays describe the corresponding groups in the same order. This is ordinary indexed traversal of known arrays, not a function-pointer framework or automatic differentiation system.

The three relative step coefficients are `0.01`, `0.003`, and `0.001`; the absolute step is the coefficient times `max(1,abs(parameter))`. Once stored in float, the positive and negative samples may not be exactly the intended distance apart. Divide by their **represented** spacing:

$$
\text{numerical derivative}
=\frac{L(\theta_+)-L(\theta_-)}{\operatorname{double}(\theta_+)-\operatorname{double}(\theta_-)}.
$$

The computed forward map contains float rounding, so finite differences of it are not exact derivatives of a smooth real-valued function at arbitrarily small steps. As in Chapter 3, decreasing the step indefinitely eventually amplifies rounding rather than improving the test. Three moderate steps, exact hand cases, and boundary tests give complementary checks.

The acceptance threshold is an absolute term plus a relative term: `8e-5 + 3e-3*max(abs(analytic),abs(numerical))`. The absolute term handles a near-zero derivative and rounded forward cancellation; the relative term accommodates truncation at the largest step, including the zero-row fixture. This is a threshold for these tiny fixtures and arithmetic, not a universal tolerance for pretrained models. All three step sizes must pass, and non-finite values fail independently of the threshold.

```c title="tests/test_ops.c"
static int numerical_case(int variant, int residual, DlBackend backend,
                           size_t *comparisons, double *worst_ratio)
{
    NormGraph g;
    graph_init(&g, variant);
    const NormGraph original = g; /* Safe value copy: no owning pointers. */
    double baseline = 0.0;
    if (!status_is(graph_forward(&g, backend, residual, &baseline), DL_OK, "graph forward") ||
        !status_is(graph_backward(&g, backend, residual), DL_OK, "graph backward")) return 0;
    float *parameters[3] = {g.x, g.scale, g.w};
    const float *derivatives[3] = {g.dx, g.ds, g.dw};
    const size_t counts[3] = {6, 3, 9};
    const char *names[3] = {"x", "scale", "w"};
    const double steps[3] = {0.01, 0.003, 0.001};
    const size_t groups = residual ? 3 : 2;
    for (size_t p = 0; p < groups; ++p) {
        for (size_t j = 0; j < counts[p]; ++j) {
            const float saved = parameters[p][j];
            const double analytic = (double)derivatives[p][j];
            for (size_t k = 0; k < 3; ++k) {
                const double h = steps[k] * fmax(1.0, fabs((double)saved));
                const float hi = (float)((double)saved + h);
                const float lo = (float)((double)saved - h);
                const double spacing = (double)hi - (double)lo;
                double plus = 0.0, minus = 0.0;
                parameters[p][j] = hi;
                const DlStatus sp = graph_forward(&g, backend, residual, &plus);
                parameters[p][j] = lo;
                const DlStatus sm = graph_forward(&g, backend, residual, &minus);
                parameters[p][j] = saved; /* Restore even if an evaluation failed. */
                if (!status_is(sp, DL_OK, "plus evaluation") ||
                    !status_is(sm, DL_OK, "minus evaluation") || spacing <= 0) return 0;
                const double numerical = (plus - minus) / spacing;
                const double tolerance = 8e-5 + 3e-3 * fmax(fabs(analytic), fabs(numerical));
                const double ratio = fabs(analytic - numerical) / tolerance;
                if (!isfinite(analytic) || !isfinite(numerical) || ratio > 1.0) {
                    fprintf(stderr, "variant=%d residual=%d backend=%d %s[%zu] h=%.3g: "
                        "analytic=%.9g numeric=%.9g ratio=%.4g\n", variant, residual,
                        (int)backend, names[p], j, h, analytic, numerical, ratio);
                    return 0;
                }
                if (ratio > *worst_ratio) *worst_ratio = ratio;
                ++*comparisons;
            }
        }
    }
    if (memcmp(g.x, original.x, sizeof g.x) ||
        memcmp(g.scale, original.scale, sizeof g.scale) ||
        memcmp(g.w, original.w, sizeof g.w)) {
        fputs("finite differences did not restore parameters\n", stderr); return 0;
    }
    return 1;
}

```

Restoring a perturbed parameter happens before inspecting either evaluation status. The checker verifies restored parameters again at the end. Forward calls overwrite only the saved activations, not the analytic gradient arrays, so the analytic reference remains the one computed at the original parameters.

After the last perturbation, the temporary activation arrays still describe the last perturbed forward, even though parameters have been restored. This function discards the local graph immediately afterward. Code that wants another backward call must first rerun forward at the restored parameters. Restoring parameters alone does not restore a training tape.

There are 81 normalization-only comparisons: three fixtures, nine input/scale coordinates, and three steps. There are 108 learned-residual comparisons: two matrix backends, eighteen input/scale/weight coordinates, and three steps. The total is **189**, in addition to the inherited Chapter 5 checker. Repeating the program in three build configurations does not create three times as many distinct mathematical fixtures.

Finish the test file with this driver. Its exit status propagates failures, rather than relying on `assert` remaining enabled in every build.

```c title="tests/test_ops.c"
int main(void)
{
    if (!hand_and_modes() || !boundary_values() || !rejected_calls() || !partial_failures()) return 1;
    size_t comparisons = 0;
    double worst_ratio = 0.0;
    for (int variant = 0; variant < 3; ++variant) {
        if (!numerical_case(variant, 0, DL_BACKEND_REF, &comparisons, &worst_ratio)) return 1;
    }
    for (int backend = 0; backend < 2; ++backend) {
        if (!numerical_case(0, 1, (DlBackend)backend, &comparisons, &worst_ratio)) return 1;
    }
    printf("RMSNorm/residual finite differences: %zu passed; worst error/tolerance=%.6g\n",
           comparisons, worst_ratio);
    puts("all normalization and residual checks passed");
    return 0;
}
```

## Build and run the cumulative program

Replace the root Makefile with the following complete file. The only functional additions to Chapter 6's build are `src/norm.c` in `CORE`, its header in `HEADERS`, and `test_ops` in `PROGRAMS`. The matrix comparison application is retained unchanged.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops

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

From the learner workspace on the Mac, run:

```sh title="Terminal — native builds and checks"
make clean
make test
```

The default compiler is `xcrun clang`. An equivalent configured compiler can be selected with `make CC=clang test`; that override is not a requirement to change the Mac's installed tools. All builds retain finite-value semantics and disabled floating-point contraction as established in Chapter 6. The sanitizer and optimized executables are distinct files.

For a focused run after an edit:

```sh title="Terminal — focus on the new operators"
make debug sanitize release
./build/debug/test_ops
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./build/sanitize/test_ops
./build/release/test_ops
```

Use the sanitizers supported by the installed toolchain. The [Clang AddressSanitizer documentation][asan] describes the runtime leak-detection option. Its availability depends on the installed platform/runtime; inability to enable that option is not evidence of a leak, and address/undefined-behavior checks should still be run. The new operators allocate no heap memory, but the cumulative suite retains the earlier owner tests.

The new executable should print the hand values and report all checks passing. The final error ratio may vary slightly by compiler and machine. One observed Clang 17, x86_64 Linux run printed:

```text title="Observed output — test_ops, x86_64 Linux"
hand row: y=[1.500000, -1.000000], norm dx=[0.171875, 0.062500]
residual row: dx=[1.171875, -1.937500]
zero rows, zero scales, large inputs, and width-one checks passed
shape, alias, epsilon, and nonfinite preflight checks passed
partial numerical failures detected; no damaged result reused
RMSNorm/residual finite differences: 189 passed; worst error/tolerance=0.169664
all normalization and residual checks passed
```

The companion source passed debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized builds in that Linux environment. The leak-enabled new test also passed. The inherited 104 finite-difference comparisons, 56 blocked shape/orientation/write-mode cases, kernel/loss checks, and comparison smoke calls passed in the cumulative test target. These are executed C checks, not M4 measurements. No macOS execution, normalization speedup, model-training throughput, or process-memory result is claimed here.

## Make three numerical mistakes deliberately

Use a temporary copy of the source and restore the correct version after each experiment. Rebuild `test_ops` after every change. A clean memory-sanitizer run is not a mathematical correctness test.

**Remove the denominator derivative.** In `dl_rmsnorm_backward`, replace `const double mean_dot = dot / (double)x.cols;` with `const double mean_dot = 0.0 * dot / (double)x.cols;`. This leaves the dot calculation present but makes the correction term zero. The first hand fixture reports `dx mode: 0.5; expected 0.171875`. The output forward values are still correct: the fault exists only in backward.

**Assign each scale contribution instead of adding it.** In the `put_value` call that writes `dscale`, replace its final `DL_ADD` with `DL_SET`. The first mode fixture reports `scale mode: -0.375; expected 0.375`. It retained the second row rather than their sum. The same defect also discards contributions present before the call when public `scale_mode` is ADD.

**Erase the residual identity path.** In the test graph's final `dl_rmsnorm_backward` call, replace `dx, residual ? DL_ADD : DL_SET` with `dx, DL_SET`. Isolated normalization checks still pass because the normalization implementation is unmodified. The learned residual finite difference fails: for the first input coordinate, one observed run gave analytic `0.670060456` versus numerical `0.970060567`. The missing value is the identity-edge incoming gradient, approximately `0.3`.

All three variations were run in separate sanitizer builds and failed explicit numerical checks without a sanitizer memory diagnostic. That distinction is the point: bounds checks cannot decide whether a valid store should have been assignment or accumulation.

For a valid variation, change only the default `.epsilon = 0.125f` in `graph_init` to `.epsilon = 0.5f`, keeping the zero-row and decoder-epsilon overrides unchanged. Predict a smaller reciprocal RMS in the ordinary fixture, but do not assume every coordinate's input derivative decreases: the direct and denominator terms change together. The analytic and numerical derivatives must still agree. This variation passed all 189 comparisons in an executed sanitizer run; its maximum error/tolerance ratio was approximately `0.180166`, not a required target value.

## Diagnose the first wrong quantity

| Symptom | First place to inspect |
|---|---|
| Changing one row changes another row's forward result | Reduction axes: divide and sum over coordinates of one row, not all tokens |
| Forward is right but input gradients resemble scaled incoming gradients | Missing denominator term or missing scales inside its dot product |
| Scale gradient equals only the last row | Assignment inside the row loop or clearing inside that loop |
| All scale gradients are too small by the row count | An extra mean reduction after the loss already supplied its scaling |
| Zero rows return zero input gradients | Treating a zero normalized output as zero sensitivity |
| Standalone norm gradients pass but residual graph gradients do not | SET erased the identity contribution, or a matrix transpose is wrong |
| NaN-filled SET outputs cause failure | Old destination was read despite overwrite semantics |
| Correct finite inputs cause a later numeric error | Intermediate/result range; discard every output, not only the failing coordinate |
| A second backward call disagrees after numerical checks | Parameters were restored but saved forward values were not recomputed |

The completed program now has a fully differentiated normalization and an explicit residual accumulation rule. It still does not own an optimizer or a full model tape, and it does not need either to test these mechanisms. The next [chapter in the roadmap](https://25d-228.github.io/gpt-c/index.html#chapter-8-swiglu-and-the-two-branch-backward-pass) builds the complete SwiGLU feed-forward sublayer, where two learned branches create another important gradient merge.

## References

These primary sources support the named normalization and architectural mechanisms. The epsilon-specific derivation, memory contracts, fixtures, and C routines in this chapter are provided directly rather than delegated to an external implementation.

- Biao Zhang and Rico Sennrich, [*Root Mean Square Layer Normalization*, version 1][rmsnorm], Section 4, equation 4, for the RMS statistic; Section 4.2, equations 8–9, for gain and normalization gradient analysis. The implementation here adds explicit positive epsilon and uses scale only.
- Hugo Touvron et al., [*LLaMA: Open and Efficient Foundation Language Models*, version 1][llama], Section 2.2, for RMSNorm before transformer sublayers. The small matrix branch in this chapter is a diagnostic composition, not a claimed reproduction of a whole transformer block.

- Clang, [AddressSanitizer usage and memory leak detection][asan], for the compile/link instrumentation and optional runtime leak checks.

[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#memory-leak-detection
[rmsnorm]: https://arxiv.org/html/1910.07467v1#S4
[llama]: https://arxiv.org/html/2302.13971v1#S2.SS2

