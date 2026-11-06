#include "dl/rope.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int require(int condition, const char *where)
{
    if (!condition) fprintf(stderr, "%s\n", where);
    return condition;
}

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

/* Only fixed, small, already-sized arrays use this test-only helper. */
static DlMatrix dense(float *p, size_t rows, size_t cols)
{
    return (DlMatrix){p, rows * cols, 0, rows, cols, cols};
}

static double dot(const float *a, const float *b, size_t n)
{
    double result = 0.0;
    for (size_t j = 0; j < n; ++j) result += (double)a[j] * (double)b[j];
    return result;
}

static int frequencies_and_hand(void)
{
    double f[4] = {9.0, 9.0, 9.0, 9.0};
    if (!status_is(dl_rope_frequencies(8, 10000.0, f, 4), DL_OK,
                   "four frequencies")) return 0;
    const double want[4] = {1.0, 0.1, 0.01, 0.001};
    for (size_t j = 0; j < 4; ++j) {
        if (!near_value(f[j], want[j], 1e-15, "frequency value")) return 0;
    }
    double checkpoint[32];
    if (!status_is(dl_rope_frequencies(64, 100000.0, checkpoint, 32),
                   DL_OK, "checkpoint frequencies") ||
        !near_value(checkpoint[0], 1.0, 0.0, "first checkpoint frequency") ||
        !near_value(checkpoint[31], 1.43301257023696e-5, 1e-18,
                    "last checkpoint frequency")) return 0;
    const double saved[4] = {f[0], f[1], f[2], f[3]};
    if (!status_is(dl_rope_frequencies(7, 10000.0, f, 4), DL_SHAPE, "odd d") ||
        !status_is(dl_rope_frequencies(0, 10000.0, f, 4), DL_SHAPE, "zero d") ||
        !status_is(dl_rope_frequencies(8, 10000.0, f, 3), DL_SHAPE, "bad count") ||
        !status_is(dl_rope_frequencies(8, 1.0, f, 4), DL_ARGUMENT, "theta one") ||
        !status_is(dl_rope_frequencies(8, NAN, f, 4), DL_ARGUMENT, "theta NaN") ||
        !status_is(dl_rope_frequencies(8, INFINITY, f, 4), DL_ARGUMENT,
                    "theta infinity") ||
        !status_is(dl_rope_frequencies(8, 10000.0, NULL, 4), DL_ARGUMENT,
                    "null frequencies") ||
        !status_is(dl_rope_frequencies(SIZE_MAX - 1, 10000.0, f, SIZE_MAX / 2),
                    DL_OVERFLOW, "frequency bytes overflow") ||
        !require(memcmp(f, saved, sizeof f) == 0, "failed constructor wrote")) {
        return 0;
    }

    /* Synthetic diagnostic frequencies, NOT the model's geometric schedule. */
    const double pi = acos(-1.0);
    const double hand_f[2] = {pi / 4.0, pi / 8.0};
    const size_t position[1] = {2};
    float x[4] = {3.0f, 1.0f, 4.0f, 2.0f};
    float y[4] = {NAN, NAN, NAN, NAN}, dx[4] = {NAN, NAN, NAN, NAN};
    const double expected[4] = {-4.0, -1.0 / sqrt(2.0), 3.0, 3.0 / sqrt(2.0)};
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(x, 1, 4)), 4,
                   position, 1, hand_f, 2, dense(y, 1, 4)), DL_OK, "hand forward")) {
        return 0;
    }
    for (size_t j = 0; j < 4; ++j) {
        if (!near_value(y[j], expected[j], 3e-7, "hand split-half value")) return 0;
    }
    float incoming[4] = {2.0f, -1.0f, 3.0f, 4.0f};
    const double grad[4] = {3.0, 3.0 / sqrt(2.0), -2.0, 5.0 / sqrt(2.0)};
    if (!status_is(dl_rope_backward(dl_matrix_read(dense(incoming, 1, 4)), 4,
                   position, 1, hand_f, 2, dense(dx, 1, 4), DL_SET), DL_OK,
                   "hand backward")) return 0;
    for (size_t j = 0; j < 4; ++j) {
        if (!near_value(dx[j], grad[j], 3e-7, "hand transpose value")) return 0;
    }
    printf("split-half hand row: [%.6f, %.6f, %.6f, %.6f]\n",
           (double)y[0], (double)y[1], (double)y[2], (double)y[3]);
    return 1;
}

static DlMatrix padded(float *p)
{
    return (DlMatrix){p, 40, 2, 3, 8, 10};
}

static int guards(const float *p)
{
    for (size_t j = 0; j < 40; ++j) {
        const int logical = j >= 2 && (j - 2) / 10 < 3 && (j - 2) % 10 < 8;
        if (!logical && p[j] != -777.0f) return 0;
    }
    return 1;
}

static int padded_modes_and_norms(void)
{
    float x[40], y[40], dy[40], dx[40], recovered[40], set_dx[40];
    for (size_t j = 0; j < 40; ++j) {
        x[j] = y[j] = dy[j] = dx[j] = recovered[j] = -777.0f;
    }
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            const size_t at = 2 + i * 10 + j;
            x[at] = (float)((int)(i * 8 + j) - 10) / 7.0f;
            dy[at] = (float)((int)((i * 8 + j) % 9) - 4) / 5.0f;
            y[at] = dx[at] = recovered[at] = NAN;
        }
    }
    float saved_x[40], saved_dy[40];
    memcpy(saved_x, x, sizeof x); memcpy(saved_dy, dy, sizeof dy);
    const size_t positions[3] = {0, 3, 8191};
    double f[2];
    if (!status_is(dl_rope_frequencies(4, 100000.0, f, 2), DL_OK, "padded f") ||
        !status_is(dl_rope_forward(dl_matrix_read(padded(x)), 4, positions, 3,
                   f, 2, padded(y)), DL_OK, "padded forward") ||
        !status_is(dl_rope_backward(dl_matrix_read(padded(y)), 4, positions, 3,
                   f, 2, padded(recovered), DL_SET), DL_OK, "inverse rotation") ||
        !status_is(dl_rope_backward(dl_matrix_read(padded(dy)), 4, positions, 3,
                   f, 2, padded(dx), DL_SET), DL_OK, "padded backward")) return 0;
    memcpy(set_dx, dx, sizeof dx);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t h = 0; h < 2; ++h) {
            const size_t start = 2 + i * 10 + h * 4;
            const double n = dot(x + start, x + start, 4);
            if (!near_value(dot(y + start, y + start, 4), n,
                            2e-6 * (1.0 + n), "head squared norm")) return 0;
        }
        for (size_t j = 0; j < 8; ++j) {
            const size_t at = 2 + i * 10 + j;
            if (!near_value(recovered[at], x[at], 1e-6, "round trip") ||
                (i == 0 && !near_value(y[at], x[at], 0.0, "position zero"))) {
                return 0;
            }
            dx[at] = 1.25f;
        }
    }
    for (int pass = 1; pass <= 2; ++pass) {
        if (!status_is(dl_rope_backward(dl_matrix_read(padded(dy)), 4,
                       positions, 3, f, 2, padded(dx), DL_ADD), DL_OK,
                       "repeated backward ADD")) return 0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                const size_t at = 2 + i * 10 + j;
                if (!near_value(dx[at], 1.25 + (double)pass * (double)set_dx[at],
                                4e-7, "ADD retains prior gradient")) return 0;
            }
        }
    }
    return require(guards(y) && guards(dx) && guards(recovered), "changed padding") &&
           require(memcmp(x, saved_x, sizeof x) == 0 &&
                   memcmp(dy, saved_dy, sizeof dy) == 0, "changed input");
}

static int sequence_and_positions(void)
{
    float x[32], full[32], chunks[32], keys[16], rotated_keys[16];
    for (size_t j = 0; j < 32; ++j) x[j] = (float)((int)(j % 11) - 5) / 3.0f;
    const size_t positions[4] = {7, 8, 9, 10};
    double f[2];
    if (!status_is(dl_rope_frequencies(4, 100000.0, f, 2), DL_OK, "sequence f") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(x, 4, 8)), 4,
                   positions, 4, f, 2, dense(full, 4, 8)), DL_OK, "whole sequence")) {
        return 0;
    }
    for (size_t i = 0; i < 4; ++i) {
        /* Retain canonical allocation bases, not x+i*8 and chunks+i*8. */
        const DlConstMatrix in = {x, 32, i * 8, 1, 8, 8};
        const DlMatrix out = {chunks, 32, i * 8, 1, 8, 8};
        if (!status_is(dl_rope_forward(in, 4, &positions[i], 1, f, 2, out),
                       DL_OK, "one-row chunk")) return 0;
        for (size_t j = 0; j < 4; ++j) keys[i * 4 + j] = x[i * 8 + j];
    }
    if (!require(memcmp(full, chunks, sizeof full) == 0,
                 "chunk changed absolute-position result") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(keys, 4, 4)), 4,
                   positions, 4, f, 2, dense(rotated_keys, 4, 4)), DL_OK,
                   "one key head versus two query heads")) return 0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (!near_value(rotated_keys[i * 4 + j], full[i * 8 + j], 0.0,
                            "frequency independent of head count")) return 0;
        }
    }

    const size_t reset_positions[4] = {4, 5, 4, 5};
    memcpy(x + 16, x, 16 * sizeof(*x));
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(x, 4, 8)), 4,
                   reset_positions, 4, f, 2, dense(full, 4, 8)), DL_OK,
                   "two flattened batch sequences") ||
        !require(memcmp(full, full + 16, 16 * sizeof(*full)) == 0,
                 "batch position reset lost")) return 0;

    float unit[4] = {1.0f, 0.0f, 1.0f, 0.0f}, large[4];
    const size_t large_p[2] = {16777216, 16777217};
    const double unit_f[1] = {1.0};
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(unit, 2, 2)), 2,
                   large_p, 2, unit_f, 1, dense(large, 2, 2)), DL_OK,
                   "integer to double positions")) return 0;
    for (size_t i = 0; i < 2; ++i) {
        if (!near_value(large[i * 2], cos((double)large_p[i]), 4e-8, "large cos") ||
            !near_value(large[i * 2 + 1], sin((double)large_p[i]), 4e-8,
                        "large sin")) return 0;
    }
    return require(fabs((double)large[0] - (double)large[2]) > 0.1,
                   "neighboring large positions collapsed");
}

static int widths_and_limits(void)
{
    const size_t widths[4] = {2, 4, 6, 64};
    const size_t positions[4] = {0, 1, 8191, DL_ROPE_MAX_POSITION};
    float x[128], y[128];
    for (size_t j = 0; j < 128; ++j) x[j] = (float)((int)(j % 13) - 6) / 8.0f;
    for (size_t w = 0; w < 4; ++w) {
        const size_t d = widths[w];
        double f[32];
        if (!status_is(dl_rope_frequencies(d, 100000.0, f, d / 2), DL_OK,
                       "varied width frequencies")) return 0;
        const DlConstMatrix input = {x, 128, 0, 1, 2 * d, 2 * d};
        const DlMatrix output = {y, 128, 0, 1, 2 * d, 2 * d};
        for (size_t p = 0; p < 4; ++p) {
            if (!status_is(dl_rope_forward(input, d,
                           &positions[p], 1, f, d / 2, output),
                           DL_OK, "varied width rotation")) return 0;
            for (size_t h = 0; h < 2; ++h) {
                const double n = dot(x + h * d, x + h * d, d);
                if (!near_value(dot(y + h * d, y + h * d, d), n,
                                2e-6 * (1.0 + n), "varied width norm")) return 0;
            }
        }
    }
    return 1;
}

static int relative_dot(void)
{
    float q[4] = {1.0f, 2.0f, 3.0f, 4.0f}, k[4] = {-2.0f, 0.5f, 1.0f, -1.0f};
    float rq[4], rk[4], shifted_q[4], shifted_k[4], relative_k[4];
    const size_t p[1] = {3}, n[1] = {9}, ps[1] = {14}, ns[1] = {20}, delta[1] = {6};
    double f[2];
    if (!status_is(dl_rope_frequencies(4, 10000.0, f, 2), DL_OK, "relative f") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(q, 1, 4)), 4,
                   p, 1, f, 2, dense(rq, 1, 4)), DL_OK, "query at m") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4,
                   n, 1, f, 2, dense(rk, 1, 4)), DL_OK, "key at n") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(q, 1, 4)), 4,
                   ps, 1, f, 2, dense(shifted_q, 1, 4)), DL_OK, "shifted query") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4,
                   ns, 1, f, 2, dense(shifted_k, 1, 4)), DL_OK, "shifted key") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4,
                   delta, 1, f, 2, dense(relative_k, 1, 4)), DL_OK,
                   "relative key")) return 0;
    const double original = dot(rq, rk, 4), shifted = dot(shifted_q, shifted_k, 4);
    if (!near_value(shifted, original, 3e-6, "common position shift") ||
        !near_value(dot(q, relative_k, 4), original, 3e-6, "relative angle n-m") ||
        !require(fabs(dot(rq, shifted_k, 4) - original) > 0.1,
                 "moving only key unexpectedly invariant")) return 0;
    printf("relative dot: original=%.8f, common-shift=%.8f\n", original, shifted);
    return 1;
}

static int unchanged(DlStatus got, DlStatus expected, const float *out,
                     const float *before, size_t count, const char *where)
{
    return status_is(got, expected, where) &&
           require(memcmp(out, before, count * sizeof(*out)) == 0,
                   "rejected call modified output");
}

static int rejected_calls(void)
{
    float x[4] = {1, 2, 3, 4}, y[4] = {9, 8, 7, 6};
    float before[4]; memcpy(before, y, sizeof y);
    size_t p[1] = {1};
    double f[2] = {1.0, 0.01};
    const DlConstMatrix in = dl_matrix_read(dense(x, 1, 4));
    const DlMatrix out = dense(y, 1, 4);
    if (!unchanged(dl_rope_forward(in, 3, p, 1, f, 2, out), DL_SHAPE,
                   y, before, 4, "reject odd width") ||
        !unchanged(dl_rope_forward(in, 0, p, 1, f, 2, out), DL_SHAPE,
                   y, before, 4, "reject zero width") ||
        !unchanged(dl_rope_forward(in, 6, p, 1, f, 3, out), DL_SHAPE,
                   y, before, 4, "head does not divide columns") ||
        !unchanged(dl_rope_forward(in, 4, p, 0, f, 2, out), DL_SHAPE,
                   y, before, 4, "wrong position count") ||
        !unchanged(dl_rope_forward(in, 4, p, 1, f, 1, out), DL_SHAPE,
                   y, before, 4, "wrong frequency count") ||
        !unchanged(dl_rope_forward(in, 4, NULL, 1, f, 2, out), DL_ARGUMENT,
                   y, before, 4, "null positions") ||
        !unchanged(dl_rope_forward(in, 4, p, 1, NULL, 2, out), DL_ARGUMENT,
                   y, before, 4, "null frequencies") ||
        !unchanged(dl_rope_backward(in, 4, p, 1, f, 2, out, (DlWriteMode)7),
                   DL_ARGUMENT, y, before, 4, "invalid mode")) return 0;
    DlMatrix bad = out; bad.cols = 2;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, bad), DL_SHAPE,
                   y, before, 4, "output shape")) return 0;
    bad = out; bad.offset = SIZE_MAX;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, bad), DL_RANGE,
                   y, before, 4, "offset outside allocation")) return 0;
    bad = out; bad.rows = 3; bad.ld = SIZE_MAX / 2 + 1;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, bad), DL_OVERFLOW,
                   y, before, 4, "view arithmetic overflow")) return 0;
    float before_x[4]; memcpy(before_x, x, sizeof x);
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, dense(x, 1, 4)),
                   DL_ALIAS, x, before_x, 4, "no in-place rotation")) return 0;
    if (SIZE_MAX > DL_ROPE_MAX_POSITION) {
        p[0] = SIZE_MAX;
        if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, out), DL_RANGE,
                       y, before, 4, "oversized position")) return 0;
        p[0] = 1;
    }
    const double invalid_f[4] = {0.0, -1.0, 2.0, NAN};
    for (size_t j = 0; j < 4; ++j) {
        f[1] = invalid_f[j];
        if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, out), DL_NUMERIC,
                       y, before, 4, "invalid frequency")) return 0;
    }
    f[1] = 0.01; x[3] = NAN;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, out), DL_NUMERIC,
                   y, before, 4, "late nonfinite input preflight")) return 0;
    x[3] = 4; y[3] = NAN; memcpy(before, y, sizeof y);
    if (!unchanged(dl_rope_backward(in, 4, p, 1, f, 2, out, DL_ADD), DL_NUMERIC,
                   y, before, 4, "nonfinite old ADD value")) return 0;

    float shared[16];
    for (size_t j = 0; j < 16; ++j) shared[j] = (float)j;
    float saved[16]; memcpy(saved, shared, sizeof shared);
    const size_t pp[2] = {1, 2};
    const DlConstMatrix strided = {shared, 16, 0, 2, 4, 8};
    const DlMatrix gaps = {shared, 16, 4, 2, 4, 8};
    if (!unchanged(dl_rope_forward(strided, 4, pp, 2, f, 2, gaps), DL_ALIAS,
                   shared, saved, 16, "conservative bounding-span alias")) return 0;
    const DlConstMatrix first = {shared, 16, 0, 1, 4, 4};
    const DlMatrix separate = {shared, 16, 4, 1, 4, 4};
    return status_is(dl_rope_forward(first, 4, p, 1, f, 2, separate), DL_OK,
                     "disjoint spans in one allocation") &&
           require(memcmp(shared, saved, 4 * sizeof(*shared)) == 0,
                   "disjoint input changed");
}

static int partial_failures(void)
{
    const double f[2] = {acos(-1.0) / 4.0, acos(-1.0) / 4.0};
    const size_t p[1] = {1};
    float x[4] = {3.0f, FLT_MAX, 4.0f, FLT_MAX}, y[4] = {9, 9, 9, 9};
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(x, 1, 4)), 4, p, 1,
                   f, 2, dense(y, 1, 4)), DL_NUMERIC, "rotation overflow") ||
        !near_value(y[0], -1.0 / sqrt(2.0), 3e-7, "earlier first committed") ||
        !near_value(y[2], 7.0 / sqrt(2.0), 4e-7, "earlier second committed") ||
        !require(y[1] == 9 && y[3] == 9, "failing pair partially committed")) return 0;
    const size_t zero[1] = {0};
    float incoming[4] = {1, FLT_MAX, 2, 0}, dx[4] = {10, FLT_MAX, 20, 13};
    return status_is(dl_rope_backward(dl_matrix_read(dense(incoming, 1, 4)), 4,
                     zero, 1, f, 2, dense(dx, 1, 4), DL_ADD), DL_NUMERIC,
                     "ADD overflow") &&
           require(dx[0] == 11 && dx[2] == 22 && dx[1] == FLT_MAX && dx[3] == 13,
                   "ADD partial-pair state");
}

static DlStatus rotation_objective(float *x, const float *incoming,
                                    const size_t *p, const double *f, double *loss)
{
    float y[24];
    const DlStatus s = dl_rope_forward(dl_matrix_read(dense(x, 3, 8)), 4,
                                      p, 3, f, 2, dense(y, 3, 8));
    if (s != DL_OK) return s;
    *loss = dot(y, incoming, 24);
    return DL_OK;
}

static int derivative(double analytic, double numeric, double *worst)
{
    const double tolerance = 2e-4 + 3e-4 * fmax(fabs(analytic), fabs(numeric));
    const double ratio = fabs(analytic - numeric) / tolerance;
    if (isfinite(ratio) && ratio > *worst) *worst = ratio;
    return near_value(numeric, analytic, tolerance, "finite difference");
}

static int isolated_gradients(size_t *checks, double *worst)
{
    float x[24], incoming[24], dx[24];
    for (size_t j = 0; j < 24; ++j) {
        x[j] = (float)((int)(j % 11) - 5) / 7.0f;
        incoming[j] = (float)((int)(j % 7) - 3) / 5.0f;
    }
    const size_t p[3] = {0, 3, 8191};
    double f[2], baseline = 0.0;
    if (!status_is(dl_rope_frequencies(4, 100000.0, f, 2), DL_OK, "gradient f") ||
        !status_is(rotation_objective(x, incoming, p, f, &baseline), DL_OK,
                   "baseline objective") ||
        !status_is(dl_rope_backward(dl_matrix_read(dense(incoming, 3, 8)), 4,
                   p, 3, f, 2, dense(dx, 3, 8), DL_SET), DL_OK,
                   "analytic gradient")) return 0;
    const double steps[3] = {0.005, 0.002, 0.001};
    for (size_t j = 0; j < 24; ++j) {
        const float saved = x[j];
        for (size_t h = 0; h < 3; ++h) {
            const float plus = (float)((double)saved + steps[h]);
            const float minus = (float)((double)saved - steps[h]);
            if (!require(plus > minus, "collapsed gradient interval")) return 0;
            double lp = 0.0, lm = 0.0;
            x[j] = plus;
            const DlStatus sp = rotation_objective(x, incoming, p, f, &lp);
            x[j] = minus;
            const DlStatus sm = rotation_objective(x, incoming, p, f, &lm);
            x[j] = saved;
            if (!status_is(sp, DL_OK, "plus objective") ||
                !status_is(sm, DL_OK, "minus objective")) return 0;
            const double numeric = (lp - lm) / ((double)plus - (double)minus);
            if (!derivative(dx[j], numeric, worst)) return 0;
            ++*checks;
        }
    }
    double replay = 0.0;
    return status_is(rotation_objective(x, incoming, p, f, &replay), DL_OK,
                     "replayed objective") &&
           near_value(replay, baseline, 0.0, "restored inputs");
}

static DlStatus score_objective(float *q, float *k, const double *f,
                                 float *rq, float *rk, double *score)
{
    const size_t qp[1] = {3}, kp[1] = {9};
    DlStatus s = dl_rope_forward(dl_matrix_read(dense(q, 1, 4)), 4, qp, 1,
                                 f, 2, dense(rq, 1, 4));
    if (s != DL_OK) return s;
    s = dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4, kp, 1,
                        f, 2, dense(rk, 1, 4));
    if (s != DL_OK) return s;
    *score = dot(rq, rk, 4);
    return DL_OK;
}

static int dot_gradients(size_t *checks, double *worst)
{
    float q[4] = {1, 2, 3, 4}, k[4] = {-2, 0.5f, 1, -1};
    float rq[4], rk[4], dq[4], dk[4];
    double f[2], score = 0.0;
    const size_t qp[1] = {3}, kp[1] = {9};
    if (!status_is(dl_rope_frequencies(4, 10000.0, f, 2), DL_OK, "score f") ||
        !status_is(score_objective(q, k, f, rq, rk, &score), DL_OK, "score") ||
        !status_is(dl_rope_backward(dl_matrix_read(dense(rk, 1, 4)), 4, qp, 1,
                   f, 2, dense(dq, 1, 4), DL_SET), DL_OK, "score dQ") ||
        !status_is(dl_rope_backward(dl_matrix_read(dense(rq, 1, 4)), 4, kp, 1,
                   f, 2, dense(dk, 1, 4), DL_SET), DL_OK, "score dK")) return 0;
    const double baseline = score;
    const double steps[3] = {0.005, 0.002, 0.001};
    for (int which = 0; which < 2; ++which) {
        float *parameter = which == 0 ? q : k;
        const float *gradient = which == 0 ? dq : dk;
        for (size_t j = 0; j < 4; ++j) {
            const float saved = parameter[j];
            for (size_t h = 0; h < 3; ++h) {
                const float plus = (float)((double)saved + steps[h]);
                const float minus = (float)((double)saved - steps[h]);
                if (!require(plus > minus, "collapsed score interval")) return 0;
                double lp = 0.0, lm = 0.0;
                parameter[j] = plus;
                const DlStatus sp = score_objective(q, k, f, rq, rk, &lp);
                parameter[j] = minus;
                const DlStatus sm = score_objective(q, k, f, rq, rk, &lm);
                parameter[j] = saved;
                if (!status_is(sp, DL_OK, "score plus") ||
                    !status_is(sm, DL_OK, "score minus")) return 0;
                const double numeric = (lp - lm) / ((double)plus - (double)minus);
                if (!derivative(gradient[j], numeric, worst)) return 0;
                ++*checks;
            }
        }
    }
    return status_is(score_objective(q, k, f, rq, rk, &score), DL_OK,
                     "score replay") &&
           near_value(score, baseline, 0.0, "restored score");
}

int main(void)
{
    size_t checks = 0;
    double worst = 0.0;
    if (!frequencies_and_hand() || !padded_modes_and_norms() ||
        !sequence_and_positions() || !widths_and_limits() || !relative_dot() ||
        !rejected_calls() || !partial_failures() ||
        !isolated_gradients(&checks, &worst) || !dot_gradients(&checks, &worst)) {
        return 1;
    }
    printf("RoPE finite differences: %zu comparisons, worst error/tolerance = %.6f\n",
           checks, worst);
    puts("RoPE checks passed");
    return 0;
}
