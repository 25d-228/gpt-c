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
