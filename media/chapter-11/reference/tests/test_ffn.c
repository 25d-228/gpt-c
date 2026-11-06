#include "dl/ffn.h"
#include "dl/norm.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
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

/* Test-only fixed storage; no dynamically owned pointers or stored views. */
enum { X, WG, WV, WO, A, B, H, Y, DY, DH, DA, DB, DX, DWG, DWV, DWO, NSLOTS };
typedef struct {
    size_t m, d, f;
    float cell[NSLOTS][64];
} FfnCase;

static DlMatrix dense(float *p, size_t rows, size_t cols)
{
    return (DlMatrix){p, rows * cols, 0, rows, cols, cols};
}

/* Only the small shapes constructed below fit these 64-float arrays. */
static DlMatrix view(FfnCase *q, int slot)
{
    size_t rows = q->m, cols = q->d;
    if (slot == WG || slot == WV || slot == DWG || slot == DWV) {
        rows = q->f;
    } else if (slot == WO || slot == DWO) {
        rows = q->d; cols = q->f;
    } else if (slot == A || slot == B || slot == H ||
               slot == DH || slot == DA || slot == DB) {
        cols = q->f;
    }
    return (DlMatrix){q->cell[slot], 64, 2, rows, cols, cols + 2};
}

static DlConstMatrix read_view(FfnCase *q, int slot)
{
    return dl_matrix_read(view(q, slot));
}

static void fill(DlMatrix v, float value)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) v.base[v.offset + i * v.ld + j] = value;
    }
}

static void load(DlMatrix v, const float *values)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j)
            v.base[v.offset + i * v.ld + j] = values[i * v.cols + j];
    }
}

static int expect(DlMatrix v, const double *values, double initial,
                   double factor, const char *where)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            const double want = initial + factor * values[i * v.cols + j];
            if (!near_value(v.base[v.offset + i * v.ld + j], want,
                            3e-6 + 2e-6 * fabs(want), where)) return 0;
        }
    }
    return 1;
}

static void reset_case(FfnCase *q, size_t m, size_t d, size_t f)
{
    *q = (FfnCase){.m = m, .d = d, .f = f};
    for (int slot = 0; slot < NSLOTS; ++slot) {
        for (size_t j = 0; j < 64; ++j) q->cell[slot][j] = -7777.0f;
        fill(view(q, slot), NAN);
    }
}

static int guards(FfnCase *q)
{
    for (int slot = 0; slot < NSLOTS; ++slot) {
        const DlMatrix v = view(q, slot);
        for (size_t j = 0; j < v.count; ++j) {
            const int active = j >= v.offset && (j - v.offset) / v.ld < v.rows
                             && (j - v.offset) % v.ld < v.cols;
            if (!active && !require(v.base[j] == -7777.0f, "padding changed")) return 0;
        }
    }
    return 1;
}

static DlFfnWeights weights(FfnCase *q)
{
    return (DlFfnWeights){read_view(q, WG), read_view(q, WV), read_view(q, WO)};
}

static DlFfnCache cache(FfnCase *q)
{
    return (DlFfnCache){view(q, A), view(q, B), view(q, H)};
}

static DlFfnScratch scratch(FfnCase *q)
{
    return (DlFfnScratch){view(q, DH), view(q, DA), view(q, DB)};
}

static DlFfnGrads grads(FfnCase *q)
{
    return (DlFfnGrads){view(q, DWG), view(q, DWV), view(q, DWO)};
}

static DlStatus forward(FfnCase *q, DlBackend backend)
{
    return dl_ffn_forward(backend, read_view(q, X), weights(q), cache(q), view(q, Y));
}

static DlStatus backward(FfnCase *q, DlBackend backend,
                          DlWriteMode xm, DlWriteMode wm)
{
    return dl_ffn_backward(backend, read_view(q, X), weights(q), cache(q),
        read_view(q, DY), scratch(q), view(q, DX), xm, grads(q), wm);
}

static int unchanged_reads(const FfnCase *q, const FfnCase *saved)
{
    const int slots[8] = {X, WG, WV, WO, A, B, H, DY};
    for (size_t i = 0; i < 8; ++i) {
        if (!require(memcmp(q->cell[slots[i]], saved->cell[slots[i]],
                            sizeof q->cell[0]) == 0, "backward changed an input")) return 0;
    }
    return 1;
}

static void hand_case(FfnCase *q)
{
    reset_case(q, 1, 2, 2);
    const float x[2] = {1, 2}, wg[4] = {(float)log(2.0), 0, 1, -0.5f};
    const float wv[4] = {2, 0, -1, 2}, wo[4] = {1, -1, 0.5f, 2};
    const float g[2] = {2, -1};
    load(view(q, X), x); load(view(q, WG), wg); load(view(q, WV), wv);
    load(view(q, WO), wo); load(view(q, DY), g);
}

static int hand_and_modes(void)
{
    const double l = log(2.0), h = 4.0 * l / 3.0, da0 = 2.0 + 2.0 * l / 3.0;
    const double av[2] = {l, 0}, bv[2] = {2, 3}, hv[2] = {h, 0}, yv[2] = {h, h/2};
    const double dh[2] = {1.5, -4}, da[2] = {da0, -6}, db[2] = {l, 0};
    const double dx[2] = {da0 * l - 6.0 + 2.0 * l, 3};
    const double dg[4] = {da0, 2*da0, -6, -12}, dv[4] = {l, 2*l, 0, 0};
    const double dout[4] = {2*h, 0, -h, 0};
    for (int backend = 0; backend < 2; ++backend) {
        FfnCase q; hand_case(&q);
        if (!status_is(forward(&q, (DlBackend)backend), DL_OK, "hand forward") ||
            !expect(view(&q, A), av, 0, 1, "A") || !expect(view(&q, B), bv, 0, 1, "B") ||
            !expect(view(&q, H), hv, 0, 1, "H") || !expect(view(&q, Y), yv, 0, 1, "Y")) return 0;
        const FfnCase saved = q; /* Safe: owns inline arrays, no pointers. */
        for (int xm = 0; xm < 2; ++xm) {
            for (int wm = 0; wm < 2; ++wm) {
                fill(view(&q, DX), xm ? 3.0f : NAN);
                for (int k = DWG; k <= DWO; ++k) fill(view(&q, k), wm ? 5.0f : NAN);
                for (int k = DH; k <= DB; ++k) fill(view(&q, k), NAN);
                if (!status_is(backward(&q, (DlBackend)backend, xm ? DL_ADD : DL_SET,
                        wm ? DL_ADD : DL_SET), DL_OK, "hand backward") ||
                    !expect(view(&q, DH), dh, 0, 1, "dH") ||
                    !expect(view(&q, DA), da, 0, 1, "dA") ||
                    !expect(view(&q, DB), db, 0, 1, "dB") ||
                    !expect(view(&q, DX), dx, xm ? 3 : 0, 1, "dX merge") ||
                    !expect(view(&q, DWG), dg, wm ? 5 : 0, 1, "dWg") ||
                    !expect(view(&q, DWV), dv, wm ? 5 : 0, 1, "dWv") ||
                    !expect(view(&q, DWO), dout, wm ? 5 : 0, 1, "dWo") ||
                    !unchanged_reads(&q, &saved) || !guards(&q)) return 0;
            }
        }
        if (!status_is(backward(&q, (DlBackend)backend, DL_ADD, DL_ADD), DL_OK, "repeat ADD") ||
            !expect(view(&q, DX), dx, 3, 2, "repeated dX") ||
            !expect(view(&q, DWG), dg, 5, 2, "repeated dWg") ||
            !expect(view(&q, DWV), dv, 5, 2, "repeated dWv") ||
            !expect(view(&q, DWO), dout, 5, 2, "repeated dWo") ||
            !unchanged_reads(&q, &saved) || !guards(&q)) return 0;
    }
    printf("hand row: A=[%.6f, 0], B=[2, 3], H=[%.6f, 0]\n", l, h);
    printf("hand backward: dA=[%.6f, -6], dB=[%.6f, 0], dX=[%.6f, 3]\n", da0, l, dx[0]);
    puts("both backends, padded views, independent modes, and repeated ADD passed");
    return 1;
}

static int activation_edges(void)
{
    FfnCase q; reset_case(&q, 8, 2, 1);
    const float a[8] = {-100, -40, -2, -1e-4f, 0, 1e-4f, 40, 100};
    const float wg[2] = {1, 0}, wv[2] = {0, 1}, wo[2] = {1, 0};
    load(view(&q, WG), wg); load(view(&q, WV), wv); load(view(&q, WO), wo);
    const DlMatrix x = view(&q, X), g = view(&q, DY);
    for (size_t i = 0; i < 8; ++i) {
        x.base[x.offset + i*x.ld] = a[i]; x.base[x.offset + i*x.ld + 1] = 1;
        g.base[g.offset + i*g.ld] = 1; g.base[g.offset + i*g.ld + 1] = 0;
    }
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "activation edges forward") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_OK, "activation edges backward")) return 0;
    const DlMatrix y = view(&q, Y), dx = view(&q, DX);
    for (size_t i = 0; i < 8; ++i) {
        /* Independent direct formula is safe on this bounded [-100,100] fixture. */
        const double v = (double)a[i], s = 1.0 / (1.0 + exp(-v));
        const double want_y = v*s, want_dx = s + v*s*(1.0-s);
        if (!near_value(y.base[y.offset + i*y.ld], want_y,
                        8.0*(double)FLT_TRUE_MIN + 2e-6*fabs(want_y), "SiLU value") ||
            !near_value(dx.base[dx.offset + i*dx.ld], want_dx,
                        8.0*(double)FLT_TRUE_MIN + 2e-6*fabs(want_dx), "SiLU slope") ||
            !near_value(dx.base[dx.offset + i*dx.ld + 1], want_y,
                        8.0*(double)FLT_TRUE_MIN + 2e-6*fabs(want_y), "value-branch slope")) return 0;
    }
    if (!require(dx.base[dx.offset + 2*dx.ld] < 0, "SiLU slope at -2 must be negative") ||
        !require(dx.base[dx.offset + 4*dx.ld] == 0.5f, "SiLU slope at zero must be 0.5") ||
        !guards(&q)) return 0;
    /* Very large negative input must not call exp(+FLT_MAX). Underflow to zero is valid. */
    x.base[x.offset] = -FLT_MAX;
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "extreme negative forward") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_OK, "extreme negative backward") ||
        !near_value(y.base[y.offset], 0, 0, "negative saturation") ||
        !near_value(dx.base[dx.offset], 0, 0, "negative saturated slope")) return 0;
    puts("negative, near-zero, saturated, and extreme finite gates passed");
    return 1;
}

static int no_mutation(const FfnCase *q, const FfnCase *saved, const char *where)
{
    return require(memcmp(q->cell, saved->cell, sizeof q->cell) == 0, where);
}

static int rejected_calls(void)
{
    FfnCase q; hand_case(&q);
    const FfnCase empty = q;
    DlFfnWeights w = weights(&q);
    DlFfnCache c = cache(&q);
    DlMatrix bad = view(&q, Y); bad.cols = 1;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, bad),
            DL_SHAPE, "late forward shape") || !no_mutation(&q, &empty, "shape mutated cache")) return 0;
    c.hidden = c.gate;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, view(&q, Y)),
            DL_ALIAS, "cache overlap") || !no_mutation(&q, &empty, "alias mutated cache")) return 0;
    c = cache(&q); c.value.offset = 63;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, view(&q, Y)),
            DL_RANGE, "range") || !no_mutation(&q, &empty, "range mutated cache")) return 0;
    c = cache(&q); c.gate.rows = SIZE_MAX;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, view(&q, Y)),
            DL_OVERFLOW, "view size overflow") || !no_mutation(&q, &empty, "overflow mutated cache")) return 0;
    if (!status_is(forward(&q, (DlBackend)99), DL_ARGUMENT, "backend") ||
        !no_mutation(&q, &empty, "bad backend wrote")) return 0;
    DlMatrix wo = view(&q, WO); wo.base[wo.offset] = INFINITY;
    const FfnCase nonfinite = q;
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_NUMERIC, "nonfinite last weight") ||
        !no_mutation(&q, &nonfinite, "preflight nonfinite wrote cache")) return 0;
    hand_case(&q);
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "prepare backward rejection")) return 0;
    const FfnCase saved = q;
    DlFfnGrads dw = grads(&q); dw.down.cols = 1;
    if (!status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), weights(&q),
            cache(&q), read_view(&q, DY), scratch(&q), view(&q, DX), DL_SET, dw, DL_SET),
            DL_SHAPE, "late backward shape") || !no_mutation(&q, &saved, "backward shape wrote")) return 0;
    DlFfnScratch work = scratch(&q); work.gate = view(&q, A);
    if (!status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), weights(&q),
            cache(&q), read_view(&q, DY), work, view(&q, DX), DL_SET, grads(&q), DL_SET),
            DL_ALIAS, "scratch aliases saved A") || !no_mutation(&q, &saved, "scratch alias wrote")) return 0;
    dw = grads(&q); dw.value = dw.gate;
    if (!status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), weights(&q),
            cache(&q), read_view(&q, DY), scratch(&q), view(&q, DX), DL_SET, dw, DL_SET),
            DL_ALIAS, "gradient outputs overlap") || !no_mutation(&q, &saved, "gradient alias wrote")) return 0;
    if (!status_is(backward(&q, DL_BACKEND_REF, (DlWriteMode)99, DL_SET), DL_ARGUMENT, "dx mode") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, (DlWriteMode)99), DL_ARGUMENT, "weight mode") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_ADD, DL_SET), DL_NUMERIC, "NaN old dx") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_ADD), DL_NUMERIC, "NaN old weights") ||
        !no_mutation(&q, &saved, "rejected mode/old destination wrote")) return 0;
    DlMatrix h = view(&q, H); h.base[h.offset] = NAN;
    const FfnCase bad_saved = q;
    if (!status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_NUMERIC, "NaN saved hidden") ||
        !no_mutation(&q, &bad_saved, "bad saved values wrote")) return 0;

    /* Input weights may share values, but derivative outputs stay separate. */
    hand_case(&q); w = weights(&q); w.value = w.gate;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, cache(&q), view(&q, Y)),
            DL_OK, "shared read-only weights") ||
        !status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), w, cache(&q),
            read_view(&q, DY), scratch(&q), view(&q, DX), DL_SET, grads(&q), DL_SET),
            DL_OK, "shared weights backward") || !guards(&q)) return 0;
    puts("shape, alias, range, overflow, mode, and nonfinite preflight checks passed");
    return 1;
}

static int partial_failures(void)
{
    FfnCase q; reset_case(&q, 1, 1, 1);
    fill(view(&q, X), 1); fill(view(&q, WG), FLT_MAX);
    fill(view(&q, WV), 2); fill(view(&q, WO), 0); fill(view(&q, DY), 1);
    fill(view(&q, H), 9); fill(view(&q, Y), 9);
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_NUMERIC, "hidden overflow despite zero down weight") ||
        !require(q.cell[A][2] == FLT_MAX && q.cell[B][2] == 2 &&
                 q.cell[H][2] == 9 && q.cell[Y][2] == 9, "unexpected forward partial state")) return 0;
    /* All forward intermediates are representable; dWo overflows afterward. */
    reset_case(&q, 1, 1, 1);
    fill(view(&q, X), 1); fill(view(&q, WG), 100);
    fill(view(&q, WV), 1); fill(view(&q, WO), 0); fill(view(&q, DY), FLT_MAX);
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "finite zero-output forward")) return 0;
    for (int slot = DH; slot < NSLOTS; ++slot) fill(view(&q, slot), 9);
    if (!status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_NUMERIC, "dWo overflow") ||
        !require(q.cell[DH][2] == 0 && q.cell[DWO][2] == 9 && q.cell[DX][2] == 9,
                 "unexpected backward partial state") || !guards(&q)) return 0;
    puts("multi-output partial numerical failures detected; damaged results discarded");
    return 1;
}

static void ordinary_case(FfnCase *q, int variant)
{
    reset_case(q, 2, 3, 5);
    const float x[6] = {0.7f, -0.5f, 1.2f, -0.8f, 0.4f, 0.3f};
    const float wg[15] = {0.6f,-0.4f,0.2f, -0.5f,0.3f,0.7f,
        1e-4f,-2e-4f,3e-4f, -0.2f,-0.1f,0.3f, 0.4f,-0.6f,-0.3f};
    const float wv[15] = {0.2f,0.5f,-0.3f, -0.4f,0.6f,0.1f,
        0.7f,-0.2f,0.4f, -0.3f,0.2f,0.5f, 0.6f,0.1f,-0.4f};
    const float wo[15] = {0.3f,-0.2f,0.4f,0.1f,-0.5f,
        -0.4f,0.6f,0.2f,-0.3f,0.1f, 0.2f,0.1f,-0.5f,0.4f,0.3f};
    const float g[6] = {0.3f,-0.4f,0.7f, -0.6f,0.2f,0.5f};
    load(view(q, X), x); load(view(q, WG), wg); load(view(q, WV), wv);
    load(view(q, WO), wo); load(view(q, DY), g);
    if (variant == 1) fill(view(q, WG), 0);
    if (variant == 2) fill(view(q, WV), 0);
}

/* Integration case: Z = U + FFN(RMSNorm(U, scale)). Inline owners only. */
typedef struct {
    FfnCase f;
    float raw[6], scale[3], z[6], draw[6], dscale[3];
} Graph;

static void graph_init(Graph *g, int variant)
{
    *g = (Graph){.raw = {0.7f,-0.5f,1.2f, -0.8f,0.4f,0.3f},
                 .scale = {1.2f,-0.7f,0.8f}};
    ordinary_case(&g->f, variant);
}

static DlStatus graph_forward(Graph *g, DlBackend backend, int residual, double *objective)
{
    FfnCase *q = &g->f;
    DlStatus s = DL_OK;
    if (residual) s = dl_rmsnorm_forward(dl_matrix_read(dense(g->raw, 2, 3)),
        dl_matrix_read(dense(g->scale, 1, 3)), 0.125f, view(q, X));
    if (s == DL_OK) s = forward(q, backend);
    if (s == DL_OK && residual) s = dl_residual_add(dl_matrix_read(dense(g->raw, 2, 3)),
        read_view(q, Y), dense(g->z, 2, 3));
    if (s != DL_OK) return s;
    const DlConstMatrix y = residual ? dl_matrix_read(dense(g->z, 2, 3)) : read_view(q, Y);
    const DlConstMatrix dy = read_view(q, DY);
    double sum = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j)
            sum += (double)y.base[y.offset + i*y.ld + j] * (double)dy.base[dy.offset + i*dy.ld + j];
    }
    if (!isfinite(sum)) return DL_NUMERIC;
    *objective = sum;
    return DL_OK;
}

static DlStatus graph_backward(Graph *g, DlBackend backend, int residual)
{
    DlStatus s = backward(&g->f, backend, DL_SET, DL_SET);
    if (s == DL_OK && residual) s = dl_residual_backward(read_view(&g->f, DY),
        dense(g->draw, 2, 3), DL_SET);
    if (s == DL_OK && residual) s = dl_rmsnorm_backward(
        dl_matrix_read(dense(g->raw, 2, 3)), dl_matrix_read(dense(g->scale, 1, 3)),
        read_view(&g->f, DX), 0.125f, dense(g->draw, 2, 3), DL_ADD,
        dense(g->dscale, 1, 3), DL_SET);
    return s;
}

static DlMatrix parameter(Graph *g, size_t group, int residual)
{
    if (group == 0) return residual ? dense(g->raw, 2, 3) : view(&g->f, X);
    if (group == 4) return dense(g->scale, 1, 3);
    const int slots[3] = {WG, WV, WO};
    return view(&g->f, slots[group - 1]);
}

static DlMatrix derivative(Graph *g, size_t group, int residual)
{
    if (group == 0) return residual ? dense(g->draw, 2, 3) : view(&g->f, DX);
    if (group == 4) return dense(g->dscale, 1, 3);
    const int slots[3] = {DWG, DWV, DWO};
    return view(&g->f, slots[group - 1]);
}

static int numerical_case(int variant, int residual, DlBackend backend,
                           size_t *comparisons, double *worst)
{
    Graph g; graph_init(&g, variant);
    double baseline = 0;
    if (!status_is(graph_forward(&g, backend, residual, &baseline), DL_OK, "gradient baseline") ||
        !status_is(graph_backward(&g, backend, residual), DL_OK, "analytic backward")) return 0;
    Graph original = g; /* This value copy duplicates inline arrays, not pointers. */
    const char *names[5] = {"input", "Wg", "Wv", "Wo", "scale"};
    const double steps[3] = {0.01, 0.003, 0.001};
    const size_t groups = residual ? 5 : 4;
    for (size_t group = 0; group < groups; ++group) {
        const DlMatrix p = parameter(&g, group, residual), d = derivative(&g, group, residual);
        for (size_t i = 0; i < p.rows; ++i) {
            for (size_t j = 0; j < p.cols; ++j) {
                float *cell = &p.base[p.offset + i*p.ld + j];
                const float saved = *cell;
                const double analytic = (double)d.base[d.offset + i*d.ld + j];
                for (size_t k = 0; k < 3; ++k) {
                    const double h = steps[k] * fmax(1.0, fabs((double)saved));
                    const float hi = (float)((double)saved + h), lo = (float)((double)saved - h);
                    const double spacing = (double)hi - (double)lo;
                    double plus = 0, minus = 0;
                    *cell = hi;
                    const DlStatus sp = graph_forward(&g, backend, residual, &plus);
                    *cell = lo;
                    const DlStatus sm = graph_forward(&g, backend, residual, &minus);
                    *cell = saved; /* Restore before ANY status/error return. */
                    if (!status_is(sp, DL_OK, "plus") || !status_is(sm, DL_OK, "minus") ||
                        !require(spacing > 0, "collapsed finite-difference interval")) return 0;
                    const double numerical = (plus - minus) / spacing;
                    const double tolerance = 8e-5 + 3e-3 * fmax(fabs(analytic), fabs(numerical));
                    const double ratio = fabs(analytic - numerical) / tolerance;
                    if (!isfinite(analytic) || !isfinite(numerical) || ratio > 1.0) {
                        fprintf(stderr, "variant=%d residual=%d backend=%d %s[%zu,%zu] h=%.3g: "
                            "analytic=%.9g numeric=%.9g ratio=%.4g\n", variant, residual,
                            (int)backend, names[group], i, j, h, analytic, numerical, ratio);
                        return 0;
                    }
                    if (ratio > *worst) *worst = ratio;
                    ++*comparisons;
                }
            }
        }
    }
    /* Restoring parameters does NOT restore the last perturbed forward cache. */
    for (size_t group = 0; group < groups; ++group) {
        const DlMatrix p = parameter(&g, group, residual), old = parameter(&original, group, residual);
        for (size_t i = 0; i < p.rows; ++i) {
            for (size_t j = 0; j < p.cols; ++j)
                if (!require(p.base[p.offset+i*p.ld+j] == old.base[old.offset+i*old.ld+j],
                             "parameter not restored")) return 0;
        }
    }
    double replay = 0;
    if (!status_is(graph_forward(&g, backend, residual, &replay), DL_OK, "restore baseline cache") ||
        !near_value(replay, baseline, 0, "baseline replay") ||
        !status_is(graph_backward(&g, backend, residual), DL_OK, "backward after replay")) return 0;
    for (size_t group = 0; group < groups; ++group) {
        const DlMatrix d = derivative(&g, group, residual), old = derivative(&original, group, residual);
        for (size_t i = 0; i < d.rows; ++i) {
            for (size_t j = 0; j < d.cols; ++j)
                if (!near_value(d.base[d.offset+i*d.ld+j], old.base[old.offset+i*old.ld+j],
                                0, "gradient replay")) return 0;
        }
    }
    return guards(&g.f);
}

int main(void)
{
    if (!hand_and_modes() || !activation_edges() || !rejected_calls() || !partial_failures()) return 1;
    size_t comparisons = 0;
    double worst = 0;
    for (int backend = 0; backend < 2; ++backend) {
        for (int variant = 0; variant < 3; ++variant) {
            if (!numerical_case(variant, 0, (DlBackend)backend, &comparisons, &worst)) return 1;
        }
        if (!numerical_case(0, 1, (DlBackend)backend, &comparisons, &worst)) return 1;
    }
    printf("SwiGLU/full residual finite differences: %zu passed; worst error/tolerance=%.6g\n",
           comparisons, worst);
    puts("all feed-forward checks passed");
    return 0;
}
