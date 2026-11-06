#include "dl/attention.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { X, WQ, WK, WV, WO, Q0, K0, V, Q, K, SCORES, PROBS, CONTEXT, Y,
       DY, DC, DP, DS, DQ, DK, DV, DQ0, DK0, DX, DWQ, DWK, DWV, DWO, NARRAY };
enum { CAP = 512 };

typedef struct {
    DlAttentionConfig c;
    DlTensor owner[NARRAY];
    DlMatrix m[NARRAY];
    size_t *positions;
    double *freq;
} Fixture;

static size_t full_checks = 0, core_checks = 0;
static double full_error = 0.0, core_error = 0.0;

static int require(int yes, const char *where)
{
    if (!yes) fprintf(stderr, "%s\n", where);
    return yes;
}

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got == want) return 1;
    fprintf(stderr, "%s: %s; expected %s\n", where,
            dl_status_name(got), dl_status_name(want));
    return 0;
}

static int near_value(double got, double want, double tol, const char *where)
{
    if (isfinite(got) && isfinite(want) && fabs(got - want) <= tol) return 1;
    fprintf(stderr, "%s: %.12g; expected %.12g, tolerance %.3g\n", where, got, want, tol);
    return 0;
}

static float *cell(DlMatrix m, size_t i, size_t j)
{
    return &m.base[m.offset + i * m.ld + j];
}

static void fill(DlMatrix m, float value)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) *cell(m, i, j) = value;
    }
}

static void fixture_destroy(Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) dl_tensor_destroy(&f->owner[a]);
    free(f->positions);
    free(f->freq);
    *f = (Fixture){0};
}

/* Small bounded TEST fixtures, not library limits. f starts as {0}. */
static int fixture_create(Fixture *f, DlAttentionConfig c, size_t width, size_t pad)
{
    if (c.batch == 0 || c.batch > 2 || c.time == 0 || c.time > 4 ||
        c.query_heads == 0 || c.query_heads > 4 || c.kv_heads == 0 ||
        c.query_heads % c.kv_heads != 0 || c.head_dim == 0 ||
        c.head_dim > 4 || c.head_dim % 2 != 0 || width == 0 || width > 4 || pad > 2)
        return 0;
    f->c = c;
    const size_t m = c.batch * c.time, q = c.query_heads * c.head_dim;
    const size_t k = c.kv_heads * c.head_dim, sr = m * c.query_heads, d = width;
    const size_t rows[NARRAY] = {m,q,k,k,d,m,m,m,m,m,sr,sr,m,m,
                                 m,m,sr,sr,m,m,m,m,m,m,q,k,k,d};
    const size_t cols[NARRAY] = {d,d,d,d,q,q,k,k,q,k,c.time,c.time,q,d,
                                 d,q,c.time,c.time,q,k,k,q,k,d,d,d,d,q};
    for (size_t a = 0; a < NARRAY; ++a) {
        const size_t ld = cols[a] + pad, off = pad != 0 ? 1 : 0;
        const size_t count = rows[a] * ld + (pad != 0 ? 2 : 0);
        if (count > CAP) goto fail;
        if (dl_tensor_create(&f->owner[a], 1, &count) != DL_OK) goto fail;
        if (dl_matrix_view(&f->owner[a], off, rows[a], cols[a], ld, &f->m[a]) != DL_OK)
            goto fail;
        for (size_t j = 0; j < count; ++j) f->owner[a].data[j] = -777.0f;
        for (size_t i = 0; i < rows[a]; ++i) {
            for (size_t j = 0; j < cols[a]; ++j) {
                *cell(f->m[a], i, j) = a < Q0 || a == DY
                    ? (float)((int)((11*i + 7*j + 3*a) % 17) - 8) / 16.0f : NAN;
            }
        }
    }
    f->positions = malloc(m * sizeof(*f->positions));
    f->freq = malloc((c.head_dim / 2) * sizeof(*f->freq));
    if (f->positions == NULL || f->freq == NULL) goto fail;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t t = 0; t < c.time; ++t) f->positions[b * c.time + t] = 3 + b + 2*t;
    }
    if (dl_rope_frequencies(c.head_dim, 100000.0, f->freq, c.head_dim / 2) != DL_OK)
        goto fail;
    return 1;
fail:
    fixture_destroy(f);
    return 0;
}

static DlAttentionWeights weights(const Fixture *f)
{
    return (DlAttentionWeights){dl_matrix_read(f->m[WQ]), dl_matrix_read(f->m[WK]),
        dl_matrix_read(f->m[WV]), dl_matrix_read(f->m[WO])};
}

static DlAttentionCache cache(const Fixture *f)
{
    return (DlAttentionCache){f->m[Q0], f->m[K0], f->m[V], f->m[Q], f->m[K],
                              f->m[SCORES], f->m[PROBS], f->m[CONTEXT]};
}

static DlAttentionScratch scratch(const Fixture *f)
{
    return (DlAttentionScratch){f->m[DC], f->m[DP], f->m[DS], f->m[DQ], f->m[DK],
                                f->m[DV], f->m[DQ0], f->m[DK0]};
}

static DlStatus forward(Fixture *f, DlBackend backend)
{
    return dl_attention_forward(backend, f->c, dl_matrix_read(f->m[X]), weights(f),
        f->positions, f->c.batch * f->c.time, f->freq, f->c.head_dim / 2,
        cache(f), f->m[Y]);
}

static DlStatus backward(Fixture *f, DlBackend backend, DlWriteMode xm, DlWriteMode wm)
{
    const DlAttentionWeightGrads dw = {f->m[DWQ], f->m[DWK], f->m[DWV], f->m[DWO]};
    return dl_attention_backward(backend, f->c, dl_matrix_read(f->m[X]), weights(f),
        f->positions, f->c.batch * f->c.time, f->freq, f->c.head_dim / 2,
        cache(f), dl_matrix_read(f->m[DY]), scratch(f), f->m[DX], xm, dw, wm);
}

static int guards(const Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) {
        const DlMatrix m = f->m[a];
        for (size_t j = 0; j < m.count; ++j) {
            const int logical = j >= m.offset && (j-m.offset)/m.ld < m.rows
                             && (j-m.offset)%m.ld < m.cols;
            if (!logical && m.base[j] != -777.0f) return 0;
        }
    }
    return 1;
}

static int matrix_zero(DlMatrix m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            if (*cell(m, i, j) != 0.0f) return 0;
        }
    }
    return 1;
}

static void save_arrays(const Fixture *f, float saved[NARRAY][CAP])
{
    for (size_t a = 0; a < NARRAY; ++a)
        memcpy(saved[a], f->owner[a].data, f->owner[a].count * sizeof(float));
}

static int same_arrays(const Fixture *f, float saved[NARRAY][CAP], size_t first, size_t end)
{
    for (size_t a = first; a < end; ++a) {
        if (memcmp(saved[a], f->owner[a].data, f->owner[a].count * sizeof(float)) != 0)
            return 0;
    }
    return 1;
}

static int mask_invariants(const Fixture *f)
{
    const DlAttentionConfig c = f->c;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                double sum = 0.0, magnitude = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    const double ds = (double)*cell(f->m[DS], sr, s);
                    if (s > t && !require(ds == 0.0 && *cell(f->m[DP], sr, s) == 0.0f,
                                          "masked derivative not zero")) return 0;
                    sum += ds;
                    magnitude += fabs(ds);
                }
                if (!near_value(sum, 0.0, 3e-7 * (1.0 + magnitude), "score row sum")) return 0;
                if (t == 0 && !near_value(*cell(f->m[DS], sr, 0), 0.0, 0.0,
                                         "one-key score derivative")) return 0;
            }
        }
    }
    return 1;
}

static int hand_trace(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){1,2,2,1,2}, 2, 2)) return 0;
    for (size_t a = X; a <= WO; ++a) fill(f.m[a], 0.0f);
    fill(f.m[DY], 0.0f);
    const float r = (float)sqrt(2.0), a = (float)log(2.0);
    *cell(f.m[X], 0, 0) = 1.0f; *cell(f.m[X], 1, 1) = 1.0f;
    *cell(f.m[WQ], 0, 1) = r; *cell(f.m[WQ], 3, 1) = r;
    *cell(f.m[WK], 0, 1) = a; *cell(f.m[WK], 1, 0) = a;
    *cell(f.m[WV], 0, 0) = 1.0f; *cell(f.m[WV], 1, 1) = 2.0f;
    *cell(f.m[WO], 0, 0) = 1.0f; *cell(f.m[WO], 1, 3) = 1.0f;
    *cell(f.m[DY], 1, 0) = 1.0f; *cell(f.m[DY], 1, 1) = 1.0f;
    f.positions[0] = 0; f.positions[1] = 0;
    int ok = status_is(forward(&f, backend), DL_OK, "hand forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "hand backward");
    const double want_ds[4] = {2.0/9, -2.0/9, -4.0/9, 4.0/9};
    const double want_dk[4] = {2.0/9, -4.0/9, -2.0/9, 4.0/9};
    const double want_dv[4] = {1.0/3, 2.0/3, 2.0/3, 1.0/3};
    const double u = log(2.0) / (9.0 * sqrt(2.0));
    const double want_dq[4] = {-2*u, 2*u, 4*u, -4*u};
    const double want_dx[4] = {1.0/3 - 4*log(2.0)/9, 4.0/3 + 2*log(2.0)/9,
                               2.0/3 + 4*log(2.0)/9, 2.0/3 - 8*log(2.0)/9};
    const double want_context[4] = {1.0/3, 4.0/3, 2.0/3, 2.0/3};
    for (size_t i = 0; ok && i < 2; ++i) {
        for (size_t j = 0; ok && j < 2; ++j) {
            ok = near_value(*cell(f.m[DS], i*2+1, j), want_ds[2*i+j], 2e-7, "hand dS") &&
                 near_value(*cell(f.m[DK], i, j), want_dk[2*i+j], 2e-7, "hand shared dK") &&
                 near_value(*cell(f.m[DV], i, j), want_dv[2*i+j], 2e-7, "hand shared dV") &&
                 near_value(*cell(f.m[DX], i, j), want_dx[2*i+j], 3e-7, "hand dX") &&
                 near_value(*cell(f.m[DWK], j, i), want_dk[2*i+j], 2e-7, "hand dWk") &&
                 near_value(*cell(f.m[DWV], j, i), want_dv[2*i+j], 2e-7, "hand dWv");
        }
    }
    for (size_t j = 0; ok && j < 4; ++j) {
        ok = near_value(*cell(f.m[DQ], 1, j), want_dq[j], 2e-7, "hand dQ") &&
             near_value(*cell(f.m[DWQ], j, 1), want_dq[j], 2e-7, "hand dWq") &&
             near_value(*cell(f.m[DQ], 0, j), 0.0, 0.0, "hand unused query") &&
             near_value(*cell(f.m[DWQ], j, 0), 0.0, 0.0, "hand unused query weight");
        for (size_t i = 0; ok && i < 2; ++i)
            ok = near_value(*cell(f.m[DWO], i, j), want_context[j], 2e-7, "hand dWo");
    }
    ok = ok && mask_invariants(&f) && require(guards(&f), "hand guard modified");
    if (ok) printf("hand shared dK[0] (%s): [%.6f, %.6f]\n",
        backend == DL_BACKEND_REF ? "reference" : "blocked",
        (double)*cell(f.m[DK], 0, 0), (double)*cell(f.m[DK], 0, 1));
    fixture_destroy(&f);
    return ok;
}

/* Scalar linear objective: its gradient at Y is exactly the supplied DY. */
static double full_objective(Fixture *f, DlBackend backend)
{
    if (forward(f, backend) != DL_OK) return NAN;
    double sum = 0.0;
    for (size_t i = 0; i < f->m[Y].rows; ++i) {
        for (size_t j = 0; j < f->m[Y].cols; ++j)
            sum += (double)*cell(f->m[Y], i, j) * (double)*cell(f->m[DY], i, j);
    }
    return sum;
}

/* Independent local objective on ROTATED Q/K/V, holding dC fixed.
 * No production attention/softmax call. Match the forward's float boundaries.
 * These tiny fixtures have T<=4; array capacities are checked at construction. */
static double core_objective(const Fixture *f)
{
    const DlAttentionConfig c = f->c;
    double loss = 0.0;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / (c.query_heads / c.kv_heads);
            for (size_t t = 0; t < c.time; ++t) {
                float score[4] = {0}, prob[4] = {0};
                double maximum = -INFINITY, total = 0.0;
                for (size_t s = 0; s <= t; ++s) {
                    double dot = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j)
                        dot += (double)*cell(f->m[Q], b*c.time+t, h*c.head_dim+j)
                             * (double)*cell(f->m[K], b*c.time+s, kh*c.head_dim+j);
                    score[s] = (float)(dot / sqrt((double)c.head_dim));
                    maximum = fmax(maximum, (double)score[s]);
                }
                for (size_t s = 0; s <= t; ++s) total += exp((double)score[s] - maximum);
                for (size_t s = 0; s <= t; ++s)
                    prob[s] = (float)(exp((double)score[s] - maximum) / total);
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double sum = 0.0;
                    for (size_t s = 0; s <= t; ++s)
                        sum += (double)prob[s] * (double)*cell(f->m[V], b*c.time+s, kh*c.head_dim+j);
                    const float context = (float)sum;
                    loss += (double)context * (double)*cell(f->m[DC], b*c.time+t, h*c.head_dim+j);
                }
            }
        }
    }
    return loss;
}

/* Every coordinate is checked at EVERY step; do not select only the best one. */
static int differences(Fixture *f, DlBackend backend, int core)
{
    const size_t inputs[5] = {core ? Q : X, core ? K : WQ, core ? V : WK, WV, WO};
    const size_t grads[5] = {core ? DQ : DX, core ? DK : DWQ, core ? DV : DWK, DWV, DWO};
    const double steps[3] = {0.01, 0.005, 0.0025};
    const size_t n = core ? 3 : 5;
    for (size_t a = 0; a < n; ++a) {
        const DlMatrix v = f->m[inputs[a]], g = f->m[grads[a]];
        for (size_t i = 0; i < v.rows; ++i) {
            for (size_t j = 0; j < v.cols; ++j) {
                float *p = cell(v, i, j);
                const float old = *p;
                const double analytic = (double)*cell(g, i, j);
                for (size_t step = 0; step < 3; ++step) {
                    const double eps = steps[step] * fmax(1.0, fabs((double)old));
                    const float plus = (float)((double)old + eps), minus = (float)((double)old - eps);
                    *p = plus;
                    const double high = core ? core_objective(f) : full_objective(f, backend);
                    *p = minus;
                    const double low = core ? core_objective(f) : full_objective(f, backend);
                    *p = old;
                    const double span = (double)plus - (double)minus;
                    const double numeric = (high - low) / span;
                    const double error = fabs(analytic - numeric);
                    const double tol = 2e-5 + 0.002 * fmax(fabs(analytic), fabs(numeric));
                    if (!(span > 0.0) || !isfinite(numeric) || !isfinite(analytic) || error > tol) {
                        fprintf(stderr, "%s gradient array %zu [%zu,%zu], h %.4g: %.9g vs %.9g (tol %.3g)\n",
                            core ? "core" : "full", inputs[a], i, j, steps[step], analytic, numeric, tol);
                        return 0;
                    }
                    if (core) { ++core_checks; core_error = fmax(core_error, error); }
                    else { ++full_checks; full_error = fmax(full_error, error); }
                }
            }
        }
    }
    return 1;
}

static int gradient_case(DlBackend backend, DlAttentionConfig c, size_t d, size_t pad)
{
    Fixture f = {0};
    if (!fixture_create(&f, c, d, pad)) return 0;
    int ok = status_is(forward(&f, backend), DL_OK, "gradient forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "gradient backward") &&
             mask_invariants(&f) && differences(&f, backend, 0);
    /* Last perturbed forward left a stale tape. Restore the baseline before reuse. */
    ok = ok && status_is(forward(&f, backend), DL_OK, "restore baseline") &&
         status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "restore derivatives") &&
         differences(&f, backend, 1) && require(guards(&f), "gradient padding modified");
    fixture_destroy(&f);
    return ok;
}

static int accumulation(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2,3,4,2,2}, 3, 2)) return 0;
    float before[NARRAY][CAP], baseline[NARRAY][CAP];
    int ok = status_is(forward(&f, backend), DL_OK, "accumulation forward");
    save_arrays(&f, before);
    ok = ok && status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "baseline backward") &&
         require(same_arrays(&f, before, X, DC), "backward changed read-only arrays");
    save_arrays(&f, baseline);
    for (int xm = 0; ok && xm < 2; ++xm) {
        for (int wm = 0; ok && wm < 2; ++wm) {
            for (size_t a = DC; a < NARRAY; ++a) fill(f.m[a], NAN);
            if (xm != 0) fill(f.m[DX], 0.25f);
            if (wm != 0) {
                for (size_t a = DWQ; a <= DWO; ++a) fill(f.m[a], -0.5f);
            }
            for (size_t rep = 1; ok && rep <= 2; ++rep) {
                ok = status_is(backward(&f, backend, (DlWriteMode)xm, (DlWriteMode)wm),
                                DL_OK, "mode/repeated backward");
                for (size_t a = DX; ok && a < NARRAY; ++a) {
                    const int add = a == DX ? xm : wm;
                    const double seed = a == DX ? 0.25 : -0.5;
                    const DlMatrix m = f.m[a];
                    for (size_t i = 0; ok && i < m.rows; ++i) {
                        for (size_t j = 0; ok && j < m.cols; ++j) {
                            const size_t index = m.offset + i*m.ld + j;
                            const double want = (add ? seed : 0.0)
                                              + (add ? (double)rep : 1.0) * (double)baseline[a][index];
                            ok = near_value(*cell(m, i, j), want, 3e-7, "SET/ADD or repeated contribution");
                        }
                    }
                }
                ok = ok && require(same_arrays(&f, baseline, DC, DX), "scratch must overwrite") &&
                     require(same_arrays(&f, before, X, DC), "saved input or tape changed");
            }
        }
    }
    ok = ok && require(guards(&f), "accumulation padding changed");
    fixture_destroy(&f);
    return ok;
}

static void copy_matrix(DlMatrix dst, DlMatrix src)
{
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) *cell(dst, i, j) = *cell(src, i, j);
    }
}

static int same_matrix(DlMatrix a, DlMatrix b, const char *where)
{
    if (a.rows != b.rows || a.cols != b.cols) return 0;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            if (!near_value(*cell(a, i, j), *cell(b, i, j), 3e-7, where)) return 0;
        }
    }
    return 1;
}

static int matched_heads(DlBackend backend, size_t kv)
{
    Fixture f = {0}, expanded = {0};
    const DlAttentionConfig c = {2,3,4,kv,2};
    if (!fixture_create(&f, c, 3, 1)) return 0;
    DlAttentionConfig ec = c;
    ec.kv_heads = c.query_heads;
    if (!fixture_create(&expanded, ec, 3, 2)) { fixture_destroy(&f); return 0; }
    const size_t shared[] = {X, WQ, WO, DY};
    for (size_t a = 0; a < 4; ++a) copy_matrix(expanded.m[shared[a]], f.m[shared[a]]);
    const size_t group = c.query_heads / c.kv_heads;
    for (size_t a = WK; a <= WV; ++a) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t j = 0; j < c.head_dim; ++j) {
                for (size_t col = 0; col < f.m[a].cols; ++col)
                    *cell(expanded.m[a], h*c.head_dim+j, col) =
                        *cell(f.m[a], (h/group)*c.head_dim+j, col);
            }
        }
    }
    int ok = status_is(forward(&f, backend), DL_OK, "shared forward") &&
             status_is(forward(&expanded, backend), DL_OK, "expanded forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "shared backward") &&
             status_is(backward(&expanded, backend, DL_SET, DL_SET), DL_OK, "expanded backward");
    const size_t equal[] = {Y, DX, DWQ, DWO, DQ};
    for (size_t a = 0; ok && a < 5; ++a)
        ok = same_matrix(f.m[equal[a]], expanded.m[equal[a]], "matched shared/nonshared result");
    const size_t reduce[] = {DK, DV, DK0, DWK, DWV};
    for (size_t a = 0; ok && a < 5; ++a) {
        const size_t which = reduce[a];
        for (size_t kh = 0; ok && kh < c.kv_heads; ++kh) {
            for (size_t j = 0; ok && j < c.head_dim; ++j) {
                const size_t n = which >= DWQ ? f.m[which].cols : f.m[which].rows;
                for (size_t other = 0; ok && other < n; ++other) {
                    double sum = 0.0;
                    for (size_t gh = 0; gh < group; ++gh) {
                        const size_t col = (kh*group+gh)*c.head_dim+j;
                        sum += which >= DWQ ? (double)*cell(expanded.m[which], col, other)
                                            : (double)*cell(expanded.m[which], other, col);
                    }
                    const size_t col = kh*c.head_dim+j;
                    const double got = which >= DWQ ? (double)*cell(f.m[which], col, other)
                                                   : (double)*cell(f.m[which], other, col);
                    ok = near_value(got, sum, 3e-7, "shared K/V gradient is SUM of expanded heads");
                }
            }
        }
    }
    ok = ok && require(guards(&f) && guards(&expanded), "sharing guard changed");
    fixture_destroy(&f);
    fixture_destroy(&expanded);
    return ok;
}

static int dependency_edges(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2,3,4,2,2}, 3, 2)) return 0;
    fill(f.m[DY], 0.0f);
    /* Only batch one's first output participates in this objective. */
    *cell(f.m[DY], 3, 0) = 1.0f;
    int ok = status_is(forward(&f, backend), DL_OK, "dependency forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "dependency backward");
    const size_t score_paths[] = {DQ, DK, DQ0, DK0, DWQ, DWK, DS};
    for (size_t a = 0; ok && a < 7; ++a)
        ok = require(matrix_zero(f.m[score_paths[a]]), "one-key objective has a score path");
    for (size_t i = 0; ok && i < f.m[DX].rows; ++i) {
        if (i == 3) continue;
        for (size_t j = 0; ok && j < f.m[DX].cols; ++j)
            ok = near_value(*cell(f.m[DX], i, j), 0.0, 0.0, "batch/future input gradient leak");
    }
    ok = ok && require(!matrix_zero(f.m[DV]) && !matrix_zero(f.m[DWV]),
                        "one-key objective must retain value derivatives") && mask_invariants(&f);
    /* A zero query projection kills dWk, but dWq need not vanish. */
    fill(f.m[WQ], 0.0f);
    fill(f.m[DY], 0.0f);
    *cell(f.m[DY], 5, 0) = 1.0f;
    ok = ok && status_is(forward(&f, backend), DL_OK, "zero-query forward") &&
         status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "zero-query backward") &&
         require(matrix_zero(f.m[DWK]) && !matrix_zero(f.m[DWQ]), "zero-query path separation");
    fill(f.m[DY], 0.0f);
    ok = ok && status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "zero upstream");
    for (size_t a = DC; ok && a < NARRAY; ++a)
        ok = require(matrix_zero(f.m[a]), "zero upstream did not clear SET result");
    ok = ok && require(guards(&f), "dependency padding changed");
    fixture_destroy(&f);
    return ok;
}

static int reject_unchanged(Fixture *f, DlBackend backend, DlWriteMode xm,
                             DlWriteMode wm, DlStatus want, const char *where)
{
    float before[NARRAY][CAP];
    save_arrays(f, before);
    return status_is(backward(f, backend, xm, wm), want, where) &&
           require(same_arrays(f, before, 0, NARRAY), "preflight failure changed storage");
}

static int failures(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2,3,4,2,2}, 3, 2)) return 0;
    int ok = status_is(forward(&f, backend), DL_OK, "failure baseline");
    ok = ok && reject_unchanged(&f, (DlBackend)99, DL_SET, DL_SET, DL_ARGUMENT, "bad backend") &&
         reject_unchanged(&f, backend, (DlWriteMode)99, DL_SET, DL_ARGUMENT, "bad dx mode") &&
         reject_unchanged(&f, backend, DL_SET, (DlWriteMode)99, DL_ARGUMENT, "bad weight mode");
    const DlMatrix old_dx = f.m[DX], old_q = f.m[DQ], old_k0 = f.m[DK0];
    f.m[DX] = f.m[DY];
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ALIAS, "dx/dy alias");
    f.m[DX] = old_dx;
    f.m[DQ] = f.m[Q];
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ALIAS, "scratch/cache alias");
    f.m[DQ] = old_q;
    f.m[DK0] = f.m[DK];
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ALIAS, "scratch/scratch alias");
    f.m[DK0] = old_k0;
    --f.m[DX].rows;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_SHAPE, "dx shape");
    f.m[DX] = old_dx;
    const size_t old_pos = f.positions[0];
    f.positions[0] = SIZE_MAX;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_RANGE, "position out of range");
    f.positions[0] = old_pos;
    const double old_freq = f.freq[0];
    f.freq[0] = NAN;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ARGUMENT, "bad frequency");
    f.freq[0] = old_freq;
    const size_t poison[] = {X, WQ, K, DY};
    for (size_t a = 0; ok && a < 4; ++a) {
        float *p = cell(f.m[poison[a]], 0, 0);
        const float old = *p;
        *p = NAN;
        ok = reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "nonfinite read input");
        *p = old;
    }
    *cell(f.m[PROBS], 0, 1) = 0.25f;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "nonzero masked probability");
    *cell(f.m[PROBS], 0, 1) = 0.0f;
    *cell(f.m[SCORES], 0, 1) = 0.0f;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "masked score sentinel");
    *cell(f.m[SCORES], 0, 1) = -INFINITY;
    *cell(f.m[PROBS], 0, 0) = 0.5f;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "probability row sum");
    *cell(f.m[PROBS], 0, 0) = 1.0f;
    ok = ok && reject_unchanged(&f, backend, DL_ADD, DL_SET, DL_NUMERIC, "NaN old dx for ADD");
    for (size_t a = DX; a < NARRAY; ++a) fill(f.m[a], 0.0f);
    *cell(f.m[DWV], 0, 0) = NAN;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_ADD, DL_NUMERIC, "NaN old later weight for ADD");
    *cell(f.m[DWV], 0, 0) = 0.0f;
    ok = ok && status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "valid masked negative infinity");
    fixture_destroy(&f);
    if (!ok || !fixture_create(&f, (DlAttentionConfig){1,1,2,1,2}, 2, 1)) return 0;
    fill(f.m[WO], 0.0f);
    *cell(f.m[WO], 0, 0) = 1.0f; *cell(f.m[WO], 0, 1) = 2.0f;
    fill(f.m[DY], 0.0f); *cell(f.m[DY], 0, 0) = FLT_MAX;
    ok = status_is(forward(&f, backend), DL_OK, "finite forward before derivative overflow") &&
         status_is(backward(&f, backend, DL_SET, DL_SET), DL_NUMERIC, "late dC overflow") &&
         require(*cell(f.m[DC], 0, 0) == FLT_MAX && isnan(*cell(f.m[DC], 0, 1)),
                 "expected partial dC, not a transactional failure") && require(guards(&f), "failure padding changed");
    fixture_destroy(&f);
    return ok;
}

int main(void)
{
    int ok = 1;
    for (int b = 0; b < 2; ++b) {
        const DlBackend backend = (DlBackend)b;
        ok = hand_trace(backend) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){2,3,4,2,2}, 3, 2) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){1,3,2,1,4}, 3, 1) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){1,2,2,2,2}, 3, 0) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){1,1,2,1,2}, 2, 1) && ok;
        ok = accumulation(backend) && ok;
        for (size_t kv = 1; kv <= 4; kv *= 2) ok = matched_heads(backend, kv) && ok;
        ok = dependency_edges(backend) && ok;
        ok = failures(backend) && ok;
    }
    if (!ok) return EXIT_FAILURE;
    printf("full attention gradients: %zu comparisons, max abs error %.6g\n", full_checks, full_error);
    printf("isolated Q/K/V gradients: %zu comparisons, max abs error %.6g\n", core_checks, core_error);
    puts("attention backward: hand, modes, shared sums, dependencies, masks, failures passed");
    return EXIT_SUCCESS;
}
