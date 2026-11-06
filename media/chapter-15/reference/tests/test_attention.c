#include "dl/attention.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { X, WQ, WK, WV, WO, Q0, K0, V, Q, K, SCORES, PROBS, CONTEXT, Y, NARRAY };

typedef struct {
    DlAttentionConfig c;
    DlTensor owner[NARRAY];
    DlMatrix m[NARRAY];
    size_t *positions;
    double *freq;
} Fixture;

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
    fprintf(stderr, "%s: %.12g; expected %.12g, tolerance %.3g\n",
            where, got, want, tol);
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

/* Small TEST fixtures only; these caps are not attention's API limits.
 * f is initialized empty. Never copy this owning struct while live. */
static int fixture_create(Fixture *f, DlAttentionConfig c, size_t d, size_t pad)
{
    if (c.batch == 0 || c.batch > 2 || c.time == 0 || c.time > 8 ||
        c.query_heads == 0 || c.query_heads > 8 || c.kv_heads == 0 ||
        c.query_heads % c.kv_heads != 0 || c.head_dim == 0 ||
        c.head_dim > 8 || c.head_dim % 2 != 0 || d == 0 || d > 8 || pad > 2)
        return 0;
    f->c = c;
    const size_t m = c.batch * c.time, q = c.query_heads * c.head_dim;
    const size_t k = c.kv_heads * c.head_dim, sr = m * c.query_heads;
    const size_t rows[NARRAY] = {m, q, k, k, d, m, m, m, m, m, sr, sr, m, m};
    const size_t cols[NARRAY] = {d, d, d, d, q, q, k, k, q, k, c.time, c.time, q, d};
    for (size_t a = 0; a < NARRAY; ++a) {
        const size_t ld = cols[a] + pad, offset = pad != 0 ? 1 : 0;
        const size_t count = rows[a] * ld + (pad != 0 ? 2 : 0);
        DlStatus s = dl_tensor_create(&f->owner[a], 1, &count);
        if (s != DL_OK) goto fail;
        s = dl_matrix_view(&f->owner[a], offset, rows[a], cols[a], ld, &f->m[a]);
        if (s != DL_OK) goto fail;
        for (size_t j = 0; j < count; ++j) f->owner[a].data[j] = -777.0f;
        for (size_t i = 0; i < rows[a]; ++i) {
            for (size_t j = 0; j < cols[a]; ++j) {
                *cell(f->m[a], i, j) = a < Q0
                    ? (float)((int)((17 * i + 13 * j + 7 * a) % 23) - 11) / 32.0f
                    : NAN;
            }
        }
    }
    f->positions = malloc(m * sizeof(*f->positions));
    f->freq = malloc((c.head_dim / 2) * sizeof(*f->freq));
    if (f->positions == NULL || f->freq == NULL) goto fail;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t t = 0; t < c.time; ++t) f->positions[b * c.time + t] = 5 + 2 * b + t;
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

static DlStatus run(Fixture *f, DlBackend backend)
{
    return dl_attention_forward(backend, f->c, dl_matrix_read(f->m[X]), weights(f),
        f->positions, f->c.batch * f->c.time, f->freq, f->c.head_dim / 2,
        cache(f), f->m[Y]);
}

static int guards(const Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) {
        const DlMatrix m = f->m[a];
        for (size_t j = 0; j < m.count; ++j) {
            const int logical = j >= m.offset && (j - m.offset) / m.ld < m.rows
                             && (j - m.offset) % m.ld < m.cols;
            if (!logical && m.base[j] != -777.0f) return 0;
        }
    }
    return 1;
}

static void outputs_fill(Fixture *f, float value)
{
    for (size_t a = Q0; a < NARRAY; ++a) fill(f->m[a], value);
}

static int outputs_are(const Fixture *f, float value)
{
    for (size_t a = Q0; a < NARRAY; ++a) {
        const DlMatrix m = f->m[a];
        for (size_t i = 0; i < m.rows; ++i) {
            for (size_t j = 0; j < m.cols; ++j) {
                if (*cell(m, i, j) != value) return 0;
            }
        }
    }
    return guards(f);
}

static int row_invariants(const Fixture *f)
{
    const DlAttentionConfig c = f->c;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                double sum = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    const float score = *cell(f->m[SCORES], sr, s);
                    const float p = *cell(f->m[PROBS], sr, s);
                    if (s > t) {
                        if (!require(isinf(score) && score < 0.0f && p == 0.0f,
                                     "masked cell is not (-infinity, zero)")) return 0;
                    } else {
                        if (!require(isfinite(score) && isfinite(p) && p >= 0.0f && p <= 1.0f,
                                     "invalid allowed score/probability")) return 0;
                        sum += (double)p;
                    }
                }
                if (!near_value(sum, 1.0, 2e-7, "probability row sum")) return 0;
                if (t == 0) {
                    const size_t kh = h / (c.query_heads / c.kv_heads);
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        if (!near_value(*cell(f->m[CONTEXT], b * c.time, h * c.head_dim + j),
                                        *cell(f->m[V], b * c.time, kh * c.head_dim + j),
                                        0.0, "one-key row equals its value")) return 0;
                    }
                }
            }
        }
    }
    return 1;
}

static int hand_trace(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){1, 3, 2, 1, 2}, 3, 2)) return 0;
    const float r = (float)sqrt(2.0), a = (float)log(2.0);
    const float x[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float wq[12] = {1, 0, -r, 0, -r, 0, 0, r, 0, 1, 0, -r};
    const float wk[6] = {0, 0, -2 * a, a, -a, a};
    const float wv[6] = {1, 0, 3, 0, 2, 1};
    const float wo[12] = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2};
    const float *src[5] = {x, wq, wk, wv, wo};
    for (size_t v = 0; v < 5; ++v) {
        for (size_t i = 0; i < f.m[v].rows; ++i) {
            for (size_t j = 0; j < f.m[v].cols; ++j)
                *cell(f.m[v], i, j) = src[v][i * f.m[v].cols + j];
        }
    }
    f.positions[0] = 0; f.positions[1] = 2; f.positions[2] = 4;
    f.freq[0] = acos(-1.0) / 4.0; /* Synthetic quarter-turn fixture. */
    int ok = status_is(run(&f, backend), DL_OK, "hand forward");
    const double want_q[12] = {1, 0, 0, 1, r, 0, 0, r, r, 0, 0, r};
    const double want_k[6] = {0, a, a, 0, 2 * a, -a};
    for (size_t i = 0; ok && i < 3; ++i) {
        for (size_t j = 0; ok && j < 4; ++j)
            ok = near_value(*cell(f.m[Q], i, j), want_q[i * 4 + j], 3e-7, "hand rotated Q");
        for (size_t j = 0; ok && j < 2; ++j)
            ok = near_value(*cell(f.m[K], i, j), want_k[i * 2 + j], 3e-7, "hand rotated K");
    }
    const double want_y[9] = {1, 1, 0, 1.0 / 3.0, 2, 4.0 / 3.0,
                             13.0 / 7.0, 15.0 / 7.0, 10.0 / 7.0};
    const double want_p[6] = {1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0,
                             4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0};
    for (size_t i = 0; ok && i < 3; ++i) {
        for (size_t j = 0; ok && j < 3; ++j)
            ok = near_value(*cell(f.m[Y], i, j), want_y[i * 3 + j], 1e-6, "hand Y");
    }
    for (size_t h = 0; ok && h < 2; ++h) {
        for (size_t s = 0; ok && s < 3; ++s)
            ok = near_value(*cell(f.m[PROBS], h * 3 + 2, s), want_p[h * 3 + s],
                            2e-7, "hand head probability");
    }
    ok = ok && row_invariants(&f) && require(guards(&f), "hand padding changed");
    if (ok) printf("hand Y[2] (%s): [%.6f, %.6f, %.6f]\n",
        backend == DL_BACKEND_REF ? "reference" : "blocked",
        (double)*cell(f.m[Y], 2, 0), (double)*cell(f.m[Y], 2, 1),
        (double)*cell(f.m[Y], 2, 2));
    fixture_destroy(&f);
    return ok;
}

/* Independent, slow double arithmetic from INPUTS, not the saved cache.
 * This checker deliberately does not call production matmul/RoPE/attention.
 * It omits intermediate float stores; compare with a tolerance, not bitwise. */
static double projection(const Fixture *f, size_t row, size_t weight, size_t col)
{
    double sum = 0.0;
    for (size_t j = 0; j < f->m[X].cols; ++j)
        sum += (double)*cell(f->m[X], row, j) * (double)*cell(f->m[weight], col, j);
    return sum;
}

static double rotated(const Fixture *f, size_t row, size_t weight,
                       size_t head, size_t j)
{
    const size_t half = f->c.head_dim / 2;
    const size_t plane = j % half, base = head * f->c.head_dim;
    const double angle = (double)f->positions[row] * f->freq[plane];
    const double u = projection(f, row, weight, base + plane);
    const double v = projection(f, row, weight, base + plane + half);
    return j < half ? cos(angle) * u - sin(angle) * v
                    : sin(angle) * u + cos(angle) * v;
}

static int oracle(const Fixture *f, double *expected)
{
    const DlAttentionConfig c = f->c;
    if (c.time > 8 || c.query_heads * c.head_dim > 32) return 0;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t t = 0; t < c.time; ++t) {
            double context[32] = {0};
            const size_t row = b * c.time + t;
            for (size_t h = 0; h < c.query_heads; ++h) {
                double score[8] = {0}, maximum = -INFINITY, den = 0.0;
                const size_t kh = h / (c.query_heads / c.kv_heads);
                for (size_t s = 0; s <= t; ++s) {
                    for (size_t j = 0; j < c.head_dim; ++j)
                        score[s] += rotated(f, row, WQ, h, j)
                                  * rotated(f, b * c.time + s, WK, kh, j);
                    score[s] /= sqrt((double)c.head_dim);
                    maximum = fmax(maximum, score[s]);
                }
                for (size_t s = 0; s <= t; ++s) den += exp(score[s] - maximum);
                for (size_t j = 0; j < c.head_dim; ++j) {
                    for (size_t s = 0; s <= t; ++s)
                        context[h * c.head_dim + j] += exp(score[s] - maximum) / den
                            * projection(f, b * c.time + s, WV, kh * c.head_dim + j);
                }
            }
            for (size_t j = 0; j < f->m[Y].cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < c.query_heads * c.head_dim; ++k)
                    sum += context[k] * (double)*cell(f->m[WO], j, k);
                expected[row * f->m[Y].cols + j] = sum;
            }
        }
    }
    return 1;
}

static int oracle_cases(void)
{
    const DlAttentionConfig cases[5] = {
        {2, 3, 4, 1, 2}, {2, 3, 4, 2, 2}, {2, 3, 4, 4, 2},
        {2, 3, 6, 2, 4}, {2, 1, 4, 2, 2}
    };
    size_t count = 0;
    double worst = 0.0;
    for (size_t ci = 0; ci < 5; ++ci) {
        for (int be = 0; be < 2; ++be) {
            Fixture f = {0};
            if (!fixture_create(&f, cases[ci], 5, ci == 2 ? 0 : 2)) return 0;
            double expected[30] = {0};
            float saved[5][512];
            int ok = 1;
            for (size_t a = 0; a < 5; ++a) {
                if (f.owner[a].count > 512) { ok = 0; break; }
                memcpy(saved[a], f.owner[a].data, f.owner[a].count * sizeof(float));
            }
            ok = ok && oracle(&f, expected) &&
                status_is(run(&f, (DlBackend)be), DL_OK, "oracle forward");
            for (size_t i = 0; ok && i < f.m[Y].rows; ++i) {
                for (size_t j = 0; ok && j < f.m[Y].cols; ++j) {
                    const double want = expected[i * f.m[Y].cols + j];
                    const double got = (double)*cell(f.m[Y], i, j);
                    worst = fmax(worst, fabs(got - want));
                    ok = near_value(got, want, 2e-6 + 1e-5 * fabs(want), "double oracle Y");
                }
            }
            for (size_t a = 0; ok && a < 5; ++a)
                ok = require(memcmp(saved[a], f.owner[a].data,
                             f.owner[a].count * sizeof(float)) == 0, "input was changed");
            ok = ok && row_invariants(&f) && require(guards(&f), "oracle guards changed");
            fixture_destroy(&f);
            if (!ok) return 0;
            ++count;
        }
    }
    printf("independent oracle: %zu cases, max absolute error %.6g\n", count, worst);
    return 1;
}

static int isolation(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2, 3, 4, 2, 2}, 5, 2)) return 0;
    /* Equal positions are legal and do NOT make future tokens visible. */
    for (size_t i = 0; i < 6; ++i) f.positions[i] = 7;
    float old_y[30] = {0};
    int ok = status_is(run(&f, backend), DL_OK, "isolation baseline");
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 5; ++j) old_y[i * 5 + j] = *cell(f.m[Y], i, j);
    }
    for (size_t j = 0; j < 5; ++j) *cell(f.m[X], 2, j) += (float)(j + 1);
    ok = ok && status_is(run(&f, backend), DL_OK, "future perturbation");
    double changed = 0.0;
    for (size_t i = 0; ok && i < 6; ++i) {
        for (size_t j = 0; ok && j < 5; ++j) {
            if (i == 2) changed += fabs((double)*cell(f.m[Y], i, j) - old_y[i * 5 + j]);
            else ok = near_value(*cell(f.m[Y], i, j), old_y[i * 5 + j], 0.0,
                                 "future/batch isolation");
        }
    }
    ok = ok && require(changed > 1e-5, "future perturbation had no measurable effect");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) old_y[i * 5 + j] = *cell(f.m[Y], i, j);
    }
    for (size_t i = 3; i < 6; ++i) {
        for (size_t j = 0; j < 5; ++j) *cell(f.m[X], i, j) -= (float)(i + j + 1);
    }
    ok = ok && status_is(run(&f, backend), DL_OK, "other batch perturbation");
    for (size_t i = 0; ok && i < 3; ++i) {
        for (size_t j = 0; ok && j < 5; ++j)
            ok = near_value(*cell(f.m[Y], i, j), old_y[i * 5 + j], 0.0, "batch isolation");
    }
    ok = ok && row_invariants(&f) && require(guards(&f), "isolation padding");
    fixture_destroy(&f);
    return ok;
}

/* Expand weights ONLY in a separate MHA fixture to prove equal effective
 * tensors, not in the production GQA cache or projection. */
static int matched_heads(DlBackend backend, size_t kv_heads)
{
    Fixture a = {0}, b = {0};
    if (!fixture_create(&a, (DlAttentionConfig){1, 3, 4, kv_heads, 2}, 5, 2) ||
        !fixture_create(&b, (DlAttentionConfig){1, 3, 4, 4, 2}, 5, 2)) {
        fixture_destroy(&a); fixture_destroy(&b); return 0;
    }
    for (size_t w = WK; w <= WV; ++w) {
        for (size_t h = 0; h < 4; ++h) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t d = 0; d < 5; ++d)
                    *cell(b.m[w], h * 2 + j, d) =
                        *cell(a.m[w], (h / (4 / kv_heads)) * 2 + j, d);
            }
        }
    }
    int ok = status_is(run(&a, backend), DL_OK, "grouped matched forward") &&
             status_is(run(&b, backend), DL_OK, "expanded matched forward");
    const size_t compare[3] = {PROBS, CONTEXT, Y};
    for (size_t n = 0; ok && n < 3; ++n) {
        const size_t v = compare[n];
        for (size_t i = 0; ok && i < a.m[v].rows; ++i) {
            for (size_t j = 0; ok && j < a.m[v].cols; ++j)
                ok = near_value(*cell(a.m[v], i, j), *cell(b.m[v], i, j),
                                1e-7, "matched MHA/GQA result");
        }
    }
    fixture_destroy(&a); fixture_destroy(&b);
    return ok;
}

static int rejected_calls(void)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2, 3, 4, 2, 2}, 5, 2)) return 0;
    outputs_fill(&f, -31.0f);
    int ok = 1;
#define REJECT(expr, code, label) do { \
    if (!status_is((expr), (code), (label)) || \
        !require(outputs_are(&f, -31.0f), "preflight changed destinations")) { \
        ok = 0; goto done; \
    } \
} while (0)
#define CALL(cfg, wt, pos, np, fr, nf, ca, out) \
    dl_attention_forward(DL_BACKEND_REF, (cfg), dl_matrix_read(f.m[X]), (wt), \
        (pos), (np), (fr), (nf), (ca), (out))
    REJECT(run(&f, (DlBackend)77), DL_ARGUMENT, "unknown backend");
    DlAttentionConfig bad = f.c;
    bad.time = 0;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "empty sequence");
    bad = f.c; bad.kv_heads = 0;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "zero KV heads");
    bad = f.c; bad.query_heads = 3;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "nondivisible heads");
    bad = f.c; bad.head_dim = 3;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "odd rotary width");
    bad = f.c; bad.batch = SIZE_MAX;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_OVERFLOW, "row product overflow");
    bad = (DlAttentionConfig){1, SIZE_MAX / 16, 4, 2, 2};
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_OVERFLOW, "storage product overflow");
    DlAttentionWeights w = weights(&f);
    --w.key.cols;
    REJECT(CALL(f.c, w, f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "wrong key projection width");
    DlMatrix y = f.m[Y]; y.ld = y.cols - 1;
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, cache(&f), y),
           DL_ARGUMENT, "invalid leading dimension");
    DlAttentionCache a = cache(&f); a.probs = a.scores;
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, a, f.m[Y]),
           DL_ALIAS, "probabilities overlap scores");
    a = cache(&f); a.q = a.q0;
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, a, f.m[Y]),
           DL_ALIAS, "in-place rotation");
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[X]),
           DL_ALIAS, "output overlaps input");
    REJECT(CALL(f.c, weights(&f), NULL, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_ARGUMENT, "null positions");
    REJECT(CALL(f.c, weights(&f), f.positions, 5, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "position count");
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 0, cache(&f), f.m[Y]),
           DL_SHAPE, "frequency count");
    f.freq[0] = NAN;
    REJECT(run(&f, DL_BACKEND_REF), DL_ARGUMENT, "nonfinite frequency");
    f.freq[0] = 1.0;
    if (SIZE_MAX > UINT32_MAX) {
        f.positions[0] = DL_ROPE_MAX_POSITION + (size_t)1;
        REJECT(run(&f, DL_BACKEND_REF), DL_RANGE, "position cap");
        f.positions[0] = 5;
    }
    const float saved = *cell(f.m[WO], 4, 7);
    *cell(f.m[WO], 4, 7) = NAN;
    REJECT(run(&f, DL_BACKEND_REF), DL_NUMERIC, "late weight preflight");
    *cell(f.m[WO], 4, 7) = saved;
    *cell(f.m[X], 5, 4) = INFINITY;
    REJECT(run(&f, DL_BACKEND_REF), DL_NUMERIC, "future input still must be finite");
done:
#undef CALL
#undef REJECT
    fixture_destroy(&f);
    return ok;
}

static int extremes_and_partial_failure(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){1, 2, 1, 1, 2}, 2, 2)) return 0;
    for (size_t a = 0; a < 5; ++a) fill(f.m[a], 0.0f);
    *cell(f.m[X], 0, 0) = *cell(f.m[X], 1, 1) = 1.0f;
    *cell(f.m[WO], 0, 0) = *cell(f.m[WO], 1, 1) = 1.0f;
    *cell(f.m[WQ], 0, 0) = *cell(f.m[WQ], 0, 1) = 1.0f;
    *cell(f.m[WK], 0, 0) = FLT_MAX; *cell(f.m[WK], 0, 1) = -FLT_MAX;
    *cell(f.m[WV], 0, 0) = 1.0f; *cell(f.m[WV], 0, 1) = 3.0f;
    *cell(f.m[WV], 1, 0) = 2.0f; *cell(f.m[WV], 1, 1) = 4.0f;
    f.positions[0] = f.positions[1] = 0;
    int ok = status_is(run(&f, backend), DL_OK, "extreme finite scores") &&
        near_value(*cell(f.m[PROBS], 1, 0), 1.0, 0.0, "extreme winner") &&
        near_value(*cell(f.m[PROBS], 1, 1), 0.0, 0.0, "extreme underflow") &&
        near_value(*cell(f.m[Y], 1, 0), 1.0, 0.0, "extreme output") &&
        row_invariants(&f);
    *cell(f.m[WQ], 0, 0) = *cell(f.m[WQ], 0, 1) = (float)sqrt(2.0);
    *cell(f.m[WK], 0, 0) = 1000.0f; *cell(f.m[WK], 0, 1) = 1001.0f;
    const double p = 1.0 / (1.0 + exp(-1.0));
    ok = ok && status_is(run(&f, backend), DL_OK, "large common score") &&
        near_value(*cell(f.m[PROBS], 1, 1), p, 1e-7, "max-shifted probability");

    outputs_fill(&f, NAN);
    *cell(f.m[WQ], 0, 0) = 1.0f; *cell(f.m[WQ], 0, 1) = FLT_MAX;
    fill(f.m[WK], 0.0f);
    *cell(f.m[WK], 1, 0) = 1.0f; *cell(f.m[WK], 0, 1) = 2.0f;
    ok = ok && status_is(run(&f, backend), DL_NUMERIC, "later score overflow") &&
        near_value(*cell(f.m[CONTEXT], 0, 0), 1.0, 0.0, "earlier row was written") &&
        require(isnan(*cell(f.m[CONTEXT], 1, 0)) && isnan(*cell(f.m[Y], 0, 0)),
                "failed score unexpectedly completed output");

    outputs_fill(&f, NAN);
    *cell(f.m[WQ], 0, 1) = 1.0f;
    *cell(f.m[WO], 1, 0) = *cell(f.m[WO], 1, 1) = FLT_MAX;
    ok = ok && status_is(run(&f, backend), DL_NUMERIC, "output projection overflow") &&
        require(isfinite(*cell(f.m[CONTEXT], 1, 0)), "cache did not finish before output failure") &&
        require(isnan(*cell(f.m[Y], 0, 1)), "overflowed Y cell was stored") &&
        require(guards(&f), "numeric failure changed padding");
    fixture_destroy(&f);
    return ok;
}

int main(void)
{
    if (!hand_trace(DL_BACKEND_REF) || !hand_trace(DL_BACKEND_BLOCKED) ||
        !oracle_cases() || !rejected_calls()) return EXIT_FAILURE;
    for (int be = 0; be < 2; ++be) {
        const DlBackend backend = (DlBackend)be;
        if (!isolation(backend) || !matched_heads(backend, 1) ||
            !matched_heads(backend, 2) || !extremes_and_partial_failure(backend))
            return EXIT_FAILURE;
    }
    puts("attention forward: hand, oracle, causality, sharing, boundaries, failures passed");
    return EXIT_SUCCESS;
}
