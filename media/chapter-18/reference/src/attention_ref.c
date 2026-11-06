#include "dl/attention.h"

#include <float.h>
#include <math.h>

/* All derived products are checked before an index uses them. */
typedef struct {
    size_t rows, query_width, kv_width, score_rows;
} Layout;

static DlStatus layout(DlAttentionConfig c, Layout *out)
{
    if (c.batch == 0 || c.time == 0 || c.query_heads == 0 || c.kv_heads == 0 ||
        c.head_dim == 0 || c.head_dim % 2 != 0 ||
        c.query_heads % c.kv_heads != 0) return DL_SHAPE;
    Layout z = {0};
    DlStatus s = dl_size_mul(c.batch, c.time, &z.rows);
    if (s != DL_OK) return s;
    s = dl_size_mul(c.query_heads, c.head_dim, &z.query_width);
    if (s != DL_OK) return s;
    s = dl_size_mul(c.kv_heads, c.head_dim, &z.kv_width);
    if (s != DL_OK) return s;
    s = dl_size_mul(z.rows, c.query_heads, &z.score_rows);
    if (s != DL_OK) return s;
    const size_t rows[3] = {z.rows, z.rows, z.score_rows};
    const size_t cols[3] = {z.query_width, z.kv_width, c.time};
    for (size_t i = 0; i < 3; ++i) {
        size_t count = 0, bytes = 0;
        s = dl_size_mul(rows[i], cols[i], &count);
        if (s != DL_OK) return s;
        s = dl_size_mul(count, sizeof(float), &bytes);
        if (s != DL_OK) return s;
    }
    *out = z;
    return DL_OK;
}

/* This forward call has exactly five inputs and nine destinations. */
static DlStatus check_views(const DlConstMatrix in[5], const DlMatrix out[9])
{
    size_t ie[5] = {0}, oe[9] = {0};
    for (size_t i = 0; i < 5; ++i) {
        const DlStatus s = dl_matrix_end(in[i], &ie[i]);
        if (s != DL_OK) return s;
    }
    for (size_t o = 0; o < 9; ++o) {
        const DlConstMatrix v = dl_matrix_read(out[o]);
        const DlStatus s = dl_matrix_end(v, &oe[o]);
        if (s != DL_OK) return s;
        for (size_t i = 0; i < 5; ++i) {
            if (v.base == in[i].base && v.offset < ie[i] &&
                in[i].offset < oe[o]) return DL_ALIAS;
        }
        for (size_t p = 0; p < o; ++p) {
            if (v.base == out[p].base && v.offset < oe[p] &&
                out[p].offset < oe[o]) return DL_ALIAS;
        }
    }
    return DL_OK;
}

static int shape(DlConstMatrix v, size_t rows, size_t cols)
{
    return v.rows == rows && v.cols == cols;
}

static int finite_cells(DlConstMatrix v)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (!isfinite(v.base[v.offset + i * v.ld + j])) return 0;
        }
    }
    return 1;
}

static DlStatus metadata(Layout z, DlAttentionConfig c,
                          const size_t *positions, size_t position_count,
                          const double *freq, size_t frequency_count)
{
    if (positions == NULL || freq == NULL) return DL_ARGUMENT;
    if (position_count != z.rows || frequency_count != c.head_dim / 2)
        return DL_SHAPE;
    size_t bytes = 0;
    DlStatus s = dl_size_mul(position_count, sizeof(*positions), &bytes);
    if (s != DL_OK) return s;
    s = dl_size_mul(frequency_count, sizeof(*freq), &bytes);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < position_count; ++i) {
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    }
    for (size_t j = 0; j < frequency_count; ++j) {
        if (!isfinite(freq[j]) || freq[j] <= 0.0 || freq[j] > 1.0)
            return DL_ARGUMENT;
    }
    return DL_OK;
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

/* Private: views, config, and all projected logical cells are already valid.
 * Each row stores only allowed finite scores; the rest are explicit sentinels. */
static DlStatus attend(DlAttentionConfig c, DlAttentionCache a)
{
    const size_t group = c.query_heads / c.kv_heads;
    const double scale = 1.0 / sqrt((double)c.head_dim);
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t t = 0; t < c.time; ++t) {
                const size_t row = b * c.time + t;
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                const size_t qa = a.q.offset + row * a.q.ld + h * c.head_dim;
                const size_t sa = a.scores.offset + sr * a.scores.ld;
                const size_t pa = a.probs.offset + sr * a.probs.ld;
                double maximum = -INFINITY;
                for (size_t s = 0; s < c.time; ++s) {
                    if (s > t) {
                        a.scores.base[sa + s] = -INFINITY;
                        a.probs.base[pa + s] = 0.0f;
                        continue;
                    }
                    const size_t kr = b * c.time + s;
                    const size_t ka = a.k.offset + kr * a.k.ld + kh * c.head_dim;
                    double dot = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        dot += (double)a.q.base[qa + j] * (double)a.k.base[ka + j];
                    }
                    const DlStatus status = store_float(&a.scores.base[sa + s],
                                                         dot * scale);
                    if (status != DL_OK) return status;
                    maximum = fmax(maximum, (double)a.scores.base[sa + s]);
                }
                double denominator = 0.0;
                for (size_t s = 0; s <= t; ++s) {
                    denominator += exp((double)a.scores.base[sa + s] - maximum);
                }
                /* Positive T and the diagonal forbid an empty allowed set. */
                if (!isfinite(denominator) || denominator <= 0.0) return DL_NUMERIC;
                for (size_t s = 0; s <= t; ++s) {
                    const double p = exp((double)a.scores.base[sa + s] - maximum)
                                   / denominator;
                    const DlStatus status = store_float(&a.probs.base[pa + s], p);
                    if (status != DL_OK) return status;
                }
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double sum = 0.0;
                    for (size_t s = 0; s <= t; ++s) {
                        const size_t vr = b * c.time + s;
                        const size_t va = a.v.offset + vr * a.v.ld + kh * c.head_dim + j;
                        sum += (double)a.probs.base[pa + s] * (double)a.v.base[va];
                    }
                    const size_t ca = a.context.offset + row * a.context.ld
                                    + h * c.head_dim + j;
                    const DlStatus status = store_float(&a.context.base[ca], sum);
                    if (status != DL_OK) return status;
                }
            }
        }
    }
    return DL_OK;
}

DlStatus dl_attention_forward(DlBackend backend, DlAttentionConfig c,
                              DlConstMatrix x, DlAttentionWeights w,
                              const size_t *positions, size_t position_count,
                              const double *inv_freq, size_t frequency_count,
                              DlAttentionCache a, DlMatrix y)
{
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED)
        return DL_ARGUMENT;
    Layout z = {0};
    DlStatus s = layout(c, &z);
    if (s != DL_OK) return s;
    const DlConstMatrix in[5] = {x, w.query, w.key, w.value, w.output};
    const DlMatrix out[9] = {a.q0, a.k0, a.v, a.q, a.k,
                             a.scores, a.probs, a.context, y};
    s = check_views(in, out);
    if (s != DL_OK) return s;
    const size_t width = x.cols;
    if (x.rows != z.rows || !shape(w.query, z.query_width, width) ||
        !shape(w.key, z.kv_width, width) || !shape(w.value, z.kv_width, width) ||
        !shape(w.output, width, z.query_width) ||
        !shape(dl_matrix_read(a.q0), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k0), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.v), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.q), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.scores), z.score_rows, c.time) ||
        !shape(dl_matrix_read(a.probs), z.score_rows, c.time) ||
        !shape(dl_matrix_read(a.context), z.rows, z.query_width) ||
        !shape(dl_matrix_read(y), z.rows, width)) return DL_SHAPE;
    s = metadata(z, c, positions, position_count, inv_freq, frequency_count);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < 5; ++i) {
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    s = dl_matmul(backend, x, DL_NOTRANS, w.query, DL_TRANS, a.q0, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, x, DL_NOTRANS, w.key, DL_TRANS, a.k0, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, x, DL_NOTRANS, w.value, DL_TRANS, a.v, DL_SET);
    if (s != DL_OK) return s;
    s = dl_rope_forward(dl_matrix_read(a.q0), c.head_dim, positions,
                        position_count, inv_freq, frequency_count, a.q);
    if (s != DL_OK) return s;
    s = dl_rope_forward(dl_matrix_read(a.k0), c.head_dim, positions,
                        position_count, inv_freq, frequency_count, a.k);
    if (s != DL_OK) return s;
    s = attend(c, a);
    if (s != DL_OK) return s;
    return dl_matmul(backend, dl_matrix_read(a.context), DL_NOTRANS,
                      w.output, DL_TRANS, y, DL_SET);
}

/* Backward borrows fourteen read views and thirteen write views. */
static DlStatus backward_views(const DlConstMatrix in[14], const DlMatrix out[13])
{
    size_t ie[14] = {0}, oe[13] = {0};
    for (size_t i = 0; i < 14; ++i) {
        const DlStatus s = dl_matrix_end(in[i], &ie[i]);
        if (s != DL_OK) return s;
    }
    for (size_t o = 0; o < 13; ++o) {
        const DlConstMatrix v = dl_matrix_read(out[o]);
        const DlStatus s = dl_matrix_end(v, &oe[o]);
        if (s != DL_OK) return s;
        for (size_t i = 0; i < 14; ++i) {
            if (v.base == in[i].base && v.offset < ie[i] &&
                in[i].offset < oe[o]) return DL_ALIAS;
        }
        for (size_t p = 0; p < o; ++p) {
            if (v.base == out[p].base && v.offset < oe[p] &&
                out[p].offset < oe[o]) return DL_ALIAS;
        }
    }
    return DL_OK;
}

/* Ordinary finite-cell validation would wrongly reject masked score sentinels. */
static DlStatus saved_rows(DlAttentionConfig c, DlAttentionCache a)
{
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                double total = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    const float score = a.scores.base[a.scores.offset + sr * a.scores.ld + s];
                    const float p = a.probs.base[a.probs.offset + sr * a.probs.ld + s];
                    if (s > t) {
                        if (score != -INFINITY || p != 0.0f) return DL_NUMERIC;
                    } else {
                        if (!isfinite(score) || !isfinite(p) || p < 0.0f || p > 1.0f)
                            return DL_NUMERIC;
                        total += (double)p;
                    }
                }
                if (fabs(total - 1.0) > 8.0 * (double)FLT_EPSILON) return DL_NUMERIC;
            }
        }
    }
    return DL_OK;
}

/* Validated views only. Materialize dP first, then differentiate softmax.
 * dP and dS above the diagonal are defined as zero: those are not variables. */
static DlStatus score_backward(DlAttentionConfig c, DlAttentionCache a,
                                DlAttentionScratch g)
{
    const size_t group = c.query_heads / c.kv_heads;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t t = 0; t < c.time; ++t) {
                const size_t row = b * c.time + t;
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                const size_t ga = g.context.offset + row * g.context.ld + h * c.head_dim;
                const size_t pa = a.probs.offset + sr * a.probs.ld;
                const size_t dpa = g.probs.offset + sr * g.probs.ld;
                const size_t dsa = g.scores.offset + sr * g.scores.ld;
                double center = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    if (s > t) {
                        g.probs.base[dpa + s] = 0.0f;
                        g.scores.base[dsa + s] = 0.0f;
                        continue;
                    }
                    const size_t va = a.v.offset + (b * c.time + s) * a.v.ld
                                    + kh * c.head_dim;
                    double dot = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        dot += (double)g.context.base[ga + j] * (double)a.v.base[va + j];
                    }
                    const DlStatus status = store_float(&g.probs.base[dpa + s], dot);
                    if (status != DL_OK) return status;
                    center += (double)a.probs.base[pa + s] * (double)g.probs.base[dpa + s];
                }
                for (size_t s = 0; s <= t; ++s) {
                    const double ds = (double)a.probs.base[pa + s]
                                    * ((double)g.probs.base[dpa + s] - center);
                    const DlStatus status = store_float(&g.scores.base[dsa + s], ds);
                    if (status != DL_OK) return status;
                }
            }
        }
    }
    return DL_OK;
}

/* Gather each derivative into one double accumulator, then store it once.
 * In particular, K/V accumulators are OUTSIDE the sharing-head loop. */
static DlStatus qkv_backward(DlAttentionConfig c, DlAttentionCache a,
                              DlAttentionScratch g)
{
    const size_t group = c.query_heads / c.kv_heads;
    const double scale = 1.0 / sqrt((double)c.head_dim);
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double sum = 0.0;
                    for (size_t s = 0; s <= t; ++s) {
                        const size_t sa = g.scores.offset + sr * g.scores.ld + s;
                        const size_t ka = a.k.offset + (b * c.time + s) * a.k.ld
                                        + kh * c.head_dim + j;
                        sum += (double)g.scores.base[sa] * (double)a.k.base[ka];
                    }
                    const size_t qa = g.q.offset + (b * c.time + t) * g.q.ld
                                    + h * c.head_dim + j;
                    const DlStatus status = store_float(&g.q.base[qa], scale * sum);
                    if (status != DL_OK) return status;
                }
            }
        }
        for (size_t kh = 0; kh < c.kv_heads; ++kh) {
            for (size_t s = 0; s < c.time; ++s) {
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double key_sum = 0.0, value_sum = 0.0;
                    for (size_t gh = 0; gh < group; ++gh) {
                        const size_t h = kh * group + gh;
                        for (size_t t = s; t < c.time; ++t) {
                            const size_t sr = (b * c.query_heads + h) * c.time + t;
                            const size_t sa = g.scores.offset + sr * g.scores.ld + s;
                            const size_t pa = a.probs.offset + sr * a.probs.ld + s;
                            const size_t qa = a.q.offset + (b * c.time + t) * a.q.ld
                                            + h * c.head_dim + j;
                            const size_t ca = g.context.offset + (b * c.time + t) * g.context.ld
                                            + h * c.head_dim + j;
                            key_sum += (double)g.scores.base[sa] * (double)a.q.base[qa];
                            value_sum += (double)a.probs.base[pa] * (double)g.context.base[ca];
                        }
                    }
                    const size_t ka = g.k.offset + (b * c.time + s) * g.k.ld
                                    + kh * c.head_dim + j;
                    const size_t va = g.v.offset + (b * c.time + s) * g.v.ld
                                    + kh * c.head_dim + j;
                    DlStatus status = store_float(&g.k.base[ka], scale * key_sum);
                    if (status != DL_OK) return status;
                    status = store_float(&g.v.base[va], value_sum);
                    if (status != DL_OK) return status;
                }
            }
        }
    }
    return DL_OK;
}

DlStatus dl_attention_backward(DlBackend backend, DlAttentionConfig c,
                               DlConstMatrix x, DlAttentionWeights w,
                               const size_t *positions, size_t position_count,
                               const double *inv_freq, size_t frequency_count,
                               DlAttentionCache a, DlConstMatrix dy,
                               DlAttentionScratch g, DlMatrix dx,
                               DlWriteMode dx_mode, DlAttentionWeightGrads dw,
                               DlWriteMode weight_mode)
{
    if ((backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) ||
        (dx_mode != DL_SET && dx_mode != DL_ADD) ||
        (weight_mode != DL_SET && weight_mode != DL_ADD)) return DL_ARGUMENT;
    Layout z = {0};
    DlStatus s = layout(c, &z);
    if (s != DL_OK) return s;
    const DlConstMatrix in[14] = {x, w.query, w.key, w.value, w.output,
        dl_matrix_read(a.q0), dl_matrix_read(a.k0), dl_matrix_read(a.v),
        dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.scores),
        dl_matrix_read(a.probs), dl_matrix_read(a.context), dy};
    const DlMatrix out[13] = {g.context, g.probs, g.scores, g.q, g.k, g.v,
        g.q0, g.k0, dx, dw.query, dw.key, dw.value, dw.output};
    s = backward_views(in, out);
    if (s != DL_OK) return s;
    const size_t d = x.cols, m = z.rows, q = z.query_width, k = z.kv_width;
    const size_t ir[14] = {m, q, k, k, d, m, m, m, m, m, z.score_rows,
                           z.score_rows, m, m};
    const size_t ic[14] = {d, d, d, d, q, q, k, k, q, k, c.time, c.time, q, d};
    const size_t orows[13] = {m, z.score_rows, z.score_rows, m, m, m, m, m,
                              m, q, k, k, d};
    const size_t ocols[13] = {q, c.time, c.time, q, k, k, q, k, d, d, d, d, q};
    for (size_t i = 0; i < 14; ++i) {
        if (!shape(in[i], ir[i], ic[i])) return DL_SHAPE;
    }
    for (size_t o = 0; o < 13; ++o) {
        if (!shape(dl_matrix_read(out[o]), orows[o], ocols[o])) return DL_SHAPE;
    }
    s = metadata(z, c, positions, position_count, inv_freq, frequency_count);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < 14; ++i) {
        if (i != 10 && i != 11 && !finite_cells(in[i])) return DL_NUMERIC;
    }
    s = saved_rows(c, a);
    if (s != DL_OK) return s;
    if (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) return DL_NUMERIC;
    if (weight_mode == DL_ADD) {
        for (size_t o = 9; o < 13; ++o) {
            if (!finite_cells(dl_matrix_read(out[o]))) return DL_NUMERIC;
        }
    }
    /* The first write occurs only after ALL structural/numeric preflight. */
    s = dl_matmul(backend, dy, DL_NOTRANS, w.output, DL_NOTRANS, g.context, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dy, DL_TRANS, dl_matrix_read(a.context), DL_NOTRANS,
                   dw.output, weight_mode);
    if (s != DL_OK) return s;
    s = score_backward(c, a, g);
    if (s != DL_OK) return s;
    s = qkv_backward(c, a, g);
    if (s != DL_OK) return s;
    s = dl_rope_backward(dl_matrix_read(g.q), c.head_dim, positions, position_count,
                         inv_freq, frequency_count, g.q0, DL_SET);
    if (s != DL_OK) return s;
    s = dl_rope_backward(dl_matrix_read(g.k), c.head_dim, positions, position_count,
                         inv_freq, frequency_count, g.k0, DL_SET);
    if (s != DL_OK) return s;
    const DlConstMatrix upstream[3] = {dl_matrix_read(g.q0), dl_matrix_read(g.k0),
                                       dl_matrix_read(g.v)};
    const DlConstMatrix weight[3] = {w.query, w.key, w.value};
    const DlMatrix gradient[3] = {dw.query, dw.key, dw.value};
    for (size_t p = 0; p < 3; ++p) {
        s = dl_matmul(backend, upstream[p], DL_TRANS, x, DL_NOTRANS,
                       gradient[p], weight_mode);
        if (s != DL_OK) return s;
        s = dl_matmul(backend, upstream[p], DL_NOTRANS, weight[p], DL_NOTRANS,
                       dx, p == 0 ? dx_mode : DL_ADD);
        if (s != DL_OK) return s;
    }
    return DL_OK;
}
