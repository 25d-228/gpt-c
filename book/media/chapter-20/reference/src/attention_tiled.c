#include "dl/attention_tiled.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define TRY(call) do { DlStatus s_ = (call); if (s_ != DL_OK) return s_; } while (0)

static DlStatus workspace_count(size_t rows, size_t d, size_t qr, size_t kc,
                                size_t *count)
{
    if (rows == 0 || d == 0 || d % 2 != 0 || qr == 0 || kc == 0) return DL_SHAPE;
    size_t total = 0, term = 0, bytes = 0;
    TRY(dl_size_mul(3, rows, &total));
    TRY(dl_size_mul(qr, kc, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(qr, d, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(2, qr, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(total, sizeof(double), &bytes));
    *count = total;
    return DL_OK;
}

void dl_tiled_workspace_destroy(DlTiledWorkspace *s)
{
    if (s == NULL) return;
    free(s->data);
    *s = (DlTiledWorkspace){0};
}

DlStatus dl_tiled_workspace_create(DlTiledWorkspace *out, size_t rows, size_t d,
                                   size_t qr, size_t kc, size_t max_bytes)
{
    if (out == NULL || max_bytes == 0) return DL_ARGUMENT;
    if (out->data != NULL || out->count != 0 || out->row_capacity != 0 ||
        out->head_dim != 0 || out->query_tile != 0 || out->key_tile != 0) return DL_STATE;
    DlTiledWorkspace next = {.row_capacity = rows, .head_dim = d,
                             .query_tile = qr, .key_tile = kc};
    TRY(workspace_count(rows, d, qr, kc, &next.count));
    size_t bytes = next.count * sizeof(*next.data); /* Checked in workspace_count. */
    if (bytes > max_bytes) return DL_RANGE;
    next.data = malloc(bytes);
    if (next.data == NULL) return DL_ALLOC;
    *out = next;
    return DL_OK;
}

typedef struct {
    size_t rows, query_width, kv_width, stat_rows;
} Layout;

static DlStatus layout(DlAttentionConfig c, Layout *z)
{
    if (c.batch == 0 || c.time == 0 || c.query_heads == 0 || c.kv_heads == 0 ||
        c.head_dim == 0 || c.head_dim % 2 != 0 || c.query_heads % c.kv_heads != 0)
        return DL_SHAPE;
    TRY(dl_size_mul(c.batch, c.time, &z->rows));
    TRY(dl_size_mul(c.query_heads, c.head_dim, &z->query_width));
    TRY(dl_size_mul(c.kv_heads, c.head_dim, &z->kv_width));
    TRY(dl_size_mul(z->rows, c.query_heads, &z->stat_rows));
    /* Deliberately no B*Hq*T*T product: that storage does not exist here. */
    return DL_OK;
}

static DlStatus workspace_check(const DlTiledWorkspace *s, Layout z, size_t d)
{
    if (s == NULL || s->data == NULL) return DL_ARGUMENT;
    size_t count = 0;
    TRY(workspace_count(s->row_capacity, s->head_dim, s->query_tile, s->key_tile, &count));
    if (s->count != count) return DL_STATE;
    if (s->head_dim != d || s->row_capacity < z.stat_rows) return DL_SHAPE;
    return DL_OK;
}

static int shape(DlConstMatrix v, size_t rows, size_t cols)
{
    return v.rows == rows && v.cols == cols;
}

static int finite_cells(DlConstMatrix v)
{
    for (size_t i = 0; i < v.rows; ++i)
        for (size_t j = 0; j < v.cols; ++j)
            if (!isfinite(v.base[v.offset + i * v.ld + j])) return 0;
    return 1;
}

/* Forward and backward share structural checks, not numeric storage. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                            const DlMatrix *out, size_t no)
{
    size_t ie[12] = {0}, oe[11] = {0};
    if (ni > 12 || no > 11) return DL_ARGUMENT;
    for (size_t i = 0; i < ni; ++i) TRY(dl_matrix_end(in[i], &ie[i]));
    for (size_t o = 0; o < no; ++o) {
        DlConstMatrix v = dl_matrix_read(out[o]);
        TRY(dl_matrix_end(v, &oe[o]));
        for (size_t i = 0; i < ni; ++i)
            if (v.base == in[i].base && v.offset < ie[i] && in[i].offset < oe[o])
                return DL_ALIAS;
        for (size_t p = 0; p < o; ++p)
            if (v.base == out[p].base && v.offset < oe[p] && out[p].offset < oe[o])
                return DL_ALIAS;
    }
    return DL_OK;
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

static size_t smaller(size_t a, size_t b) { return a < b ? a : b; }

/* All geometry and input finiteness were checked by the public entry point. */
static DlStatus attend(DlAttentionConfig c, DlConstMatrix q, DlConstMatrix k,
                       DlConstMatrix v, DlTiledWorkspace *s, DlMatrix out)
{
    const size_t qr = s->query_tile, kc = s->key_tile, d = c.head_dim;
    double *tile = s->data + 3 * s->row_capacity;
    double *u = tile + qr * kc;
    double *maximum = u + qr * d;
    double *denominator = maximum + qr;
    const double scale = 1.0 / sqrt((double)d);
    const size_t group = c.query_heads / c.kv_heads;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t qb = 0; qb < c.time; ) {
                const size_t nq = smaller(qr, c.time - qb);
                for (size_t r = 0; r < nq; ++r) {
                    maximum[r] = -INFINITY;
                    denominator[r] = 0.0;
                    for (size_t j = 0; j < d; ++j) u[r * d + j] = 0.0;
                }
                /* All key blocks beginning at qb+nq or later are wholly masked. */
                for (size_t kb = 0; kb < qb + nq; ) {
                    const size_t nk = smaller(kc, c.time - kb);
                    /* Form a Q/K score tile, never a pre-existing square matrix. */
                    for (size_t r = 0; r < nq; ++r) {
                        const size_t t = qb + r;
                        const size_t qa = q.offset + (b * c.time + t) * q.ld + h * d;
                        for (size_t col = 0; col < nk; ++col) {
                            const size_t key_time = kb + col;
                            if (key_time > t) { tile[r * kc + col] = -INFINITY; continue; }
                            const size_t ka = k.offset + (b * c.time + key_time) * k.ld + kh * d;
                            double dot = 0.0;
                            for (size_t j = 0; j < d; ++j)
                                dot += (double)q.base[qa + j] * (double)k.base[ka + j];
                            float score = 0.0f;
                            TRY(store_float(&score, dot * scale));
                            tile[r * kc + col] = (double)score;
                        }
                    }
                    for (size_t r = 0; r < nq; ++r) {
                        const size_t t = qb + r;
                        /* This row may be fully masked within a diagonal tile. */
                        if (kb > t) continue;
                        const size_t allowed = smaller(nk, t - kb + 1);
                        double tile_max = -INFINITY;
                        for (size_t col = 0; col < allowed; ++col)
                            tile_max = fmax(tile_max, tile[r * kc + col]);
                        const double next_max = fmax(maximum[r], tile_max);
                        const double alpha = denominator[r] == 0.0 ? 0.0
                                           : exp(maximum[r] - next_max);
                        denominator[r] *= alpha;
                        for (size_t j = 0; j < d; ++j) u[r * d + j] *= alpha;
                        for (size_t col = 0; col < allowed; ++col) {
                            const double beta = exp(tile[r * kc + col] - next_max);
                            denominator[r] += beta;
                            const size_t va = v.offset + (b * c.time + kb + col) * v.ld + kh * d;
                            for (size_t j = 0; j < d; ++j)
                                u[r * d + j] += beta * (double)v.base[va + j];
                        }
                        maximum[r] = next_max;
                    }
                    kb += nk; /* Actual tail length; no overflowing kb+kc. */
                }
                for (size_t r = 0; r < nq; ++r) {
                    if (!isfinite(denominator[r]) || denominator[r] <= 0.0 ||
                        !isfinite(maximum[r])) return DL_NUMERIC;
                    const size_t t = qb + r;
                    const size_t sr = (b * c.query_heads + h) * c.time + t;
                    const double lse = maximum[r] + log(denominator[r]);
                    if (!isfinite(lse)) return DL_NUMERIC;
                    s->data[sr] = maximum[r];
                    s->data[s->row_capacity + sr] = denominator[r];
                    s->data[2 * s->row_capacity + sr] = lse;
                    const size_t oa = out.offset + (b * c.time + t) * out.ld + h * d;
                    for (size_t j = 0; j < d; ++j)
                        TRY(store_float(&out.base[oa + j], u[r * d + j] / denominator[r]));
                }
                qb += nq;
            }
        }
    }
    return DL_OK;
}

DlStatus dl_attention_tiled_core(DlAttentionConfig c, DlConstMatrix q,
                                 DlConstMatrix k, DlConstMatrix v,
                                 DlTiledWorkspace *s, DlMatrix context)
{
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(workspace_check(s, z, c.head_dim));
    const DlConstMatrix in[] = {q, k, v};
    TRY(check_views(in, 3, &context, 1));
    if (!shape(q, z.rows, z.query_width) || !shape(k, z.rows, z.kv_width) ||
        !shape(v, z.rows, z.kv_width) ||
        !shape(dl_matrix_read(context), z.rows, z.query_width)) return DL_SHAPE;
    for (size_t i = 0; i < 3; ++i) if (!finite_cells(in[i])) return DL_NUMERIC;
    return attend(c, q, k, v, s, context);
}

DlStatus dl_attention_tiled_forward(DlBackend backend, DlAttentionConfig c,
                                    DlConstMatrix x, DlAttentionWeights w,
                                    const size_t *positions, size_t position_count,
                                    const double *freq, size_t frequency_count,
                                    DlTiledAttentionCache a,
                                    DlTiledWorkspace *s, DlMatrix y)
{
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) return DL_ARGUMENT;
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(workspace_check(s, z, c.head_dim));
    const DlConstMatrix in[] = {x, w.query, w.key, w.value, w.output};
    const DlMatrix out[] = {a.q0, a.k0, a.v, a.q, a.k, a.context, y};
    TRY(check_views(in, 5, out, 7));
    const size_t width = x.cols;
    if (x.rows != z.rows || !shape(w.query, z.query_width, width) ||
        !shape(w.key, z.kv_width, width) || !shape(w.value, z.kv_width, width) ||
        !shape(w.output, width, z.query_width) ||
        !shape(dl_matrix_read(a.q0), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k0), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.v), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.q), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.context), z.rows, z.query_width) ||
        !shape(dl_matrix_read(y), z.rows, width)) return DL_SHAPE;
    if (positions == NULL || freq == NULL) return DL_ARGUMENT;
    if (position_count != z.rows || frequency_count != c.head_dim / 2) return DL_SHAPE;
    size_t bytes = 0;
    TRY(dl_size_mul(position_count, sizeof(*positions), &bytes));
    TRY(dl_size_mul(frequency_count, sizeof(*freq), &bytes));
    for (size_t i = 0; i < position_count; ++i)
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    for (size_t i = 0; i < frequency_count; ++i)
        if (!isfinite(freq[i]) || freq[i] <= 0.0 || freq[i] > 1.0) return DL_ARGUMENT;
    for (size_t i = 0; i < 5; ++i) if (!finite_cells(in[i])) return DL_NUMERIC;
    TRY(dl_matmul(backend, x, DL_NOTRANS, w.query, DL_TRANS, a.q0, DL_SET));
    TRY(dl_matmul(backend, x, DL_NOTRANS, w.key, DL_TRANS, a.k0, DL_SET));
    TRY(dl_matmul(backend, x, DL_NOTRANS, w.value, DL_TRANS, a.v, DL_SET));
    TRY(dl_rope_forward(dl_matrix_read(a.q0), c.head_dim, positions, position_count,
                         freq, frequency_count, a.q));
    TRY(dl_rope_forward(dl_matrix_read(a.k0), c.head_dim, positions, position_count,
                         freq, frequency_count, a.k));
    TRY(attend(c, dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.v), s, a.context));
    return dl_matmul(backend, dl_matrix_read(a.context), DL_NOTRANS, w.output, DL_TRANS, y, DL_SET);
}

static DlStatus backward_count(size_t d, size_t qr, size_t kc, size_t *count)
{
    if (d == 0 || d % 2 != 0 || qr == 0 || kc == 0) return DL_SHAPE;
    size_t total = 0, term = 0, bytes = 0;
    TRY(dl_size_mul(qr, kc, &term));
    TRY(dl_size_mul(2, term, &total));
    TRY(dl_size_mul(qr, d, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(kc, d, &term));
    TRY(dl_size_mul(2, term, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_add(total, qr, &total));
    TRY(dl_size_mul(total, sizeof(double), &bytes));
    *count = total;
    return DL_OK;
}

void dl_tiled_backward_workspace_destroy(DlTiledBackwardWorkspace *s)
{
    if (s == NULL) return;
    free(s->data);
    *s = (DlTiledBackwardWorkspace){0};
}

DlStatus dl_tiled_backward_workspace_create(DlTiledBackwardWorkspace *out,
    size_t d, size_t qr, size_t kc, size_t max_bytes)
{
    if (out == NULL || max_bytes == 0) return DL_ARGUMENT;
    if (out->data != NULL || out->count != 0 || out->head_dim != 0 ||
        out->query_tile != 0 || out->key_tile != 0) return DL_STATE;
    DlTiledBackwardWorkspace next = {.head_dim = d, .query_tile = qr, .key_tile = kc};
    TRY(backward_count(d, qr, kc, &next.count));
    size_t bytes = next.count * sizeof(*next.data);
    if (bytes > max_bytes) return DL_RANGE;
    next.data = malloc(bytes);
    if (next.data == NULL) return DL_ALLOC;
    *out = next;
    return DL_OK;
}

static DlStatus backward_state(DlAttentionConfig c, Layout z,
    const DlTiledWorkspace *forward, const DlTiledBackwardWorkspace *s)
{
    TRY(workspace_check(forward, z, c.head_dim));
    if (s == NULL || s->data == NULL) return DL_ARGUMENT;
    size_t count = 0;
    TRY(backward_count(s->head_dim, s->query_tile, s->key_tile, &count));
    if (s->count != count) return DL_STATE;
    if (s->head_dim != c.head_dim) return DL_SHAPE;
    if (s->data == forward->data) return DL_ALIAS;
    for (size_t sr = 0; sr < z.stat_rows; ++sr) {
        double maximum = forward->data[sr];
        double sum = forward->data[forward->row_capacity + sr];
        double lse = forward->data[2 * forward->row_capacity + sr];
        /* One allowed maximum contributes exp(0)=1. Roundoff allows a tiny
         * excess over the number of allowed terms, never an arbitrary factor. */
        double allowed = (double)(sr % c.time + 1);
        if (!isfinite(maximum) || fabs(maximum) > (double)FLT_MAX ||
            !isfinite(sum) || sum < 1.0 || sum > allowed * (1.0 + 1e-12) ||
            !isfinite(lse)) return DL_NUMERIC;
    }
    return DL_OK;
}

static size_t at(DlConstMatrix a, size_t row, size_t column)
{
    return a.offset + row * a.ld + column;
}

/* One score/probability and score-derivative tile. Center is local to its
 * query rows. Masked cells are explicit zero and never enter a reduction. */
static DlStatus derivative_tile(DlAttentionConfig c,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dc, const DlTiledWorkspace *f,
    size_t b, size_t h, size_t qb, size_t nq, size_t kb, size_t nk,
    size_t stride, double *prob, double *ds, double *center)
{
    size_t d = c.head_dim, kh = h / (c.query_heads / c.kv_heads);
    double scale = 1.0 / sqrt((double)d);
    for (size_t r = 0; r < nq; ++r) {
        size_t t = qb + r, row = b * c.time + t;
        size_t sr = (b * c.query_heads + h) * c.time + t;
        size_t qa = at(q, row, h * d), ca = at(context, row, h * d);
        size_t ga = at(dc, row, h * d);
        center[r] = 0.0;
        for (size_t j = 0; j < d; ++j)
            center[r] += (double)context.base[ca + j] * (double)dc.base[ga + j];
        if (!isfinite(center[r])) return DL_NUMERIC;
        for (size_t col = 0; col < nk; ++col) {
            size_t index = r * stride + col, key = kb + col;
            prob[index] = ds[index] = 0.0;
            if (key > t) continue;
            size_t ka = at(k, b * c.time + key, kh * d);
            size_t va = at(v, b * c.time + key, kh * d);
            double dot = 0.0, dp = 0.0;
            for (size_t j = 0; j < d; ++j) {
                dot += (double)q.base[qa + j] * (double)k.base[ka + j];
                dp += (double)dc.base[ga + j] * (double)v.base[va + j];
            }
            float score = 0.0f;
            TRY(store_float(&score, dot * scale));
            double delta = (double)score - f->data[sr];
            if (delta > 0.0) return DL_NUMERIC; /* Inconsistent saved maximum. */
            double p = exp(delta) / f->data[f->row_capacity + sr];
            double g = p * (dp - center[r]);
            if (!isfinite(p) || !isfinite(g)) return DL_NUMERIC;
            prob[index] = p;
            ds[index] = g;
        }
    }
    return DL_OK;
}

static DlStatus write_gradient(DlMatrix out, size_t row, size_t col,
                                double value, DlWriteMode mode)
{
    float *cell = &out.base[out.offset + row * out.ld + col];
    if (mode == DL_ADD) value += (double)*cell;
    return store_float(cell, value);
}

/* Two sweeps give each output tile one owner. No partially rounded dQ is
 * repeatedly read from float memory, and all sharing heads enter dK/dV. */
static DlStatus backward_compute(DlAttentionConfig c,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dc, const DlTiledWorkspace *f,
    DlTiledBackwardWorkspace *s, DlMatrix dq, DlMatrix dk, DlMatrix dv,
    DlWriteMode mode)
{
    size_t qr = s->query_tile, kc = s->key_tile, d = c.head_dim;
    size_t group = c.query_heads / c.kv_heads;
    double scale = 1.0 / sqrt((double)d);
    double *prob = s->data, *ds = prob + qr * kc;
    double *gq = ds + qr * kc, *gk = gq + qr * d;
    double *gv = gk + kc * d, *center = gv + kc * d;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            size_t kh = h / group;
            for (size_t qb = 0; qb < c.time; ) {
                size_t nq = smaller(qr, c.time - qb);
                for (size_t i = 0; i < nq * d; ++i) gq[i] = 0.0;
                for (size_t kb = 0; kb < qb + nq; ) {
                    size_t nk = smaller(kc, c.time - kb);
                    TRY(derivative_tile(c, q, k, v, context, dc, f,
                        b, h, qb, nq, kb, nk, kc, prob, ds, center));
                    for (size_t r = 0; r < nq; ++r)
                        for (size_t col = 0; col < nk && kb + col <= qb + r; ++col) {
                            size_t ka = at(k, b * c.time + kb + col, kh * d);
                            for (size_t j = 0; j < d; ++j)
                                gq[r * d + j] += ds[r * kc + col] * (double)k.base[ka + j];
                        }
                    kb += nk;
                }
                for (size_t r = 0; r < nq; ++r)
                    for (size_t j = 0; j < d; ++j)
                        TRY(write_gradient(dq, b * c.time + qb + r, h * d + j,
                                            scale * gq[r * d + j], mode));
                qb += nq;
            }
        }
        for (size_t kh = 0; kh < c.kv_heads; ++kh) {
            for (size_t kb = 0; kb < c.time; ) {
                size_t nk = smaller(kc, c.time - kb);
                for (size_t i = 0; i < nk * d; ++i) gk[i] = gv[i] = 0.0;
                for (size_t gh = 0; gh < group; ++gh) {
                    size_t h = kh * group + gh;
                    /* Queries earlier than kb cannot consume any of this tile. */
                    for (size_t qb = kb; qb < c.time; ) {
                        size_t nq = smaller(qr, c.time - qb);
                        TRY(derivative_tile(c, q, k, v, context, dc, f,
                            b, h, qb, nq, kb, nk, kc, prob, ds, center));
                        for (size_t r = 0; r < nq; ++r) {
                            size_t qa = at(q, b * c.time + qb + r, h * d);
                            size_t ga = at(dc, b * c.time + qb + r, h * d);
                            for (size_t col = 0; col < nk && kb + col <= qb + r; ++col)
                                for (size_t j = 0; j < d; ++j) {
                                    gk[col * d + j] += ds[r * kc + col] * (double)q.base[qa + j];
                                    gv[col * d + j] += prob[r * kc + col] * (double)dc.base[ga + j];
                                }
                        }
                        qb += nq;
                    }
                }
                for (size_t col = 0; col < nk; ++col)
                    for (size_t j = 0; j < d; ++j) {
                        TRY(write_gradient(dk, b * c.time + kb + col, kh * d + j,
                                            scale * gk[col * d + j], mode));
                        TRY(write_gradient(dv, b * c.time + kb + col, kh * d + j,
                                            gv[col * d + j], mode));
                    }
                kb += nk;
            }
        }
    }
    return DL_OK;
}

DlStatus dl_attention_tiled_core_backward(DlAttentionConfig c,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dc,
    const DlTiledWorkspace *f, DlTiledBackwardWorkspace *s,
    DlMatrix dq, DlMatrix dk, DlMatrix dv, DlWriteMode mode)
{
    if (mode != DL_SET && mode != DL_ADD) return DL_ARGUMENT;
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(backward_state(c, z, f, s));
    const DlConstMatrix in[] = {q, k, v, context, dc};
    const DlMatrix out[] = {dq, dk, dv};
    TRY(check_views(in, 5, out, 3));
    size_t widths[] = {z.query_width, z.kv_width, z.kv_width, z.query_width, z.query_width};
    for (size_t i = 0; i < 5; ++i) {
        if (!shape(in[i], z.rows, widths[i])) return DL_SHAPE;
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    for (size_t o = 0; o < 3; ++o) {
        if (!shape(dl_matrix_read(out[o]), z.rows, widths[o])) return DL_SHAPE;
        if (mode == DL_ADD && !finite_cells(dl_matrix_read(out[o]))) return DL_NUMERIC;
    }
    return backward_compute(c, q, k, v, context, dc, f, s, dq, dk, dv, mode);
}

DlStatus dl_attention_tiled_backward(DlBackend backend, DlAttentionConfig c,
    DlConstMatrix x, DlAttentionWeights w,
    const size_t *positions, size_t position_count,
    const double *freq, size_t frequency_count,
    DlTiledAttentionCache a, const DlTiledWorkspace *f,
    DlConstMatrix dy, DlTiledAttentionScratch g,
    DlTiledBackwardWorkspace *s, DlMatrix dx, DlWriteMode dx_mode,
    DlAttentionWeightGrads dw, DlWriteMode weight_mode)
{
    if ((backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) ||
        (dx_mode != DL_SET && dx_mode != DL_ADD) ||
        (weight_mode != DL_SET && weight_mode != DL_ADD)) return DL_ARGUMENT;
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(backward_state(c, z, f, s));
    const DlConstMatrix in[] = {x, w.query, w.key, w.value, w.output,
        dl_matrix_read(a.q0), dl_matrix_read(a.k0), dl_matrix_read(a.v),
        dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.context), dy};
    const DlMatrix out[] = {g.context, g.q, g.k, g.v, g.q0, g.k0,
                            dx, dw.query, dw.key, dw.value, dw.output};
    TRY(check_views(in, 12, out, 11));
    size_t m = z.rows, d = x.cols, q = z.query_width, k = z.kv_width;
    const size_t ir[] = {m,q,k,k,d,m,m,m,m,m,m,m};
    const size_t ic[] = {d,d,d,d,q,q,k,k,q,k,q,d};
    const size_t orows[] = {m,m,m,m,m,m,m,q,k,k,d};
    const size_t ocols[] = {q,q,k,k,q,k,d,d,d,d,q};
    for (size_t i = 0; i < 12; ++i) {
        if (!shape(in[i], ir[i], ic[i])) return DL_SHAPE;
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    for (size_t o = 0; o < 11; ++o)
        if (!shape(dl_matrix_read(out[o]), orows[o], ocols[o])) return DL_SHAPE;
    if (positions == NULL || freq == NULL) return DL_ARGUMENT;
    if (position_count != m || frequency_count != c.head_dim / 2) return DL_SHAPE;
    size_t bytes = 0;
    TRY(dl_size_mul(position_count, sizeof(*positions), &bytes));
    TRY(dl_size_mul(frequency_count, sizeof(*freq), &bytes));
    for (size_t i = 0; i < position_count; ++i)
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    for (size_t i = 0; i < frequency_count; ++i)
        if (!isfinite(freq[i]) || freq[i] <= 0.0 || freq[i] > 1.0) return DL_ARGUMENT;
    if (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) return DL_NUMERIC;
    for (size_t o = 7; o < 11; ++o)
        if (weight_mode == DL_ADD && !finite_cells(dl_matrix_read(out[o]))) return DL_NUMERIC;
    TRY(dl_matmul(backend, dy, DL_NOTRANS, w.output, DL_NOTRANS, g.context, DL_SET));
    TRY(dl_matmul(backend, dy, DL_TRANS, dl_matrix_read(a.context), DL_NOTRANS,
                   dw.output, weight_mode));
    TRY(backward_compute(c, dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.v),
        dl_matrix_read(a.context), dl_matrix_read(g.context), f, s, g.q, g.k, g.v, DL_SET));
    TRY(dl_rope_backward(dl_matrix_read(g.q), c.head_dim, positions, position_count,
                         freq, frequency_count, g.q0, DL_SET));
    TRY(dl_rope_backward(dl_matrix_read(g.k), c.head_dim, positions, position_count,
                         freq, frequency_count, g.k0, DL_SET));
    const DlConstMatrix upstream[] = {dl_matrix_read(g.q0), dl_matrix_read(g.k0),
                                       dl_matrix_read(g.v)};
    const DlConstMatrix weights[] = {w.query, w.key, w.value};
    const DlMatrix gradients[] = {dw.query, dw.key, dw.value};
    for (size_t p = 0; p < 3; ++p) {
        TRY(dl_matmul(backend, upstream[p], DL_TRANS, x, DL_NOTRANS,
                       gradients[p], weight_mode));
        TRY(dl_matmul(backend, upstream[p], DL_NOTRANS, weights[p], DL_NOTRANS,
                       dx, p == 0 ? dx_mode : DL_ADD));
    }
    return DL_OK;
}

#undef TRY
