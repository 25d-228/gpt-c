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

/* Used with at most five inputs and seven float outputs. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                            const DlMatrix *out, size_t no)
{
    size_t ie[5] = {0}, oe[7] = {0};
    if (ni > 5 || no > 7) return DL_ARGUMENT;
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

#undef TRY
