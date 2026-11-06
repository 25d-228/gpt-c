#include "dl/cache.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

typedef struct {
    DlMatrix hidden, norm, branch, middle, q0, k0, q, context;
    DlMatrix scores, probs;          /* ONE row of length capacity, reused. */
    DlFfnCache ffn;
} Views;

static DlStatus place(DlTensor *arena, size_t *cursor, size_t rows,
                       size_t cols, DlMatrix *view)
{
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    if (arena->data != NULL)
        RETURN_IF_ERROR(dl_matrix_view(arena, *cursor, rows, cols, cols, view));
    *cursor = end;
    return DL_OK;
}

static DlStatus layout(DlTensor *arena, DlModelConfig c, size_t rows,
                        size_t capacity, Views *v, size_t *count)
{
    size_t cursor = 0;
    size_t kv = c.kv_heads * (c.width / c.query_heads);
#define PLACE(rows_, cols_, member_) \
    RETURN_IF_ERROR(place(arena, &cursor, rows_, cols_, &v->member_))
    PLACE(rows, c.width, hidden);
    PLACE(rows, c.width, norm);
    PLACE(rows, c.width, branch);
    PLACE(rows, c.width, middle);
    PLACE(rows, c.width, q0);
    PLACE(rows, kv, k0);
    PLACE(rows, c.width, q);
    PLACE(rows, c.width, context);
    PLACE(rows, c.ffn_width, ffn.gate);
    PLACE(rows, c.ffn_width, ffn.value);
    PLACE(rows, c.ffn_width, ffn.hidden);
    PLACE(1, capacity, scores);
    PLACE(1, capacity, probs);
#undef PLACE
    *count = cursor;
    return DL_OK;
}

void dl_cache_destroy(DlKvCache *s)
{
    if (s == NULL) return;
    free(s->inv_freq);
    free(s->positions);
    dl_tensor_destroy(&s->arena);
    dl_tensor_destroy(&s->history);
    *s = (DlKvCache){0};
}

DlStatus dl_cache_create(DlKvCache *out, const DlWeights *weights,
                          size_t capacity, size_t max_float_bytes)
{
    if (out == NULL || weights == NULL || weights->params == NULL ||
        capacity == 0 || max_float_bytes == 0) return DL_ARGUMENT;
    if (!weights->ready) return DL_STATE;
    if (out->weights != NULL || out->history.data != NULL || out->arena.data != NULL ||
        out->positions != NULL || out->inv_freq != NULL ||
        out->capacity != 0 || out->length != 0) return DL_STATE;
    if (capacity - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlModelConfig c = weights->config;
    size_t params = 0, values = 0;
    RETURN_IF_ERROR(dl_model_config_validate(c, &params, &values));
    size_t d = c.width / c.query_heads, kv = c.kv_heads * d;
    DlKvCache next = {.weights = weights, .capacity = capacity};
    Views views = {0};
    size_t scratch = 0, history = 0, total = 0, bytes = 0;
    size_t position_bytes = 0, frequency_bytes = 0;
    RETURN_IF_ERROR(layout(&next.arena, c, capacity, capacity, &views, &scratch));
    RETURN_IF_ERROR(dl_size_mul(c.layers, 2, &history));
    RETURN_IF_ERROR(dl_size_mul(history, capacity, &history));
    RETURN_IF_ERROR(dl_size_mul(history, kv, &history));
    RETURN_IF_ERROR(dl_size_add(history, scratch, &total));
    RETURN_IF_ERROR(dl_size_mul(total, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(capacity, sizeof(*next.positions), &position_bytes));
    RETURN_IF_ERROR(dl_size_mul(d / 2, sizeof(*next.inv_freq), &frequency_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    const size_t shape[] = {c.layers, 2, capacity, kv};
    DlStatus status = dl_tensor_create(&next.history, 4, shape);
    if (status == DL_OK) status = dl_tensor_create(&next.arena, 1, &scratch);
    if (status == DL_OK) {
        next.positions = malloc(position_bytes);
        next.inv_freq = malloc(frequency_bytes);
        if (next.positions == NULL || next.inv_freq == NULL) status = DL_ALLOC;
    }
    if (status == DL_OK)
        status = dl_rope_frequencies(d, c.rope_theta, next.inv_freq, d / 2);
    if (status != DL_OK) { dl_cache_destroy(&next); return status; }
    *out = next;
    return DL_OK;
}

static DlStatus live(const DlKvCache *s)
{
    if (s == NULL || s->weights == NULL || s->history.data == NULL ||
        s->arena.data == NULL || s->positions == NULL || s->inv_freq == NULL)
        return DL_STATE;
    if (!s->weights->ready || s->length > s->capacity) return DL_STATE;
    return DL_OK;
}

DlStatus dl_cache_reset(DlKvCache *s)
{
    RETURN_IF_ERROR(live(s));
    s->length = 0;
    return DL_OK;
}

/* Only constructed, unchanged owners reach these internal helpers. */
static DlConstMatrix weight(const DlWeights *w, size_t index)
{
    return dl_matrix_read(w->params[index].value);
}

static DlStatus history_view(DlKvCache *s, size_t layer, size_t kind,
                               size_t start, size_t rows, DlMatrix *out)
{
    size_t kv = s->weights->config.kv_heads *
                (s->weights->config.width / s->weights->config.query_heads);
    /* The constructor checked 2*L*C*K, and caller bounds every index. */
    size_t offset = ((2 * layer + kind) * s->capacity + start) * kv;
    return dl_matrix_view(&s->history, offset, rows, kv, kv, out);
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

/* Materialize one allowed score/probability ROW, not an online softmax.
 * Inputs/views are private, sized by layout/history_view; no unknown aliases. */
static DlStatus attend(DlModelConfig c, size_t start, Views v,
                        DlConstMatrix key, DlConstMatrix value)
{
    size_t d = c.width / c.query_heads, group = c.query_heads / c.kv_heads;
    double scale = 1.0 / sqrt((double)d);
    float *scores = v.scores.base + v.scores.offset;
    float *probs = v.probs.base + v.probs.offset;
    for (size_t r = 0; r < v.q.rows; ++r) {
        size_t allowed = start + r + 1;
        for (size_t h = 0; h < c.query_heads; ++h) {
            size_t kh = h / group;
            size_t qa = v.q.offset + r * v.q.ld + h * d;
            double maximum = -INFINITY;
            for (size_t t = 0; t < allowed; ++t) {
                size_t ka = key.offset + t * key.ld + kh * d;
                double dot = 0.0;
                for (size_t j = 0; j < d; ++j)
                    dot += (double)v.q.base[qa + j] * (double)key.base[ka + j];
                RETURN_IF_ERROR(store_float(&scores[t], dot * scale));
                maximum = fmax(maximum, (double)scores[t]);
            }
            double denominator = 0.0;
            for (size_t t = 0; t < allowed; ++t)
                denominator += exp((double)scores[t] - maximum);
            if (!isfinite(denominator) || denominator <= 0.0) return DL_NUMERIC;
            for (size_t t = 0; t < allowed; ++t)
                RETURN_IF_ERROR(store_float(&probs[t],
                    exp((double)scores[t] - maximum) / denominator));
            for (size_t j = 0; j < d; ++j) {
                double sum = 0.0;
                for (size_t t = 0; t < allowed; ++t)
                    sum += (double)probs[t] *
                        (double)value.base[value.offset + t * value.ld + kh * d + j];
                RETURN_IF_ERROR(store_float(&v.context.base[
                    v.context.offset + r * v.context.ld + h * d + j], sum));
            }
        }
    }
    return DL_OK;
}

static DlStatus run(DlKvCache *s, DlBackend backend, const size_t *ids,
                      size_t count, DlMatrix logits)
{
    RETURN_IF_ERROR(live(s));
    if (ids == NULL || (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED))
        return DL_ARGUMENT;
    const DlWeights *w = s->weights;
    DlModelConfig c = w->config;
    if (count == 0 || count > s->capacity - s->length) return DL_RANGE;
    size_t end = 0;
    RETURN_IF_ERROR(dl_matrix_end(dl_matrix_read(logits), &end));
    if (logits.rows != 1 || logits.cols != c.vocab) return DL_SHAPE;
    if (logits.base == s->history.data || logits.base == s->arena.data ||
        logits.base == w->arena.data) return DL_ALIAS;
    for (size_t i = 0; i < count; ++i) if (ids[i] >= c.vocab) return DL_RANGE;
    Views v = {0};
    size_t scratch = 0, start = s->length, next = start + count;
    RETURN_IF_ERROR(layout(&s->arena, c, count, s->capacity, &v, &scratch));
    for (size_t i = 0; i < count; ++i) s->positions[i] = start + i;
    RETURN_IF_ERROR(dl_embedding_forward(weight(w, 0), ids, count, v.hidden));
    for (size_t layer = 0; layer < c.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        size_t d = c.width / c.query_heads;
        DlMatrix key = {0}, value = {0}, new_key = {0}, new_value = {0};
        RETURN_IF_ERROR(history_view(s, layer, 0, 0, next, &key));
        RETURN_IF_ERROR(history_view(s, layer, 1, 0, next, &value));
        RETURN_IF_ERROR(history_view(s, layer, 0, start, count, &new_key));
        RETURN_IF_ERROR(history_view(s, layer, 1, start, count, &new_value));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.hidden),
            weight(w, base + DL_P_ATTN_NORM), c.epsilon, v.norm));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.norm), DL_NOTRANS,
            weight(w, base + DL_P_QUERY), DL_TRANS, v.q0, DL_SET));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.norm), DL_NOTRANS,
            weight(w, base + DL_P_KEY), DL_TRANS, v.k0, DL_SET));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.norm), DL_NOTRANS,
            weight(w, base + DL_P_VALUE), DL_TRANS, new_value, DL_SET));
        RETURN_IF_ERROR(dl_rope_forward(dl_matrix_read(v.q0), d, s->positions,
            count, s->inv_freq, d / 2, v.q));
        RETURN_IF_ERROR(dl_rope_forward(dl_matrix_read(v.k0), d, s->positions,
            count, s->inv_freq, d / 2, new_key));
        RETURN_IF_ERROR(attend(c, start, v, dl_matrix_read(key), dl_matrix_read(value)));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.context), DL_NOTRANS,
            weight(w, base + DL_P_OUTPUT), DL_TRANS, v.branch, DL_SET));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.hidden), dl_matrix_read(v.branch), v.middle));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.middle),
            weight(w, base + DL_P_FFN_NORM), c.epsilon, v.norm));
        DlFfnWeights fw = {weight(w, base + DL_P_GATE), weight(w, base + DL_P_UP),
                           weight(w, base + DL_P_DOWN)};
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(v.norm), fw, v.ffn, v.branch));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.middle), dl_matrix_read(v.branch), v.hidden));
    }
    DlMatrix last = v.hidden, norm = v.norm;
    last.offset += (count - 1) * last.ld;
    last.rows = norm.rows = 1;
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(last),
        weight(w, w->param_count - 1), c.epsilon, norm));
    RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(norm), DL_NOTRANS,
        weight(w, 0), DL_TRANS, logits, DL_SET));
    s->length = next;             /* Sole commit point, including final logits. */
    return DL_OK;
}

DlStatus dl_cache_prefill(DlKvCache *s, DlBackend backend,
                           const size_t *ids, size_t count, DlMatrix logits)
{
    RETURN_IF_ERROR(live(s));
    if (s->length != 0) return DL_STATE;
    return run(s, backend, ids, count, logits);
}

DlStatus dl_cache_decode(DlKvCache *s, DlBackend backend,
                          size_t id, DlMatrix logits)
{
    RETURN_IF_ERROR(live(s));
    if (s->length == 0) return DL_STATE;
    return run(s, backend, &id, 1, logits);
}

#undef RETURN_IF_ERROR
