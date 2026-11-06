#include "dl/inference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

void dl_weights_destroy(DlWeights *w)
{
    if (w == NULL) return;
    free(w->params);
    dl_tensor_destroy(&w->arena);
    *w = (DlWeights){0};
}

static DlStatus weight_record(DlWeights *w, size_t index, size_t *cursor,
                               const char *name, size_t rows, size_t cols,
                               int decay)
{
    DlWeight *p = &w->params[index];
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    int n = snprintf(p->name, sizeof(p->name), "%s", name);
    if (n < 0 || (size_t)n >= sizeof(p->name)) return DL_RANGE;
    RETURN_IF_ERROR(dl_matrix_view(&w->arena, *cursor, rows, cols, cols, &p->value));
    p->decay = decay;
    *cursor = end;
    return DL_OK;
}

static DlStatus weight_directory(DlWeights *w)
{
    DlModelConfig c = w->config;
    size_t cursor = 0;
    RETURN_IF_ERROR(weight_record(w, 0, &cursor, "token_embedding", c.vocab, c.width, 1));
    size_t kv = c.kv_heads * (c.width / c.query_heads);
    const char *names[] = {"attn_norm", "query", "key", "value", "output",
                           "ffn_norm", "gate", "up", "down"};
    const size_t rows[] = {1, c.width, kv, kv, c.width, 1,
                           c.ffn_width, c.ffn_width, c.width};
    const size_t cols[] = {c.width, c.width, c.width, c.width, c.width,
                           c.width, c.width, c.width, c.ffn_width};
    for (size_t layer = 0; layer < c.layers; ++layer) {
        for (size_t slot = 0; slot < DL_BLOCK_PARAMS; ++slot) {
            char name[64];
            int n = snprintf(name, sizeof(name), "blocks.%zu.%s", layer, names[slot]);
            if (n < 0 || (size_t)n >= sizeof(name)) return DL_RANGE;
            int scale = slot == DL_P_ATTN_NORM || slot == DL_P_FFN_NORM;
            RETURN_IF_ERROR(weight_record(w, 1 + layer * DL_BLOCK_PARAMS + slot,
                &cursor, name, rows[slot], cols[slot], !scale));
        }
    }
    RETURN_IF_ERROR(weight_record(w, w->param_count - 1, &cursor,
                                  "final_norm", 1, c.width, 0));
    return cursor == w->arena.count ? DL_OK : DL_STATE;
}

DlStatus dl_weights_create(DlWeights *out, DlModelConfig config,
                           size_t max_float_bytes)
{
    if (out == NULL || max_float_bytes == 0) return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->ready) return DL_STATE;
    DlWeights next = {.config = config};
    size_t count = 0, bytes = 0, directory_bytes = 0;
    RETURN_IF_ERROR(dl_model_config_validate(config, &next.param_count, &count));
    RETURN_IF_ERROR(dl_size_mul(count, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(next.param_count, sizeof(*next.params), &directory_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    next.params = malloc(directory_bytes);
    if (next.params == NULL) return DL_ALLOC;
    for (size_t i = 0; i < next.param_count; ++i) next.params[i] = (DlWeight){0};
    DlStatus status = dl_tensor_create(&next.arena, 1, &count);
    if (status == DL_OK) status = weight_directory(&next);
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    for (size_t i = 0; i < count; ++i) next.arena.data[i] = 0.0f;
    *out = next;
    return DL_OK;
}

DlStatus dl_weights_from_model(DlWeights *out, const DlModel *model,
                               size_t max_float_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL) return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->ready) return DL_STATE;
    DlWeights next = {0};
    RETURN_IF_ERROR(dl_weights_create(&next, model->config, max_float_bytes));
    DlStatus status = DL_OK;
    for (size_t p = 0; p < next.param_count && status == DL_OK; ++p) {
        const DlParameter *source = &model->params[p];
        DlMatrix target = next.params[p].value;
        if (strcmp(source->name, next.params[p].name) != 0 ||
            source->value.shape[0] != target.rows || source->value.shape[1] != target.cols) {
            status = DL_SHAPE;
            break;
        }
        for (size_t i = 0; i < source->value.count; ++i) {
            float value = source->value.data[i];
            if (!isfinite(value)) { status = DL_NUMERIC; break; }
            target.base[target.offset + i] = value;
        }
    }
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    next.ready = 1;
    *out = next;
    return DL_OK;
}

typedef struct {
    DlMatrix hidden, norm, branch, middle;
    DlAttentionCache attention;
    DlFfnCache ffn;
} Views;

static DlStatus place(DlTensor *arena, size_t *cursor, size_t rows, size_t cols,
                       DlMatrix *view)
{
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    if (arena->data != NULL)
        RETURN_IF_ERROR(dl_matrix_view(arena, *cursor, rows, cols, cols, view));
    *cursor = end;
    return DL_OK;
}

static DlStatus layout(DlTensor *arena, DlModelConfig c, size_t time,
                        Views *v, size_t *count)
{
    size_t cursor = 0, score_rows = 0;
    size_t kv = c.kv_heads * (c.width / c.query_heads);
    RETURN_IF_ERROR(dl_size_mul(time, c.query_heads, &score_rows));
#define PLACE(rows_, cols_, member_) \
    RETURN_IF_ERROR(place(arena, &cursor, rows_, cols_, &v->member_))
    PLACE(time, c.width, hidden);
    PLACE(time, c.width, norm);
    PLACE(time, c.width, branch);
    PLACE(time, c.width, middle);
    PLACE(time, c.width, attention.q0);
    PLACE(time, kv, attention.k0);
    PLACE(time, kv, attention.v);
    PLACE(time, c.width, attention.q);
    PLACE(time, kv, attention.k);
    PLACE(score_rows, time, attention.scores);
    PLACE(score_rows, time, attention.probs);
    PLACE(time, c.width, attention.context);
    PLACE(time, c.ffn_width, ffn.gate);
    PLACE(time, c.ffn_width, ffn.value);
    PLACE(time, c.ffn_width, ffn.hidden);
#undef PLACE
    *count = cursor;
    return DL_OK;
}

void dl_inference_destroy(DlInference *s)
{
    if (s == NULL) return;
    free(s->inv_freq);
    free(s->positions);
    dl_tensor_destroy(&s->arena);
    *s = (DlInference){0};
}

DlStatus dl_inference_create(DlInference *out, const DlWeights *weights,
                             size_t capacity, size_t max_float_bytes)
{
    if (out == NULL || weights == NULL || weights->params == NULL ||
        capacity == 0 || max_float_bytes == 0) return DL_ARGUMENT;
    if (!weights->ready) return DL_STATE;
    if (out->weights != NULL || out->arena.data != NULL ||
        out->positions != NULL || out->inv_freq != NULL || out->capacity != 0)
        return DL_STATE;
    if (capacity - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlInference next = {.weights = weights, .capacity = capacity};
    Views views = {0};
    size_t count = 0, bytes = 0, position_bytes = 0, frequency_bytes = 0;
    size_t frequencies = weights->config.width / weights->config.query_heads / 2;
    RETURN_IF_ERROR(layout(&next.arena, weights->config, capacity, &views, &count));
    RETURN_IF_ERROR(dl_size_mul(count, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(capacity, sizeof(*next.positions), &position_bytes));
    RETURN_IF_ERROR(dl_size_mul(frequencies, sizeof(*next.inv_freq), &frequency_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    DlStatus status = dl_tensor_create(&next.arena, 1, &count);
    if (status == DL_OK) {
        next.positions = malloc(position_bytes);
        next.inv_freq = malloc(frequency_bytes);
        if (next.positions == NULL || next.inv_freq == NULL) status = DL_ALLOC;
    }
    if (status == DL_OK) {
        for (size_t i = 0; i < capacity; ++i) next.positions[i] = i;
        status = dl_rope_frequencies(weights->config.width / weights->config.query_heads,
                                     weights->config.rope_theta, next.inv_freq, frequencies);
    }
    if (status != DL_OK) { dl_inference_destroy(&next); return status; }
    *out = next;
    return DL_OK;
}

static DlConstMatrix weight(const DlWeights *w, size_t index)
{
    return dl_matrix_read(w->params[index].value);
}

DlStatus dl_inference_forward(DlInference *s, DlBackend backend,
                              const size_t *ids, size_t length,
                              size_t first_row, DlMatrix logits)
{
    if (s == NULL || s->weights == NULL || s->arena.data == NULL || ids == NULL)
        return DL_ARGUMENT;
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) return DL_ARGUMENT;
    const DlWeights *w = s->weights;
    if (!w->ready) return DL_STATE;
    DlModelConfig c = w->config;
    if (length == 0 || length > s->capacity || first_row >= length) return DL_RANGE;
    size_t end = 0;
    RETURN_IF_ERROR(dl_matrix_end(dl_matrix_read(logits), &end));
    if (logits.cols != c.vocab || logits.rows > length - first_row) return DL_SHAPE;
    if (logits.base == s->arena.data || logits.base == w->arena.data) return DL_ALIAS;
    for (size_t i = 0; i < length; ++i) if (ids[i] >= c.vocab) return DL_RANGE;
    Views v = {0};
    size_t count = 0;
    RETURN_IF_ERROR(layout(&s->arena, c, length, &v, &count));
    DlAttentionConfig a = {1, length, c.query_heads, c.kv_heads, c.width / c.query_heads};
    RETURN_IF_ERROR(dl_embedding_forward(weight(w, 0), ids, length, v.hidden));
    for (size_t layer = 0; layer < c.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        DlAttentionWeights aw = {weight(w, base + DL_P_QUERY), weight(w, base + DL_P_KEY),
            weight(w, base + DL_P_VALUE), weight(w, base + DL_P_OUTPUT)};
        DlFfnWeights fw = {weight(w, base + DL_P_GATE), weight(w, base + DL_P_UP),
                           weight(w, base + DL_P_DOWN)};
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.hidden),
            weight(w, base + DL_P_ATTN_NORM), c.epsilon, v.norm));
        RETURN_IF_ERROR(dl_attention_forward(backend, a, dl_matrix_read(v.norm), aw,
            s->positions, length, s->inv_freq, a.head_dim / 2, v.attention, v.branch));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.hidden), dl_matrix_read(v.branch), v.middle));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.middle),
            weight(w, base + DL_P_FFN_NORM), c.epsilon, v.norm));
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(v.norm), fw, v.ffn, v.branch));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.middle), dl_matrix_read(v.branch), v.hidden));
    }
    DlMatrix selected = v.hidden;
    selected.offset += first_row * selected.ld; /* Bounded by length above. */
    selected.rows = logits.rows;
    DlMatrix norm = v.norm;
    norm.rows = logits.rows;
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(selected),
        weight(w, w->param_count - 1), c.epsilon, norm));
    return dl_matmul(backend, dl_matrix_read(norm), DL_NOTRANS,
                     weight(w, 0), DL_TRANS, logits, DL_SET);
}

#undef RETURN_IF_ERROR
