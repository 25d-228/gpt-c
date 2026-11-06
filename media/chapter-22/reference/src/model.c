#include "dl/model.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RETURN_IF_ERROR(call) do {       \
    DlStatus status_ = (call);           \
    if (status_ != DL_OK) return status_;\
} while (0)

static DlStatus add_product(size_t *sum, size_t a, size_t b)
{
    size_t product = 0;
    RETURN_IF_ERROR(dl_size_mul(a, b, &product));
    return dl_size_add(*sum, product, sum);
}

DlStatus dl_model_config_validate(DlModelConfig c,
                                  size_t *parameter_count, size_t *value_count)
{
    if (parameter_count == NULL || value_count == NULL ||
        parameter_count == value_count) return DL_ARGUMENT;
    if (c.vocab < 2 || c.layers == 0 || c.width == 0 ||
        c.query_heads == 0 || c.kv_heads == 0 || c.ffn_width == 0)
        return DL_SHAPE;
    if (c.width % c.query_heads != 0 || c.query_heads % c.kv_heads != 0 ||
        (c.width / c.query_heads) % 2 != 0) return DL_SHAPE;
    if (!isfinite(c.epsilon) || c.epsilon <= 0.0f ||
        !isfinite(c.init_std) || c.init_std <= 0.0f ||
        !isfinite(c.rope_theta) || c.rope_theta <= 1.0) return DL_RANGE;
    size_t kv = 0, per_block = 0, values = 0, params = 0, bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(c.kv_heads, c.width / c.query_heads, &kv));
    /* Two D*D, two K*D, three F*D, two D scales per block. */
    for (size_t i = 0; i < 2; ++i) {
        RETURN_IF_ERROR(add_product(&per_block, c.width, c.width));
        RETURN_IF_ERROR(add_product(&per_block, kv, c.width));
        RETURN_IF_ERROR(dl_size_add(per_block, c.width, &per_block));
    }
    for (size_t i = 0; i < 3; ++i)
        RETURN_IF_ERROR(add_product(&per_block, c.ffn_width, c.width));
    RETURN_IF_ERROR(add_product(&values, c.vocab, c.width));
    RETURN_IF_ERROR(add_product(&values, c.layers, per_block));
    RETURN_IF_ERROR(dl_size_add(values, c.width, &values));
    RETURN_IF_ERROR(dl_size_mul(c.layers, DL_BLOCK_PARAMS, &params));
    RETURN_IF_ERROR(dl_size_add(params, 2, &params));
    RETURN_IF_ERROR(dl_size_mul(params, sizeof(DlParameter), &bytes));
    RETURN_IF_ERROR(dl_size_mul(values, 2 * sizeof(float), &bytes));
    *parameter_count = params;
    *value_count = values;
    return DL_OK;
}

static DlStatus read_setting(FILE *file, const char *key, char value[128])
{
    char line[128];
    size_t used = 0;
    int ch = 0;
    while ((ch = fgetc(file)) != '\n') {
        if (ch == EOF) return ferror(file) ? DL_STATE : DL_ARGUMENT;
        if (ch == 0 || used == sizeof(line) - 1) return DL_ARGUMENT;
        line[used++] = (char)ch;
    }
    if (used != 0 && line[used - 1] == '\r') --used;
    line[used] = '\0';
    size_t key_size = strlen(key);
    if (used <= key_size + 1 || strncmp(line, key, key_size) != 0 ||
        line[key_size] != '=') return DL_ARGUMENT;
    for (size_t i = key_size + 1; i < used; ++i)
        if (isspace((unsigned char)line[i])) return DL_ARGUMENT;
    memcpy(value, line + key_size + 1, used - key_size);
    return DL_OK;
}

static DlStatus decimal_u64(const char *text, uint64_t *out)
{
    uint64_t value = 0;
    if (*text == '\0') return DL_ARGUMENT;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return DL_ARGUMENT;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return DL_RANGE;
        value = value * 10 + digit;
    }
    *out = value;
    return DL_OK;
}

static DlStatus parse_settings(FILE *file, DlModelConfig *c,
                                size_t *batch, size_t *time)
{
    const char *keys[] = {
        "vocab", "layers", "width", "query_heads", "kv_heads", "ffn_width",
        "batch", "time", "epsilon", "rope_theta", "init_std", "seed"
    };
    size_t *sizes[] = {&c->vocab, &c->layers, &c->width, &c->query_heads,
                       &c->kv_heads, &c->ffn_width, batch, time};
    char value[128];
    for (size_t i = 0; i < 12; ++i) {
        RETURN_IF_ERROR(read_setting(file, keys[i], value));
        if (i < 8 || i == 11) {
            uint64_t integer = 0;
            RETURN_IF_ERROR(decimal_u64(value, &integer));
            if (i == 11) c->seed = integer;
            else {
                if (integer > SIZE_MAX) return DL_RANGE;
                *sizes[i] = (size_t)integer;
            }
        } else {
            char *end = NULL;
            errno = 0;
            double number = strtod(value, &end);
            if (end == value || *end != '\0') return DL_ARGUMENT;
            if (errno == ERANGE || !isfinite(number) || number <= 0.0)
                return DL_RANGE;
            if (i == 9) c->rope_theta = number;
            else {
                if (number > FLT_MAX || number < FLT_MIN) return DL_RANGE;
                if (i == 8) c->epsilon = (float)number;
                else c->init_std = (float)number;
            }
        }
    }
    if (fgetc(file) != EOF) return DL_ARGUMENT;
    if (ferror(file)) return DL_STATE;
    size_t parameters = 0, values = 0, rows = 0;
    RETURN_IF_ERROR(dl_model_config_validate(*c, &parameters, &values));
    if (*batch == 0 || *time == 0 || *time - 1 > DL_ROPE_MAX_POSITION)
        return DL_RANGE;
    return dl_size_mul(*batch, *time, &rows);
}

DlStatus dl_model_config_load(const char *path, DlModelConfig *config,
                              size_t *batch, size_t *time)
{
    if (path == NULL || config == NULL || batch == NULL || time == NULL ||
        batch == time) return DL_ARGUMENT;
    FILE *file = fopen(path, "rb");
    if (file == NULL) return DL_STATE;
    DlModelConfig candidate = {0};
    size_t b = 0, t = 0;
    DlStatus status = parse_settings(file, &candidate, &b, &t);
    if (fclose(file) != 0 && status == DL_OK) status = DL_STATE;
    if (status == DL_OK) {
        *config = candidate;
        *batch = b;
        *time = t;
    }
    return status;
}

static DlMatrix dense_view(DlTensor *tensor)
{
    return (DlMatrix){tensor->data, tensor->count, 0,
                      tensor->shape[0], tensor->shape[1], tensor->shape[1]};
}

DlMatrix dl_parameter_value(DlModel *m, size_t index)
{
    return dense_view(&m->params[index].value);
}

DlMatrix dl_parameter_grad(DlModel *m, size_t index)
{
    return dense_view(&m->params[index].grad);
}

static DlConstMatrix weight(DlModel *m, size_t index)
{
    return dl_matrix_read(dl_parameter_value(m, index));
}

static DlStatus create_parameter(DlParameter *p, const char *name,
                                  size_t rows, size_t cols, int scale,
                                  double stddev, DlRng *rng)
{
    int n = snprintf(p->name, sizeof(p->name), "%s", name);
    if (n < 0 || (size_t)n >= sizeof(p->name)) return DL_RANGE;
    const size_t shape[] = {rows, cols};
    RETURN_IF_ERROR(dl_tensor_create(&p->value, 2, shape));
    RETURN_IF_ERROR(dl_tensor_create(&p->grad, 2, shape));
    p->decay = !scale;
    double limit = sqrt(3.0) * stddev;
    if (!isfinite(limit) || limit > FLT_MAX) return DL_RANGE;
    for (size_t i = 0; i < p->value.count; ++i) {
        p->value.data[i] = scale ? 1.0f
            : (float)((2.0 * dl_rng_uniform(rng) - 1.0) * limit);
        p->grad.data[i] = 0.0f;
    }
    return DL_OK;
}

static DlStatus allocate_parameters(DlModel *m)
{
    size_t bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(m->param_count, sizeof(*m->params), &bytes));
    m->params = malloc(bytes);
    if (m->params == NULL) return DL_ALLOC;
    for (size_t i = 0; i < m->param_count; ++i)
        m->params[i] = (DlParameter){0};
    DlModelConfig c = m->config;
    DlRng rng = {c.seed};
    RETURN_IF_ERROR(create_parameter(&m->params[0], "token_embedding",
                                      c.vocab, c.width, 0, c.init_std, &rng));
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
            double stddev = c.init_std;
            if (slot == DL_P_OUTPUT || slot == DL_P_DOWN)
                stddev /= sqrt(2.0 * (double)c.layers);
            size_t index = 1 + layer * DL_BLOCK_PARAMS + slot;
            RETURN_IF_ERROR(create_parameter(&m->params[index], name,
                rows[slot], cols[slot], scale, stddev, &rng));
        }
    }
    return create_parameter(&m->params[m->param_count - 1], "final_norm",
                             1, c.width, 1, c.init_std, &rng);
}

void dl_model_destroy(DlModel *m)
{
    if (m == NULL) return;
    if (m->params != NULL) {
        for (size_t i = 0; i < m->param_count; ++i) {
            dl_tensor_destroy(&m->params[i].value);
            dl_tensor_destroy(&m->params[i].grad);
        }
    }
    free(m->params);
    *m = (DlModel){0};
}

DlStatus dl_model_create(DlModel *out, DlModelConfig config)
{
    if (out == NULL) return DL_ARGUMENT;
    if (out->params != NULL || out->param_count != 0) return DL_STATE;
    DlModel candidate = {0};
    candidate.config = config;
    RETURN_IF_ERROR(dl_model_config_validate(config, &candidate.param_count,
                                              &candidate.value_count));
    DlStatus status = allocate_parameters(&candidate);
    if (status != DL_OK) {
        dl_model_destroy(&candidate);
        return status;
    }
    candidate.revision = 1;
    candidate.gradients_valid = 1;   /* Constructor initialized all to zero. */
    *out = candidate;               /* Sole ownership transfer. */
    return DL_OK;
}

static DlStatus place(DlTrainTape *t, size_t *cursor, size_t rows, size_t cols,
                       DlMatrix *view)
{
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    if (t->arena.data != NULL)
        RETURN_IF_ERROR(dl_matrix_view(&t->arena, *cursor, rows, cols, cols, view));
    *cursor = end;
    return DL_OK;
}

static DlStatus place_attention(DlTrainTape *t, size_t *cursor,
                                 DlMatrix *q0, DlMatrix *k0, DlMatrix *v,
                                 DlMatrix *q, DlMatrix *k, DlMatrix *scores,
                                 DlMatrix *probs, DlMatrix *context)
{
    size_t m = t->rows, d = t->model->config.width;
    size_t kv = t->attention_config.kv_heads * t->attention_config.head_dim;
    size_t score_rows = 0;
    RETURN_IF_ERROR(dl_size_mul(m, t->attention_config.query_heads, &score_rows));
    RETURN_IF_ERROR(place(t, cursor, m, d, q0));
    RETURN_IF_ERROR(place(t, cursor, m, kv, k0));
    RETURN_IF_ERROR(place(t, cursor, m, kv, v));
    RETURN_IF_ERROR(place(t, cursor, m, d, q));
    RETURN_IF_ERROR(place(t, cursor, m, kv, k));
    if (t->attention_mode == DL_ATTENTION_MATERIALIZED) {
        RETURN_IF_ERROR(place(t, cursor, score_rows, t->attention_config.time, scores));
        RETURN_IF_ERROR(place(t, cursor, score_rows, t->attention_config.time, probs));
    } /* Tiled views stay {0}; no square count is even computed. */
    return place(t, cursor, m, d, context);
}

static DlStatus layout_tape(DlTrainTape *t, size_t *count)
{
    DlModelConfig c = t->model->config;
    size_t cursor = 0, m = t->rows;
    for (size_t i = 0; i <= c.layers; ++i)
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->hidden[i]));
    for (size_t i = 0; i < c.layers; ++i) {
        DlBlockTape *b = &t->blocks[i];
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->norm1));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->attention_out));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->middle));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->norm2));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->ffn_out));
        DlAttentionCache *a = &b->attention;
        RETURN_IF_ERROR(place_attention(t, &cursor, &a->q0, &a->k0, &a->v,
            &a->q, &a->k, &a->scores, &a->probs, &a->context));
        RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &b->ffn.gate));
        RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &b->ffn.value));
        RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &b->ffn.hidden));
    }
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->final_norm));
    RETURN_IF_ERROR(place(t, &cursor, m, c.vocab, &t->logits));
    RETURN_IF_ERROR(place(t, &cursor, m, c.vocab, &t->dlogits));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->g));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->other));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->du));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->dn));
    DlAttentionScratch *a = &t->attention_scratch;
    RETURN_IF_ERROR(place_attention(t, &cursor, &a->q0, &a->k0, &a->v,
        &a->q, &a->k, &a->scores, &a->probs, &a->context));
    RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &t->ffn_scratch.hidden));
    RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &t->ffn_scratch.gate));
    RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &t->ffn_scratch.value));
    *count = cursor;
    return DL_OK;
}

static DlStatus allocate_tiled_state(DlTrainTape *t, size_t qr, size_t kc,
                                      size_t *remaining)
{
    size_t layers = t->model->config.layers, bytes = 0, n = 0;
    RETURN_IF_ERROR(dl_size_mul(layers, sizeof(*t->tiled), &bytes));
    t->tiled = malloc(bytes);
    if (t->tiled == NULL) return DL_ALLOC;
    for (size_t i = 0; i < layers; ++i) t->tiled[i] = (DlTiledWorkspace){0};
    RETURN_IF_ERROR(dl_size_mul(t->rows, t->attention_config.query_heads, &n));
    for (size_t i = 0; i < layers; ++i) {
        if (*remaining == 0) return DL_RANGE;
        RETURN_IF_ERROR(dl_tiled_workspace_create(&t->tiled[i], n,
            t->attention_config.head_dim, qr, kc, *remaining));
        *remaining -= t->tiled[i].count * sizeof(double);
    }
    if (*remaining == 0) return DL_RANGE;
    RETURN_IF_ERROR(dl_tiled_backward_workspace_create(&t->tiled_backward,
        t->attention_config.head_dim, qr, kc, *remaining));
    *remaining -= t->tiled_backward.count * sizeof(double);
    return DL_OK;
}

static DlStatus allocate_tape(DlTrainTape *t, size_t max_bytes, size_t qr, size_t kc)
{
    size_t layers = t->model->config.layers, bytes = 0, entries = 0;
    RETURN_IF_ERROR(dl_size_add(layers, 1, &entries));
    RETURN_IF_ERROR(dl_size_mul(entries, sizeof(*t->hidden), &bytes));
    t->hidden = malloc(bytes);
    if (t->hidden == NULL) return DL_ALLOC;
    for (size_t i = 0; i < entries; ++i) t->hidden[i] = (DlMatrix){0};
    RETURN_IF_ERROR(dl_size_mul(layers, sizeof(*t->blocks), &bytes));
    t->blocks = malloc(bytes);
    if (t->blocks == NULL) return DL_ALLOC;
    for (size_t i = 0; i < layers; ++i) t->blocks[i] = (DlBlockTape){0};
    size_t count = 0;
    RETURN_IF_ERROR(layout_tape(t, &count));
    RETURN_IF_ERROR(dl_size_mul(count, sizeof(float), &bytes));
    if (bytes > max_bytes) return DL_RANGE;
    size_t remaining = max_bytes - bytes;
    if (t->attention_mode == DL_ATTENTION_TILED)
        RETURN_IF_ERROR(allocate_tiled_state(t, qr, kc, &remaining));
    RETURN_IF_ERROR(dl_tensor_create(&t->arena, 1, &count));
    RETURN_IF_ERROR(layout_tape(t, &count));
    RETURN_IF_ERROR(dl_size_mul(t->rows, 3, &entries));
    RETURN_IF_ERROR(dl_size_mul(entries, sizeof(*t->metadata), &bytes));
    t->metadata = malloc(bytes);
    if (t->metadata == NULL) return DL_ALLOC;
    t->ids = t->metadata;
    t->targets = t->metadata + t->rows;
    t->positions = t->metadata + 2 * t->rows;
    for (size_t i = 0; i < t->rows; ++i)
        t->positions[i] = i % t->attention_config.time;
    entries = t->attention_config.head_dim / 2;
    RETURN_IF_ERROR(dl_size_mul(entries, sizeof(*t->inv_freq), &bytes));
    t->inv_freq = malloc(bytes);
    if (t->inv_freq == NULL) return DL_ALLOC;
    return dl_rope_frequencies(t->attention_config.head_dim,
                               t->model->config.rope_theta, t->inv_freq, entries);
}

void dl_train_tape_destroy(DlTrainTape *t)
{
    if (t == NULL) return;
    dl_tiled_backward_workspace_destroy(&t->tiled_backward);
    if (t->tiled != NULL) {
        for (size_t i = 0; i < t->model->config.layers; ++i)
            dl_tiled_workspace_destroy(&t->tiled[i]);
    }
    free(t->tiled);
    free(t->inv_freq);
    free(t->metadata);
    free(t->blocks);
    free(t->hidden);
    dl_tensor_destroy(&t->arena);
    *t = (DlTrainTape){0};
}

static DlStatus tape_create(DlTrainTape *out, const DlModel *model,
    size_t batch, size_t time, DlAttentionMode mode, size_t qr, size_t kc, size_t max_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL ||
        batch == 0 || time == 0 || max_bytes == 0) return DL_ARGUMENT;
    if (out->model != NULL || out->arena.data != NULL || out->hidden != NULL ||
        out->blocks != NULL || out->metadata != NULL || out->inv_freq != NULL ||
        out->tiled != NULL || out->tiled_backward.data != NULL)
        return DL_STATE;
    if (time - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlTrainTape candidate = {0};
    candidate.model = model;
    candidate.attention_mode = mode;
    candidate.attention_config = (DlAttentionConfig){batch, time,
        model->config.query_heads, model->config.kv_heads,
        model->config.width / model->config.query_heads};
    RETURN_IF_ERROR(dl_size_mul(batch, time, &candidate.rows));
    DlStatus status = allocate_tape(&candidate, max_bytes, qr, kc);
    if (status != DL_OK) {
        dl_train_tape_destroy(&candidate);
        return status;
    }
    *out = candidate;
    return DL_OK;
}

DlStatus dl_train_tape_create(DlTrainTape *out, const DlModel *model,
                             size_t batch, size_t time, size_t max_float_bytes)
{
    return tape_create(out, model, batch, time, DL_ATTENTION_MATERIALIZED,
                       0, 0, max_float_bytes);
}

DlStatus dl_train_tape_create_tiled(DlTrainTape *out, const DlModel *model,
    size_t batch, size_t time, size_t qr, size_t kc, size_t max_bytes)
{
    if (qr == 0 || kc == 0) return DL_SHAPE;
    return tape_create(out, model, batch, time, DL_ATTENTION_TILED, qr, kc, max_bytes);
}

DlStatus dl_train_tape_payload_bytes(const DlTrainTape *t, size_t *out)
{
    if (t == NULL || out == NULL || t->model == NULL || t->arena.data == NULL)
        return DL_ARGUMENT;
    size_t total = 0, doubles = t->tiled_backward.count, bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(t->arena.count, sizeof(float), &total));
    if (t->tiled != NULL)
        for (size_t i = 0; i < t->model->config.layers; ++i)
            RETURN_IF_ERROR(dl_size_add(doubles, t->tiled[i].count, &doubles));
    RETURN_IF_ERROR(dl_size_mul(doubles, sizeof(double), &bytes));
    RETURN_IF_ERROR(dl_size_add(total, bytes, &total));
    *out = total;
    return DL_OK;
}

static DlTiledAttentionCache tiled_cache(DlAttentionCache a)
{
    return (DlTiledAttentionCache){a.q0, a.k0, a.v, a.q, a.k, a.context};
}

static DlTiledAttentionScratch tiled_scratch(DlAttentionScratch a)
{
    return (DlTiledAttentionScratch){a.context, a.q, a.k, a.v, a.q0, a.k0};
}

static DlAttentionWeights attention_weights(DlModel *m, size_t base)
{
    return (DlAttentionWeights){weight(m, base + DL_P_QUERY),
        weight(m, base + DL_P_KEY), weight(m, base + DL_P_VALUE),
        weight(m, base + DL_P_OUTPUT)};
}

static DlAttentionWeightGrads attention_grads(DlModel *m, size_t base)
{
    return (DlAttentionWeightGrads){dl_parameter_grad(m, base + DL_P_QUERY),
        dl_parameter_grad(m, base + DL_P_KEY), dl_parameter_grad(m, base + DL_P_VALUE),
        dl_parameter_grad(m, base + DL_P_OUTPUT)};
}

static DlFfnWeights ffn_weights(DlModel *m, size_t base)
{
    return (DlFfnWeights){weight(m, base + DL_P_GATE),
                          weight(m, base + DL_P_UP), weight(m, base + DL_P_DOWN)};
}

static DlFfnGrads ffn_grads(DlModel *m, size_t base)
{
    return (DlFfnGrads){dl_parameter_grad(m, base + DL_P_GATE),
        dl_parameter_grad(m, base + DL_P_UP), dl_parameter_grad(m, base + DL_P_DOWN)};
}

static DlStatus finite_parameters(DlModel *m, int gradients)
{
    for (size_t p = 0; p < m->param_count; ++p) {
        DlTensor *a = gradients ? &m->params[p].grad : &m->params[p].value;
        for (size_t i = 0; i < a->count; ++i)
            if (!isfinite(a->data[i])) return DL_NUMERIC;
    }
    return DL_OK;
}

DlStatus dl_model_mark_changed(DlModel *m)
{
    if (m == NULL || m->params == NULL) return DL_ARGUMENT;
    if (m->revision == UINT64_MAX) return DL_OVERFLOW;
    ++m->revision;
    m->gradients_valid = 0;
    return DL_OK;
}

DlStatus dl_model_zero_grad(DlModel *m)
{
    if (m == NULL || m->params == NULL) return DL_ARGUMENT;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].grad.count; ++i)
            m->params[p].grad.data[i] = 0.0f;
    m->gradients_valid = 1;
    return DL_OK;
}

DlStatus dl_model_forward(DlModel *m, DlTrainTape *t, DlBackend backend,
                          const size_t *ids, const size_t *targets,
                          size_t count, double *loss)
{
    if (t == NULL) return DL_ARGUMENT;
    t->ready = 0;
    t->embedding_grad = (DlMatrix){0};
    if (m == NULL || m->params == NULL || t->model != m ||
        t->arena.data == NULL || ids == NULL || targets == NULL || loss == NULL)
        return DL_ARGUMENT;
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED)
        return DL_ARGUMENT;
    if (count != t->rows) return DL_SHAPE;
    for (size_t i = 0; i < count; ++i)
        if (ids[i] >= m->config.vocab || targets[i] >= m->config.vocab)
            return DL_RANGE;
    RETURN_IF_ERROR(finite_parameters(m, 0));
    for (size_t i = 0; i < count; ++i) {
        t->ids[i] = ids[i];
        t->targets[i] = targets[i];
    }
    RETURN_IF_ERROR(dl_embedding_forward(weight(m, 0), t->ids, count, t->hidden[0]));
    for (size_t layer = 0; layer < m->config.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        DlBlockTape *b = &t->blocks[layer];
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(t->hidden[layer]),
            weight(m, base + DL_P_ATTN_NORM), m->config.epsilon, b->norm1));
        if (t->attention_mode == DL_ATTENTION_TILED) {
            RETURN_IF_ERROR(dl_attention_tiled_forward(backend, t->attention_config,
                dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
                count, t->inv_freq, t->attention_config.head_dim / 2,
                tiled_cache(b->attention), &t->tiled[layer], b->attention_out));
        } else {
        RETURN_IF_ERROR(dl_attention_forward(backend, t->attention_config,
            dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
            count, t->inv_freq, t->attention_config.head_dim / 2,
            b->attention, b->attention_out));
        }
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(t->hidden[layer]),
            dl_matrix_read(b->attention_out), b->middle));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(b->middle),
            weight(m, base + DL_P_FFN_NORM), m->config.epsilon, b->norm2));
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(b->norm2),
            ffn_weights(m, base), b->ffn, b->ffn_out));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(b->middle),
            dl_matrix_read(b->ffn_out), t->hidden[layer + 1]));
    }
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(t->hidden[m->config.layers]),
        weight(m, m->param_count - 1), m->config.epsilon, t->final_norm));
    RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(t->final_norm), DL_NOTRANS,
        weight(m, 0), DL_TRANS, t->logits, DL_SET));
    double candidate_loss = 0.0;
    RETURN_IF_ERROR(dl_cross_entropy_mean(dl_matrix_read(t->logits), t->targets,
        count, &candidate_loss, NULL, DL_SET));
    t->backend = backend;
    t->revision = m->revision;
    t->ready = 1;
    *loss = candidate_loss;
    return DL_OK;
}

DlStatus dl_model_backward(DlModel *m, DlTrainTape *t, DlWriteMode mode)
{
    if (m == NULL || m->params == NULL || t == NULL || t->model != m)
        return DL_ARGUMENT;
    if (mode != DL_SET && mode != DL_ADD) return DL_ARGUMENT;
    if (!t->ready || t->revision != m->revision) return DL_STATE;
    if (mode == DL_ADD) {
        if (!m->gradients_valid) return DL_STATE;
        RETURN_IF_ERROR(finite_parameters(m, 1));
    }
    t->ready = 0;                    /* Consume before any gradient writes. */
    t->embedding_grad = (DlMatrix){0};
    m->gradients_valid = 0;
    double ignored_loss = 0.0;
    RETURN_IF_ERROR(dl_cross_entropy_mean(dl_matrix_read(t->logits), t->targets,
        t->rows, &ignored_loss, &t->dlogits, DL_SET));
    RETURN_IF_ERROR(dl_matmul(t->backend, dl_matrix_read(t->dlogits), DL_TRANS,
        dl_matrix_read(t->final_norm), DL_NOTRANS, dl_parameter_grad(m, 0), mode));
    RETURN_IF_ERROR(dl_matmul(t->backend, dl_matrix_read(t->dlogits), DL_NOTRANS,
        weight(m, 0), DL_NOTRANS, t->other, DL_SET));
    RETURN_IF_ERROR(dl_rmsnorm_backward(dl_matrix_read(t->hidden[m->config.layers]),
        weight(m, m->param_count - 1), dl_matrix_read(t->other), m->config.epsilon,
        t->g, DL_SET, dl_parameter_grad(m, m->param_count - 1), mode));
    DlMatrix g = t->g, dx = t->other; /* Swap borrowed views, never owners. */
    for (size_t remaining = m->config.layers; remaining != 0; --remaining) {
        size_t layer = remaining - 1, base = 1 + layer * DL_BLOCK_PARAMS;
        DlBlockTape *b = &t->blocks[layer];
        RETURN_IF_ERROR(dl_residual_backward(dl_matrix_read(g), t->du, DL_SET));
        RETURN_IF_ERROR(dl_ffn_backward(t->backend, dl_matrix_read(b->norm2),
            ffn_weights(m, base), b->ffn, dl_matrix_read(g), t->ffn_scratch,
            t->dn, DL_SET, ffn_grads(m, base), mode));
        RETURN_IF_ERROR(dl_rmsnorm_backward(dl_matrix_read(b->middle),
            weight(m, base + DL_P_FFN_NORM), dl_matrix_read(t->dn), m->config.epsilon,
            t->du, DL_ADD, dl_parameter_grad(m, base + DL_P_FFN_NORM), mode));
        RETURN_IF_ERROR(dl_residual_backward(dl_matrix_read(t->du), dx, DL_SET));
        if (t->attention_mode == DL_ATTENTION_TILED) {
            RETURN_IF_ERROR(dl_attention_tiled_backward(t->backend, t->attention_config,
                dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
                t->rows, t->inv_freq, t->attention_config.head_dim / 2,
                tiled_cache(b->attention), &t->tiled[layer], dl_matrix_read(t->du),
                tiled_scratch(t->attention_scratch), &t->tiled_backward,
                t->dn, DL_SET, attention_grads(m, base), mode));
        } else {
        RETURN_IF_ERROR(dl_attention_backward(t->backend, t->attention_config,
            dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
            t->rows, t->inv_freq, t->attention_config.head_dim / 2, b->attention,
            dl_matrix_read(t->du), t->attention_scratch, t->dn, DL_SET,
            attention_grads(m, base), mode));
        }
        RETURN_IF_ERROR(dl_rmsnorm_backward(dl_matrix_read(t->hidden[layer]),
            weight(m, base + DL_P_ATTN_NORM), dl_matrix_read(t->dn), m->config.epsilon,
            dx, DL_ADD, dl_parameter_grad(m, base + DL_P_ATTN_NORM), mode));
        DlMatrix temporary = g;
        g = dx;
        dx = temporary;
    }
    RETURN_IF_ERROR(dl_embedding_backward(dl_matrix_read(g), t->ids, t->rows,
                                           dl_parameter_grad(m, 0), DL_ADD));
    t->embedding_grad = g;
    m->gradients_valid = 1;
    return DL_OK;
}

DlStatus dl_model_sgd(DlModel *m, double eta)
{
    if (m == NULL || m->params == NULL || !isfinite(eta) || eta <= 0.0)
        return DL_ARGUMENT;
    if (!m->gradients_valid) return DL_STATE;
    if (m->revision == UINT64_MAX) return DL_OVERFLOW;
    for (size_t p = 0; p < m->param_count; ++p) {
        DlParameter *a = &m->params[p];
        for (size_t i = 0; i < a->value.count; ++i) {
            double w = a->value.data[i], g = a->grad.data[i];
            double next = w - eta * g;
            if (!isfinite(w) || !isfinite(g) || !isfinite(next) || fabs(next) > FLT_MAX)
                return DL_NUMERIC;
        }
    }
    for (size_t p = 0; p < m->param_count; ++p) {
        DlParameter *a = &m->params[p];
        for (size_t i = 0; i < a->value.count; ++i)
            a->value.data[i] = (float)((double)a->value.data[i] - eta * a->grad.data[i]);
    }
    return dl_model_mark_changed(m);
}

#undef RETURN_IF_ERROR
