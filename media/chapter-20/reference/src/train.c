#include "dl/train.h"

#include <math.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

DlStatus dl_schedule_rate(DlSchedule s, uint64_t completed, double *out)
{
    if (out == NULL || !isfinite(s.peak) || !isfinite(s.floor) ||
        s.floor <= 0.0 || s.peak < s.floor || s.total == 0 ||
        s.total > DL_OPT_MAX_COUNT || s.warmup >= s.total) return DL_ARGUMENT;
    if (completed >= s.total) return DL_RANGE;
    uint64_t next = completed + 1;
    uint64_t anchor = s.warmup == 0 ? 1 : s.warmup;
    double rate = s.peak;
    if (next <= anchor) {
        if (s.warmup != 0) rate *= (double)next / (double)s.warmup;
    } else {
        double remaining = (double)(s.total - next) /
                           (double)(s.total - anchor);
        rate = s.floor + (s.peak - s.floor) * remaining;
    }
    if (!isfinite(rate) || rate <= 0.0) return DL_NUMERIC;
    *out = rate;
    return DL_OK;
}

static DlStatus state_check(const DlAdamW *o, const DlTrainState *s,
                             const DlByteData *d, uint64_t *group)
{
    if (o == NULL || s == NULL || d == NULL || group == NULL ||
        o->model == NULL || o->model->params == NULL || o->moments == NULL ||
        o->sum == NULL || d->bytes == NULL) return DL_ARGUMENT;
    if (o->model->config.vocab != 256 || s->batch == 0 || s->time == 0 ||
        s->microbatches == 0 || (s->fixed != 0 && s->fixed != 1) ||
        (s->backend != DL_BACKEND_REF && s->backend != DL_BACKEND_BLOCKED))
        return DL_ARGUMENT;
    if ((s->attention != DL_ATTENTION_MATERIALIZED && s->attention != DL_ATTENTION_TILED) ||
        (s->attention == DL_ATTENTION_MATERIALIZED && (s->query_tile != 0 || s->key_tile != 0)) ||
        (s->attention == DL_ATTENTION_TILED && (s->query_tile == 0 || s->key_tile == 0)))
        return DL_ARGUMENT;
    if (o->revision != o->model->revision || o->count != o->model->value_count ||
        o->active || o->tokens != 0 || o->model->gradients_valid) return DL_STATE;
    double ignored = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate(s->schedule, 0, &ignored));
    size_t rows = 0, targets = 0;
    RETURN_IF_ERROR(dl_size_mul(s->batch, s->time, &rows));
    RETURN_IF_ERROR(dl_size_mul(rows, s->microbatches, &targets));
    if (targets > DL_OPT_MAX_COUNT || d->count > d->capacity ||
        d->count <= rows || s->time - 1 > DL_ROPE_MAX_POSITION)
        return DL_RANGE;
    if (o->updates > s->schedule.total ||
        o->updates > UINT64_MAX / (uint64_t)targets) return DL_RANGE;
    if (s->tokens_seen != o->updates * (uint64_t)targets) return DL_STATE;
    *group = (uint64_t)targets;
    return DL_OK;
}

DlStatus dl_train_boundary(const DlAdamW *o, const DlTrainState *s,
                           const DlByteData *training, const DlByteData *valid)
{
    uint64_t group = 0;
    RETURN_IF_ERROR(state_check(o, s, training, &group));
    return state_check(o, s, valid, &group);
}

DlStatus dl_train_update(DlAdamW *o, DlTrainState *s, const DlByteData *data,
                         DlByteBatch *batch, DlTrainTape *tape, DlTrainStats *stats)
{
    if (batch == NULL || tape == NULL || stats == NULL) return DL_ARGUMENT;
    uint64_t group = 0;
    RETURN_IF_ERROR(state_check(o, s, data, &group));
    if (batch->storage == NULL || batch->batch != s->batch ||
        batch->time != s->time || tape->model != o->model ||
        tape->arena.data == NULL || tape->rows != batch->count ||
        tape->attention_config.batch != s->batch ||
        tape->attention_config.time != s->time) return DL_SHAPE;
    if (tape->attention_mode != s->attention) return DL_SHAPE;
    if (s->attention == DL_ATTENTION_TILED &&
        (tape->tiled == NULL || tape->tiled[0].query_tile != s->query_tile ||
         tape->tiled[0].key_tile != s->key_tile)) return DL_SHAPE;
    if (s->tokens_seen > UINT64_MAX - group) return DL_OVERFLOW;
    double rate = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate(s->schedule, o->updates, &rate));
    DlRng next_rng = s->rng; /* Speculative stream; commit only after step. */
    DlStatus status = dl_adamw_begin(o);
    if (status != DL_OK) return status;
    double weighted_loss = 0.0;
    for (size_t i = 0; i < s->microbatches; ++i) {
        if (s->fixed) {
            size_t cursor = 0;
            int has = 0;
            status = dl_byte_batch_next(data, batch, &cursor, &has);
            if (status == DL_OK && !has) status = DL_RANGE;
        } else status = dl_byte_batch_sample(data, batch, &next_rng);
        double loss = 0.0;
        if (status == DL_OK)
            status = dl_model_forward(o->model, tape, s->backend,
                                      batch->ids, batch->targets, batch->count, &loss);
        if (status == DL_OK) status = dl_model_backward(o->model, tape, DL_SET);
        if (status == DL_OK) status = dl_adamw_add_mean(o, (uint64_t)batch->count);
        if (status != DL_OK) break;
        weighted_loss += loss * (double)batch->count;
        if (!isfinite(weighted_loss)) { status = DL_NUMERIC; break; }
    }
    DlTrainStats result = {.loss = weighted_loss / (double)group};
    if (status == DL_OK) status = dl_adamw_step(o, rate, &result.optim);
    if (status != DL_OK) {
        tape->ready = 0;
        /* No update has committed; abort cannot fail for this attached owner. */
        DlStatus cleanup = dl_adamw_abort(o);
        return cleanup == DL_OK ? status : cleanup;
    }
    s->rng = next_rng;
    s->tokens_seen += group;
    *stats = result;
    return DL_OK;
}

DlStatus dl_evaluate(DlModel *model, DlTrainTape *tape, DlBackend backend,
                      const DlByteData *data, DlByteBatch *batch,
                      size_t limit, DlEvalStats *out)
{
    if (model == NULL || tape == NULL || data == NULL || batch == NULL ||
        out == NULL || model->params == NULL || batch->storage == NULL)
        return DL_ARGUMENT;
    if (model->config.vocab != 256 || tape->model != model ||
        tape->rows != batch->count || tape->attention_config.batch != batch->batch ||
        tape->attention_config.time != batch->time) return DL_SHAPE;
    DlEvalStats result = {0};
    size_t cursor = 0;
    double total_loss = 0.0;
    for (;;) {
        if (limit != 0 && result.batches == limit) break;
        int has = 0;
        RETURN_IF_ERROR(dl_byte_batch_next(data, batch, &cursor, &has));
        if (!has) break;
        double loss = 0.0;
        RETURN_IF_ERROR(dl_model_forward(model, tape, backend, batch->ids,
                                         batch->targets, batch->count, &loss));
        total_loss += (double)batch->count * loss;
        if (!isfinite(total_loss)) return DL_NUMERIC;
        ++result.batches;
    }
    if (cursor == 0) return DL_RANGE;
    result.tokens = (uint64_t)cursor;
    result.unscored = data->count - 1 - cursor;
    result.loss = total_loss / (double)cursor;
    *out = result;
    return DL_OK;
}

#undef RETURN_IF_ERROR
