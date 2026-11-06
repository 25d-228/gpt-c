#include "dl/optim.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

static int config_valid(DlAdamWConfig c)
{
    return isfinite(c.beta1) && c.beta1 >= 0.0 && c.beta1 < 1.0 &&
           isfinite(c.beta2) && c.beta2 >= 0.0 && c.beta2 < 1.0 &&
           isfinite(c.epsilon) && c.epsilon > 0.0 &&
           isfinite(c.weight_decay) && c.weight_decay >= 0.0 &&
           isfinite(c.clip_norm) && c.clip_norm >= 0.0;
}

static DlStatus attached(const DlAdamW *o)
{
    if (o == NULL || o->model == NULL || o->model->params == NULL ||
        o->moments == NULL || o->sum == NULL || o->count == 0)
        return DL_ARGUMENT;
    if (o->revision != o->model->revision || o->count != o->model->value_count)
        return DL_STATE;
    return DL_OK;
}

void dl_adamw_destroy(DlAdamW *o)
{
    if (o == NULL) return;
    free(o->sum);
    free(o->moments);
    *o = (DlAdamW){0};
}

DlStatus dl_adamw_create(DlAdamW *out, DlModel *model, DlAdamWConfig config,
                         size_t max_state_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL ||
        model->value_count == 0 || !config_valid(config) || max_state_bytes == 0)
        return DL_ARGUMENT;
    if (out->model != NULL || out->moments != NULL || out->sum != NULL ||
        out->count != 0 || out->updates != 0 || out->tokens != 0 || out->active)
        return DL_STATE;
    size_t two_p = 0, moment_bytes = 0, sum_bytes = 0, total = 0;
    RETURN_IF_ERROR(dl_size_mul(model->value_count, 2, &two_p));
    RETURN_IF_ERROR(dl_size_mul(two_p, sizeof(float), &moment_bytes));
    RETURN_IF_ERROR(dl_size_mul(model->value_count, sizeof(double), &sum_bytes));
    RETURN_IF_ERROR(dl_size_add(moment_bytes, sum_bytes, &total));
    if (total > max_state_bytes) return DL_RANGE;
    DlAdamW candidate = {0};
    candidate.model = model;
    candidate.config = config;
    candidate.count = model->value_count;
    candidate.revision = model->revision;
    candidate.moments = calloc(two_p, sizeof(float));
    candidate.sum = calloc(candidate.count, sizeof(double));
    if (candidate.moments == NULL || candidate.sum == NULL) {
        dl_adamw_destroy(&candidate);
        return DL_ALLOC;
    }
    *out = candidate;
    return DL_OK;
}

DlStatus dl_adamw_begin(DlAdamW *o)
{
    RETURN_IF_ERROR(attached(o));
    RETURN_IF_ERROR(dl_model_zero_grad(o->model));
    for (size_t i = 0; i < o->count; ++i) o->sum[i] = 0.0;
    o->tokens = 0;
    o->active = 1;
    o->model->gradients_valid = 0; /* Cleared storage is not a fresh backward. */
    return DL_OK;
}

DlStatus dl_adamw_add_mean(DlAdamW *o, uint64_t tokens)
{
    RETURN_IF_ERROR(attached(o));
    if (!o->active || !o->model->gradients_valid) return DL_STATE;
    if (tokens == 0) return DL_ARGUMENT;
    if (o->tokens > DL_OPT_MAX_COUNT || tokens > DL_OPT_MAX_COUNT - o->tokens)
        return DL_OVERFLOW;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlTensor *g = &o->model->params[p].grad;
        for (size_t i = 0; i < g->count; ++i, ++at) {
            double next = o->sum[at] + (double)tokens * (double)g->data[i];
            if (!isfinite(g->data[i]) || !isfinite(o->sum[at]) || !isfinite(next))
                return DL_NUMERIC;
        }
    }
    at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlTensor *g = &o->model->params[p].grad;
        for (size_t i = 0; i < g->count; ++i, ++at)
            o->sum[at] += (double)tokens * (double)g->data[i];
    }
    o->tokens += tokens;
    o->model->gradients_valid = 0;
    return DL_OK;
}

/* Scaled sum of squares: norm^2 = scale^2 * ssq. No float square. */
static DlStatus mean_norm(const DlAdamW *o, double *out)
{
    double scale = 0.0, ssq = 1.0;
    for (size_t i = 0; i < o->count; ++i) {
        double a = fabs(o->sum[i] / (double)o->tokens);
        if (!isfinite(a)) return DL_NUMERIC;
        if (a == 0.0) continue;
        if (scale < a) {
            double ratio = scale / a;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = a;
        } else {
            double ratio = a / scale;
            ssq += ratio * ratio;
        }
    }
    double norm = scale * sqrt(ssq);
    if (!isfinite(norm)) return DL_NUMERIC;
    *out = norm;
    return DL_OK;
}

/* Unlike 1-pow(beta,t), this does not subtract two nearly equal numbers. */
static double correction(double beta, uint64_t update)
{
    return beta == 0.0 ? 1.0 : -expm1((double)update * log(beta));
}

typedef struct {
    float weight, first, second;
} Candidate;

static DlStatus candidate_at(const DlAdamW *o, size_t at, double weight,
                              int decay, double lr, double clip,
                              double correction1, double correction2,
                              Candidate *out)
{
    double old_m = o->moments[at];
    double old_v = o->moments[o->count + at];
    double g = (o->sum[at] / (double)o->tokens) * clip;
    if (!isfinite(weight) || !isfinite(old_m) || !isfinite(old_v) ||
        old_v < 0.0 || !isfinite(g)) return DL_NUMERIC;
    double m = o->config.beta1 * old_m + (1.0 - o->config.beta1) * g;
    double v = o->config.beta2 * old_v + (1.0 - o->config.beta2) * g * g;
    if (!isfinite(m) || fabs(m) > FLT_MAX || !isfinite(v) || v > FLT_MAX)
        return DL_NUMERIC;
    /* Bias-correct the moments that will actually be stored, not extra bits. */
    float stored_m = (float)m, stored_v = (float)v;
    double mhat = (double)stored_m / correction1;
    double vhat = (double)stored_v / correction2;
    double adaptive = mhat / (sqrt(vhat) + o->config.epsilon);
    double shrink = decay ? 1.0 - lr * o->config.weight_decay : 1.0;
    double next = weight * shrink - lr * adaptive;
    if (!isfinite(mhat) || !isfinite(vhat) || !isfinite(adaptive) ||
        !isfinite(next) || fabs(next) > FLT_MAX) return DL_NUMERIC;
    *out = (Candidate){(float)next, stored_m, stored_v};
    return DL_OK;
}

DlStatus dl_adamw_step(DlAdamW *o, double learning_rate, DlOptimStats *stats)
{
    RETURN_IF_ERROR(attached(o));
    if (stats == NULL || !isfinite(learning_rate) || learning_rate <= 0.0)
        return DL_ARGUMENT;
    if (!o->active || o->tokens == 0 || o->model->gradients_valid)
        return DL_STATE;
    if (o->updates >= DL_OPT_MAX_COUNT || o->tokens > DL_OPT_MAX_COUNT ||
        o->model->revision == UINT64_MAX) return DL_OVERFLOW;
    double decay_step = learning_rate * o->config.weight_decay;
    if (!isfinite(decay_step) || decay_step > 1.0) return DL_RANGE;
    uint64_t update = o->updates + 1;
    double c1 = correction(o->config.beta1, update);
    double c2 = correction(o->config.beta2, update);
    if (!isfinite(c1) || c1 <= 0.0 || !isfinite(c2) || c2 <= 0.0)
        return DL_NUMERIC;
    double norm = 0.0;
    RETURN_IF_ERROR(mean_norm(o, &norm));
    double clip = 1.0;
    if (o->config.clip_norm > 0.0 && norm > o->config.clip_norm)
        clip = o->config.clip_norm / norm;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlParameter *a = &o->model->params[p];
        for (size_t i = 0; i < a->value.count; ++i, ++at) {
            Candidate trial;
            RETURN_IF_ERROR(candidate_at(o, at, a->value.data[i], a->decay,
                learning_rate, clip, c1, c2, &trial));
        }
    }
    /* Single-threaded: repeating unchanged inputs cannot acquire a new error.
     * Recompute instead of allocating a second model-sized candidate buffer. */
    RETURN_IF_ERROR(dl_model_mark_changed(o->model));
    at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        DlParameter *a = &o->model->params[p];
        for (size_t i = 0; i < a->value.count; ++i, ++at) {
            Candidate next = {0};
            (void)candidate_at(o, at, a->value.data[i], a->decay,
                              learning_rate, clip, c1, c2, &next);
            a->value.data[i] = next.weight;
            o->moments[at] = next.first;
            o->moments[o->count + at] = next.second;
        }
    }
    *stats = (DlOptimStats){update, o->tokens, learning_rate, norm, clip};
    o->updates = update;
    o->revision = o->model->revision;
    o->tokens = 0;
    o->active = 0;
    return DL_OK;
}

DlStatus dl_adamw_abort(DlAdamW *o)
{
    RETURN_IF_ERROR(dl_adamw_begin(o));
    o->active = 0;
    return DL_OK;
}

DlStatus dl_adamw_restore(DlAdamW *o, const float *weights,
                          const float *first, const float *second,
                          size_t count, uint64_t updates)
{
    RETURN_IF_ERROR(attached(o));
    if (weights == NULL || first == NULL || second == NULL) return DL_ARGUMENT;
    if (count != o->count) return DL_SHAPE;
    if (o->active || o->tokens != 0 || o->model->gradients_valid) return DL_STATE;
    if (updates > DL_OPT_MAX_COUNT) return DL_RANGE;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(weights[i]) || !isfinite(first[i]) ||
            !isfinite(second[i]) || second[i] < 0.0f) return DL_NUMERIC;
    /* This is the last operation that can fail. Keep owners in their places. */
    RETURN_IF_ERROR(dl_model_mark_changed(o->model));
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        DlParameter *parameter = &o->model->params[p];
        for (size_t i = 0; i < parameter->value.count; ++i, ++at) {
            parameter->value.data[i] = weights[at];
            parameter->grad.data[i] = 0.0f;
            o->moments[at] = first[at];
            o->moments[count + at] = second[at];
            o->sum[at] = 0.0;
        }
    }
    o->updates = updates;
    o->revision = o->model->revision;
    return DL_OK;
}

#undef RETURN_IF_ERROR
