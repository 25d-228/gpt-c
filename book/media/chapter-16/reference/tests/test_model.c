#include "dl/model.h"

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t coordinate_checks = 0, direction_checks = 0;
static double coordinate_error = 0.0, direction_error = 0.0;

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

static int close_value(double a, double b, double absolute, double relative,
                        const char *where)
{
    double tolerance = absolute + relative * fmax(fabs(a), fabs(b));
    if (isfinite(a) && isfinite(b) && fabs(a - b) <= tolerance) return 1;
    fprintf(stderr, "%s: %.10g versus %.10g (tolerance %.3g)\n", where, a, b, tolerance);
    return 0;
}

#define OK(call) status_is((call), DL_OK, #call)

typedef struct {
    DlModel model;
    DlTrainTape tape;
    size_t ids[8], targets[8];
} Fixture;

static void fixture_destroy(Fixture *f)
{
    dl_train_tape_destroy(&f->tape);
    dl_model_destroy(&f->model);
    *f = (Fixture){0};
}

/* Bounds keep this TEST executable small; they are not model-library limits. */
static int fixture_create(Fixture *f, DlModelConfig config, size_t batch, size_t time)
{
    if (batch == 0 || batch > 2 || time < 2 || time > 4 ||
        config.vocab < 4 || config.vocab > 16 || config.width > 16 ||
        config.ffn_width > 40 || config.layers > 2) return 0;
    if (!OK(dl_model_create(&f->model, config))) goto fail;
    if (!OK(dl_train_tape_create(&f->tape, &f->model, batch, time,
                                 64 * 1024 * 1024))) goto fail;
    const size_t sequence[] = {1, 2, 1, 3, 2};
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < time; ++t) {
            f->ids[b * time + t] = sequence[t];
            f->targets[b * time + t] = sequence[t + 1];
        }
    }
    return 1;
fail:
    fixture_destroy(f);
    return 0;
}

static int evaluate(Fixture *f, DlBackend backend, double *loss)
{
    return OK(dl_model_forward(&f->model, &f->tape, backend, f->ids,
                               f->targets, f->tape.rows, loss));
}

static void flatten(DlModel *m, int gradients, float *out)
{
    size_t at = 0;
    for (size_t p = 0; p < m->param_count; ++p) {
        DlTensor *a = gradients ? &m->params[p].grad : &m->params[p].value;
        for (size_t i = 0; i < a->count; ++i) out[at++] = a->data[i];
    }
}

static int restore(DlModel *m, const float *values)
{
    if (!OK(dl_model_mark_changed(m))) return 0;
    size_t at = 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].value.count; ++i)
            m->params[p].value.data[i] = values[at++];
    return 1;
}

static int rng_and_ownership(DlModelConfig c)
{
    DlRng rng = {0};
    if (!require(dl_rng_next(&rng) == UINT64_C(0xe220a8397b1dcdaf), "RNG first word") ||
        !require(dl_rng_next(&rng) == UINT64_C(0x6e789e6aa1b965f4), "RNG second word"))
        return 0;
    for (size_t i = 0; i < 1000; ++i) {
        double u = dl_rng_uniform(&rng);
        if (!require(u >= 0.0 && u < 1.0, "uniform half-open interval")) return 0;
    }
    DlModel a = {0}, b = {0}, empty = {0};
    DlTrainTape rejected = {0};
    int ok = 0;
    if (!OK(dl_model_create(&a, c)) || !OK(dl_model_create(&b, c))) goto done;
    size_t total = 0, scales = 0;
    for (size_t p = 0; p < a.param_count; ++p) {
        DlParameter *x = &a.params[p], *y = &b.params[p];
        total += x->value.count;
        scales += !x->decay;
        if (!require(strcmp(x->name, y->name) == 0, "stable parameter names") ||
            !require(memcmp(x->value.data, y->value.data,
                            x->value.count * sizeof(float)) == 0, "same-seed values") ||
            !require(x->value.data != x->grad.data, "value/gradient ownership")) goto done;
        double limit = sqrt(3.0) * c.init_std;
        if (p > 0 && p + 1 < a.param_count &&
            ((p - 1) % DL_BLOCK_PARAMS == DL_P_OUTPUT ||
             (p - 1) % DL_BLOCK_PARAMS == DL_P_DOWN)) limit /= sqrt(2.0 * (double)c.layers);
        for (size_t i = 0; i < x->value.count; ++i) {
            if (!require(x->grad.data[i] == 0.0f, "initialized gradient") ||
                !require(x->decay ? fabs(x->value.data[i]) <= limit + 1e-7
                                 : x->value.data[i] == 1.0f, "initialization rule")) goto done;
        }
        for (size_t q = 0; q < p; ++q)
            if (!require(x->value.data != a.params[q].value.data &&
                         strcmp(x->name, a.params[q].name) != 0, "unique directory")) goto done;
    }
    if (!require(total == a.value_count && scales == 2 * c.layers + 1,
                 "parameter counts") ||
        !status_is(dl_model_create(&a, c), DL_STATE, "nonempty model") ||
        !status_is(dl_train_tape_create(&rejected, &a, 1, 4, 1), DL_RANGE,
                   "arena budget after descriptor allocation") ||
        !require(rejected.model == NULL && rejected.hidden == NULL, "failed tape unchanged"))
        goto done;
    DlModelConfig bad = c;
    bad.width = SIZE_MAX;
    size_t n = 71, v = 72;
    if (!require(dl_model_config_validate(bad, &n, &v) != DL_OK && n == 71 && v == 72,
                 "invalid config count outputs unchanged")) goto done;
    bad = c; bad.layers = SIZE_MAX;
    if (!status_is(dl_model_config_validate(bad, &n, &v), DL_OVERFLOW, "count overflow"))
        goto done;
    bad = c; bad.init_std = FLT_MAX;
    if (!status_is(dl_model_create(&empty, bad), DL_RANGE, "partial parameter construction") ||
        !require(empty.params == NULL && empty.param_count == 0, "failed model unchanged"))
        goto done;
    dl_model_destroy(&b);
    ++c.seed;
    if (!OK(dl_model_create(&b, c)) ||
        !require(memcmp(a.params[0].value.data, b.params[0].value.data,
                        a.params[0].value.count * sizeof(float)) != 0, "different seed"))
        goto done;
    ok = 1;
done:
    dl_train_tape_destroy(&rejected);
    dl_model_destroy(&empty);
    dl_model_destroy(&b);
    dl_model_destroy(&a);
    dl_model_destroy(&a);
    return ok;
}

static int tied_gradient(Fixture *f)
{
    DlTrainTape *t = &f->tape;
    DlModel *m = &f->model;
    const size_t d = m->config.width, v = m->config.vocab;
    for (size_t token = 0; token < v; ++token) {
        for (size_t j = 0; j < d; ++j) {
            double output_path = 0.0;
            for (size_t row = 0; row < t->rows; ++row)
                output_path += (double)t->dlogits.base[t->dlogits.offset + row * v + token]
                    * t->final_norm.base[t->final_norm.offset + row * d + j];
            float expected = (float)output_path;
            for (size_t row = 0; row < t->rows; ++row) {
                if (t->ids[row] == token)
                    expected = (float)((double)expected + t->embedding_grad.base[
                        t->embedding_grad.offset + row * d + j]);
            }
            if (!close_value(m->params[0].grad.data[token * d + j], expected,
                             2e-7, 2e-6, "two tied-table paths")) return 0;
        }
    }
    return 1;
}

static int coordinate_derivatives(Fixture *f, DlBackend backend)
{
    DlModel *m = &f->model;
    double loss = 0.0;
    if (!evaluate(f, backend, &loss) || !OK(dl_model_backward(m, &f->tape, DL_SET)) ||
        !tied_gradient(f)) return 0;
    /* Forward/mark_changed never overwrite parameter gradient bytes. */
    const float steps[] = {0.004f, 0.002f, 0.001f};
    for (size_t p = 0; p < m->param_count; ++p) {
        DlParameter *a = &m->params[p];
        size_t strongest = 0;
        for (size_t i = 1; i < a->grad.count; ++i)
            if (fabsf(a->grad.data[i]) > fabsf(a->grad.data[strongest])) strongest = i;
        if (!require(fabsf(a->grad.data[strongest]) > 1e-7f,
                     "fixture must exercise EVERY parameter family")) return 0;
        size_t selected[] = {strongest, (strongest + a->value.count / 2) % a->value.count,
                             (strongest + a->value.count - 1) % a->value.count};
        for (size_t which = 0; which < 3; ++which) {
            size_t i = selected[which];
            const float original = a->value.data[i];
            const double analytic = a->grad.data[i];
            for (size_t s = 0; s < 3; ++s) {
                float plus = original + steps[s], minus = original - steps[s];
                if (!OK(dl_model_mark_changed(m))) return 0;
                a->value.data[i] = plus;
                double high = 0.0, low = 0.0;
                if (!evaluate(f, backend, &high) || !OK(dl_model_mark_changed(m))) return 0;
                a->value.data[i] = minus;
                if (!evaluate(f, backend, &low) || !OK(dl_model_mark_changed(m))) return 0;
                a->value.data[i] = original;
                double numeric = (high - low) / ((double)plus - minus);
                coordinate_error = fmax(coordinate_error, fabs(numeric - analytic));
                ++coordinate_checks;
                if (!close_value(numeric, analytic, 3e-5, 0.015, a->name)) return 0;
            }
        }
    }
    return 1;
}

static int directional_derivatives(Fixture *f, DlBackend backend,
                                    float *original, float *analytic, float *direction)
{
    DlModel *m = &f->model;
    double loss = 0.0;
    if (!evaluate(f, backend, &loss) || !OK(dl_model_backward(m, &f->tape, DL_SET))) return 0;
    flatten(m, 0, original);
    flatten(m, 1, analytic);
    DlRng rng = {123};
    const double steps[] = {0.01, 0.005, 0.0025};
    for (size_t trial = 0; trial < 3; ++trial) {
        double norm2 = 0.0;
        for (size_t i = 0; i < m->value_count; ++i) {
            direction[i] = (float)(2.0 * dl_rng_uniform(&rng) - 1.0);
            norm2 += (double)direction[i] * direction[i];
        }
        double norm = sqrt(norm2);
        for (size_t i = 0; i < m->value_count; ++i)
            direction[i] = (float)(direction[i] / norm);
        for (size_t s = 0; s < 3; ++s) {
            double high = 0.0, low = 0.0, projected = 0.0;
            for (size_t side = 0; side < 2; ++side) {
                if (!OK(dl_model_mark_changed(m))) return 0;
                size_t at = 0;
                for (size_t p = 0; p < m->param_count; ++p) {
                    for (size_t i = 0; i < m->params[p].value.count; ++i, ++at) {
                        float plus = (float)(original[at] + steps[s] * direction[at]);
                        float minus = (float)(original[at] - steps[s] * direction[at]);
                        m->params[p].value.data[i] = side == 0 ? plus : minus;
                        if (side == 0)
                            projected += analytic[at] * ((double)plus - minus) / (2.0 * steps[s]);
                    }
                }
                if (!evaluate(f, backend, side == 0 ? &high : &low)) return 0;
            }
            if (!restore(m, original)) return 0;
            double numeric = (high - low) / (2.0 * steps[s]);
            direction_error = fmax(direction_error, fabs(numeric - projected));
            ++direction_checks;
            if (!close_value(numeric, projected, 2e-5, 0.01, "whole-model direction")) return 0;
        }
    }
    return 1;
}

static int state_accumulation_update(Fixture *f, float *original, float *analytic)
{
    DlModel *m = &f->model;
    DlTrainTape *t = &f->tape;
    double before = 0.0, after = 0.0;
    if (!evaluate(f, DL_BACKEND_REF, &before)) return 0;
    /* Forward owns a snapshot of IDs and targets, not caller references. */
    size_t id = f->ids[0], target = f->targets[0];
    f->ids[0] = m->config.vocab;
    f->targets[0] = m->config.vocab;
    if (!OK(dl_model_backward(m, t, DL_SET))) return 0;
    f->ids[0] = id; f->targets[0] = target;
    flatten(m, 0, original); flatten(m, 1, analytic);
    if (!status_is(dl_model_backward(m, t, DL_SET), DL_STATE, "consumed tape")) return 0;
    if (!evaluate(f, DL_BACKEND_REF, &after) || !OK(dl_model_backward(m, t, DL_ADD))) return 0;
    size_t at = 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].grad.count; ++i, ++at)
            if (!close_value(m->params[p].grad.data[i], 2.0 * analytic[at],
                             3e-6, 3e-6, "ADD equals two batch means")) return 0;
    if (!evaluate(f, DL_BACKEND_BLOCKED, &after) || !OK(dl_model_backward(m, t, DL_SET))) return 0;
    at = 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].grad.count; ++i, ++at)
            if (!close_value(m->params[p].grad.data[i], analytic[at],
                             3e-6, 3e-6, "blocked gradients and SET reset")) return 0;
    if (!close_value(before, after, 1e-7, 1e-7, "backend loss") ||
        !evaluate(f, DL_BACKEND_REF, &after) || !OK(dl_model_mark_changed(m)) ||
        !status_is(dl_model_backward(m, t, DL_SET), DL_STATE, "stale tape") ||
        !evaluate(f, DL_BACKEND_REF, &after) ||
        !status_is(dl_model_backward(m, t, DL_ADD), DL_STATE, "invalid ADD boundary")) return 0;
    if (!OK(dl_model_zero_grad(m)) || !OK(dl_model_backward(m, t, DL_ADD))) return 0;

    /* A late NaN must not allow an earlier weight to be updated. */
    DlTensor *last = &m->params[m->param_count - 1].grad;
    float saved = last->data[last->count - 1];
    last->data[last->count - 1] = NAN;
    if (!status_is(dl_model_sgd(m, 0.001), DL_NUMERIC, "preflight whole update")) return 0;
    at = 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].value.count; ++i, ++at)
            if (!require(m->params[p].value.data[i] == original[at], "failed update unchanged"))
                return 0;
    last->data[last->count - 1] = saved;
    flatten(m, 1, analytic);
    const double eta = 0.001;
    if (!evaluate(f, DL_BACKEND_REF, &before) || !OK(dl_model_sgd(m, eta)) ||
        !status_is(dl_model_backward(m, t, DL_SET), DL_STATE, "update invalidates tape") ||
        !status_is(dl_model_sgd(m, eta), DL_STATE, "no accidental second update")) return 0;
    at = 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].value.count; ++i, ++at) {
            float expected = (float)((double)original[at] - eta * analytic[at]);
            if (!require(m->params[p].value.data[i] == expected, "each unique parameter updated once"))
                return 0;
        }
    if (!evaluate(f, DL_BACKEND_REF, &after) || !require(after < before, "descent smoke check"))
        return 0;
    printf("  plain step: %.9f -> %.9f\n", before, after);

    if (!restore(m, original) || !evaluate(f, DL_BACKEND_REF, &before)) return 0;
    f->targets[0] = m->config.vocab;
    double sentinel = 123.0;
    if (!status_is(dl_model_forward(m, t, DL_BACKEND_REF, f->ids, f->targets,
                                    t->rows, &sentinel), DL_RANGE, "invalid target") ||
        !require(sentinel == 123.0 && !t->ready, "failed forward invalidates tape")) return 0;
    f->targets[0] = target;
    if (!evaluate(f, DL_BACKEND_REF, &before)) return 0;
    /* Deliberate test-only corruption of saved state forces failed backward. */
    size_t where = t->blocks[0].ffn.gate.offset;
    t->arena.data[where] = NAN;
    if (!status_is(dl_model_backward(m, t, DL_SET), DL_NUMERIC, "failed backward") ||
        !require(!m->gradients_valid && !t->ready, "failed backward state") ||
        !status_is(dl_model_sgd(m, eta), DL_STATE, "refuse damaged gradient") ||
        !evaluate(f, DL_BACKEND_REF, &before) || !OK(dl_model_backward(m, t, DL_SET))) return 0;
    return 1;
}

static int dependency_and_zero(Fixture *f, float *original)
{
    DlModel *m = &f->model;
    DlTrainTape *t = &f->tape;
    double loss = 0.0;
    float saved_logits[8 * 16];
    if (!evaluate(f, DL_BACKEND_REF, &loss)) return 0;
    memcpy(saved_logits, t->arena.data + t->logits.offset,
           t->rows * m->config.vocab * sizeof(float));
    size_t index = t->attention_config.time - 1, old = f->ids[index];
    f->ids[index] = (old + 1) % m->config.vocab;
    if (!evaluate(f, DL_BACKEND_REF, &loss)) return 0;
    for (size_t row = 0; row < t->rows; ++row) {
        if (row == index) continue;
        for (size_t j = 0; j < m->config.vocab; ++j)
            if (!require(t->arena.data[t->logits.offset + row * m->config.vocab + j] ==
                         saved_logits[row * m->config.vocab + j],
                         "future-token and other-batch isolation")) return 0;
    }
    f->ids[index] = old;
    flatten(m, 0, original);
    if (!OK(dl_model_mark_changed(m))) return 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].value.count; ++i)
            m->params[p].value.data[i] = 0.0f;
    if (!evaluate(f, DL_BACKEND_REF, &loss) ||
        !close_value(loss, log((double)m->config.vocab), 1e-12, 1e-12, "zero model loss") ||
        !OK(dl_model_backward(m, t, DL_SET))) return 0;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].grad.count; ++i)
            if (!require(m->params[p].grad.data[i] == 0.0f, "zero model gradient")) return 0;
    return restore(m, original);
}

static int hand_forward(void)
{
    DlModelConfig c = {4, 1, 2, 1, 1, 2, 0.5f, 0.1f, 10000.0, 0};
    Fixture f = {0};
    int ok = 0;
    if (!fixture_create(&f, c, 1, 2) || !OK(dl_model_mark_changed(&f.model))) goto done;
    const float table[8] = {1, 0, 0, 2, 1, 1, -1, 0};
    for (size_t p = 0; p < f.model.param_count; ++p) {
        DlParameter *a = &f.model.params[p];
        for (size_t i = 0; i < a->value.count; ++i)
            a->value.data[i] = a->decay ? 0.0f : 1.0f;
    }
    memcpy(f.model.params[0].value.data, table, sizeof(table));
    const size_t identity[] = {DL_P_VALUE, DL_P_OUTPUT, DL_P_GATE, DL_P_UP, DL_P_DOWN};
    for (size_t p = 0; p < sizeof(identity) / sizeof(identity[0]); ++p) {
        float *a = f.model.params[1 + identity[p]].value.data;
        a[0] = 1.0f; a[3] = 1.0f;
    }
    f.model.params[f.model.param_count - 1].value.data[1] = 2.0f;
    f.ids[0] = 0; f.ids[1] = 1; f.targets[0] = 1; f.targets[1] = 2;
    double loss = 0.0;
    if (!evaluate(&f, DL_BACKEND_REF, &loss)) goto done;
    /* Q=K=0 makes causal probabilities [1,0] and [1/2,1/2].
     * Independent two-coordinate arithmetic: NO model/operator calls here. */
    float n1[2][2], u[2][2], y[2][2], z[2][2], expected[2][4];
    for (size_t row = 0; row < 2; ++row) {
        double x0 = table[2 * row], x1 = table[2 * row + 1];
        double r = 1.0 / sqrt((x0 * x0 + x1 * x1) / 2.0 + 0.5);
        n1[row][0] = (float)(x0 * r); n1[row][1] = (float)(x1 * r);
        for (size_t j = 0; j < 2; ++j) {
            float context = row == 0 ? n1[0][j]
                : (float)(((double)n1[0][j] + n1[1][j]) / 2.0);
            u[row][j] = (float)((double)table[2 * row + j] + context);
        }
        r = 1.0 / sqrt(((double)u[row][0] * u[row][0] +
                       (double)u[row][1] * u[row][1]) / 2.0 + 0.5);
        for (size_t j = 0; j < 2; ++j) {
            float n2 = (float)(u[row][j] * r);
            float hidden = (float)((double)n2 * n2 / (1.0 + exp(-(double)n2)));
            y[row][j] = (float)((double)u[row][j] + hidden);
        }
        r = 1.0 / sqrt(((double)y[row][0] * y[row][0] +
                       (double)y[row][1] * y[row][1]) / 2.0 + 0.5);
        z[row][0] = (float)(y[row][0] * r);
        z[row][1] = (float)(2.0 * y[row][1] * r);
        for (size_t token = 0; token < 4; ++token) {
            expected[row][token] = (float)((double)z[row][0] * table[2 * token]
                                + (double)z[row][1] * table[2 * token + 1]);
            if (!close_value(f.tape.arena.data[f.tape.logits.offset + row * 4 + token],
                             expected[row][token], 3e-6, 2e-6, "hand-scheduled logits")) goto done;
        }
    }
    double expected_loss = 0.0;
    for (size_t row = 0; row < 2; ++row) {
        double maximum = expected[row][0], sum = 0.0;
        for (size_t j = 1; j < 4; ++j) maximum = fmax(maximum, expected[row][j]);
        for (size_t j = 0; j < 4; ++j) sum += exp(expected[row][j] - maximum);
        expected_loss += (maximum - expected[row][f.targets[row]] + log(sum)) / 2.0;
    }
    if (!close_value(loss, expected_loss, 3e-6, 2e-6, "hand-scheduled loss")) goto done;
    printf("hand logits row 1: [%.6f, %.6f, %.6f, %.6f], loss=%.9f\n",
           expected[1][0], expected[1][1], expected[1][2], expected[1][3], loss);
    ok = 1;
done:
    fixture_destroy(&f);
    return ok;
}

static int run_case(DlModelConfig c, size_t batch, size_t time)
{
    Fixture f = {0};
    DlTensor vectors[3] = {{0}};
    int ok = 0;
    if (!fixture_create(&f, c, batch, time)) goto done;
    for (size_t i = 0; i < 3; ++i)
        if (!OK(dl_tensor_create(&vectors[i], 1, &f.model.value_count))) goto done;
    printf("model: L=%zu B=%zu T=%zu, %zu tensors, %zu values, %zu tape floats\n",
           c.layers, batch, time, f.model.param_count, f.model.value_count, f.tape.arena.count);
    for (size_t backend = 0; backend < 2; ++backend) {
        if (!coordinate_derivatives(&f, (DlBackend)backend) ||
            !directional_derivatives(&f, (DlBackend)backend, vectors[0].data,
                                      vectors[1].data, vectors[2].data)) goto done;
    }
    if (!state_accumulation_update(&f, vectors[0].data, vectors[1].data) ||
        !dependency_and_zero(&f, vectors[0].data)) goto done;
    ok = 1;
done:
    for (size_t i = 0; i < 3; ++i) dl_tensor_destroy(&vectors[i]);
    fixture_destroy(&f);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc > 2) {
        fprintf(stderr, "usage: %s [configs/tiny.conf]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *path = argc == 2 ? argv[1] : "configs/tiny.conf";
    DlModelConfig c = {0};
    size_t batch = 0, time = 0;
    if (!OK(dl_model_config_load(path, &c, &batch, &time))) return EXIT_FAILURE;
    if (!hand_forward() || !rng_and_ownership(c) || !run_case(c, batch, time)) return EXIT_FAILURE;
    /* A one-block fixture cannot establish correct reverse block order. */
    c.layers = 2;
    c.seed = 73;
    if (!run_case(c, 2, 3)) return EXIT_FAILURE;
    printf("coordinate checks: %zu, max absolute error %.6g\n", coordinate_checks, coordinate_error);
    printf("direction checks: %zu, max absolute error %.6g\n", direction_checks, direction_error);
    puts("model checks passed");
    return EXIT_SUCCESS;
}

#undef OK
