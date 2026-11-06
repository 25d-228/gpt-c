#include "dl/optim.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

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
    fprintf(stderr, "%s: %.12g versus %.12g (tolerance %.3g)\n",
            where, a, b, tolerance);
    return 0;
}

#define CHECK(yes, where) do { if (!require((yes), (where))) goto done; } while (0)
#define STATUS(call, want) do { if (!status_is((call), (want), #call)) goto done; } while (0)
#define OK(call) STATUS((call), DL_OK)
#define NEAR(a, b, tol, where) do { \
    if (!close_value((a), (b), (tol), (tol), (where))) goto done; \
} while (0)

static DlModelConfig small_config(void)
{
    return (DlModelConfig){.vocab = 4, .layers = 1, .width = 4,
        .query_heads = 2, .kv_heads = 1, .ffn_width = 6,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0,
        .seed = UINT64_C(42)};
}

static DlAdamWConfig ordinary_config(void)
{
    return (DlAdamWConfig){0.9, 0.999, 1e-8, 0.1, 1.0};
}

/* Synthetic gradients isolate the optimizer from the already-tested model.
 * The second coordinate belongs to the FINAL norm, not the table tensor. */
static DlStatus synthetic_mean(DlAdamW *o, float table, float norm, uint64_t n)
{
    DlStatus status = dl_model_zero_grad(o->model);
    if (status != DL_OK) return status;
    o->model->params[0].grad.data[0] = table;
    o->model->params[o->model->param_count - 1].grad.data[0] = norm;
    return dl_adamw_add_mean(o, n);
}

static DlStatus set_two_weights(DlModel *m)
{
    DlStatus status = dl_model_mark_changed(m);
    if (status != DL_OK) return status;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].value.count; ++i)
            m->params[p].value.data[i] = 0.0f;
    m->params[0].value.data[0] = 2.0f;
    m->params[m->param_count - 1].value.data[0] = 2.0f;
    return DL_OK;
}

/* Fixed TEST-only capacity, checked before any copy. No struct-padding compare. */
enum { TEST_CAP = 4096 };
typedef struct {
    float weight[TEST_CAP], grad[TEST_CAP], moments[2 * TEST_CAP];
    double sum[TEST_CAP];
    uint64_t updates, tokens, model_revision, optim_revision;
    int active, gradients_valid;
} Image;

static int remember(const DlAdamW *o, Image *image)
{
    if (o->count > TEST_CAP) return 0;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlParameter *a = &o->model->params[p];
        memcpy(image->weight + at, a->value.data, a->value.count * sizeof(float));
        memcpy(image->grad + at, a->grad.data, a->grad.count * sizeof(float));
        at += a->value.count;
    }
    memcpy(image->moments, o->moments, 2 * o->count * sizeof(float));
    memcpy(image->sum, o->sum, o->count * sizeof(double));
    image->updates = o->updates;
    image->tokens = o->tokens;
    image->model_revision = o->model->revision;
    image->optim_revision = o->revision;
    image->active = o->active;
    image->gradients_valid = o->model->gradients_valid;
    return 1;
}

static int unchanged(const DlAdamW *o, const Image *image)
{
    if (o->count > TEST_CAP) return 0;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlParameter *a = &o->model->params[p];
        if (memcmp(image->weight + at, a->value.data, a->value.count * sizeof(float)) ||
            memcmp(image->grad + at, a->grad.data, a->grad.count * sizeof(float)))
            return 0;
        at += a->value.count;
    }
    return memcmp(image->moments, o->moments, 2 * o->count * sizeof(float)) == 0 &&
        memcmp(image->sum, o->sum, o->count * sizeof(double)) == 0 &&
        image->updates == o->updates && image->tokens == o->tokens &&
        image->model_revision == o->model->revision &&
        image->optim_revision == o->revision && image->active == o->active &&
        image->gradients_valid == o->model->gradients_valid;
}

static int scalar_steps(void)
{
    int ok = 0;
    DlModel m = {0};
    DlAdamW o = {0};
    DlAdamWConfig c = {0.5, 0.25, 0.1, 0.5, 0.0};
    DlOptimStats stats = {0};
    OK(dl_model_create(&m, small_config()));
    OK(set_two_weights(&m));
    OK(dl_adamw_create(&o, &m, c, 1024 * 1024));
    size_t norm_at = m.value_count - m.config.width;
    const float gradients[] = {2.0f, -1.0f, 0.0f, 3.0f};
    const double rates[] = {0.2, 0.2, 0.1, 0.05};
    float rm = 0.0f, rv = 0.0f, rw = 2.0f, rn = 2.0f;
    uint64_t revision = m.revision;
    for (size_t step = 0; step < 4; ++step) {
        OK(dl_adamw_begin(&o));
        OK(synthetic_mean(&o, gradients[step], gradients[step], 1));
        OK(synthetic_mean(&o, gradients[step], gradients[step], 2));
        STATUS(dl_adamw_add_mean(&o, 2), DL_STATE);
        CHECK(o.updates == step && o.tokens == 3, "microbatches are not updates");
        OK(dl_adamw_step(&o, rates[step], &stats));
        /* Independent scalar recurrence: long-double arithmetic and powl.
         * Deliberate float stores match the declared moment/weight contract. */
        long double g = gradients[step], eta = rates[step];
        rm = (float)(0.5L * rm + 0.5L * g);
        rv = (float)(0.25L * rv + 0.75L * g * g);
        long double mh = rm / (1.0L - powl(0.5L, (long double)(step + 1)));
        long double vh = rv / (1.0L - powl(0.25L, (long double)(step + 1)));
        long double delta = eta * mh / (sqrtl(vh) + (long double)c.epsilon);
        rw = (float)((long double)rw * (1.0L - eta * 0.5L) - delta);
        rn = (float)((long double)rn - delta);
        NEAR(o.moments[0], rm, 2e-7, "first moment");
        NEAR(o.moments[o.count], rv, 2e-7, "second moment");
        NEAR(o.moments[norm_at], rm, 2e-7, "norm moment excludes decay");
        NEAR(m.params[0].value.data[0], rw, 2e-7, "matrix scalar update");
        NEAR(m.params[m.param_count - 1].value.data[0], rn, 2e-7, "norm scalar update");
        CHECK(stats.update == step + 1 && stats.tokens == 3 &&
              m.revision == revision + step + 1 && !m.gradients_valid && !o.active,
              "completed update lifecycle");
        if (step == 0) {
            NEAR(rw, 1.6095238095238096, 2e-7, "hand first weight");
            NEAR(rn, 1.8095238095238095, 2e-7, "hand no-decay weight");
        }
        if (step == 1) {
            NEAR(rm, 0.0, 0.0, "opposite gradient cancels first moment");
            NEAR(rv, 1.5, 0.0, "hand second raw moment");
            NEAR(rw, 1.4485714285714286, 2e-7, "hand second weight");
        }
        STATUS(dl_adamw_step(&o, rates[step], &stats), DL_STATE);
    }
    puts("scalar: four reference updates, bias correction and decay passed");
    ok = 1;
done:
    dl_adamw_destroy(&o);
    dl_model_destroy(&m);
    return ok;
}

static int accumulation_and_clipping(void)
{
    int ok = 0;
    DlModel m = {0};
    DlAdamW o = {0};
    DlOptimStats stats = {0};
    DlAdamWConfig c = {0.0, 0.0, 0.1, 0.0, 2.0};
    OK(dl_model_create(&m, small_config()));
    OK(dl_adamw_create(&o, &m, c, 1024 * 1024));
    size_t norm_at = m.value_count - m.config.width;
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 3.0f, 4.0f, 10));
    OK(dl_adamw_step(&o, 0.01, &stats));
    NEAR(stats.grad_norm, 5.0, 1e-14, "global 3-4-5 norm");
    NEAR(stats.clip_scale, 0.4, 1e-14, "one global clipping scale");
    NEAR(o.moments[0], 1.2, 2e-7, "table clipped mean");
    NEAR(o.moments[norm_at], 1.6, 2e-7, "norm clipped mean");
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 3.0f, 0.0f, 2));
    OK(synthetic_mean(&o, 0.0f, 4.0f, 6));
    NEAR(o.sum[0], 6.0, 0.0, "two tokens times first mean");
    NEAR(o.sum[norm_at], 24.0, 0.0, "six tokens times second mean");
    CHECK(o.tokens == 8, "actual scored-token denominator");
    OK(dl_adamw_step(&o, 0.01, &stats));
    double norm = sqrt(0.75 * 0.75 + 3.0 * 3.0);
    NEAR(stats.grad_norm, norm, 1e-14, "unequal-token mean norm");
    NEAR(o.moments[0], 0.75 * 2.0 / norm, 2e-7, "weighted clipped mean");
    NEAR(o.moments[norm_at], 3.0 * 2.0 / norm, 2e-7, "second weighted mean");
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 100.0f, 0.0f, 1));
    OK(synthetic_mean(&o, -100.0f, 2.0f, 1));
    OK(dl_adamw_step(&o, 0.01, &stats));
    NEAR(stats.grad_norm, 1.0, 1e-14, "cancellation before clipping");
    NEAR(stats.clip_scale, 1.0, 0.0, "no clipping of the final small mean");
    NEAR(o.moments[0], 0.0, 0.0, "opposing contributions cancel");
    NEAR(o.moments[norm_at], 1.0, 0.0, "do not clip individual batches");
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 7.0f, 9.0f, 1));
    OK(dl_adamw_begin(&o)); /* Discard a partial group, NOT the moments. */
    CHECK(o.tokens == 0 && o.updates == 3 && !m.gradients_valid,
          "restart keeps update clock, clears gradient ownership");
    for (size_t i = 0; i < o.count; ++i) NEAR(o.sum[i], 0.0, 0.0, "clear sum");
    OK(synthetic_mean(&o, 0.0f, 0.0f, 1));
    OK(dl_adamw_step(&o, 0.01, &stats));
    NEAR(stats.grad_norm, 0.0, 0.0, "zero norm");
    NEAR(stats.clip_scale, 1.0, 0.0, "zero norm avoids division");
    puts("accumulation: unequal tokens, global clipping, cancellation and restart passed");
    ok = 1;
done:
    dl_adamw_destroy(&o);
    dl_model_destroy(&m);
    return ok;
}

static int failure_checks(void)
{
    int ok = 0;
    DlModel m = {0};
    DlAdamW o = {0}, rejected = {0};
    DlAdamWConfig c = {0.5, 0.25, 0.1, 0.0, 0.0};
    DlOptimStats stats = {77, 88, 9.0, 10.0, 11.0};
    Image before;
    OK(dl_model_create(&m, small_config()));
    OK(set_two_weights(&m));
    /* This value is deliberately huge; no model forward uses this fixture. */
    OK(dl_model_mark_changed(&m));
    DlTensor *last = &m.params[m.param_count - 1].value;
    last->data[last->count - 1] = FLT_MAX;
    size_t bytes = m.value_count * (2 * sizeof(float) + sizeof(double));
    DlAdamWConfig bad = c;
    bad.beta1 = 1.0;
    STATUS(dl_adamw_create(&rejected, &m, bad, bytes), DL_ARGUMENT);
    bad = c; bad.epsilon = 0.0;
    STATUS(dl_adamw_create(&rejected, &m, bad, bytes), DL_ARGUMENT);
    bad = c; bad.clip_norm = NAN;
    STATUS(dl_adamw_create(&rejected, &m, bad, bytes), DL_ARGUMENT);
    STATUS(dl_adamw_create(&rejected, &m, c, bytes - 1), DL_RANGE);
    CHECK(rejected.model == NULL && rejected.sum == NULL && rejected.moments == NULL,
          "rejected constructor did not acquire ownership");
    OK(dl_adamw_create(&o, &m, c, bytes));
    STATUS(dl_adamw_create(&o, &m, c, bytes), DL_STATE);
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_STATE);

    OK(dl_adamw_begin(&o));
    STATUS(dl_adamw_add_mean(&o, 1), DL_STATE);
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_STATE);
    OK(dl_model_zero_grad(&m));
    STATUS(dl_adamw_add_mean(&o, 0), DL_ARGUMENT);
    OK(synthetic_mean(&o, 1.0f, 0.0f, 2));
    OK(dl_model_zero_grad(&m));
    m.params[0].grad.data[0] = 5.0f;
    DlTensor *last_grad = &m.params[m.param_count - 1].grad;
    last_grad->data[last_grad->count - 1] = NAN;
    CHECK(remember(&o, &before), "capture late bad gradient");
    STATUS(dl_adamw_add_mean(&o, 1), DL_NUMERIC);
    CHECK(unchanged(&o, &before), "late NaN did not partially accumulate");
    last_grad->data[last_grad->count - 1] = INFINITY;
    CHECK(remember(&o, &before), "capture infinite gradient");
    STATUS(dl_adamw_add_mean(&o, 1), DL_NUMERIC);
    CHECK(unchanged(&o, &before), "infinite gradient rejected unchanged");
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_STATE); /* Unconsumed gradient. */
    OK(dl_adamw_begin(&o));
    OK(dl_model_zero_grad(&m));
    m.params[0].grad.data[0] = 1.0f;
    last_grad->data[last_grad->count - 1] = -1.0f;
    OK(dl_adamw_add_mean(&o, 1));
    CHECK(remember(&o, &before), "capture candidate overflow");
    STATUS(dl_adamw_step(&o, 1e38, &stats), DL_NUMERIC);
    CHECK(unchanged(&o, &before), "late weight overflow did not commit earlier weights/moments");
    CHECK(stats.update == 77 && stats.tokens == 88 && stats.learning_rate == 9.0 &&
          stats.grad_norm == 10.0 && stats.clip_scale == 11.0, "failed step leaves statistics");
    STATUS(dl_adamw_step(&o, NAN, &stats), DL_ARGUMENT);
    STATUS(dl_adamw_step(&o, 0.0, &stats), DL_ARGUMENT);
    STATUS(dl_adamw_step(&o, 0.01, NULL), DL_ARGUMENT);
    CHECK(unchanged(&o, &before), "bad call arguments leave all state");

    /* Explicit test-only corruption; ordinary callers never edit this storage. */
    o.moments[2 * o.count - 1] = -1.0f;
    CHECK(remember(&o, &before), "capture corrupt second moment");
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_NUMERIC);
    CHECK(unchanged(&o, &before), "negative second moment rejected atomically");
    o.moments[2 * o.count - 1] = 0.0f;
    o.updates = DL_OPT_MAX_COUNT;
    CHECK(remember(&o, &before), "capture update limit");
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_OVERFLOW);
    CHECK(unchanged(&o, &before), "update limit preserved");
    o.updates = 0;
    uint64_t old_revision = m.revision;
    m.revision = UINT64_MAX; o.revision = UINT64_MAX;
    CHECK(remember(&o, &before), "capture revision limit");
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_OVERFLOW);
    CHECK(unchanged(&o, &before), "revision exhaustion preserved");
    m.revision = old_revision; o.revision = old_revision;
    OK(dl_adamw_step(&o, 0.01, &stats)); /* Retry has no extra bias-correction tick. */
    CHECK(stats.update == 1, "only successful calls advance schedule/update count");
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 0.0f, 0.0f, DL_OPT_MAX_COUNT));
    OK(dl_model_zero_grad(&m));
    CHECK(remember(&o, &before), "capture token count limit");
    STATUS(dl_adamw_add_mean(&o, 1), DL_OVERFLOW);
    CHECK(unchanged(&o, &before), "exact-count limit leaves the accumulation intact");
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, FLT_MAX, 0.0f, 1));
    CHECK(remember(&o, &before), "capture unrepresentable second moment");
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_NUMERIC);
    CHECK(unchanged(&o, &before), "finite gradient can still overflow a float moment");
    OK(dl_model_mark_changed(&m));
    STATUS(dl_adamw_begin(&o), DL_STATE);
    STATUS(dl_adamw_add_mean(&o, 1), DL_STATE);
    STATUS(dl_adamw_step(&o, 0.01, &stats), DL_STATE);
    dl_adamw_destroy(&o);
    OK(dl_adamw_create(&o, &m, c, bytes));
    CHECK(o.updates == 0, "explicit reattachment resets optimizer history");
    puts("failure: construction, nonfinite data, late overflow and state limits passed");
    ok = 1;
done:
    dl_adamw_destroy(&rejected);
    dl_adamw_destroy(&o);
    dl_adamw_destroy(&o); /* Repeatable empty destruction. */
    dl_model_destroy(&m);
    return ok;
}

static int extremes_and_decay(void)
{
    int ok = 0;
    DlModel m = {0};
    DlAdamW o = {0};
    DlOptimStats stats = {0};
    Image before;
    DlAdamWConfig c = ordinary_config();
    OK(dl_model_create(&m, small_config()));
    OK(dl_adamw_create(&o, &m, c, 1024 * 1024));
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 0.0f, 0.0f, 17));
    CHECK(remember(&o, &before), "capture pure decay");
    STATUS(dl_adamw_step(&o, 11.0, &stats), DL_RANGE);
    CHECK(unchanged(&o, &before), "negative shrink factor rejected");
    OK(dl_adamw_step(&o, 0.01, &stats));
    size_t at = 0;
    for (size_t p = 0; p < m.param_count; ++p) {
        DlParameter *a = &m.params[p];
        for (size_t i = 0; i < a->value.count; ++i, ++at) {
            double expected = (double)before.weight[at] * (a->decay ? 0.999 : 1.0);
            NEAR(a->value.data[i], expected, 1e-7, "unique decay once, norm scales never");
            NEAR(o.moments[at], 0.0, 0.0, "decay excluded from first moment");
            NEAR(o.moments[o.count + at], 0.0, 0.0, "decay excluded from second moment");
        }
    }
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, FLT_MAX, FLT_MAX, 1));
    OK(dl_adamw_step(&o, 0.01, &stats));
    NEAR(stats.grad_norm, sqrt(2.0) * (double)FLT_MAX, 1e-14, "norm exceeds float range safely");
    CHECK(stats.clip_scale > 0.0 && stats.clip_scale < 1e-38,
          "tiny positive clipping coefficient kept in double");
    NEAR(o.moments[0], 0.1 / sqrt(2.0), 2e-7, "huge finite gradients clipped before square");
    float first = o.moments[0], second = o.moments[o.count];
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 0.0f, 0.0f, 1));
    OK(dl_adamw_step(&o, 0.01, &stats));
    NEAR(o.moments[0], 0.9 * first, 2e-7, "zero gradient does not erase momentum");
    NEAR(o.moments[o.count], 0.999 * second, 2e-7, "zero gradient still ages second moment");
    NEAR(stats.grad_norm, 0.0, 0.0, "momentum is not part of gradient norm");
    dl_adamw_destroy(&o);
    c.beta1 = nextafter(1.0, 0.0); c.beta2 = c.beta1;
    c.weight_decay = 0.0; c.clip_norm = 0.0;
    OK(dl_adamw_create(&o, &m, c, 1024 * 1024));
    OK(dl_adamw_begin(&o));
    OK(synthetic_mean(&o, 1.0f, 0.0f, 1));
    double weight = m.params[0].value.data[0];
    OK(dl_adamw_step(&o, 0.01, &stats));
    NEAR(m.params[0].value.data[0], weight - 0.01 / (1.0 + c.epsilon), 2e-7,
         "bias correction close to beta one");
    puts("numerics: pure decay, huge/zero gradients and near-one betas passed");
    ok = 1;
done:
    dl_adamw_destroy(&o);
    dl_model_destroy(&m);
    return ok;
}

static int combined_vs_micro(DlBackend backend, int unequal)
{
    int ok = 0;
    DlModel full = {0}, micro = {0};
    DlTrainTape large = {0}, one = {0}, two = {0};
    DlAdamW a = {0}, b = {0};
    DlOptimStats sa = {0}, sb = {0};
    DlModelConfig config = {.vocab = 8, .layers = 2, .width = 8,
        .query_heads = 4, .kv_heads = 2, .ffn_width = 12,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0,
        .seed = UINT64_C(42)};
    DlAdamWConfig optim = {0.9, 0.99, 1e-5, 0.1, 0.15};
    const size_t ids[] = {1, 2, 1, 3, 1, 4, 4, 2, 5};
    const size_t targets[] = {2, 1, 3, 1, 4, 2, 2, 5, 1};
    double max_gradient_difference = 0.0, max_weight_difference = 0.0;
    OK(dl_model_create(&full, config));
    OK(dl_model_create(&micro, config));
    OK(dl_train_tape_create(&large, &full, 3, 3, 1024 * 1024));
    OK(dl_train_tape_create(&one, &micro, 1, 3, 1024 * 1024));
    OK(dl_train_tape_create(&two, &micro, 2, 3, 1024 * 1024));
    OK(dl_adamw_create(&a, &full, optim, 1024 * 1024));
    OK(dl_adamw_create(&b, &micro, optim, 1024 * 1024));
    for (size_t update = 0; update < 2; ++update) {
        double loss_full = 0.0, loss_micro = 0.0;
        OK(dl_adamw_begin(&a));
        OK(dl_adamw_begin(&b));
        OK(dl_model_forward(&full, &large, backend, ids, targets, 9, &loss_full));
        OK(dl_model_backward(&full, &large, DL_SET));
        OK(dl_adamw_add_mean(&a, 9));
        size_t offset = 0;
        while (offset < 9) {
            DlTrainTape *tape = unequal && offset == 3 ? &two : &one;
            double loss = 0.0;
            OK(dl_model_forward(&micro, tape, backend, ids + offset,
                                 targets + offset, tape->rows, &loss));
            OK(dl_model_backward(&micro, tape, DL_SET));
            OK(dl_adamw_add_mean(&b, (uint64_t)tape->rows));
            loss_micro += (double)tape->rows * loss / 9.0;
            offset += tape->rows;
        }
        NEAR(loss_full, loss_micro, 2e-6, "same token-weighted objective");
        CHECK(a.tokens == 9 && b.tokens == 9 && a.updates == update &&
              b.updates == update, "update clock independent of partition");
        for (size_t i = 0; i < a.count; ++i) {
            double ga = a.sum[i] / 9.0, gb = b.sum[i] / 9.0;
            max_gradient_difference = fmax(max_gradient_difference, fabs(ga - gb));
            NEAR(ga, gb, 3e-6, "combined gradient versus token-weighted microbatches");
        }
        /* Make a deliberately pending forward; the update must stale its tape. */
        double pending_loss = 0.0;
        OK(dl_model_forward(&full, &large, backend, ids, targets, 9, &pending_loss));
        double lr = 0.002 / (double)(update + 1);
        OK(dl_adamw_step(&a, lr, &sa));
        OK(dl_adamw_step(&b, lr, &sb));
        STATUS(dl_model_backward(&full, &large, DL_SET), DL_STATE);
        CHECK(sa.update == update + 1 && sb.update == update + 1 &&
              sa.tokens == 9 && sb.tokens == 9 && sa.clip_scale < 1.0,
              "matched clipped updates consumed each group once");
        NEAR(sa.grad_norm, sb.grad_norm, 3e-6, "same global norm");
        for (size_t p = 0; p < full.param_count; ++p) {
            for (size_t i = 0; i < full.params[p].value.count; ++i) {
                double x = full.params[p].value.data[i], y = micro.params[p].value.data[i];
                max_weight_difference = fmax(max_weight_difference, fabs(x - y));
                NEAR(x, y, 3e-6, "combined versus accumulated updated weight");
            }
        }
        for (size_t i = 0; i < 2 * a.count; ++i)
            NEAR(a.moments[i], b.moments[i], 3e-6, "matched moment histories");
    }
    printf("model: %s partition %s, two updates passed (max |dg| %.3g, |dw| %.3g)\n",
           backend == DL_BACKEND_REF ? "ref" : "blocked", unequal ? "1+2" : "1+1+1",
           max_gradient_difference, max_weight_difference);
    ok = 1;
done:
    dl_adamw_destroy(&b);
    dl_adamw_destroy(&a);
    dl_train_tape_destroy(&two);
    dl_train_tape_destroy(&one);
    dl_train_tape_destroy(&large);
    dl_model_destroy(&micro);
    dl_model_destroy(&full);
    return ok;
}

int main(void)
{
    int ok = 1;
    ok &= scalar_steps();
    ok &= accumulation_and_clipping();
    ok &= failure_checks();
    ok &= extremes_and_decay();
    ok &= combined_vs_micro(DL_BACKEND_REF, 0);
    ok &= combined_vs_micro(DL_BACKEND_REF, 1);
    ok &= combined_vs_micro(DL_BACKEND_BLOCKED, 0);
    ok &= combined_vs_micro(DL_BACKEND_BLOCKED, 1);
    if (!ok) return 1;
    puts("all optimizer checks passed");
    return 0;
}
