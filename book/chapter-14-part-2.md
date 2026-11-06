# Chapter 14 · Part 2 — Checking optimizer state and equivalent batches

[Part 1](https://25d-228.github.io/gpt-c/chapter-14.html) derives and implements the complete optimizer. This part supplies the executable checks and build integration for the same **4.00-hour chapter**. There is no second implementation-hour allocation. Start from the Chapter 13 snapshot and add Part 1's header/source before creating the files below, or inspect the complete [Chapter 14 companion](https://25d-228.github.io/gpt-c/media/chapter-14/reference/Makefile).

The central comparison is not merely that an optimizer produces a decreasing number. We will compare a full batch with exactly equivalent microbatches, including unequal token counts, then compare their updated weights and moment histories. Separate hand calculations test the update formula, because two equally wrong optimizer runs could still agree with one another.

Create `tests/test_optim.c`. Its filename-labeled blocks, in reading order with one blank line between them, form the complete file. The fixed-capacity snapshot in this test is only for tiny fixtures; it is not an implementation limit or a training-checkpoint format. Every failure exits through owner cleanup.

## Check values without depending on assertions

The test helpers follow the earlier chapters. A failed status or non-finite comparison produces a diagnostic and a nonzero process exit. The macros jump to each test function's `done` label so a failed numerical expectation does not leak its model, tapes, or optimizer.


```c title="tests/test_optim.c"
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
```

`NEAR` uses both an absolute and a relative term. A tolerance of zero is reserved for results that are exact in these fixtures, such as zero counts or exact cancellation of binary-representable numbers. Reference weight checks use a small nonzero tolerance because the declared implementation stores floats.

The small optimizer-only fixture has vocabulary four, one block, width four, two query heads sharing one key/value head, and feed-forward width six. Its directory has 148 scalar values. We can test real directory iteration without allocating a large decoder or running a forward pass for a known synthetic gradient.

`synthetic_mean` first makes the model gradient arrays valid zeros, then writes a controlled table coordinate and a controlled norm-scale coordinate. This is test setup, not a substitute for the real-model integration below. `set_two_weights` legally announces an edit before modifying values, and is called before constructing the optimizer that will borrow the model.


```c title="tests/test_optim.c"
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
```

The two controlled coordinates belong to different parameter tensors. This matters for distinguishing global clipping from independent per-tensor clipping. They also have different decay flags. Both share the same incoming gradient in the scalar-update test, so any difference in their moment histories would reveal an incorrect coupling between decay and the loss derivative.

## Compare rejected calls with their exact prior state

The optimizer promises not to partly commit a rejected add or step. Checking only `updates == 0` would miss a changed early weight or moment. The next helpers copy the tiny model's logical arrays and relevant counters before a call, then compare all of them afterward.

We compare array bytes when testing **unchanged storage**, including a deliberately inserted NaN bit pattern. We do not use byte equality to compare independently evaluated floating-point algorithms, and we do not compare C struct padding. Metadata fields are compared individually. The fixed test capacity is checked before copying.


```c title="tests/test_optim.c"
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
```

This helper neither reads a file nor creates an audit artifact. It is ordinary test-local memory used to make one failure promise observable. All implementation arrays retain their existing owners; the copies are embedded in the test's stack object and require no separate free.

## Reproduce the hand calculation and continue its history

The scalar test runs the first two updates calculated in Part 1, then two more updates. Each update is assembled from two synthetic microbatches with one and two tokens, but their means are identical. Consequently the final mean equals the supplied scalar derivative. This isolates the distinction between the number of add calls, three total targets, and one completed update.

The independent scalar recurrence uses long-double expressions and `powl`, not the optimizer's `correction` or candidate helper. It deliberately rounds moment and weight storage to float, matching the public precision contract. On a platform where `long double` has no more precision than double, the separate expressions and hand constants still provide the check; no extended-precision hardware assumption is required.


```c title="tests/test_optim.c"
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
```

The first pair of weights should be approximately `1.6095238` for the matrix and `1.8095238` for the no-decay norm. On update two the first moment becomes exactly zero and the second becomes 1.5. The matrix still shrinks to approximately `1.4485714`. The test then changes the rates to expose a hidden assumption that the first rate remains fixed.

The last `step` call inside each iteration is intentionally rejected. It proves that stale sum bytes cannot cause a second weight update. The next `begin` retains the existing moments and successful-update clock, then starts a new group. The following test examines normalization and clipping independently of these scalar values.

## Exercise token weighting and one global norm

Set both betas to zero for this test. Then the first stored moment is exactly the current clipped gradient, up to float rounding, and the second is its square. This makes the clipping decision visible directly in optimizer state rather than infer it from Adam's nearly scale-invariant first weight step.

The first group places three and four in different tensors and checks norm five, coefficient 0.4, and first moments 1.2 and 1.6. The second uses the unequal two-token/six-token example. The third combines large opposing contributions before making the norm decision. Finally, a partially accumulated group is discarded and replaced by a zero gradient without resetting the update history.


```c title="tests/test_optim.c"
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
```

For the unequal case, the pre-clipping mean is `(0.75, 3)`, with norm approximately `3.0923292192`. Threshold two produces coefficient approximately `0.6467616668`. The test compares the stored moments with that **single** coefficient applied to both coordinates.

For the cancellation case the combined mean is `(0, 1)`. Its coefficient must remain one even though each contributing batch separately had norm about one hundred. A per-microbatch clipping implementation cannot satisfy this expectation. The final zero group also catches an unconditional `threshold / norm` that divides by zero.

## Reject invalid construction before acquiring ownership

The failure test begins by creating a model, making its final scalar weight very large, and attaching an optimizer. This fixture is never used for model forward computation: it exists to make a late candidate overflow observable.

Bad beta, epsilon, and NaN threshold inputs must be rejected. A byte budget one byte below the exact array requirement must fail before construction. Reconstructing over a live owner must not silently lose its original allocations.


```c title="tests/test_optim.c"
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
```

We continue the **same** `failure_checks` function below. First test empty and unconsumed states. Then put a NaN at the last gradient coordinate while an earlier coordinate is finite and nonzero. A one-pass write-as-you-check implementation would already have modified the sum when it reaches that NaN; our snapshot comparison catches that partial mutation.

The later oversized-rate case is different: every gradient is finite. An early candidate weight remains within float range, but the final huge weight plus its adaptive change does not. The entire step, including all moments and the caller's statistics, must remain unchanged. A valid retry afterward must still be update one.


```c title="tests/test_optim.c"
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
```

Continue by explicitly corrupting internal state only inside this test. A negative second moment, an exhausted update count, and an exhausted model revision are independently rejected. These edits are not examples of a supported optimizer client interface; they are controlled ways to reach failure paths without performing an astronomical number of updates.

The token-count test reaches the exact-integer limit without allocating that many examples: the synthetic mean has zero values but a very large declared count. A further token would exceed the permitted range. The final finite-gradient case shows that non-finite input checks alone are insufficient: a float second moment can overflow even when the input derivative itself is finite.


```c title="tests/test_optim.c"
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
```

The final external revision change must detach the optimizer logically. Destroying it and constructing a new one is an explicit history reset. Silently overwriting its saved revision would make the check pass while concealing that the old moments belonged to a different weight history.

These tests distinguish call-level nonmutation from group-level recovery. A rejected `add_mean` preserves accepted earlier sums, but a real backward failure invalidates the temporary model gradients. The integration caller must stop, discard or recompute the intended group, and not mistake a partial group for a full update.

## Isolate decay and extreme finite arithmetic

A fresh optimizer with zero gradients has zero moments. Every matrix coordinate must therefore shrink once by `0.999` for learning rate `0.01` and decay `0.1`; every norm coordinate must remain unchanged. This visits the entire actual parameter directory, including its single tied-table entry. It also checks that neither moment contains a decay contribution.

Then use two `FLT_MAX` gradients with clipping enabled. Their double norm is finite even though it exceeds float range; the clipped mean fits in the moment storage. A following zero gradient must age, not erase, those moments. Finally, betas set to the largest representable double below one exercise small correction denominators.


```c title="tests/test_optim.c"
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
```

`nextafter(1.0, 0.0)` returns the neighboring representable double toward zero, so the final test is genuinely close to one without rounding the coefficient to the invalid value one. At update one with gradient one, bias correction should recover corrected moments near one. It is not enough that the computed denominator is nonzero; the resulting step must have the expected magnitude.

The large-gradient success does not mean clipping repairs NaN or infinity. Those inputs were rejected before accumulation in the preceding test. Nor does zero gradient mean no update: retained first moments and decoupled decay can still move parameters. The test inspects their distinct state effects.

## Integrate two equivalent decoder computations

Now use actual model forward and backward calls. The configuration has two blocks, vocabulary eight, width eight, four query heads, two key/value heads, and feed-forward width twelve. Each model owns 1,064 parameter values. Both are initialized with the same explicit seed, so they begin with equal weights without copying live owner structs.

The input/target arrays contain three independent sequences of three targets each. One model computes a combined batch. The other uses either three one-sequence microbatches or an unequal one-sequence/two-sequence partition. No targets, causal contexts, or RoPE positions change between the compared computations.

The loop performs two real updates with different supplied learning rates. Before each update it compares the token-weighted losses and **every** gradient coordinate. After each update it compares **every** parameter value and both moment arrays. This second update checks that the equivalent histories remain compatible, not merely that two fresh optimizers have a similar first step.


```c title="tests/test_optim.c"
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
```

The `offset` counts flattened target positions, not parameter coordinates. With the unequal partition, `one.rows` is three and `two.rows` is six. Each `DL_SET` backward replaces the micro model's temporary gradient; each add immediately transfers the weighted contribution to the persistent accumulator. Nothing updates a weight inside that loop.

`loss_micro` uses the same target weights as the derivatives. The combined batch has nine targets, so the individual means contribute with factors `3/9` and `6/9`, or three factors of `3/9` for the equal partition. A simple average of two losses would be wrong in the unequal case.

The additional `full` forward creates a pending tape **after** its gradient has been accepted. Forward does not change the accumulator or gradient validity. The successful update then advances the model revision, so trying to differentiate that old tape must return `DL_STATE`. This tests the connection between optimizer commits and the model's existing saved-state rules.

The integration tolerances are `3e-6 + 3e-6 * max(abs(a), abs(b))` for gradients, weights, and moments, and a similar `2e-6` rule for loss. They are not licenses to loosen checks until a wrong normalization passes. Part 1's hand examples and the unequal partition should fail substantially under those mistakes. The maximum observed discrepancies are printed to make the numerical scale visible.

## Run every independent check

The complete driver uses bitwise AND assignment on integer success flags so one failed group does not prevent the later groups from reporting their own diagnostic. Every group still cleans up before returning. Any failure ultimately makes the process exit nonzero.


```c title="tests/test_optim.c"
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
```

## Add the optimizer to the existing build

Replace the root Makefile below. `src/optim.c` joins `CORE`, its header joins `HEADERS`, and `test_optim` joins the existing test list. Debug, sanitizer, and optimized builds remain separate. All previous applications and the asset-preparation tests remain available. The compiler default still uses the native Apple toolchain; another compatible C17 compiler can be selected through `CC`.


```make title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2 -fno-lto
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/rng.c src/model.c src/data.c src/optim.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/rng.h include/dl/model.h include/dl/data.h include/dl/optim.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data test_optim

.PHONY: all debug sanitize release test test-assets clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data

build/debug/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

test-assets:
	python3 tests/test_prepare.py

clean:
	rm -rf build
```

From the learner implementation directory, run the focused checks first:

```sh
make build/debug/test_optim build/sanitize/test_optim build/release/test_optim
./build/debug/test_optim
./build/sanitize/test_optim
./build/release/test_optim
```

On a machine without `xcrun`, select a compatible installed compiler explicitly, for example `make CC=clang build/debug/test_optim`. That override is not a request to install a Linux VM on the Mac. The new code uses C17, the standard math library, and the already-established model interfaces; it does not introduce a particular OS release or accelerator requirement.

Then run the inherited checks:

```sh
make test
make test-assets
```

`make test` builds and runs all twelve C test programs in three modes and runs the earlier small kernel-comparison command. Its timing output is unrelated to optimizer performance. `make test-assets` runs the nine network-free preparation tests from Chapter 13; it does not download TinyStories.

A successful optimizer run prints the following structure. The final digits in the discrepancy measurements may vary by compiler, math library, and machine:

```text
scalar: four reference updates, bias correction and decay passed
accumulation: unequal tokens, global clipping, cancellation and restart passed
failure: construction, nonfinite data, late overflow and state limits passed
numerics: pure decay, huge/zero gradients and near-one betas passed
model: ref partition 1+1+1, two updates passed (max |dg| 8.94e-08, |dw| 1.49e-08)
model: ref partition 1+2, two updates passed (max |dg| 7.95e-08, |dw| 1.49e-08)
model: blocked partition 1+1+1, two updates passed (max |dg| 8.94e-08, |dw| 1.49e-08)
model: blocked partition 1+2, two updates passed (max |dg| 7.95e-08, |dw| 1.49e-08)
all optimizer checks passed
```

These lines, including the shown discrepancies, were observed with Clang 17 on x86_64 Linux in debug, sanitizer, and optimized configurations. The cumulative C suite and nine inherited Python tests passed without compiler warnings; a leak-enabled sanitizer run of the new optimizer test reported no leaks. No native M4/macOS execution, optimizer throughput measurement, sustained text training, or process-memory measurement is implied by those checks.

The four model cases exercise eight matched update boundaries. They do not constitute a corpus training result or a fresh finite-difference gradient suite. The model's existing gradient checks remain in the cumulative tests; this chapter tests how already-computed derivatives become optimizer state and weights.

## Change one mechanism and predict the failure

Work on a temporary copy of the implementation and restore each deliberate defect before continuing. Do not change expected values to accommodate a mutation.

**Drop token weighting.** Replace the two occurrences of `(double)tokens * (double)g->data[i]` in `dl_adamw_add_mean` with `(double)g->data[i]`, leaving the token denominator unchanged. The accumulator now stores a sum of means, not a token sum. The scalar moment check and the unequal-token checks should fail. A superficially reasonable loss printout would not establish that the update represents the intended objective.

**Use the token clock for bias correction.** In the two `correction` calls in `dl_adamw_step`, use `o->tokens` instead of `update`. The first synthetic group has three tokens but must be update one. The hand update should fail. Combined and microbatch model runs may still agree, because both use nine targets and share the same incorrect clock. That is why equivalence alone does not validate AdamW.

**Put decay into the derivative.** Immediately after calculating `g` in `candidate_at`, add `weight * o->config.weight_decay` when `decay` is true. The pure-decay test now observes nonzero moments from a zero loss gradient, and the hand matrix/norm comparison fails. Restore the separate shrink term without contaminating the moments.

**Skip clipping.** Remove the threshold branch that changes `clip` from one. The 3–4–5 example must fail its coefficient/moment expectations, and the huge-gradient step should reject its unrepresentable second moment instead of succeeding. A normal first-step parameter change can look similar despite missing clipping; inspect the moments and reported coefficient as well.

**Clear moment history at every begin.** Add a loop that zeros both moment halves in `dl_adamw_begin`. The first scalar update still has zero initial moments as expected, but the second and later updates fail their recurrence checks. Both equivalent model paths can still agree because both have been changed in the same wrong way.

These five isolated incorrect variants were compiled and run under the sanitizer configuration and failed numerical/state expectations without sanitizer memory diagnostics. Their purpose is to distinguish an algorithmic bug from an invalid memory access, not to manufacture a larger test-count claim.

For a valid variation, change only the real-model integration fixture's clipping threshold from `0.15` to `0.075`. Both compared computations should still agree before and after two updates, while the clipping scale becomes smaller. This variation passed in the same environment. It changes the training rule consistently rather than weakening a test. As another local experiment, print the first-step weight change and moment norm with two clipping thresholds and explain why the weight change may move less than proportionally.

## Diagnose the first mismatch, not the last symptom

| Observation | Likely mechanism to inspect |
|---|---|
| Equal microbatches pass, unequal ones fail | Missing target weights or an average of means |
| Combined and microbatch updates agree, hand scalar steps fail | Shared error in correction clock, epsilon placement, or decay convention |
| Norm scales shrink with zero fresh moments and zero gradient | Decay membership was inferred incorrectly or ignored |
| Clipping coefficient changes when a shared table is counted twice | Directory enumeration is not unique |
| NaN is reported but earlier weights or moments changed | Validation and committed writes were interleaved |
| Every update behaves like the first | Moments or the update clock are being reset |
| Second `add_mean` unexpectedly succeeds without new gradients | Consumption state is not being enforced |
| Step fails with a pending valid gradient | The most recent backward has not been added to the accumulator |
| A huge finite derivative fails despite successful backward | Check moment float range and whether clipping was enabled |
| Backward accepts a tape created before an update | Model revision invalidation was omitted |

A reported failure is not a reason to blindly retry the entire loop. First identify whether the rejected operation left the group unchanged, whether backward invalidated temporary gradients, or whether an external edit detached the optimizer. The recovery point follows from that distinction.

## Where the next chapter begins

We now have a complete C forward/backward model, byte batches, and an explicit checked update step. A full trainer must still choose data windows, call the group lifecycle in the right order, evaluate held-out data without disturbing training state, select learning rates, and serialize a coherent restart boundary. Those are the next chapter's mechanisms, not hidden helpers assumed by the tests here.

The important invariant carried forward is simple: a successful optimizer step consumes one well-defined token mean and advances one update clock exactly once. Checkpoint saves will be restricted to completed update boundaries, where no partly accumulated derivative is an implicit missing dependency.
