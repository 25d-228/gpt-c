# Chapter 12 · Part 3 — Checking the assembled model

A loss near $\log V$ establishes almost nothing about a decoder. The all-zero model has exactly that loss. A decreasing loss is stronger evidence, but a gradient missing one of several paths can still point downhill. The assembled model therefore needs checks that distinguish the correct graph, the derivative of that graph, and the state in which its derivative may be used.

**Starting state:** the complete files from [Part 1](https://25d-228.github.io/gpt-c/chapter-12.html) and [Part 2](https://25d-228.github.io/gpt-c/chapter-12-part-2.html), plus the unchanged Chapter 11 numerical operators and tests. Create `tests/test_model.c` from all filename-labeled C blocks below in order, separated by one blank line. Replace the root Makefile with the complete final listing. This completes the same **4.50-hour** chapter.

## Choose checks that expose composition mistakes

The diagnostic model uses the loaded file: one block, width 16, four query heads, two key/value heads, FFN width 40, vocabulary 16, batch one, time four. It exercises all eleven parameter tensors without needing a text corpus. A second fixture uses two blocks, batch two, and time three. One block cannot reveal incorrect block-reversal order; one sequence cannot reveal a crossed batch boundary.

The tests use input IDs `[1,2,1,3]` and targets `[2,1,3,2]` in the four-token case. These are adjacent windows of the explicitly stated sequence `[1,2,1,3,2]`, so there is no hidden target preparation. Repeated ID 1 forces lookup-gradient accumulation. The model itself does not shift targets: its caller must supply the intended targets.

Three kinds of numerical checks complement one another. An independent two-coordinate forward calculation checks the schedule, normalization placement, causal averaging, SwiGLU, and tied output projection. Selected finite differences check every parameter family. Whole-model directions change every coordinate together, checking the combined derivative rather than another isolated layer.

Start with explicit diagnostic helpers. None of the checks depends on `assert`, which can disappear in an optimized build. `close_value` rejects non-finite operands and combines an absolute tolerance with a scale-dependent relative tolerance. The `OK` macro formats a failed function call and returns a Boolean; unlike the model's error-return macro, it does not itself leave the calling function.

```c title="tests/test_model.c"
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
```

The fixture owns a model and a tape, and contains bounded input arrays. These bounds belong to this test executable, not the numerical library: at most two sequences of four tokens, two blocks, vocabulary 16, width 16, and FFN width 40. The default configuration fits them. A larger training configuration will need a training application rather than silently turning this diagnostic into an expensive test run.

Destruction follows the borrowing relation: tape first, model second. Construction checks every returned status and uses the same cleanup path after failure. The metadata copy in forward will let us mutate the fixture's caller-side IDs after forward without changing the saved example.

```c title="tests/test_model.c"
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
```

`flatten` traverses the unique directory, producing a flat test snapshot of values or gradients. It does not reorder tensors or treat the output table as another parameter. `restore` calls `dl_model_mark_changed` before writing any values, so a numerical derivative probe cannot accidentally leave an old tape eligible for backward.

```c title="tests/test_model.c"
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
```

## Check initialization and partial construction

The first checks establish that a fixed seed produces the documented integer stream, that generated fractions stay in their half-open interval, and that two model owners constructed with the same configuration have equal values but distinct ownership. Norm scales must be one, every gradient must be initialized, and each matrix must fit its specified initialization bound.

Directory names and value pointers must be unique. Total values and scale counts must agree with the formulas from Part 1. Two failure cases then exercise cleanup: a tiny arena budget rejects a tape after descriptor allocation, and an excessive uniform bound rejects a model after its first value/gradient allocations. The tests also reject a nonempty output owner and overflow in the layer-dependent parameter count.

```c title="tests/test_model.c"
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
```

The different-seed test changes only the configuration of a newly constructed model; it does not reseed or mutate an existing one. Equal initial values are tested byte-for-byte **within the same build**. This checks reproducibility of this program, not cross-platform identity of all floating-point library operations.

## Reconstruct both uses of the table independently

After a successful backward, the tape exposes `dlogits`, `final_norm`, and `embedding_grad`. These are enough to reconstruct the table derivative directly with scalar loops. The output-path dot product is rounded once to float, matching the matrix kernel. Each repeated-ID lookup contribution is then added and rounded in row order, matching the embedding scatter.

This check covers **every table coordinate**, including rows never selected as inputs. Such rows still receive output-projection gradients because their vocabulary logits participate in the loss. If the last scatter uses SET, the unselected rows immediately expose the erased output contribution.

```c title="tests/test_model.c"
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
```

The reconstructed derivative is independent of the model's scheduling code but intentionally uses the saved intermediate derivatives. It localizes a tying mistake; it cannot by itself establish that `embedding_grad` was correctly produced by the blocks. Numerical perturbations check that larger question next.

## Perturb selected coordinates in every parameter family

For one parameter coordinate $\theta_i$, the central difference is

$$
g_i^{\mathrm{num}}=
\frac{\mathcal L(\theta_i^+)-\mathcal L(\theta_i^-)}
{\theta_i^+-\theta_i^-},
$$

where $\theta_i^+$ and $\theta_i^-$ are the actual stored float values after perturbation. All other values are unchanged. The denominator uses their represented spacing, not an assumed exact $2h$.

The code first computes one complete analytic gradient. In each tensor it selects the largest-magnitude gradient coordinate and two separated coordinates. Selecting the strongest coordinate is deliberate: a broken branch should not escape because three arbitrarily chosen gradients happened to be close to zero. The fixture must produce a largest gradient exceeding $10^{-7}$ in **every** parameter tensor. This selection uses the analytic gradient only to choose probes, not to calculate the independent finite-difference losses.

Each coordinate is checked at all three requested float steps, `0.004`, `0.002`, and `0.001`. No “best of three” result hides a failed comparison. Tolerance is $3\times10^{-5}+0.015\max(|g^{\mathrm{num}}|,|g^{\mathrm{analytic}}|)$. Small gradients are limited mostly by the absolute term; larger gradients can show finite-step curvature error.

```c title="tests/test_model.c"
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
```

Calling `mark_changed` invalidates gradient **eligibility**, not the existing gradient bytes. The test can still read its saved analytic values while probing new forward losses. It must recompute a valid gradient before an update; no probe is allowed to reuse an old tape for backward.

The one-block case makes $11\times3\times3\times2=198$ comparisons; the two-block case makes $20\times3\times3\times2=360$. The final factor is the two projection backends. Together these are **558 coordinate checks**, not a claim to check every coordinate of the model.

A smaller step is not automatically more accurate. On the checked Linux build, adding a `0.0005` step produced approximately `0.000800656` for a two-block FFN scale derivative whose analytic value was `0.000860915`; forward float rounding was already significant relative to the tiny loss difference. The current three-step range passes without increasing the tolerance. This is a reason to inspect error versus step, not to repeatedly loosen a tolerance until an implementation passes.

## Change the whole model along a direction

Let $q$ be a deterministic random direction, normalized so $\sum_i q_i^2\approx1$. In exact arithmetic, the directional derivative is

$$
\left.\frac{d}{dh}\mathcal L(\theta+hq)\right|_{h=0}
=\sum_i\bar\theta_iq_i.
$$

Float storage perturbs the intended direction slightly. For each coordinate, construct stored endpoints $\theta_i^+$ and $\theta_i^-$. Compare the numerical quotient with the derivative projected onto their effective secant direction:

$$
\frac{\mathcal L(\theta^+)-\mathcal L(\theta^-)}{2h}
\quad\text{against}\quad
\sum_i\bar\theta_i\frac{\theta_i^+-\theta_i^-}{2h}.
$$

This adjustment accounts for represented endpoints; it does not remove nonlinear finite-step error or rounding inside forward. Three directions, three steps, two backends, and two model sizes produce **36 directional checks**. The tolerance is $2\times10^{-5}+0.01\max(|g^{\mathrm{num}}|,|g^{\mathrm{projected}}|)$.

```c title="tests/test_model.c"
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
```

All model coordinates are restored between directions. The direction generator is separate from model initialization, so adding a test draw cannot alter the model being checked. This scalar comparison is sensitive to missing paths across the model, but cancellation is possible; that is why it supplements, rather than replaces, per-family coordinate checks and the explicit tied-table reconstruction.

## Check accumulation and saved-state rules

The next function deliberately changes caller-side IDs and targets after forward. Backward must still succeed because the tape copied the valid original arrays. It then rejects a second backward on the consumed tape, adds a second batch mean, and compares every gradient with twice the original. A blocked-backend SET pass must replace that accumulated result and agree with the reference result.

It also marks a model generation changed without changing a numerical weight. Backward must reject the stale tape anyway: generation matching is a state rule, not a comparison of loss values. An ADD call after the invalidation must reject the invalid accumulation boundary. Clearing gradients restores a valid zero boundary without consuming the current forward.

```c title="tests/test_model.c"
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
```

Continue the **same function**. A NaN is placed at the end of the last gradient array. The update must inspect the entire proposed update before touching the first weight. After rejection, every parameter value is compared with its saved original.

Once the test restores that gradient cell, a step of `0.001` is applied. Every coordinate must equal the individually calculated float update exactly within this build; this catches duplicate directory updates. The model must reject both backward through a pre-update tape and a second update with now-ineligible gradients. Finally, a new forward must have lower loss on this fixture.

```c title="tests/test_model.c"
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
```

The final continuation checks failed forward and failed backward. An out-of-range target must leave the scalar sentinel unchanged and invalidate old tape readiness. A deliberately corrupted saved gate then forces a backward numerical failure after some parameter derivatives may already have been written. The update must refuse those gradients. A fresh forward followed by SET must recover normally.

The direct NaN writes below are test-only fault injection. Normal application code must not edit saved tape contents or fabricate gradient-valid flags. These faults are not left in the model after the function returns successfully.

```c title="tests/test_model.c"
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
```

## Preserve causality across the full graph

Changing the last input token of the first sequence may change that token's logits, but not earlier logits and not any other batch item's logits. The comparison uses a fixed number of layers and unchanged weights; it tests the complete graph's dependency structure, not merely the attention operator in isolation.

Then every parameter is set to zero. Every hidden value and logit is zero, so loss is $\log V$. All parameter gradients are also zero here: the final table and final scale are both zero, and neither route carries a nonzero derivative back into the graph. This degenerate check is useful precisely because it demonstrates that a plausible uniform loss does not imply a working learning signal. The real initializer starts scales at one and matrices at nonzero random values.

```c title="tests/test_model.c"
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
```

## Calculate a complete two-coordinate decoder by hand

The forward check uses $D=F=2$, vocabulary four, one block, two tokens, and epsilon $0.5$. The table rows are $(1,0)$, $(0,2)$, $(1,1)$, and $(-1,0)$; input IDs are 0 and 1, and targets are 1 and 2. Query and key matrices are zero, so allowed probabilities are exactly `[1]` at the first position and `[1/2,1/2]` at the second. RoPE cannot change zero queries or keys.

Value, attention-output, gate, up, and down matrices are identities. Both block normalization scales are ones. The final scale is $(1,2)$. The independent calculation uses scalar arithmetic and explicit float storage points, with no calls to the model's numerical operators.

For the first token, the first normalization returns $(1,0)$, attention returns $(1,0)$, and the middle residual is $(2,0)$. Its next normalized value is approximately $(1.264911,0)$. SwiGLU with identity matrices gives $(1.247793,0)$, so the block output is approximately $(3.247793,0)$.

For the second token, first normalization gives $(0,1.264911)$. Averaging the two allowed value rows gives $(0.5,0.632456)$. Adding its original embedding yields $(0.5,2.632456)$. After the second normalization and SwiGLU, the block output is approximately $(0.534322,3.964429)$. Final normalization with scale $(1,2)$ gives approximately $(0.183259,2.719400)$. Projecting with the shared table therefore gives logits near `[0.183259,5.438800,2.902659,-0.183259]`.

```c title="tests/test_model.c"
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
```

Unlike a derivative-consistency test, this fixture can reject a model whose forward and backward both implement the same wrong normalization placement. It also checks the independent loss calculation. Identity projection matrices do not test every possible weight orientation; the inherited non-square operator tests supply that separate coverage.

## Run both model sizes and release every owner

`run_case` allocates three flat test vectors through the existing tensor owner. They hold original values, analytic gradients, and a direction. These are diagnostic allocations, not part of the production tape or the arena count reported by the model.

The command-line entry point permits one optional configuration path. It actually loads the file, runs the hand fixture and ownership checks, then the configured model. A second model with two blocks and seed 73 is always included. The test's shape bounds remain explicit in `fixture_create`; do not feed this executable the full training configuration and expect it to act as a training CLI.

```c title="tests/test_model.c"
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
```

This completes `tests/test_model.c`. The `EXIT_FAILURE` returns preserve failures under debug, sanitizer, and optimized builds. The two global error counters are diagnostic outputs only; there are no global model, gradient, or random states.

## Integrate the new modules into all build modes

Replace the root Makefile with the following complete version. The new `rng.c` and `model.c` join the shared core; the new header dependencies and `test_model` executable join the existing targets. The config is intentionally not a compile-time dependency: it is read when the test runs, so changing it does not require recompilation.

```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2 -fno-lto
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/rng.c src/model.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/rng.h include/dl/model.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model

.PHONY: all debug sanitize release test clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare

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

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

clean:
	rm -rf build
```

From the implementation root on the native Mac toolchain:

```sh title="Terminal — cumulative checks and the model executable"
make clean
make test
./build/debug/test_model configs/tiny.conf
```

A compatible non-Apple environment can override the compiler, for example `make CC=clang test`. The new model requires only the already used C17 and mathematical-library features. `make test` runs all ten test executables in three build modes and retains the inherited kernel-comparison smoke invocation; it is not a whole-model performance benchmark. The [Clang sanitizer guide][asan] documents the diagnostic build mechanism.

The new executable's observed output with Clang 17 on x86_64 Linux was:

```text title="Observed Linux output — default tiny configuration"
hand logits row 1: [0.183259, 5.438800, 2.902659, -0.183259], loss=2.408078371
model: L=1 B=1 T=4, 11 tensors, 2992 values, 2688 tape floats
  plain step: 3.451081853 -> 3.426729256
model: L=2 B=2 T=3, 20 tensors, 5712 values, 5808 tape floats
  plain step: 2.951561302 -> 2.919813147
coordinate checks: 558, max absolute error 8.58711e-05
direction checks: 36, max absolute error 1.45447e-05
model checks passed
```

All three build modes passed without compiler warnings, as did the inherited tests. A Linux leak-enabled sanitizer run of the model executable reported no leaks, including the partial-construction failures. No native M4/macOS execution or model-speed measurement is implied. Printed final digits can vary with compatible floating-point libraries; the explicit tolerances and invariants, not copying the exact decimal output, define the numerical checks.

The absolute coordinate maximum is larger than the absolute tolerance alone because each comparison also includes its stated relative term. The reported maximum is not a tolerance normalized score. Preserve the per-comparison checks when interpreting summary numbers.

## Change the file and predict the memory result

Changing time from four to three leaves the parameter directory unchanged but changes the activation arena. For one block and the same batch/width, the formula in Part 2 predicts **1,968 floats**, not 2,688. Create a separate temporary config rather than editing the baseline:

```sh title="Terminal — a valid shape variation and a rejected setting"
tmp=$(mktemp -d)
sed 's/time=4/time=3/' configs/tiny.conf > "$tmp/three.conf"
./build/sanitize/test_model "$tmp/three.conf"
sed 's/width=16/width=12/' configs/tiny.conf > "$tmp/odd-head.conf"
if ./build/debug/test_model "$tmp/odd-head.conf"; then
    printf 'ERROR: expected the odd head width to be rejected\n' >&2
    rm -rf "$tmp"
    exit 1
fi
rm -rf "$tmp"
```

Width 12 is divisible by four query heads, but its head width is three, so the RoPE pairing requirement fails. The time-three variation passed the model checks and reported the predicted 1,968 arena floats. The loader also rejected missing and duplicate settings, negative dimensions, zero epsilon, non-finite theta, integer overflow, extra lines, embedded zero bytes, overlong lines, and a missing final newline in separate malformed-file checks.

## Break one connection at a time

Keep a clean copy of `src/model.c`, make one isolated change, rebuild the model test, observe the failure, and restore the original before trying another. These are diagnostic experiments, not changes to retain in the implementation.

**Erase the output-table path.** Change the final `dl_embedding_backward` call from `DL_ADD` to `DL_SET`. The complete tied-table check fails immediately on a vocabulary row with no input lookup contribution. In the checked build it observed zero where the output projection required approximately `-0.0410557`. This is why a table row absent from the input is a useful probe.

**Erase a residual identity.** In the second normalization backward call, change the destination mode for `t->du` from ADD to SET. The table reconstruction still uses whatever input gradient was produced, but the model coordinate derivative rejects that wrong gradient. One observed comparison was approximately `-0.840156` numerically versus `-0.799423` analytically.

**Visit blocks in forward order during backward.** Replace `layer=remaining-1` with `layer=m->config.layers-remaining` in the reverse loop. The one-block fixture cannot distinguish these schedules. The two-block fixture rejects it: the incoming derivative belongs to the last block, not to whichever block happens to have a compatible shape.

**Ignore parameter revisions.** Remove the revision comparison from backward's ready-state test. Numerical derivatives may still pass when the parameter bytes happen not to have changed, but the stale-tape state test fails because the call incorrectly succeeds. This catches a lifecycle defect rather than a calculus defect.

All four isolated variants failed their intended checks on the Linux sanitizer build without sanitizer memory or undefined-behavior reports. They demonstrate logical failures within valid allocations. A clean final build must pass after restoration; do not leave an intentionally broken variant in the normal test target.

## Diagnose the first broken boundary

Wrong hand logits with no derivative failure usually means forward and backward agree on the wrong graph. Inspect normalization placement, the two residual additions, final scale, and use of the table at directory entry zero. Wrong table reconstruction with otherwise finite derivatives points to the SET/ADD choice at the two tied uses or incorrect lookup IDs.

Failures only in two-block gradients point toward reverse order, overwritten saved state, or the `g`/`dx` view swap. Failures only on ADD point toward an invalid accumulation boundary or an internal path that accidentally uses the external mode instead of always adding. A nearly uniform factor of batch/token count points toward repeated loss averaging. Errors growing at the smallest finite-difference step call for inspecting float loss differences before changing the derivative.

A successful forward followed by `DL_STATE` in backward is not a numerical mystery: check readiness, parameter revision, model identity, and ADD eligibility. After an actual backward numerical failure, assume **all** parameter gradients are invalid, even when only one block reported the error. A subsequent forward does not clear those bytes; use SET for a single-batch recomputation or clear and replay the intended accumulation sequence.

The decoder now computes a complete loss and all unique parameter gradients in C, with a small working update. This is not a corpus-training result. **Chapter 13 — Byte text, shifted targets, and bounded data preparation** will supply real input windows and separate training/validation data without changing the model's mathematical operators.

[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#usage

