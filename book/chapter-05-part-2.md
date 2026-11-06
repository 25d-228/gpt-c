# Chapter 5 — Checking gradients and numerical failures

**Part 2 — Tests for embeddings, stable loss, and elementary backward passes**

Begin with [Part 1](https://25d-228.github.io/gpt-c/chapter-05.html), including the extended `kernels.h` and `kernels.c`, new loss module, and complete `tests/tiny.h`. Keep Chapter 4's common/tensor modules and `tests/test_kernels.c`. This part supplies every remaining test and build rule for the same Chapter 5 program; it does not assume a future model or optimizer.

The graph uses separate embedding and projection arrays, input IDs `[0,1,0]`, and next-token targets `[1,0,2]`. Its objective is a natural-log cross-entropy averaged over three prediction rows. Gradient buffers obey the `DL_SET`/`DL_ADD` contract: a fresh result does not read old cells, and an added contribution does not silently clear existing ones.

> **Learning objectives**
> Detect a lost repeated-token contribution; distinguish overflow from cancellation; verify that invalid requests do not mutate output; compare manual gradients with stored-value finite differences; and restore parameter state on every trial path.

**Implementation allocation:** the 1.25-hour checking/integration portion of Chapter 5's **3.50-hour total**. Reading is separate; no additional chapter hours are assigned to this second page.

## Test numerical meaning as well as memory safety

Create `tests/test_loss.c`. Its small helpers fail explicitly rather than rely on `assert`, which an optimized build can disable. `near` checks finiteness before tolerances, retaining Chapter 3's NaN protection. `guards` classifies offsets using each view's physical pitch; only nonlogical cells must retain the sentinel.

```c title="tests/test_loss.c"
#include "tiny.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static void status_is(DlStatus got, DlStatus want, const char *message)
{
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %s, expected %s\n", message,
            dl_status_name(got), dl_status_name(want));
        exit(EXIT_FAILURE);
    }
}

static void near(double got, double want, double atol, double rtol,
                 const char *message)
{
    if (!isfinite(got) || !isfinite(want) ||
        fabs(got - want) > atol + rtol * fmax(fabs(got), fabs(want))) {
        fprintf(stderr, "FAIL: %s: got %.17g, expected %.17g\n",
            message, got, want);
        exit(EXIT_FAILURE);
    }
}

static void fill(float *data, size_t count, float value)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = value;
    }
}

static void guards(DlMatrix m, float sentinel)
{
    for (size_t k = 0; k < m.count; ++k) {
        int logical = 0;
        for (size_t i = 0; i < m.rows; ++i) {
            const size_t start = m.offset + i * m.ld;
            if (k >= start && k - start < m.cols) {
                logical = 1;
            }
        }
        if (!logical) {
            require(m.base[k] == sentinel, "padding/guard changed");
        }
    }
}

```

### A known mean, poisoned outputs, and late failures

The next test uses padded views with different pitches for logits and gradients. Logical logits are zero; input padding is NaN. Output logical cells start as NaN, so a mistaken old-value read in `DL_SET` becomes visible. First check the exact binary fractions from the two-row, four-class example, then call `DL_ADD` on the same buffers and require exactly twice the gradient.

```c title="tests/test_loss.c"
static void test_uniform_and_contracts(void)
{
    float z[14], g[17];
    fill(z, 14, NAN);
    fill(g, 17, 91.0f);
    DlMatrix zm = {.base = z, .count = 14, .offset = 1,
        .rows = 2, .cols = 4, .ld = 6};
    DlMatrix gm = {.base = g, .count = 17, .offset = 2,
        .rows = 2, .cols = 4, .ld = 7};
    const size_t targets[2] = {0, 3};
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            z[zm.offset + i * zm.ld + j] = 0.0f;
            g[gm.offset + i * gm.ld + j] = NAN;
        }
    }
    double loss = -7.0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_SET), DL_OK, "uniform SET");
    near(loss, log(4.0), 1e-14, 1e-14, "uniform mean loss");
    for (size_t i = 0; i < 2; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < 4; ++j) {
            const double want = j == targets[i] ? -0.375 : 0.125;
            const double got = (double)g[gm.offset + i * gm.ld + j];
            near(got, want, 0.0, 0.0, "uniform gradient");
            sum += got;
        }
        near(sum, 0.0, 1e-7, 0.0, "row gradient sum");
    }
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_ADD), DL_OK, "uniform ADD");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            near((double)g[gm.offset + i * gm.ld + j],
                j == targets[i] ? -0.75 : 0.25, 0.0, 0.0, "doubled gradient");
        }
    }
    guards(gm, 91.0f);
```

Continue this same test with a valid first target and invalid second target. A snapshot taken with `memcpy` permits a byte-for-byte unchanged-output check with `memcmp`; both functions operate on specified byte counts, supplied here by `sizeof` of actual arrays. This is an output-preservation test, not a floating-point equality criterion for independently computed results.

The later cases exercise a wrong gradient shape, overlapping logits/gradient storage, null pointers, an invalid mode, empty rows, and a non-finite value in the last logical logit. Preflight must find that last error before writing the first row. A NaN in an ADD destination is also rejected, but value-only evaluation must not depend on any unused gradient array.

```c title="tests/test_loss.c"
    float saved[17];
    memcpy(saved, g, sizeof saved);
    loss = -7.0;
    const size_t bad[2] = {0, 4};
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), bad, 2,
        &loss, &gm, DL_SET), DL_RANGE, "late invalid target");
    require(loss == -7.0 && memcmp(saved, g, sizeof saved) == 0,
        "invalid target changed output");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 1,
        &loss, &gm, DL_SET), DL_SHAPE, "target count");
    DlMatrix wrong = gm;
    wrong.rows = 1;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &wrong, DL_SET), DL_SHAPE, "gradient shape");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &zm, DL_SET), DL_ALIAS, "loss output alias");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), NULL, 2,
        &loss, NULL, DL_SET), DL_ARGUMENT, "null targets");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        NULL, NULL, DL_SET), DL_ARGUMENT, "null loss");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, NULL, (DlWriteMode)99), DL_ARGUMENT, "invalid mode");
    DlMatrix empty = zm;
    empty.rows = 0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(empty), targets, 0,
        &loss, NULL, DL_SET), DL_ARGUMENT, "zero rows");
    const float nonfinite[3] = {NAN, INFINITY, -INFINITY};
    for (size_t k = 0; k < 3; ++k) {
        z[10] = nonfinite[k];
        status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
            &loss, &gm, DL_SET), DL_NUMERIC, "non-finite late logit");
        require(loss == -7.0 && memcmp(saved, g, sizeof saved) == 0,
            "preflight failure changed output");
    }
    z[10] = 0.0f;
    g[12] = NAN;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_ADD), DL_NUMERIC, "ADD requires finite old gradient");
    require(loss == -7.0 && g[2] == saved[2], "ADD preflight wrote early row");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, NULL, DL_SET), DL_OK, "value-only loss");
    near(loss, log(4.0), 1e-14, 1e-14, "value-only equals training loss");
    printf("uniform: loss = %.9f; target gradient = -0.375\n", loss);
}

```

### Separate overflow, cancellation, and saturation

The next function targets different failure mechanisms rather than assuming one “large input” covers them all. A shared shift checks the exponential scale. Four `FLT_MAX` logits check the order of target subtraction. Opposite finite extremes require a double loss. `[0,-50]` checks a tiny positive loss **and** its tiny negative target derivative. A single-class vocabulary has exactly zero loss and derivative.

The tiny-gradient comparison deliberately has zero absolute tolerance and a relative allowance: an absolute tolerance of `1e-6` would incorrectly accept zero in place of a value around `1e-22`. Tolerances must expose the failure being tested.

```c title="tests/test_loss.c"
static void test_extreme_logits(void)
{
    size_t target[1] = {2};
    float z[3] = {1000.0f, 999.0f, 998.0f}, g[3];
    DlMatrix zm = tiny_matrix(z, 1, 3);
    DlMatrix gm = tiny_matrix(g, 1, 3);
    double loss = -1.0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), target, 1,
        &loss, &gm, DL_SET), DL_OK, "shifted large logits");
    const double expected = 2.0 + log1p(exp(-1.0) + exp(-2.0));
    near(loss, expected, 1e-14, 1e-14, "large-logit loss");
    const double original = loss;
    for (size_t i = 0; i < 3; ++i) {
        z[i] -= 1000.0f;
    }
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), target, 1,
        &loss, NULL, DL_SET), DL_OK, "shift invariance");
    near(loss, original, 1e-14, 1e-14, "common shift");
    float huge[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    target[0] = 0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(tiny_matrix(huge, 1, 4)),
        target, 1, &loss, NULL, DL_SET), DL_OK, "huge tied maximum");
    near(loss, log(4.0), 1e-14, 1e-14, "do not add maximum then subtract");
    float extreme[2] = {FLT_MAX, -FLT_MAX}, eg[2];
    DlMatrix em = tiny_matrix(extreme, 1, 2);
    DlMatrix egm = tiny_matrix(eg, 1, 2);
    target[0] = 1;
    status_is(dl_cross_entropy_mean(dl_matrix_read(em), target, 1,
        &loss, &egm, DL_SET), DL_OK, "opposite finite extremes");
    near(loss, 2.0 * (double)FLT_MAX, 0.0, 1e-15, "wide target subtraction");
    require(loss > (double)FLT_MAX && eg[0] == 1.0f && eg[1] == -1.0f,
        "extreme loss needs double, gradient does not overflow");
    extreme[0] = 0.0f;
    extreme[1] = -50.0f;
    target[0] = 0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(em), target, 1,
        &loss, &egm, DL_SET), DL_OK, "confident finite row");
    const double q = exp(-50.0);
    near(loss, log1p(q), 0.0, 1e-12, "tiny positive loss");
    near((double)eg[0], -q / (1.0 + q), 0.0, 2e-7, "tiny target derivative");
    near((double)eg[1], q / (1.0 + q), 0.0, 2e-7, "tiny other derivative");
    require(loss > 0.0 && eg[0] < 0.0f && eg[1] > 0.0f,
        "confidence must not silently erase representable derivatives");
    printf("confident: loss = %.9e; dtarget = %.9e\n", loss, (double)eg[0]);
    float single[1] = {3000.0f}, sg[1] = {NAN};
    DlMatrix sm = tiny_matrix(sg, 1, 1);
    status_is(dl_cross_entropy_mean(dl_matrix_read(tiny_matrix(single, 1, 1)),
        target, 1, &loss, &sm, DL_SET), DL_OK, "one class");
    require(loss == 0.0 && sg[0] == 0.0f, "one-class value and derivative");
    puts("extreme logits: shift, tied maxima, wide loss, one class passed");
}

```

### Make repeated IDs collide at a known destination

The gather/scatter fixture uses ID 2 twice, a table pitch of three, and output pitch of four. The first pass proves that unused rows become zero under SET; the second starts every gradient cell at 10 and proves that ADD preserves those earlier contributions. Padding remains 91 throughout. The input `dy` and output table have different shapes and pitches, so replacing one leading dimension with another is not harmless.

```c title="tests/test_loss.c"
static void test_embedding(void)
{
    float table[14], out[14], dt[14], dy[6] = {1, 2, 3, 4, 5, 6};
    fill(table, 14, 91.0f);
    fill(out, 14, 91.0f);
    fill(dt, 14, 91.0f);
    DlMatrix tm = {.base = table, .count = 14, .offset = 1,
        .rows = 4, .cols = 2, .ld = 3};
    DlMatrix om = {.base = out, .count = 14, .offset = 2,
        .rows = 3, .cols = 2, .ld = 4};
    DlMatrix dm = tm;
    dm.base = dt;
    const DlConstMatrix dyr = dl_matrix_read(tiny_matrix(dy, 3, 2));
    const size_t ids[3] = {2, 1, 2};
    for (size_t v = 0; v < 4; ++v) {
        for (size_t j = 0; j < 2; ++j) {
            table[1 + v * 3 + j] = (float)(1 + 2 * v + j);
            dt[1 + v * 3 + j] = NAN;
        }
    }
    status_is(dl_embedding_forward(dl_matrix_read(tm), ids, 3, om),
        DL_OK, "padded gather");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            near((double)out[2 + i * 4 + j], (double)(1 + 2 * ids[i] + j),
                0.0, 0.0, "gathered value");
        }
    }
    guards(om, 91.0f);
    status_is(dl_embedding_backward(dyr, ids, 3, dm, DL_SET), DL_OK, "scatter SET");
    const double expected[8] = {0, 0, 3, 4, 6, 8, 0, 0};
    for (size_t v = 0; v < 4; ++v) {
        for (size_t j = 0; j < 2; ++j) {
            near((double)dt[1 + v * 3 + j], expected[v * 2 + j],
                0.0, 0.0, "repeated-ID sum");
            dt[1 + v * 3 + j] = 10.0f;
        }
    }
    status_is(dl_embedding_backward(dyr, ids, 3, dm, DL_ADD), DL_OK, "scatter ADD");
    for (size_t v = 0; v < 4; ++v) {
        for (size_t j = 0; j < 2; ++j) {
            near((double)dt[1 + v * 3 + j], 10.0 + expected[v * 2 + j],
                0.0, 0.0, "preserved prior contribution");
        }
    }
    guards(dm, 91.0f);
```

Continue the test by asking for an invalid final ID in SET mode. The entire table must remain unchanged; zeroing before ID validation would violate that contract. Next reject self-overlap, which otherwise could clear the incoming gradient before it is read. Finally, two visits carrying `FLT_MAX` deliberately overflow their combined float destination. This last case must report a numeric failure **after** some writes, matching—not contradicting—the documented boundary.

```c title="tests/test_loss.c"
    float saved[14], saved_out[14];
    memcpy(saved, dt, sizeof saved);
    memcpy(saved_out, out, sizeof saved_out);
    const size_t bad[3] = {2, 1, 4};
    status_is(dl_embedding_backward(dyr, bad, 3, dm, DL_SET), DL_RANGE, "late bad ID");
    require(memcmp(saved, dt, sizeof saved) == 0, "bad ID cleared table");
    status_is(dl_embedding_forward(dl_matrix_read(tm), bad, 3, om),
        DL_RANGE, "gather late bad ID");
    require(memcmp(saved_out, out, sizeof saved_out) == 0, "bad ID copied rows");
    status_is(dl_embedding_backward(dyr, ids, 2, dm, DL_SET), DL_SHAPE, "ID count");
    status_is(dl_embedding_backward(dyr, NULL, 3, dm, DL_SET), DL_ARGUMENT, "null IDs");
    status_is(dl_embedding_backward(dyr, ids, 3, dm, (DlWriteMode)99),
        DL_ARGUMENT, "embedding mode");
    DlMatrix alias = dm;
    alias.rows = 3;
    status_is(dl_embedding_backward(dl_matrix_read(alias), ids, 3, dm, DL_SET),
        DL_ALIAS, "scatter cannot clear input");
    DlMatrix gather_alias = tm;
    gather_alias.rows = 3;
    status_is(dl_embedding_forward(dl_matrix_read(tm), ids, 3, gather_alias),
        DL_ALIAS, "gather cannot overwrite table");
    DlMatrix narrow = dm;
    narrow.cols = 1;
    status_is(dl_embedding_backward(dyr, ids, 3, narrow, DL_SET), DL_SHAPE,
        "embedding feature width");
    require(memcmp(saved, dt, sizeof saved) == 0, "structural error wrote table");
    dy[0] = FLT_MAX;
    dy[4] = FLT_MAX;
    status_is(dl_embedding_backward(dyr, ids, 3, dm, DL_SET),
        DL_NUMERIC, "repeated-ID overflow");
    require(dt[7] == FLT_MAX && dt[1] == 0.0f,
        "numeric error follows documented partial writes");
    puts("embedding: padded gather, repeated scatter, ADD, rejection, overflow passed");
}

```

### Check the connected graph against the hand calculation

The final fixture compares every coordinate of $G_H$, $G_E$, and $G_W$ against the derived values. It then applies both projection backward functions in ADD mode and checks doubled results. Notice that the embedding gradient is not recomputed from the now-doubled $G_H$; its original hand check has already completed.

```c title="tests/test_loss.c"
static void test_tiny_graph(void)
{
    Tiny t;
    tiny_init(&t);
    fill(t.dz, 12, NAN);
    fill(t.dh, 6, NAN);
    fill(t.de, 8, NAN);
    fill(t.dw, 8, NAN);
    double loss = -1.0;
    status_is(tiny_eval(&t, &loss, 1), DL_OK, "tiny graph");
    const double a = log(2.0);
    near(loss, (log(3.0) + 2.0 * log(6.0)) / 3.0, 2e-8, 2e-8, "tiny loss");
    const double dh[6] = {-a/9.0, a/6.0, a/6.0, 2.0*a/9.0,
        2.0*a/9.0, -a/6.0};
    const double de[8] = {a/9.0, 0.0, a/6.0, 2.0*a/9.0, 0, 0, 0, 0};
    const double dw[8] = {1.0/9.0, -5.0/18.0, -1.0/9.0, 1.0/18.0,
        -2.0/9.0, 1.0/9.0, 2.0/9.0, 1.0/9.0};
    for (size_t k = 0; k < 8; ++k) {
        near((double)t.de[k], de[k], 2e-8, 2e-7, "tiny dE");
        near((double)t.dw[k], dw[k], 2e-8, 2e-7, "tiny dW");
    }
    for (size_t k = 0; k < 6; ++k) {
        near((double)t.dh[k], dh[k], 2e-8, 2e-7, "tiny dH");
    }
    const DlConstMatrix dz = dl_matrix_read(tiny_matrix(t.dz, 3, 4));
    status_is(dl_linear_backward_input(dz, dl_matrix_read(tiny_matrix(t.w, 4, 2)),
        tiny_matrix(t.dh, 3, 2), DL_ADD), DL_OK, "input gradient ADD");
    status_is(dl_linear_backward_weight(dz, dl_matrix_read(tiny_matrix(t.h, 3, 2)),
        tiny_matrix(t.dw, 4, 2), DL_ADD), DL_OK, "weight gradient ADD");
    for (size_t k = 0; k < 6; ++k) {
        near((double)t.dh[k], 2.0 * dh[k], 4e-8, 2e-7, "input gradient doubled");
    }
    for (size_t k = 0; k < 8; ++k) {
        near((double)t.dw[k], 2.0 * dw[k], 4e-8, 2e-7, "weight gradient doubled");
    }
    printf("tiny graph: loss = %.9f; dE[0] = [%.9f, %.9f]\n",
        loss, (double)t.de[0], (double)t.de[1]);
}

int main(void)
{
    test_uniform_and_contracts();
    test_extreme_logits();
    test_embedding();
    test_tiny_graph();
    puts("loss and elementary backward checks passed");
    return EXIT_SUCCESS;
}
```

These tests complement the existing `test_kernels.c`, which already covers all four transpose combinations, non-square/padded layouts, guard values, and reference-product failures. Do not delete that earlier test just because a connected example now passes. Conversely, correct memory bounds do not prove that a derivative uses the right normalization or accumulates all paths.

## Check parameters by changing the values the program reads

A hand-derived result is valuable but covers one fixture. We also approximate a parameter derivative by running the actual forward graph twice. For parameter $\theta$, choose a scale $h=\eta\max(1,|\theta|)$, with positive relative step $\eta$, and store float endpoints:

$$
\theta_- = \operatorname{float}(\theta-h),\qquad
\theta_+ = \operatorname{float}(\theta+h),\qquad
 g_{\mathrm{num}}=\frac{L(\theta_+)-L(\theta_-)}{\theta_+-\theta_-}.
$$

The denominator uses the values actually stored, widened to double before subtraction, not the requested `2*h`. Endpoint rounding can make the two distances slightly unequal. This is a secant over the represented endpoints, close to a centered difference at the original parameter when those distances are nearly symmetric; it is not a magic cure for forward-evaluation rounding.

The analytical gradients describe the underlying real-valued operations evaluated at stored parameter values. We do not attempt to differentiate the discontinuous rounding-to-float map. A step that is too small can yield unchanged intermediate floats even when the parameter itself changed. This is why finite differences are approximate diagnostics over a useful step range, not bitwise ground truth.

For ordinary fixture derivatives, compare using:

$$
|g_{\mathrm{analytic}}-g_{\mathrm{num}}|
\le 3\times10^{-5}+2\times10^{-3}
\max(|g_{\mathrm{analytic}}|,|g_{\mathrm{num}}|).
$$

The absolute term covers near-zero coordinates; the relative term scales with nonzero gradients. These are fixture tolerances, not defaults for all future operators. A factor-of-three mean-reduction mistake or a missing repeated-ID contribution is much larger. Print the failing array name, coordinate, step, values, and tolerance so the check locates an error instead of saying only “gradient failed.”

Create `tests/gradcheck.c` with these endpoint and comparison helpers. Range checks occur before converting double endpoints to float. We reject a step that does not straddle the original stored value rather than silently report a derivative of zero.

```c title="tests/gradcheck.c"
#include "tiny.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Float endpoints are the values actually written into model storage. */
static int endpoints(float original, double relative_step,
                     float *minus, float *plus)
{
    if (!isfinite(original) || !isfinite(relative_step) || relative_step <= 0.0) {
        return 0;
    }
    const double step = relative_step * fmax(1.0, fabs((double)original));
    const double low = (double)original - step;
    const double high = (double)original + step;
    if (!isfinite(low) || !isfinite(high) ||
        fabs(low) > (double)FLT_MAX || fabs(high) > (double)FLT_MAX) {
        return 0;
    }
    *minus = (float)low;
    *plus = (float)high;
    return *minus < original && original < *plus;
}

static int compare(const char *name, size_t index, double step,
                    double analytic, double numerical, double *worst)
{
    const double tolerance = 3e-5 + 2e-3 * fmax(fabs(analytic), fabs(numerical));
    const double ratio = fabs(analytic - numerical) / tolerance;
    if (!isfinite(analytic) || !isfinite(numerical) || ratio > 1.0) {
        fprintf(stderr, "FAIL %s[%zu] step=%g: analytic=%.9g numerical=%.9g "
            "tolerance=%.3g\n", name, index, step, analytic, numerical, tolerance);
        return 0;
    }
    if (ratio > *worst) {
        *worst = ratio;
    }
    return 1;
}

```

For each coordinate of an eight-element parameter array, modify the actual array used by `tiny_eval`. Save both forward statuses, restore the original float, and only then return on any error. Value-only evaluation does not touch the analytical gradient arrays, so they remain the baseline derivatives throughout the sweep. The working hidden/logit arrays do change; the final baseline rerun will rebuild them.

```c title="tests/gradcheck.c"
static int check_parameters(Tiny *t, const char *name, float *parameter,
                             const float *gradient, double step)
{
    double worst = 0.0;
    for (size_t k = 0; k < 8; ++k) {
        const float original = parameter[k];
        float minus = 0.0f, plus = 0.0f;
        if (!endpoints(original, step, &minus, &plus)) {
            fprintf(stderr, "FAIL %s[%zu]: step has no two finite endpoints\n", name, k);
            return 0;
        }
        double lp = 0.0, lm = 0.0;
        parameter[k] = plus;
        const DlStatus sp = tiny_eval(t, &lp, 0);
        parameter[k] = minus;
        const DlStatus sm = tiny_eval(t, &lm, 0);
        parameter[k] = original; /* Restore before any status/accuracy return. */
        if (sp != DL_OK || sm != DL_OK) {
            fprintf(stderr, "FAIL %s[%zu]: evaluations %s / %s\n", name, k,
                dl_status_name(sp), dl_status_name(sm));
            return 0;
        }
        const double numerical = (lp - lm) / ((double)plus - (double)minus);
        if (!compare(name, k, step, (double)gradient[k], numerical, &worst)) {
            return 0;
        }
    }
    printf("%s: 8 coordinates, step=%g, worst tolerance fraction=%.4g\n",
        name, step, worst);
    return 1;
}

```

The `worst tolerance fraction` is the largest error divided by its allowed tolerance; a value below one passes. It is not a probability or a percent-accuracy score. The checker tests **all** coordinates, including unused embedding rows whose analytical derivatives should be zero.

First isolate the loss, perturbing its eight logits while holding targets fixed. This distinguishes a loss-derivative error from a projection or scatter error. The loss remains a C computation; no Python tensor library supplies these values.

```c title="tests/gradcheck.c"
static int check_direct_loss(void)
{
    float z[8] = {0.2f, -0.7f, 1.1f, 0.3f, -0.6f, 0.5f, 0.8f, -0.1f};
    float gradient[8], original_array[8];
    memcpy(original_array, z, sizeof z);
    DlMatrix zm = tiny_matrix(z, 2, 4);
    DlMatrix gm = tiny_matrix(gradient, 2, 4);
    const size_t targets[2] = {2, 0};
    double loss = 0.0, worst = 0.0;
    DlStatus status = dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_SET);
    if (status != DL_OK) {
        fprintf(stderr, "FAIL direct loss: %s\n", dl_status_name(status));
        return 0;
    }
    for (size_t k = 0; k < 8; ++k) {
        const float original = z[k];
        float minus = 0.0f, plus = 0.0f;
        if (!endpoints(original, 0.003, &minus, &plus)) {
            return 0;
        }
        double lp = 0.0, lm = 0.0;
        z[k] = plus;
        const DlStatus sp = dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
            &lp, NULL, DL_SET);
        z[k] = minus;
        const DlStatus sm = dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
            &lm, NULL, DL_SET);
        z[k] = original;
        if (sp != DL_OK || sm != DL_OK) {
            fprintf(stderr, "FAIL direct loss coordinate %zu\n", k);
            return 0;
        }
        const double numerical = (lp - lm) / ((double)plus - (double)minus);
        if (!compare("logits", k, 0.003, (double)gradient[k], numerical, &worst)) {
            return 0;
        }
    }
    if (memcmp(original_array, z, sizeof z) != 0) {
        fputs("FAIL direct loss did not restore storage\n", stderr);
        return 0;
    }
    printf("logits: 8 coordinates, worst tolerance fraction=%.4g\n", worst);
    return 1;
}

```

Then test the whole graph twice: once with the hand-calculated fixture, and once with asymmetric decimal weights that do not preserve its convenient zeros. Each graph checks both eight-element parameter arrays at three step scales: `0.01`, `0.003`, and `0.001`. That is $2\times2\times8\times3=96$ parameter comparisons, plus eight direct-loss comparisons.

`memcpy` preserves the original parameter arrays. After all trials, the code checks their exact stored bytes and recomputes the original loss. Equality here is intentional: the same serial code, inputs, and execution settings are rerun, not compared with another implementation. `tiny_eval` must not secretly accumulate state during a value-only call.

```c title="tests/gradcheck.c"
static int check_graph(int varied)
{
    Tiny t;
    tiny_init(&t);
    if (varied) {
        const float e[8] = {0.7f, -0.2f, 0.3f, 0.9f, -0.4f, 0.6f, 0.1f, -0.8f};
        const float w[8] = {0.2f, -0.1f, -0.3f, 0.4f, 0.5f, 0.2f, -0.2f, -0.4f};
        memcpy(t.e, e, sizeof e);
        memcpy(t.w, w, sizeof w);
    }
    float e_before[8], w_before[8];
    memcpy(e_before, t.e, sizeof t.e);
    memcpy(w_before, t.w, sizeof t.w);
    double baseline = 0.0;
    DlStatus status = tiny_eval(&t, &baseline, 1);
    if (status != DL_OK) {
        fprintf(stderr, "FAIL baseline graph: %s\n", dl_status_name(status));
        return 0;
    }
    printf("graph fixture %d: baseline loss=%.9f\n", varied, baseline);
    const double steps[3] = {0.01, 0.003, 0.001};
    for (size_t k = 0; k < 3; ++k) {
        if (!check_parameters(&t, "E", t.e, t.de, steps[k]) ||
            !check_parameters(&t, "W", t.w, t.dw, steps[k])) {
            return 0;
        }
    }
    double restored_loss = 0.0;
    status = tiny_eval(&t, &restored_loss, 0);
    if (status != DL_OK || memcmp(e_before, t.e, sizeof t.e) != 0 ||
        memcmp(w_before, t.w, sizeof t.w) != 0 || restored_loss != baseline) {
        fputs("FAIL restored graph differs from baseline\n", stderr);
        return 0;
    }
    return 1;
}

int main(void)
{
    if (!check_direct_loss() || !check_graph(0) || !check_graph(1)) {
        return EXIT_FAILURE;
    }
    puts("104 finite-difference comparisons passed; parameter storage restored");
    return EXIT_SUCCESS;
}
```

A failure at one small step is not permission to relax every tolerance until the test turns green. First inspect the coordinate and compare several steps. Systematic errors that stay the same as the step changes point toward mathematics or indexing. Errors that grow only at the smallest steps can come from float rounding in the forward path. The asymmetric fixture prevents relying exclusively on cancellations that happen to make one bad expression look correct.

## Build all three programs without losing the old test

Replace the root Makefile with the following. The default compiler is still native `xcrun clang`; another compatible C17 compiler can be selected explicitly with `CC=...`. No external numerical library is added. `<math.h>` and the platform math implementation provide scalar `exp`, `log1p`, and the other standard functions; the table lookup, projection, reduction, and backward computation remain our own C loops.

The pattern rule `build/debug/%: tests/%.c` substitutes the same name into both paths. `$<` is Make's first prerequisite, the driver source; `$@` is the output target. `$(addprefix ...)` forms three executable paths. In the test recipe, `$$mode` and `$$name` pass single-dollar variables to the shell rather than asking Make to expand them. `set -e` stops on the first failing executable. Keep literal tab characters before recipe commands.

```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck

.PHONY: all debug sanitize release test clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS))
sanitize: $(addprefix build/sanitize/,$(PROGRAMS))
release: $(addprefix build/release/,$(PROGRAMS))

build/debug/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	done

clean:
	rm -rf build
```

The executable names now match their source stems with underscores: Chapter 4's former `test-kernels` becomes `test_kernels`. Run `make clean` once during this transition so an old binary cannot be mistaken for a new check. The old source file itself is unchanged.

From the learner workspace root:

```sh
make clean
make test
./build/debug/test_loss
./build/debug/gradcheck
```

A non-Apple build with Clang uses:

```sh
make CC=clang test
```

`make test` runs the old kernel test, new loss/backward test, and gradient checker in debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized configurations. It does not time them. On a sanitizer platform that supports leak detection, a separate `ASAN_OPTIONS=detect_leaks=1` run can exercise it; the new routines themselves allocate no heap storage.

A representative `test_loss` run prints:

```text
uniform: loss = 1.386294361; target gradient = -0.375
confident: loss = 1.928749848e-22; dtarget = -1.928749893e-22
extreme logits: shift, tied maxima, wide loss, one class passed
embedding: padded gather, repeated scatter, ADD, rejection, overflow passed
tiny graph: loss = 1.560710410; dE[0] = [0.077016354, -0.000000007]
loss and elementary backward checks passed
```

The near-zero final coordinate is not an omitted scatter contribution; it comes from float approximations and is checked against zero with an explicit tolerance. The handwritten exact loss differs in the last printed decimal because the stored `log(2)` is a float.

The gradient checker reports each array and step, followed by:

```text
104 finite-difference comparisons passed; parameter storage restored
```

In the checked Clang 17 x86_64 Linux builds, the largest tolerance fraction across these fixtures was approximately `0.0354`, below one. Debug, sanitizer, and optimized checks passed there. These are not M4/macOS observations, and small differences in math-library results may change the printed fractions. The tests' mathematical expectations and tolerances, rather than exact diagnostic decimals, determine success. No throughput or model-training result is being claimed by this operator check.

## Change one mechanism and predict the failure

Make experiments in a disposable workspace copy, not in the only working kernel. Rebuild the changed executable rather than run a stale binary. Restore each change before the next experiment.

### Turn scatter-add into last-write-wins

In `dl_embedding_backward`, replace the sum expression with only the incoming value:

```c
const double sum = (double)dy.base[dy.offset + i * dy.ld + j];
```

Run `make sanitize` and `./build/sanitize/test_loss`. The repeated-ID fixture should report `5` instead of `6` for the first coordinate of table row 2. This is not an out-of-bounds access, so AddressSanitizer is not expected to diagnose it. The arithmetic check does. Run `./build/sanitize/gradcheck` separately: the whole-graph parameter check should also expose the missing occurrence.

### Differentiate the sum while reporting the mean

Restore scatter. In `dl_cross_entropy_mean`, change only the gradient assignment to:

```c
double value = numerator / denominator;
```

Leave `total += row_loss / rows` unchanged. For the two-row uniform fixture, the target derivative becomes `-0.75` instead of `-0.375`. The direct-loss gradient check also fails because it differentiates the reported scalar mean. This experiment shows why a numerically plausible scalar loss cannot certify a backward pass. Do not fix it by dividing again in projection or scatter: that would couple unrelated operators to a hidden loss convention.

### Round away a confident loss

Restore the mean derivative. Replace `log1p(other)` with `log(1.0 + other)` in the row loss. The confident fixture should return zero instead of a tiny positive value. The ordinary finite-difference fixtures can still pass; they were not designed to probe this scale. Keep both boundary tests and moderate-scale gradient checks.

For a separate prediction, replace the fixture's IDs `[0,1,0]` with `[0,1,2]` and keep its targets unchanged. Recompute the expected forward/gradient values before changing the hand test. There is now no second lookup contribution to row 0, while row 2 is no longer unused. A passing gradient checker alone will not preserve the old hand-calculated numbers because the function has genuinely changed.

## Diagnose the first incorrect layer

| Symptom | Likely cause | Focused check |
|---|---|---|
| Correct loss but every gradient differs by $M$ | Summed versus mean objective mismatch | Direct loss finite differences before any projection |
| Only repeated IDs fail | Assignment or clearing inside the scatter loop | IDs `[2,1,2]`, incoming `[1,2]` and `[5,6]` |
| Unused embedding rows are nonzero after SET | Table not fully cleared, or stale accumulation | Rows 2 and 3 in the tiny graph |
| Large logits overflow | Exponential before shifting, or float subtraction before widening | `[1000,999,998]` and opposite `FLT_MAX` values |
| Four huge equal logits have loss zero | Maximum added before target subtraction | Expect `log(4)` regardless of common finite value |
| A confident loss or target derivative becomes zero too early | `log(1+r)` or `p_target-1` cancellation | `[0,-50]` with relative checks for tiny values |
| Weight gradients have transposed shape | Stored weight convention changed | Require `[N,K]`, compute `G^T X` |
| ADD gives an unexpected multiple | Caller reused a partially accumulated gradient | Check initialization and each contribution once |
| Finite differences drift after the first coordinate | Parameter not restored or forward has hidden state | Original arrays and baseline loss after the sweep |

The finished module boundary now accepts token IDs, evaluates a stable mean objective, and returns derivatives to both independent parameter arrays. The next chapter changes matrix-product execution order and measures it; it must preserve the function, layout, and accumulation rules established here.

## References and prerequisite code

The numerical definitions and primary-source references are collected in [Part 1](https://25d-228.github.io/gpt-c/chapter-05.html#references). The runnable state requires both parts and the named Chapter 4 modules, not any external tensor framework. Companion sources include [test_loss.c](https://25d-228.github.io/gpt-c/media/chapter-05/reference/tests/test_loss.c), [gradcheck.c](https://25d-228.github.io/gpt-c/media/chapter-05/reference/tests/gradcheck.c), [tiny.h](https://25d-228.github.io/gpt-c/media/chapter-05/reference/tests/tiny.h), and the [Makefile](https://25d-228.github.io/gpt-c/media/chapter-05/reference/Makefile).
