# Chapter 11 — Checking attention gradients and accumulation

A single numerical derivative test is not enough to locate a shared-head bug. The final input gradient combines query, key, and value paths; a wrong sum can be obscured by another path's larger magnitude. We will check intermediate derivatives, independent local objectives, the complete sublayer, and the accumulation rules separately.

**Prerequisites:** [Part 1](https://25d-228.github.io/gpt-c/chapter-11.html), including the replaced `include/dl/attention.h` and appended `src/attention_ref.c`, plus the unchanged Chapter 10 workspace. This is the second half of the same **4.50-hour** chapter, not additional implementation hours.

Create `tests/test_attention_backward.c` from **all blocks with that filename below**, in order with one blank line between blocks. Keep Chapter 10's `tests/test_attention.c` unchanged: independent forward correctness remains necessary even when numerical differentiation agrees with backward. Replace the root Makefile with the complete listing near the end. The [completed companion snapshot](https://25d-228.github.io/gpt-c/media/chapter-11/reference/Makefile) contains the entire cumulative workspace, not just these additions.

## Make incorrect addresses visible without enormous test tensors

Use small asymmetric shapes, nonzero offsets, and physical row padding. Query and key/value widths differ in shared-head cases. Input width differs from query width in most cases. This makes an accidental transpose or wrong leading dimension less likely to produce a plausible result.

The fixture owns one allocation per matrix. Its views borrow each owner's canonical base/count; they are not separately allocated rows. `NARRAY` enumerates the input, weights, forward cache, output, incoming derivative, eight gradient scratch arrays, input derivative, and four parameter derivatives. `CAP` bounds saved copies in the tests; it is **not** a limit in the attention interface.

Start with the includes, fixture type, counters, diagnostics, and logical-cell helpers:

```c title="tests/test_attention_backward.c"
#include "dl/attention.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { X, WQ, WK, WV, WO, Q0, K0, V, Q, K, SCORES, PROBS, CONTEXT, Y,
       DY, DC, DP, DS, DQ, DK, DV, DQ0, DK0, DX, DWQ, DWK, DWV, DWO, NARRAY };
enum { CAP = 512 };

typedef struct {
    DlAttentionConfig c;
    DlTensor owner[NARRAY];
    DlMatrix m[NARRAY];
    size_t *positions;
    double *freq;
} Fixture;

static size_t full_checks = 0, core_checks = 0;
static double full_error = 0.0, core_error = 0.0;

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

static int near_value(double got, double want, double tol, const char *where)
{
    if (isfinite(got) && isfinite(want) && fabs(got - want) <= tol) return 1;
    fprintf(stderr, "%s: %.12g; expected %.12g, tolerance %.3g\n", where, got, want, tol);
    return 0;
}

static float *cell(DlMatrix m, size_t i, size_t j)
{
    return &m.base[m.offset + i * m.ld + j];
}

static void fill(DlMatrix m, float value)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) *cell(m, i, j) = value;
    }
}
```

`near_value` rejects nonfinite values before checking a tolerance, avoiding the NaN-comparison trap from Chapter 3. `cell` uses a view's physical leading dimension. The fixture constructor, not `cell`, establishes bounds; this helper is intentionally only for already-validated test shapes.

The constructor accepts at most two sequences, four tokens, four query heads, head width four, and input width four. Under these explicit caps its small size calculations are bounded; the production API instead uses the checked layout arithmetic from Chapter 10. Padding contains `-777`; outputs begin as NaN. An uninitialized `SET` read or accidental write into padding becomes observable.

```c title="tests/test_attention_backward.c"
static void fixture_destroy(Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) dl_tensor_destroy(&f->owner[a]);
    free(f->positions);
    free(f->freq);
    *f = (Fixture){0};
}

/* Small bounded TEST fixtures, not library limits. f starts as {0}. */
static int fixture_create(Fixture *f, DlAttentionConfig c, size_t width, size_t pad)
{
    if (c.batch == 0 || c.batch > 2 || c.time == 0 || c.time > 4 ||
        c.query_heads == 0 || c.query_heads > 4 || c.kv_heads == 0 ||
        c.query_heads % c.kv_heads != 0 || c.head_dim == 0 ||
        c.head_dim > 4 || c.head_dim % 2 != 0 || width == 0 || width > 4 || pad > 2)
        return 0;
    f->c = c;
    const size_t m = c.batch * c.time, q = c.query_heads * c.head_dim;
    const size_t k = c.kv_heads * c.head_dim, sr = m * c.query_heads, d = width;
    const size_t rows[NARRAY] = {m,q,k,k,d,m,m,m,m,m,sr,sr,m,m,
                                 m,m,sr,sr,m,m,m,m,m,m,q,k,k,d};
    const size_t cols[NARRAY] = {d,d,d,d,q,q,k,k,q,k,c.time,c.time,q,d,
                                 d,q,c.time,c.time,q,k,k,q,k,d,d,d,d,q};
    for (size_t a = 0; a < NARRAY; ++a) {
        const size_t ld = cols[a] + pad, off = pad != 0 ? 1 : 0;
        const size_t count = rows[a] * ld + (pad != 0 ? 2 : 0);
        if (count > CAP) goto fail;
        if (dl_tensor_create(&f->owner[a], 1, &count) != DL_OK) goto fail;
        if (dl_matrix_view(&f->owner[a], off, rows[a], cols[a], ld, &f->m[a]) != DL_OK)
            goto fail;
        for (size_t j = 0; j < count; ++j) f->owner[a].data[j] = -777.0f;
        for (size_t i = 0; i < rows[a]; ++i) {
            for (size_t j = 0; j < cols[a]; ++j) {
                *cell(f->m[a], i, j) = a < Q0 || a == DY
                    ? (float)((int)((11*i + 7*j + 3*a) % 17) - 8) / 16.0f : NAN;
            }
        }
    }
    f->positions = malloc(m * sizeof(*f->positions));
    f->freq = malloc((c.head_dim / 2) * sizeof(*f->freq));
    if (f->positions == NULL || f->freq == NULL) goto fail;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t t = 0; t < c.time; ++t) f->positions[b * c.time + t] = 3 + b + 2*t;
    }
    if (dl_rope_frequencies(c.head_dim, 100000.0, f->freq, c.head_dim / 2) != DL_OK)
        goto fail;
    return 1;
fail:
    fixture_destroy(f);
    return 0;
}
```

Failure jumps to one cleanup path. Destroying empty owners is allowed; after cleanup the fixture is reset to its empty state. Do not copy this owning struct while it is live. The deterministic input formula generates both signs and avoids relying on an external random library or dataset.

Positions vary by batch and token; ordinary gradient cases therefore exercise the existing rotation transpose rather than only position zero. The four-coordinate head case also exercises two different frequencies. The hand calculation will explicitly replace positions with zero to isolate its arithmetic.

## Connect every borrowed view to the public calls

These helpers only assemble arguments; they do not implement any model computation. Forward writes the original cache and $Y$. Backward borrows those saved views, reads `DY`, writes the eight separate scratch views, and applies the two requested accumulation modes. The parameter order is query, key, value, output throughout.

```c title="tests/test_attention_backward.c"
static DlAttentionWeights weights(const Fixture *f)
{
    return (DlAttentionWeights){dl_matrix_read(f->m[WQ]), dl_matrix_read(f->m[WK]),
        dl_matrix_read(f->m[WV]), dl_matrix_read(f->m[WO])};
}

static DlAttentionCache cache(const Fixture *f)
{
    return (DlAttentionCache){f->m[Q0], f->m[K0], f->m[V], f->m[Q], f->m[K],
                              f->m[SCORES], f->m[PROBS], f->m[CONTEXT]};
}

static DlAttentionScratch scratch(const Fixture *f)
{
    return (DlAttentionScratch){f->m[DC], f->m[DP], f->m[DS], f->m[DQ], f->m[DK],
                                f->m[DV], f->m[DQ0], f->m[DK0]};
}

static DlStatus forward(Fixture *f, DlBackend backend)
{
    return dl_attention_forward(backend, f->c, dl_matrix_read(f->m[X]), weights(f),
        f->positions, f->c.batch * f->c.time, f->freq, f->c.head_dim / 2,
        cache(f), f->m[Y]);
}

static DlStatus backward(Fixture *f, DlBackend backend, DlWriteMode xm, DlWriteMode wm)
{
    const DlAttentionWeightGrads dw = {f->m[DWQ], f->m[DWK], f->m[DWV], f->m[DWO]};
    return dl_attention_backward(backend, f->c, dl_matrix_read(f->m[X]), weights(f),
        f->positions, f->c.batch * f->c.time, f->freq, f->c.head_dim / 2,
        cache(f), dl_matrix_read(f->m[DY]), scratch(f), f->m[DX], xm, dw, wm);
}
```

The next helpers inspect guards and exact zero results, or save allocation bytes for unchanged-state checks. Saving the whole owner, not only logical cells, detects writes into padding. The saved arrays are local test copies of **float data**, not copies of owning tensors. `memcmp` is appropriate here because the promise being checked is “no byte was written,” including the existing NaN initialization; it is not a numerical approximate-equality test.

```c title="tests/test_attention_backward.c"
static int guards(const Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) {
        const DlMatrix m = f->m[a];
        for (size_t j = 0; j < m.count; ++j) {
            const int logical = j >= m.offset && (j-m.offset)/m.ld < m.rows
                             && (j-m.offset)%m.ld < m.cols;
            if (!logical && m.base[j] != -777.0f) return 0;
        }
    }
    return 1;
}

static int matrix_zero(DlMatrix m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            if (*cell(m, i, j) != 0.0f) return 0;
        }
    }
    return 1;
}

static void save_arrays(const Fixture *f, float saved[NARRAY][CAP])
{
    for (size_t a = 0; a < NARRAY; ++a)
        memcpy(saved[a], f->owner[a].data, f->owner[a].count * sizeof(float));
}

static int same_arrays(const Fixture *f, float saved[NARRAY][CAP], size_t first, size_t end)
{
    for (size_t a = first; a < end; ++a) {
        if (memcmp(saved[a], f->owner[a].data, f->owner[a].count * sizeof(float)) != 0)
            return 0;
    }
    return 1;
}
```

`same_arrays` receives an exclusive end index. The interval `[X,DC)` contains all original inputs, forward state, $Y$, and incoming `DY`. Backward must not modify any of them. `[DC,DX)` contains only scratch. Comparing that interval across repeated calls establishes that scratch is overwritten consistently, independent of the final gradient modes.

## Check the mask and a complete hand derivative

The score-gradient row sum should be approximately zero because a common score shift leaves the probabilities unchanged. Masked `DP` and `DS` are exact zeros by our storage convention. A one-key score derivative is exactly zero on these finite fixtures: its saved probability is exactly one.

```c title="tests/test_attention_backward.c"
static int mask_invariants(const Fixture *f)
{
    const DlAttentionConfig c = f->c;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                double sum = 0.0, magnitude = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    const double ds = (double)*cell(f->m[DS], sr, s);
                    if (s > t && !require(ds == 0.0 && *cell(f->m[DP], sr, s) == 0.0f,
                                          "masked derivative not zero")) return 0;
                    sum += ds;
                    magnitude += fabs(ds);
                }
                if (!near_value(sum, 0.0, 3e-7 * (1.0 + magnitude), "score row sum")) return 0;
                if (t == 0 && !near_value(*cell(f->m[DS], sr, 0), 0.0, 0.0,
                                         "one-key score derivative")) return 0;
            }
        }
    }
    return 1;
}
```

Now construct the two-token, two-query/one-key-value-head calculation from Part 1. The objective uses both coordinates of the final output. It gives different upstream vectors to the two query heads, so losing either head is visible. Checking only identical upstream vectors would make some sharing mistakes harder to diagnose.

For $a=\log2$, $u=a/(9\sqrt2)$, the expected score rows are $(2/9,-2/9)$ and $(-4/9,4/9)$; the shared key derivatives are $(2/9,-4/9)$ and $(-2/9,4/9)$. The test checks those arrays, the values, queries, input derivative, and all four weight derivatives. Because $X$ is the identity, weight orientation can be checked directly rather than inferred from a final scalar.

```c title="tests/test_attention_backward.c"
static int hand_trace(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){1,2,2,1,2}, 2, 2)) return 0;
    for (size_t a = X; a <= WO; ++a) fill(f.m[a], 0.0f);
    fill(f.m[DY], 0.0f);
    const float r = (float)sqrt(2.0), a = (float)log(2.0);
    *cell(f.m[X], 0, 0) = 1.0f; *cell(f.m[X], 1, 1) = 1.0f;
    *cell(f.m[WQ], 0, 1) = r; *cell(f.m[WQ], 3, 1) = r;
    *cell(f.m[WK], 0, 1) = a; *cell(f.m[WK], 1, 0) = a;
    *cell(f.m[WV], 0, 0) = 1.0f; *cell(f.m[WV], 1, 1) = 2.0f;
    *cell(f.m[WO], 0, 0) = 1.0f; *cell(f.m[WO], 1, 3) = 1.0f;
    *cell(f.m[DY], 1, 0) = 1.0f; *cell(f.m[DY], 1, 1) = 1.0f;
    f.positions[0] = 0; f.positions[1] = 0;
    int ok = status_is(forward(&f, backend), DL_OK, "hand forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "hand backward");
    const double want_ds[4] = {2.0/9, -2.0/9, -4.0/9, 4.0/9};
    const double want_dk[4] = {2.0/9, -4.0/9, -2.0/9, 4.0/9};
    const double want_dv[4] = {1.0/3, 2.0/3, 2.0/3, 1.0/3};
    const double u = log(2.0) / (9.0 * sqrt(2.0));
    const double want_dq[4] = {-2*u, 2*u, 4*u, -4*u};
    const double want_dx[4] = {1.0/3 - 4*log(2.0)/9, 4.0/3 + 2*log(2.0)/9,
                               2.0/3 + 4*log(2.0)/9, 2.0/3 - 8*log(2.0)/9};
    const double want_context[4] = {1.0/3, 4.0/3, 2.0/3, 2.0/3};
    for (size_t i = 0; ok && i < 2; ++i) {
        for (size_t j = 0; ok && j < 2; ++j) {
            ok = near_value(*cell(f.m[DS], i*2+1, j), want_ds[2*i+j], 2e-7, "hand dS") &&
                 near_value(*cell(f.m[DK], i, j), want_dk[2*i+j], 2e-7, "hand shared dK") &&
                 near_value(*cell(f.m[DV], i, j), want_dv[2*i+j], 2e-7, "hand shared dV") &&
                 near_value(*cell(f.m[DX], i, j), want_dx[2*i+j], 3e-7, "hand dX") &&
                 near_value(*cell(f.m[DWK], j, i), want_dk[2*i+j], 2e-7, "hand dWk") &&
                 near_value(*cell(f.m[DWV], j, i), want_dv[2*i+j], 2e-7, "hand dWv");
        }
    }
    for (size_t j = 0; ok && j < 4; ++j) {
        ok = near_value(*cell(f.m[DQ], 1, j), want_dq[j], 2e-7, "hand dQ") &&
             near_value(*cell(f.m[DWQ], j, 1), want_dq[j], 2e-7, "hand dWq") &&
             near_value(*cell(f.m[DQ], 0, j), 0.0, 0.0, "hand unused query") &&
             near_value(*cell(f.m[DWQ], j, 0), 0.0, 0.0, "hand unused query weight");
        for (size_t i = 0; ok && i < 2; ++i)
            ok = near_value(*cell(f.m[DWO], i, j), want_context[j], 2e-7, "hand dWo");
    }
    ok = ok && mask_invariants(&f) && require(guards(&f), "hand guard modified");
    if (ok) printf("hand shared dK[0] (%s): [%.6f, %.6f]\n",
        backend == DL_BACKEND_REF ? "reference" : "blocked",
        (double)*cell(f.m[DK], 0, 0), (double)*cell(f.m[DK], 0, 1));
    fixture_destroy(&f);
    return ok;
}
```

The arrays named `want_...` use independent algebraic values rather than a second call to backward. Position zero removes rotation rounding from this hand example, but the weights still contain float approximations to $\sqrt2$ and $\log2$, so nonzero tolerances remain appropriate.

A first-row query gradient of zero must not be generalized to the first key/value gradient. The latter receives contributions from the later output. This is the backward version of causal information flow: future inputs cannot influence earlier outputs, but earlier inputs can influence later outputs.

## Define two objectives that test different parts of the graph

For the complete sublayer use

$$
L_{\mathrm{full}}(X,W_q,W_k,W_v,W_o)
 =\sum_{r,i}Y_{r,i}\,\mathrm{DY}_{r,i}.
$$

`DY` stays fixed while one input or weight coordinate is perturbed. Its gradient at $Y$ is exactly the supplied `DY`, with no extra averaging. This check differentiates the complete forward call, including both rotations and all projections.

```c title="tests/test_attention_backward.c"
/* Scalar linear objective: its gradient at Y is exactly the supplied DY. */
static double full_objective(Fixture *f, DlBackend backend)
{
    if (forward(f, backend) != DL_OK) return NAN;
    double sum = 0.0;
    for (size_t i = 0; i < f->m[Y].rows; ++i) {
        for (size_t j = 0; j < f->m[Y].cols; ++j)
            sum += (double)*cell(f->m[Y], i, j) * (double)*cell(f->m[DY], i, j);
    }
    return sum;
}
```

The isolated objective instead treats **rotated** $Q,K,V$ as its independent inputs and holds `DC` fixed. It computes attention anew without calling the production attention, normalization, matrix, or rotation routines, and returns $L_{\mathrm{core}}=\sum C\,\mathrm{DC}$.

A correct full input gradient could conceal offsetting path errors; this local objective compares each of $\bar Q$, $\bar K$, and $\bar V$ independently. It deliberately reproduces the forward's score, probability, and context float stores rather than silently changing to an all-double function. It is an independent local numerical evaluation, not an autograd implementation and not a proof against every possible shared convention error. The hand fixture and Chapter 10's independent forward reference address that additional risk.

```c title="tests/test_attention_backward.c"
/* Independent local objective on ROTATED Q/K/V, holding dC fixed.
 * No production attention/softmax call. Match the forward's float boundaries.
 * These tiny fixtures have T<=4; array capacities are checked at construction. */
static double core_objective(const Fixture *f)
{
    const DlAttentionConfig c = f->c;
    double loss = 0.0;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / (c.query_heads / c.kv_heads);
            for (size_t t = 0; t < c.time; ++t) {
                float score[4] = {0}, prob[4] = {0};
                double maximum = -INFINITY, total = 0.0;
                for (size_t s = 0; s <= t; ++s) {
                    double dot = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j)
                        dot += (double)*cell(f->m[Q], b*c.time+t, h*c.head_dim+j)
                             * (double)*cell(f->m[K], b*c.time+s, kh*c.head_dim+j);
                    score[s] = (float)(dot / sqrt((double)c.head_dim));
                    maximum = fmax(maximum, (double)score[s]);
                }
                for (size_t s = 0; s <= t; ++s) total += exp((double)score[s] - maximum);
                for (size_t s = 0; s <= t; ++s)
                    prob[s] = (float)(exp((double)score[s] - maximum) / total);
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double sum = 0.0;
                    for (size_t s = 0; s <= t; ++s)
                        sum += (double)prob[s] * (double)*cell(f->m[V], b*c.time+s, kh*c.head_dim+j);
                    const float context = (float)sum;
                    loss += (double)context * (double)*cell(f->m[DC], b*c.time+t, h*c.head_dim+j);
                }
            }
        }
    }
    return loss;
}
```

The arrays of four scores/probabilities are safe because the fixture constructor enforces $T\le4$. No forward production limit is inferred from those test arrays. Only allowed keys enter maximum subtraction, exponentiation, normalization, or the weighted value sum.

## Perturb represented values at three step sizes

For one float input coordinate $x$, choose $\epsilon=h\max(1,|x|)$, round $x+\epsilon$ and $x-\epsilon$ to floats, and call the objective at those represented values. Denote them $x_+$ and $x_-$. The numerical derivative is

$$
g_{\mathrm{num}}=\frac{L(x_+)-L(x_-)}{x_+-x_-}.
$$

Using the represented denominator avoids pretending that float storage accepted an exact $2\epsilon$ spacing. It does **not** remove roundoff inside forward. Very small steps amplify that error; large steps introduce truncation error. We check every coordinate at $h=0.01$, $0.005$, and $0.0025$, not only whichever step happens to look best.

The acceptance rule is

$$
|g_{\mathrm{analytic}}-g_{\mathrm{num}}|
 \le 2\times10^{-5}+0.002\max(|g_{\mathrm{analytic}}|,|g_{\mathrm{num}}|).
$$

The absolute term covers small-gradient finite-difference noise; the relative term scales with larger derivatives. This tolerance is for these bounded, moderate-valued float fixtures. It is not a universal training tolerance, nor permission to ignore a failed hand calculation. The observed worst absolute discrepancies below are much smaller than that absolute floor.

```c title="tests/test_attention_backward.c"
/* Every coordinate is checked at EVERY step; do not select only the best one. */
static int differences(Fixture *f, DlBackend backend, int core)
{
    const size_t inputs[5] = {core ? Q : X, core ? K : WQ, core ? V : WK, WV, WO};
    const size_t grads[5] = {core ? DQ : DX, core ? DK : DWQ, core ? DV : DWK, DWV, DWO};
    const double steps[3] = {0.01, 0.005, 0.0025};
    const size_t n = core ? 3 : 5;
    for (size_t a = 0; a < n; ++a) {
        const DlMatrix v = f->m[inputs[a]], g = f->m[grads[a]];
        for (size_t i = 0; i < v.rows; ++i) {
            for (size_t j = 0; j < v.cols; ++j) {
                float *p = cell(v, i, j);
                const float old = *p;
                const double analytic = (double)*cell(g, i, j);
                for (size_t step = 0; step < 3; ++step) {
                    const double eps = steps[step] * fmax(1.0, fabs((double)old));
                    const float plus = (float)((double)old + eps), minus = (float)((double)old - eps);
                    *p = plus;
                    const double high = core ? core_objective(f) : full_objective(f, backend);
                    *p = minus;
                    const double low = core ? core_objective(f) : full_objective(f, backend);
                    *p = old;
                    const double span = (double)plus - (double)minus;
                    const double numeric = (high - low) / span;
                    const double error = fabs(analytic - numeric);
                    const double tol = 2e-5 + 0.002 * fmax(fabs(analytic), fabs(numeric));
                    if (!(span > 0.0) || !isfinite(numeric) || !isfinite(analytic) || error > tol) {
                        fprintf(stderr, "%s gradient array %zu [%zu,%zu], h %.4g: %.9g vs %.9g (tol %.3g)\n",
                            core ? "core" : "full", inputs[a], i, j, steps[step], analytic, numeric, tol);
                        return 0;
                    }
                    if (core) { ++core_checks; core_error = fmax(core_error, error); }
                    else { ++full_checks; full_error = fmax(full_error, error); }
                }
            }
        }
    }
    return 1;
}
```

`inputs` and `grads` choose matching arrays. The conditional `core` selects three isolated inputs or five full-sublayer inputs; the formulas themselves are unchanged. Both perturbed values are evaluated before restoring the original coordinate. Even when a comparison fails, that coordinate has already been restored.

A more subtle issue remains: the last complete forward call used the **minus** perturbation and overwrote its cache. Restoring a weight value does not restore that cache. Before reusing the tape or switching to the isolated check, rerun the baseline forward and backward. The driver makes that step explicit:

```c title="tests/test_attention_backward.c"
static int gradient_case(DlBackend backend, DlAttentionConfig c, size_t d, size_t pad)
{
    Fixture f = {0};
    if (!fixture_create(&f, c, d, pad)) return 0;
    int ok = status_is(forward(&f, backend), DL_OK, "gradient forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "gradient backward") &&
             mask_invariants(&f) && differences(&f, backend, 0);
    /* Last perturbed forward left a stale tape. Restore the baseline before reuse. */
    ok = ok && status_is(forward(&f, backend), DL_OK, "restore baseline") &&
         status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "restore derivatives") &&
         differences(&f, backend, 1) && require(guards(&f), "gradient padding modified");
    fixture_destroy(&f);
    return ok;
}
```

The main program uses four configurations with both projection backends. The table lists dimensions in this chapter's notation; the code's configuration order is batch, time, query heads, key/value heads, head width.

| Case | $(B,T,H_q,H_{kv},d)$ | $D$ | Full independent coordinates | Isolated $Q/K/V$ coordinates |
|---|---|---:|---:|---:|
| Grouped heads, two sequences | $(2,3,4,2,2)$ | 3 | 90 | 96 |
| Multi-query, two rotary planes | $(1,3,2,1,4)$ | 3 | 81 | 48 |
| Multi-head, no sharing | $(1,2,2,2,2)$ | 3 | 54 | 24 |
| One-token sequence | $(1,1,2,1,2)$ | 2 | 26 | 8 |

There are $251\times3\times2=1506$ full comparisons and $176\times3\times2=1056$ isolated comparisons. Changing to the blocked backend affects projection products, not the scalar attention loops. Exercising both checks their integration while preserving that distinction.

## Test independent modes and repeated accumulation

A correct fresh derivative can still be unusable in a model if a second call erases earlier contributions. Save a baseline `SET/SET` result, then exercise all four combinations of input and weight modes. For `ADD`, seed input derivatives with $0.25$ and parameter derivatives with $-0.5$. For `SET`, leave old cells as NaN.

After one call, an `ADD` result should equal seed plus one derivative. After two identical calls, it should equal seed plus twice that derivative. A `SET` result remains one derivative on both calls. Scratch must always match the baseline scratch, and the entire forward state must remain untouched.

```c title="tests/test_attention_backward.c"
static int accumulation(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2,3,4,2,2}, 3, 2)) return 0;
    float before[NARRAY][CAP], baseline[NARRAY][CAP];
    int ok = status_is(forward(&f, backend), DL_OK, "accumulation forward");
    save_arrays(&f, before);
    ok = ok && status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "baseline backward") &&
         require(same_arrays(&f, before, X, DC), "backward changed read-only arrays");
    save_arrays(&f, baseline);
    for (int xm = 0; ok && xm < 2; ++xm) {
        for (int wm = 0; ok && wm < 2; ++wm) {
            for (size_t a = DC; a < NARRAY; ++a) fill(f.m[a], NAN);
            if (xm != 0) fill(f.m[DX], 0.25f);
            if (wm != 0) {
                for (size_t a = DWQ; a <= DWO; ++a) fill(f.m[a], -0.5f);
            }
            for (size_t rep = 1; ok && rep <= 2; ++rep) {
                ok = status_is(backward(&f, backend, (DlWriteMode)xm, (DlWriteMode)wm),
                                DL_OK, "mode/repeated backward");
                for (size_t a = DX; ok && a < NARRAY; ++a) {
                    const int add = a == DX ? xm : wm;
                    const double seed = a == DX ? 0.25 : -0.5;
                    const DlMatrix m = f.m[a];
                    for (size_t i = 0; ok && i < m.rows; ++i) {
                        for (size_t j = 0; ok && j < m.cols; ++j) {
                            const size_t index = m.offset + i*m.ld + j;
                            const double want = (add ? seed : 0.0)
                                              + (add ? (double)rep : 1.0) * (double)baseline[a][index];
                            ok = near_value(*cell(m, i, j), want, 3e-7, "SET/ADD or repeated contribution");
                        }
                    }
                }
                ok = ok && require(same_arrays(&f, baseline, DC, DX), "scratch must overwrite") &&
                     require(same_arrays(&f, before, X, DC), "saved input or tape changed");
            }
        }
    }
    ok = ok && require(guards(&f), "accumulation padding changed");
    fixture_destroy(&f);
    return ok;
}
```

The input derivative is assembled through three matrix products, each with its own float store. Adding a seed and repeating the call can change the last bits compared with adding to an already-rounded final baseline. Therefore final gradient comparisons use a small tolerance. Scratch has identical inputs and traversal on every call, so that particular comparison is byte-for-byte within the same executable.

This test preserves a caller-supplied activation-gradient seed, the mechanism needed for an existing residual contribution. It does not yet assemble a complete decoder block; that integration follows in Chapter 12. No backward routine here silently introduces a residual or normalization operation.

## Compare shared parameters with matched expanded heads

To test sharing independently, construct a second fixture with one key/value head per query head. Copy $X,W_q,W_o$, and `DY`. For each query head, copy its shared key/value weight rows into the corresponding separate rows of the expanded fixture. Positions and frequencies are identical because both constructors use the same batch/time/head-width metadata.

The forward functions now represent the same effective keys and values. Outputs, input derivatives, query derivatives, and query/output weight gradients should agree. The expanded key/value gradients need **not** be equal to one another: their query heads can use those identical keys differently. Sum the expanded derivatives within each group before comparing with the shared derivative.

These small copying and comparison helpers operate only on fixture shapes established below:

```c title="tests/test_attention_backward.c"
static void copy_matrix(DlMatrix dst, DlMatrix src)
{
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) *cell(dst, i, j) = *cell(src, i, j);
    }
}

static int same_matrix(DlMatrix a, DlMatrix b, const char *where)
{
    if (a.rows != b.rows || a.cols != b.cols) return 0;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            if (!near_value(*cell(a, i, j), *cell(b, i, j), 3e-7, where)) return 0;
        }
    }
    return 1;
}
```

For weight gradients, the head dimension is in the **row** axis. For rotated key/value and unrotated key gradients, it is in the **column** axis. The reduction code distinguishes those layouts rather than applying one flattening formula to both. `DK0` is included to check that summing before the rotation transpose agrees, within rounding, with summing separately rotated expanded contributions.

```c title="tests/test_attention_backward.c"
static int matched_heads(DlBackend backend, size_t kv)
{
    Fixture f = {0}, expanded = {0};
    const DlAttentionConfig c = {2,3,4,kv,2};
    if (!fixture_create(&f, c, 3, 1)) return 0;
    DlAttentionConfig ec = c;
    ec.kv_heads = c.query_heads;
    if (!fixture_create(&expanded, ec, 3, 2)) { fixture_destroy(&f); return 0; }
    const size_t shared[] = {X, WQ, WO, DY};
    for (size_t a = 0; a < 4; ++a) copy_matrix(expanded.m[shared[a]], f.m[shared[a]]);
    const size_t group = c.query_heads / c.kv_heads;
    for (size_t a = WK; a <= WV; ++a) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t j = 0; j < c.head_dim; ++j) {
                for (size_t col = 0; col < f.m[a].cols; ++col)
                    *cell(expanded.m[a], h*c.head_dim+j, col) =
                        *cell(f.m[a], (h/group)*c.head_dim+j, col);
            }
        }
    }
    int ok = status_is(forward(&f, backend), DL_OK, "shared forward") &&
             status_is(forward(&expanded, backend), DL_OK, "expanded forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "shared backward") &&
             status_is(backward(&expanded, backend, DL_SET, DL_SET), DL_OK, "expanded backward");
    const size_t equal[] = {Y, DX, DWQ, DWO, DQ};
    for (size_t a = 0; ok && a < 5; ++a)
        ok = same_matrix(f.m[equal[a]], expanded.m[equal[a]], "matched shared/nonshared result");
    const size_t reduce[] = {DK, DV, DK0, DWK, DWV};
    for (size_t a = 0; ok && a < 5; ++a) {
        const size_t which = reduce[a];
        for (size_t kh = 0; ok && kh < c.kv_heads; ++kh) {
            for (size_t j = 0; ok && j < c.head_dim; ++j) {
                const size_t n = which >= DWQ ? f.m[which].cols : f.m[which].rows;
                for (size_t other = 0; ok && other < n; ++other) {
                    double sum = 0.0;
                    for (size_t gh = 0; gh < group; ++gh) {
                        const size_t col = (kh*group+gh)*c.head_dim+j;
                        sum += which >= DWQ ? (double)*cell(expanded.m[which], col, other)
                                            : (double)*cell(expanded.m[which], other, col);
                    }
                    const size_t col = kh*c.head_dim+j;
                    const double got = which >= DWQ ? (double)*cell(f.m[which], col, other)
                                                   : (double)*cell(f.m[which], other, col);
                    ok = near_value(got, sum, 3e-7, "shared K/V gradient is SUM of expanded heads");
                }
            }
        }
    }
    ok = ok && require(guards(&f) && guards(&expanded), "sharing guard changed");
    fixture_destroy(&f);
    fixture_destroy(&expanded);
    return ok;
}
```

With four query heads, the driver tests one, two, and four key/value heads: MQA, an intermediate GQA case, and MHA. Each uses two independent sequences and both projection backends, giving six matched-head comparisons. These are matched **effective tensors**, not assertions that differently initialized architectures must produce the same answer.

## Check dependencies that scalar losses can hide

Place all incoming output gradient at the first token of batch one. No query in that objective can read future tokens or batch zero. All score/query/key paths vanish because the one-key distribution is constant; the value path remains. Check zero input gradients at every other row rather than merely checking the final loss.

Then set all query projection weights to zero and move the objective to a later output. The keys receive zero derivative because each score is a dot product with a zero query. Queries can still receive a nonzero derivative through nonzero keys. Finally use zero incoming gradient and require every `SET` derivative to become zero, replacing the previous nonzero result.

```c title="tests/test_attention_backward.c"
static int dependency_edges(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2,3,4,2,2}, 3, 2)) return 0;
    fill(f.m[DY], 0.0f);
    /* Only batch one's first output participates in this objective. */
    *cell(f.m[DY], 3, 0) = 1.0f;
    int ok = status_is(forward(&f, backend), DL_OK, "dependency forward") &&
             status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "dependency backward");
    const size_t score_paths[] = {DQ, DK, DQ0, DK0, DWQ, DWK, DS};
    for (size_t a = 0; ok && a < 7; ++a)
        ok = require(matrix_zero(f.m[score_paths[a]]), "one-key objective has a score path");
    for (size_t i = 0; ok && i < f.m[DX].rows; ++i) {
        if (i == 3) continue;
        for (size_t j = 0; ok && j < f.m[DX].cols; ++j)
            ok = near_value(*cell(f.m[DX], i, j), 0.0, 0.0, "batch/future input gradient leak");
    }
    ok = ok && require(!matrix_zero(f.m[DV]) && !matrix_zero(f.m[DWV]),
                        "one-key objective must retain value derivatives") && mask_invariants(&f);
    /* A zero query projection kills dWk, but dWq need not vanish. */
    fill(f.m[WQ], 0.0f);
    fill(f.m[DY], 0.0f);
    *cell(f.m[DY], 5, 0) = 1.0f;
    ok = ok && status_is(forward(&f, backend), DL_OK, "zero-query forward") &&
         status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "zero-query backward") &&
         require(matrix_zero(f.m[DWK]) && !matrix_zero(f.m[DWQ]), "zero-query path separation");
    fill(f.m[DY], 0.0f);
    ok = ok && status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "zero upstream");
    for (size_t a = DC; ok && a < NARRAY; ++a)
        ok = require(matrix_zero(f.m[a]), "zero upstream did not clear SET result");
    ok = ok && require(guards(&f), "dependency padding changed");
    fixture_destroy(&f);
    return ok;
}
```

A zero forward quantity does not automatically imply every derivative involving it is zero. For a product $xy$, at $x=0$ the derivative with respect to $y$ vanishes, but the derivative with respect to $x$ is still $y$. The zero-query test applies that elementary distinction to the score dot product.

## Reject malformed calls without touching any allocation

For a preflight rejection, save every allocation's bytes, invoke backward, and require both the expected status and unchanged storage. This is intentionally stronger than checking one destination sentinel. It also catches accidental writes into saved inputs during validation.

```c title="tests/test_attention_backward.c"
static int reject_unchanged(Fixture *f, DlBackend backend, DlWriteMode xm,
                             DlWriteMode wm, DlStatus want, const char *where)
{
    float before[NARRAY][CAP];
    save_arrays(f, before);
    return status_is(backward(f, backend, xm, wm), want, where) &&
           require(same_arrays(f, before, 0, NARRAY), "preflight failure changed storage");
}
```

The failure cases cover mode/backend values, input/output and scratch/scratch aliasing, shape mismatch, rotary metadata, nonfinite read inputs, malformed probability/mask state, and nonfinite old `ADD` cells. The modified views are restored before the owning fixture is destroyed; views never transfer ownership.

The last case is different: all inputs and the forward result are finite, but an incoming derivative of `FLT_MAX` multiplied by a weight of two overflows float storage. The first context-gradient cell is written before the second fails. This explicitly demonstrates the nontransactional arithmetic boundary rather than assuming every reported failure leaves outputs unchanged.

```c title="tests/test_attention_backward.c"
static int failures(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2,3,4,2,2}, 3, 2)) return 0;
    int ok = status_is(forward(&f, backend), DL_OK, "failure baseline");
    ok = ok && reject_unchanged(&f, (DlBackend)99, DL_SET, DL_SET, DL_ARGUMENT, "bad backend") &&
         reject_unchanged(&f, backend, (DlWriteMode)99, DL_SET, DL_ARGUMENT, "bad dx mode") &&
         reject_unchanged(&f, backend, DL_SET, (DlWriteMode)99, DL_ARGUMENT, "bad weight mode");
    const DlMatrix old_dx = f.m[DX], old_q = f.m[DQ], old_k0 = f.m[DK0];
    f.m[DX] = f.m[DY];
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ALIAS, "dx/dy alias");
    f.m[DX] = old_dx;
    f.m[DQ] = f.m[Q];
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ALIAS, "scratch/cache alias");
    f.m[DQ] = old_q;
    f.m[DK0] = f.m[DK];
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ALIAS, "scratch/scratch alias");
    f.m[DK0] = old_k0;
    --f.m[DX].rows;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_SHAPE, "dx shape");
    f.m[DX] = old_dx;
    const size_t old_pos = f.positions[0];
    f.positions[0] = SIZE_MAX;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_RANGE, "position out of range");
    f.positions[0] = old_pos;
    const double old_freq = f.freq[0];
    f.freq[0] = NAN;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_ARGUMENT, "bad frequency");
    f.freq[0] = old_freq;
    const size_t poison[] = {X, WQ, K, DY};
    for (size_t a = 0; ok && a < 4; ++a) {
        float *p = cell(f.m[poison[a]], 0, 0);
        const float old = *p;
        *p = NAN;
        ok = reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "nonfinite read input");
        *p = old;
    }
    *cell(f.m[PROBS], 0, 1) = 0.25f;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "nonzero masked probability");
    *cell(f.m[PROBS], 0, 1) = 0.0f;
    *cell(f.m[SCORES], 0, 1) = 0.0f;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "masked score sentinel");
    *cell(f.m[SCORES], 0, 1) = -INFINITY;
    *cell(f.m[PROBS], 0, 0) = 0.5f;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_SET, DL_NUMERIC, "probability row sum");
    *cell(f.m[PROBS], 0, 0) = 1.0f;
    ok = ok && reject_unchanged(&f, backend, DL_ADD, DL_SET, DL_NUMERIC, "NaN old dx for ADD");
    for (size_t a = DX; a < NARRAY; ++a) fill(f.m[a], 0.0f);
    *cell(f.m[DWV], 0, 0) = NAN;
    ok = ok && reject_unchanged(&f, backend, DL_SET, DL_ADD, DL_NUMERIC, "NaN old later weight for ADD");
    *cell(f.m[DWV], 0, 0) = 0.0f;
    ok = ok && status_is(backward(&f, backend, DL_SET, DL_SET), DL_OK, "valid masked negative infinity");
    fixture_destroy(&f);
    if (!ok || !fixture_create(&f, (DlAttentionConfig){1,1,2,1,2}, 2, 1)) return 0;
    fill(f.m[WO], 0.0f);
    *cell(f.m[WO], 0, 0) = 1.0f; *cell(f.m[WO], 0, 1) = 2.0f;
    fill(f.m[DY], 0.0f); *cell(f.m[DY], 0, 0) = FLT_MAX;
    ok = status_is(forward(&f, backend), DL_OK, "finite forward before derivative overflow") &&
         status_is(backward(&f, backend, DL_SET, DL_SET), DL_NUMERIC, "late dC overflow") &&
         require(*cell(f.m[DC], 0, 0) == FLT_MAX && isnan(*cell(f.m[DC], 0, 1)),
                 "expected partial dC, not a transactional failure") && require(guards(&f), "failure padding changed");
    fixture_destroy(&f);
    return ok;
}
```

The final partial-write expectation is specific to this one-row fixture and the two current matrix kernels. The interface does **not** promise an arbitrary failure's exact write subset. For all later numerical failures the portable response is to discard the entire result, including partially updated `ADD` destinations.

Normal masked negative infinities are accepted in an otherwise valid tape. This matters because a blanket finite-score scan would make every sequence longer than one fail backward. Conversely, arbitrary NaN or infinity is not accepted as a substitute for a mask.

The failure test uses `SIZE_MAX` as an out-of-range rotary position on the target 64-bit Mac and the tested 64-bit Linux host. The supported rotary cap is `UINT32_MAX`; this particular test assumption is not a claim of a separate 32-bit target port.

## Run every numerical case with both projection backends

Finish the test file with the driver. Each case is evaluated on the left side of `&& ok`, so a previous failure does not prevent the remaining top-level cases from reporting their own diagnostic. Within a case, dependent work stops when its required setup fails. The program returns a nonzero exit code if any case fails.

```c title="tests/test_attention_backward.c"
int main(void)
{
    int ok = 1;
    for (int b = 0; b < 2; ++b) {
        const DlBackend backend = (DlBackend)b;
        ok = hand_trace(backend) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){2,3,4,2,2}, 3, 2) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){1,3,2,1,4}, 3, 1) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){1,2,2,2,2}, 3, 0) && ok;
        ok = gradient_case(backend, (DlAttentionConfig){1,1,2,1,2}, 2, 1) && ok;
        ok = accumulation(backend) && ok;
        for (size_t kv = 1; kv <= 4; kv *= 2) ok = matched_heads(backend, kv) && ok;
        ok = dependency_edges(backend) && ok;
        ok = failures(backend) && ok;
    }
    if (!ok) return EXIT_FAILURE;
    printf("full attention gradients: %zu comparisons, max abs error %.6g\n", full_checks, full_error);
    printf("isolated Q/K/V gradients: %zu comparisons, max abs error %.6g\n", core_checks, core_error);
    puts("attention backward: hand, modes, shared sums, dependencies, masks, failures passed");
    return EXIT_SUCCESS;
}
```

Replace the root Makefile with the following complete file. The only functional change from Chapter 10 is adding `test_attention_backward` to `PROGRAMS`; all inherited test executables and the comparison smoke call remain. Make recipes begin with tabs.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward

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

From the learner workspace on the target Mac, run:

```sh title="Terminal — cumulative checks"
make test
```

The default compiler is still `xcrun clang`. For a compatible non-Apple compiler, choose it explicitly; the authoring checks used `make CC=clang test` on Linux. No new dependency, special compiler version, or invented LTS requirement is introduced. The required features are the existing C17 mode, libm functions, and the chosen diagnostic flags. [Clang's floating-point controls][clang-fp] document why finite-value assumptions from fast-math are incompatible with deliberate NaN and infinity checks.

To rebuild and run only the new executable in each mode:

```sh title="Terminal — focused backward checks"
make build/debug/test_attention_backward build/sanitize/test_attention_backward build/release/test_attention_backward
./build/debug/test_attention_backward
./build/sanitize/test_attention_backward
./build/release/test_attention_backward
```

A successful observed run with Clang 17 on x86_64 Linux printed:

```text
hand shared dK[0] (reference): [0.222222, -0.444444]
hand shared dK[0] (blocked): [0.222222, -0.444444]
full attention gradients: 1506 comparisons, max abs error 2.37583e-06
isolated Q/K/V gradients: 1056 comparisons, max abs error 1.03778e-06
attention backward: hand, modes, shared sums, dependencies, masks, failures passed
```

Comparison counts are determined by the fixed fixtures. Last-digit errors are observed values, not portable output requirements. All three builds passed, as did the inherited cumulative tests. A separate leak-enabled run of the new sanitizer executable reported no leaks. These are not M4/macOS execution results or performance measurements. [AddressSanitizer's documentation][asan] explains the memory diagnostics; it does not make mathematical correctness claims.

The new source snapshot contains 28 files. Only the attention header/source and Makefile change from Chapter 10, and the new backward test is added. Existing forward tests, earlier operator files, and the comparison application remain unchanged. Copying the whole snapshot into a separate scratch directory is a convenient way to run the reference without modifying an implementation in progress.

## Make four wrong implementations fail for different reasons

Use a temporary copy, make **one** change, rebuild the new executable, inspect its first numerical failure, and restore the correct source before the next change. These experiments are diagnostic variations, not missing required implementation work.

**Visit only the first sharing head.** In `qkv_backward`, change `gh < group` to `gh < 1`. The key/value sum now omits other consumers. In the hand case, the first input coordinate becomes about $0.33333334$ instead of $0.02526792$ in the observed run. Inspect the omitted second component of the first shared-key derivative: it should receive $-4/9$ from head one. MHA alone would not expose this bug because its group size is already one.

**Remove softmax's row subtraction.** Replace `- center` in the score-derivative expression with `- 0.0 * center` for the experiment. This keeps the variable referenced but deliberately discards its contribution. The first checked score derivative becomes about $1/3$ instead of $2/9$. The score-gradient row sum also loses its common-shift invariant. A one-key row incorrectly develops a score path.

**Omit the key-path score scale.** Change `scale * key_sum` to `key_sum` in the key-gradient store only. The first hand key derivative becomes about $0.31426969$ instead of $0.22222222$, a factor of $\sqrt2$. Query and value paths may still pass their local checks; inspect the key path rather than changing a global tolerance.

**Overwrite each input-projection path.** Change the final input-product mode from `p == 0 ? dx_mode : DL_ADD` to `dx_mode`. With `DL_SET`, the last value path erases query and key contributions. The hand input derivative again becomes about $1/3$ instead of $0.02526792$, but this time the saved key/value intermediate derivatives remain correct. Comparing those intermediates distinguishes this mistake from missing sharing heads.

All four variants returned numerical failures in sanitizer builds without an AddressSanitizer or UndefinedBehaviorSanitizer report in the authoring environment. Their addresses remain in bounds. This is why the test suite checks the intended function as well as memory validity.

For a valid variation, change only the first `gradient_case` in `main` from `{2,3,4,2,2}` to `{2,4,4,2,2}`. The constructor already supports four tokens. Predict that the square probability/score state grows with $T^2$, while the number of perturbed input/query/key/value coordinates grows linearly with $T$; parameter counts stay fixed. This variation passed, producing 1,542 full and 1,248 isolated comparisons. The modified fixture and binaries are not part of the normal source state.

## Locate the earliest wrong derivative

If $\bar C$ is wrong, inspect `DY`, output-weight orientation, and the output projection's input derivative. If $\bar P$ is wrong, inspect the value-head map, value stride, and the head slice of $\bar C$. If $\bar P$ is correct but $\bar S$ is wrong, inspect the allowed-key reduction and its row scalar before checking any projection.

Correct $\bar S$ with wrong $\bar Q$ points toward the scale, key address, or query-row reduction. Wrong shared $\bar K$ or $\bar V$ points toward the group/time reduction and its initialization. Correct rotated gradients but wrong unrotated ones point toward the saved position/frequency metadata or use of the forward rotation instead of its transpose.

Correct projected gradients with wrong parameter gradients usually indicate TN orientation or input-row selection. Correct parameter and intermediate gradients with wrong $\bar X$ suggests a missing `ADD`, a discarded caller seed, or a wrong weight matrix in one of its three paths. A uniform extra factor suggests repeated loss averaging or division by the sharing count.

The full materialized attention sublayer now has an implemented and checked backward reference. The next chapter composes embeddings, normalization, attention, feed-forward layers, residuals, the final normalization, and the tied output table into a complete decoder. Preserve the independent mode rules and saved-state lifetimes when connecting those pieces; composition should not change their derivatives.

## Technical references

The mathematical primary references are identified beside the derivations in [Part 1](https://25d-228.github.io/gpt-c/chapter-11.html#technical-references). The fixtures, expected values, independent local objective, and complete test program are specified here; they do not require an external tensor framework.

Clang, [floating-point behavior controls][clang-fp] and [AddressSanitizer usage][asan], document the compiler and diagnostic features used in the commands. Numerical tests remain necessary even when these diagnostics report no memory or undefined-behavior error.

[clang-fp]: https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior
[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#usage
