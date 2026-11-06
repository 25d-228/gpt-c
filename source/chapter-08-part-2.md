# Chapter 8, Part 2 — Checking feed-forward branches and saved state

This is the second half of [Chapter 8 — SwiGLU and the two-branch backward pass](https://25d-228.github.io/gpt-c/chapter-08.html). Part 1 supplies the complete interface and implementation. Here we build an executable that exposes intermediate values, checks the full reverse graph, exercises malformed calls and partial numerical failures, and composes the sublayer with RMSNorm and a residual addition.

**Prerequisites:** the entire Chapter 7 workspace and Part 1's new `include/dl/ffn.h` and `src/ffn.c`. Keep all existing source and tests. The blocks named `tests/test_ffn.c` concatenate in order into one new file; the final Makefile replaces the root file. The matching [test file](https://25d-228.github.io/gpt-c/media/chapter-08/reference/tests/test_ffn.c) and [Makefile](https://25d-228.github.io/gpt-c/media/chapter-08/reference/Makefile) are also included as companions.

**Implementation estimate:** the two parts share 3.50 hours, including 1.25 for checks, lifetimes, and integration. There is no additional estimate for this page. All model computations below are C calls to the routines being learned.

## Make numerical failures readable

Create `tests/test_ffn.c` with the following includes and basic diagnostics. A failed condition returns a nonzero process status through `main`; correctness does not depend on debug-only assertions. `near_value` rejects nonfinite values explicitly rather than relying on a comparison involving NaN.

The test suite distinguishes four kinds of failure: a wrong returned status, a wrong numerical value, a changed input or padding region, and a disagreement between analytic and numerical derivatives. That separation helps locate the first incorrect operation instead of treating every failure as a tolerance problem.

```c title="tests/test_ffn.c"
#include "dl/ffn.h"
#include "dl/norm.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int require(int condition, const char *where)
{
    if (!condition) fprintf(stderr, "%s\n", where);
    return condition;
}

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got != want) {
        fprintf(stderr, "%s: %s; expected %s\n", where,
                dl_status_name(got), dl_status_name(want));
        return 0;
    }
    return 1;
}

static int near_value(double got, double want, double tolerance, const char *where)
{
    if (!isfinite(got) || !isfinite(want) || fabs(got - want) > tolerance) {
        fprintf(stderr, "%s: %.12g; expected %.12g, tolerance %.3g\n",
                where, got, want, tolerance);
        return 0;
    }
    return 1;
}

```

## Use padded arrays without hiding their ownership

`FfnCase` owns sixteen inline arrays. The enumeration gives each role a named integer index; `NSLOTS` is the final count. There are no pointers stored inside this fixture, so copying it by value duplicates the bytes rather than leaving views aimed at the old object. We construct temporary views on demand.

Each array has 64 floats. Only the small shapes explicitly created in this file are supported by the fixture helper. The library has no 64-element bound; larger callers use checked allocations and truthful views. `view` adds a two-element initial offset and two unused elements between logical rows. That makes `ld` different from width and prevents a dense-only implementation from passing accidentally.

The shape mapping is deliberate. Gate/value weights and their gradients have $F$ rows and $D$ columns. Down weights and their gradients have $D$ rows and $F$ columns. Saved activations and their local derivatives have $M$ rows and $F$ columns. Inputs, outputs, incoming gradients, and input gradients have $M$ rows and $D$ columns.

`reset_case` fills nonlogical storage with a recognizable finite sentinel and fills all logical cells with NaN. Fixtures then initialize every required input. A successful SET call must overwrite its logical destinations without reading those old NaNs.

```c title="tests/test_ffn.c"
/* Test-only fixed storage; no dynamically owned pointers or stored views. */
enum { X, WG, WV, WO, A, B, H, Y, DY, DH, DA, DB, DX, DWG, DWV, DWO, NSLOTS };
typedef struct {
    size_t m, d, f;
    float cell[NSLOTS][64];
} FfnCase;

static DlMatrix dense(float *p, size_t rows, size_t cols)
{
    return (DlMatrix){p, rows * cols, 0, rows, cols, cols};
}

/* Only the small shapes constructed below fit these 64-float arrays. */
static DlMatrix view(FfnCase *q, int slot)
{
    size_t rows = q->m, cols = q->d;
    if (slot == WG || slot == WV || slot == DWG || slot == DWV) {
        rows = q->f;
    } else if (slot == WO || slot == DWO) {
        rows = q->d; cols = q->f;
    } else if (slot == A || slot == B || slot == H ||
               slot == DH || slot == DA || slot == DB) {
        cols = q->f;
    }
    return (DlMatrix){q->cell[slot], 64, 2, rows, cols, cols + 2};
}

static DlConstMatrix read_view(FfnCase *q, int slot)
{
    return dl_matrix_read(view(q, slot));
}

static void fill(DlMatrix v, float value)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) v.base[v.offset + i * v.ld + j] = value;
    }
}

static void load(DlMatrix v, const float *values)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j)
            v.base[v.offset + i * v.ld + j] = values[i * v.cols + j];
    }
}

static int expect(DlMatrix v, const double *values, double initial,
                   double factor, const char *where)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            const double want = initial + factor * values[i * v.cols + j];
            if (!near_value(v.base[v.offset + i * v.ld + j], want,
                            3e-6 + 2e-6 * fabs(want), where)) return 0;
        }
    }
    return 1;
}

static void reset_case(FfnCase *q, size_t m, size_t d, size_t f)
{
    *q = (FfnCase){.m = m, .d = d, .f = f};
    for (int slot = 0; slot < NSLOTS; ++slot) {
        for (size_t j = 0; j < 64; ++j) q->cell[slot][j] = -7777.0f;
        fill(view(q, slot), NAN);
    }
}

```

`expect` accepts an expected array, an initial value, and a factor. This lets the same hand-derived result check SET, ADD with a known previous contribution, and a repeated ADD call. It does not use the implementation's output to invent its expected derivative.

`guards` checks every nonlogical cell after work. The role builders below return the view-group structs from Part 1. The small `forward` and `backward` helpers only assemble arguments; they do not implement any missing mathematical operation.

`unchanged_reads` compares the full arrays containing $X$, all weights, all three saved activations, and the incoming gradient after backward. A const view does not enforce this property by itself, and scratch reuse can accidentally destroy saved inputs even when output numbers happen to look plausible.

```c title="tests/test_ffn.c"
static int guards(FfnCase *q)
{
    for (int slot = 0; slot < NSLOTS; ++slot) {
        const DlMatrix v = view(q, slot);
        for (size_t j = 0; j < v.count; ++j) {
            const int active = j >= v.offset && (j - v.offset) / v.ld < v.rows
                             && (j - v.offset) % v.ld < v.cols;
            if (!active && !require(v.base[j] == -7777.0f, "padding changed")) return 0;
        }
    }
    return 1;
}

static DlFfnWeights weights(FfnCase *q)
{
    return (DlFfnWeights){read_view(q, WG), read_view(q, WV), read_view(q, WO)};
}

static DlFfnCache cache(FfnCase *q)
{
    return (DlFfnCache){view(q, A), view(q, B), view(q, H)};
}

static DlFfnScratch scratch(FfnCase *q)
{
    return (DlFfnScratch){view(q, DH), view(q, DA), view(q, DB)};
}

static DlFfnGrads grads(FfnCase *q)
{
    return (DlFfnGrads){view(q, DWG), view(q, DWV), view(q, DWO)};
}

static DlStatus forward(FfnCase *q, DlBackend backend)
{
    return dl_ffn_forward(backend, read_view(q, X), weights(q), cache(q), view(q, Y));
}

static DlStatus backward(FfnCase *q, DlBackend backend,
                          DlWriteMode xm, DlWriteMode wm)
{
    return dl_ffn_backward(backend, read_view(q, X), weights(q), cache(q),
        read_view(q, DY), scratch(q), view(q, DX), xm, grads(q), wm);
}

static int unchanged_reads(const FfnCase *q, const FfnCase *saved)
{
    const int slots[8] = {X, WG, WV, WO, A, B, H, DY};
    for (size_t i = 0; i < 8; ++i) {
        if (!require(memcmp(q->cell[slots[i]], saved->cell[slots[i]],
                            sizeof q->cell[0]) == 0, "backward changed an input")) return 0;
    }
    return 1;
}

```

## Check the hand row before using finite differences

The next functions instantiate the row calculated in Part 1. The expected $dA$, $dB$, $dX$, and three weight-gradient arrays come from that calculation. The loop runs both matrix backends and all four combinations of input/weight write modes.

For SET, the old destination is NaN; for ADD, $dX$ starts at three and each weight gradient starts at five. After all four mode combinations, one more ADD must produce the old contribution plus **two** copies of the sublayer contribution. Neither the input nor the saved cache may change.

The second gate has a zero forward preactivation but a derivative of $-6$ in this fixture. Checking that value directly prevents a blanket “zero activation, zero derivative” rule from surviving behind a global gradient norm.

```c title="tests/test_ffn.c"
static void hand_case(FfnCase *q)
{
    reset_case(q, 1, 2, 2);
    const float x[2] = {1, 2}, wg[4] = {(float)log(2.0), 0, 1, -0.5f};
    const float wv[4] = {2, 0, -1, 2}, wo[4] = {1, -1, 0.5f, 2};
    const float g[2] = {2, -1};
    load(view(q, X), x); load(view(q, WG), wg); load(view(q, WV), wv);
    load(view(q, WO), wo); load(view(q, DY), g);
}

static int hand_and_modes(void)
{
    const double l = log(2.0), h = 4.0 * l / 3.0, da0 = 2.0 + 2.0 * l / 3.0;
    const double av[2] = {l, 0}, bv[2] = {2, 3}, hv[2] = {h, 0}, yv[2] = {h, h/2};
    const double dh[2] = {1.5, -4}, da[2] = {da0, -6}, db[2] = {l, 0};
    const double dx[2] = {da0 * l - 6.0 + 2.0 * l, 3};
    const double dg[4] = {da0, 2*da0, -6, -12}, dv[4] = {l, 2*l, 0, 0};
    const double dout[4] = {2*h, 0, -h, 0};
    for (int backend = 0; backend < 2; ++backend) {
        FfnCase q; hand_case(&q);
        if (!status_is(forward(&q, (DlBackend)backend), DL_OK, "hand forward") ||
            !expect(view(&q, A), av, 0, 1, "A") || !expect(view(&q, B), bv, 0, 1, "B") ||
            !expect(view(&q, H), hv, 0, 1, "H") || !expect(view(&q, Y), yv, 0, 1, "Y")) return 0;
        const FfnCase saved = q; /* Safe: owns inline arrays, no pointers. */
        for (int xm = 0; xm < 2; ++xm) {
            for (int wm = 0; wm < 2; ++wm) {
                fill(view(&q, DX), xm ? 3.0f : NAN);
                for (int k = DWG; k <= DWO; ++k) fill(view(&q, k), wm ? 5.0f : NAN);
                for (int k = DH; k <= DB; ++k) fill(view(&q, k), NAN);
                if (!status_is(backward(&q, (DlBackend)backend, xm ? DL_ADD : DL_SET,
                        wm ? DL_ADD : DL_SET), DL_OK, "hand backward") ||
                    !expect(view(&q, DH), dh, 0, 1, "dH") ||
                    !expect(view(&q, DA), da, 0, 1, "dA") ||
                    !expect(view(&q, DB), db, 0, 1, "dB") ||
                    !expect(view(&q, DX), dx, xm ? 3 : 0, 1, "dX merge") ||
                    !expect(view(&q, DWG), dg, wm ? 5 : 0, 1, "dWg") ||
                    !expect(view(&q, DWV), dv, wm ? 5 : 0, 1, "dWv") ||
                    !expect(view(&q, DWO), dout, wm ? 5 : 0, 1, "dWo") ||
                    !unchanged_reads(&q, &saved) || !guards(&q)) return 0;
            }
        }
        if (!status_is(backward(&q, (DlBackend)backend, DL_ADD, DL_ADD), DL_OK, "repeat ADD") ||
            !expect(view(&q, DX), dx, 3, 2, "repeated dX") ||
            !expect(view(&q, DWG), dg, 5, 2, "repeated dWg") ||
            !expect(view(&q, DWV), dv, 5, 2, "repeated dWv") ||
            !expect(view(&q, DWO), dout, 5, 2, "repeated dWo") ||
            !unchanged_reads(&q, &saved) || !guards(&q)) return 0;
    }
    printf("hand row: A=[%.6f, 0], B=[2, 3], H=[%.6f, 0]\n", l, h);
    printf("hand backward: dA=[%.6f, -6], dB=[%.6f, 0], dX=[%.6f, 3]\n", da0, l, dx[0]);
    puts("both backends, padded views, independent modes, and repeated ADD passed");
    return 1;
}

```

The hand row also separates the two input paths. A wrong final SET keeps only $dX^{(v)}=(2\log2,0)$, while dropping the value branch keeps only $dX^{(g)}$. Both can return `DL_OK`, and both are rejected by `expect`.

## Isolate the scalar activation inside the complete sublayer

We do not need a second public activation API to test SiLU. Choose $D=2,F=1$ and an input row $(a,1)$, with gate weights $(1,0)$, value weights $(0,1)$, and down weights $(1,0)^T$. Then

$$
A=a,\quad B=1,\quad H=\phi(a),\quad Y=(\phi(a),0).
$$

With incoming gradient $(1,0)$, the two input derivatives are $(\phi'(a),\phi(a))$. Thus one full forward/backward call exposes both the scalar slope and the product's value-branch derivative.

The fixture evaluates negative, near-zero, zero, and strongly positive gates. Its independent expected-value formula uses a literal sigmoid only for the bounded interval $[-100,100]$, where `exp(-a)` fits double. That formula is **not** proposed as the general library implementation. A second pass replaces one negative gate with `-FLT_MAX` to exercise safe exponential underflow without evaluating `exp(+FLT_MAX)`.

`FLT_TRUE_MIN` is the smallest positive float value supported by the implementation, including subnormals when present. It is distinct from `FLT_MIN`, the smallest positive normal float. A few such units of absolute tolerance account for tiny stored values, while the relative term checks ordinary magnitudes. This suite retains the floating-point assumptions and no-fast-math builds established in Chapter 3; it does not enable flush-to-zero arithmetic.

```c title="tests/test_ffn.c"
static int activation_edges(void)
{
    FfnCase q; reset_case(&q, 8, 2, 1);
    const float a[8] = {-100, -40, -2, -1e-4f, 0, 1e-4f, 40, 100};
    const float wg[2] = {1, 0}, wv[2] = {0, 1}, wo[2] = {1, 0};
    load(view(&q, WG), wg); load(view(&q, WV), wv); load(view(&q, WO), wo);
    const DlMatrix x = view(&q, X), g = view(&q, DY);
    for (size_t i = 0; i < 8; ++i) {
        x.base[x.offset + i*x.ld] = a[i]; x.base[x.offset + i*x.ld + 1] = 1;
        g.base[g.offset + i*g.ld] = 1; g.base[g.offset + i*g.ld + 1] = 0;
    }
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "activation edges forward") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_OK, "activation edges backward")) return 0;
    const DlMatrix y = view(&q, Y), dx = view(&q, DX);
    for (size_t i = 0; i < 8; ++i) {
        /* Independent direct formula is safe on this bounded [-100,100] fixture. */
        const double v = (double)a[i], s = 1.0 / (1.0 + exp(-v));
        const double want_y = v*s, want_dx = s + v*s*(1.0-s);
        if (!near_value(y.base[y.offset + i*y.ld], want_y,
                        8.0*(double)FLT_TRUE_MIN + 2e-6*fabs(want_y), "SiLU value") ||
            !near_value(dx.base[dx.offset + i*dx.ld], want_dx,
                        8.0*(double)FLT_TRUE_MIN + 2e-6*fabs(want_dx), "SiLU slope") ||
            !near_value(dx.base[dx.offset + i*dx.ld + 1], want_y,
                        8.0*(double)FLT_TRUE_MIN + 2e-6*fabs(want_y), "value-branch slope")) return 0;
    }
    if (!require(dx.base[dx.offset + 2*dx.ld] < 0, "SiLU slope at -2 must be negative") ||
        !require(dx.base[dx.offset + 4*dx.ld] == 0.5f, "SiLU slope at zero must be 0.5") ||
        !guards(&q)) return 0;
    /* Very large negative input must not call exp(+FLT_MAX). Underflow to zero is valid. */
    x.base[x.offset] = -FLT_MAX;
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "extreme negative forward") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_OK, "extreme negative backward") ||
        !near_value(y.base[y.offset], 0, 0, "negative saturation") ||
        !near_value(dx.base[dx.offset], 0, 0, "negative saturated slope")) return 0;
    puts("negative, near-zero, saturated, and extreme finite gates passed");
    return 1;
}

```

At $a=-2$, the test requires a negative slope. At zero it requires exactly $0.5$ in the current arithmetic. These targeted checks complement finite differences: a derivative can be broadly close on ordinary positive inputs while still implementing the wrong function around its negative region.

## Reject an invalid whole call without touching its destinations

The following functions exercise a late forward shape error, overlapping saved arrays, an out-of-range offset, overflowing view geometry, an invalid backend, and a nonfinite last weight. None should refresh even the first cache array.

After a successful forward, they similarly reject a wrong final weight-gradient shape, scratch that aliases a saved preactivation, overlapping gradient destinations, invalid modes, NaN old ADD outputs, and a nonfinite saved hidden value. The memory comparison covers the fixture's float arrays, not unspecified padding bytes of a structure.

A final valid case deliberately supplies the same read-only weight view to both branches. This is legal input sharing, but the weight-gradient destinations remain separate. It checks the alias rule; it does not silently turn two parameter roles into one automatically accumulated parameter.

```c title="tests/test_ffn.c"
static int no_mutation(const FfnCase *q, const FfnCase *saved, const char *where)
{
    return require(memcmp(q->cell, saved->cell, sizeof q->cell) == 0, where);
}

static int rejected_calls(void)
{
    FfnCase q; hand_case(&q);
    const FfnCase empty = q;
    DlFfnWeights w = weights(&q);
    DlFfnCache c = cache(&q);
    DlMatrix bad = view(&q, Y); bad.cols = 1;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, bad),
            DL_SHAPE, "late forward shape") || !no_mutation(&q, &empty, "shape mutated cache")) return 0;
    c.hidden = c.gate;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, view(&q, Y)),
            DL_ALIAS, "cache overlap") || !no_mutation(&q, &empty, "alias mutated cache")) return 0;
    c = cache(&q); c.value.offset = 63;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, view(&q, Y)),
            DL_RANGE, "range") || !no_mutation(&q, &empty, "range mutated cache")) return 0;
    c = cache(&q); c.gate.rows = SIZE_MAX;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, c, view(&q, Y)),
            DL_OVERFLOW, "view size overflow") || !no_mutation(&q, &empty, "overflow mutated cache")) return 0;
    if (!status_is(forward(&q, (DlBackend)99), DL_ARGUMENT, "backend") ||
        !no_mutation(&q, &empty, "bad backend wrote")) return 0;
    DlMatrix wo = view(&q, WO); wo.base[wo.offset] = INFINITY;
    const FfnCase nonfinite = q;
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_NUMERIC, "nonfinite last weight") ||
        !no_mutation(&q, &nonfinite, "preflight nonfinite wrote cache")) return 0;
    hand_case(&q);
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "prepare backward rejection")) return 0;
    const FfnCase saved = q;
    DlFfnGrads dw = grads(&q); dw.down.cols = 1;
    if (!status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), weights(&q),
            cache(&q), read_view(&q, DY), scratch(&q), view(&q, DX), DL_SET, dw, DL_SET),
            DL_SHAPE, "late backward shape") || !no_mutation(&q, &saved, "backward shape wrote")) return 0;
    DlFfnScratch work = scratch(&q); work.gate = view(&q, A);
    if (!status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), weights(&q),
            cache(&q), read_view(&q, DY), work, view(&q, DX), DL_SET, grads(&q), DL_SET),
            DL_ALIAS, "scratch aliases saved A") || !no_mutation(&q, &saved, "scratch alias wrote")) return 0;
    dw = grads(&q); dw.value = dw.gate;
    if (!status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), weights(&q),
            cache(&q), read_view(&q, DY), scratch(&q), view(&q, DX), DL_SET, dw, DL_SET),
            DL_ALIAS, "gradient outputs overlap") || !no_mutation(&q, &saved, "gradient alias wrote")) return 0;
    if (!status_is(backward(&q, DL_BACKEND_REF, (DlWriteMode)99, DL_SET), DL_ARGUMENT, "dx mode") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, (DlWriteMode)99), DL_ARGUMENT, "weight mode") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_ADD, DL_SET), DL_NUMERIC, "NaN old dx") ||
        !status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_ADD), DL_NUMERIC, "NaN old weights") ||
        !no_mutation(&q, &saved, "rejected mode/old destination wrote")) return 0;
    DlMatrix h = view(&q, H); h.base[h.offset] = NAN;
    const FfnCase bad_saved = q;
    if (!status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_NUMERIC, "NaN saved hidden") ||
        !no_mutation(&q, &bad_saved, "bad saved values wrote")) return 0;

    /* Input weights may share values, but derivative outputs stay separate. */
    hand_case(&q); w = weights(&q); w.value = w.gate;
    if (!status_is(dl_ffn_forward(DL_BACKEND_REF, read_view(&q, X), w, cache(&q), view(&q, Y)),
            DL_OK, "shared read-only weights") ||
        !status_is(dl_ffn_backward(DL_BACKEND_REF, read_view(&q, X), w, cache(&q),
            read_view(&q, DY), scratch(&q), view(&q, DX), DL_SET, grads(&q), DL_SET),
            DL_OK, "shared weights backward") || !guards(&q)) return 0;
    puts("shape, alias, range, overflow, mode, and nonfinite preflight checks passed");
    return 1;
}

```

The checks use actual library return values. For example, increasing a view's row count to `SIZE_MAX` is rejected by checked geometry arithmetic before any attempt to read such a matrix. It is not a test that really allocates or traverses an enormous buffer.

The snapshot comparisons include NaN-filled cells. `memcmp` compares their saved object representations without interpreting NaNs as equal floating-point numbers. This is a local untouched-memory check, not an artifact checksum workflow.

## Observe numerical failure after earlier writes

A numerical execution failure is not the same as a malformed-call rejection. The next two fixtures reproduce Part 1's range examples. In the first, gate and value projections have been written when the hidden product overflows; the down weight happens to be zero, but that cannot repair the missing finite hidden value. In the second, forward succeeds, $dH$ is written as zero, and the output-weight derivative then exceeds float range.

We fill destinations with nine so the partial state is visible. We never use those values as a valid result, and we do not retry ADD on them. The fixture is reset before the second experiment.

```c title="tests/test_ffn.c"
static int partial_failures(void)
{
    FfnCase q; reset_case(&q, 1, 1, 1);
    fill(view(&q, X), 1); fill(view(&q, WG), FLT_MAX);
    fill(view(&q, WV), 2); fill(view(&q, WO), 0); fill(view(&q, DY), 1);
    fill(view(&q, H), 9); fill(view(&q, Y), 9);
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_NUMERIC, "hidden overflow despite zero down weight") ||
        !require(q.cell[A][2] == FLT_MAX && q.cell[B][2] == 2 &&
                 q.cell[H][2] == 9 && q.cell[Y][2] == 9, "unexpected forward partial state")) return 0;
    /* All forward intermediates are representable; dWo overflows afterward. */
    reset_case(&q, 1, 1, 1);
    fill(view(&q, X), 1); fill(view(&q, WG), 100);
    fill(view(&q, WV), 1); fill(view(&q, WO), 0); fill(view(&q, DY), FLT_MAX);
    if (!status_is(forward(&q, DL_BACKEND_REF), DL_OK, "finite zero-output forward")) return 0;
    for (int slot = DH; slot < NSLOTS; ++slot) fill(view(&q, slot), 9);
    if (!status_is(backward(&q, DL_BACKEND_REF, DL_SET, DL_SET), DL_NUMERIC, "dWo overflow") ||
        !require(q.cell[DH][2] == 0 && q.cell[DWO][2] == 9 && q.cell[DX][2] == 9,
                 "unexpected backward partial state") || !guards(&q)) return 0;
    puts("multi-output partial numerical failures detected; damaged results discarded");
    return 1;
}

```

These cases show why a multi-output API needs one documented failure policy. Looking only at whether $Y$ or $dX$ changed would miss the modified saved/scratch arrays. If a future training step receives this failure, it must discard the step's affected results rather than update a subset of parameters.

## Construct non-square, multi-row derivative fixtures

Finite differences will check $M=2,D=3,F=5$. The gate weights include a very small row, so the ordinary fixture contains positive, negative, and near-zero preactivations. The other weight matrices and incoming gradients are asymmetric. Non-square shapes and mixed signs make a mistaken transpose less likely to pass by coincidence.

There are three isolated variants: ordinary weights; every gate weight zero; and every value weight zero. Both zero variants produce zero outputs but need not produce zero **parameter** derivatives. With $W_g=0$, the SiLU slope at zero permits a nonzero $dW_g$; with $W_v=0$, a nonzero gate activation permits a nonzero $dW_v$. Those are particularly useful tests of the two branches.

The `Graph` structure additionally owns a raw input, normalization scales, a residual output, and their gradients. Like `FfnCase`, it contains inline arrays rather than stored pointers, so a value copy is a valid independent snapshot. No heap allocation or external model library is involved.

```c title="tests/test_ffn.c"
static void ordinary_case(FfnCase *q, int variant)
{
    reset_case(q, 2, 3, 5);
    const float x[6] = {0.7f, -0.5f, 1.2f, -0.8f, 0.4f, 0.3f};
    const float wg[15] = {0.6f,-0.4f,0.2f, -0.5f,0.3f,0.7f,
        1e-4f,-2e-4f,3e-4f, -0.2f,-0.1f,0.3f, 0.4f,-0.6f,-0.3f};
    const float wv[15] = {0.2f,0.5f,-0.3f, -0.4f,0.6f,0.1f,
        0.7f,-0.2f,0.4f, -0.3f,0.2f,0.5f, 0.6f,0.1f,-0.4f};
    const float wo[15] = {0.3f,-0.2f,0.4f,0.1f,-0.5f,
        -0.4f,0.6f,0.2f,-0.3f,0.1f, 0.2f,0.1f,-0.5f,0.4f,0.3f};
    const float g[6] = {0.3f,-0.4f,0.7f, -0.6f,0.2f,0.5f};
    load(view(q, X), x); load(view(q, WG), wg); load(view(q, WV), wv);
    load(view(q, WO), wo); load(view(q, DY), g);
    if (variant == 1) fill(view(q, WG), 0);
    if (variant == 2) fill(view(q, WV), 0);
}

/* Integration case: Z = U + FFN(RMSNorm(U, scale)). Inline owners only. */
typedef struct {
    FfnCase f;
    float raw[6], scale[3], z[6], draw[6], dscale[3];
} Graph;

static void graph_init(Graph *g, int variant)
{
    *g = (Graph){.raw = {0.7f,-0.5f,1.2f, -0.8f,0.4f,0.3f},
                 .scale = {1.2f,-0.7f,0.8f}};
    ordinary_case(&g->f, variant);
}

```

## Compose normalization, SwiGLU, and the identity path

For the integration case let $U$ be the raw input and $\gamma$ the normalization scale. The graph is

$$
N=\operatorname{RMSNorm}(U,\gamma),\qquad
Z=U+\operatorname{FFN}(N).
$$

The fixture's `f.X` buffer holds $N$ in this case, while `raw` holds $U$. For the isolated cases, `f.X` is the independent input directly. The flag `residual` chooses the complete graph; it is a test harness choice, not an implicit mode inside the sublayer API.

Use a scalar objective

$$
L=\sum_{i,j}G_{ij}Z_{ij}
$$

for the integrated graph, or the same weighted sum of $Y$ for the isolated sublayer. $G$ is the fixed incoming-gradient fixture. Differentiating this sum gives exactly $G$ at the output, which makes it a useful way to test arbitrary reverse directions without adding another loss implementation.

Backward first obtains $dN$ from the complete feed-forward routine. The residual identity edge initializes $dU$ with $G$, then normalization adds its branch contribution to that same $dU$ and computes $d\gamma$. The incoming gradient to the feed-forward branch is also $G$: addition's derivative is identity on each operand. Both edges may read the same incoming storage; no duplicate array is required merely to copy a read-only gradient.

The epsilon is the explicit float `0.125f` in both normalization calls. No optimizer changes a weight between forward and backward.

```c title="tests/test_ffn.c"
static DlStatus graph_forward(Graph *g, DlBackend backend, int residual, double *objective)
{
    FfnCase *q = &g->f;
    DlStatus s = DL_OK;
    if (residual) s = dl_rmsnorm_forward(dl_matrix_read(dense(g->raw, 2, 3)),
        dl_matrix_read(dense(g->scale, 1, 3)), 0.125f, view(q, X));
    if (s == DL_OK) s = forward(q, backend);
    if (s == DL_OK && residual) s = dl_residual_add(dl_matrix_read(dense(g->raw, 2, 3)),
        read_view(q, Y), dense(g->z, 2, 3));
    if (s != DL_OK) return s;
    const DlConstMatrix y = residual ? dl_matrix_read(dense(g->z, 2, 3)) : read_view(q, Y);
    const DlConstMatrix dy = read_view(q, DY);
    double sum = 0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j)
            sum += (double)y.base[y.offset + i*y.ld + j] * (double)dy.base[dy.offset + i*dy.ld + j];
    }
    if (!isfinite(sum)) return DL_NUMERIC;
    *objective = sum;
    return DL_OK;
}

static DlStatus graph_backward(Graph *g, DlBackend backend, int residual)
{
    DlStatus s = backward(&g->f, backend, DL_SET, DL_SET);
    if (s == DL_OK && residual) s = dl_residual_backward(read_view(&g->f, DY),
        dense(g->draw, 2, 3), DL_SET);
    if (s == DL_OK && residual) s = dl_rmsnorm_backward(
        dl_matrix_read(dense(g->raw, 2, 3)), dl_matrix_read(dense(g->scale, 1, 3)),
        read_view(&g->f, DX), 0.125f, dense(g->draw, 2, 3), DL_ADD,
        dense(g->dscale, 1, 3), DL_SET);
    return s;
}

```

The integrated graph is a complete feed-forward residual sublayer, not a full transformer block: it intentionally has no attention yet. Its purpose is to catch interactions that an isolated activation test cannot, particularly a missing branch inside SwiGLU or a SET that erases the residual identity derivative.

The two accessors below select the actual parameter and gradient arrays for each finite-difference group. In the isolated cases there are four groups: input, gate weights, value weights, and down weights. The integrated case adds the normalization scale and perturbs the raw input instead of the derived normalized buffer.

Returning a view by value here is safe because it points into the still-live `Graph` owned by `numerical_case`; returning a view into an accessor's own local array would not be safe. The fixed group counts below constrain the indices passed to these test-only accessors.

```c title="tests/test_ffn.c"
static DlMatrix parameter(Graph *g, size_t group, int residual)
{
    if (group == 0) return residual ? dense(g->raw, 2, 3) : view(&g->f, X);
    if (group == 4) return dense(g->scale, 1, 3);
    const int slots[3] = {WG, WV, WO};
    return view(&g->f, slots[group - 1]);
}

static DlMatrix derivative(Graph *g, size_t group, int residual)
{
    if (group == 0) return residual ? dense(g->draw, 2, 3) : view(&g->f, DX);
    if (group == 4) return dense(g->dscale, 1, 3);
    const int slots[3] = {DWG, DWV, DWO};
    return view(&g->f, slots[group - 1]);
}

```

## Perturb actual float storage and restore matching forward state

For each scalar parameter $p$, first run the unperturbed C forward and backward. Save its analytic derivative. Then choose representable float endpoints

$$
p_+=\operatorname{float}(p+h),\qquad
p_-=\operatorname{float}(p-h),
$$

and compare against

$$
g_{\mathrm{num}}=\frac{L(p_+)-L(p_-)}{\operatorname{double}(p_+)-\operatorname{double}(p_-)}.
$$

The denominator is the represented endpoint spacing, not blindly $2h$. Each trial runs the **whole C forward graph**. It does not overwrite the analytic gradient arrays. Restore the parameter before inspecting evaluation statuses so a failed trial does not leave the fixture perturbed.

Three relative step scales, `0.01`, `0.003`, and `0.001`, give different truncation/rounding tradeoffs. The step multiplies `max(1,abs(p))`; the mixed tolerance is

$$
8\times10^{-5}+3\times10^{-3}\max(|g_{\mathrm{analytic}}|,|g_{\mathrm{num}}|).
$$

This is a tolerance for these small float fixtures, not a universal constant for every future model. A near-zero derivative needs an absolute term; a larger derivative also needs a relative term. A growing error as $h$ shrinks can reflect forward rounding, while a persistent large discrepancy across steps usually points back to the implementation or saved state.

There is one additional lifetime trap: restoring parameters does not restore $A,B,H$ from the last perturbed forward. After all comparisons, the checker verifies parameter restoration, reruns a baseline forward, and only then reruns backward. It compares the replayed objective and gradients with the original baseline. The exact replay check is within one deterministic executable and one unchanged input state; it is not a cross-platform bitwise-equality claim.

```c title="tests/test_ffn.c"
static int numerical_case(int variant, int residual, DlBackend backend,
                           size_t *comparisons, double *worst)
{
    Graph g; graph_init(&g, variant);
    double baseline = 0;
    if (!status_is(graph_forward(&g, backend, residual, &baseline), DL_OK, "gradient baseline") ||
        !status_is(graph_backward(&g, backend, residual), DL_OK, "analytic backward")) return 0;
    Graph original = g; /* This value copy duplicates inline arrays, not pointers. */
    const char *names[5] = {"input", "Wg", "Wv", "Wo", "scale"};
    const double steps[3] = {0.01, 0.003, 0.001};
    const size_t groups = residual ? 5 : 4;
    for (size_t group = 0; group < groups; ++group) {
        const DlMatrix p = parameter(&g, group, residual), d = derivative(&g, group, residual);
        for (size_t i = 0; i < p.rows; ++i) {
            for (size_t j = 0; j < p.cols; ++j) {
                float *cell = &p.base[p.offset + i*p.ld + j];
                const float saved = *cell;
                const double analytic = (double)d.base[d.offset + i*d.ld + j];
                for (size_t k = 0; k < 3; ++k) {
                    const double h = steps[k] * fmax(1.0, fabs((double)saved));
                    const float hi = (float)((double)saved + h), lo = (float)((double)saved - h);
                    const double spacing = (double)hi - (double)lo;
                    double plus = 0, minus = 0;
                    *cell = hi;
                    const DlStatus sp = graph_forward(&g, backend, residual, &plus);
                    *cell = lo;
                    const DlStatus sm = graph_forward(&g, backend, residual, &minus);
                    *cell = saved; /* Restore before ANY status/error return. */
                    if (!status_is(sp, DL_OK, "plus") || !status_is(sm, DL_OK, "minus") ||
                        !require(spacing > 0, "collapsed finite-difference interval")) return 0;
                    const double numerical = (plus - minus) / spacing;
                    const double tolerance = 8e-5 + 3e-3 * fmax(fabs(analytic), fabs(numerical));
                    const double ratio = fabs(analytic - numerical) / tolerance;
                    if (!isfinite(analytic) || !isfinite(numerical) || ratio > 1.0) {
                        fprintf(stderr, "variant=%d residual=%d backend=%d %s[%zu,%zu] h=%.3g: "
                            "analytic=%.9g numeric=%.9g ratio=%.4g\n", variant, residual,
                            (int)backend, names[group], i, j, h, analytic, numerical, ratio);
                        return 0;
                    }
                    if (ratio > *worst) *worst = ratio;
                    ++*comparisons;
                }
            }
        }
    }
    /* Restoring parameters does NOT restore the last perturbed forward cache. */
    for (size_t group = 0; group < groups; ++group) {
        const DlMatrix p = parameter(&g, group, residual), old = parameter(&original, group, residual);
        for (size_t i = 0; i < p.rows; ++i) {
            for (size_t j = 0; j < p.cols; ++j)
                if (!require(p.base[p.offset+i*p.ld+j] == old.base[old.offset+i*old.ld+j],
                             "parameter not restored")) return 0;
        }
    }
    double replay = 0;
    if (!status_is(graph_forward(&g, backend, residual, &replay), DL_OK, "restore baseline cache") ||
        !near_value(replay, baseline, 0, "baseline replay") ||
        !status_is(graph_backward(&g, backend, residual), DL_OK, "backward after replay")) return 0;
    for (size_t group = 0; group < groups; ++group) {
        const DlMatrix d = derivative(&g, group, residual), old = derivative(&original, group, residual);
        for (size_t i = 0; i < d.rows; ++i) {
            for (size_t j = 0; j < d.cols; ++j)
                if (!near_value(d.base[d.offset+i*d.ld+j], old.base[old.offset+i*old.ld+j],
                                0, "gradient replay")) return 0;
        }
    }
    return guards(&g.f);
}

```

The isolated fixture has $6+15+15+15=51$ coordinates. Three isolated variants, two backends, and three step sizes give $51\cdot3\cdot2\cdot3=918$ comparisons. The integrated graph adds three normalization scales, so its $54\cdot2\cdot3=324$ comparisons bring the total to **1,242**.

Every weight family is checked, not just a norm of their concatenation or one randomly selected coordinate. The input checks expose both branch contributions; the integrated input checks also expose the residual identity path and the normalization derivative. Earlier chapter tests remain separate executables and still run.

Finish `tests/test_ffn.c` with the complete entry point below. It runs structural and numerical boundary checks before finite differences, so an obvious contract failure is reported without a flood of later derivative errors.

```c title="tests/test_ffn.c"
int main(void)
{
    if (!hand_and_modes() || !activation_edges() || !rejected_calls() || !partial_failures()) return 1;
    size_t comparisons = 0;
    double worst = 0;
    for (int backend = 0; backend < 2; ++backend) {
        for (int variant = 0; variant < 3; ++variant) {
            if (!numerical_case(variant, 0, (DlBackend)backend, &comparisons, &worst)) return 1;
        }
        if (!numerical_case(0, 1, (DlBackend)backend, &comparisons, &worst)) return 1;
    }
    printf("SwiGLU/full residual finite differences: %zu passed; worst error/tolerance=%.6g\n",
           comparisons, worst);
    puts("all feed-forward checks passed");
    return 0;
}
```

## Build the cumulative program

Replace the root Makefile with this complete version. It adds `src/ffn.c`, its header dependency, and `test_ffn` while preserving all earlier test executables and the kernel comparison application. Keep the tab characters before recipe commands.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn

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

On the native Mac, run from the learner workspace:

```sh title="Terminal — cumulative builds"
make clean
make test
```

The default compiler remains `xcrun clang`. A compatible configured compiler can be selected with `make CC=clang test`; this override is not a requirement to change an already suitable Mac toolchain. Debug, sanitizer, and optimized builds live in separate directories. The build keeps the established C17, finite-value, and no-contraction settings rather than enabling fast-math just to obtain a smaller timing result.

For a focused rerun after an edit:

```sh title="Terminal — focus on the feed-forward sublayer"
make debug sanitize release
./build/debug/test_ffn
./build/sanitize/test_ffn
./build/release/test_ffn
```

Where the installed sanitizer runtime supports leak detection, the focused command can also be run as:

```sh title="Terminal — optional runtime leak check"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./build/sanitize/test_ffn
```

The [Clang AddressSanitizer documentation][asan] describes this runtime option and platform differences. An unavailable leak detector is not a numerical failure; continue using the address and undefined-behavior instrumentation supported by the installed runtime. The new sublayer allocates nothing internally, but the cumulative suite still includes the existing owner tests.

One observed Clang 17, x86_64 Linux run printed:

```text title="Observed output — test_ffn, x86_64 Linux"
hand row: A=[0.693147, 0], B=[2, 3], H=[0.924196, 0]
hand backward: dA=[2.462098, -6], dB=[0.693147, 0], dX=[-2.907109, 3]
both backends, padded views, independent modes, and repeated ADD passed
negative, near-zero, saturated, and extreme finite gates passed
shape, alias, range, overflow, mode, and nonfinite preflight checks passed
multi-output partial numerical failures detected; damaged results discarded
SwiGLU/full residual finite differences: 1242 passed; worst error/tolerance=0.338303
all feed-forward checks passed
```

The exact last error ratio is diagnostic, not a target value. It must remain at most one under the stated acceptance rule. The companion code passed debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized cumulative builds in that Linux environment without compiler warnings. The new leak-enabled run also passed. The inherited 104 loss/embedding comparisons, 189 normalization/residual comparisons, 56 blocked cases, other kernel checks, and comparison smoke calls continued to pass.

Those are executed C checks, not M4 performance measurements. No native macOS execution, full-model throughput, training run, or process-memory result is claimed for this chapter. The comparison smoke calls verify the inherited application's interface; they do not measure this feed-forward sublayer.

## Break one derivative at a time

Use a temporary copy of the workspace and restore the correct version after each variation. Rebuild the sanitizer executable each time. No variation below changes array bounds; memory instrumentation alone cannot detect the mathematical error.

### Drop the value branch

In `src/ffn.c`, change the expression passed to the `scratch.value` store from `g * z.value` to `0.0 * g * z.value`. The hand fixture immediately reports a zero $dB_0$ instead of $\log2\approx0.693147$. Forward remains unchanged.

For a focused finite-difference diagnosis, temporarily replace the first conditional in `main` with the following diagnostic version. Its permanently false left operand skips the preliminary fixtures while leaving the finite-difference loop unchanged. This is a temporary test selection, not a final passing test configuration.

```c title="Diagnostic replacement — first conditional in main"
if (0 && (!hand_and_modes() || !activation_edges() ||
          !rejected_calls() || !partial_failures())) return 1;
```

In one executed run, the first ordinary input comparison gave analytic `-0.148661450` versus numerical `-0.101596457`, with error/tolerance about `89.48`. The discrepancy is the missing value-path contribution, not random rounding noise. Restore both the implementation and normal test selection before proceeding.

### Overwrite the first input contribution

Change only the final matrix call in `dl_ffn_backward` from `dx, DL_ADD` to `dx, DL_SET`. The hand check now gives `1.386294` instead of `-2.907109` for the first input coordinate: it retained only the value branch.

Using the same temporary finite-difference-only selection, one observed first comparison gave analytic `0.0470632091` versus numerical `-0.101596457`, with ratio about `386.3`. A backward pass can have individually correct matrix products and still be wrong because its write mode loses an edge.

### Differentiate sigmoid but forget multiplication by the gate input

Change the helper's return from `(Silu){a * s, s + a * ds}` to `(Silu){a * s, s + 0.0 * a * ds}`. The expression retains the local variables but drops the second term of the product rule. The hand fixture reports $dA_0=2$ instead of approximately $2.462098$.

The focused finite-difference run gave analytic `-0.0757671967` versus numerical `-0.101596457`, ratio about `67.13`, at the same first input coordinate. The scalar negative-gate test would also catch an incorrect sign at $a=-2$ once the earlier hand failure is repaired.

All three faults were executed in separate sanitizer builds. Each failed an explicit hand check and, with preliminary checks temporarily skipped, failed finite differences. None produced an AddressSanitizer or undefined-behavior diagnostic in these runs. Restore the correct helper and normal `main` before treating the suite as valid.

## Make a valid change and predict which gradients change

For a useful variation, multiply every logical value weight in `ordinary_case` by $c=-2$ immediately after loading the fixture and **before** its zero-variant overrides. Insert this complete block there; do not change the hand fixture or the derivative formulas:

```c title="Variation — insert in ordinary_case after loading weights"
DlMatrix value_weights = view(q, WV);
for (size_t i = 0; i < value_weights.rows; ++i) {
    for (size_t j = 0; j < value_weights.cols; ++j) {
        value_weights.base[value_weights.offset + i * value_weights.ld + j] *= -2.0f;
    }
}
```

For fixed isolated input, gate weights, down weights, and incoming $G$, scaling $W_v$ by $c$ scales $B,H,Y,dA,dW_g,dW_o$, and the complete branch input gradient by $c$. It does **not** scale $dH$ or $dB$, and $dW_v=dB^TX$ stays unchanged: the derivative of a linear function with respect to its current weight does not gain an extra factor just because that weight's value changed. This is a prediction about the new parameter point, not a reparameterization with $c$ inside the differentiation graph.

For the integrated residual graph, only the learned branch's raw-input contribution scales; the identity contribution $G$ does not. Therefore the full raw-input gradient is not simply multiplied by $c$. Its decomposition is $G+c\,dU^{(\mathrm{branch})}$. This distinction makes the variation a useful cross-check of the Chapter 7 accumulation rule.

The modified fixtures still passed all 1,242 finite-difference comparisons in an executed sanitizer run, with maximum error/tolerance approximately `0.354464`. That observation checks analytic/numerical consistency at the changed weights; the scaling predictions above follow directly from the equations. Restore the baseline fixture afterward so subsequent chapter snapshots share the same starting state.

## Diagnose the first quantity that diverges

| Symptom | First place to inspect |
|---|---|
| Gate projection is correct but the value projection is not | Both projections must read the same $X$; check stored weight orientation |
| Only negative gates disagree | Stable sigmoid sign branch, SiLU rather than ReLU, and the full scalar slope |
| A zero gate has zero $dA$ despite nonzero $dH B$ | SiLU slope at zero is $1/2$, not zero |
| Down-weight derivatives pass but one projection family fails | The two different product derivatives and the transposed row reduction |
| All input checks fail while several weight checks pass | Missing branch, wrong matrix orientation, or SET used for the second $dX$ path |
| ADD works for weights but not for the input | Input mode is independent; existing $dX$ must be included exactly once |
| A second backward disagrees after finite differences | Parameters were restored but matching forward cache was not recomputed |
| Dense fixtures pass and padded fixtures fail | Each view needs its own offset and leading dimension |
| A valid zero final output nevertheless returns `DL_NUMERIC` | A required intermediate or derivative may be unrepresentable |
| Changing optimizer state between forward/backward changes gradients unexpectedly | This backward requires the original weights and saved activations |

The completed workspace now has a manually differentiated, three-matrix SwiGLU sublayer and a tested composition with pre-normalization and a residual path. The next [chapter in the roadmap](https://25d-228.github.io/gpt-c/index.html#chapter-9-rotary-positions-as-explicit-coordinate-transformations) develops rotary positions as explicit coordinate transformations before attention is assembled.

## References

The architectural and scalar-activation references are discussed in [Part 1](https://25d-228.github.io/gpt-c/chapter-08.html#references). The executable fixtures, write-mode checks, finite differences, and composed graph on this page are specified directly in the chapter rather than delegated to another implementation.

- Clang, [AddressSanitizer: usage and memory leak detection][asan], for compiler/runtime instrumentation and the optional leak-detection setting.

[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#memory-leak-detection

