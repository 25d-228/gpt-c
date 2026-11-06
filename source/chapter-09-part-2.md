# Chapter 9, Part 2 — Checking coordinate conventions and absolute positions

[Part 1](https://25d-228.github.io/gpt-c/chapter-09.html) implements split-half rotary positions and their transpose derivatives. This part makes the coordinate convention, position identity, and failure behavior observable. A correct-looking norm is not enough: we need a hand-computed layout fixture, independent position checks, and numerical derivatives of a small composed expression.

**Prerequisites:** the complete Chapter 8 workspace plus Part 1's `include/dl/rope.h` and `src/rope.c`. Keep all earlier code. The blocks labeled `tests/test_rope.c` concatenate in order into one complete new test file. The final Makefile replaces the root Makefile. Matching [tests](https://25d-228.github.io/gpt-c/media/chapter-09/reference/tests/test_rope.c) and [Makefile](https://25d-228.github.io/gpt-c/media/chapter-09/reference/Makefile) are included as companions.

**Implementation estimate:** the two parts share **2.50 hours**, including 1.00 for these checks and integration. This page does not add another chapter or another estimate.

## Small diagnostics with explicit failure statuses

Create `tests/test_rope.c`. `require` and `status_is` return a boolean-like `int`, which eventually determines the process exit status. There are no debug-only assertions. `near_value` rejects nonfinite values before testing a numerical difference.

The `dense` helper is deliberately limited to fixed, already-sized arrays in these tests. It is not a substitute for the checked dynamic tensor constructor. The `dot` helper widens both float inputs before multiplying, then reduces in double. We use it to form scalar test objectives and squared norms, not to introduce another model backend.

```c title="tests/test_rope.c"
#include "dl/rope.h"

#include <float.h>
#include <math.h>
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

/* Only fixed, small, already-sized arrays use this test-only helper. */
static DlMatrix dense(float *p, size_t rows, size_t cols)
{
    return (DlMatrix){p, rows * cols, 0, rows, cols, cols};
}

static double dot(const float *a, const float *b, size_t n)
{
    double result = 0.0;
    for (size_t j = 0; j < n; ++j) result += (double)a[j] * (double)b[j];
    return result;
}

```

## Separate frequency generation from coordinate rotation

The first fixture checks the geometric frequency constructor against $[1,0.1,0.01,0.001]$ and the selected checkpoint's first/last frequencies. It checks shape/base errors without allowing them to change an existing frequency vector. The oversized count is rejected by byte arithmetic **before** the supplied small output can be accessed with that count.

The hand row then uses the two synthetic frequencies from Part 1. `acos(-1.0)` supplies a double approximation to $\pi$; no nonstandard `M_PI` macro or hardcoded decimal angle is required. The cosine of the represented quarter-turn need not be exactly zero, so the test uses a small tolerance. The expected coordinates and derivative are independent expressions, not output copied from another call to the same implementation.

```c title="tests/test_rope.c"
static int frequencies_and_hand(void)
{
    double f[4] = {9.0, 9.0, 9.0, 9.0};
    if (!status_is(dl_rope_frequencies(8, 10000.0, f, 4), DL_OK,
                   "four frequencies")) return 0;
    const double want[4] = {1.0, 0.1, 0.01, 0.001};
    for (size_t j = 0; j < 4; ++j) {
        if (!near_value(f[j], want[j], 1e-15, "frequency value")) return 0;
    }
    double checkpoint[32];
    if (!status_is(dl_rope_frequencies(64, 100000.0, checkpoint, 32),
                   DL_OK, "checkpoint frequencies") ||
        !near_value(checkpoint[0], 1.0, 0.0, "first checkpoint frequency") ||
        !near_value(checkpoint[31], 1.43301257023696e-5, 1e-18,
                    "last checkpoint frequency")) return 0;
    const double saved[4] = {f[0], f[1], f[2], f[3]};
    if (!status_is(dl_rope_frequencies(7, 10000.0, f, 4), DL_SHAPE, "odd d") ||
        !status_is(dl_rope_frequencies(0, 10000.0, f, 4), DL_SHAPE, "zero d") ||
        !status_is(dl_rope_frequencies(8, 10000.0, f, 3), DL_SHAPE, "bad count") ||
        !status_is(dl_rope_frequencies(8, 1.0, f, 4), DL_ARGUMENT, "theta one") ||
        !status_is(dl_rope_frequencies(8, NAN, f, 4), DL_ARGUMENT, "theta NaN") ||
        !status_is(dl_rope_frequencies(8, INFINITY, f, 4), DL_ARGUMENT,
                    "theta infinity") ||
        !status_is(dl_rope_frequencies(8, 10000.0, NULL, 4), DL_ARGUMENT,
                    "null frequencies") ||
        !status_is(dl_rope_frequencies(SIZE_MAX - 1, 10000.0, f, SIZE_MAX / 2),
                    DL_OVERFLOW, "frequency bytes overflow") ||
        !require(memcmp(f, saved, sizeof f) == 0, "failed constructor wrote")) {
        return 0;
    }

    /* Synthetic diagnostic frequencies, NOT the model's geometric schedule. */
    const double pi = acos(-1.0);
    const double hand_f[2] = {pi / 4.0, pi / 8.0};
    const size_t position[1] = {2};
    float x[4] = {3.0f, 1.0f, 4.0f, 2.0f};
    float y[4] = {NAN, NAN, NAN, NAN}, dx[4] = {NAN, NAN, NAN, NAN};
    const double expected[4] = {-4.0, -1.0 / sqrt(2.0), 3.0, 3.0 / sqrt(2.0)};
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(x, 1, 4)), 4,
                   position, 1, hand_f, 2, dense(y, 1, 4)), DL_OK, "hand forward")) {
        return 0;
    }
    for (size_t j = 0; j < 4; ++j) {
        if (!near_value(y[j], expected[j], 3e-7, "hand split-half value")) return 0;
    }
    float incoming[4] = {2.0f, -1.0f, 3.0f, 4.0f};
    const double grad[4] = {3.0, 3.0 / sqrt(2.0), -2.0, 5.0 / sqrt(2.0)};
    if (!status_is(dl_rope_backward(dl_matrix_read(dense(incoming, 1, 4)), 4,
                   position, 1, hand_f, 2, dense(dx, 1, 4), DL_SET), DL_OK,
                   "hand backward")) return 0;
    for (size_t j = 0; j < 4; ++j) {
        if (!near_value(dx[j], grad[j], 3e-7, "hand transpose value")) return 0;
    }
    printf("split-half hand row: [%.6f, %.6f, %.6f, %.6f]\n",
           (double)y[0], (double)y[1], (double)y[2], (double)y[3]);
    return 1;
}

```

The expected printed row is `[-4.000000, -0.707107, 3.000000, 2.121320]`. A first coordinate of `-1` suggests adjacent pairing rather than split-half. A first input gradient of `-3` rather than `3` suggests applying forward again instead of the transpose. Those errors have different causes and should not be “fixed” by broadening a tolerance.

The constructor and transform tests are deliberately separate. A special quarter-turn table can expose rotation signs precisely, but it cannot prove that `pow` uses the right model exponent. Conversely, a correct frequency list cannot prove the coordinate pairing.

## Exercise padding, independent heads, and accumulation

The next fixture uses three rows, two heads per row, and head width four. Each 40-float array has a two-element prefix and physical row stride ten, even though only eight cells per row are logical. Prefix, row gaps, and suffix keep a sentinel value. SET destinations begin with NaNs in logical cells, making an accidental dependence on old output visible.

Positions are `[0,3,8191]`. The first row must equal its input numerically. Every head's squared norm should be preserved within tolerance. Applying backward to the rotated vector should approximately recover the input. Backward on an unrelated incoming-gradient array establishes a SET result, then two ADD calls must retain a preexisting contribution of 1.25.

These tests are not all independent proofs of model compatibility. A consistently wrong pairing can pass them. They verify memory traversal, transform reversibility, and accumulation after the hand fixture establishes the intended convention.

```c title="tests/test_rope.c"
static DlMatrix padded(float *p)
{
    return (DlMatrix){p, 40, 2, 3, 8, 10};
}

static int guards(const float *p)
{
    for (size_t j = 0; j < 40; ++j) {
        const int logical = j >= 2 && (j - 2) / 10 < 3 && (j - 2) % 10 < 8;
        if (!logical && p[j] != -777.0f) return 0;
    }
    return 1;
}

static int padded_modes_and_norms(void)
{
    float x[40], y[40], dy[40], dx[40], recovered[40], set_dx[40];
    for (size_t j = 0; j < 40; ++j) {
        x[j] = y[j] = dy[j] = dx[j] = recovered[j] = -777.0f;
    }
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            const size_t at = 2 + i * 10 + j;
            x[at] = (float)((int)(i * 8 + j) - 10) / 7.0f;
            dy[at] = (float)((int)((i * 8 + j) % 9) - 4) / 5.0f;
            y[at] = dx[at] = recovered[at] = NAN;
        }
    }
    float saved_x[40], saved_dy[40];
    memcpy(saved_x, x, sizeof x); memcpy(saved_dy, dy, sizeof dy);
    const size_t positions[3] = {0, 3, 8191};
    double f[2];
    if (!status_is(dl_rope_frequencies(4, 100000.0, f, 2), DL_OK, "padded f") ||
        !status_is(dl_rope_forward(dl_matrix_read(padded(x)), 4, positions, 3,
                   f, 2, padded(y)), DL_OK, "padded forward") ||
        !status_is(dl_rope_backward(dl_matrix_read(padded(y)), 4, positions, 3,
                   f, 2, padded(recovered), DL_SET), DL_OK, "inverse rotation") ||
        !status_is(dl_rope_backward(dl_matrix_read(padded(dy)), 4, positions, 3,
                   f, 2, padded(dx), DL_SET), DL_OK, "padded backward")) return 0;
    memcpy(set_dx, dx, sizeof dx);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t h = 0; h < 2; ++h) {
            const size_t start = 2 + i * 10 + h * 4;
            const double n = dot(x + start, x + start, 4);
            if (!near_value(dot(y + start, y + start, 4), n,
                            2e-6 * (1.0 + n), "head squared norm")) return 0;
        }
        for (size_t j = 0; j < 8; ++j) {
            const size_t at = 2 + i * 10 + j;
            if (!near_value(recovered[at], x[at], 1e-6, "round trip") ||
                (i == 0 && !near_value(y[at], x[at], 0.0, "position zero"))) {
                return 0;
            }
            dx[at] = 1.25f;
        }
    }
    for (int pass = 1; pass <= 2; ++pass) {
        if (!status_is(dl_rope_backward(dl_matrix_read(padded(dy)), 4,
                       positions, 3, f, 2, padded(dx), DL_ADD), DL_OK,
                       "repeated backward ADD")) return 0;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                const size_t at = 2 + i * 10 + j;
                if (!near_value(dx[at], 1.25 + (double)pass * (double)set_dx[at],
                                4e-7, "ADD retains prior gradient")) return 0;
            }
        }
    }
    return require(guards(y) && guards(dx) && guards(recovered), "changed padding") &&
           require(memcmp(x, saved_x, sizeof x) == 0 &&
                   memcmp(dy, saved_dy, sizeof dy) == 0, "changed input");
}

```

`guards` uses short-circuit evaluation before subtracting the two-element prefix, so its unsigned arithmetic does not underflow for indices zero and one. The input byte comparisons are legitimate unchanged-storage checks: they compare an array with a copied image of its own original bytes, not independent floating-point calculations.

The ADD expectation uses the float SET result as a convenient nearby value. The actual routine computes a double contribution and combines it with the old float before narrowing; it does not necessarily store the same rounded contribution as a SET call before adding. The tolerance covers that final-rounding difference without excusing a missing whole contribution.

## Make the absolute-position contract executable

Process four rows at positions `[7,8,9,10]` in one call, then process each row independently with its original absolute position. The matrix views retain the original allocation bases and full counts while changing `offset`. The position pointer can point to one element of the separate `size_t` array, since that API uses a pointer/count rather than the canonical-base matrix-view scheme.

The two paths perform the same scalar operations on the same inputs in the same build; here a byte-exact comparison is meaningful. It is not a claim of bitwise equality across platforms, math libraries, or independently imported model implementations.

A second call rotates only the first head copied from every row. Its result must match the first head of the two-head call. Frequencies belong to head width, not head count. A reset-position fixture then repeats both contents and positions across two flattened sequences.

Finally, positions $2^{24}$ and $2^{24}+1$ are applied to the unit vector $(1,0)$ with frequency one. Their outputs should be $(\cos p,\sin p)$ with distinct angles. A float cast of the position would collapse these neighboring integers, even though ordinary short-context tests could still pass.

```c title="tests/test_rope.c"
static int sequence_and_positions(void)
{
    float x[32], full[32], chunks[32], keys[16], rotated_keys[16];
    for (size_t j = 0; j < 32; ++j) x[j] = (float)((int)(j % 11) - 5) / 3.0f;
    const size_t positions[4] = {7, 8, 9, 10};
    double f[2];
    if (!status_is(dl_rope_frequencies(4, 100000.0, f, 2), DL_OK, "sequence f") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(x, 4, 8)), 4,
                   positions, 4, f, 2, dense(full, 4, 8)), DL_OK, "whole sequence")) {
        return 0;
    }
    for (size_t i = 0; i < 4; ++i) {
        /* Retain canonical allocation bases, not x+i*8 and chunks+i*8. */
        const DlConstMatrix in = {x, 32, i * 8, 1, 8, 8};
        const DlMatrix out = {chunks, 32, i * 8, 1, 8, 8};
        if (!status_is(dl_rope_forward(in, 4, &positions[i], 1, f, 2, out),
                       DL_OK, "one-row chunk")) return 0;
        for (size_t j = 0; j < 4; ++j) keys[i * 4 + j] = x[i * 8 + j];
    }
    if (!require(memcmp(full, chunks, sizeof full) == 0,
                 "chunk changed absolute-position result") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(keys, 4, 4)), 4,
                   positions, 4, f, 2, dense(rotated_keys, 4, 4)), DL_OK,
                   "one key head versus two query heads")) return 0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (!near_value(rotated_keys[i * 4 + j], full[i * 8 + j], 0.0,
                            "frequency independent of head count")) return 0;
        }
    }

    const size_t reset_positions[4] = {4, 5, 4, 5};
    memcpy(x + 16, x, 16 * sizeof(*x));
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(x, 4, 8)), 4,
                   reset_positions, 4, f, 2, dense(full, 4, 8)), DL_OK,
                   "two flattened batch sequences") ||
        !require(memcmp(full, full + 16, 16 * sizeof(*full)) == 0,
                 "batch position reset lost")) return 0;

    float unit[4] = {1.0f, 0.0f, 1.0f, 0.0f}, large[4];
    const size_t large_p[2] = {16777216, 16777217};
    const double unit_f[1] = {1.0};
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(unit, 2, 2)), 2,
                   large_p, 2, unit_f, 1, dense(large, 2, 2)), DL_OK,
                   "integer to double positions")) return 0;
    for (size_t i = 0; i < 2; ++i) {
        if (!near_value(large[i * 2], cos((double)large_p[i]), 4e-8, "large cos") ||
            !near_value(large[i * 2 + 1], sin((double)large_p[i]), 4e-8,
                        "large sin")) return 0;
    }
    return require(fabs((double)large[0] - (double)large[2]) > 0.1,
                   "neighboring large positions collapsed");
}

```

The large-position check demonstrates representation, not long-context model quality. It requires no allocation proportional to the position: there are just two rows. An absolute token position is a scalar input, not an instruction to allocate all preceding rows.

## Avoid a hidden power-of-two or short-context assumption

Head widths two, four, six, and sixty-four are all even and valid. Width six exposes an implementation that only handles powers of two. Each case has two heads and is exercised at positions zero, one, 8191, and the API's maximum accepted position. These are norm checks; they do not assert a small common-shift error for every very large angle.

```c title="tests/test_rope.c"
static int widths_and_limits(void)
{
    const size_t widths[4] = {2, 4, 6, 64};
    const size_t positions[4] = {0, 1, 8191, DL_ROPE_MAX_POSITION};
    float x[128], y[128];
    for (size_t j = 0; j < 128; ++j) x[j] = (float)((int)(j % 13) - 6) / 8.0f;
    for (size_t w = 0; w < 4; ++w) {
        const size_t d = widths[w];
        double f[32];
        if (!status_is(dl_rope_frequencies(d, 100000.0, f, d / 2), DL_OK,
                       "varied width frequencies")) return 0;
        const DlConstMatrix input = {x, 128, 0, 1, 2 * d, 2 * d};
        const DlMatrix output = {y, 128, 0, 1, 2 * d, 2 * d};
        for (size_t p = 0; p < 4; ++p) {
            if (!status_is(dl_rope_forward(input, d,
                           &positions[p], 1, f, d / 2, output),
                           DL_OK, "varied width rotation")) return 0;
            for (size_t h = 0; h < 2; ++h) {
                const double n = dot(x + h * d, x + h * d, d);
                if (!near_value(dot(y + h * d, y + h * d, d), n,
                                2e-6 * (1.0 + n), "varied width norm")) return 0;
            }
        }
    }
    return 1;
}

```

The test arrays have 128 floats and the temporary frequency array has 32 doubles. The two matrix views retain the full 128-element capacity while exposing only `2*d` logical columns, so all these fixtures keep truthful canonical bases and allocation counts. The helper's fixed arrays are not the library's dimension limit. Larger callers must allocate truthful storage and supply validated views.

## Check the relative-position identity without attention infrastructure

Use $q=[1,2,3,4]$, $k=[-2,0.5,1,-1]$, width four, and base 10000. Rotate the query at position 3 and the key at position 9. Then shift both by 11. Compare those dot products with $q^{\mathsf T}R(6)k$, because $9-3=6$. Moving only the key is intentionally checked **not** to preserve the result.

In exact arithmetic the two split-half contributions are

$$
\begin{aligned}
q^{\mathsf T}R(6)k
&=\cos 6-7\sin 6\\
&\quad-3\cos(0.06)+4\sin(0.06).
\end{aligned}
$$

The first plane uses $(q_0,q_2)=(1,3)$ and $(k_0,k_2)=(-2,1)$, giving dot coefficient 1 and sine coefficient $-7$. The second uses $(2,4)$ and $(0.5,-1)$, giving coefficients $-3$ and 4. This supplies an additional way to inspect the relative-angle sign by hand. The implementation comparison allows small errors from two separate float rotation results versus a single relative rotation.

```c title="tests/test_rope.c"
static int relative_dot(void)
{
    float q[4] = {1.0f, 2.0f, 3.0f, 4.0f}, k[4] = {-2.0f, 0.5f, 1.0f, -1.0f};
    float rq[4], rk[4], shifted_q[4], shifted_k[4], relative_k[4];
    const size_t p[1] = {3}, n[1] = {9}, ps[1] = {14}, ns[1] = {20}, delta[1] = {6};
    double f[2];
    if (!status_is(dl_rope_frequencies(4, 10000.0, f, 2), DL_OK, "relative f") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(q, 1, 4)), 4,
                   p, 1, f, 2, dense(rq, 1, 4)), DL_OK, "query at m") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4,
                   n, 1, f, 2, dense(rk, 1, 4)), DL_OK, "key at n") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(q, 1, 4)), 4,
                   ps, 1, f, 2, dense(shifted_q, 1, 4)), DL_OK, "shifted query") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4,
                   ns, 1, f, 2, dense(shifted_k, 1, 4)), DL_OK, "shifted key") ||
        !status_is(dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4,
                   delta, 1, f, 2, dense(relative_k, 1, 4)), DL_OK,
                   "relative key")) return 0;
    const double original = dot(rq, rk, 4), shifted = dot(shifted_q, shifted_k, 4);
    if (!near_value(shifted, original, 3e-6, "common position shift") ||
        !near_value(dot(q, relative_k, 4), original, 3e-6, "relative angle n-m") ||
        !require(fabs(dot(rq, shifted_k, 4) - original) > 0.1,
                 "moving only key unexpectedly invariant")) return 0;
    printf("relative dot: original=%.8f, common-shift=%.8f\n", original, shifted);
    return 1;
}

```

A representative run prints an original value around `0.16133285` and a common-shift value around `0.16133332`. Their difference is much smaller than the fixture tolerance; exact last digits are not the cross-platform contract. A value being small does not by itself mean the operator is wrong: the four signed terms above substantially cancel.

## Reject malformed calls before modifying the destination

`unchanged` combines a status expectation with a byte comparison against the old output. The tests reject odd/zero head width, incompatible columns, wrong array lengths, null metadata pointers, an invalid mode, bad matrix geometry, oversized positions, invalid frequencies, nonfinite input, and nonfinite old ADD storage.

The existing geometry validator checks `offset >= count` before computing a span. An offset of `SIZE_MAX` therefore reports `DL_RANGE`, not an arithmetic overflow. A separate malformed row-stride product exercises `DL_OVERFLOW`. These expected statuses follow the actual validation order; they are not interchangeable labels for “something went wrong.”

Two strided views whose logical cells alternate within the same allocation still have overlapping bounding spans and are rejected. Two nonoverlapping contiguous slices of that allocation are accepted. No test relies on subtracting unrelated C pointers or on a fabricated allocation capacity surviving validation.

```c title="tests/test_rope.c"
static int unchanged(DlStatus got, DlStatus expected, const float *out,
                     const float *before, size_t count, const char *where)
{
    return status_is(got, expected, where) &&
           require(memcmp(out, before, count * sizeof(*out)) == 0,
                   "rejected call modified output");
}

static int rejected_calls(void)
{
    float x[4] = {1, 2, 3, 4}, y[4] = {9, 8, 7, 6};
    float before[4]; memcpy(before, y, sizeof y);
    size_t p[1] = {1};
    double f[2] = {1.0, 0.01};
    const DlConstMatrix in = dl_matrix_read(dense(x, 1, 4));
    const DlMatrix out = dense(y, 1, 4);
    if (!unchanged(dl_rope_forward(in, 3, p, 1, f, 2, out), DL_SHAPE,
                   y, before, 4, "reject odd width") ||
        !unchanged(dl_rope_forward(in, 0, p, 1, f, 2, out), DL_SHAPE,
                   y, before, 4, "reject zero width") ||
        !unchanged(dl_rope_forward(in, 6, p, 1, f, 3, out), DL_SHAPE,
                   y, before, 4, "head does not divide columns") ||
        !unchanged(dl_rope_forward(in, 4, p, 0, f, 2, out), DL_SHAPE,
                   y, before, 4, "wrong position count") ||
        !unchanged(dl_rope_forward(in, 4, p, 1, f, 1, out), DL_SHAPE,
                   y, before, 4, "wrong frequency count") ||
        !unchanged(dl_rope_forward(in, 4, NULL, 1, f, 2, out), DL_ARGUMENT,
                   y, before, 4, "null positions") ||
        !unchanged(dl_rope_forward(in, 4, p, 1, NULL, 2, out), DL_ARGUMENT,
                   y, before, 4, "null frequencies") ||
        !unchanged(dl_rope_backward(in, 4, p, 1, f, 2, out, (DlWriteMode)7),
                   DL_ARGUMENT, y, before, 4, "invalid mode")) return 0;
    DlMatrix bad = out; bad.cols = 2;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, bad), DL_SHAPE,
                   y, before, 4, "output shape")) return 0;
    bad = out; bad.offset = SIZE_MAX;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, bad), DL_RANGE,
                   y, before, 4, "offset outside allocation")) return 0;
    bad = out; bad.rows = 3; bad.ld = SIZE_MAX / 2 + 1;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, bad), DL_OVERFLOW,
                   y, before, 4, "view arithmetic overflow")) return 0;
    float before_x[4]; memcpy(before_x, x, sizeof x);
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, dense(x, 1, 4)),
                   DL_ALIAS, x, before_x, 4, "no in-place rotation")) return 0;
    if (SIZE_MAX > DL_ROPE_MAX_POSITION) {
        p[0] = SIZE_MAX;
        if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, out), DL_RANGE,
                       y, before, 4, "oversized position")) return 0;
        p[0] = 1;
    }
    const double invalid_f[4] = {0.0, -1.0, 2.0, NAN};
    for (size_t j = 0; j < 4; ++j) {
        f[1] = invalid_f[j];
        if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, out), DL_NUMERIC,
                       y, before, 4, "invalid frequency")) return 0;
    }
    f[1] = 0.01; x[3] = NAN;
    if (!unchanged(dl_rope_forward(in, 4, p, 1, f, 2, out), DL_NUMERIC,
                   y, before, 4, "late nonfinite input preflight")) return 0;
    x[3] = 4; y[3] = NAN; memcpy(before, y, sizeof y);
    if (!unchanged(dl_rope_backward(in, 4, p, 1, f, 2, out, DL_ADD), DL_NUMERIC,
                   y, before, 4, "nonfinite old ADD value")) return 0;

    float shared[16];
    for (size_t j = 0; j < 16; ++j) shared[j] = (float)j;
    float saved[16]; memcpy(saved, shared, sizeof shared);
    const size_t pp[2] = {1, 2};
    const DlConstMatrix strided = {shared, 16, 0, 2, 4, 8};
    const DlMatrix gaps = {shared, 16, 4, 2, 4, 8};
    if (!unchanged(dl_rope_forward(strided, 4, pp, 2, f, 2, gaps), DL_ALIAS,
                   shared, saved, 16, "conservative bounding-span alias")) return 0;
    const DlConstMatrix first = {shared, 16, 0, 1, 4, 4};
    const DlMatrix separate = {shared, 16, 4, 1, 4, 4};
    return status_is(dl_rope_forward(first, 4, p, 1, f, 2, separate), DL_OK,
                     "disjoint spans in one allocation") &&
           require(memcmp(shared, saved, 4 * sizeof(*shared)) == 0,
                   "disjoint input changed");
}

```

The oversized-position test is conditional because it needs `size_t` to represent a value larger than the API cap. The intended native Mac and the verification host both have that capacity. This is a test portability guard, not permission for the model to silently truncate a larger position.

The constructor checks and the transform checks use distinct status rules: an invalid model base is `DL_ARGUMENT`; an invalid numerical frequency supplied to a transform is `DL_NUMERIC`. After any unsuccessful call, examine the returned status and follow the operator's documented output rules rather than assuming a particular validation stage was reached.

## Observe a partial numerical failure

The first case below uses the overflowing rotation from Part 1. Coordinates zero and two are committed by the first plane. Coordinates one and three retain their old values because the second plane's two candidates are checked before either store.

The second case uses position zero, so the transform itself is an identity. ADD still overflows when the incoming coordinate and old destination are both `FLT_MAX`. Successful earlier pairs remain written. This separates rotation arithmetic from accumulation arithmetic and demonstrates why retrying ADD on the same result is unsafe.

```c title="tests/test_rope.c"
static int partial_failures(void)
{
    const double f[2] = {acos(-1.0) / 4.0, acos(-1.0) / 4.0};
    const size_t p[1] = {1};
    float x[4] = {3.0f, FLT_MAX, 4.0f, FLT_MAX}, y[4] = {9, 9, 9, 9};
    if (!status_is(dl_rope_forward(dl_matrix_read(dense(x, 1, 4)), 4, p, 1,
                   f, 2, dense(y, 1, 4)), DL_NUMERIC, "rotation overflow") ||
        !near_value(y[0], -1.0 / sqrt(2.0), 3e-7, "earlier first committed") ||
        !near_value(y[2], 7.0 / sqrt(2.0), 4e-7, "earlier second committed") ||
        !require(y[1] == 9 && y[3] == 9, "failing pair partially committed")) return 0;
    const size_t zero[1] = {0};
    float incoming[4] = {1, FLT_MAX, 2, 0}, dx[4] = {10, FLT_MAX, 20, 13};
    return status_is(dl_rope_backward(dl_matrix_read(dense(incoming, 1, 4)), 4,
                     zero, 1, f, 2, dense(dx, 1, 4), DL_ADD), DL_NUMERIC,
                     "ADD overflow") &&
           require(dx[0] == 11 && dx[2] == 22 && dx[1] == FLT_MAX && dx[3] == 13,
                   "ADD partial-pair state");
}

```

An AddressSanitizer success is not a numerical success. These failures are expected range checks on valid memory, and should produce `DL_NUMERIC` without a memory diagnostic. Conversely, invalid pointers or an untruthful capacity remain caller bugs; the API cannot convert arbitrary undefined behavior into a status code.

## Differentiate a scalar objective built entirely in C

For the isolated operator, define

$$
L(X)=\sum_{i=0}^{2}\sum_{j=0}^{7}G_{ij}\,\operatorname{RoPE}(X)_{ij},
$$

where $G$ is a fixed incoming-gradient array. Its analytic derivative is the transpose rotation applied to $G$. The `rotation_objective` helper performs the actual C forward call and a double dot product; no Python function replaces the computation being tested.

For each input coordinate we perturb the actual float storage at three nominal steps. Let $x_+$ and $x_-$ be the represented float endpoints. The numerical derivative is

$$
\widehat g=\frac{L(x_+)-L(x_-)}{\operatorname{double}(x_+)-\operatorname{double}(x_-)}.
$$

Using the represented spacing avoids pretending that both nominal increments survived float rounding exactly. It does not remove rounding from the forward outputs. The tested function is linear before floating-point stores, so a discrepancy here is dominated by arithmetic rounding rather than a nonlinear finite-difference truncation term. There are $24\times3=72$ comparisons.

```c title="tests/test_rope.c"
static DlStatus rotation_objective(float *x, const float *incoming,
                                    const size_t *p, const double *f, double *loss)
{
    float y[24];
    const DlStatus s = dl_rope_forward(dl_matrix_read(dense(x, 3, 8)), 4,
                                      p, 3, f, 2, dense(y, 3, 8));
    if (s != DL_OK) return s;
    *loss = dot(y, incoming, 24);
    return DL_OK;
}

static int derivative(double analytic, double numeric, double *worst)
{
    const double tolerance = 2e-4 + 3e-4 * fmax(fabs(analytic), fabs(numeric));
    const double ratio = fabs(analytic - numeric) / tolerance;
    if (isfinite(ratio) && ratio > *worst) *worst = ratio;
    return near_value(numeric, analytic, tolerance, "finite difference");
}

static int isolated_gradients(size_t *checks, double *worst)
{
    float x[24], incoming[24], dx[24];
    for (size_t j = 0; j < 24; ++j) {
        x[j] = (float)((int)(j % 11) - 5) / 7.0f;
        incoming[j] = (float)((int)(j % 7) - 3) / 5.0f;
    }
    const size_t p[3] = {0, 3, 8191};
    double f[2], baseline = 0.0;
    if (!status_is(dl_rope_frequencies(4, 100000.0, f, 2), DL_OK, "gradient f") ||
        !status_is(rotation_objective(x, incoming, p, f, &baseline), DL_OK,
                   "baseline objective") ||
        !status_is(dl_rope_backward(dl_matrix_read(dense(incoming, 3, 8)), 4,
                   p, 3, f, 2, dense(dx, 3, 8), DL_SET), DL_OK,
                   "analytic gradient")) return 0;
    const double steps[3] = {0.005, 0.002, 0.001};
    for (size_t j = 0; j < 24; ++j) {
        const float saved = x[j];
        for (size_t h = 0; h < 3; ++h) {
            const float plus = (float)((double)saved + steps[h]);
            const float minus = (float)((double)saved - steps[h]);
            if (!require(plus > minus, "collapsed gradient interval")) return 0;
            double lp = 0.0, lm = 0.0;
            x[j] = plus;
            const DlStatus sp = rotation_objective(x, incoming, p, f, &lp);
            x[j] = minus;
            const DlStatus sm = rotation_objective(x, incoming, p, f, &lm);
            x[j] = saved;
            if (!status_is(sp, DL_OK, "plus objective") ||
                !status_is(sm, DL_OK, "minus objective")) return 0;
            const double numeric = (lp - lm) / ((double)plus - (double)minus);
            if (!derivative(dx[j], numeric, worst)) return 0;
            ++*checks;
        }
    }
    double replay = 0.0;
    return status_is(rotation_objective(x, incoming, p, f, &replay), DL_OK,
                     "replayed objective") &&
           near_value(replay, baseline, 0.0, "restored inputs");
}

```

The tolerance is $2\times10^{-4}+3\times10^{-4}\max(|g|,|\widehat g|)$ for these specific small fixtures. It is not a license to accept a transpose-sign error. `worst` records the largest error divided by its allowed tolerance; a passing finite ratio is at most one.

Each perturbation is restored **before** either status is checked. This avoids leaving the parameter altered on a reported forward failure. The final objective replay is exact within the same executable because the original float values, positions, and frequencies have been restored. Unlike the feed-forward cache in Chapter 8, these backward calls do not read saved activation arrays; restoring a frequency or position after the fact would still not excuse having used the wrong one during backward.

## Differentiate the query/key dot product through both rotations

Next define one unscaled score

$$
S(q,k)=\operatorname{RoPE}_{3}(q)^{\mathsf T}\operatorname{RoPE}_{9}(k).
$$

After forward, the incoming derivative for the rotated query is the rotated key, and vice versa. Apply backward with position 3 for the query and position 9 for the key. That tests two distinct reverse paths and their connection to the correct position metadata.

Perturb all four query coordinates and all four key coordinates at three steps, giving another $8\times3=24$ comparisons. The total is **96**. This is a scalar integration fixture, not a complete attention layer: there is no head sharing, score scaling, causal mask, or probability normalization to hide a positional error.

```c title="tests/test_rope.c"
static DlStatus score_objective(float *q, float *k, const double *f,
                                 float *rq, float *rk, double *score)
{
    const size_t qp[1] = {3}, kp[1] = {9};
    DlStatus s = dl_rope_forward(dl_matrix_read(dense(q, 1, 4)), 4, qp, 1,
                                 f, 2, dense(rq, 1, 4));
    if (s != DL_OK) return s;
    s = dl_rope_forward(dl_matrix_read(dense(k, 1, 4)), 4, kp, 1,
                        f, 2, dense(rk, 1, 4));
    if (s != DL_OK) return s;
    *score = dot(rq, rk, 4);
    return DL_OK;
}

static int dot_gradients(size_t *checks, double *worst)
{
    float q[4] = {1, 2, 3, 4}, k[4] = {-2, 0.5f, 1, -1};
    float rq[4], rk[4], dq[4], dk[4];
    double f[2], score = 0.0;
    const size_t qp[1] = {3}, kp[1] = {9};
    if (!status_is(dl_rope_frequencies(4, 10000.0, f, 2), DL_OK, "score f") ||
        !status_is(score_objective(q, k, f, rq, rk, &score), DL_OK, "score") ||
        !status_is(dl_rope_backward(dl_matrix_read(dense(rk, 1, 4)), 4, qp, 1,
                   f, 2, dense(dq, 1, 4), DL_SET), DL_OK, "score dQ") ||
        !status_is(dl_rope_backward(dl_matrix_read(dense(rq, 1, 4)), 4, kp, 1,
                   f, 2, dense(dk, 1, 4), DL_SET), DL_OK, "score dK")) return 0;
    const double baseline = score;
    const double steps[3] = {0.005, 0.002, 0.001};
    for (int which = 0; which < 2; ++which) {
        float *parameter = which == 0 ? q : k;
        const float *gradient = which == 0 ? dq : dk;
        for (size_t j = 0; j < 4; ++j) {
            const float saved = parameter[j];
            for (size_t h = 0; h < 3; ++h) {
                const float plus = (float)((double)saved + steps[h]);
                const float minus = (float)((double)saved - steps[h]);
                if (!require(plus > minus, "collapsed score interval")) return 0;
                double lp = 0.0, lm = 0.0;
                parameter[j] = plus;
                const DlStatus sp = score_objective(q, k, f, rq, rk, &lp);
                parameter[j] = minus;
                const DlStatus sm = score_objective(q, k, f, rq, rk, &lm);
                parameter[j] = saved;
                if (!status_is(sp, DL_OK, "score plus") ||
                    !status_is(sm, DL_OK, "score minus")) return 0;
                const double numeric = (lp - lm) / ((double)plus - (double)minus);
                if (!derivative(gradient[j], numeric, worst)) return 0;
                ++*checks;
            }
        }
    }
    return status_is(score_objective(q, k, f, rq, rk, &score), DL_OK,
                     "score replay") &&
           near_value(score, baseline, 0.0, "restored score");
}

```

`score_objective` overwrites its two temporary rotated vectors on every evaluation. The analytic `dq` and `dk` arrays are separate, so finite differences cannot overwrite the gradients they are being compared against. The last replay restores the forward outputs to the baseline inputs as well as checking the objective value.

One important limitation remains: finite differences compare a backward implementation with **its own forward function**. If both use adjacent pairing, both may be derivatives of the same unintended function. The hand-calculated split-half fixture tests a different question: whether that function is the one the model requires.

## Assemble the test executable

Append the complete driver. A failed check stops the suite with exit status one. A passing run prints the two numerical traces, the finite-difference count and worst ratio, and the final success line.

```c title="tests/test_rope.c"
int main(void)
{
    size_t checks = 0;
    double worst = 0.0;
    if (!frequencies_and_hand() || !padded_modes_and_norms() ||
        !sequence_and_positions() || !widths_and_limits() || !relative_dot() ||
        !rejected_calls() || !partial_failures() ||
        !isolated_gradients(&checks, &worst) || !dot_gradients(&checks, &worst)) {
        return 1;
    }
    printf("RoPE finite differences: %zu comparisons, worst error/tolerance = %.6f\n",
           checks, worst);
    puts("RoPE checks passed");
    return 0;
}
```

## Replace the Makefile and run the complete workspace

Replace the root `Makefile` with the following. It adds `src/rope.c`, `include/dl/rope.h`, and `test_rope` while retaining every earlier program and the separate debug/sanitizer/optimized directories. Tab indentation in the recipe lines matters. The target is still a compatible native Apple Clang toolchain; the language and diagnostic requirements have not changed.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope

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

From the learner workspace root on the Mac:

```sh
make build/debug/test_rope
./build/debug/test_rope
make test
```

The full test target runs all seven test executables and the existing comparison smoke call in each build configuration. A compatible compiler invoked directly can be selected without editing the Makefile:

```sh
make CC=clang test
ASAN_OPTIONS=detect_leaks=1 ./build/sanitize/test_rope
```

The optional leak-detection setting depends on the sanitizer platform. This chapter's library functions allocate no heap storage, so the key new memory checks are bounds, initialization discipline, and untouched inputs/padding. The cumulative project still includes its earlier ownership tests.

A representative **observed x86_64 Linux, Clang 17** run of the new executable is:

```text
split-half hand row: [-4.000000, -0.707107, 3.000000, 2.121320]
relative dot: original=0.16133285, common-shift=0.16133332
RoPE finite differences: 96 comparisons, worst error/tolerance = 0.197335
RoPE checks passed
```

Debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized cumulative builds were executed in that environment; the new test and inherited tests passed without compiler warnings. The new leak-enabled run reported no leaks. No M4/macOS execution, rotary performance measurement, pretrained-model comparison, or cache implementation is implied. The Mac commands describe how to run the same source natively; exact last printed digits are not promised to match this host.

## Four mistakes that pass different subsets of checks

Make each change separately in a disposable copy of the workspace, rebuild, inspect the first failure, and restore the source before trying the next. Do not broaden the tolerances or modify expected values to accommodate an intentional defect.

**Change the coordinate convention.** In `rope_apply`, replace the two index lines with:

```c
const size_t a = h * head_dim + 2 * j;
const size_t b = a + 1;
```

This produces adjacent pairing in both forward and backward. The first hand coordinate becomes `-1` instead of `-4`. Norm preservation still holds, and both finite-difference fixtures can still pass because the two directions agree on the wrong function. In an isolated verification run of this variant, the width/norm checks and all 96 gradient comparisons passed while the split-half hand fixture failed. The repair is the intended indexing, not more derivative tests of the same unintended forward.

**Use forward rotation during backward.** Replace `if (transpose) s = -s;` with `if (transpose) s = sin(angle);`. The latter recomputes the original sine instead of negating it. Forward output is unchanged, but the first hand gradient is `-3` instead of `3`. An isolated numerical derivative also disagrees. Check the transpose's off-diagonal signs before inspecting matrix weight orientations elsewhere in the model.

**Use the local row as the position.** Replace the angle expression with `(double)i * inv_freq[j]`. Whole-sequence rows get local positions 0, 1, 2, 3; every one-row call gets zero. The whole-sequence/chunk comparison fails even though each computed transform is still a rotation. Passing `positions[i]` is necessary for chunking and cache append semantics, not just an optional interface preference.

**Pass the integer through float.** Replace the angle expression with `(double)(float)positions[i] * inv_freq[j]`. Ordinary short positions and the hand fixture still work. The adjacent-large-position check fails: on the verification host the second cosine is about `0.626323` instead of `0.994384`. The missing integer bit was lost before conversion back to double; later double trigonometry cannot reconstruct it.

All four isolated variants returned explicit test failures without sanitizer memory diagnostics. They are mathematical/layout defects on valid arrays, which is why memory tooling alone cannot find them.

## A valid variation and a useful prediction

In `isolated_gradients` only, change the frequency constructor's base from `100000.0` to `10000.0`. Keep the same base/table for its forward and backward calls, and leave the checkpoint-frequency fixture and hand expectations untouched. The 96-comparison suite should still pass: the slow plane changes, but the derivative rule remains the transpose of the chosen transform. This variation was executed successfully in the verification environment.

Before running, predict which rows and planes change. Position-zero output cannot change. Plane zero has frequency one for either base and also cannot change. At nonzero positions, the second plane of a width-four head changes because its frequency moves from $1/\sqrt{100000}$ to $1/\sqrt{10000}$. A model configured with the original base would nevertheless be incompatible with an unexplained change to that base; mathematically valid is not the same as checkpoint-compatible.

## What is ready for attention

The workspace now has explicit split-half forward/backward rotations, a small model-specific frequency vector, and per-row absolute positions. Query and key calls can have different head counts without changing head width or frequencies. Backward can overwrite or add one contribution, and the tests distinguish convention errors, wrong reverse signs, lost absolute positions, representation loss, and numerical overflow.

The next chapter forms query, key, and value projections, maps query heads to shared key/value heads, applies these rotations, masks future scores, normalizes permitted scores, and accumulates values. That future integration must preserve this chapter's coordinate convention and absolute-position contract. It will not rotate values or silently change the frequency table to make an import comparison pass.

## References

The [RoFormer rotation construction and relative-position formulation][rope] underlie the identities checked here. The [selected model configuration][config] supplies the checkpoint-specific frequency/head settings. [WG14 N1570][c-draft], Sections 6.7.10, 7.12.4, and 7.12.7.4, describes the compile-time and trigonometric/power interfaces used in the complete C listings. All fixtures, expected numerical expressions, and diagnostics needed to run this chapter are supplied above.

[rope]: https://arxiv.org/html/2104.09864v5
[config]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
[c-draft]: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
