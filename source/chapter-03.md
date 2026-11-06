# Chapter 3 — Floating-point arithmetic and numerical derivatives

A buffer can have the right lifetime, extent, and stride and still contain the wrong numerical answer. Adding a small value to a large one can lose the small value. Subtracting nearby computed values can expose errors that were previously insignificant. A derivative check can become worse when its step becomes smaller, even though the differentiation formula is correct.

This chapter builds a standalone C experiment around those three observations. Its inputs are small enough to calculate by hand; its checks are explicit enough to fail on a wrong answer. The same habits will later apply to long matrix reductions, normalization, stable losses, and handwritten backward passes. No transformer knowledge or calculus course is required here.

> **Learning objectives**
> Distinguish stored precision from intermediate precision; inspect the floating-point format rather than guess it; reject non-finite results before applying tolerances; stabilize an exponential sum; derive a derivative and the chain rule from small changes; and interpret a finite-difference sweep without treating the smallest step as the best one.

**Prerequisites:** [Chapter 1 — Native C builds and a debuggable numerical program](https://25d-228.github.io/gpt-c/chapter-01.html) and [Chapter 2 — Owned buffers, borrowed views, and failure cleanup](https://25d-228.github.io/gpt-c/chapter-02.html). Their files remain unchanged. We reuse their pointer, array, struct, status-return, and build vocabulary, not their source modules. Every new function is included below.

**Implementation estimate: 2.25 hours**—1.25 for the program and build target, and 1.00 for running the checks and the three small variations. Reading is separate. Record actual hands-on time; the length of the explanation is not an additional coding estimate.

## Give the experiment its own directory

From the learner workspace root:

```sh title="Terminal — from the learner workspace root"
mkdir -p examples/03-numerics
cd examples/03-numerics
```

Write two persistent files, `numerics.c` and `Makefile`. Append every C block titled `examples/03-numerics/numerics.c` in order; together they form the complete translation unit. Blocks marked **Illustration** or **Variation** are not additional baseline source. The companion [numerics.c][source-c] and [Makefile][source-make] contain exactly the complete listings. `build/` holds only disposable outputs.

There is no allocation, dataset, Python dependency, or model state in this program. Arrays and result structs belong to their calling functions. Helpers borrow array elements or fill a caller-owned struct; they retain no pointer. This makes it possible to study arithmetic without simultaneously debugging a tensor allocator.

## Precision is a spacing, not a number of printed decimals

Think of a positive binary floating-point number as a significand multiplied by a power of two. For a **normal** value we can write

$$
x = (1.b_1b_2\ldots b_{p-1})_2\,2^e.
$$

Each $b_j$ is either zero or one; the subscript 2 means binary notation. The leading one and the following $p-1$ bits give $p$ bits of precision. The exponent $e$ moves the binary point. A sign supplies negative values. This is a mathematical description, not permission to reinterpret a float through an incompatible integer pointer. The [representation concepts][fp-concepts] distinguish significand precision from exponent range.

For the binary32 and binary64 formats used by this experiment, $p$ is 24 for `float` and 53 for `double`. The program checks those format characteristics. The [IEEE-format parameter examples][fp-ieee] give the corresponding C constants. Merely knowing `sizeof(float)` is not enough to establish a format.

Within the interval from $2^e$ up to, but not including, $2^{e+1}$, increasing the last significand bit changes the value by

$$
\operatorname{gap}=2^{e-(p-1)}.
$$

This follows directly from the place value of that last bit. For `float`, the spacing above one is $2^{-23}$, approximately $1.1920929\times10^{-7}$. Above $2^{24}=16\,777\,216$, the spacing is $2^{24-23}=2$. The representable values there begin `16777216`, `16777218`, `16777220`. `16777217` lies between two of them.

Our experiment uses round-to-nearest, with a halfway case choosing the value whose least significant retained bit is even. Consequently, storing `16777217` as binary32 produces `16777216`. This is not truncation of all floating-point values; it is the tie case for these particular neighbors. The [rounding description][fp-rounding] defines that rule. The program observes the active rounding mode and never changes it.

### Read the format through its headers

`<float.h>` describes the implementation's floating-point types. The `FLT_` prefix generally refers to `float`; `DBL_` refers to `double`. Some particularly useful quantities are:

| Name | Meaning for this experiment |
|---|---|
| `FLT_RADIX` | Radix of the significand representation; two here |
| `FLT_MANT_DIG`, `DBL_MANT_DIG` | Significand digits in that radix; 24 and 53 here |
| `FLT_EPSILON`, `DBL_EPSILON` | Distance from one to the next larger representable value |
| `FLT_MAX`, `DBL_MAX` | Largest positive finite value |
| `FLT_MIN` | Smallest positive **normal** float, not the most negative float |
| `FLT_TRUE_MIN` | Smallest positive float, including subnormal values |
| `FLT_EVAL_METHOD` | Whether expressions use their source type's precision or a wider evaluation format |

The [floating-point parameter documentation][fp-parameters] defines the precision, range, and epsilon constants. Epsilon is not a universal error allowance: its definition concerns spacing at one, not the total error of an arbitrary algorithm.

A **subnormal** value lies below the normal positive range and uses a reduced-precision representation near zero. For binary32, the minimum positive normal is $2^{-126}$, while the minimum positive subnormal is $2^{-149}$. Subnormal spacing remains $2^{-149}$ rather than shrinking with the magnitude, so relative precision deteriorates toward zero. An even smaller result can round to zero. The [classification interfaces][fp-classes] distinguish normal, subnormal, zero, infinity, and NaN. Our program prints the limits; it does not change the processor's subnormal handling.

`static` on a function at file scope gives it **internal linkage**: its name is private to this translation unit. It does not mean the function is executed only once, and it does not give its local variables permanent storage. There is no header because nothing in a second translation unit calls these helpers.

Begin `numerics.c` with the headers and the environment check:

```c title="examples/03-numerics/numerics.c — headers and environment"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static int expect(int condition, const char *name) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        return 0;
    }
    return 1;
}

static int check_environment(void) {
    printf("sizes: float=%zu double=%zu bytes\n", sizeof(float), sizeof(double));
    printf("radix=%d mantissa bits: float=%d double=%d eval_method=%d\n",
           FLT_RADIX, FLT_MANT_DIG, DBL_MANT_DIG, FLT_EVAL_METHOD);
    printf("epsilon: float=%.9g double=%.17g\n",
           (double)FLT_EPSILON, DBL_EPSILON);
    printf("float range: true_min=%.9g min_normal=%.9g max=%.9g\n",
           (double)FLT_TRUE_MIN, (double)FLT_MIN, (double)FLT_MAX);
    return expect(FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                  DBL_MANT_DIG == 53 && FLT_MAX_EXP == 128 &&
                  DBL_MAX_EXP == 1024 && FLT_EVAL_METHOD == 0 &&
                  fegetround() == FE_TONEAREST,
                  "binary32/binary64, source precision, nearest rounding");
}
```


`<fenv.h>` provides `fegetround` and `FE_TONEAREST`. `<math.h>` provides the classification, absolute-value, neighboring-value, exponential, and logarithm interfaces used below. `<stddef.h>` supplies `size_t`; `<stdio.h>` and `<stdlib.h>` supply output and exit status definitions. None of these headers allocates a tensor or supplies a differentiation engine.

The conjunction in `check_environment` rejects a different arithmetic configuration before applying this experiment's exact rounding expectations. `FLT_EVAL_METHOD == 0` asks for evaluation in the source type rather than a wider expression format. The range checks use the C exponent parameters, not a claim that all C implementations have identical formats. On the native target, use a normal compatible Clang configuration. On a different implementation, inspect a failed environment check rather than removing it to obtain a passing message.

`%zu` prints a `size_t`. `%g` selects a compact decimal representation, and `.17` asks for 17 significant digits. Promoting a float to double for `printf` cannot recover digits lost earlier; it only prints the stored float value in a wider format. We use ordinary return-value checks instead of relying on assertions that a build could disable.

## Watch an addition disappear

Consider the three-element array

$$
a=[16\,777\,216,\;1,\;-16\,777\,216].
$$

Its exact sum is one. Sequential float accumulation visits these states:

| Input just consumed | Exact addition at this step | Stored float accumulator |
|---|---:|---:|
| `16777216` | $0+16\,777\,216$ | `16777216` |
| `1` | $16\,777\,216+1$ | `16777216` |
| `-16777216` | $16\,777\,216-16\,777\,216$ | `0` |

The second addition rounded away the unit. The final subtraction is exact for its already-rounded operands; it cannot restore what was lost. This distinction matters: cancellation is not a rule that every subtraction is inaccurate. Subtracting nearby computed values can expose **earlier** errors because the desired result is much smaller than either operand.

Reorder the inputs to `[16777216, -16777216, 1]`. The running results become `16777216`, `0`, `1`. Use a double accumulator on the original input order, and `16777217` is representable at the intermediate step, so the final result is also one.

Append the reductions and their driver:

```c title="examples/03-numerics/numerics.c — summation and rounding"
/* Borrow n initialized elements; NULL is permitted only when n == 0. */
static float sum_f32(const float *values, size_t n) {
    float total = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        total += values[i];
    }
    return total;
}

static double sum_f64(const float *values, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += (double)values[i];
    }
    return total;
}

static int check_rounding(void) {
    const float values[3] = {16777216.0f, 1.0f, -16777216.0f};
    const float reordered[3] = {16777216.0f, -16777216.0f, 1.0f};
    const float narrow = sum_f32(values, 3);
    const float reordered_sum = sum_f32(reordered, 3);
    const double wide = sum_f64(values, 3);
    const float stored = (float)16777217.0;
    const float tenth = 0.1f;
    const float gap_one = nextafterf(1.0f, INFINITY) - 1.0f;
    const float gap_large = nextafterf(values[0], INFINITY) - values[0];
    printf("gaps above 1 and 2^24: %.9g and %.9g\n",
           (double)gap_one, (double)gap_large);
    printf("sum: float=%.0f reordered=%.0f double=%.0f late_cast=%.0f\n",
           (double)narrow, (double)reordered_sum, wide, (double)narrow);
    printf("stored 16777217 as float, then widened: %.0f\n", (double)stored);
    printf("0.1f widened=%.17g; 0.1 double=%.17g\n", (double)tenth, 0.1);
    return expect(narrow == 0.0f && reordered_sum == 1.0f && wide == 1.0,
                  "known summation-order results") &&
           expect(stored == 16777216.0f && gap_one == FLT_EPSILON &&
                  gap_large == 2.0f, "known rounding and spacing") &&
           expect(sum_f32(NULL, 0) == 0.0f && sum_f64(NULL, 0) == 0.0,
                  "empty reductions access no input");
}
```

Both loops visit contiguous float elements at offsets `i * sizeof(float)`. `sum_f64` does not copy the input to a double array. It converts one loaded float to double, adds it to one double accumulator, and advances. The input still occupies three float elements.

For `n > 0`, the caller must supply that many initialized live elements. A null pointer is allowed only for `n == 0`, when no subscript is evaluated. These low-level reduction helpers have the same borrowed-extent convention as Chapter 1; they do not attempt to validate arbitrary pointers or reject non-finite elements. The fixtures here are finite, and the later checked exponential helper will have a different explicit validation boundary.

`nextafterf(x, INFINITY)` asks for the next representable float above `x`; subtracting `x` measures the local gap. The [neighboring-value interface][fp-neighbor] defines the direction argument and result type. `INFINITY` is used as a direction, not as an element to add to the sum.

The `stored` example isolates **storage rounding**: `16777217.0` starts as a representable double and the explicit `(float)` conversion narrows it. Converting `stored` back to double therefore gives `16777216`. The `tenth` example makes a similar point with a familiar decimal: neither decimal printing nor widening makes the original `0.1f` exact.

### Place the conversion before the operation that needs it

The same decision will apply to a dot product. For float inputs and a double accumulator, these expressions are different:

```c title="Illustration — alternative expressions inside a future dot-product loop"
/* Multiply in double, then accumulate in double. */
total += (double)a[i] * (double)b[i];

/* Multiply in float first; widening happens only after that product. */
total += (double)(a[i] * b[i]);
```

Here `total` would be a double and `a` and `b` would be valid float input arrays. These are alternatives, not two consecutive additions to paste into the current program. In the second expression, any product rounding or float-range overflow has already happened before the cast. Choosing a wider destination does not retroactively widen all subexpressions.

Double accumulation is useful as a small reference calculation, not a guarantee of an exact answer for arbitrary inputs. It has finite precision too. Nor does this experiment establish which accumulator is faster on the M4. Keep storage, arithmetic precision, order of reduction, and measured performance as separate questions.

## A tolerance check must first decide whether there is a number to compare

An infinity is not a very large finite model result. A NaN, meaning “not a number,” is not a large error that an ordinary comparison will necessarily reject. IEEE-style arithmetic can produce these values through range overflow or invalid operations; `<math.h>` provides classification rather than requiring us to inspect representation bits. See [infinity and NaN behavior][fp-special].

A particularly dangerous test is:

```c title="Illustration — a rejection-only test is insufficient"
if (fabs(actual - expected) > tolerance) {
    return 0;
}
return 1;
```

When the difference is NaN, the greater-than comparison is false. The code falls through to success. It is not enough to write “fail on a large error”; the test must require a **finite acceptable error**. Similarly, checking `actual == expected` before rejecting non-finite values would accept two positive infinities. That is not acceptable for our model-result checks.

For finite values $a$ and $b$, define an absolute-plus-relative comparison:

$$
|a-b| \leq a_{\mathrm{tol}} + r_{\mathrm{tol}}\max(|a|,|b|).
$$

Here $a$ is the actual result, $b$ is the reference, $a_{\mathrm{tol}}$ is a nonnegative absolute tolerance in the result's units, and $r_{\mathrm{tol}}$ is a nonnegative dimensionless relative tolerance. This chapter uses a symmetric scale so neither argument is silently privileged.

Near zero, use the absolute term. Comparing zero with $10^{-9}$ under an absolute tolerance of $10^{-8}$ succeeds without dividing by a zero reference. At a scale near one million, comparing `1000001` with `1000000` under relative tolerance $2\times10^{-6}$ allows an error of about two, so a difference of one passes. Neither example justifies using those tolerances for every later kernel.

Append the comparison helper and its test table:

```c title="examples/03-numerics/numerics.c — finite-aware comparisons"
/* Conservative test helper: invalid or overflowing error bounds fail. */
static int close_finite(double actual, double expected,
                        double atol, double rtol) {
    if (!isfinite(actual) || !isfinite(expected) ||
        !isfinite(atol) || !isfinite(rtol) || atol < 0.0 || rtol < 0.0) {
        return 0;
    }
    const double error = fabs(actual - expected);
    const double scale = fmax(fabs(actual), fabs(expected));
    const double bound = atol + rtol * scale;
    return isfinite(error) && isfinite(bound) && error <= bound;
}

static int check_comparisons(void) {
    typedef struct {
        double actual, expected, atol, rtol;
        int wanted;
        const char *name;
    } Case;
    const Case cases[] = {
        {0.0, 1e-9, 1e-8, 0.0, 1, "absolute tolerance near zero"},
        {1000001.0, 1000000.0, 0.0, 2e-6, 1, "relative tolerance at scale"},
        {1.0, 1.1, 1e-8, 1e-6, 0, "ordinary mismatch"},
        {0.0, -0.0, 0.0, 0.0, 1, "signed zeros are finite and equal"},
        {NAN, 1.0, 1e-8, 1e-6, 0, "NaN actual"},
        {1.0, NAN, 1e-8, 1e-6, 0, "NaN expected"},
        {INFINITY, INFINITY, 1e-8, 1e-6, 0, "equal infinities still fail"},
        {-INFINITY, 0.0, 1e-8, 1e-6, 0, "negative infinity"},
        {1.0, 1.0, -1.0, 0.0, 0, "negative tolerance"},
        {1.0, 1.0, NAN, 0.0, 0, "NaN tolerance"},
        {1.0, 1.0, 0.0, INFINITY, 0, "infinite tolerance"},
        {DBL_MAX, -DBL_MAX, 0.0, 2.0, 0, "overflowed error and bound"}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const Case c = cases[i];
        if (!expect(close_finite(c.actual, c.expected, c.atol, c.rtol) ==
                    c.wanted, c.name)) {
            return 0;
        }
    }
    const double nan_error = fabs((double)NAN - 1.0);
    printf("NaN: rejection-only (error > tolerance)=%d; finite check=%d\n",
           nan_error > 1e-6, close_finite(NAN, 1.0, 1e-8, 1e-6));
    return 1;
}
```

`isfinite` accepts any of the ordinary floating-point types and excludes infinities and NaNs; its [documented semantics][fp-classes] are exactly what the first guard needs. `fabs` takes the magnitude of a double. `fmax` selects a larger value, but is not our substitute for non-finite validation. We reject invalid operands before using it.

The final line checks **both** the computed error and the computed bound. Even finite double inputs can yield an overflowing subtraction or an overflowing tolerance expression. The final test case intentionally exercises that on the required IEEE-style environment: a naive `infinity <= infinity` would succeed. This helper instead reports failure. It is a conservative diagnostic, not a fully scaled comparison algorithm for arbitrary values near `DBL_MAX`; an overflowed bound is rejected even if a real-arithmetic inequality with an enormous tolerance would be true. Normal model checks should use meaningful small tolerances, not enlarge them until the bound overflows.

The test table is an array of structs. Each row supplies four operands, the wanted Boolean result, and a diagnostic name. `const Case c = cases[i]` copies a small value; unlike a live owner from Chapter 2, it has no resource that could be freed twice. `c.name` borrows a string literal that exists throughout the program. `sizeof(cases) / sizeof(cases[0])` counts actual array elements at the declaration's scope.

The program explicitly supplies a quiet `NAN` and infinities through `INFINITY` rather than using an integer division by zero to manufacture a failure. It runs with the normal nontrapping floating-point environment and does not inspect exception flags or promise to preserve them. The numerical test's output-state guarantees below concern result structs, not the floating-point status register.

## Keep an exponential sum in a useful scale

Let a vector of three scores be $s=[1000,1001,1002]$. We want information about

$$
Z=\sum_{i=0}^{2}\exp(s_i).
$$

The index $i$ selects one scalar from the three-element array. $Z$ is a scalar, not another array. `exp` denotes $e$ raised to its argument; `log` below denotes the natural logarithm. Their C interfaces and range behavior are described in [exponentials and logarithms][fp-exp].

Directly evaluating these exponentials exceeds float range. Switching the entire calculation to double does not rescue `exp(1000)` either. The useful transformation is algebraic, not merely a wider type.

Set $m=\max_i s_i$ and define a shifted sum $\ell$:

$$
\ell=\sum_{i=0}^{2}\exp(s_i-m),\qquad
Z=\exp(m)\ell.
$$

This follows from $\exp(s_i)=\exp(m)\exp(s_i-m)$: the common factor can be taken outside the sum. For our scores, $m=1002$, the shifted arguments are `[-2, -1, 0]`, and the loop accumulates:

| Index | Shifted argument | Exponential, approximately | Running $\ell$, approximately |
|---:|---:|---:|---:|
| 0 | $-2$ | 0.135335283236613 | 0.135335283236613 |
| 1 | $-1$ | 0.367879441171442 | 0.503214724408055 |
| 2 | $0$ | 1 | 1.503214724408055 |

Every exponent argument is nonpositive, so an individual shifted exponential cannot overflow positively. At least one maximum contributes exactly one. For a nonempty finite row of length $n$, the mathematical denominator therefore satisfies $1\leq\ell\leq n$.

We have **not** made the raw mathematical $Z$ small enough to store. Multiplying by `exp(m)` would recreate the problem. Instead, we can retain the pair $(m,\ell)$, compute normalized weights as $\exp(s_i-m)/\ell$, or calculate

$$
\log Z=m+\log\ell.
$$

For this example the last quantity is approximately `1002.4076059644444`, which is easily representable as a finite double. The full normalization, loss, and their backward passes belong to later chapters; here we implement the scale-preserving sum itself.

### Validate first, then commit one small result

Append the result type and the complete routine:

```c title="examples/03-numerics/numerics.c — shifted exponential sum"
typedef struct {
    double maximum;
    double sum;
} ShiftedSum;

/* Nonempty finite float input, at most 4096 elements. Output unchanged on 0. */
static int shifted_exp_sum(const float *scores, size_t n, ShiftedSum *out) {
    if (scores == NULL || out == NULL || n == 0 || n > 4096) {
        return 0;
    }
    if (!isfinite(scores[0])) {
        return 0;
    }
    double maximum = (double)scores[0];
    for (size_t i = 1; i < n; ++i) {
        if (!isfinite(scores[i])) {
            return 0;
        }
        maximum = fmax(maximum, (double)scores[i]);
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double shifted = (double)scores[i] - maximum;
        if (!isfinite(shifted)) {
            return 0;
        }
        sum += exp(shifted);
    }
    if (!isfinite(sum) || sum <= 0.0) {
        return 0;
    }
    const ShiftedSum result = {.maximum = maximum, .sum = sum};
    *out = result;
    return 1;
}
```

The input is a borrowed contiguous array of `n` initialized floats. `out` must point to a live writable `ShiftedSum` object, separate from the input storage. `n` must be between one and 4096. Null pointers and invalid metadata are reportable argument failures; a non-null invalid pointer or a false extent remains a caller error, as in Chapter 2.

The limit 4096 keeps this foundation interface deliberately bounded. It is not a model context-length decision or a numerical requirement for all softmax implementations. In particular, `n > 4096` is rejected before any input access, so the negative test can safely supply that metadata with a smaller valid array.

The first scan checks all scores and finds the maximum. The second scan calculates exponentials in the chosen scale and accumulates them in double. There is no array of probabilities and no heap allocation, but there are two passes over an already-stored input. This is not yet streaming or tiled attention.

Look closely at `const double shifted = (double)scores[i] - maximum;`. Conversion happens **before** subtraction. Two finite float extremes differ by roughly twice `FLT_MAX`, outside float range but comfortably within double range. Writing `(double)(scores[i] - a_float_maximum)` would perform the subtraction in the wrong precision. The explicit `isfinite(shifted)` guard turns such an unexpected intermediate overflow into a failure; the extreme-input test below then catches the precision mistake.

The local `result` is assigned to `*out` only after validation and calculation succeed. All earlier returns leave both output fields unchanged. A zero return is an error status; it is not an exponential sum of zero. Reusing Chapter 2's transactional-output convention keeps errors from masquerading as plausible numerical data.

Very negative shifted arguments can underflow to zero. For `[0, -1000]`, the small contribution is negligible at the result's precision while the maximum still contributes one. That is different from an invalid row. This function rejects every non-finite score, including negative infinity; it does not treat a special value as a hidden mask. Masking will later be an explicit algorithmic decision.

### Check both the scale and the failure path

Append the driver:

```c title="examples/03-numerics/numerics.c — exponential checks"
static int check_exponentials(void) {
    const float scores[3] = {1000.0f, 1001.0f, 1002.0f};
    const float translated[3] = {0.0f, 1.0f, 2.0f};
    float naive = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        naive += expf(scores[i]);
    }
    ShiftedSum high = {.maximum = -7.0, .sum = -9.0};
    ShiftedSum low = {.maximum = -7.0, .sum = -9.0};
    if (!expect(shifted_exp_sum(scores, 3, &high) &&
                shifted_exp_sum(translated, 3, &low), "valid shifted sums")) {
        return 0;
    }
    const double log_total = high.maximum + log(high.sum);
    printf("raw expf sum finite=%d; shifted max=%.0f sum=%.15g\n",
           !!isfinite(naive), high.maximum, high.sum);
    printf("log-sum-exp=%.15g; translated sum=%.15g\n", log_total, low.sum);
    if (!expect(isinf(naive) && naive > 0.0f, "raw exponential overflow") ||
        !expect(high.maximum == 1002.0 && low.maximum == 2.0 &&
                close_finite(high.sum, 1.503214724408055, 1e-14, 1e-13) &&
                close_finite(high.sum, low.sum, 1e-14, 1e-13) &&
                close_finite(log_total, 1002.4076059644444, 2e-12, 0.0),
                "shifted values and log-sum-exp")) {
        return 0;
    }
    const float bad[3][2] = {{0.0f, NAN}, {0.0f, INFINITY}, {0.0f, -INFINITY}};
    for (size_t i = 0; i < 3; ++i) {
        ShiftedSum unchanged = {.maximum = -7.0, .sum = -9.0};
        if (!expect(!shifted_exp_sum(bad[i], 2, &unchanged) &&
                    unchanged.maximum == -7.0 && unchanged.sum == -9.0,
                    "reject non-finite input without changing output")) {
            return 0;
        }
    }
    ShiftedSum unchanged = {.maximum = -7.0, .sum = -9.0};
    if (!expect(!shifted_exp_sum(NULL, 3, &unchanged) &&
                !shifted_exp_sum(scores, 0, &unchanged) &&
                !shifted_exp_sum(scores, 4097, &unchanged) &&
                !shifted_exp_sum(scores, 3, NULL) &&
                unchanged.maximum == -7.0 && unchanged.sum == -9.0,
                "invalid exponential arguments")) {
        return 0;
    }
    const float tail[2] = {0.0f, -1000.0f};
    const float extremes[2] = {FLT_MAX, -FLT_MAX};
    ShiftedSum edge = {.maximum = 0.0, .sum = 0.0};
    return expect(shifted_exp_sum(tail, 2, &edge) && edge.sum == 1.0,
                  "negligible tail does not remove the maximum term") &&
           expect(shifted_exp_sum(extremes, 2, &edge) && edge.sum == 1.0 &&
                  edge.maximum == (double)FLT_MAX, "widen before subtracting");
}
```

The high scores and the translated scores `[0,1,2]` have the same shifted arguments, so their sums agree. Their maxima differ by 1000; consequently their log-sum-exp values differ by 1000, not zero. Translating scores preserves normalized ratios, not the unnormalized sum itself.

The expected denominator and log-sum-exp are decimal fixtures calculated independently from the three-term expression. The denominator uses a small combined tolerance, and the log-sum-exp uses an absolute tolerance of `2e-12`. At a scale near 1002, a double's spacing is about $2^{-43}$, approximately $1.14\times10^{-13}$; that absolute tolerance allows a small amount of library and evaluation rounding without accepting a unit-scale error. These are checks for this bounded example, not accuracy specifications for all math libraries or future kernels.

The three rows in `bad` are separate two-float arrays containing NaN, positive infinity, and negative infinity. Each attempted call must preserve `{-7,-9}`. Negative sentinels are useful here because they cannot be a valid positive shifted sum. The other negative tests exercise null pointers, zero length, and an over-limit length. The normal target contains no invalid memory access.

`!!isfinite(naive)` converts the macro's arbitrary nonzero “true” result to exactly one for printing: `!` negates truth, and a second `!` negates it again. Do not assume every C predicate already returns the integer one.

## Define a derivative through an experiment with a small change

A function maps an input to an output. A **derivative** asks how quickly that output changes locally when the input changes. For a scalar function $f$, input $x$, and nonzero small change $h$, the forward difference is

$$
\frac{f(x+h)-f(x)}{h}.
$$

If this ratio approaches a value as $h$ approaches zero from either side, that value is the derivative $f'(x)$, also written $df/dx$. The prime is notation for a derivative, not multiplication. The derivative's units are output units per input unit. The [finite-difference reference][finite-difference] relates nearby function samples to this local slope.

For $f(x)=x^2$, expand the numerator:

$$
(x+h)^2-x^2=2xh+h^2.
$$

Divide by $h$ to obtain $2x+h$. As $h$ becomes arbitrarily small in real arithmetic, the extra $h$ vanishes, giving $f'(x)=2x$. This is a derivation, not a request to remember a rule for powers before understanding it.

A real-arithmetic limit does not mean that a C program should repeatedly reduce `h` until it becomes zero. The computer stores only finitely many inputs near `x`. We will deliberately see where that mathematical limit stops being a usable numerical experiment.

### Compose two local rates

Consider the two-step calculation

$$
u=3x+1,\qquad y=u^2.
$$

At $x=2$, the intermediate $u$ is seven and the output $y$ is 49. Changing $x$ by a small $h$ changes $u$ by exactly $3h$. Expanding the new output gives

$$
(u+3h)^2-u^2=6uh+9h^2.
$$

Dividing by $h$ and taking the small-change limit yields $6u=42$. The same answer comes from multiplying the two local rates:

$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
=(2u)(3)=42.
$$

This is the **chain rule**. The first factor tells how a change in `u` affects `y`; the second tells how a change in `x` affects `u`. Multiplication converts an input change through the intermediate variable. At this point all shapes are scalar: `x`, `u`, `y`, and each derivative are single numbers.

Append the runnable chain-rule check and the polynomial routines that the difference experiment will use:

```c title="examples/03-numerics/numerics.c — chain rule and polynomial"
static int check_chain_rule(void) {
    const double x = 2.0;
    const double u = 3.0 * x + 1.0;
    const double y = u * u;
    const double dy_du = 2.0 * u;
    const double du_dx = 3.0;
    const double dy_dx = dy_du * du_dx;
    printf("chain rule: x=%.0f u=%.0f y=%.0f dy/dx=%.0f\n", x, u, y, dy_dx);
    return expect(y == 49.0 && dy_dx == 42.0, "two-step chain rule");
}

static double cubic_f64(double x) {
    return (x * x) * x - 2.0 * x;
}

static float cubic_f32(float x) {
    return (x * x) * x - 2.0f * x;
}

static double cubic_derivative(double x) {
    return 3.0 * x * x - 2.0;
}
```

The code keeps the intermediate `u` and names both local rates. There is no symbolic differentiator or hidden automatic differentiation. The exact comparisons for `49` and `42` are appropriate because these particular small integers and operations are exactly representable in the required format; they are not a general policy for comparing computed model outputs.

### A variable used along several paths receives several contributions

Now take

$$
f(x)=x^3-2x.
$$

Expanding $(x+h)^3-2(x+h)$ and subtracting $x^3-2x$ gives

$$
f(x+h)-f(x)=(3x^2-2)h+3xh^2+h^3.
$$

After division by $h$, the terms containing $h$ vanish in the limit, so

$$
f'(x)=3x^2-2.
$$

At $x=1.5$, $x^2=2.25$, hence $f'(1.5)=6.75-2=4.75$. The function value is different: $f(1.5)=3.375-3=0.375$. A derivative is not the output of the forward calculation.

We can also expose the paths in the expression. Let $u=x\cdot x$, $v=u\cdot x$, and $w=2x$, then $f=v-w$. Differentiating $u$ gives $du/dx=2x$. A small change in $v$ has a contribution from changing `u` and another from its separate `x` input:

$$
\frac{dv}{dx}=x\frac{du}{dx}+u=2x^2+x^2=3x^2.
$$

The two terms follow by expanding $(u+\Delta u)(x+h)-ux= x\Delta u+uh+\Delta u\,h$ and dividing by $h$; the final product of small changes vanishes in the limit. Subtract the derivative of `w`, which is two, to recover the same formula. Later, this “one contribution from each path” will become actual additions into gradient buffers. Overwriting one contribution with another would differentiate a different graph.

## A centered difference has a calculable error

Use values on both sides of `x`:

$$
D_h(x)=\frac{f(x+h)-f(x-h)}{2h},\qquad h>0.
$$

$D_h(x)$ is an estimate of $f'(x)$; its denominator is the distance between the requested sample points in exact arithmetic. For our cubic,

$$
\begin{aligned}
f(x+h)&=f(x)+(3x^2-2)h+3xh^2+h^3,\\
f(x-h)&=f(x)-(3x^2-2)h+3xh^2-h^3.
\end{aligned}
$$

Subtracting cancels both $f(x)$ and the $3xh^2$ terms. Therefore

$$
D_h(x)=3x^2-2+h^2=f'(x)+h^2.
$$

For this polynomial, the expression is exact in real arithmetic: there are no omitted higher powers. The **truncation error**, meaning the difference between this finite-step approximation and the derivative, is $h^2$. For a general sufficiently smooth function, centered differences similarly have a leading error proportional to $h^2$; the [Taylor-expansion treatment][finite-difference] gives the general argument.

Work one complete sample at $x=1.5$ and $h=0.1$:

$$
f(1.6)=4.096-3.2=0.896,\qquad
f(1.4)=2.744-2.8=-0.056.
$$

Their difference is `0.952`; division by `0.2` gives `4.76`. The derivative is `4.75`, so the error is `0.01`, exactly $h^2$. At $h=0.01$, the real-arithmetic result is `4.7501`, with error `0.0001`. Up to this point, making the step ten times smaller reduces this error by a factor of 100.

### Rounding is divided by the step too

Suppose the two evaluated function values contain errors $\delta_+$ and $\delta_-$. For the moment, suppose the sample points themselves were exact. The computed quotient differs from the ideal one by

$$
\frac{\delta_+-\delta_-}{2h}.
$$

A conservative magnitude bound is $(|\delta_+|+|\delta_-|)/(2h)$. Decreasing $h$ shrinks the polynomial's $h^2$ error but amplifies the evaluation errors. In a rough model with fixed positive constants $A$ and $B$,

$$
E(h)\approx Ah^2+\frac{B\varepsilon}{h}.
$$

Here $E$ models absolute derivative error and $\varepsilon$ represents the arithmetic's rounding scale. $A$ depends on the function's curvature variation; $B$ also depends on value scale and the evaluation algorithm. This is a useful explanation of competing trends, not an exact law or a promised smooth curve. In the actual program, rounding `x+h` and `x-h` introduces an additional source of error.

Balancing the two modeled terms suggests a step scale proportional to the cube root of $\varepsilon$, with problem-dependent factors. It does **not** imply that `FLT_EPSILON` or `DBL_EPSILON` is the right perturbation. Nor does it produce one step suitable for every parameter, operator, and loss. A small sweep is more informative than hiding a single arbitrary step in a checker.

The derivative we want is the derivative of the mathematical operation. The rounded C function is a discrete approximation to it. At sufficiently fine scales it has flat regions and jumps, so taking ever smaller samples of that rounded function does not recover the desired real-arithmetic derivative.

## Implement the perturbation in the type being tested

We will keep the actual sample coordinates as well as the derivative estimate:

```c title="examples/03-numerics/numerics.c — double centered difference"
typedef struct {
    double plus;
    double minus;
    double derivative;
} Difference;

/* No pointer retained or allocation performed. Output unchanged on failure. */
static int centered_f64(double x, double h, Difference *out) {
    if (out == NULL || !isfinite(x) || !isfinite(h) || h <= 0.0) {
        return 0;
    }
    const double plus = x + h;
    const double minus = x - h;
    if (!isfinite(plus) || !isfinite(minus) || plus == x || minus == x) {
        return 0;
    }
    const double upper = cubic_f64(plus);
    const double lower = cubic_f64(minus);
    const double numerator = upper - lower;
    const double denominator = 2.0 * h;
    if (!isfinite(upper) || !isfinite(lower) || !isfinite(numerator) ||
        !isfinite(denominator) || denominator <= 0.0) {
        return 0;
    }
    const double derivative = numerator / denominator;
    if (!isfinite(derivative)) {
        return 0;
    }
    const Difference result = {.plus = plus, .minus = minus,
                               .derivative = derivative};
    *out = result;
    return 1;
}
```

`Difference.plus` and `.minus` are the representable values actually used, not merely descriptions of an intended perturbation. The helper receives `x` and `h` by value, so it cannot modify the caller's original input. It fills a caller-owned output only on success. No saved input needs restoration in this scalar example.

A nonpositive or non-finite step is invalid. If `x+h` or `x-h` rounds back to `x`, one side did not move, so the routine refuses to label the resulting quotient a usable centered check. It also rejects overflow in the polynomial evaluations, numerator, denominator, or final quotient. It has no special recovery for enabled floating-point traps; the program uses the normal masked-exception environment described earlier.

The guard checks values before division. An invalid step cannot become a silently accepted `0/0`. The `Difference result` initialization and final assignment implement the same unchanged-output rule as the exponential helper. The caller must not read a failed output as though it were a newly computed answer.

For the float experiment, changing only the destination field to `float` would be inadequate. The perturbed coordinates, forward evaluation, subtraction, and division must actually happen in float. Append the separate routine:

```c title="examples/03-numerics/numerics.c — float centered difference"
static int centered_f32(float x, float h, Difference *out) {
    if (out == NULL || !isfinite(x) || !isfinite(h) || h <= 0.0f) {
        return 0;
    }
    const float plus = x + h;
    const float minus = x - h;
    if (!isfinite(plus) || !isfinite(minus) || plus == x || minus == x) {
        return 0;
    }
    const float upper = cubic_f32(plus);
    const float lower = cubic_f32(minus);
    const float numerator = upper - lower;
    const float denominator = 2.0f * h;
    if (!isfinite(upper) || !isfinite(lower) || !isfinite(numerator) ||
        !isfinite(denominator) || denominator <= 0.0f) {
        return 0;
    }
    const float derivative = numerator / denominator;
    if (!isfinite(derivative)) {
        return 0;
    }
    const Difference result = {.plus = (double)plus, .minus = (double)minus,
                               .derivative = (double)derivative};
    *out = result;
    return 1;
}
```

Here every intermediate from `plus` through `derivative` is float, and the literals carry the `f` suffix where they participate in float arithmetic. Only the finished values are widened into the reporting struct. The function `cubic_f32` also uses float intermediates; calling `cubic_f64` and narrowing its result would test a different computation.

The caller will convert the table's requested double step to float once when calling this helper. We report the requested decimal step for readability and retain the actual sampled coordinates for diagnosis. Around `1.5f`, adjacent floats are separated by about `1.1920929e-7`. A requested `1e-7f` therefore moves by one such spacing, not exactly the requested decimal amount. A requested `1e-8f` moves neither way and is rejected.

These routines deliberately implement the displayed textbook quotient with denominator `2*h`. A later variation replaces it with the actual representable span `plus-minus` to separate sample-displacement error from function-evaluation error. That replacement cannot recreate information already lost while evaluating the polynomial, and unequal left/right displacements need not define a perfectly centered sample about the original `x`.

## Sweep the step instead of hiding it

The next routine compares each estimate to the independently derived `4.75` and prints its absolute error. The float and double columns evaluate the same mathematical function in different precisions. An unusable step is printed as text, not as a fake zero derivative or a result that enters a minimum-error calculation.

Append the sweep and its boundary checks:

```c title="examples/03-numerics/numerics.c — step sweep and checks"
static int check_differences(void) {
    const double x = 1.5;
    const double exact = cubic_derivative(x);
    const double steps[] = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
                            1e-7, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18};
    printf("cubic: f(1.5)=%.3f exact derivative=%.2f\n", cubic_f64(x), exact);
    puts("         h     float estimate    abs error    double estimate    abs error");
    for (size_t i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i) {
        Difference narrow = {.plus = 0.0, .minus = 0.0, .derivative = 0.0};
        Difference wide = {.plus = 0.0, .minus = 0.0, .derivative = 0.0};
        const int ok32 = centered_f32((float)x, (float)steps[i], &narrow);
        const int ok64 = centered_f64(x, steps[i], &wide);
        printf("%10.1e", steps[i]);
        if (ok32) {
            printf(" %18.10g %12.3e", narrow.derivative,
                   fabs(narrow.derivative - exact));
        } else {
            printf(" %18s %12s", "unusable", "-");
        }
        if (ok64) {
            printf(" %18.10g %12.3e\n", wide.derivative,
                   fabs(wide.derivative - exact));
        } else {
            printf(" %18s %12s\n", "unusable", "-");
        }
    }
    Difference probe = {.plus = -7.0, .minus = -9.0, .derivative = -11.0};
    if (!expect(centered_f32(1.5f, 1e-7f, &probe), "visible float step")) {
        return 0;
    }
    printf("float h=1e-7: actual steps +%.9g and -%.9g\n",
           probe.plus - x, x - probe.minus);
    if (!expect(exact == 4.75 && centered_f64(x, 1e-5, &probe) &&
                close_finite(probe.derivative, exact, 1e-8, 1e-8) &&
                !close_finite(probe.derivative, 4.5, 1e-8, 1e-8),
                "useful double step accepts correct, rejects wrong derivative") ||
        !expect(centered_f32(1.5f, 1e-2f, &probe) &&
                close_finite(probe.derivative, exact, 3e-4, 1e-5),
                "useful float step")) {
        return 0;
    }
    probe = (Difference){.plus = -7.0, .minus = -9.0, .derivative = -11.0};
    return expect(!centered_f64(x, 0.0, &probe) &&
                  !centered_f64(x, -1e-5, &probe) &&
                  !centered_f64(x, NAN, &probe) &&
                  !centered_f64(INFINITY, 1e-5, &probe) &&
                  !centered_f64(x, 1e-18, &probe) &&
                  !centered_f32(1.5f, 1e-8f, &probe) &&
                  !centered_f64(x, 1e-5, NULL) &&
                  probe.plus == -7.0 && probe.minus == -9.0 &&
                  probe.derivative == -11.0,
                  "invalid or collapsed steps leave output unchanged");
}
```

The per-step structs are initialized so their state is always defined. The `ok32` and `ok64` statuses control whether their values can be used. `%18.10g` gives the estimate a field at least 18 characters wide and up to ten significant digits; `%12.3e` prints its error in scientific notation. The error column can reveal a discrepancy hidden by the rounded estimate column.

The two passing-step tests serve different purposes. In double, `h=1e-5` is small enough to reduce this cubic's truncation error while still leaving room above point collapse. The bound `1e-8 + 1e-8 * max(|estimate|, 4.75)` is about `5.75e-8`, comfortably above the observed rounding error for this small experiment and far below the error of the deliberately wrong derivative `4.5`. In float, `h=1e-2` deliberately tolerates an error on the scale of the known $h^2=10^{-4}$ term; its bound is about `3.475e-4`.

No assertion demands monotonic improvement as `h` shrinks, a particular best row, or matching last bits on every machine. The program does require at least these useful steps to distinguish the mathematical derivative from a substantial mistake. The exact `4.75` guard prevents a changed analytic formula from silently becoming the only source of truth.

The final block reinitializes `probe` with a **compound literal**, `(Difference){...}`, the struct-value syntax introduced in Chapter 2. Each invalid call returns without changing it. The short-circuit `&&` sequence stops at the first violated expectation, and the outer `expect` names the group. The null-output test is valid because the helper rejects it before dereferencing it.

Later, a parameter gradient checker will perturb actual model storage rather than a scalar passed by value. That checker must restore the original parameter after each trial, rerun the required forward calculation, and avoid stale saved state. This example establishes the numerical vocabulary without pretending that those future ownership and recomputation requirements already exist.

## Connect the executable and build all three configurations

Finish `numerics.c` with its entry point:

```c title="examples/03-numerics/numerics.c — entry point"
int main(void) {
    if (!check_environment() || !check_rounding() || !check_comparisons() ||
        !check_exponentials() || !check_chain_rule() || !check_differences()) {
        return EXIT_FAILURE;
    }
    puts("all numerical checks passed");
    return EXIT_SUCCESS;
}
```

`||` short-circuits: a failed check stops subsequent groups and produces a nonzero exit status. The final message is reached only when every group succeeds. This permits `make test` to distinguish a failed numerical expectation from an ordinary printed diagnostic about an intentionally bad method.

Create the complete Makefile. Recipe indentation must be a literal tab, as in Chapter 1:

```makefile title="examples/03-numerics/Makefile"
CC = xcrun clang
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2
LDLIBS = -lm

.PHONY: all debug sanitize release test clean

all: debug

debug: build/debug/numerics
sanitize: build/sanitize/numerics
release: build/release/numerics

build/debug/numerics: numerics.c Makefile
	mkdir -p build/debug
	$(CC) $(DEBUG_FLAGS) numerics.c -o $@ $(LDLIBS)

build/sanitize/numerics: numerics.c Makefile
	mkdir -p build/sanitize
	$(CC) $(SAN_FLAGS) numerics.c -o $@ $(LDLIBS)

build/release/numerics: numerics.c Makefile
	mkdir -p build/release
	$(CC) $(RELEASE_FLAGS) numerics.c -o $@ $(LDLIBS)

test: debug sanitize release
	./build/debug/numerics
	./build/sanitize/numerics
	./build/release/numerics

clean:
	rm -rf build
```

This small single-file program is compiled and linked in one command per configuration. Its Makefile is a prerequisite of each executable, so changing its flags triggers rebuilding. Outputs are separated under `build/debug/`, `build/sanitize/`, and `build/release/` rather than mixed in one directory.

The two numerical flags matter. `-fno-fast-math` preserves the ordinary assumptions required by our non-finite and evaluation-order checks. `-ffp-contract=off` then disables fusion of a multiply and add into a single rounded operation, keeping this foundation experiment's arithmetic steps easier to compare. The [Clang floating-point controls][clang-fp] document both behaviors. Fused multiply-add can be useful and more accurate; disabling it here is an experimental choice, not a prohibition for the later engine.

`-O2` enables ordinary optimization without changing that policy. Do not add `-Ofast`, `-ffast-math`, or finite-only assumptions to this executable and continue interpreting its NaN tests as valid. This is particularly important because the compiler can remove checks made irrelevant by assumptions you explicitly allowed.

`-lm` requests the math library at link time, after the source or object arguments. The normal native Apple toolchain accepts it; it also supplies the usual separate math-library linkage in the Linux checks. The program depends on standard C numerical interfaces, not glibc-specific allocation or processor-control extensions. Another compatible compiler is acceptable with equivalent floating-point controls; an unsupported flag or different evaluation format should be diagnosed, not hidden behind an exact version requirement.

Run from `examples/03-numerics/`:

```sh title="Terminal — native Mac build and numerical checks"
make clean
make test
```

To use a compatible directly installed Clang rather than `xcrun clang`:

```sh title="Terminal — explicit compiler override"
make clean
make CC=clang test
```

The checks do not require a particular macOS release, a GPU, or large memory. The largest baseline fixture is tiny. All normal calls use live local objects and valid element extents. The sanitizer build checks memory and selected undefined behavior; it does not establish the accuracy of a derivative or guarantee detection of NaNs. The numerical comparisons remain necessary.

## Read a complete run

The following output was observed with Clang 17.0.0 on x86_64 Linux using the displayed source and flags. Debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized builds all completed successfully. This is not a recorded M4/macOS execution; last digits and library formatting can differ on another toolchain.

```text title="Observed Linux output — baseline program"
sizes: float=4 double=8 bytes
radix=2 mantissa bits: float=24 double=53 eval_method=0
epsilon: float=1.1920929e-07 double=2.2204460492503131e-16
float range: true_min=1.40129846e-45 min_normal=1.17549435e-38 max=3.40282347e+38
gaps above 1 and 2^24: 1.1920929e-07 and 2
sum: float=0 reordered=1 double=1 late_cast=0
stored 16777217 as float, then widened: 16777216
0.1f widened=0.10000000149011612; 0.1 double=0.10000000000000001
NaN: rejection-only (error > tolerance)=0; finite check=0
raw expf sum finite=0; shifted max=1002 sum=1.50321472440806
log-sum-exp=1002.40760596444; translated sum=1.50321472440806
chain rule: x=2 u=7 y=49 dy/dx=42
cubic: f(1.5)=0.375 exact derivative=4.75
         h     float estimate    abs error    double estimate    abs error
   1.0e-01        4.760000706    1.000e-02               4.76    1.000e-02
   1.0e-02        4.750096798    9.680e-05             4.7501    1.000e-04
   1.0e-03         4.75025177    2.518e-04           4.750001    1.000e-06
   1.0e-04        4.751682281    1.682e-03         4.75000001    9.999e-09
   1.0e-05        4.756450653    6.451e-03               4.75    1.310e-10
   1.0e-06        4.529953003    2.200e-01               4.75    4.463e-10
   1.0e-07        5.960464478    1.210e+00        4.750000002    2.218e-09
   1.0e-08           unusable            -        4.749999971    2.887e-08
   1.0e-10           unusable            -        4.750000393    3.930e-07
   1.0e-12           unusable            -        4.750422278    4.223e-04
   1.0e-14           unusable            -        4.751754545    1.755e-03
   1.0e-16           unusable            -           unusable            -
   1.0e-18           unusable            -           unusable            -
float h=1e-7: actual steps +1.1920929e-07 and -1.1920929e-07
all numerical checks passed
```

Start with the sum line: float order gives zero, reordering gives one, and double accumulation gives one. `late_cast=0` proves that widening the final answer does not repair the lost intermediate. The decimal `0.1f` line makes stored approximation visible without implying that all decimal fractions are inaccurate; binary fractions such as `0.5` are exactly representable.

The two zeros on the NaN diagnostic line have different meanings. The rejection-only expression returns zero because it **did not reject** the bad error. The proper comparison returns zero because the values **did not pass**. A Boolean is useful only together with the question it answers.

The raw exponential sum is non-finite, while the shifted pair and log-sum-exp remain finite. That is the intended demonstration, not a failed program. The routine with the explicit success status never returns the raw overflowing sum as its answer.

In the derivative table, the large-step double errors initially track $h^2$. At `1e-5`, the estimate is already very close to `4.75`. Smaller steps then expose rounding, and `1e-16` is unusable at this `x`. The float column turns upward much earlier. Its `1e-7` row is about `5.96`, despite the correct analytic derivative being `4.75`; the printed actual displacement helps explain why this is not a good numerical test.

“Unusable” is neither success nor an analytic-gradient failure. It means that the stated numerical experiment could not be formed under the routine's contract. Likewise, a finite estimate is not automatically accurate. We need a suitable step region, independently understood expected values, and a meaningful tolerance together.

## Change one mechanism at a time

Keep these variations separate and restore the baseline after each. The unchanged companion source provides a reference for the functions modified below. The variations are part of the one-hour experiment allowance, not an additional project.

### Remove a contribution from the derivative

Temporarily replace only `cubic_derivative` with:

```c title="Variation — temporarily replace cubic_derivative"
static double cubic_derivative(double x) {
    return 3.0 * x * x;
}
```

The forward functions still compute $x^3-2x$, but the analytic formula forgot the second path's derivative. It now reports `6.75`. Across useful centered-difference steps, estimates remain near `4.75`, so the discrepancy approaches two rather than decreasing into a rounding-sized band. `make test` must fail the useful-double-step check, including its independent `4.75` anchor.

That pattern differs from the baseline's small-step deterioration: an omitted derivative term causes a substantial disagreement even in the good step region. Do not repair it by increasing the tolerance or shrinking `h` further. Restore `return 3.0 * x * x - 2.0;` and rerun the test.

### Let a NaN pass a rejection-only comparison

Temporarily replace only `close_finite` with this intentionally wrong implementation:

```c title="Variation — temporarily replace close_finite with a faulty checker"
static int close_finite(double actual, double expected,
                        double atol, double rtol) {
    const double error = fabs(actual - expected);
    const double bound = atol + rtol * fmax(fabs(actual), fabs(expected));
    return !(error > bound);
}
```

The first ordinary finite cases still behave plausibly. The `NaN actual` row then fails because the faulty helper returns one for a result that must not pass. The source accesses valid memory; an address sanitizer is not the mechanism that detects this mistake. Restore the complete finite guards and the final finite-aware acceptance condition before continuing.

### Divide by the represented span

In `centered_f32` only, temporarily replace the denominator declaration with:

```c title="Variation — one line inside centered_f32"
    const float denominator = plus - minus;
```

At the printed `h=1e-7`, both represented displacements are one float spacing, so this denominator is about `2.3841858e-7` instead of about `2.0e-7`. With the displayed non-fused Linux build, the estimate changes from about `5.960464478` to `5`, **not** to `4.75`. The forward evaluations and their subtraction are still rounded. A better displacement measurement alone cannot undo their error.

The useful float step at `1e-2` still lies within the chapter's tolerance in that observed build. More importantly, `1e-8` is still rejected: a different denominator does not make unchanged sample coordinates useful. Restore `const float denominator = 2.0f * h;` so the named baseline continues to implement the formula derived in the chapter.

After restoring all three variations:

```sh title="Terminal — leave the named baseline passing"
make clean
make test
```

## Diagnose the numerical layer that failed

| Symptom | First useful question or change |
|---|---|
| The format check fails before the arithmetic tests | Inspect the printed format, `FLT_EVAL_METHOD`, rounding mode, and compiler flags. Do not apply exact binary32 examples to a different evaluation model without adaptation. |
| `sum_f64` gives zero for the three-element fixture | Is the accumulator really double, and are you widening each element before addition rather than widening the final float result? |
| A non-finite result passes | Are you rejecting non-finite inputs and bounds before accepting a comparison? Did finite-only compiler assumptions remove the guards? |
| Every shifted exponential becomes zero on very negative scores | Was the maximum initialized to an actual element, not arbitrarily to zero? Does the maximum's shifted argument equal zero? |
| The extreme-score test fails | Did subtraction happen in float before conversion? Inspect expression types, not only the output field type. |
| Useful finite-difference steps disagree by a substantial nearly constant amount | Recheck the analytic expression and every computational path. An omitted contribution is not ordinary small-step roundoff. |
| Errors improve and then worsen as the step shrinks | Inspect function-value precision, subtraction cancellation, and actual represented perturbations before changing the derivative. |
| The reported estimate is zero at a tiny step | Check whether either sample rounded back to the original input. The baseline rejects that step instead of reporting a misleading estimate. |
| Debug and optimized results differ substantially | Verify that both use the stated non-fast-math and contraction flags, then check uninitialized data and other undefined behavior. Last-bit differences alone are not proof of a bug. |
| Sanitizers are quiet but a derivative is wrong | Memory validity and numerical correctness are different properties. Read the scalar trace and explicit expectations. |

The boundary between an invalid argument and a caller mistake remains explicit. Our checked helpers reject null pointers, invalid counts, non-finite values, or unusable steps. They cannot determine whether an arbitrary non-null pointer actually refers to enough live memory. Keep Chapter 2's lifetime and extent rules while adding these numerical checks; one layer does not replace another.

## Carry these observations into the tensor build

You now have three independently runnable foundation programs: a compiled and debuggable dot product, an ownership-and-view example, and a numerical diagnostic. This chapter adds a bounded shifted-exponential routine, conservative finite comparisons, a two-step manual chain rule, and float-versus-double finite-difference experiments. None relies on an unseen future framework.

The next chapter starts the actual decoder's tensor storage and matrix-product modules. The numerical choices will remain visible: element type, accumulator type, reduction order, non-finite policy, and tolerance. A matrix-product loop is built from reductions like the ones here; its test must know which values and shape it intends to compute before performance changes can be evaluated.

## References

**Representations and C interfaces:** GNU C Library Reference Manual, [floating-point representation concepts][fp-concepts], [floating-point parameters][fp-parameters], and [IEEE parameter examples][fp-ieee]. Consult the used constants and interfaces rather than adopting machine-specific implementation details. GNU C Language Manual, [rounding][fp-rounding]. GNU C Library Reference Manual, [number classification][fp-classes], [infinity and NaN behavior][fp-special], [neighboring representable values][fp-neighbor], and [exponential/logarithm functions][fp-exp]. These references describe interfaces available in the target toolchain; using them does not require installing glibc on macOS.

**Compiler semantics:** Clang Compiler User's Manual, [floating-point controls][clang-fp], specifically the descriptions of `-ffast-math`, `-fno-fast-math`, `-ffp-contract`, and expression evaluation. Their semantic effects matter more here than matching an exact compiler release.

**Numerical differences:** Qingkai Kong, Timmy Siauw, and Alexandre Bayen, *Python Programming and Numerical Methods*, [finite-difference approximations and Taylor expansions][finite-difference]. The reference supplies broader mathematical context; this chapter derives its own cubic example and executes all model-related numerical experiments in C.

[fp-concepts]: https://sourceware.org/glibc/manual/latest/html_node/Floating-Point-Concepts.html
[fp-parameters]: https://sourceware.org/glibc/manual/latest/html_node/Floating-Point-Parameters.html
[fp-ieee]: https://sourceware.org/glibc/manual/latest/html_node/IEEE-Floating-Point.html
[fp-rounding]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Rounding.html
[fp-classes]: https://sourceware.org/glibc/manual/latest/html_node/Floating-Point-Classes.html
[fp-special]: https://sourceware.org/glibc/manual/latest/html_node/Infinity-and-NaN.html
[fp-neighbor]: https://sourceware.org/glibc/manual/2.42/html_node/FP-Bit-Twiddling.html
[fp-exp]: https://sourceware.org/glibc/manual/latest/html_node/Exponents-and-Logarithms.html
[clang-fp]: https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior
[finite-difference]: https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter20.02-Finite-Difference-Approximating-Derivatives.html
[source-c]: https://25d-228.github.io/gpt-c/media/chapter-03/reference/numerics.c
[source-make]: https://25d-228.github.io/gpt-c/media/chapter-03/reference/Makefile
