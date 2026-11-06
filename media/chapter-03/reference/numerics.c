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

int main(void) {
    if (!check_environment() || !check_rounding() || !check_comparisons() ||
        !check_exponentials() || !check_chain_rule() || !check_differences()) {
        return EXIT_FAILURE;
    }
    puts("all numerical checks passed");
    return EXIT_SUCCESS;
}
