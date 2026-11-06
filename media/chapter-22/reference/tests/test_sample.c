#include "dl/sample.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(yes) do { if (!(yes)) { \
    fprintf(stderr, "FAILED: %s (line %d)\n", #yes, __LINE__); return 0; \
} } while (0)

static int boundaries(void)
{
    const double p[] = {0.0, 0.25, 0.5, 0.25, 0.0};
    const double draws[] = {0.0, nextafter(0.25, 0.0), 0.25,
                            nextafter(0.75, 0.0), 0.75, nextafter(1.0, 0.0)};
    const size_t expected[] = {1, 1, 2, 2, 3, 3};
    size_t token = 99;
    for (size_t i = 0; i < 6; ++i) {
        CHECK(dl_categorical(p, 5, draws[i], &token) == DL_OK);
        CHECK(token == expected[i]);
    }
    /* Deterministic midpoint coverage, not a statistical/flaky RNG assertion. */
    size_t counts[5] = {0};
    for (size_t i = 0; i < 1000; ++i) {
        CHECK(dl_categorical(p, 5, ((double)i + 0.5) / 1000.0, &token) == DL_OK);
        ++counts[token];
    }
    CHECK(counts[0] == 0 && counts[1] == 250 && counts[2] == 500 &&
          counts[3] == 250 && counts[4] == 0);
    const double rounded[] = {0.1, 0.2, 0.6999999999999998, 0.0};
    CHECK(dl_categorical(rounded, 4, nextafter(1.0, 0.0), &token) == DL_OK && token == 2);
    puts("sampling: six half-open boundaries; 1000 midpoint draws give 250/500/250");
    return 1;
}

static int temperatures(void)
{
    const float logits[] = {0.0f, 0.6931471805599453f, 1.3862943611198906f};
    const double temperatures[] = {1.0, 0.5, 2.0};
    const double sqrt2 = sqrt(2.0);
    const double expected[3][3] = {
        {1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0},
        {1.0 / 21.0, 4.0 / 21.0, 16.0 / 21.0},
        {1.0 / (3.0 + sqrt2), sqrt2 / (3.0 + sqrt2), 2.0 / (3.0 + sqrt2)}
    };
    double p[3] = {0}, shifted[3] = {0};
    for (size_t t = 0; t < 3; ++t) {
        CHECK(dl_sample_probabilities(logits, 3, temperatures[t], p) == DL_OK);
        for (size_t i = 0; i < 3; ++i) CHECK(fabs(p[i] - expected[t][i]) < 3e-8);
    }
    const float x[] = {1.0f, 2.0f, 3.0f}, y[] = {101.0f, 102.0f, 103.0f};
    CHECK(dl_sample_probabilities(x, 3, 0.7, p) == DL_OK);
    CHECK(dl_sample_probabilities(y, 3, 0.7, shifted) == DL_OK);
    for (size_t i = 0; i < 3; ++i) CHECK(p[i] == shifted[i]);
    const float extremes[] = {-FLT_MAX, FLT_MAX, FLT_MAX};
    CHECK(dl_sample_probabilities(extremes, 3, DBL_MIN, p) == DL_OK);
    CHECK(p[0] == 0.0 && p[1] == 0.5 && p[2] == 0.5);
    CHECK(dl_sample_probabilities(extremes, 3, DBL_MAX, p) == DL_OK);
    for (size_t i = 0; i < 3; ++i) CHECK(fabs(p[i] - 1.0 / 3.0) < 1e-15);
    size_t token = 99;
    CHECK(dl_greedy(extremes, 3, &token) == DL_OK && token == 1);
    CHECK(dl_sample_probabilities(extremes, 3, DBL_TRUE_MIN, p) == DL_OK);
    CHECK(dl_categorical(p, 3, 0.75, &token) == DL_OK && token == 2);
    const float one[] = {-123.0f};
    CHECK(dl_sample_probabilities(one, 1, 1.0, p) == DL_OK && p[0] == 1.0);
    CHECK(dl_greedy(one, 1, &token) == DL_OK && token == 0);
    puts("sampling: three hand distributions, shift invariance, extreme temperatures, greedy ties");
    return 1;
}

static int failures_and_rng(void)
{
    const float good[] = {0.0f, 1.0f, 2.0f};
    const double bad_temperatures[] = {0.0, -1.0, NAN, INFINITY};
    double scratch[3] = {7.0, 8.0, 9.0};
    size_t token = 99;
    DlRng rng = {42};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(dl_sample_next(good, 3, bad_temperatures[i], &rng, scratch, &token) == DL_ARGUMENT);
        CHECK(rng.state == 42 && token == 99 && scratch[0] == 7.0);
    }
    const float bad[] = {100.0f, NAN, -INFINITY};
    CHECK(dl_greedy(bad, 3, &token) == DL_NUMERIC && token == 99);
    CHECK(dl_sample_next(bad, 3, 1.0, &rng, scratch, &token) == DL_NUMERIC);
    CHECK(rng.state == 42 && token == 99 && scratch[0] == 7.0);
    const double invalid[5][3] = {{0, 0, 0}, {0.2, 0.2, 0.2},
        {-0.1, 0.5, 0.6}, {NAN, 0, 1}, {INFINITY, 0, 0}};
    for (size_t i = 0; i < 5; ++i)
        CHECK(dl_categorical(invalid[i], 3, 0.5, &token) == DL_NUMERIC && token == 99);
    const double p[] = {0.25, 0.5, 0.25};
    const double invalid_draws[] = {-0.1, 1.0, NAN, INFINITY};
    for (size_t i = 0; i < 4; ++i)
        CHECK(dl_categorical(p, 3, invalid_draws[i], &token) == DL_ARGUMENT && token == 99);
    CHECK(dl_greedy(NULL, 3, &token) == DL_ARGUMENT);
    CHECK(dl_greedy(good, 0, &token) == DL_ARGUMENT);
    CHECK(dl_sample_probabilities(good, 3, 1.0, NULL) == DL_ARGUMENT);
    CHECK(dl_categorical(p, 0, 0.0, &token) == DL_ARGUMENT);
    DlRng expected = rng, replay = rng;
    for (size_t i = 0; i < 64; ++i) {
        double draw = dl_rng_uniform(&expected);
        size_t want = 99, again = 99;
        CHECK(dl_sample_probabilities(good, 3, 0.7, scratch) == DL_OK);
        CHECK(dl_categorical(scratch, 3, draw, &want) == DL_OK);
        CHECK(dl_sample_next(good, 3, 0.7, &rng, scratch, &token) == DL_OK);
        CHECK(dl_sample_next(good, 3, 0.7, &replay, scratch, &again) == DL_OK);
        CHECK(token == want && again == token && rng.state == expected.state && replay.state == rng.state);
    }
    puts("sampling: invalid inputs preserve RNG/output; 64 sampled draws replay exactly");
    return 1;
}

int main(void)
{
    if (!boundaries() || !temperatures() || !failures_and_rng()) return EXIT_FAILURE;
    puts("all sampling checks passed");
    return EXIT_SUCCESS;
}

#undef CHECK
