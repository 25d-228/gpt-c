#include "dl/sample.h"

#include <math.h>

DlStatus dl_greedy(const float *logits, size_t count, size_t *token)
{
    if (logits == NULL || token == NULL || count == 0) return DL_ARGUMENT;
    size_t best = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(logits[i])) return DL_NUMERIC;
        if (logits[i] > logits[best]) best = i;
    }
    *token = best;
    return DL_OK;
}

DlStatus dl_sample_probabilities(const float *logits, size_t count,
                                 double temperature, double *probabilities)
{
    if (logits == NULL || probabilities == NULL || count == 0 ||
        !isfinite(temperature) || temperature <= 0.0) return DL_ARGUMENT;
    double maximum = -(double)INFINITY;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(logits[i])) return DL_NUMERIC;
        if ((double)logits[i] > maximum) maximum = logits[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        /* Widen BEFORE subtraction; subtract BEFORE dividing by temperature. */
        probabilities[i] = exp(((double)logits[i] - maximum) / temperature);
        sum += probabilities[i];
    }
    for (size_t i = 0; i < count; ++i) probabilities[i] /= sum;
    return DL_OK;
}

DlStatus dl_categorical(const double *probabilities, size_t count,
                        double uniform, size_t *token)
{
    if (probabilities == NULL || token == NULL || count == 0 ||
        !isfinite(uniform) || uniform < 0.0 || uniform >= 1.0) return DL_ARGUMENT;
    double sum = 0.0;
    size_t last = count;
    for (size_t i = 0; i < count; ++i) {
        double p = probabilities[i];
        if (!isfinite(p) || p < 0.0 || p > 1.0) return DL_NUMERIC;
        sum += p;
        if (p > 0.0) last = i;
    }
    if (last == count || !isfinite(sum) || fabs(sum - 1.0) > 1e-10) return DL_NUMERIC;
    double target = uniform * sum, cumulative = 0.0;
    for (size_t i = 0; i < count; ++i) {
        cumulative += probabilities[i];
        if (probabilities[i] > 0.0 && target < cumulative) {
            *token = i;
            return DL_OK;
        }
    }
    *token = last; /* Only a rounded endpoint can reach this fallback. */
    return DL_OK;
}

DlStatus dl_sample_next(const float *logits, size_t count, double temperature,
                        DlRng *rng, double *scratch, size_t *token)
{
    if (rng == NULL || token == NULL) return DL_ARGUMENT;
    DlStatus status = dl_sample_probabilities(logits, count, temperature, scratch);
    if (status != DL_OK) return status;
    DlRng next = *rng;
    size_t selected = 0;
    status = dl_categorical(scratch, count, dl_rng_uniform(&next), &selected);
    if (status != DL_OK) return status;
    *rng = next;
    *token = selected;
    return DL_OK;
}
