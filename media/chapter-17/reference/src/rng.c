#include "dl/rng.h"

/* SplitMix64 mixing constants and recurrence: Sebastiano Vigna (2015),
 * public domain. https://prng.di.unimi.it/splitmix64.c
 * State is explicit here rather than a file-scope variable. */
uint64_t dl_rng_next(DlRng *rng)
{
    rng->state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = rng->state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

double dl_rng_uniform(DlRng *rng)
{
    return (double)(dl_rng_next(rng) >> 11) * 0x1.0p-53;
}

DlStatus dl_rng_bounded(DlRng *rng, uint64_t bound, uint64_t *out)
{
    if (rng == NULL || out == NULL || bound == 0) return DL_ARGUMENT;
    /* Unsigned subtraction wraps modulo 2^64: threshold = 2^64 % bound. */
    uint64_t threshold = (UINT64_C(0) - bound) % bound;
    uint64_t word;
    do {
        word = dl_rng_next(rng);
    } while (word < threshold);
    *out = word % bound;
    return DL_OK;
}
