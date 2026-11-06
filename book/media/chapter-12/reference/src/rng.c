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
