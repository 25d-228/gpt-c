#ifndef DL_RNG_H
#define DL_RNG_H

#include "dl/common.h"
#include <stdint.h>

typedef struct { uint64_t state; } DlRng;

/* rng is nonnull, initialized with ANY uint64_t seed, and exclusively owned.
 * Deterministic non-cryptographic stream. No allocation or global state. */
uint64_t dl_rng_next(DlRng *rng);
/* One draw; 53-bit grid in [0,1), represented exactly in binary64. */
double dl_rng_uniform(DlRng *rng);

/* Rejection sampling in [0,bound); no float conversion or modulo bias.
 * Nonzero bound, nonnull independent rng/out. Invalid arguments change neither.
 * A successful call can consume more than one generator word. */
DlStatus dl_rng_bounded(DlRng *rng, uint64_t bound, uint64_t *out);

#endif
