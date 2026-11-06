#ifndef DL_RNG_H
#define DL_RNG_H

#include <stdint.h>

typedef struct { uint64_t state; } DlRng;

/* rng is nonnull, initialized with ANY uint64_t seed, and exclusively owned.
 * Deterministic non-cryptographic stream. No allocation or global state. */
uint64_t dl_rng_next(DlRng *rng);
/* One draw; 53-bit grid in [0,1), represented exactly in binary64. */
double dl_rng_uniform(DlRng *rng);

#endif
