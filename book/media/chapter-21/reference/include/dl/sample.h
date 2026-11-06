#ifndef DL_SAMPLE_H
#define DL_SAMPLE_H

#include "dl/common.h"
#include "dl/rng.h"

/* Independent, live, disjoint arrays/outputs of stated lengths; count>0.
 * All logits finite (even entries that would lose). Ties choose lowest ID.
 * Output ID unchanged on failure; no allocation or RNG in these three calls. */
DlStatus dl_greedy(const float *logits, size_t count, size_t *token);
/* Finite temperature>0; zero is NOT a greedy mode. Double probabilities.
 * Preflight failure leaves probabilities unchanged. Extreme negative scaled
 * differences may become -infinity; their exp is zero, not an invalid input.
 * No top-k/p filtering. At least one maximal logit has unnormalized weight 1. */
DlStatus dl_sample_probabilities(const float *logits, size_t count,
                                 double temperature, double *probabilities);
/* Finite 0<=p<=1, |sum(p)-1|<=1e-10; finite 0<=u<1. Renormalize the accepted
 * rounding residue using target=u*sum(p). Strict '<' implements half-open
 * intervals. Skip zero mass. A rounded terminal endpoint uses the LAST
 * positive entry, never a trailing zero. Output unchanged on failure. */
DlStatus dl_categorical(const double *probabilities, size_t count,
                        double uniform, size_t *token);
/* scratch[count] doubles may be overwritten. RNG/token commit only on success;
 * exactly ONE uniform draw per successful sampled token, even if mass is
 * concentrated at one ID. Does not consume a training or model RNG. */
DlStatus dl_sample_next(const float *logits, size_t count, double temperature,
                        DlRng *rng, double *scratch, size_t *token);

#endif
