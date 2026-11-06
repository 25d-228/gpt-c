#ifndef DL_ROPE_H
#define DL_ROPE_H

#include "dl/kernels.h"

#include <stdint.h>

/* Integer-position contract, not a model's trained context length. */
#define DL_ROPE_MAX_POSITION ((size_t)UINT32_MAX)

/* Write head_dim/2 doubles into caller-owned frequency storage.
 * head_dim is positive and even; count must equal head_dim/2.
 * theta is finite and > 1. Frequencies are theta^(-2*j/head_dim).
 * No allocation. On any reported failure, leave inv_freq unchanged.
 * The output must be live, correctly aligned, and large enough for count. */
DlStatus dl_rope_frequencies(size_t head_dim, double theta,
                             double *inv_freq, size_t count);

/* x/y are [M,H*head_dim]; H is inferred separately on each call.
 * Pair j with j+head_dim/2 WITHIN each head (split-half, not adjacent).
 * positions is a separate live size_t[M] array of absolute positions.
 * Each position is <= DL_ROPE_MAX_POSITION; repetitions/resets are allowed.
 * inv_freq is a separate live double[head_dim/2] array with values in (0,1].
 * Float, position, and frequency storage must not overlap each other.
 * Arrays retain canonical bases/counts and truthful live capacities.
 * x/y bounding spans must be disjoint, even for exact in-place requests.
 * No allocation, retained pointers, cached angles, or hidden position counter.
 * Input/old ADD logical cells must be finite. Padding is untouched.
 * Structural/preflight errors do not write. Later DL_NUMERIC can leave
 * earlier pairs written: discard the entire result and do not retry ADD.
 * Each pair's two results are range-checked before either is stored.
 * Frequencies, angles, and pair arithmetic use double; arrays store float. */
DlStatus dl_rope_forward(DlConstMatrix x, size_t head_dim,
                         const size_t *positions, size_t position_count,
                         const double *inv_freq, size_t frequency_count,
                         DlMatrix y);

/* Apply the rotation TRANSPOSE to dy, then SET or ADD into dx.
 * dy/dx have identical [M,H*head_dim] shapes. Positions/frequencies must
 * match the forward whose gradient is being propagated. Neither forward
 * x nor y is needed: the Jacobian depends only on those fixed settings.
 * No derivative of positions/frequencies and no extra loss averaging. */
DlStatus dl_rope_backward(DlConstMatrix dy, size_t head_dim,
                          const size_t *positions, size_t position_count,
                          const double *inv_freq, size_t frequency_count,
                          DlMatrix dx, DlWriteMode mode);

#endif
