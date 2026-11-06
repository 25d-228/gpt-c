#ifndef DL_NORM_H
#define DL_NORM_H

#include "dl/kernels.h"

/* All views use canonical bases/counts and truthful live storage.
 * Logical cells must be finite; padding is neither read nor written.
 * Every output span must be disjoint from all input spans and other outputs.
 * Inputs may share storage. No allocation or ownership transfer occurs.
 * Structural and preflight non-finite errors leave all outputs unchanged.
 * Later DL_NUMERIC may leave partial writes: discard ALL outputs of the call.
 * SET never reads old output. ADD requires initialized finite logical cells. */

/* x [M,D], scale [1,D], y [M,D]; finite epsilon > 0, inside sqrt.
 * y[i,j] = scale[j] * x[i,j] / sqrt(mean_j(x[i,j]^2) + epsilon).
 * No bias, mean subtraction, row mixing, or stored statistics; overwrite y. */
DlStatus dl_rmsnorm_forward(DlConstMatrix x, DlConstMatrix scale,
                            float epsilon, DlMatrix y);

/* x and scale are the UNCHANGED inputs of the matching successful forward;
 * epsilon must match it. dy [M,D] is the incoming gradient.
 * dx [M,D] and dscale [1,D] have independently selected SET/ADD modes.
 * Recompute row statistics from x; y is not needed. Keep x/scale live until
 * backward finishes. No averaging over M: dy already contains loss scaling.
 * Row reductions use double. dscale adds one row at a time into FP32 storage,
 * like embedding scatter-add; it rounds after each row, not once per call. */
DlStatus dl_rmsnorm_backward(DlConstMatrix x, DlConstMatrix scale,
                             DlConstMatrix dy, float epsilon,
                             DlMatrix dx, DlWriteMode dx_mode,
                             DlMatrix dscale, DlWriteMode scale_mode);

/* Same positive shape for both inputs and output; overwrite out = a + b.
 * No in-place output, even if one input is no longer needed by the caller. */
DlStatus dl_residual_add(DlConstMatrix a, DlConstMatrix b, DlMatrix out);

/* Reverse ONE edge of a residual addition: dst = dy or dst += dy.
 * Call once per operand edge. For y = x + x, call twice into the SAME dx:
 * first SET, then ADD (or ADD twice when dx already has contributions).
 * dy and dst cannot overlap, including an exactly identical view. */
DlStatus dl_residual_backward(DlConstMatrix dy, DlMatrix dst,
                              DlWriteMode mode);

#endif
