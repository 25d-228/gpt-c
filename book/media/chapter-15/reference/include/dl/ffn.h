#ifndef DL_FFN_H
#define DL_FFN_H

#include "dl/kernels.h"

typedef struct {
    DlConstMatrix gate, value, down; /* [F,D], [F,D], [D,F] */
} DlFfnWeights;

typedef struct {
    DlMatrix gate, value, hidden;   /* A, B, H: each [M,F] */
} DlFfnCache;

typedef struct {
    DlMatrix hidden, gate, value;   /* dH, dA, dB: each [M,F] */
} DlFfnScratch;

typedef struct {
    DlMatrix gate, value, down;     /* dWg, dWv, Wo gradient */
} DlFfnGrads;

/* All structs BORROW views; no allocation, transfer, or implicit freeing.
 * Positive M,D,F; canonical live bases/counts; padding is not accessed.
 * Every written span must be disjoint from all inputs and other outputs.
 * Read-only inputs may share storage. No in-place activation/scratch reuse.
 * Invalid geometry/shapes/flags/aliases and preflight non-finites do not write.
 * Later DL_NUMERIC may partially write MULTIPLE outputs; discard all of them.
 * No fallback or ADD retry on damaged gradients. */

/* X [M,D]; overwrite cache and Y [M,D]:
 * A=X Wg^T, B=X Wv^T, H=SiLU(A)*B, Y=H Wo^T.
 * No bias, normalization, residual, dropout, or implicit row averaging. */
DlStatus dl_ffn_forward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                        DlFfnCache cache, DlMatrix y);

/* X, weights, and cache must match a successful, UNCHANGED forward.
 * dy [M,D] is the incoming gradient. Cache is read-only during this call
 * despite its writable view type (used by forward); no saved Y is needed.
 * Always overwrite scratch. dx_mode controls the FIRST dX branch only;
 * the second branch always ADDs. weight_mode applies to all three dW arrays.
 * SET never reads old output. ADD requires initialized finite logical cells.
 * Arrays use float; scalar arithmetic/reductions use double. Each matrix call
 * stores float results; the two dX contributions therefore round separately. */
DlStatus dl_ffn_backward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                         DlFfnCache cache, DlConstMatrix dy,
                         DlFfnScratch scratch, DlMatrix dx,
                         DlWriteMode dx_mode, DlFfnGrads dw,
                         DlWriteMode weight_mode);

#endif
