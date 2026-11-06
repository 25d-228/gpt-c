#ifndef DL_LOSS_H
#define DL_LOSS_H

#include "dl/kernels.h"

/* logits [M,V], M,V > 0; targets is a separate live size_t[M] array.
 * Mean over M rows, natural logarithm, unweighted, no masks/ignored targets.
 * Every target is in [0,V); every logical logit must be finite.
 * loss points to a separate writable double, not into any input/output.
 * dlogits == NULL evaluates loss only. Otherwise *dlogits has shape [M,V],
 * is disjoint from logits, and receives d(mean loss)/d(logits).
 * SET never reads old gradients; ADD requires initialized finite gradients.
 * mode must be valid even in value-only calls. No allocation or saved tape.
 * Structural, ID, and preflight non-finite errors change no output.
 * Later DL_NUMERIC can leave partial gradient writes; loss is committed only
 * on success. Always discard the entire result after any failed call.
 * Keep canonical allocation bases; caller guarantees truthful live storage. */
DlStatus dl_cross_entropy_mean(DlConstMatrix logits, const size_t *targets,
                               size_t target_count, double *loss,
                               DlMatrix *dlogits, DlWriteMode mode);

#endif
