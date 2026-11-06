#ifndef DL_CACHE_H
#define DL_CACHE_H

#include "dl/inference.h"

typedef struct {
    const DlWeights *weights;       /* Immutable, stable-address borrow. */
    size_t capacity, length;        /* length counts COMMITTED input tokens. */
    DlTensor history;              /* [layers,2,capacity,Hkv*d]: rotated K, V. */
    DlTensor arena;                /* Reusable prompt/one-row scratch. */
    size_t *positions;             /* capacity absolute positions. */
    double *inv_freq;              /* head_dim/2 frequencies. */
} DlKvCache;

/* Empty {0} owner only. Budget counts BOTH float arenas, not metadata/stack.
 * Positive capacity, last position <= DL_ROPE_MAX_POSITION. A nonzero
 * weights.position_limit also bounds capacity. No allocation
 * during prefill/decode/reset. All borrowing caches/workspaces must die before
 * weights; never copy/move live owners, edit metadata/weights, or use concurrently.
 * Construction failure leaves out unchanged; destroy is repeatable. */
DlStatus dl_cache_create(DlKvCache *out, const DlWeights *weights,
                          size_t capacity, size_t max_float_bytes);
void dl_cache_destroy(DlKvCache *cache);
/* Forget the transcript by setting length=0, not by erasing bytes. Not a wipe. */
DlStatus dl_cache_reset(DlKvCache *cache);

/* B=1, dense causal attention, no eviction, padding mask, dropout or sampling.
 * Prefill requires EMPTY history and 1<=count<=capacity; it processes the full
 * prompt layer by layer. Decode requires NONEMPTY history and consumes exactly
 * one token at absolute position length. Each returns [1,V] logits predicting
 * the token AFTER its last input, not the input itself. Logits are caller-owned
 * with canonical base/count; their ENTIRE allocation must differ from weights,
 * history and scratch. IDs/metadata and all cross-type storage are disjoint.
 * Invalid IDs/backend/shape/alias/capacity/state are rejected before mutation.
 * Success commits length only after every block AND final projection succeed.
 * On later numeric failure: old length and ALL committed K/V entries survive;
 * tentative suffix/scratch and logits are invalid. Retrying valid inputs is safe
 * because every reachable new entry is overwritten before it can be read.
 * Prefill never appends to an existing transcript: call reset first. Reset does
 * not change another cache, caller logits, weights, or sampling RNG state. */
DlStatus dl_cache_prefill(DlKvCache *cache, DlBackend backend,
                           const size_t *ids, size_t count, DlMatrix logits);
DlStatus dl_cache_decode(DlKvCache *cache, DlBackend backend,
                          size_t id, DlMatrix logits);

#endif
