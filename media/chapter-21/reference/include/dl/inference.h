#ifndef DL_INFERENCE_H
#define DL_INFERENCE_H

#include "dl/model.h"
#include "dl/attention_tiled.h"

typedef struct {
    char name[64];
    DlMatrix value;                  /* Borrow into the single weight arena. */
    int decay;                      /* Format metadata, not an optimizer. */
} DlWeight;

typedef struct {
    DlModelConfig config;
    DlWeight *params;               /* Owns param_count view records. */
    DlTensor arena;                 /* Owns ONLY P weight floats. */
    size_t param_count;
    size_t position_limit;          /* 0: training origin; imported: max prefix. */
    int ready;                     /* Set only by a successful copy/load. */
} DlWeights;

typedef struct {
    const DlWeights *weights;      /* Stable-address immutable borrow. */
    size_t capacity;                /* Maximum ONE-sequence prefix length. */
    DlTiledWorkspace tiled;         /* data==NULL: materialized; otherwise tiled. */
    DlTensor arena;                 /* One reusable layer, no backward data. */
    size_t *positions;              /* Owns capacity absolute positions. */
    double *inv_freq;               /* Owns head_dim/2 frequencies. */
} DlInference;

/* Empty {0} owners only. No live owner copies, metadata edits or concurrency.
 * Budgets bound float payloads, not metadata/allocator/stack/process memory.
 * Constructors leave out unchanged on failure. Destroy is repeatable.
 * create allocates an UNREADY directory for the checkpoint loader; it zeros
 * the float payload but does not initialize a usable model. from_model copies
 * values only, never gradients, and commits ready=1 after checking finiteness.
 * A nonzero position_limit bounds the capacity of EVERY inference/cache owner.
 * After copy/load, weights are immutable. Destroy all borrowing workspaces
 * before their weights. Do not reload, move, or mutate a borrowed weight owner. */
DlStatus dl_weights_create(DlWeights *out, DlModelConfig config,
                           size_t max_float_bytes);
DlStatus dl_weights_from_model(DlWeights *out, const DlModel *model,
                               size_t max_float_bytes);
void dl_weights_destroy(DlWeights *weights);
DlStatus dl_inference_create(DlInference *out, const DlWeights *weights,
                             size_t capacity, size_t max_float_bytes);
/* Same immutable-weight borrow, but no square scores/probabilities. This budget
 * includes BOTH the float arena AND all double tile/statistics payload. The
 * positions/frequencies/metadata and called-kernel stack are counted separately.
 * query_tile=16, key_tile=32 are starting choices, not architectural constants.
 * Attention strategy is fixed at creation; backend still selects matmul only. */
DlStatus dl_inference_create_tiled(DlInference *out, const DlWeights *weights,
                                   size_t capacity, size_t query_tile,
                                   size_t key_tile, size_t max_payload_bytes);
void dl_inference_destroy(DlInference *workspace);

/* Full-prefix B=1 baseline, no persistent KV state and no allocations here.
 * ids[length] is a separate live array, 1<=length<=capacity, each ID<V.
 * Compute all hidden prefix rows, but final norm/projection only for rows
 * [first_row, first_row+logits.rows). logits.cols==V; nonempty valid view.
 * Caller owns logits, with canonical base/count metadata. Its ENTIRE owner
 * must differ from both arenas; cross-type metadata/IDs are also disjoint.
 * No targets, loss, gradients, moments, dropout, RNG, or hidden append counter.
 * Invalid arguments/IDs/geometry leave logits unchanged. Later numeric errors
 * can partially write logits; discard the WHOLE result on any failure.
 * Scratch is disposable on failure; a fresh full call safely overwrites it.
 * Logits remain caller-owned and survive the next workspace call. */
DlStatus dl_inference_forward(DlInference *workspace, DlBackend backend,
                              const size_t *ids, size_t length,
                              size_t first_row, DlMatrix logits);

/* Synchronous read-only observation. Each view is borrowed ONLY until return;
 * copy values to retain them. Callback must not mutate/reenter any live owner,
 * weights, input IDs, or output. A callback failure stops the forward call;
 * discard ALL logits and trace records from that failed attempt. NULL disables
 * observation. No trace allocations; the ordinary forward wrapper is unchanged.
 * Block index is zero-based; final_norm/logits use config.layers. */
typedef DlStatus (*DlTraceFn)(void *context, const char *name, size_t index,
                               DlConstMatrix value);
DlStatus dl_inference_forward_trace(DlInference *workspace, DlBackend backend,
                                    const size_t *ids, size_t length,
                                    size_t first_row, DlMatrix logits,
                                    DlTraceFn trace, void *context);

#endif
