#ifndef DL_OPTIM_H
#define DL_OPTIM_H

#include "dl/model.h"

/* Counts up to 2^53 are exactly representable as binary64 integers. */
#define DL_OPT_MAX_COUNT UINT64_C(9007199254740992)

typedef struct {
    double beta1, beta2, epsilon, weight_decay, clip_norm;
} DlAdamWConfig;

typedef struct {
    DlModel *model;                  /* Borrow: stable address, outlives us. */
    DlAdamWConfig config;
    float *moments;                  /* Owner: [first moment P][second P]. */
    double *sum;                    /* Owner: P token-weighted gradients. */
    size_t count;
    uint64_t updates, tokens, revision;
    int active;
} DlAdamW;

typedef struct {
    uint64_t update, tokens;
    double learning_rate, grad_norm, clip_scale;
} DlOptimStats;

/* Config: finite 0<=beta<1, epsilon>0, decay>=0, clip>=0. clip==0 disables
 * clipping. Empty {0} owner only. Budget covers 2*P floats + P doubles;
 * metadata/allocator overhead excluded. Failure leaves out unchanged.
 * Never copy a live owner, edit metadata/config, or move its borrowed model.
 * Constructed model metadata remains immutable. No concurrent access. */
DlStatus dl_adamw_create(DlAdamW *out, DlModel *model, DlAdamWConfig config,
                         size_t max_state_bytes);
void dl_adamw_destroy(DlAdamW *optim);

/* Start/restart one update group: zero sum and model gradients, tokens=0,
 * gradients_valid=0. Retain moments and completed update count. No weight
 * change. A previous partial group is deliberately discarded. An outside
 * weight edit invalidates this optimizer; destroy/recreate rather than
 * silently attaching old moments to externally changed parameters. */
DlStatus dl_adamw_begin(DlAdamW *optim);

/* Consume a FRESH single-microbatch mean from model.grad, normally produced
 * by backward(..., DL_SET); tokens is its actual number of scored targets.
 * Accumulate tokens*grad in double. Never feed a sum of batch means (ADD).
 * Success marks model gradients consumed (gradients_valid=0); bytes remain.
 * Failure changes neither sum/count nor model gradients/validity. Combined
 * tokens <= 2^53. A failed backward requires begin/recompute the whole group. */
DlStatus dl_adamw_add_mean(DlAdamW *optim, uint64_t tokens);

/* Needs an active nonempty group and no unconsumed model gradients.
 * finite lr>0; 0<=lr*decay<=1 (nonnegative shrink factor).
 * Form the token mean, clip its ONE global norm, then update unique params.
 * Moment storage is float; recurrences and candidate weights use double.
 * Every candidate is checked before ANY weight/moment/count/revision write.
 * Failure leaves the complete call state and stats unchanged. A successful
 * call increments updates/revision, consumes the group, invalidates old tapes
 * and model gradients; sum/grad bytes are cleared by the NEXT begin, not here.
 * stats is a separate caller-owned object, disjoint from all optimizer/model
 * storage. No use of model SGD or external weight edits while attached. */
DlStatus dl_adamw_step(DlAdamW *optim, double learning_rate, DlOptimStats *stats);

/* Discard a partial group (or clear stale boundary scratch), retaining weights,
 * moments, update count and revision. Does not rewind any external RNG. */
DlStatus dl_adamw_abort(DlAdamW *optim);
/* Coordinated restore into a STABLE model/optimizer pair at a boundary.
 * Three independent flat arrays, each count floats in parameter-directory
 * order. They must not overlap any live model/optimizer storage or each other.
 * Checks all values (second moments >=0) and revision capacity before commit.
 * On success advances the LIVE revision, installs weights/moments/update count,
 * clears gradient/sum scratch, and invalidates all old tapes by revision.
 * Does not change optimizer config; the checkpoint must match it beforehand.
 * On failure all live state unchanged. Not safe for concurrent readers. */
DlStatus dl_adamw_restore(DlAdamW *optim, const float *weights,
                          const float *first, const float *second,
                          size_t count, uint64_t updates);

#endif
