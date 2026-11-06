#ifndef DL_ATTENTION_TILED_H
#define DL_ATTENTION_TILED_H

#include "dl/attention.h"

typedef struct {
    double *data;                  /* Sole owned allocation; never copy a live owner. */
    size_t count, row_capacity, head_dim, query_tile, key_tile;
} DlTiledWorkspace;

typedef struct {
    DlMatrix q0, k0, v, q, k, context; /* Same linear shapes as DlAttentionCache. */
} DlTiledAttentionCache;

/* Empty {0} only; positive dimensions, even head_dim. Budget counts ALL double
 * payload: three row-statistic planes plus score tile, numerators and state.
 * No time-by-time allocation. Checked arithmetic; failure leaves out unchanged.
 * Successful create leaves numeric cells uninitialized. Destroy is repeatable.
 * No metadata edits, owner copies, concurrent calls, or cross-type aliasing.
 * After SUCCESS, for sr=(b*Hq+h)*T+t, data[sr] is maximum,
 * data[row_capacity+sr] is scaled sum, data[2*row_capacity+sr] is log-sum-exp.
 * Only the first B*Hq*T cells in each plane are meaningful. The next call
 * overwrites them. On ANY failed computation discard all statistics/output.
 * Keep maximum and sum: log-sum-exp alone loses a small log(sum) at huge scores. */
DlStatus dl_tiled_workspace_create(DlTiledWorkspace *out, size_t row_capacity,
                                   size_t head_dim, size_t query_tile,
                                   size_t key_tile, size_t max_bytes);
void dl_tiled_workspace_destroy(DlTiledWorkspace *workspace);

/* Already projected/rotated Q [B*T,Hq*d], K/V [B*T,Hkv*d]. Output [B*T,Hq*d].
 * Full-sequence causal attention, key s<=query t, within the same batch item.
 * All input logical cells must be finite (even a masked future row). Old output
 * and scratch cells are unread. Float bounding spans must not overlap writes;
 * read-only inputs may overlap. Metadata/double workspace are disjoint from ALL
 * float storage. Canonical allocation bases/counts and lifetimes are required.
 * Scores use double dot products, then one checked FLOAT rounding, as in ref.
 * Online maxima/sums/numerators and tile exponentials use double; probabilities
 * are NOT individually rounded to float. Only final context cells are stored.
 * No malloc in compute. Preflight rejects without writes. Later DL_NUMERIC may
 * partially write output and statistics; discard them and rerun a whole call.
 * This is not a persistent KV cache and is not a materialized backward tape. */
DlStatus dl_attention_tiled_core(DlAttentionConfig config,
                                 DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
                                 DlTiledWorkspace *workspace, DlMatrix context);

/* Complete attention sublayer: projection, RoPE, tiled core, output projection.
 * Same X/weight/position/frequency shapes and lifetime rules as attention.h.
 * Six linear cache views plus Y are disjoint writes. Matrix backend controls
 * ONLY projections. Preserve successful linear cache and statistics for backward.
 * No residual, norm, dropout, bias, allocation or hidden counter. */
DlStatus dl_attention_tiled_forward(DlBackend backend, DlAttentionConfig config,
                                    DlConstMatrix x, DlAttentionWeights weights,
                                    const size_t *positions, size_t position_count,
                                    const double *inv_freq, size_t frequency_count,
                                    DlTiledAttentionCache cache,
                                    DlTiledWorkspace *workspace, DlMatrix y);

/* This choice is independent of DlBackend (the matrix-product implementation). */
typedef enum {
    DL_ATTENTION_MATERIALIZED = 0,
    DL_ATTENTION_TILED = 1
} DlAttentionMode;

typedef struct {
    double *data;
    size_t count, head_dim, query_tile, key_tile;
} DlTiledBackwardWorkspace;

typedef struct {
    DlMatrix context, q, k, v, q0, k0;
} DlTiledAttentionScratch;

/* Empty owner only. Count = 2*R*S + R*d + 2*S*d + R doubles.
 * No sequence-length-dependent allocation; no allocation during backward.
 * Same lifetime, canonical storage and no-owner-copy rules as forward workspace.
 * Forward and backward owners must be different allocations. */
DlStatus dl_tiled_backward_workspace_create(DlTiledBackwardWorkspace *out,
    size_t head_dim, size_t query_tile, size_t key_tile, size_t max_bytes);
void dl_tiled_backward_workspace_destroy(DlTiledBackwardWorkspace *workspace);

/* Differentiate ONE successful, unchanged tiled forward. Q/K already rotated.
 * context and dcontext [B*T,Hq*d], dQ same; dK/dV [B*T,Hkv*d].
 * Preserve Q,K,V,context and forward row stats. No freshness proof is possible.
 * Backward tile sizes may differ from forward; saved geometry must match.
 * All three outputs use mode. SET ignores old cells; ADD requires finite cells.
 * Preflight validates metadata/views/aliases/finiteness/basic statistic ranges,
 * with no writes. Later numeric failure invalidates ALL outputs/scratch,
 * including earlier ADD writes; do not retry ADD without restoring the baseline.
 * Stats and all float inputs remain unchanged; padding is untouched.
 * Reconstruction uses exp(score-max)/sum, never exp(score-lse) alone.
 * P,dP,dS and gradient reductions remain double; float stores occur once per
 * gradient cell. Center=dot(float context,dcontext), so equality with a
 * materialized float-probability reference is numerical, not bitwise.
 * No averaging by batch, time, or number of sharing heads. */
DlStatus dl_attention_tiled_core_backward(DlAttentionConfig config,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dcontext,
    const DlTiledWorkspace *forward, DlTiledBackwardWorkspace *work,
    DlMatrix dq, DlMatrix dk, DlMatrix dv, DlWriteMode mode);

/* Complete sublayer backward; matches dl_attention_tiled_forward, not a
 * materialized tape. Linear scratch is overwritten. Independent dx/weight
 * SET/ADD modes have the same semantics as attention.h. All eleven write
 * spans are mutually disjoint and disjoint from twelve float read spans.
 * Includes preflight before any projection/gradient writes. On later failure
 * discard every destination, even if an earlier projection succeeded. */
DlStatus dl_attention_tiled_backward(DlBackend backend, DlAttentionConfig config,
    DlConstMatrix x, DlAttentionWeights weights,
    const size_t *positions, size_t position_count,
    const double *inv_freq, size_t frequency_count,
    DlTiledAttentionCache cache, const DlTiledWorkspace *forward,
    DlConstMatrix dy, DlTiledAttentionScratch scratch,
    DlTiledBackwardWorkspace *work, DlMatrix dx, DlWriteMode dx_mode,
    DlAttentionWeightGrads dw, DlWriteMode weight_mode);

#endif
