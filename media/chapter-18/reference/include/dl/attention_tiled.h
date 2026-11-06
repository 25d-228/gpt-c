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
 * ONLY projections. Training keeps its materialized forward/backward pair in
 * this chapter. No residual, norm, dropout, bias, allocation or hidden counter. */
DlStatus dl_attention_tiled_forward(DlBackend backend, DlAttentionConfig config,
                                    DlConstMatrix x, DlAttentionWeights weights,
                                    const size_t *positions, size_t position_count,
                                    const double *inv_freq, size_t frequency_count,
                                    DlTiledAttentionCache cache,
                                    DlTiledWorkspace *workspace, DlMatrix y);

#endif
