#ifndef DL_ATTENTION_H
#define DL_ATTENTION_H

#include "dl/rope.h"

typedef struct {
    size_t batch, time, query_heads, kv_heads, head_dim;
} DlAttentionConfig;

typedef struct {
    DlConstMatrix query, key, value, output;
} DlAttentionWeights;

typedef struct {
    DlMatrix q0, k0, v;        /* Before RoPE: [M,Q], [M,K], [M,K]. */
    DlMatrix q, k;             /* Rotated: [M,Q], [M,K]. */
    DlMatrix scores, probs;    /* [B*Hq*T,T], row (b*Hq+h)*T+t. */
    DlMatrix context;          /* [M,Q], concatenated head results. */
} DlAttentionCache;

/* Full, fixed-length causal self-attention. M=B*T, Q=Hq*d, K=Hkv*d.
 * Positive B,T,Hq,Hkv; positive even d; Hq is divisible by Hkv.
 * X/Y [M,D]; Wq [Q,D], Wk/Wv [K,D], Wo [D,Q]. D need not equal Q.
 * Head h reads K/V head h/(Hq/Hkv); no physical K/V repetition.
 * Time s is allowed iff s<=t WITHIN the SAME batch item. No padding mask,
 * packed documents, cache append, bias, dropout, residual, or normalization.
 * positions[M] supplies absolute RoPE positions, NOT mask membership.
 * Frequencies/positions follow rope.h; all metadata storage is disjoint
 * from float storage and each other. Actual arrays must be live and sized.
 * All structs borrow. No allocation, ownership transfer, retained pointers,
 * or hidden state. Every written bounding span is disjoint from all inputs
 * and other outputs; read-only inputs may overlap. No in-place use.
 * All destinations are overwritten; their old logical cells are NOT read.
 * Padding is untouched. Masked scores are -INFINITY; masked probs are zero.
 * Other stored values must be finite floats. Dot products, max subtraction,
 * exp/normalization and value reductions use double. Scores and probabilities
 * are rounded to float at their explicit storage boundaries.
 * Invalid metadata/views/shapes/aliases and nonfinite inputs do not write.
 * A later DL_NUMERIC can partially write MULTIPLE buffers: discard ALL output
 * and cache contents. No retry that consumes partially written state.
 * Preserve X, weights, successful cache, positions and frequencies for the
 * later backward pass. This cache is a training tape, NOT a decode KV cache. */
DlStatus dl_attention_forward(DlBackend backend, DlAttentionConfig config,
                              DlConstMatrix x, DlAttentionWeights weights,
                              const size_t *positions, size_t position_count,
                              const double *inv_freq, size_t frequency_count,
                              DlAttentionCache cache, DlMatrix y);

#endif
