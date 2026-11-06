# Chapter 19 — Recomputed tiled backward without a quadratic tape

> **Learning objectives.** Reconstruct one probability tile from saved normalization, derive every query/key/value gradient without a square tape, and connect the matching forward/backward pair to the existing C trainer. Understand why head sharing requires addition, why tile ownership affects rounding, and why an attention execution policy belongs in a resumable run.

This is one **7.00-hour implementation chapter** presented in five connected parts. Allow **4.50 hours** for the backward operator, tape layout, and training integration, and **2.50 hours** for gradient, memory, and continuation checks. Reading and unattended computation are separate. The estimates assume Chapter 18's working code; most functions reproduced below are retained dependencies rather than new work.

**Prerequisites:** [Chapter 11's materialized backward](https://25d-228.github.io/gpt-c/chapter-11.html), [Chapter 12's model tape](https://25d-228.github.io/gpt-c/chapter-12-part-2.html), [Chapter 15's trainer/checkpoints](https://25d-228.github.io/gpt-c/chapter-15.html), and the complete [Chapter 18 source state](https://25d-228.github.io/gpt-c/chapter-18.html). All 71 prerequisite files are in that chapter's companion directory. Chapter 19 replaces eleven files and adds two tests; its [complete source snapshot](https://25d-228.github.io/gpt-c/media/chapter-19/reference/Makefile) contains 73 files. The final part gives exact build/run commands. Do not combine selected new headers with old implementations.

Part 1 develops the operator. [Part 2](https://25d-228.github.io/gpt-c/chapter-19-part-2.html) removes square allocations from the model tape. [Part 3](https://25d-228.github.io/gpt-c/chapter-19-part-3.html) integrates training and continuation. [Part 4](https://25d-228.github.io/gpt-c/chapter-19-part-4.html) checks the core and complete sublayer. [Part 5](https://25d-228.github.io/gpt-c/chapter-19-part-5.html) checks complete models, runs training, and accounts for memory.

## What backward actually needs

Chapter 18 already computed context without storing the full score or probability matrix. Its inference workspace was reused from layer to layer. That is not yet a training tape: backward for block zero cannot use normalization statistics left over from the last block.

For each block, preserve the normalized input, projection intermediates, rotated query/key arrays, unrotated values, attention context, and row normalization statistics until that block's backward finishes. Preserve weights, positions, and frequencies too. We will keep a separate forward workspace per block and reuse **one** backward workspace as the reverse traversal visits blocks sequentially.

The large arrays we remove are scores, probabilities, probability derivatives, and score derivatives. We do **not** remove all activations, change causal visibility, introduce sparse attention, or make pairwise arithmetic linear in sequence length.

For one batch item and one query head, write the query at time $t$ as $q_t\in\mathbb R^d$, the key/value at time $s$ as $k_s,v_s\in\mathbb R^d$, and the incoming context gradient as $g_t\in\mathbb R^d$. The causal condition is $0\leq s\leq t<T$. Here $d$ is head width, $T$ is sequence length, and $\tau=1/\sqrt d$ is the score scale. Define

$$
a_{ts}=\tau\sum_{j=0}^{d-1}q_{t,j}k_{s,j},\qquad
p_{ts}=\frac{\exp(a_{ts}-m_t)}{\ell_t},\qquad
C_{t,j}=\sum_{s=0}^{t}p_{ts}v_{s,j}.
$$

The row maximum is $m_t=\max_{s\leq t}a_{ts}$, and the scaled sum is $\ell_t=\sum_{s\leq t}\exp(a_{ts}-m_t)$. Chapter 18 also saves $L_t=m_t+\log\ell_t$. We retain its explicit float score-rounding boundary, so the implementation's $a_{ts}$ is the checked float-rounded scaled dot product widened back to double.

Recompute each score tile from unchanged Q and K, then reconstruct $p_{ts}$ from **maximum and scaled sum**. At scores near $7\times10^{37}$, adding $\log 3$ to a double can produce the original number. Consequently `exp(score - lse)` can be one for every tied key instead of one third. The split representation still divides by three. Backward needs the same safeguard as forward.

## Derive the missing row reduction

Let $\mathcal L$ be the scalar objective and use a bar to denote its derivative. From the weighted sum,

$$
\bar p_{ts}=\sum_{j=0}^{d-1}g_{t,j}v_{s,j},\qquad
\bar v_{s,j}=\sum_{t=s}^{T-1}p_{ts}g_{t,j}.
$$

Differentiating one softmax row gives

$$
\frac{\partial p_{tu}}{\partial a_{ts}}
=p_{tu}\bigl(\mathbf 1_{u=s}-p_{ts}\bigr),\qquad
\bar a_{ts}=p_{ts}\left(\bar p_{ts}-D_t\right),
$$

where $\mathbf 1_{u=s}$ is one when the indices agree and zero otherwise. The apparent obstacle is the row-wide sum $D_t=\sum_{u\leq t}p_{tu}\bar p_{tu}$. Substitute the definition of $\bar p$ and exchange two finite sums:

$$
D_t=\sum_{u=0}^{t}p_{tu}\sum_{j=0}^{d-1}g_{t,j}v_{u,j}
=\sum_{j=0}^{d-1}g_{t,j}\sum_{u=0}^{t}p_{tu}v_{u,j}
=\sum_{j=0}^{d-1}g_{t,j}C_{t,j}.
$$

Now the reduction needs only the saved context row and its incoming gradient, each of width $d$, rather than a sequence-length probability row. This identity and the tile recomputation strategy are described in [FlashAttention, version 2, Appendix B.2 and B.4](https://arxiv.org/html/2205.14135v2). Our implementation below is scalar CPU C, not the paper's CUDA kernel or a claim about its hardware speedups.

Finally, differentiate the scaled dot product:

$$
\bar q_{t,j}=\tau\sum_{s=0}^{t}\bar a_{ts}k_{s,j},\qquad
\bar k_{s,j}=\tau\sum_{t=s}^{T-1}\bar a_{ts}q_{t,j}.
$$

The factor $\tau$ appears in **both** derivatives. Values do not receive this factor, because they entered the weighted sum rather than the score.

### Add the head-sharing dimension

Let $H_q$ denote query heads, $H_{kv}$ key/value heads, and $G=H_q/H_{kv}$ the positive integral sharing ratio. Query head $h$ uses key/value head $k_h=\lfloor h/G\rfloor$. A key/value head $k$ therefore accumulates query heads $kG$ through $(k+1)G-1$:

$$
\bar K_{b,s,k,j}=\tau\sum_{h=kG}^{(k+1)G-1}\sum_{t=s}^{T-1}
\bar a_{b,h,t,s}Q_{b,t,h,j},
$$

$$
\bar V_{b,s,k,j}=\sum_{h=kG}^{(k+1)G-1}\sum_{t=s}^{T-1}
p_{b,h,t,s}g_{b,t,h,j}.
$$

Here $b$ selects a batch item. There is no factor $1/G$. Sharing a parameter means summing all uses, not averaging them. Two identical consuming heads with identical incoming gradients double the key/value derivative.

## Calculate one row before arranging tiles

Use scores $[0,\log2,\log4]$ and values $[(1,0),(0,2),(3,1)]$, as in Chapter 18. The probabilities are $[1,2,4]/7$, context is $(13/7,8/7)$, and choose incoming gradient $g=(2,-1)$. Then

$$
\bar p=[2,-2,5],\qquad D=2(13/7)-8/7=18/7,
$$

$$
\bar a=\left[-\frac4{49},-\frac{64}{49},\frac{68}{49}\right].
$$

The score derivatives sum to zero. Increasing all scores equally cannot change softmax, so this is an independent algebraic check. Each value receives its probability times $(2,-1)$.

Choose $q=(1,0)$ and keys $(0,0)$, $(\sqrt2\log2,0)$, $(\sqrt2\log4,0)$, giving the stated scores with $d=2$. The first query-gradient coordinate is

$$
\bar q_0=-\frac{64}{49}\log2+\frac{68}{49}\log4
=\frac{72}{49}\log2\approx1.018502.
$$

With **two** identical query heads, the shared first key-gradient coordinates become approximately $[-0.115446,-1.847136,1.962582]$. The shared value gradients are $[(4/7,-2/7),(8/7,-4/7),(16/7,-8/7)]$. Part 4 puts upstream signal only on these final query rows and checks these numbers without obtaining the expectations from another implementation.

## Choose who owns a gradient tile

A common tiled traversal updates partial `dQ` in float after every key tile. It is valid but introduces repeated float rounding. Here we use two sweeps so each destination tile has exactly one reduction owner:

| Sweep | Outer ownership | What remains double | When it is stored |
|---|---|---|---|
| Query sweep | Batch, query head, query tile | One $R\times d$ query-gradient tile | After every visible key tile has contributed |
| Key/value sweep | Batch, key/value head, key tile | Two $S\times d$ key/value-gradient tiles | After every sharing head and visible query has contributed |

$R$ is query-tile capacity and $S$ is key-tile capacity. Each sweep reconstructs a bounded probability/score-derivative tile. This deliberately repeats score, exponential, and incoming-value-dot work. It preserves double reduction until a single float store and avoids an extra sequence-sized double gradient arena. It does **not** eliminate quadratic pair work or promise a speedup.

The backward workspace owns

$$
W_b=2RS+Rd+2Sd+R
$$

doubles: probabilities, score derivatives, query reduction, key reduction, value reduction, and row centers. At $R=2,S=2,d=2$, that is $8+4+8+2=22$ doubles. At fixed tiles and head width it is independent of sequence length.

Pointers such as `double *ds = prob + qr * kc` are borrowed slices of one checked allocation. They are not independent owners. As in Chapter 2, destroy the one base allocation, never a slice. The `enum` below assigns the default zero value to materialized attention, so existing zero-initialized training state retains its previous behavior.

## The public interface

Replace `include/dl/attention_tiled.h` completely. The forward signatures are retained. The new backward interface requires a successful, unchanged tiled forward; it cannot accept a materialized probability tape as a substitute. Forward and backward tile sizes may differ for an isolated operator. The trainer later fixes both as part of its execution policy.

```c title="include/dl/attention_tiled.h"
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
```

Preflight can detect wrong geometry, aliases, nonfinite cells, and impossible basic normalization ranges. It cannot establish that all saved values belong to the same original computation. For example, a stale but finite context row can pass every shape check. The model-level revision/ready state guards that lifecycle; low-level callers must uphold it.

SET never reads old destination gradients. ADD reads finite old cells once when the complete reduction is stored. Both preserve padding. A late arithmetic failure may leave earlier outputs written, including ADD destinations. Discard the whole failed result and restart from a restored gradient boundary, not from those partly accumulated cells.

## Forward support and shared structural checks

Replace `src/attention_tiled.c` by concatenating the following blocks in order with one blank line between them. The forward arithmetic is retained; the shared view checker now admits the twelve read spans and eleven write spans needed by the complete backward wrapper. No compute routine allocates memory.

### Retain checked forward workspace ownership

`workspace_count` proves both the element count and its byte conversion before allocation. `row_capacity` counts query-head rows, not token rows. Destruction zeroes the owner so partial construction and repeated cleanup remain safe. The `Layout` record contains only checked geometry.

```c title="src/attention_tiled.c"
#include "dl/attention_tiled.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define TRY(call) do { DlStatus s_ = (call); if (s_ != DL_OK) return s_; } while (0)

static DlStatus workspace_count(size_t rows, size_t d, size_t qr, size_t kc,
                                size_t *count)
{
    if (rows == 0 || d == 0 || d % 2 != 0 || qr == 0 || kc == 0) return DL_SHAPE;
    size_t total = 0, term = 0, bytes = 0;
    TRY(dl_size_mul(3, rows, &total));
    TRY(dl_size_mul(qr, kc, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(qr, d, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(2, qr, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(total, sizeof(double), &bytes));
    *count = total;
    return DL_OK;
}

void dl_tiled_workspace_destroy(DlTiledWorkspace *s)
{
    if (s == NULL) return;
    free(s->data);
    *s = (DlTiledWorkspace){0};
}

DlStatus dl_tiled_workspace_create(DlTiledWorkspace *out, size_t rows, size_t d,
                                   size_t qr, size_t kc, size_t max_bytes)
{
    if (out == NULL || max_bytes == 0) return DL_ARGUMENT;
    if (out->data != NULL || out->count != 0 || out->row_capacity != 0 ||
        out->head_dim != 0 || out->query_tile != 0 || out->key_tile != 0) return DL_STATE;
    DlTiledWorkspace next = {.row_capacity = rows, .head_dim = d,
                             .query_tile = qr, .key_tile = kc};
    TRY(workspace_count(rows, d, qr, kc, &next.count));
    size_t bytes = next.count * sizeof(*next.data); /* Checked in workspace_count. */
    if (bytes > max_bytes) return DL_RANGE;
    next.data = malloc(bytes);
    if (next.data == NULL) return DL_ALLOC;
    *out = next;
    return DL_OK;
}

typedef struct {
    size_t rows, query_width, kv_width, stat_rows;
} Layout;
```

### Validate geometry without forming a square count

`rows=B*T`, `query_width=Hq*d`, `kv_width=Hkv*d`, and `stat_rows=B*Hq*T`. The layout deliberately never multiplies statistic rows by time. A very long sequence must not be rejected merely because an allocation we have removed would have overflowed.

```c title="src/attention_tiled.c"
static DlStatus layout(DlAttentionConfig c, Layout *z)
{
    if (c.batch == 0 || c.time == 0 || c.query_heads == 0 || c.kv_heads == 0 ||
        c.head_dim == 0 || c.head_dim % 2 != 0 || c.query_heads % c.kv_heads != 0)
        return DL_SHAPE;
    TRY(dl_size_mul(c.batch, c.time, &z->rows));
    TRY(dl_size_mul(c.query_heads, c.head_dim, &z->query_width));
    TRY(dl_size_mul(c.kv_heads, c.head_dim, &z->kv_width));
    TRY(dl_size_mul(z->rows, c.query_heads, &z->stat_rows));
    /* Deliberately no B*Hq*T*T product: that storage does not exist here. */
    return DL_OK;
}

static DlStatus workspace_check(const DlTiledWorkspace *s, Layout z, size_t d)
{
    if (s == NULL || s->data == NULL) return DL_ARGUMENT;
    size_t count = 0;
    TRY(workspace_count(s->row_capacity, s->head_dim, s->query_tile, s->key_tile, &count));
    if (s->count != count) return DL_STATE;
    if (s->head_dim != d || s->row_capacity < z.stat_rows) return DL_SHAPE;
    return DL_OK;
}
```

### Compare canonical float spans

A matrix view uses its original allocation base plus offset/leading dimension. The checker rejects overlapping written bounding spans; read/read sharing is legal. Metadata and double owners remain disjoint caller obligations. This is conservative about padding, matching the existing kernels. `store_float` rejects nonfinite or out-of-range results before conversion.

```c title="src/attention_tiled.c"
static int shape(DlConstMatrix v, size_t rows, size_t cols)
{
    return v.rows == rows && v.cols == cols;
}

static int finite_cells(DlConstMatrix v)
{
    for (size_t i = 0; i < v.rows; ++i)
        for (size_t j = 0; j < v.cols; ++j)
            if (!isfinite(v.base[v.offset + i * v.ld + j])) return 0;
    return 1;
}

/* Forward and backward share structural checks, not numeric storage. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                            const DlMatrix *out, size_t no)
{
    size_t ie[12] = {0}, oe[11] = {0};
    if (ni > 12 || no > 11) return DL_ARGUMENT;
    for (size_t i = 0; i < ni; ++i) TRY(dl_matrix_end(in[i], &ie[i]));
    for (size_t o = 0; o < no; ++o) {
        DlConstMatrix v = dl_matrix_read(out[o]);
        TRY(dl_matrix_end(v, &oe[o]));
        for (size_t i = 0; i < ni; ++i)
            if (v.base == in[i].base && v.offset < ie[i] && in[i].offset < oe[o])
                return DL_ALIAS;
        for (size_t p = 0; p < o; ++p)
            if (v.base == out[p].base && v.offset < oe[p] && out[p].offset < oe[o])
                return DL_ALIAS;
    }
    return DL_OK;
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

static size_t smaller(size_t a, size_t b) { return a < b ? a : b; }
```

### Keep the matched online forward

The next complete function is the Chapter 18 forward computation. Scores round through float, then online maxima, sums, and weighted numerators use double. Every completed query-head row saves maximum, scaled sum, and log-sum-exp. A key tile may cross the diagonal, so each row uses only its permitted prefix. Backward must reproduce those same score and mask conventions.

```c title="src/attention_tiled.c"
/* All geometry and input finiteness were checked by the public entry point. */
static DlStatus attend(DlAttentionConfig c, DlConstMatrix q, DlConstMatrix k,
                       DlConstMatrix v, DlTiledWorkspace *s, DlMatrix out)
{
    const size_t qr = s->query_tile, kc = s->key_tile, d = c.head_dim;
    double *tile = s->data + 3 * s->row_capacity;
    double *u = tile + qr * kc;
    double *maximum = u + qr * d;
    double *denominator = maximum + qr;
    const double scale = 1.0 / sqrt((double)d);
    const size_t group = c.query_heads / c.kv_heads;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t qb = 0; qb < c.time; ) {
                const size_t nq = smaller(qr, c.time - qb);
                for (size_t r = 0; r < nq; ++r) {
                    maximum[r] = -INFINITY;
                    denominator[r] = 0.0;
                    for (size_t j = 0; j < d; ++j) u[r * d + j] = 0.0;
                }
                /* All key blocks beginning at qb+nq or later are wholly masked. */
                for (size_t kb = 0; kb < qb + nq; ) {
                    const size_t nk = smaller(kc, c.time - kb);
                    /* Form a Q/K score tile, never a pre-existing square matrix. */
                    for (size_t r = 0; r < nq; ++r) {
                        const size_t t = qb + r;
                        const size_t qa = q.offset + (b * c.time + t) * q.ld + h * d;
                        for (size_t col = 0; col < nk; ++col) {
                            const size_t key_time = kb + col;
                            if (key_time > t) { tile[r * kc + col] = -INFINITY; continue; }
                            const size_t ka = k.offset + (b * c.time + key_time) * k.ld + kh * d;
                            double dot = 0.0;
                            for (size_t j = 0; j < d; ++j)
                                dot += (double)q.base[qa + j] * (double)k.base[ka + j];
                            float score = 0.0f;
                            TRY(store_float(&score, dot * scale));
                            tile[r * kc + col] = (double)score;
                        }
                    }
                    for (size_t r = 0; r < nq; ++r) {
                        const size_t t = qb + r;
                        /* This row may be fully masked within a diagonal tile. */
                        if (kb > t) continue;
                        const size_t allowed = smaller(nk, t - kb + 1);
                        double tile_max = -INFINITY;
                        for (size_t col = 0; col < allowed; ++col)
                            tile_max = fmax(tile_max, tile[r * kc + col]);
                        const double next_max = fmax(maximum[r], tile_max);
                        const double alpha = denominator[r] == 0.0 ? 0.0
                                           : exp(maximum[r] - next_max);
                        denominator[r] *= alpha;
                        for (size_t j = 0; j < d; ++j) u[r * d + j] *= alpha;
                        for (size_t col = 0; col < allowed; ++col) {
                            const double beta = exp(tile[r * kc + col] - next_max);
                            denominator[r] += beta;
                            const size_t va = v.offset + (b * c.time + kb + col) * v.ld + kh * d;
                            for (size_t j = 0; j < d; ++j)
                                u[r * d + j] += beta * (double)v.base[va + j];
                        }
                        maximum[r] = next_max;
                    }
                    kb += nk; /* Actual tail length; no overflowing kb+kc. */
                }
                for (size_t r = 0; r < nq; ++r) {
                    if (!isfinite(denominator[r]) || denominator[r] <= 0.0 ||
                        !isfinite(maximum[r])) return DL_NUMERIC;
                    const size_t t = qb + r;
                    const size_t sr = (b * c.query_heads + h) * c.time + t;
                    const double lse = maximum[r] + log(denominator[r]);
                    if (!isfinite(lse)) return DL_NUMERIC;
                    s->data[sr] = maximum[r];
                    s->data[s->row_capacity + sr] = denominator[r];
                    s->data[2 * s->row_capacity + sr] = lse;
                    const size_t oa = out.offset + (b * c.time + t) * out.ld + h * d;
                    for (size_t j = 0; j < d; ++j)
                        TRY(store_float(&out.base[oa + j], u[r * d + j] / denominator[r]));
                }
                qb += nq;
            }
        }
    }
    return DL_OK;
}
```

### Expose core and sublayer forward calls

The core accepts already projected/rotated arrays. The wrapper projects Q/K/V, rotates Q/K once, computes context, and applies the output projection. Backward will reverse exactly this graph; no normalization or residual is hidden in either wrapper.

```c title="src/attention_tiled.c"
DlStatus dl_attention_tiled_core(DlAttentionConfig c, DlConstMatrix q,
                                 DlConstMatrix k, DlConstMatrix v,
                                 DlTiledWorkspace *s, DlMatrix context)
{
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(workspace_check(s, z, c.head_dim));
    const DlConstMatrix in[] = {q, k, v};
    TRY(check_views(in, 3, &context, 1));
    if (!shape(q, z.rows, z.query_width) || !shape(k, z.rows, z.kv_width) ||
        !shape(v, z.rows, z.kv_width) ||
        !shape(dl_matrix_read(context), z.rows, z.query_width)) return DL_SHAPE;
    for (size_t i = 0; i < 3; ++i) if (!finite_cells(in[i])) return DL_NUMERIC;
    return attend(c, q, k, v, s, context);
}

DlStatus dl_attention_tiled_forward(DlBackend backend, DlAttentionConfig c,
                                    DlConstMatrix x, DlAttentionWeights w,
                                    const size_t *positions, size_t position_count,
                                    const double *freq, size_t frequency_count,
                                    DlTiledAttentionCache a,
                                    DlTiledWorkspace *s, DlMatrix y)
{
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) return DL_ARGUMENT;
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(workspace_check(s, z, c.head_dim));
    const DlConstMatrix in[] = {x, w.query, w.key, w.value, w.output};
    const DlMatrix out[] = {a.q0, a.k0, a.v, a.q, a.k, a.context, y};
    TRY(check_views(in, 5, out, 7));
    const size_t width = x.cols;
    if (x.rows != z.rows || !shape(w.query, z.query_width, width) ||
        !shape(w.key, z.kv_width, width) || !shape(w.value, z.kv_width, width) ||
        !shape(w.output, width, z.query_width) ||
        !shape(dl_matrix_read(a.q0), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k0), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.v), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.q), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.context), z.rows, z.query_width) ||
        !shape(dl_matrix_read(y), z.rows, width)) return DL_SHAPE;
    if (positions == NULL || freq == NULL) return DL_ARGUMENT;
    if (position_count != z.rows || frequency_count != c.head_dim / 2) return DL_SHAPE;
    size_t bytes = 0;
    TRY(dl_size_mul(position_count, sizeof(*positions), &bytes));
    TRY(dl_size_mul(frequency_count, sizeof(*freq), &bytes));
    for (size_t i = 0; i < position_count; ++i)
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    for (size_t i = 0; i < frequency_count; ++i)
        if (!isfinite(freq[i]) || freq[i] <= 0.0 || freq[i] > 1.0) return DL_ARGUMENT;
    for (size_t i = 0; i < 5; ++i) if (!finite_cells(in[i])) return DL_NUMERIC;
    TRY(dl_matmul(backend, x, DL_NOTRANS, w.query, DL_TRANS, a.q0, DL_SET));
    TRY(dl_matmul(backend, x, DL_NOTRANS, w.key, DL_TRANS, a.k0, DL_SET));
    TRY(dl_matmul(backend, x, DL_NOTRANS, w.value, DL_TRANS, a.v, DL_SET));
    TRY(dl_rope_forward(dl_matrix_read(a.q0), c.head_dim, positions, position_count,
                         freq, frequency_count, a.q));
    TRY(dl_rope_forward(dl_matrix_read(a.k0), c.head_dim, positions, position_count,
                         freq, frequency_count, a.k));
    TRY(attend(c, dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.v), s, a.context));
    return dl_matmul(backend, dl_matrix_read(a.context), DL_NOTRANS, w.output, DL_TRANS, y, DL_SET);
}
```

## Allocate only the backward tiles

The backward owner has no row capacity because it retains no per-sequence statistics. Its count formula is independent of $B$, $T$, and head counts. `max_bytes` includes every double in this owner. An overflow or inadequate budget leaves the caller's empty owner unchanged.

```c title="src/attention_tiled.c"
static DlStatus backward_count(size_t d, size_t qr, size_t kc, size_t *count)
{
    if (d == 0 || d % 2 != 0 || qr == 0 || kc == 0) return DL_SHAPE;
    size_t total = 0, term = 0, bytes = 0;
    TRY(dl_size_mul(qr, kc, &term));
    TRY(dl_size_mul(2, term, &total));
    TRY(dl_size_mul(qr, d, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_mul(kc, d, &term));
    TRY(dl_size_mul(2, term, &term));
    TRY(dl_size_add(total, term, &total));
    TRY(dl_size_add(total, qr, &total));
    TRY(dl_size_mul(total, sizeof(double), &bytes));
    *count = total;
    return DL_OK;
}

void dl_tiled_backward_workspace_destroy(DlTiledBackwardWorkspace *s)
{
    if (s == NULL) return;
    free(s->data);
    *s = (DlTiledBackwardWorkspace){0};
}

DlStatus dl_tiled_backward_workspace_create(DlTiledBackwardWorkspace *out,
    size_t d, size_t qr, size_t kc, size_t max_bytes)
{
    if (out == NULL || max_bytes == 0) return DL_ARGUMENT;
    if (out->data != NULL || out->count != 0 || out->head_dim != 0 ||
        out->query_tile != 0 || out->key_tile != 0) return DL_STATE;
    DlTiledBackwardWorkspace next = {.head_dim = d, .query_tile = qr, .key_tile = kc};
    TRY(backward_count(d, qr, kc, &next.count));
    size_t bytes = next.count * sizeof(*next.data);
    if (bytes > max_bytes) return DL_RANGE;
    next.data = malloc(bytes);
    if (next.data == NULL) return DL_ALLOC;
    *out = next;
    return DL_OK;
}
```

## Check saved normalization before writing

The allowed count for statistic row `sr` is `sr % time + 1`, since layout is batch, query head, then time. At least one maximum contributes one to the scaled sum. The upper bound permits a tiny double-rounding excess over the number of visible keys. A valid large maximum is still allowed; log-sum-exp need not differ numerically from that maximum.

These are sanity checks, not a second forward or a proof of freshness. The later score reconstruction also rejects a recomputed score greater than its saved maximum.

```c title="src/attention_tiled.c"
static DlStatus backward_state(DlAttentionConfig c, Layout z,
    const DlTiledWorkspace *forward, const DlTiledBackwardWorkspace *s)
{
    TRY(workspace_check(forward, z, c.head_dim));
    if (s == NULL || s->data == NULL) return DL_ARGUMENT;
    size_t count = 0;
    TRY(backward_count(s->head_dim, s->query_tile, s->key_tile, &count));
    if (s->count != count) return DL_STATE;
    if (s->head_dim != c.head_dim) return DL_SHAPE;
    if (s->data == forward->data) return DL_ALIAS;
    for (size_t sr = 0; sr < z.stat_rows; ++sr) {
        double maximum = forward->data[sr];
        double sum = forward->data[forward->row_capacity + sr];
        double lse = forward->data[2 * forward->row_capacity + sr];
        /* One allowed maximum contributes exp(0)=1. Roundoff allows a tiny
         * excess over the number of allowed terms, never an arbitrary factor. */
        double allowed = (double)(sr % c.time + 1);
        if (!isfinite(maximum) || fabs(maximum) > (double)FLT_MAX ||
            !isfinite(sum) || sum < 1.0 || sum > allowed * (1.0 + 1e-12) ||
            !isfinite(lse)) return DL_NUMERIC;
    }
    return DL_OK;
}
```

## Recompute probability and score derivative together

`at` maps a logical row/column to a padded view. `derivative_tile` first computes `center[r]` from the **saved context** and incoming context gradient. For each visible pair it calculates the float-rounded score, reconstructs double probability, computes the incoming-value dot product, and writes double score derivative.

The score scale belongs in score reconstruction here and in Q/K differentiation later. It does not belong in `dp`. Masked cells are zero and their key/value contents do not participate in a reduction. The complete tile function follows; no full score row is hidden behind it.

```c title="src/attention_tiled.c"
static size_t at(DlConstMatrix a, size_t row, size_t column)
{
    return a.offset + row * a.ld + column;
}

/* One score/probability and score-derivative tile. Center is local to its
 * query rows. Masked cells are explicit zero and never enter a reduction. */
static DlStatus derivative_tile(DlAttentionConfig c,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dc, const DlTiledWorkspace *f,
    size_t b, size_t h, size_t qb, size_t nq, size_t kb, size_t nk,
    size_t stride, double *prob, double *ds, double *center)
{
    size_t d = c.head_dim, kh = h / (c.query_heads / c.kv_heads);
    double scale = 1.0 / sqrt((double)d);
    for (size_t r = 0; r < nq; ++r) {
        size_t t = qb + r, row = b * c.time + t;
        size_t sr = (b * c.query_heads + h) * c.time + t;
        size_t qa = at(q, row, h * d), ca = at(context, row, h * d);
        size_t ga = at(dc, row, h * d);
        center[r] = 0.0;
        for (size_t j = 0; j < d; ++j)
            center[r] += (double)context.base[ca + j] * (double)dc.base[ga + j];
        if (!isfinite(center[r])) return DL_NUMERIC;
        for (size_t col = 0; col < nk; ++col) {
            size_t index = r * stride + col, key = kb + col;
            prob[index] = ds[index] = 0.0;
            if (key > t) continue;
            size_t ka = at(k, b * c.time + key, kh * d);
            size_t va = at(v, b * c.time + key, kh * d);
            double dot = 0.0, dp = 0.0;
            for (size_t j = 0; j < d; ++j) {
                dot += (double)q.base[qa + j] * (double)k.base[ka + j];
                dp += (double)dc.base[ga + j] * (double)v.base[va + j];
            }
            float score = 0.0f;
            TRY(store_float(&score, dot * scale));
            double delta = (double)score - f->data[sr];
            if (delta > 0.0) return DL_NUMERIC; /* Inconsistent saved maximum. */
            double p = exp(delta) / f->data[f->row_capacity + sr];
            double g = p * (dp - center[r]);
            if (!isfinite(p) || !isfinite(g)) return DL_NUMERIC;
            prob[index] = p;
            ds[index] = g;
        }
    }
    return DL_OK;
}
```

## Reduce and store each gradient once

`write_gradient` adds an old destination only in ADD mode. In the query sweep, `gq` is cleared **outside** the key loop. In the key/value sweep, `gk` and `gv` are cleared **outside** the sharing-head loop. Moving either initialization inward silently discards earlier contributions without causing an invalid memory access.

Starting the second sweep's query block at `kb` is safe: query times below the first key cannot see any key in that tile. It need not use the same query-block boundaries as the first sweep. Every visible pair is nevertheless visited once in each sweep. Tail increments use actual remaining lengths, not an unchecked `start + configured_capacity`.

```c title="src/attention_tiled.c"
static DlStatus write_gradient(DlMatrix out, size_t row, size_t col,
                                double value, DlWriteMode mode)
{
    float *cell = &out.base[out.offset + row * out.ld + col];
    if (mode == DL_ADD) value += (double)*cell;
    return store_float(cell, value);
}

/* Two sweeps give each output tile one owner. No partially rounded dQ is
 * repeatedly read from float memory, and all sharing heads enter dK/dV. */
static DlStatus backward_compute(DlAttentionConfig c,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dc, const DlTiledWorkspace *f,
    DlTiledBackwardWorkspace *s, DlMatrix dq, DlMatrix dk, DlMatrix dv,
    DlWriteMode mode)
{
    size_t qr = s->query_tile, kc = s->key_tile, d = c.head_dim;
    size_t group = c.query_heads / c.kv_heads;
    double scale = 1.0 / sqrt((double)d);
    double *prob = s->data, *ds = prob + qr * kc;
    double *gq = ds + qr * kc, *gk = gq + qr * d;
    double *gv = gk + kc * d, *center = gv + kc * d;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            size_t kh = h / group;
            for (size_t qb = 0; qb < c.time; ) {
                size_t nq = smaller(qr, c.time - qb);
                for (size_t i = 0; i < nq * d; ++i) gq[i] = 0.0;
                for (size_t kb = 0; kb < qb + nq; ) {
                    size_t nk = smaller(kc, c.time - kb);
                    TRY(derivative_tile(c, q, k, v, context, dc, f,
                        b, h, qb, nq, kb, nk, kc, prob, ds, center));
                    for (size_t r = 0; r < nq; ++r)
                        for (size_t col = 0; col < nk && kb + col <= qb + r; ++col) {
                            size_t ka = at(k, b * c.time + kb + col, kh * d);
                            for (size_t j = 0; j < d; ++j)
                                gq[r * d + j] += ds[r * kc + col] * (double)k.base[ka + j];
                        }
                    kb += nk;
                }
                for (size_t r = 0; r < nq; ++r)
                    for (size_t j = 0; j < d; ++j)
                        TRY(write_gradient(dq, b * c.time + qb + r, h * d + j,
                                            scale * gq[r * d + j], mode));
                qb += nq;
            }
        }
        for (size_t kh = 0; kh < c.kv_heads; ++kh) {
            for (size_t kb = 0; kb < c.time; ) {
                size_t nk = smaller(kc, c.time - kb);
                for (size_t i = 0; i < nk * d; ++i) gk[i] = gv[i] = 0.0;
                for (size_t gh = 0; gh < group; ++gh) {
                    size_t h = kh * group + gh;
                    /* Queries earlier than kb cannot consume any of this tile. */
                    for (size_t qb = kb; qb < c.time; ) {
                        size_t nq = smaller(qr, c.time - qb);
                        TRY(derivative_tile(c, q, k, v, context, dc, f,
                            b, h, qb, nq, kb, nk, kc, prob, ds, center));
                        for (size_t r = 0; r < nq; ++r) {
                            size_t qa = at(q, b * c.time + qb + r, h * d);
                            size_t ga = at(dc, b * c.time + qb + r, h * d);
                            for (size_t col = 0; col < nk && kb + col <= qb + r; ++col)
                                for (size_t j = 0; j < d; ++j) {
                                    gk[col * d + j] += ds[r * kc + col] * (double)q.base[qa + j];
                                    gv[col * d + j] += prob[r * kc + col] * (double)dc.base[ga + j];
                                }
                        }
                        qb += nq;
                    }
                }
                for (size_t col = 0; col < nk; ++col)
                    for (size_t j = 0; j < d; ++j) {
                        TRY(write_gradient(dk, b * c.time + kb + col, kh * d + j,
                                            scale * gk[col * d + j], mode));
                        TRY(write_gradient(dv, b * c.time + kb + col, kh * d + j,
                                            gv[col * d + j], mode));
                    }
                kb += nk;
            }
        }
    }
    return DL_OK;
}
```

## Enter the core only after complete preflight

The five read views are Q, K, V, context, and incoming context gradient. The three destinations are Q/K/V gradients. Mode validation, saved-state checks, all shapes/spans, all finite inputs, and all old ADD cells are checked before `backward_compute` can touch scratch or output. SET permits old NaNs because it overwrites rather than depends on them.

```c title="src/attention_tiled.c"
DlStatus dl_attention_tiled_core_backward(DlAttentionConfig c,
    DlConstMatrix q, DlConstMatrix k, DlConstMatrix v,
    DlConstMatrix context, DlConstMatrix dc,
    const DlTiledWorkspace *f, DlTiledBackwardWorkspace *s,
    DlMatrix dq, DlMatrix dk, DlMatrix dv, DlWriteMode mode)
{
    if (mode != DL_SET && mode != DL_ADD) return DL_ARGUMENT;
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(backward_state(c, z, f, s));
    const DlConstMatrix in[] = {q, k, v, context, dc};
    const DlMatrix out[] = {dq, dk, dv};
    TRY(check_views(in, 5, out, 3));
    size_t widths[] = {z.query_width, z.kv_width, z.kv_width, z.query_width, z.query_width};
    for (size_t i = 0; i < 5; ++i) {
        if (!shape(in[i], z.rows, widths[i])) return DL_SHAPE;
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    for (size_t o = 0; o < 3; ++o) {
        if (!shape(dl_matrix_read(out[o]), z.rows, widths[o])) return DL_SHAPE;
        if (mode == DL_ADD && !finite_cells(dl_matrix_read(out[o]))) return DL_NUMERIC;
    }
    return backward_compute(c, q, k, v, context, dc, f, s, dq, dk, dv, mode);
}
```

## Reverse the complete sublayer

Let input $X$ have shape $[BT,D]$, concatenated context $C$ have shape $[BT,Q]$ with $Q=H_qd$, and output weight $W_o$ have shape $[D,Q]$. Since $Y=CW_o^\top$, first compute $\bar C=\bar YW_o$ and $\bar W_o=\bar Y^\top C$. After the tiled core, apply each RoPE transpose to obtain the unrotated query/key derivatives.

For each projection $U=XW^\top$, accumulate $\bar W=\bar U^\top X$ and $\bar X=\bar U W$. The Q path uses the caller's input mode; K and V use ADD to retain the preceding paths. Weight mode is independent. The wrapper permits $D\ne Q$; the tests use $D=6,Q=8$ to make orientation mistakes visible.

The wrapper preflights all destinations before writing even the first output-weight derivative. Later arithmetic can still fail after some writes, so successful preflight is not transactional backward execution.

```c title="src/attention_tiled.c"
DlStatus dl_attention_tiled_backward(DlBackend backend, DlAttentionConfig c,
    DlConstMatrix x, DlAttentionWeights w,
    const size_t *positions, size_t position_count,
    const double *freq, size_t frequency_count,
    DlTiledAttentionCache a, const DlTiledWorkspace *f,
    DlConstMatrix dy, DlTiledAttentionScratch g,
    DlTiledBackwardWorkspace *s, DlMatrix dx, DlWriteMode dx_mode,
    DlAttentionWeightGrads dw, DlWriteMode weight_mode)
{
    if ((backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) ||
        (dx_mode != DL_SET && dx_mode != DL_ADD) ||
        (weight_mode != DL_SET && weight_mode != DL_ADD)) return DL_ARGUMENT;
    Layout z = {0};
    TRY(layout(c, &z));
    TRY(backward_state(c, z, f, s));
    const DlConstMatrix in[] = {x, w.query, w.key, w.value, w.output,
        dl_matrix_read(a.q0), dl_matrix_read(a.k0), dl_matrix_read(a.v),
        dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.context), dy};
    const DlMatrix out[] = {g.context, g.q, g.k, g.v, g.q0, g.k0,
                            dx, dw.query, dw.key, dw.value, dw.output};
    TRY(check_views(in, 12, out, 11));
    size_t m = z.rows, d = x.cols, q = z.query_width, k = z.kv_width;
    const size_t ir[] = {m,q,k,k,d,m,m,m,m,m,m,m};
    const size_t ic[] = {d,d,d,d,q,q,k,k,q,k,q,d};
    const size_t orows[] = {m,m,m,m,m,m,m,q,k,k,d};
    const size_t ocols[] = {q,q,k,k,q,k,d,d,d,d,q};
    for (size_t i = 0; i < 12; ++i) {
        if (!shape(in[i], ir[i], ic[i])) return DL_SHAPE;
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    for (size_t o = 0; o < 11; ++o)
        if (!shape(dl_matrix_read(out[o]), orows[o], ocols[o])) return DL_SHAPE;
    if (positions == NULL || freq == NULL) return DL_ARGUMENT;
    if (position_count != m || frequency_count != c.head_dim / 2) return DL_SHAPE;
    size_t bytes = 0;
    TRY(dl_size_mul(position_count, sizeof(*positions), &bytes));
    TRY(dl_size_mul(frequency_count, sizeof(*freq), &bytes));
    for (size_t i = 0; i < position_count; ++i)
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    for (size_t i = 0; i < frequency_count; ++i)
        if (!isfinite(freq[i]) || freq[i] <= 0.0 || freq[i] > 1.0) return DL_ARGUMENT;
    if (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) return DL_NUMERIC;
    for (size_t o = 7; o < 11; ++o)
        if (weight_mode == DL_ADD && !finite_cells(dl_matrix_read(out[o]))) return DL_NUMERIC;
    TRY(dl_matmul(backend, dy, DL_NOTRANS, w.output, DL_NOTRANS, g.context, DL_SET));
    TRY(dl_matmul(backend, dy, DL_TRANS, dl_matrix_read(a.context), DL_NOTRANS,
                   dw.output, weight_mode));
    TRY(backward_compute(c, dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.v),
        dl_matrix_read(a.context), dl_matrix_read(g.context), f, s, g.q, g.k, g.v, DL_SET));
    TRY(dl_rope_backward(dl_matrix_read(g.q), c.head_dim, positions, position_count,
                         freq, frequency_count, g.q0, DL_SET));
    TRY(dl_rope_backward(dl_matrix_read(g.k), c.head_dim, positions, position_count,
                         freq, frequency_count, g.k0, DL_SET));
    const DlConstMatrix upstream[] = {dl_matrix_read(g.q0), dl_matrix_read(g.k0),
                                       dl_matrix_read(g.v)};
    const DlConstMatrix weights[] = {w.query, w.key, w.value};
    const DlMatrix gradients[] = {dw.query, dw.key, dw.value};
    for (size_t p = 0; p < 3; ++p) {
        TRY(dl_matmul(backend, upstream[p], DL_TRANS, x, DL_NOTRANS,
                       gradients[p], weight_mode));
        TRY(dl_matmul(backend, upstream[p], DL_NOTRANS, weights[p], DL_NOTRANS,
                       dx, p == 0 ? dx_mode : DL_ADD));
    }
    return DL_OK;
}

#undef TRY
```

## Precision is part of the interpretation

The identity $D_t=g_t^\top C_t$ is exact over real numbers. Our saved `context` is float, however, while its online numerator and denominator were double. Therefore that dot product can differ slightly from recomputing $\sum p\bar p$ with all-double probabilities. Chapter 11 additionally rounds probabilities, incoming probability derivatives, and score derivatives to float. The two paths implement the same differentiable attention equations but have different rounding boundaries.

We do not differentiate the discontinuous float conversion itself; this is the same continuous-operator convention used by the earlier gradient checks. Near float quantization boundaries, a numerical difference quotient includes that rounding noise. Use a finite-difference step large enough to resolve the intended change, test an independent formula and hand example, and inspect absolute as well as scaled error. Do not demand bitwise gradient equality, and do not use a relaxed tolerance to excuse a missing sharing head or missing $1/\sqrt d$.

A single visible key has probability one and zero score derivative; only the value path remains. Very large finite inputs can still produce a gradient outside float range, which returns `DL_NUMERIC`. Neither numerical stability nor tiling means every finite input has a representable finite derivative.

Continue with [Part 2 — A linear training tape and paired dispatch](https://25d-228.github.io/gpt-c/chapter-19-part-2.html).
