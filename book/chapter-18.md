# Chapter 18 — Online normalization and tiled attention forward

A causal attention row needs its weighted sum, not a permanent list of every probability. The materialized implementation nevertheless allocates both scores and probabilities for every query, head, and batch item. This chapter changes that storage schedule: generate a bounded query/key tile, merge its contribution into a running normalization state, and discard the tile. The attention formula and its causal dependencies remain the same.

**Prerequisites:** Chapter 10's materialized attention, Chapter 16's immutable weight owner and reusable inference workspace, and the complete [Chapter 17 source state](https://25d-228.github.io/gpt-c/media/chapter-17/reference/Makefile). Keep the existing trainer, materialized backward pass, checkpoint format, and persistent key/value cache. The new forward is an additional inference strategy, not a replacement for a backward pass that still needs stored probabilities.

**Implementation estimate:** **5.00 hours** for the entire chapter: 3.25 for online state, tiling, and inference integration; 1.75 for equivalence, masking, storage, and failure checks. Reading and unattended computation are separate. This part derives and implements attention; [Part 2](https://25d-228.github.io/gpt-c/chapter-18-part-2.html) integrates inference and generation; [Part 3](https://25d-228.github.io/gpt-c/chapter-18-part-3.html) supplies complete executable checks and experiments. These are three parts of one chapter.

The [71-file companion](https://25d-228.github.io/gpt-c/media/chapter-18/reference/Makefile) is the cumulative source state. Create `include/dl/attention_tiled.h`, `src/attention_tiled.c`, and `tests/test_tiled.c`. Replace `include/dl/inference.h`, `src/inference.c`, `app/generate.c`, `tests/test_generate_cli.py`, and the Makefile with the complete listings here. The other 63 prerequisite files are unchanged. Concatenate blocks bearing the same filename in part order, with one blank line between blocks. Untitled explanatory snippets are not extra source files.

> **Learning objectives**
> Derive and trace normalization under a moving maximum; generate scores in two-dimensional tiles; handle causal edges without reading masked values; preserve grouped-head sharing; distinguish mathematical equivalence from floating-point identity; and account for every owned attention buffer rather than only the score tile.

## Three different meanings of “remember less”

Chapter 16 recomputes the complete prefix and retains square score and probability arrays in one reusable layer workspace. Chapter 17 remembers old keys and values across generation calls. Its attention calculation still uses a materialized score row and probability row, each as long as the cache capacity. It already avoids square inference scratch, but it is not an online softmax implementation.

Here we return to full-prefix execution and remove its square arrays. Queries and keys are still computed for every prefix row; the generator still recomputes the prefix after selecting each new token. What changes is the temporary attention representation within each call. The persistent cache remains available as a separate `cache` strategy. Combining tiled prefill with persistent history is possible, but is not necessary to establish this mechanism or its reference checks.

Neither smaller storage nor a tiled loop proves a speed improvement. This C implementation uses ordinary heap memory and scalar score loops. It does not control GPU on-chip storage. The original [FlashAttention paper, Section 3.1 and Appendix B.3](https://arxiv.org/html/2205.14135v2#S3.SS1), describes tiling and normalization statistics in a GPU-oriented algorithm. We use those mathematical ideas, not that kernel or its performance results.

## Fix the shapes before changing the schedule

Let $B$ be the batch size, $T$ the number of tokens in each sequence, $H_q$ the number of query heads, $H_{kv}$ the number of key/value heads, and $d$ the even head dimension. All are positive and $H_q$ is divisible by $H_{kv}$. Write $M=BT$, $Q_w=H_qd$, and $K_w=H_{kv}d$. Already-rotated queries have shape $[M,Q_w]$; already-rotated keys and unrotated values have shape $[M,K_w]$. The concatenated context has shape $[M,Q_w]$.

Head $h$ uses key/value head

$$
g(h)=\left\lfloor\frac{h}{H_q/H_{kv}}\right\rfloor.
$$

For batch item $b$, query time $t$, and allowed key time $s\leq t$, define the score $a_{b,h,t,s}$, probability $p_{b,h,t,s}$, and context coordinate $o_{b,t,h,j}$ by

$$
a_{b,h,t,s}=\frac{1}{\sqrt d}\sum_{j=0}^{d-1}
 q_{b,t,h,j}k_{b,s,g(h),j},
$$

$$
p_{b,h,t,s}=\frac{\exp(a_{b,h,t,s})}{\sum_{r=0}^{t}\exp(a_{b,h,t,r})},
\qquad
o_{b,t,h,j}=\sum_{s=0}^{t}p_{b,h,t,s}v_{b,s,g(h),j}.
$$

The index $r$ in the denominator is another allowed key time, not a query-tile row. Keys belonging to another batch item never participate. The causal mask depends on sequence time, not the numerical RoPE position. RoPE has already acted on the coordinates before this core is called.

The materialized implementation stores $2BH_qT^2$ floats for scores and probabilities, including masked entries. Its actual allowed dot products number $BH_qT(T+1)/2$. Our new implementation keeps that same set of dot products but not their complete storage. Tiling is not sparse attention and does not make the quadratic arithmetic disappear.

## A normalization state whose scale may change

Temporarily suppress batch, head, and query indices. Suppose a nonempty set $A$ of allowed keys has been visited. Maintain one maximum $m$, one scaled sum $\ell$, and a vector $u$ of $d$ weighted numerators:

$$
m=\max_{s\in A}a_s,\qquad
\ell=\sum_{s\in A}\exp(a_s-m),\qquad
u_j=\sum_{s\in A}\exp(a_s-m)v_{s,j}.
$$

At the end of the row, $o_j=u_j/\ell$. A stored numerator is **not yet a normalized output**. That distinction determines the update equation.

Now visit a new tile of allowed keys $E$. Let $\mu=\max_{s\in E}a_s$ and $m'=\max(m,\mu)$. Every old exponential must be expressed relative to this new maximum. For an old score,

$$
\exp(a_s-m')=\exp(a_s-m)\exp(m-m').
$$

Thus one common factor $\alpha=\exp(m-m')$ rescales both old sums. Defining $\beta_s=\exp(a_s-m')$ for new keys gives

$$
\ell'=\alpha\ell+\sum_{s\in E}\beta_s,
\qquad
u'_j=\alpha u_j+\sum_{s\in E}\beta_s v_{s,j}.
$$

This is also a direct proof of the invariant: substitute the old sums, distribute $\alpha$, and the two sums become the union $A\cup E$ expressed in one scale. Tiles visit disjoint key ranges. No probability needs to be stored after its weighted contribution has been added.

A tile could first produce its own $\ell_E$ and $u_E$ relative to $\mu$ and then merge them with the factor $\exp(\mu-m')$. Our code instead computes each $\beta_s$ directly relative to $m'$. That removes a separate tile-numerator buffer while implementing the same union of sums. The [online-normalizer paper, Sections 3 and 3.1](https://arxiv.org/html/1805.02867v1#S3), develops the running-normalizer update and its merge operation. The vector numerator here follows by applying the same rescaling to each weighted coordinate.

An empty history uses $\ell=0$ and $u=0$. Its maximum is a sentinel, `-INFINITY`, not a score. On the first nonempty tile, set $\alpha=0$ explicitly. For a nonempty row, at least one maximum contributes an exponential of one. All exponential arguments are nonpositive. Extremely small weights may underflow to zero; an exponential of a large positive score is never formed.

## Work one row before writing the loops

Use the ideal scores $[0,\log 2,\log 4]$ and values $[(1,0),(0,2),(3,1)]$. The unnormalized weights are $[1,2,4]$. Direct arithmetic gives

$$
o=\left(\frac{1+0+12}{7},\frac{0+4+4}{7}\right)
  =\left(\frac{13}{7},\frac{8}{7}\right).
$$

Process one key at a time:

| Last key visited | Maximum $m$ | Scaled sum $\ell$ | Numerator $u$ |
|---|---:|---:|---:|
| 0 | $0$ | $1$ | $(1,0)$ |
| 1 | $\log 2$ | $1.5$ | $(0.5,2)$ |
| 2 | $\log 4$ | $1.75$ | $(3.25,2)$ |

The second and third keys each raise the maximum by $\log 2$, so $\alpha=1/2$. At the last key, the old numerator $(0.5,2)$ becomes $(0.25,1)$ before adding $(3,1)$. Dividing the final numerator by $1.75$ recovers the direct answer. Forgetting to rescale the numerator gives the wrong weighted average even when the denominator is correct.

The same row can use a two-key first tile. Its maximum is $\log 2$, its exponential weights are $[1/2,1]$, and its state is immediately $(m,\ell,u)=(\log2,1.5,(0.5,2))$. The last one-key tile produces the same mathematical result. Floating-point addition can change the last bits under different partitions, but not the function we intend to evaluate.

The executable fixture does not feed these scores to the kernel. It supplies $q=(1,0)$ and keys $(0,0)$, $(\sqrt2\log2,0)$, and $(\sqrt2\log4,0)$ with $d=2$. The kernel must actually form the scaled dot products. Keys are stored as floats, so the test accepts a small tolerance around the ideal rational answer rather than demanding exact fractions.

## Generate a real query/key tile

Let $R$ be the maximum query rows per tile and $S$ the maximum key columns per tile. Start with $R=16$ and $S=32$. For one batch item and one query head, `qb` is the first query time and `kb` the first key time. Actual edge lengths are

$$
n_q=\min(R,T-q_b),\qquad n_k=\min(S,T-k_b).
$$

The temporary score tile has physical row stride $S$, including when $n_k<S$. Tile entry $(r,c)$ uses query time $t=q_b+r$ and key time $s=k_b+c$. Its address is `tile[r*S + c]`, not `tile[r*nk + c]`. The numerator's address is `u[r*d + j]`.

For an example with $B=2,T=5,H_q=4,H_{kv}=2,d=2$, take batch $b=1$, head $h=3$, query $t=2$, and key $s=1$. The shared key head is $g(3)=1$. The query lives in matrix row $1\cdot5+2=7$, coordinates 6 and 7. The key lives in row 6, coordinates 2 and 3. Each matrix's actual leading dimension and offset still apply. A padded query access is `q.offset + 7*q.ld + 6`, not a dense-shape shortcut.

Saved row statistics use head-major rows, as the materialized score matrix did:

$$
\operatorname{sr}(b,h,t)=(bH_q+h)T+t.
$$

This example's statistic index is $(1\cdot4+3)5+2=37$. Context remains token-major: its matrix row is 7 and its head begins at coordinate 6. Confusing these two flattenings can preserve finite values while mixing heads or batches.

### The diagonal is an irregular boundary inside a rectangular tile

A whole key tile beginning at or after `qb+nq` is masked for every query in this query tile. The outer key loop stops before such tiles. Within a tile that crosses the diagonal, an early query can have no allowed columns even while a later query has some. Skip that query row when `kb > t`. Otherwise the allowed column count is `min(nk, t-kb+1)`.

For `qb=0`, `nq=3`, and key tiles of width 2, the tile beginning at key 2 is entirely masked for query times 0 and 1. Query time 2 uses only its first column. We do not subtract two negative-infinity sentinels or evaluate a masked dot product. We also avoid an unsigned `t-kb` when `kb>t`.

Every complete row has at least its diagonal key. An entirely masked complete row is therefore impossible under this API's positive-length causal contract. Supporting padding masks or packed documents would require a separate policy for genuinely empty rows; no such policy is silently inferred here.

## Declare the owner and the two computation interfaces

Create `include/dl/attention_tiled.h`. `DlTiledWorkspace` owns one double allocation. The row capacity is a capacity in **query-head rows**, not token rows. `DlTiledAttentionCache` contains borrowed float views for the six linear intermediates. It has no score or probability view.

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
 * ONLY projections. Training keeps its materialized forward/backward pair in
 * this chapter. No residual, norm, dropout, bias, allocation or hidden counter. */
DlStatus dl_attention_tiled_forward(DlBackend backend, DlAttentionConfig config,
                                    DlConstMatrix x, DlAttentionWeights weights,
                                    const size_t *positions, size_t position_count,
                                    const double *inv_freq, size_t frequency_count,
                                    DlTiledAttentionCache cache,
                                    DlTiledWorkspace *workspace, DlMatrix y);

#endif
```

The core accepts projected, rotated queries and keys directly. This makes its normalization independently testable. The full sublayer supplies projections, RoPE, and output projection around that core. `DlBackend` continues to select only the matrix product implementation; it does not encode the attention storage strategy.

A live workspace must not be copied by struct assignment: that would create two owners of the same allocation. Only a temporary owner constructed locally is transferred to an initially empty caller at successful creation. The owner and its double storage must not overlap float arrays, position arrays, or other metadata. As with earlier chapters, C cannot infer truthful pointer lifetimes or hidden aliases from arbitrary pointers.

### Count the entire double allocation

Let $N$ be the row capacity. The allocation contains three $N$-element statistic planes, one $R\times S$ tile, one $R\times d$ numerator array, and two $R$-element state arrays:

$$
N_{\rm double}=3N+RS+Rd+2R.
$$

The three planes store maximum, scaled sum, and log-sum-exp. The working maximum and denominator arrays are separate because only the current query tile is being updated. There is no second tile-probability allocation: exponentials are scalar temporaries consumed immediately. There are no packed query/key/value tiles; the loops read their declared input views directly.

Start `src/attention_tiled.c` with the constructor and destructor. Checked multiplication must precede allocation and pointer partitioning. A rejected budget does not become an attempted huge allocation.

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
```

`workspace_count` checks the byte product as well as the element count. The later `next.count * sizeof(*next.data)` is safe because that same product has already succeeded. `malloc` deliberately leaves numbers uninitialized: every computation initializes the active tile and state, and successful rows overwrite their statistic cells. The tests poison scratch with NaNs to expose unintended reads of previous contents.

For the small owner in the limit test, $N=3,d=2,R=S=1$. Its count is $9+1+2+2=14$ doubles. The exact 112-byte payload budget is accepted on the course's eight-byte-double target; a budget of eight bytes is rejected. Metadata and allocator overhead are not included in that payload bound.

## Preflight without requiring a square size

Continue the source with its layout and geometry checks. The layout checks $BT$, $H_qd$, $H_{kv}d$, and $BH_qT$. It never computes $BH_qT^2$. A supposedly memory-bounded implementation should not retain a hidden square overflow check from an array it no longer owns.

```c title="src/attention_tiled.c"
typedef struct {
    size_t rows, query_width, kv_width, stat_rows;
} Layout;

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

/* Used with at most five inputs and seven float outputs. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                            const DlMatrix *out, size_t no)
{
    size_t ie[5] = {0}, oe[7] = {0};
    if (ni > 5 || no > 7) return DL_ARGUMENT;
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

`check_views` compares bounding spans only when the canonical allocation bases match. This preserves the existing view contract: do not rebase a pointer into the middle of an allocation and pretend it is an independent owner. Padding is conservatively included in a bounding span when checking destructive overlap. Read-only inputs can share storage; outputs cannot overlap any input or each other.

The tiny `smaller` helper makes edge lengths explicit without new macros. It is ordinary `size_t` arithmetic, not an unchecked tensor utility. The full sublayer additionally checks positions and frequencies before starting any projection. Geometry, invalid metadata, and nonfinite input failures therefore leave destinations and statistics untouched.

### Match the reference's score boundary, not every rounding operation

Both implementations accumulate each query/key dot product in double and multiply by $1/\sqrt d$. Both then check that the score fits a finite float and round it once to float. The tile stores that rounded value in a double cell. This apparent extra conversion is intentional: it isolates the normalization/storage change from a different definition of the score input.

The reference next rounds **each normalized probability** to float before its weighted-value reduction. The online implementation has no such probability store. It keeps its sums in double and rounds the completed context to float. Consequently bitwise equality with materialized attention is not the contract. With cancellation among value coordinates, even tiny probability rounding can matter; the tests use absolute-plus-relative tolerances and an independent oracle with matching score boundaries.

A double score outside the float range is still rejected even though the online tile could hold it. That preserves the established numerical input domain. Changing that domain would be a separate feature, not a side effect of removing an array.

## Merge every tile into the running state

The next complete function is the computational kernel. Its nested loops are batch, query head, query tile, key tile, and the logical cells of that tile. Heads are processed serially, so a single tile allocation serves all heads and batch items.

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

At entry, pointers partition the checked double owner. `tile` follows the three statistic planes; `u` follows the entire physical tile, not just the active tail. `maximum` follows all $Rd$ numerator cells, and `denominator` follows all $R$ maxima. These offsets are products already checked by the constructor.

For each query tile, all active numerators begin at zero and all active denominators begin at zero. Only allowed score cells execute a query/key dot product. Masked tile cells receive `-INFINITY`, but the merge loop never includes them. `allowed` is computed only after the row-level empty-tile branch.

Notice the order in the merge: find the new maximum; compute `alpha`; multiply **both** the previous denominator and every previous numerator by it; then add all new terms. The new terms use `next_max`, not the old maximum or the tile maximum alone. A variable holding the old maximum is updated only after those operations.

Finally, the code stores the statistics and divides each numerator by its denominator. Division happens once per completed output coordinate, not once per tile. The increments use the actual `nk` and `nq` tail sizes. Since each is bounded by the remaining length, advancing cannot overflow even when the requested tile dimensions are larger than the sequence.

### Why save more than log-sum-exp?

For ordinary magnitudes, the row log-normalizer

$$
\lambda=m+\log\ell
$$

permits reconstructing a probability as $p_s=\exp(a_s-\lambda)$. That looks like one scalar per row. But the addition can lose the entire $\log\ell$ term when $m$ is enormous. Double precision does not remove that possibility.

For example, let three scores all be approximately $7.07\times10^{37}$. Their correct scaled sum is three and every probability is $1/3$. At that magnitude, adding $\log3$ to a double maximum does not change the represented value. A saved `lse` then equals `maximum`; reconstructing a tied maximum with `exp(score-lse)` incorrectly gives one.

Retaining $m$ and $\ell$ avoids this particular cancellation:

$$
p_s=\frac{\exp(a_s-m)}{\ell}.
$$

The test constructs this case with finite float query/key coordinates of about $10^{19}$. It checks that the weighted average and split statistics remain usable, while explicitly showing the failure of the single-scalar reconstruction. We still save log-sum-exp for inspection and ordinary comparisons. It is not the only numerical state available to the next chapter.

This costs three doubles per query-head row rather than one. That is a deliberate linear-storage tradeoff. It does not justify claiming that finite precision now has no limitations, or that arbitrary out-of-range scores are accepted.

### Failure is not a transaction over output rows

With query tiles of one row, query 0 can finish and be stored before query 1 encounters a score overflow. Its output and statistics are already changed. The public result is nevertheless a failure: discard the entire result and all row statistics. Do not use the valid-looking earlier row as evidence of a successful call.

A new full call overwrites its active state before reading it, so retrying after fixing the input is safe. This differs from the persistent-cache contract in Chapter 17: here there is no committed prefix length to preserve. The tests distinguish preflight failures, which leave cells untouched, from arithmetic failures, which may leave partial results.

## Wrap the core without hidden preprocessing

The public core first checks configuration, workspace, views, shapes, and all logical input values. The complete sublayer then wraps the same internal `attend` function with the existing projection and rotary routines. Its projections write finite float outputs or return an error; successful RoPE calls do the same. It can therefore call the internal kernel without repeating the public core's input scan.

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

#undef TRY
```

The full sublayer accepts an external hidden width independent of $Q_w$; its output weight has shape $[D,Q_w]$. Decoder configurations use $Q_w=D$, but the standalone batched test intentionally uses $D=6$ and $Q_w=8$. That test would expose a wrapper that accidentally used hidden width as attention width.

All six float cache views and the final output are disjoint writes. A late output-projection failure invalidates the context and saved statistics too, even if they were numerically sound before the failure. A successful caller must preserve any state needed for later use; an inference workspace will intentionally overwrite this layer's state when processing the next layer.

The trainer still calls `dl_attention_forward` and `dl_attention_backward` together. This chapter does not substitute a probability-free forward into a backward function that expects probability matrices. [Part 2](https://25d-228.github.io/gpt-c/chapter-18-part-2.html) now gives the new strategy a complete, runnable inference path; the following chapter will implement and integrate its matching backward pass.
