# Chapter 10 — Causal grouped-query attention, forward

A feed-forward layer changes the features of one token without consulting another token. Attention does something different: it uses one token's projected features to choose a weighted combination of value vectors from other permitted positions. We will now implement that movement of information, including the projections before and after it. There is no attention-library call inside the implementation.

The central questions are concrete. Which key belongs to this query head? Which sequence owns that key? Which positions are permitted? Over which entries does the denominator sum? Where do the head results go in the output row? Getting any one of these wrong can produce finite, plausible-looking numbers.

**Prerequisites:** the complete [Chapter 9 workspace](https://25d-228.github.io/gpt-c/media/chapter-09/reference/Makefile), including its headers, seven source modules, tests, and comparison application. Chapters 4–6 supply checked matrix views and both matrix-product backends; Chapter 5 supplies the stable-normalization reasoning; Chapter 9 supplies split-half rotary positions. Chapters 7–8 explain how sublayers keep borrowed saved state. No model loader, tokenizer, training loop, or future chapter is required to run this chapter.

**Implementation estimate:** **4.00 hours** across both parts: 2.75 for projections, masking, and layout plumbing, and 1.25 for traces and tests. Reading is separate. This part derives and implements the entire forward sublayer. [Part 2 — Checking causal attention and shared heads](https://25d-228.github.io/gpt-c/chapter-10-part-2.html) supplies the complete executable checks, replacement Makefile, and diagnostic variations. Together they are one chapter.

> **Learning objectives**
> Follow a query through projection, rotation, scoring, causal normalization, value accumulation, and output projection; implement contiguous groups of shared key/value heads without repeating their storage; keep batch boundaries separate from position metadata; and preserve an inspectable forward tape for differentiation.

## First follow two tokens through one head

A **query** asks which permitted positions should contribute. A **key** supplies the features against which that query is scored. A **value** supplies the features that are actually averaged. These are roles in a calculation, not three different token sequences. In self-attention all three are learned projections of the same input rows.

Begin after projection and rotation, with two coordinates per head. At the second token, let the query be $q=(\sqrt{2},0)$. Let the two keys be $k_0=(0,\log 2)$ and $k_1=(\log 2,0)$, and the values be $v_0=(1,0)$ and $v_1=(0,2)$. Throughout this chapter, $\log$ means the natural logarithm.

The head width is $d=2$. Compute a scaled dot product for each allowed key:

$$
z_0=\frac{q\cdot k_0}{\sqrt{d}}=0,
\qquad
z_1=\frac{q\cdot k_1}{\sqrt{d}}=\log 2.
$$

Exponentiating gives unnormalized weights $1$ and $2$. Dividing by their sum gives probabilities $p_0=1/3$ and $p_1=2/3$. The head output is therefore

$$
c=p_0v_0+p_1v_1
 =\left(\frac13,\frac43\right).
$$

The keys decide the weights; the values supply the coordinates of the result. Replacing a key changes the probabilities. Replacing a value does not change these probabilities in this isolated calculation, but changes the weighted sum. In the complete layer, both are functions of the input, so changing an input row can change both.

At the first token, only key zero is allowed. Its probability is exactly one in our implementation, irrespective of its finite score. Its head result is $v_0$. It must not average $v_0$ and $v_1$ merely because both are present in memory.

[Scaled dot-product attention][attention] defines the score scale, row normalization, and value combination. We will implement each step with explicit indices instead of treating the compact matrix expression as executable documentation.

## Define the axes before flattening them

Use the following symbols locally in this chapter. In particular, uppercase $D$ is the input/model width; lowercase $d$ is the width of **one head**. The roadmap writes those as $d$ and $d_h$, respectively.

| Symbol | Meaning |
|---|---|
| $B$ | Number of independent sequences in this call |
| $T$ | Tokens per sequence; every sequence has this fixed length |
| $D$ | Coordinates in each input and final output row |
| $H_q$ | Number of query heads |
| $H_{kv}$ | Number of stored key heads and stored value heads |
| $d$ | Coordinates in one query, key, or value head; positive and even |
| $M=BT$ | Flattened token-row count |
| $Q_w=H_qd$, $K_w=H_{kv}d$ | Query and key/value projection widths |

The input is logically $X\in\mathbb R^{B\times T\times D}$. The matrix kernels see $[M,D]$. Sequence $b$ and token $t$ occupy row

$$
r(b,t)=bT+t.
$$

This flattening does not authorize attention between different sequences. Projection acts independently on every row, so flattening is harmless there. Attention must recover $b$ and $t$ and keep key rows inside the same sequence.

Weights follow the orientation established in Chapter 4: one output feature per weight **row**. Let

$$
W_q\in\mathbb R^{Q_w\times D},\quad
W_k,W_v\in\mathbb R^{K_w\times D},\quad
W_o\in\mathbb R^{D\times Q_w}.
$$

The three first projections are

$$
Q^0=XW_q^{\mathsf T},\qquad
K^0=XW_k^{\mathsf T},\qquad
V=XW_v^{\mathsf T}.
$$

For example, the query coordinate for head $h$, coordinate $j$, token row $r$ is

$$
Q^0_{r,hd+j}=\sum_{i=0}^{D-1}X_{r,i}(W_q)_{hd+j,i}.
$$

Here $i$ ranges over input features; $j$ ranges over features **within a head**. Splitting into heads changes the interpretation of columns, not their stored order. There is no separate transpose or copy to create a head tensor.

The model configurations use $D=Q_w$. The operator also accepts a different positive $D$, which lets a small non-square fixture expose orientation mistakes. This does not change the eventual model's configuration. The key and value widths are smaller than the query width whenever heads share them.

## A group shares keys and values, not queries or probabilities

Require $H_q$ to be divisible by $H_{kv}$, with both positive. Define the number of query heads per group as

$$
g=H_q/H_{kv}.
$$

Query head $h$ reads key/value head

$$
k(h)=\left\lfloor h/g\right\rfloor.
$$

Integer division of nonnegative `size_t` values implements this directly:

```c
const size_t group = config.query_heads / config.kv_heads;
const size_t key_head = query_head / group;
```

For $H_q=6,H_{kv}=2$, the mapping is:

| Query head | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Key/value head | 0 | 0 | 0 | 1 | 1 | 1 |

`query_head % kv_heads` would instead produce $0,1,0,1,0,1$. Both expressions yield valid indices. Only the first implements our contiguous grouping convention. The mistake can survive memory checking and probability-sum checks.

[Grouped-query attention][gqa] shares a key/value pair of heads among each group of queries. Its endpoints are one shared key/value head, often called multi-query attention, and one key/value head per query head, ordinary multi-head attention. The [documented attention reference formulation][sdpa] expresses contiguous repetition of key/value heads; our index mapping selects the same effective vectors without making repeated arrays.

Each query head still has its own projected query. Even within one group, different queries generally produce different scores, different probability rows, and different weighted value combinations. The saved probability shape contains $H_q$, **not** $H_{kv}$.

For the selected [SmolLM2-135M configuration][config], $D=576,H_q=9,H_{kv}=3,d=64$: three query heads share one key/value head. The projection widths are 576 and 192. Its attention has neither biases nor dropout. Our laboratory's byte-model configuration similarly uses eight query heads and two key/value heads, giving groups of four.

## Rotate the queries and keys, then score allowed pairs

For flattened token row $r$, `positions[r]` supplies its absolute rotary position. Rotate each head of $Q^0$ and $K^0$ using the split-half rule from Chapter 9, producing $Q$ and $K$. Both calls use the same head width and frequency vector, but infer different numbers of heads from their matrix widths. Values are **not** rotated.

For batch $b$, query head $h$, query time $t$, and key time $s$, the finite score is

$$
z_{b,h,t,s}
 =\frac{1}{\sqrt d}\sum_{j=0}^{d-1}
 Q_{bT+t,hd+j}\,K_{bT+s,k(h)d+j},
\qquad 0\leq s\leq t.
$$

The scale uses the number of terms in the dot product, $d$, not $D$, $T$, $H_q$, or $K_w$. A useful explanation is to temporarily assume independent query/key coordinates with zero means and unit variances. Each product then has variance one; summing $d$ independent products gives variance $d$. Dividing the sum by $\sqrt d$ gives variance one. This is an illustrative model of scale, not a claim that learned coordinates remain independent or unit-variance. The original [attention paper, Section 3.2.1][attention], motivates this scaling to avoid increasingly extreme softmax inputs as head width grows.

**Causality is a rule about sequence order:** key time $s$ is permitted when $s\leq t$. The diagonal is included. A training input row at time $t$ contains token $t$ and is used to predict the next token, so including that row is not access to its future target.

Absolute rotary positions do a different job. They change angles, not mask membership. Common full-sequence positions are $0,1,\ldots,T-1$ separately for each batch item. Nonzero offsets and repeated positions are accepted by this low-level operator, but they never merge sequences or make later rows visible. In particular, three rows all labeled with position seven still have a triangular causal mask.

This is full-sequence attention for fixed-length training or prefill. There is no padding mask, packed-document boundary mask, cross-attention, or past key/value cache in the interface. Calling it with one row sees only that row, not previously processed rows hidden somewhere in the program.

## Normalize only the permitted scores

For a fixed $(b,h,t)$, the probability of key $s$ is

$$
p_s=\frac{\exp(z_s)}{\sum_{u=0}^{t}\exp(z_u)},
\qquad 0\leq s\leq t.
$$

The key-time sum is the denominator's only axis. It does not sum across query times, heads, features, or batches. Factor out the exponential of the largest permitted score, $m=\max_{0\leq u\leq t}z_u$:

$$
p_s=\frac{\exp(z_s-m)}{\sum_{u=0}^{t}\exp(z_u-m)}.
$$

Every exponent argument is nonpositive; at least one is zero. In exact arithmetic the denominator lies between one and $t+1$. Small terms can underflow to zero without making the denominator zero. For scores $(1000,1001)$, direct exponentiation is unsuitable even in double precision, but the shifted arguments are $(-1,0)$, giving approximately $(0.268941,0.731059)$.

The subtraction also needs an appropriate type. Two finite float scores near opposite ends of the range can have a difference outside the float range. The implementation subtracts **after** widening each score to double. Casting an already-overflowed float difference is too late.

We store a full square score plane per head for inspection. Future entries contain `-INFINITY`; their corresponding probabilities contain exactly zero. The mask branch writes these sentinels without calculating a future dot product. Maximum, denominator, and weighted-value loops visit only $s\leq t$. They do not treat a masked sentinel as an ordinary finite input.

Masking after normalization is not equivalent. Scores $(0,0)$ normalize to $(1/2,1/2)$. If only the first key is permitted and the second probability is then set to zero, the row becomes $(1/2,0)$, not $(1,0)$. The forbidden key has already diluted the denominator. The output can depend on its score even though its final probability is zero.

Our API cannot represent an arbitrary all-masked row. It rejects $T=0$ with `DL_SHAPE`, and every valid row includes its diagonal. The denominator guard still reports an impossible/non-finite normalization as `DL_NUMERIC`. We do not silently choose a uniform distribution over an empty set. A future arbitrary-mask extension would need an explicit empty-row policy rather than inheriting one accidentally from infinity arithmetic.

## Average values and place each head directly in the concatenated row

For coordinate $j$ of query head $h$, accumulate

$$
C_{bT+t,hd+j}
 =\sum_{s=0}^{t}p_{b,h,t,s}\,V_{bT+s,k(h)d+j}.
$$

$C$ has shape $[M,Q_w]$. The word **concatenation** means that head zero occupies columns $0$ through $d-1$, head one occupies the next $d$ columns, and so on. We write those columns directly. There is no later concatenation allocation or copy.

The final projection mixes the head features:

$$
Y=CW_o^{\mathsf T},\qquad Y\in\mathbb R^{M\times D}.
$$

A final output coordinate may combine several heads. It is not generally a convex combination of original input coordinates: the weighted sum is over projected values, and the output matrix can contain negative or greater-than-one coefficients. No residual addition or normalization is hidden here. The decoder will compose those separately.

## Trace actual padded addresses

The projected matrices use token-major rows. Scores and probabilities instead use logical shape $[B,H_q,T,T]$, represented as a matrix $[BH_qT,T]$. Its row index is

$$
a(b,h,t)=(bH_q+h)T+t.
$$

These two flattenings are intentionally different. A probability row belongs to one query **head and time**; a projected row contains all heads for one token.

Take $B=2,T=3,H_q=4,H_{kv}=2,d=2$. Consider query batch $b=1$, head $h=3$, time $t=2$, reading key time $s=1$, coordinate $j=1$. Then $r=5$, key row is 4, shared head is 1, and probability row is 23.

Suppose each view starts at element offset 1 and has two padding floats after its logical row. The following are element offsets from each view's **own** canonical allocation base:

| Access | Calculation | Offset |
|---|---|---:|
| Query coordinate | $1+5\cdot10+(3\cdot2+1)$ | 58 |
| Key/value coordinate | $1+4\cdot6+(1\cdot2+1)$ | 28 |
| Score/probability for key 1 | $1+23\cdot5+1$ | 117 |
| Concatenated context coordinate | $1+5\cdot10+(3\cdot2+1)$ | 58 |

The two offsets of 58 refer to different allocations. Equal offsets do not imply aliasing. Conversely, rebasing a pointer into an allocation would conceal shared ownership from the existing alias checker. Keep canonical base/count metadata, including for slices.

Using logical width instead of physical `ld` would read a padding float as data. Using flattened query row five as the causal time would permit keys from another sequence or beyond the end of this one. Neither error is repaired by a stable softmax.

## Declare borrowed inputs and a complete forward tape

Create `include/dl/attention.h`. The configuration stores only geometry; input width comes from `x.cols`. The weights and cache structs contain borrowed matrix views, not owners.

```c title="include/dl/attention.h"
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
```

`q0` and `k0` preserve the projection results before rotation. `q` and `k` preserve their rotated versions; `v` preserves the unrotated value projection. `scores` and `probs` make every normalization row inspectable. `context` saves the head outputs before the final projection. All nine destinations, including `y`, are independent writable spans.

The word `cache` here means **saved state from one forward call**, often called a training tape. It is not the persistent key/value cache used during incremental generation. Running forward again into the same views replaces the previous tape. The function does not retain any pointer after it returns; the caller is responsible for keeping the owners alive.

| State kept unchanged until its backward consumer | Reason |
|---|---|
| $X$ and projection weights | Projection derivatives need the original inputs/weights |
| Rotated $Q,K$, unrotated $V$, and probabilities | The attention derivative needs these effective vectors and weights |
| Concatenated context | The output-projection weight derivative uses it |
| Positions and frequencies | Rotation backward must apply the matching transpose |
| $Q^0,K^0$, scores | Useful inspection state; not all are mathematically necessary for the minimal backward tape |

We deliberately keep more than the smallest possible tape. Removing arrays before understanding their consumers would obscure rather than teach the memory tradeoff. With dense FP32 arrays, the eight cache fields contain

$$
3M(Q_w+K_w)+2BH_qT^2
$$

floats: three query-width arrays, three key/value-width arrays, and two score-sized arrays. This excludes inputs, final outputs, weights, gradients, metadata, and padding. At the byte model's initial $B=2,T=128,H_q=8,H_{kv}=2,d=32$, the two score-sized arrays alone occupy 2 MiB; the linear cache arrays add 0.9375 MiB per layer. These are calculated payloads, not measured process memory. Sharing key/value heads reduces their width but does **not** remove the $H_qT^2$ probability storage.

Every call is an overwrite operation. Old destination values may be uninitialized or NaN; the implementation must never read them. Input logical cells must all be finite. Structural validation and non-finite-input checks happen before the first write. Later arithmetic can still overflow, so `DL_NUMERIC` may leave several cache arrays partially or fully written. Discard the **whole** failed call's outputs. A plausible earlier context row is not a valid partial result to continue training with.

There is an important distinction between data dependencies and validation. A finite future-row perturbation cannot affect an earlier result of a successful call. A future NaN, however, causes our whole-input preflight to reject the call. Causality does not promise that malformed future storage will be ignored by error checking.

## Validate sizes before using any index product

Create `src/attention_ref.c`. All filename-labeled blocks for this path, in order, form the complete file. The private `Layout` stores checked derived dimensions. It does not allocate anything or duplicate the caller's owner metadata.

```c title="src/attention_ref.c"
#include "dl/attention.h"

#include <float.h>
#include <math.h>

/* All derived products are checked before an index uses them. */
typedef struct {
    size_t rows, query_width, kv_width, score_rows;
} Layout;

static DlStatus layout(DlAttentionConfig c, Layout *out)
{
    if (c.batch == 0 || c.time == 0 || c.query_heads == 0 || c.kv_heads == 0 ||
        c.head_dim == 0 || c.head_dim % 2 != 0 ||
        c.query_heads % c.kv_heads != 0) return DL_SHAPE;
    Layout z = {0};
    DlStatus s = dl_size_mul(c.batch, c.time, &z.rows);
    if (s != DL_OK) return s;
    s = dl_size_mul(c.query_heads, c.head_dim, &z.query_width);
    if (s != DL_OK) return s;
    s = dl_size_mul(c.kv_heads, c.head_dim, &z.kv_width);
    if (s != DL_OK) return s;
    s = dl_size_mul(z.rows, c.query_heads, &z.score_rows);
    if (s != DL_OK) return s;
    const size_t rows[3] = {z.rows, z.rows, z.score_rows};
    const size_t cols[3] = {z.query_width, z.kv_width, c.time};
    for (size_t i = 0; i < 3; ++i) {
        size_t count = 0, bytes = 0;
        s = dl_size_mul(rows[i], cols[i], &count);
        if (s != DL_OK) return s;
        s = dl_size_mul(count, sizeof(float), &bytes);
        if (s != DL_OK) return s;
    }
    *out = z;
    return DL_OK;
}
```

The short-circuit `||` expressions matter: `kv_heads == 0` is tested before a remainder operation could divide by zero. Positive even `head_dim` is required by full-width RoPE. Divisibility also ensures that `query_heads / kv_heads` is at least one.

Checking `batch*time` alone is insufficient. Query width, key/value width, score rows, element counts, and byte counts can overflow independently. A score row index uses $(bH_q+h)T+t$, whose maximum lies within the checked $BH_qT$ count. The additional $T$ factor is needed for the square score storage. View validation will separately establish the actual padded spans.

The next helpers apply the same canonical-base/bounding-span rule as Chapters 4 and 8. Read-only inputs may overlap each other. Every destination must be disjoint from all five inputs and from the other eight destinations, including gaps inside their bounding spans.

```c title="src/attention_ref.c"
/* This forward call has exactly five inputs and nine destinations. */
static DlStatus check_views(const DlConstMatrix in[5], const DlMatrix out[9])
{
    size_t ie[5] = {0}, oe[9] = {0};
    for (size_t i = 0; i < 5; ++i) {
        const DlStatus s = dl_matrix_end(in[i], &ie[i]);
        if (s != DL_OK) return s;
    }
    for (size_t o = 0; o < 9; ++o) {
        const DlConstMatrix v = dl_matrix_read(out[o]);
        const DlStatus s = dl_matrix_end(v, &oe[o]);
        if (s != DL_OK) return s;
        for (size_t i = 0; i < 5; ++i) {
            if (v.base == in[i].base && v.offset < ie[i] &&
                in[i].offset < oe[o]) return DL_ALIAS;
        }
        for (size_t p = 0; p < o; ++p) {
            if (v.base == out[p].base && v.offset < oe[p] &&
                out[p].offset < oe[o]) return DL_ALIAS;
        }
    }
    return DL_OK;
}

static int shape(DlConstMatrix v, size_t rows, size_t cols)
{
    return v.rows == rows && v.cols == cols;
}

static int finite_cells(DlConstMatrix v)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (!isfinite(v.base[v.offset + i * v.ld + j])) return 0;
        }
    }
    return 1;
}
```

The fixed array sizes are private implementation facts: this forward has five inputs and nine destinations. They are not limits on model dimensions. `shape` compares logical dimensions only; `dl_matrix_end` already checks offsets, strides, capacity arithmetic, and positive extents. `finite_cells` walks logical cells, not padding. Padding may contain sentinel values and is neither computation input nor temporary workspace.

The metadata arrays have different element types from the matrices. As in Chapter 9, their alignment, truthful capacities, lifetimes, and disjoint storage remain caller preconditions. C cannot establish them from a pointer and a claimed length. We can still reject mismatched counts, unsupported positions, and invalid frequency values before invoking a projection.

```c title="src/attention_ref.c"
static DlStatus metadata(Layout z, DlAttentionConfig c,
                          const size_t *positions, size_t position_count,
                          const double *freq, size_t frequency_count)
{
    if (positions == NULL || freq == NULL) return DL_ARGUMENT;
    if (position_count != z.rows || frequency_count != c.head_dim / 2)
        return DL_SHAPE;
    size_t bytes = 0;
    DlStatus s = dl_size_mul(position_count, sizeof(*positions), &bytes);
    if (s != DL_OK) return s;
    s = dl_size_mul(frequency_count, sizeof(*freq), &bytes);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < position_count; ++i) {
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    }
    for (size_t j = 0; j < frequency_count; ++j) {
        if (!isfinite(freq[j]) || freq[j] <= 0.0 || freq[j] > 1.0)
            return DL_ARGUMENT;
    }
    return DL_OK;
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}
```

`store_float` rejects a non-finite or out-of-range arithmetic result before narrowing it. This contract also applies to **allowed scores**. A query/key dot product that fits in double but cannot be stored as a finite float returns `DL_NUMERIC`; it is not quietly turned into an infinity and passed to softmax. Masked `-INFINITY` is a separately written sentinel, not an accepted arithmetic score.

The helper does not promise rollback. If the next score fails after the previous query row was completed, those earlier writes remain. The wrapper must propagate the failure without applying the output projection to incomplete context.

## Implement one causal probability row and its value reduction

Append the complete private `attend` function. The enclosing public call has already validated geometry, aliases, metadata, and inputs, and has produced finite projected/rotated arrays. This private function is not a second public API with looser contracts.

```c title="src/attention_ref.c"
/* Private: views, config, and all projected logical cells are already valid.
 * Each row stores only allowed finite scores; the rest are explicit sentinels. */
static DlStatus attend(DlAttentionConfig c, DlAttentionCache a)
{
    const size_t group = c.query_heads / c.kv_heads;
    const double scale = 1.0 / sqrt((double)c.head_dim);
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t t = 0; t < c.time; ++t) {
                const size_t row = b * c.time + t;
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                const size_t qa = a.q.offset + row * a.q.ld + h * c.head_dim;
                const size_t sa = a.scores.offset + sr * a.scores.ld;
                const size_t pa = a.probs.offset + sr * a.probs.ld;
                double maximum = -INFINITY;
                for (size_t s = 0; s < c.time; ++s) {
                    if (s > t) {
                        a.scores.base[sa + s] = -INFINITY;
                        a.probs.base[pa + s] = 0.0f;
                        continue;
                    }
                    const size_t kr = b * c.time + s;
                    const size_t ka = a.k.offset + kr * a.k.ld + kh * c.head_dim;
                    double dot = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        dot += (double)a.q.base[qa + j] * (double)a.k.base[ka + j];
                    }
                    const DlStatus status = store_float(&a.scores.base[sa + s],
                                                         dot * scale);
                    if (status != DL_OK) return status;
                    maximum = fmax(maximum, (double)a.scores.base[sa + s]);
                }
                double denominator = 0.0;
                for (size_t s = 0; s <= t; ++s) {
                    denominator += exp((double)a.scores.base[sa + s] - maximum);
                }
                /* Positive T and the diagonal forbid an empty allowed set. */
                if (!isfinite(denominator) || denominator <= 0.0) return DL_NUMERIC;
                for (size_t s = 0; s <= t; ++s) {
                    const double p = exp((double)a.scores.base[sa + s] - maximum)
                                   / denominator;
                    const DlStatus status = store_float(&a.probs.base[pa + s], p);
                    if (status != DL_OK) return status;
                }
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double sum = 0.0;
                    for (size_t s = 0; s <= t; ++s) {
                        const size_t vr = b * c.time + s;
                        const size_t va = a.v.offset + vr * a.v.ld + kh * c.head_dim + j;
                        sum += (double)a.probs.base[pa + s] * (double)a.v.base[va];
                    }
                    const size_t ca = a.context.offset + row * a.context.ld
                                    + h * c.head_dim + j;
                    const DlStatus status = store_float(&a.context.base[ca], sum);
                    if (status != DL_OK) return status;
                }
            }
        }
    }
    return DL_OK;
}
```

The loop order is batch, query head, query time. At the start of each row, `maximum` is reset to negative infinity. The diagonal guarantees at least one finite score replaces it. The branch on `s > t` writes both masked sentinels and skips the dot product entirely. Notice that both `kr` and `vr` include `b * c.time`; forgetting either one can read another batch item's valid memory without any out-of-bounds access.

After storing each allowed score as float, we widen that **stored score** to update the maximum. Softmax therefore normalizes exactly the scores visible in `cache.scores`, not an unrecorded higher-precision alternative. The denominator and numerator exponentials use double. Each probability is stored as float, and the value loop reads those saved float probabilities. The finite-precision execution is thus explicit:

```text
float Q/K -> double dot and scale -> float score
float score -> double max subtraction, exp and normalization -> float probability
float probability/V -> double weighted sum -> float context
float context/Wo -> double matrix reduction -> float Y
```

In real arithmetic, allowed probabilities sum to one. Separately rounded float probabilities need not sum to exactly one, so tests use a small tolerance. The one-key case is stronger: its shifted score is exactly zero, its denominator is one, and its probability is exactly one. Its context equals the selected value row exactly for finite float values.

The denominator loop deliberately recomputes exponentials in the subsequent probability loop. That costs work but avoids another temporary array and never stores unnormalized exponentials in the saved probability tape. This is a reference implementation whose numerical boundaries should be easy to inspect, not a claim of an optimal softmax kernel.

Concatenation happens at `h * c.head_dim + j`. The value lookup instead uses `kh * c.head_dim + j`. Confusing these offsets either duplicates the wrong head or exceeds the narrower key/value width. No projection or normalization is averaged across heads.

## Connect projections, rotations, and the complete sublayer

Append the public function. It performs a whole-call preflight before any projection writes, then calls the already-taught matrix and rotary kernels in data-dependency order.

```c title="src/attention_ref.c"
DlStatus dl_attention_forward(DlBackend backend, DlAttentionConfig c,
                              DlConstMatrix x, DlAttentionWeights w,
                              const size_t *positions, size_t position_count,
                              const double *inv_freq, size_t frequency_count,
                              DlAttentionCache a, DlMatrix y)
{
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED)
        return DL_ARGUMENT;
    Layout z = {0};
    DlStatus s = layout(c, &z);
    if (s != DL_OK) return s;
    const DlConstMatrix in[5] = {x, w.query, w.key, w.value, w.output};
    const DlMatrix out[9] = {a.q0, a.k0, a.v, a.q, a.k,
                             a.scores, a.probs, a.context, y};
    s = check_views(in, out);
    if (s != DL_OK) return s;
    const size_t width = x.cols;
    if (x.rows != z.rows || !shape(w.query, z.query_width, width) ||
        !shape(w.key, z.kv_width, width) || !shape(w.value, z.kv_width, width) ||
        !shape(w.output, width, z.query_width) ||
        !shape(dl_matrix_read(a.q0), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k0), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.v), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.q), z.rows, z.query_width) ||
        !shape(dl_matrix_read(a.k), z.rows, z.kv_width) ||
        !shape(dl_matrix_read(a.scores), z.score_rows, c.time) ||
        !shape(dl_matrix_read(a.probs), z.score_rows, c.time) ||
        !shape(dl_matrix_read(a.context), z.rows, z.query_width) ||
        !shape(dl_matrix_read(y), z.rows, width)) return DL_SHAPE;
    s = metadata(z, c, positions, position_count, inv_freq, frequency_count);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < 5; ++i) {
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    s = dl_matmul(backend, x, DL_NOTRANS, w.query, DL_TRANS, a.q0, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, x, DL_NOTRANS, w.key, DL_TRANS, a.k0, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, x, DL_NOTRANS, w.value, DL_TRANS, a.v, DL_SET);
    if (s != DL_OK) return s;
    s = dl_rope_forward(dl_matrix_read(a.q0), c.head_dim, positions,
                        position_count, inv_freq, frequency_count, a.q);
    if (s != DL_OK) return s;
    s = dl_rope_forward(dl_matrix_read(a.k0), c.head_dim, positions,
                        position_count, inv_freq, frequency_count, a.k);
    if (s != DL_OK) return s;
    s = attend(c, a);
    if (s != DL_OK) return s;
    return dl_matmul(backend, dl_matrix_read(a.context), DL_NOTRANS,
                      w.output, DL_TRANS, y, DL_SET);
}
```

Each matrix call uses `DL_NOTRANS` for the activations and `DL_TRANS` for weights stored as output-by-input. The last projection has the same orientation rule even though its weight shape differs from the three first projections. All matrix destinations use `DL_SET`.

The `backend` parameter selects reference or blocked **projection matrix products**. The query/key dot products and weighted-value loops inside `attend` remain scalar and identical for both selections. The filename `attention_ref.c` refers to the materialized reference attention algorithm, not a restriction to the reference projection backend. There is no silent fallback.

The full preflight is intentionally more conservative than the requirements of any one nested call. For example, it checks `w.output` for a NaN before computing `q0`, even though the output matrix is not consumed until the last step. Similarly, it checks that scores and probabilities do not overlap, although neither is a matrix-product output during projection. Without the whole-call checks, a later structural error could arrive only after earlier buffers had been overwritten.

Finite inputs do not guarantee finite arithmetic. A projection may overflow; a rotated pair may exceed float range; an allowed score or a final output may fail its store. These failures are propagated immediately. No normalization, clipping, or replacement by zero is used to disguise them.

The next part constructs real owners for every borrowed view, follows a three-token/two-query-head example all the way to $Y$, and tests independent sequence boundaries. The complete forward code above already exists; those tests do not rely on an unwritten backward routine.

## Technical references

Vaswani et al., [*Attention Is All You Need*, version 7, Sections 3.2.1–3.2.3][attention]: scaled dot products, independently projected heads, concatenation/output projection, and decoder masking. The C weight orientation and every buffer convention are defined here rather than inferred from the paper's matrix notation.

Ainslie et al., [*GQA*, version 3, Section 2.2][gqa]: grouped sharing and the multi-query/multi-head endpoints. The grouping convention is implemented by integer division, not physical repetition.

Su et al., [*RoFormer*, version 5, Section 3.2][rope]: rotary query/key transformations. Chapter 9 supplies the full split-half implementation reused here.

PyTorch, [documented scaled-dot-product attention formulation][sdpa]: an independent reference for contiguous key/value repetition and a lower-triangular self-attention mask. It is a technical reference, not a required dependency or a replacement for the C loops.

HuggingFaceTB, [SmolLM2-135M configuration][config]: the selected inference configuration's head counts, dimensions, bias/dropout settings, and rotary metadata. Asset loading later validates the downloaded revision; this chapter downloads no weights.

[attention]: https://arxiv.org/html/1706.03762v7
[gqa]: https://arxiv.org/html/2305.13245v3
[rope]: https://arxiv.org/html/2104.09864v5
[sdpa]: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
[config]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json

