# Chapter 11 — Attention backward and shared-head accumulation

One key/value head can be read by several query heads. Forward makes that sharing look inexpensive: each reader uses the same address. Backward must account for every reader. A result that keeps only the last head's contribution can have the right shape, remain finite, and produce a decreasing training loss. None of those observations makes it the derivative of the forward program.

We will reverse the complete sublayer from Chapter 10, including the output projection, weighted values, row normalization, scaled scores, rotary positions, and the three input projections. The result is a materialized derivative reference: its intermediate gradients remain available for inspection before later chapters remove quadratic storage.

**Prerequisites:** the complete [Chapter 10 source snapshot](https://25d-228.github.io/gpt-c/media/chapter-10/reference/Makefile). Keep its eight headers, eight source modules, tests, comparison application, and Makefile. Chapter 5 supplies projection derivatives and finite differences; Chapter 7 explains residual accumulation; Chapter 9 supplies rotation-transpose backward. No training loop, external tensor framework, or future model code is needed.

**Implementation estimate:** **4.50 hours** across both parts: 3.00 for the backward implementation and integration, and 1.50 for shared-head, mask, and gradient checks. Reading is separate. This part derives and implements the operators. [Part 2 — Checking attention gradients and accumulation](https://25d-228.github.io/gpt-c/chapter-11-part-2.html) supplies the complete test program, Makefile, runnable commands, and diagnostic variations. Together they form one chapter.

> **Learning objectives**
> Derive a softmax vector–Jacobian product without storing its Jacobian; follow a gradient through shared keys and values; distinguish local query derivatives from shared key/value derivatives; preserve earlier input and parameter contributions; and use independent local and end-to-end checks to diagnose errors.

## A gradient has the shape of the quantity it describes

Let $L$ be a scalar objective. For any array $A$, write $\bar A=\partial L/\partial A$ for its gradient. In C, a name such as `g.q` or `DQ` stores $\bar Q$. The bar is not another forward activation, and `dQ` does not mean the head dimension multiplied by $Q$.

We retain Chapter 10's axes: $B$ sequences, $T$ tokens per sequence, $D$ input/output features, $H_q$ query heads, $H_{kv}$ key/value heads, and even head width $d$. Define $M=BT$, $Q_w=H_qd$, $K_w=H_{kv}d$, and sharing-group size $G=H_q/H_{kv}$. All dimensions are positive. Query head $h$ reads key/value head $k(h)=\lfloor h/G\rfloor$.

| Quantity | Stored shape | Gradient shape |
|---|---|---|
| Input $X$ and output $Y$ | $[M,D]$ | $[M,D]$ |
| Unrotated/rotated queries $Q^0,Q$; context $C$ | $[M,Q_w]$ | $[M,Q_w]$ |
| Unrotated/rotated keys $K^0,K$; values $V$ | $[M,K_w]$ | $[M,K_w]$ |
| Scores $S$ and probabilities $P$ | $[BH_qT,T]$ | $[BH_qT,T]$ |
| Query weights $W_q$ | $[Q_w,D]$ | $[Q_w,D]$ |
| Key/value weights $W_k,W_v$ | $[K_w,D]$ | $[K_w,D]$ |
| Output weights $W_o$ | $[D,Q_w]$ | $[D,Q_w]$ |

The gradient of a shared array has the **shared array's shape**, not one copy per consumer. The extra query heads contribute to that shape through addition. GQA's head-sharing definition and its MQA/MHA endpoints are described in [Ainslie et al., Section 2.2][gqa]. The sum in backward follows from differentiating this reuse; it is not an additional averaging convention.

The caller supplies $\bar Y$. An eventual loss may already include a token or batch mean. This operator cannot infer that normalization and must not apply it again. For tests we choose a linear objective $L=\sum_{r,i}Y_{r,i}U_{r,i}$, where $U$ is a fixed array. Then $\bar Y=U$ exactly; this exercises the sublayer without hiding an extra loss derivative.

## Start at the output projection

The last forward operation was $Y=CW_o^{\mathsf T}$. Written as a scalar,

$$
Y_{r,i}=\sum_{a=0}^{Q_w-1}C_{r,a}(W_o)_{i,a}.
$$

For one context entry, sum the incoming gradients of all outputs that used it. For one weight entry, sum the contributions of every token row that used that weight:

$$
\bar C_{r,a}=\sum_{i=0}^{D-1}\bar Y_{r,i}(W_o)_{i,a},
\qquad
(\bar W_o)_{i,a}=\sum_{r=0}^{M-1}\bar Y_{r,i}C_{r,a}.
$$

Therefore $\bar C=\bar YW_o$ and $\bar W_o=\bar Y^{\mathsf T}C$. The weight remains stored as `[output,input]`; backward changes the product flags, not the weight layout. `dl_matmul` uses NN for the first product and TN for the second. The context gradient is new scratch, so it uses `DL_SET`. A parameter gradient may already contain another microbatch's contribution, so its mode belongs to the caller.

The context was a concatenation of head results. Its backward needs no copy or transpose: head $h$ reads columns $hd$ through $hd+d-1$ of $\bar C$. That is the same column interpretation used in forward.

## Reverse one weighted sum

Temporarily fix a batch item $b$, query head $h$, and query time $t$. Suppress those fixed indices. Let $p_s$ be the probability assigned to key time $s$, and let $v_{s,j}$ be coordinate $j$ of its shared value head. Only $0\le s\le t$ is allowed. Forward computed

$$
c_j=\sum_{s=0}^{t}p_s v_{s,j}.
$$

Holding the other inputs fixed, $\partial c_j/\partial p_s=v_{s,j}$ and $\partial c_j/\partial v_{s,j}=p_s$. Applying the incoming gradient $g_j=\bar c_j$ gives

$$
\bar p_s=\sum_{j=0}^{d-1}g_jv_{s,j},
\qquad
\bar v_{s,j}\mathrel{+}=p_sg_j.
$$

The first expression is a dot product and creates one probability derivative for this query row. The second is a **contribution**, not the whole value derivative: later query times and other query heads can read the same value. All those uses must be included before its final derivative is stored.

This separates two often-confused paths. A value changes what is averaged; its derivative uses probabilities. A key changes the probabilities; its derivative must travel through softmax and the score dot product. A correct value derivative does not establish a correct key derivative.

## Differentiate softmax without a square Jacobian

For this one allowed row, write $p_u=e^{z_u}/Z$, where $Z=\sum_{v=0}^{t}e^{z_v}$. Differentiating the numerator and denominator gives

$$
\frac{\partial p_u}{\partial z_s}
 =\frac{\delta_{us}e^{z_u}Z-e^{z_u}e^{z_s}}{Z^2}
 =p_u(\delta_{us}-p_s).
$$

Here $\delta_{us}$ is one when $u=s$, otherwise zero. The whole Jacobian would have $(t+1)^2$ entries, but we only need its product with the incoming vector $\bar p$. Substitute the derivative above into the chain rule:

$$
\bar z_s=\sum_{u=0}^{t}\bar p_u p_u(\delta_{us}-p_s)
        =p_s\left(\bar p_s-\sum_{u=0}^{t}p_u\bar p_u\right).
$$

Define the row scalar $\mu=\sum_u p_u\bar p_u$. Compute $\mu$ once, then visit the row again to store $p_s(\bar p_s-\mu)$. The entire operation is linear in the row length; no explicit Jacobian is allocated. [FlashAttention Appendix B.2 and Algorithm 3][flash] give a primary reference for the materialized backward relationships. Our code below is a scalar C implementation with the course's masks, sharing, and float-storage boundaries, not a GPU kernel.

The subtraction has a useful invariant. In exact arithmetic,

$$
\sum_s\bar z_s=\sum_s p_s\bar p_s-\mu\sum_s p_s=\mu-\mu=0.
$$

Adding the same constant to all allowed scores does not change softmax; the derivative in that direction is zero. Stored float probabilities need not sum to exactly one, so the executable checks a small tolerance, not bitwise zero for general rows.

For a one-key row, $p_0=1$ exactly, $\mu=\bar p_0$, and $\bar z_0=0$. Yet $\bar v_0=g$ can be nonzero. “The attention weights cannot change” is not the same as “attention has no gradient.”

Forward subtracts a maximum for numerical stability. We differentiate the unchanged normalized function, not a new trainable maximum. Differentiating the algebraically equivalent shifted expression makes its common-shift terms cancel. Treating a maximum's derivative as an extra uncancelled path would change the function being differentiated.

Above the causal diagonal, probabilities were explicit zero and the score entries were `-INFINITY` sentinels. They are not free score variables. We store **zero masked score derivatives** and also zero masked `dP` entries as a storage convention. We do not evaluate `0 * -INFINITY`, differentiate the sentinel, or extend the allowed-row normalization into masked entries.

## Reverse scaled scores and include every sharing head

Restore the full indices. Let $\alpha=1/\sqrt d$. For an allowed key,

$$
S_{b,h,t,s}=\alpha\sum_{j=0}^{d-1}Q_{b,t,h,j}K_{b,s,k(h),j}.
$$

One query belongs to one query head and one time, so its derivative sums over that row's allowed keys:

$$
\bar Q_{b,t,h,j}=\alpha\sum_{s=0}^{t}\bar S_{b,h,t,s}K_{b,s,k(h),j}.
$$

A stored key belongs to a key/value head, not a query head. Its derivative must collect every query in the sharing group and every time that is allowed to read this key:

$$
\bar K_{b,s,k,j}
 =\alpha\sum_{h=kG}^{(k+1)G-1}\sum_{t=s}^{T-1}
       \bar S_{b,h,t,s}Q_{b,t,h,j}.
$$

The corresponding value derivative is

$$
\bar V_{b,s,k,j}
 =\sum_{h=kG}^{(k+1)G-1}\sum_{t=s}^{T-1}
       P_{b,h,t,s}\bar C_{b,t,h,j}.
$$

The lower bound $t=s$ in the last two expressions is causality read backward: key $s$ can influence queries at $s$ and later, not earlier. The batch index remains fixed. There is no gradient contribution from another sequence merely because its rows are adjacent in the allocation.

A factor $1/G$ appears nowhere. To see why, imagine a shared scalar $w$ used twice, with total objective $L=f(w)+g(w)$. Its derivative is $f'(w)+g'(w)$, not their mean. Expanding a key/value head into several identical copies and then treating those copies as tied parameters gives the same sum. This differs from **mean-pooling weights while converting a checkpoint**, which is a different operation discussed in the GQA paper.

## Choose a write order that makes sharing visible

A scatter implementation could zero the entire $\bar K$ and $\bar V$ arrays once, then add each query's contribution. Zeroing either array inside the query-head loop would erase earlier heads. Repeatedly adding into float storage would also round after every contribution.

Here we use the equivalent **gather** order. For each destination key/value coordinate, initialize two double accumulators to zero, visit all sharing heads and allowed query times, then store each completed result once as float. The zero occurs outside both consumer loops. This retains the double-reduction convention established by the matrix reference kernel and avoids atomics or hidden heap scratch in this single-threaded implementation.

The query derivative is also gathered: one accumulator per query coordinate sums all allowed keys. Every scratch destination is overwritten, so stale NaNs in old scratch are harmless. This does not make a partially failed result usable; a later overflow can leave some destinations written and others untouched.

## Follow two sharing heads by hand

Use $B=1,T=2,D=2,H_q=2,H_{kv}=1,d=2$. Let $X$ be the two-by-two identity matrix and both position values be zero, so the rotations are identities. Write $r=\sqrt2$ and $a=\log2$. Choose

$$
W_q=\begin{bmatrix}0&r\\0&0\\0&0\\0&r\end{bmatrix},
\quad
W_k=\begin{bmatrix}0&a\\a&0\end{bmatrix},
\quad
W_v=\begin{bmatrix}1&0\\0&2\end{bmatrix},
\quad
W_o=\begin{bmatrix}1&0&0&0\\0&0&0&1\end{bmatrix}.
$$

At time one, the two queries are $(r,0)$ and $(0,r)$. The shared keys are $(0,a)$ and $(a,0)$; the shared values are $(1,0)$ and $(0,2)$. Choose the objective $L=Y_{1,0}+Y_{1,1}$. All entries of $\bar Y$ are zero except the last row, which is $(1,1)$.

The output matrix selects the first coordinate of head zero and second coordinate of head one. Thus their incoming context gradients are $(1,0)$ and $(0,1)$ respectively. Follow each path:

| At time one | Query head zero | Query head one |
|---|---|---|
| Allowed scores | $(0,a)$ | $(a,0)$ |
| Probabilities | $(1/3,2/3)$ | $(2/3,1/3)$ |
| Head context | $(1/3,4/3)$ | $(2/3,2/3)$ |
| Incoming context gradient | $(1,0)$ | $(0,1)$ |
| Probability derivative | $(1,0)$ | $(0,2)$ |
| Row scalar $\mu$ | $1/3$ | $2/3$ |
| Score derivative | $(2/9,-2/9)$ | $(-4/9,4/9)$ |

For the first shared key, head zero contributes $(2/9,0)$ after the score scale cancels its query's $r$. Head one contributes $(0,-4/9)$. Therefore

$$
\bar K_0=(2/9,-4/9),\qquad \bar K_1=(-2/9,4/9).
$$

For values, add probability times incoming context gradient:

$$
\bar V_0=\tfrac13(1,0)+\tfrac23(0,1)=(1/3,2/3),
$$

$$
\bar V_1=\tfrac23(1,0)+\tfrac13(0,1)=(2/3,1/3).
$$

Set $u=a/(9\sqrt2)$. The two query derivatives at time one are $(-2u,2u)$ and $(4u,-4u)$. Both query derivatives at time zero are zero because that output has no incoming gradient. Nevertheless, **the key and value at time zero have nonzero derivatives**: the later output used them.

Since $X$ is the identity, the query/key/value weight gradients are the transposes of the corresponding projected gradients. Both rows of $\bar W_o$ equal the concatenated context $(1/3,4/3,2/3,2/3)$. Summing the three input-projection paths yields

$$
\bar X_0=\left(\frac13-\frac{4a}{9},\;\frac43+\frac{2a}{9}\right),
\qquad
\bar X_1=\left(\frac23+\frac{4a}{9},\;\frac23-\frac{8a}{9}\right).
$$

For example, the first coordinate of $\bar X_0$ is about $0.02526792$. Keeping only the value path produces $1/3$ instead. This is an in-bounds, finite result, but it is not a rounding discrepancy. Part 2 builds these exact matrices and checks intermediate derivatives as well as the final input and all four weight gradients.

## Undo rotations, then merge the three input paths

For one column-vector head, forward used $q=R(p)q^0$ at the supplied position $p$. The chain rule gives $\bar q^0=R(p)^{\mathsf T}\bar q$. For a two-coordinate plane with cosine $c$ and sine $s$,

$$
\begin{bmatrix}\bar q^0_0\\\bar q^0_1\end{bmatrix}
 =\begin{bmatrix}c&s\\-s&c\end{bmatrix}
  \begin{bmatrix}\bar q_0\\\bar q_1\end{bmatrix}.
$$

Use the Chapter 9 function `dl_rope_backward` with the **same positions and frequencies** as forward. Apply it once to query gradients and once to the already-summed key gradients. Rotation is linear, so summing key contributions before the transpose is valid. Values were not rotated and must not be rotated now. The rotation formulation is grounded in [RoFormer Section 3.2][rope]; the split-half pairing and its tested C implementation remain exactly those from Chapter 9.

The projection derivatives are

$$
\bar W_q=(\bar Q^0)^{\mathsf T}X,\quad
\bar W_k=(\bar K^0)^{\mathsf T}X,\quad
\bar W_v=\bar V^{\mathsf T}X,
$$

$$
\bar X_{\text{attention}}
 =\bar Q^0W_q+\bar K^0W_k+\bar VW_v.
$$

Each weight gradient receives a TN matrix product. The input receives three NN products. On the first input product, use the caller's `dx_mode`; on the next two, always use `DL_ADD`. With `DL_SET`, the first path starts a new input derivative. With `DL_ADD`, it preserves an incoming contribution, such as a residual path from Chapter 7. Neither mode permits the second or third path to erase its predecessors.

All four weight gradients share a separate `weight_mode`. Choosing `dx_mode=DL_SET` while `weight_mode=DL_ADD` is useful when a fresh activation gradient is needed for each microbatch but parameters accumulate across microbatches. The reverse combination is also well-defined and is tested. “Backward always adds” and “backward always overwrites” are both too imprecise for composition.

## Extend the interface without changing the forward call

Replace `include/dl/attention.h` with the next two blocks in order, separated by one blank line. The first block retains the forward interface and its storage requirements. The second adds backward scratch, parameter-gradient views, and the new call. These structs contain **borrowed views**, not owning allocations.

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
```

The scratch names denote gradients of their matching forward quantities. Their order is intentionally different from `DlAttentionCache`: backward first creates a context gradient, then probability and score gradients. Keeping a distinct type avoids suggesting that a gradient tape is a valid forward tape.

`cache` still uses the forward's mutable-view type for compatibility, but backward treats all eight views as read-only. The caller must keep them, input rows, weights, positions, and frequencies unchanged until all backward consumers finish. A valid shape cannot prove that a cache belongs to the current weight values.

```c title="include/dl/attention.h"
typedef struct {
    DlMatrix context, probs, scores; /* dC [M,Q], dP/dS [B*Hq*T,T]. */
    DlMatrix q, k, v;               /* Rotated dQ [M,Q], dK/dV [M,K]. */
    DlMatrix q0, k0;                /* Before RoPE: [M,Q], [M,K]. */
} DlAttentionScratch;

typedef struct {
    DlMatrix query, key, value, output;
} DlAttentionWeightGrads;

/* Reverse one successful forward with UNCHANGED X, weights, cache, positions
 * and frequencies. These are borrowed read-only even though cache uses the
 * forward's mutable-view type. No provenance check can prove they still match.
 * dy [M,D]; dx [M,D]; each weight gradient has its weight's shape.
 * Scratch is overwritten, including masked dP/dS = 0. Old scratch is unread.
 * dx_mode and weight_mode are independent; weight_mode applies to all four
 * weight gradients. SET never reads their old cells; ADD requires finite
 * initialized cells. No averaging by tokens, batch, or head-sharing group.
 * Shared K/V derivatives sum ALL consuming query heads. dx sums Q/K/V paths.
 * Every write span is disjoint from all inputs and other write spans. Read
 * inputs may overlap. Keep truthful canonical allocation base/count metadata;
 * cross-type separation and lifetime requirements from forward still apply.
 * Preflight checks shapes, aliases, metadata, finite inputs and old ADD cells,
 * allowed finite scores, masked -INFINITY scores, and probability range/mask/
 * row sums (tolerance 8*FLT_EPSILON). Preflight errors write nothing.
 * This detects malformed tapes, not stale or consistently corrupted tapes.
 * Arithmetic uses double reductions with explicit float gradient stores.
 * A later DL_NUMERIC invalidates ALL scratch/destinations, including earlier
 * successful ADD writes. Do not retry an ADD on partially written gradients.
 * No allocation, in-place use, retained pointers, or hidden counters. */
DlStatus dl_attention_backward(DlBackend backend, DlAttentionConfig config,
                               DlConstMatrix x, DlAttentionWeights weights,
                               const size_t *positions, size_t position_count,
                               const double *inv_freq, size_t frequency_count,
                               DlAttentionCache cache, DlConstMatrix dy,
                               DlAttentionScratch scratch, DlMatrix dx,
                               DlWriteMode dx_mode, DlAttentionWeightGrads dw,
                               DlWriteMode weight_mode);

#endif
```

Scratch and each gradient output must have disjoint bounding spans from every input and every other output. Even two nonoverlapping sets of logical cells are rejected when their row-padding spans overlap. Read-only inputs may overlap one another. The canonical owner base/count metadata from Chapter 4 is still mandatory; rebasing a pointer to disguise an overlap defeats the check and violates the interface.

`DL_SET` outputs and all scratch can start as NaN because old cells are not read. `DL_ADD` output cells must be initialized and finite. Zero is an ordinary valid seed, not a special allocation state. The call allocates nothing and never frees caller storage. The test driver owns all allocations and releases them even on failure.

## Check the complete call before writing any destination

Append the following blocks, in order with a blank line between them, to the **unchanged end** of Chapter 10's `src/attention_ref.c`. Do not replace its existing includes, `Layout`, helpers, `attend`, or `dl_attention_forward`. The earlier file already includes `<float.h>` and `<math.h>` and provides `layout`, `shape`, `finite_cells`, `metadata`, and `store_float`. Each is fully implemented in Chapter 10; no unnamed helper is required.

Backward has fourteen read views: input, four weights, eight cache matrices, and incoming output gradient. It has thirteen write views: eight scratch matrices, input gradient, and four weight gradients. The next helper validates each view's declared bounds, then checks every write span against all reads and earlier writes. It compares offsets only when the views name the same canonical allocation; it does not order unrelated C pointers.

```c title="src/attention_ref.c"
/* Backward borrows fourteen read views and thirteen write views. */
static DlStatus backward_views(const DlConstMatrix in[14], const DlMatrix out[13])
{
    size_t ie[14] = {0}, oe[13] = {0};
    for (size_t i = 0; i < 14; ++i) {
        const DlStatus s = dl_matrix_end(in[i], &ie[i]);
        if (s != DL_OK) return s;
    }
    for (size_t o = 0; o < 13; ++o) {
        const DlConstMatrix v = dl_matrix_read(out[o]);
        const DlStatus s = dl_matrix_end(v, &oe[o]);
        if (s != DL_OK) return s;
        for (size_t i = 0; i < 14; ++i) {
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
```

`dl_matrix_end` returns an exclusive end in float elements. Two spans in the same owner overlap when each starts before the other ends. A structural rejection therefore happens before any derivative is initialized. As before, C metadata cannot establish whether a caller's allocation is actually live or its claimed count is truthful.

Scores require a different numerical check from ordinary arrays. Masked entries must remain negative infinity; allowed entries must be finite. Every probability must obey the matching mask, and an allowed row must sum close to one. We use eight float epsilons as a modest corruption-detection tolerance, not an exact proof of provenance. A normalized but incorrect probability row can pass this test, so preserving the successful forward tape is still a caller obligation.

```c title="src/attention_ref.c"
/* Ordinary finite-cell validation would wrongly reject masked score sentinels. */
static DlStatus saved_rows(DlAttentionConfig c, DlAttentionCache a)
{
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                double total = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    const float score = a.scores.base[a.scores.offset + sr * a.scores.ld + s];
                    const float p = a.probs.base[a.probs.offset + sr * a.probs.ld + s];
                    if (s > t) {
                        if (score != -INFINITY || p != 0.0f) return DL_NUMERIC;
                    } else {
                        if (!isfinite(score) || !isfinite(p) || p < 0.0f || p > 1.0f)
                            return DL_NUMERIC;
                        total += (double)p;
                    }
                }
                if (fabs(total - 1.0) > 8.0 * (double)FLT_EPSILON) return DL_NUMERIC;
            }
        }
    }
    return DL_OK;
}
```

The score array is inspected here but does not participate in derivative arithmetic. The original unrotated queries and keys are likewise retained and validated even though the rotation-transpose derivative does not need their values. Those diagnostic buffers were deliberate choices in Chapter 10; removing them later requires a corresponding explicit interface and memory-accounting change.

The public entry point will validate all remaining input cells and all old `ADD` cells before the first write. That includes a later parameter gradient, not merely the first one used. If the old value gradient contains NaN, we reject before updating the output-weight gradient. This preserves the distinction between a preflight failure and a later arithmetic failure.

## Materialize probability and score derivatives

The first numerical helper consumes the context gradient already computed by the entry point. It uses four address bases per query row: incoming context, saved probabilities, probability gradients, and score gradients. Remember the two flattened row conventions:

$$
r(b,t)=bT+t,\qquad a(b,h,t)=(bH_q+h)T+t.
$$

For example, with $B=2,T=3,H_q=4,H_{kv}=2,d=2$, batch one, query head three, time two uses projection row $5$, score row $23$, and shared key/value head $1$. Its head columns begin at $6$ in a query/context row, but at $2$ in a key/value row. Physical addresses also include each view's own `offset` and `ld`; do not substitute a logical width for a padded leading dimension.

```c title="src/attention_ref.c"
/* Validated views only. Materialize dP first, then differentiate softmax.
 * dP and dS above the diagonal are defined as zero: those are not variables. */
static DlStatus score_backward(DlAttentionConfig c, DlAttentionCache a,
                                DlAttentionScratch g)
{
    const size_t group = c.query_heads / c.kv_heads;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t t = 0; t < c.time; ++t) {
                const size_t row = b * c.time + t;
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                const size_t ga = g.context.offset + row * g.context.ld + h * c.head_dim;
                const size_t pa = a.probs.offset + sr * a.probs.ld;
                const size_t dpa = g.probs.offset + sr * g.probs.ld;
                const size_t dsa = g.scores.offset + sr * g.scores.ld;
                double center = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    if (s > t) {
                        g.probs.base[dpa + s] = 0.0f;
                        g.scores.base[dsa + s] = 0.0f;
                        continue;
                    }
                    const size_t va = a.v.offset + (b * c.time + s) * a.v.ld
                                    + kh * c.head_dim;
                    double dot = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        dot += (double)g.context.base[ga + j] * (double)a.v.base[va + j];
                    }
                    const DlStatus status = store_float(&g.probs.base[dpa + s], dot);
                    if (status != DL_OK) return status;
                    center += (double)a.probs.base[pa + s] * (double)g.probs.base[dpa + s];
                }
                for (size_t s = 0; s <= t; ++s) {
                    const double ds = (double)a.probs.base[pa + s]
                                    * ((double)g.probs.base[dpa + s] - center);
                    const DlStatus status = store_float(&g.scores.base[dsa + s], ds);
                    if (status != DL_OK) return status;
                }
            }
        }
    }
    return DL_OK;
}
```

The first inner loop computes $\bar P$ and its row scalar. `dot` widens **both float factors** before multiplication. `store_float` rejects nonfinite or out-of-range results and stores an accepted float derivative. `center` then uses that **stored** value. The second loop computes $\bar S$ only for allowed keys. Masked destinations were explicitly zeroed even if their old storage was NaN.

This reference uses float saved activations and float intermediate gradients, with double local products and reductions. Its intended derivative is the calculus of the continuous operators evaluated using these numerical intermediates, not the mathematical derivative of float rounding as a discontinuous function. Finite differences must therefore use a useful perturbation range rather than an arbitrarily tiny step. Part 2 checks several steps and includes explicit tolerances.

In exact arithmetic, $\mu=\bar c\cdot c$ as well: expand $c=\sum_s p_sv_s$ and exchange the sums. That identity will be useful for tiled backward. We deliberately compute the explicit $\sum_s p_s\bar p_s$ here, so the materialized intermediates remain inspectable. Substituting the separately rounded stored context would change the rounding path; it should not be described as bitwise identical.

## Gather query, shared-key, and shared-value gradients

The next function follows the three summations derived earlier. The query loop varies `s` over allowed keys. The key/value loop instead fixes `s`, then varies the sharing head and the query time `t` that can consume it. Neither loop crosses `b`.

```c title="src/attention_ref.c"
/* Gather each derivative into one double accumulator, then store it once.
 * In particular, K/V accumulators are OUTSIDE the sharing-head loop. */
static DlStatus qkv_backward(DlAttentionConfig c, DlAttentionCache a,
                              DlAttentionScratch g)
{
    const size_t group = c.query_heads / c.kv_heads;
    const double scale = 1.0 / sqrt((double)c.head_dim);
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            const size_t kh = h / group;
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double sum = 0.0;
                    for (size_t s = 0; s <= t; ++s) {
                        const size_t sa = g.scores.offset + sr * g.scores.ld + s;
                        const size_t ka = a.k.offset + (b * c.time + s) * a.k.ld
                                        + kh * c.head_dim + j;
                        sum += (double)g.scores.base[sa] * (double)a.k.base[ka];
                    }
                    const size_t qa = g.q.offset + (b * c.time + t) * g.q.ld
                                    + h * c.head_dim + j;
                    const DlStatus status = store_float(&g.q.base[qa], scale * sum);
                    if (status != DL_OK) return status;
                }
            }
        }
        for (size_t kh = 0; kh < c.kv_heads; ++kh) {
            for (size_t s = 0; s < c.time; ++s) {
                for (size_t j = 0; j < c.head_dim; ++j) {
                    double key_sum = 0.0, value_sum = 0.0;
                    for (size_t gh = 0; gh < group; ++gh) {
                        const size_t h = kh * group + gh;
                        for (size_t t = s; t < c.time; ++t) {
                            const size_t sr = (b * c.query_heads + h) * c.time + t;
                            const size_t sa = g.scores.offset + sr * g.scores.ld + s;
                            const size_t pa = a.probs.offset + sr * a.probs.ld + s;
                            const size_t qa = a.q.offset + (b * c.time + t) * a.q.ld
                                            + h * c.head_dim + j;
                            const size_t ca = g.context.offset + (b * c.time + t) * g.context.ld
                                            + h * c.head_dim + j;
                            key_sum += (double)g.scores.base[sa] * (double)a.q.base[qa];
                            value_sum += (double)a.probs.base[pa] * (double)g.context.base[ca];
                        }
                    }
                    const size_t ka = g.k.offset + (b * c.time + s) * g.k.ld
                                    + kh * c.head_dim + j;
                    const size_t va = g.v.offset + (b * c.time + s) * g.v.ld
                                    + kh * c.head_dim + j;
                    DlStatus status = store_float(&g.k.base[ka], scale * key_sum);
                    if (status != DL_OK) return status;
                    status = store_float(&g.v.base[va], value_sum);
                    if (status != DL_OK) return status;
                }
            }
        }
    }
    return DL_OK;
}
```

`key_sum` and `value_sum` are initialized once for one destination coordinate, **before** `gh`. The selected query head is `kh * group + gh`, not just `gh` and not `kh % group`. Moving either initialization into that loop keeps only a suffix of the group. Dividing the stored answer by `group` silently changes a shared-parameter sum into a mean.

The score scale appears in the query and key stores but not in the value store. Values do not pass through the scaled dot product. Using the full query width instead of `head_dim` would be just as incorrect here as in forward.

Each scratch cell is stored exactly once by this gather, so no preliminary whole-array zero is required. This is not an omission of gradient initialization: the double accumulators provide it locally. A failed store may leave earlier coordinates valid-looking, but the caller must discard the entire failed result. Reconstructing a partially successful shared sum would require a different failure protocol.

## Connect validation, projections, rotations, and accumulation

The entry point below is the complete orchestration. Its `in`, `out`, and expected-shape arrays use identical ordering. `ir`/`ic` describe input rows/columns; `orows`/`ocols` describe output rows/columns. After layout and alias checks, it rejects shape mismatches, invalid rotary metadata, malformed tapes, and invalid old accumulation destinations. Only then does it begin the reverse computation.

The repeated parameter/input projection code uses three small view arrays. Copying those view structs borrows the same live bytes; it does not duplicate or transfer an owner. `p == 0 ? dx_mode : DL_ADD` is the precise merge rule for the three input paths. No operator divides by $M$, $B$, $T$, or $G$.

```c title="src/attention_ref.c"
DlStatus dl_attention_backward(DlBackend backend, DlAttentionConfig c,
                               DlConstMatrix x, DlAttentionWeights w,
                               const size_t *positions, size_t position_count,
                               const double *inv_freq, size_t frequency_count,
                               DlAttentionCache a, DlConstMatrix dy,
                               DlAttentionScratch g, DlMatrix dx,
                               DlWriteMode dx_mode, DlAttentionWeightGrads dw,
                               DlWriteMode weight_mode)
{
    if ((backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) ||
        (dx_mode != DL_SET && dx_mode != DL_ADD) ||
        (weight_mode != DL_SET && weight_mode != DL_ADD)) return DL_ARGUMENT;
    Layout z = {0};
    DlStatus s = layout(c, &z);
    if (s != DL_OK) return s;
    const DlConstMatrix in[14] = {x, w.query, w.key, w.value, w.output,
        dl_matrix_read(a.q0), dl_matrix_read(a.k0), dl_matrix_read(a.v),
        dl_matrix_read(a.q), dl_matrix_read(a.k), dl_matrix_read(a.scores),
        dl_matrix_read(a.probs), dl_matrix_read(a.context), dy};
    const DlMatrix out[13] = {g.context, g.probs, g.scores, g.q, g.k, g.v,
        g.q0, g.k0, dx, dw.query, dw.key, dw.value, dw.output};
    s = backward_views(in, out);
    if (s != DL_OK) return s;
    const size_t d = x.cols, m = z.rows, q = z.query_width, k = z.kv_width;
    const size_t ir[14] = {m, q, k, k, d, m, m, m, m, m, z.score_rows,
                           z.score_rows, m, m};
    const size_t ic[14] = {d, d, d, d, q, q, k, k, q, k, c.time, c.time, q, d};
    const size_t orows[13] = {m, z.score_rows, z.score_rows, m, m, m, m, m,
                              m, q, k, k, d};
    const size_t ocols[13] = {q, c.time, c.time, q, k, k, q, k, d, d, d, d, q};
    for (size_t i = 0; i < 14; ++i) {
        if (!shape(in[i], ir[i], ic[i])) return DL_SHAPE;
    }
    for (size_t o = 0; o < 13; ++o) {
        if (!shape(dl_matrix_read(out[o]), orows[o], ocols[o])) return DL_SHAPE;
    }
    s = metadata(z, c, positions, position_count, inv_freq, frequency_count);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < 14; ++i) {
        if (i != 10 && i != 11 && !finite_cells(in[i])) return DL_NUMERIC;
    }
    s = saved_rows(c, a);
    if (s != DL_OK) return s;
    if (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) return DL_NUMERIC;
    if (weight_mode == DL_ADD) {
        for (size_t o = 9; o < 13; ++o) {
            if (!finite_cells(dl_matrix_read(out[o]))) return DL_NUMERIC;
        }
    }
    /* The first write occurs only after ALL structural/numeric preflight. */
    s = dl_matmul(backend, dy, DL_NOTRANS, w.output, DL_NOTRANS, g.context, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dy, DL_TRANS, dl_matrix_read(a.context), DL_NOTRANS,
                   dw.output, weight_mode);
    if (s != DL_OK) return s;
    s = score_backward(c, a, g);
    if (s != DL_OK) return s;
    s = qkv_backward(c, a, g);
    if (s != DL_OK) return s;
    s = dl_rope_backward(dl_matrix_read(g.q), c.head_dim, positions, position_count,
                         inv_freq, frequency_count, g.q0, DL_SET);
    if (s != DL_OK) return s;
    s = dl_rope_backward(dl_matrix_read(g.k), c.head_dim, positions, position_count,
                         inv_freq, frequency_count, g.k0, DL_SET);
    if (s != DL_OK) return s;
    const DlConstMatrix upstream[3] = {dl_matrix_read(g.q0), dl_matrix_read(g.k0),
                                       dl_matrix_read(g.v)};
    const DlConstMatrix weight[3] = {w.query, w.key, w.value};
    const DlMatrix gradient[3] = {dw.query, dw.key, dw.value};
    for (size_t p = 0; p < 3; ++p) {
        s = dl_matmul(backend, upstream[p], DL_TRANS, x, DL_NOTRANS,
                       gradient[p], weight_mode);
        if (s != DL_OK) return s;
        s = dl_matmul(backend, upstream[p], DL_NOTRANS, weight[p], DL_NOTRANS,
                       dx, p == 0 ? dx_mode : DL_ADD);
        if (s != DL_OK) return s;
    }
    return DL_OK;
}
```

The intermediate lifetimes are now explicit. $\bar C$ is needed by probability and value backward; $\bar P$ is needed until score backward finishes; $\bar S$ is needed for query/key backward; $\bar Q$ and $\bar K$ are needed for their inverse rotations; $\bar Q^0,\bar K^0,\bar V$ are needed for projection-weight and input products. The saved context supplies $\bar W_o$. The saved input supplies the other three weight gradients. The original weights supply all four input-side products.

Do not update weights between forward and backward. Do not reuse a forward tape for a different input merely because it has the same shape. Do not overwrite one gradient scratch view with another's storage before its final consumer. This deliberately simple implementation keeps all eight gradient scratch arrays live and disjoint; later memory optimization must justify any reuse by those consumer boundaries.

## Understand the failure boundary and memory cost

Preflight failures write nothing. They include malformed view metadata, dimensions, overlap, rotary metadata, nonfinite read inputs, nonfinite old `ADD` cells, invalid score sentinels, and malformed probability rows. Those checks examine all inputs even when a particular incoming derivative is zero. This is a validity policy, not a claim that every input has a nonzero mathematical dependency.

Once arithmetic starts, `DL_NUMERIC` may follow writes to scratch and parameter gradients. For example, an output-weight/input-gradient product can exceed `FLT_MAX` even though both operands and the preceding forward result were finite. In `ADD` mode, some old contributions may already have changed. **Do not retry an `ADD` call on that partially written state.** Recompute from a known accumulation boundary or discard the batch and clear/rebuild its gradient state. There is no allocation to clean up inside this call; the caller still owns and releases all buffers.

The eight scratch matrices require

$$
3M(Q_w+K_w)+2BH_qT^2
$$

float elements. This equals the payload of the eight forward-cache matrices, though the meanings and lifetime differ. Retaining both together costs $6M(Q_w+K_w)+4BH_qT^2$ floats, excluding weights, parameter gradients, input/output arrays, leading-dimension padding, and matrix-kernel local workspace. This is a storage calculation, not measured process memory.

For the roadmap's byte-model microbatch, $B=2,T=128,H_q=8,Q_w=256,K_w=64$. One square attention matrix is exactly 1 MiB with four-byte floats. The two retained forward matrices and two backward matrices occupy 4 MiB for this layer; their linear forward-plus-backward arrays add 1.875 MiB. The reference is feasible at this size, but the square part grows by four when $T$ doubles. We have not removed that storage here, nor claimed an attention speedup.

The completed backward is intentionally explicit enough to serve as a later comparison target. First run the hand, dependency, sharing, accumulation, and finite-difference checks in Part 2. A plausible loss and clean sanitizer run cannot replace those numerical checks.

## Technical references

[Dao et al., *FlashAttention*, version 2, Appendix B.2 and Algorithm 3 in B.4][flash] provide the standard attention backward identities. This chapter derives them for the specified causal, grouped, scaled layout and implements its own scalar materialized loops. Appendix B.4's tiled algorithm is a later topic, not an implementation hidden here.

[Ainslie et al., *GQA*, version 3, Section 2.2][gqa] defines shared key/value groups and the MQA/MHA endpoints. The expanded-head test in Part 2 applies the chain rule to tied copies; it does not compare unrelated parameter sets or adopt checkpoint-conversion averaging as a gradient rule.

[Su et al., *RoFormer*, version 5, Section 3.2][rope] supplies the rotation formulation. The existing split-half implementation and its independent convention tests in Chapter 9 remain prerequisites. A consistently wrong forward/backward coordinate convention can still pass derivative self-consistency tests.

[flash]: https://arxiv.org/html/2205.14135v2
[gqa]: https://arxiv.org/html/2305.13245v3
[rope]: https://arxiv.org/html/2104.09864v5
