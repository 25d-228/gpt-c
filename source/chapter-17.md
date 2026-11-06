# Chapter 17 — Prefill and incremental key/value caching

The full-prefix generator is correct but forgetful. After choosing a new token, it projects and processes every earlier token again. This chapter preserves the information that a causal decoder can safely reuse: each block's already-rotated keys and unrotated values. A prompt is processed once. Later calls consume one new input token without recomputing old token representations.

**Prerequisites:** Chapters 9–11 for rotary coordinates, causal attention, and grouped heads; Chapter 12 for the decoder's parameter directory; and the complete [Chapter 16 source state](https://25d-228.github.io/gpt-c/media/chapter-16/reference/Makefile). In particular, retain `DlWeights`, `dl_inference_forward`, the weight-only checkpoint loader, and both selection functions. The unchanged full-prefix path is an independent storage-schedule reference, not a function called from cached decoding.

**Implementation estimate:** **4.50 hours** for this entire chapter: 3.00 for prefill, decode, storage, and state transitions; 1.50 for equivalence, boundary, and failure checks. Reading and unattended computation are separate. Three linked parts form one chapter. This part derives and allocates the cache; [Part 2](https://25d-228.github.io/gpt-c/chapter-17-part-2.html) implements computation and generation; [Part 3](https://25d-228.github.io/gpt-c/chapter-17-part-3.html) supplies complete tests, builds, and actual continuations.

The [68-file companion](https://25d-228.github.io/gpt-c/media/chapter-17/reference/Makefile) contains the entire prerequisite state. Create `include/dl/cache.h`, `src/cache.c`, and `tests/test_cache.c`. Replace `app/generate.c`, `tests/test_generate_cli.py`, and the root Makefile with the complete listings here. The other 62 prerequisite files are unchanged. Concatenate blocks labeled with the same filename in part order, with one blank line between blocks. No earlier function is silently patched or omitted.

> **Learning objectives**
> Separate selected tokens from consumed tokens; derive a fixed-stride, per-layer shared-head history; reuse absolute rotary coordinates correctly; implement layer-wise prompt prefill and one-row decoding; and make numerical failure preserve the committed prefix without copying the entire cache.

## Which computation may be remembered?

Consider position $t$ in block $\ell$. Causal attention reads only positions $0$ through $t$. RMSNorm normalizes coordinates within a row, not across token positions. Projections, residual additions, and SwiGLU also operate row by row. Therefore appending a token at position $t+1$ cannot change the earlier row's computation.

That argument extends through the blocks. The input embedding at an earlier position is unchanged. If its input to block $\ell$ is unchanged, every operation producing that block's output depends only on unchanged prefix rows. Its output, and hence the next block's input, is unchanged. By induction, earlier keys and values at every block remain valid. This requires fixed weights, the same prefix, the same positions, and the deterministic inference operators already implemented. It is not permission to reuse a cache after changing a prompt or training the weights.

Hugging Face's versioned [caching explanation, “Attention matrices” and “Cache class”][cache-doc] describes the same per-layer reuse and absolute cache-position requirement. Its high-level cache classes are not dependencies here; the storage and computation below are C code.

Why not cache old queries? A new prediction requires the **new** query compared with old and new keys. Earlier queries produced earlier outputs, which are not recomputed. Why not share one cache across blocks? Block 1 projects the output of block 0, not the original embedding; its keys and values generally differ. Temporary activation buffers may be reused across blocks. Persistent histories may not.

## Consumed input is not selected output

Let a prompt contain $P$ token IDs, at absolute positions $0$ through $P-1$. Prefill consumes all $P$ and returns the last row's logits. Those logits predict position $P$. Selection chooses a token ID, but selection alone does not compute that token's key or value.

| Operation | Consumed positions | Committed length | Logits predict |
|---|---|---:|---:|
| Prefill three IDs | 0, 1, 2 | 3 | position 3 |
| Select an ID from those logits | unchanged | 3 | no new logits |
| Decode that selected ID | additionally 3 | 4 | position 4 |
| Select the final requested ID and stop | unchanged | 4 | no new logits |

A request for $N>0$ new tokens makes one prefill and $N-1$ decode calls. It produces a transcript of length $P+N$, while the cache contains $P+N-1$ consumed inputs. A zero-new-token request needs neither prefill nor decode. We retain the application's conservative requirement that the entire transcript fit its capacity, including the final unconsumed token.

This distinction prevents an off-by-one error that can look like poor model quality: forwarding the prompt's last token twice uses the wrong absolute position and repeats its contribution. The implementation has separate `prefill` and `decode` entry points, with an empty/nonempty-state requirement, rather than guessing the caller's intention from an array length.

## Shapes and a physical address

Use one sequence, no padding, and no sliding window. Let $L$ be the number of decoder blocks, $C$ the allocated sequence capacity, $D$ the model width, $F$ the feed-forward width, and $V$ the vocabulary size. Let $H_q$ and $H_{kv}$ be the query and key/value head counts. Define

$$
d=D/H_q,\qquad K=H_{kv}d,\qquad g=H_q/H_{kv}.
$$

Here $d$ is an even head width, $K$ is the combined key/value width, and $g$ is the number of query heads sharing one key/value head. Query head $h$ reads key/value head $\lfloor h/g\rfloor$. With four query heads and two key/value heads, the mapping is $[0,0,1,1]$, not $[0,1,0,1]$. This grouping follows [GQA, Section 2.2][gqa]; no physical replication of shared heads is required.

The owned history has shape $[L,2,C,K]$. The second dimension selects rotated keys (`kind=0`) or values (`kind=1`). Within each plane, tokens are rows and heads are contiguous portions of each row. A coordinate has float-element offset

$$
\operatorname{offset}(\ell,k,t,h,j)
=\big((2\ell+k)C+t\big)K+hd+j.
$$

Here $k\in\{0,1\}$, $0\leq t<C$, $0\leq h<H_{kv}$, and $0\leq j<d$. Multiply by `sizeof(float)` only when counting bytes, not when indexing a `float *`.

For $L=2$, $C=5$, $H_{kv}=2$, and $d=2$, $K=4$ and the owner has 80 floats. Block 1's key at token 3, shared head 1, coordinate 0 is at `((2*1+0)*5+3)*4+1*2+0 = 54`. Its value is at offset 74. Changing the committed length from three to four does **not** change either plane's stride. Replacing $C$ with the current length in this address would relocate older rows on every append.

All views retain the owner's original `base` and `count`. A suffix view changes `offset` and `rows`; it does not invent a new allocation base. This preserves the alias checks taught in Chapter 4.

## Absolute positions and one rotation per new key

Let $x_t^\ell$ denote a block's pre-attention normalized row. New projections are $q_t^0=W_qx_t^\ell$, $k_t^0=W_kx_t^\ell$, and $v_t=W_vx_t^\ell$, with per-head partitions understood. Store

$$
q_t=R(t)q_t^0,\qquad k_t=R(t)k_t^0.
$$

$R(t)$ is Chapter 9's split-half rotary transformation, evaluated separately within each head. Values are not rotated. [RoFormer, Section 3.2][rope] gives the rotary construction and its relative-position dependence; the coordinate convention in our code remains the one already tested in Chapter 9.

If the committed length is $S$, a new row with local index $r$ has absolute position $S+r$. During initial prefill $S=0$. During one-token decode $r=0$, but the position is $S$, not zero. Cached keys already contain their rotations. Rotating the entire stored prefix again would change the computation rather than save work.

For query head $h$, local row $r$, and shared head $h_k=\lfloor h/g\rfloor$, compute

$$
s_{r,h,t}=\frac{1}{\sqrt d}
\sum_{j=0}^{d-1}q_{r,hd+j}\,k_{t,h_kd+j},
\qquad 0\leq t\leq S+r.
$$

The new query attends to the committed prefix **and itself**. The diagonal is included. In prefill, later prompt keys may already have been projected, but they must remain outside this row's allowed set. An allocated slot is not necessarily a visible token.

Normalize only the $S+r+1$ allowed scores, then form the weighted-value sum. The implementation keeps the same rounding boundaries as materialized attention: double dot products, float stored scores, double maximum subtraction and normalization, float stored probabilities, and double value reductions before a float output store. Matching these boundaries makes disagreements more informative than comparisons between two unrelated precision choices.

## Materializing a row is not tiled online attention

We process all prompt projections as matrices, block by block. Inside attention, one query/head row uses two temporary arrays of capacity $C$: its scores and probabilities. After producing that row's context, the next row reuses the arrays. No backward pass needs them during inference.

This removes a square inference score allocation without implementing online softmax. We still store every allowed score for the current row and scan it several times. Chapter 18 will maintain an online normalization state across tiles; Chapter 19 will address backward recomputation. The unchanged full-prefix reference and the training tape still materialize their complete score/probability matrices.

The transient matrices for $n$ new rows are seven $[n,D]$ arrays (`hidden`, `norm`, `branch`, `middle`, `q0`, `q`, `context`), one $[n,K]$ unrotated-key array, and three $[n,F]$ feed-forward arrays. Values project directly into the new history suffix, and RoPE writes the new keys there. With the two score rows, scratch contains

$$
A(n,C)=7nD+nK+3nF+2C
$$

floats. Allocate it for $n=C$ so a full-capacity prompt fits. Decode maps only $n=1$ rows, without reallocating or shrinking that owner. The larger prompt capacity remains reserved. Persistent history contains

$$
B(C)=2LCK
$$

floats. Cache creation checks the **sum** $A(C,C)+B(C)$ against its float-payload budget. It also owns $C$ positions of type `size_t` and $d/2$ frequency doubles. Caller logits, probability scratch used by sampling, weights, allocator overhead, and the blocked kernel's fixed stack arrays are separate. Allocation arithmetic is not a process-memory measurement.

## Declare the owner and its boundary

Create the following header. `DlKvCache` owns two float allocations and two metadata arrays. Its `weights` pointer borrows the already-ready `DlWeights`; it neither retains a training model nor loads another copy of the weights. Several caches may borrow the same immutable weights, provided their lifetimes and callers remain independent.

```c title="include/dl/cache.h"
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
 * Positive capacity, last position <= DL_ROPE_MAX_POSITION. No allocation
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
```

The public `length` is a commit boundary, not a progress counter inside a block loop. The caller may inspect it but must not edit it. Owner structs stay at stable addresses. Destroy every cache and full-prefix workspace before destroying the borrowed weights. Reloading the same weight owner in place is not an allowed way to change the model beneath a cache.

## Count the workspace before allocating it

Start `src/cache.c` with its includes, return helper, and private view record. `RETURN_IF_ERROR` returns from the enclosing function; its temporary variable has block scope inside the `do` statement, as in earlier chapters. No global state is introduced.

```c title="src/cache.c"
#include "dl/cache.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

typedef struct {
    DlMatrix hidden, norm, branch, middle, q0, k0, q, context;
    DlMatrix scores, probs;          /* ONE row of length capacity, reused. */
    DlFfnCache ffn;
} Views;
```

`place` has two uses. With an empty tensor, it counts a shape using checked multiplication and addition. With an allocated tensor, it constructs a bounded borrowed view at the same cursor. `layout` runs first for the maximum prompt, then again for each call's actual new-row count. It never lays out persistent history: that owner's fixed capacity strides must not move with the temporary layout.

```c title="src/cache.c"
static DlStatus place(DlTensor *arena, size_t *cursor, size_t rows,
                       size_t cols, DlMatrix *view)
{
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    if (arena->data != NULL)
        RETURN_IF_ERROR(dl_matrix_view(arena, *cursor, rows, cols, cols, view));
    *cursor = end;
    return DL_OK;
}

static DlStatus layout(DlTensor *arena, DlModelConfig c, size_t rows,
                        size_t capacity, Views *v, size_t *count)
{
    size_t cursor = 0;
    size_t kv = c.kv_heads * (c.width / c.query_heads);
#define PLACE(rows_, cols_, member_) \
    RETURN_IF_ERROR(place(arena, &cursor, rows_, cols_, &v->member_))
    PLACE(rows, c.width, hidden);
    PLACE(rows, c.width, norm);
    PLACE(rows, c.width, branch);
    PLACE(rows, c.width, middle);
    PLACE(rows, c.width, q0);
    PLACE(rows, kv, k0);
    PLACE(rows, c.width, q);
    PLACE(rows, c.width, context);
    PLACE(rows, c.ffn_width, ffn.gate);
    PLACE(rows, c.ffn_width, ffn.value);
    PLACE(rows, c.ffn_width, ffn.hidden);
    PLACE(1, capacity, scores);
    PLACE(1, capacity, probs);
#undef PLACE
    *count = cursor;
    return DL_OK;
}
```

There is no `PLACE(rows, K, value)` or rotated-key temporary. Those outputs go to history. The score arrays are `[1,C]`, not `[H_q*n,C]`: attention processes query heads and rows sequentially, consuming their scores before reuse. The final cursor equals $7nD+nK+3nF+2C$. On decode this remaps scratch near the beginning of its allocation; no retained history pointer points there.

## Construction and cleanup are separate from execution

The destructor handles a fully built owner and every partially constructed state. The constructor builds a local `next`, and publishes it only after all allocations and frequency construction succeed. This is the same output-unchanged-on-failure pattern used by the buffer and weight owners.

```c title="src/cache.c"
void dl_cache_destroy(DlKvCache *s)
{
    if (s == NULL) return;
    free(s->inv_freq);
    free(s->positions);
    dl_tensor_destroy(&s->arena);
    dl_tensor_destroy(&s->history);
    *s = (DlKvCache){0};
}
```

The constructor validates the immutable configuration before deriving $d$ or $K$. It checks the rotary-position limit using `capacity - 1`, after proving capacity is positive. Checked products cover both arenas and the metadata byte counts before the first allocation. A successful float budget check does not guarantee `malloc` success; failure still releases every allocation already obtained.

```c title="src/cache.c"
DlStatus dl_cache_create(DlKvCache *out, const DlWeights *weights,
                          size_t capacity, size_t max_float_bytes)
{
    if (out == NULL || weights == NULL || weights->params == NULL ||
        capacity == 0 || max_float_bytes == 0) return DL_ARGUMENT;
    if (!weights->ready) return DL_STATE;
    if (out->weights != NULL || out->history.data != NULL || out->arena.data != NULL ||
        out->positions != NULL || out->inv_freq != NULL ||
        out->capacity != 0 || out->length != 0) return DL_STATE;
    if (capacity - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlModelConfig c = weights->config;
    size_t params = 0, values = 0;
    RETURN_IF_ERROR(dl_model_config_validate(c, &params, &values));
    size_t d = c.width / c.query_heads, kv = c.kv_heads * d;
    DlKvCache next = {.weights = weights, .capacity = capacity};
    Views views = {0};
    size_t scratch = 0, history = 0, total = 0, bytes = 0;
    size_t position_bytes = 0, frequency_bytes = 0;
    RETURN_IF_ERROR(layout(&next.arena, c, capacity, capacity, &views, &scratch));
    RETURN_IF_ERROR(dl_size_mul(c.layers, 2, &history));
    RETURN_IF_ERROR(dl_size_mul(history, capacity, &history));
    RETURN_IF_ERROR(dl_size_mul(history, kv, &history));
    RETURN_IF_ERROR(dl_size_add(history, scratch, &total));
    RETURN_IF_ERROR(dl_size_mul(total, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(capacity, sizeof(*next.positions), &position_bytes));
    RETURN_IF_ERROR(dl_size_mul(d / 2, sizeof(*next.inv_freq), &frequency_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    const size_t shape[] = {c.layers, 2, capacity, kv};
    DlStatus status = dl_tensor_create(&next.history, 4, shape);
    if (status == DL_OK) status = dl_tensor_create(&next.arena, 1, &scratch);
    if (status == DL_OK) {
        next.positions = malloc(position_bytes);
        next.inv_freq = malloc(frequency_bytes);
        if (next.positions == NULL || next.inv_freq == NULL) status = DL_ALLOC;
    }
    if (status == DL_OK)
        status = dl_rope_frequencies(d, c.rope_theta, next.inv_freq, d / 2);
    if (status != DL_OK) { dl_cache_destroy(&next); return status; }
    *out = next;
    return DL_OK;
}
```

The history starts uninitialized. That is intentional: no element is readable merely because it was allocated. Each successful call writes every new visible key and value before attention consumes it. Tests poison unused storage with NaNs to make accidental reads observable. Production code need not clear the entire capacity before a prompt.

The internal `live` check detects an absent constructed owner or an impossible public length. It is not a general validator for arbitrarily forged structs or dangling pointers. C cannot infer a pointer's lifetime; unchanged, truthful ownership metadata remains a caller requirement.

```c title="src/cache.c"
static DlStatus live(const DlKvCache *s)
{
    if (s == NULL || s->weights == NULL || s->history.data == NULL ||
        s->arena.data == NULL || s->positions == NULL || s->inv_freq == NULL)
        return DL_STATE;
    if (!s->weights->ready || s->length > s->capacity) return DL_STATE;
    return DL_OK;
}

DlStatus dl_cache_reset(DlKvCache *s)
{
    RETURN_IF_ERROR(live(s));
    s->length = 0;
    return DL_OK;
}
```

Reset changes only visibility. Old bytes remain allocated, but length zero makes all of them unreachable to the next computation until they are overwritten. This is not a secure erasure API. A second cache borrowing the same weights is unaffected, and caller-owned logits do not disappear when the first cache resets.

Continue with [Part 2 — Computing and committing a cached step](https://25d-228.github.io/gpt-c/chapter-17-part-2.html).

## Technical references

The implementation and storage arithmetic above are developed directly in this chapter. The external references establish the underlying inference mechanisms:

- [Transformers v4.57.1, “Caching,” especially “Attention matrices” and “Cache class”][cache-doc]: per-layer reuse, newly consumed inputs, and cache positions. A versioned conceptual reference, not a Python runtime requirement.
- [Ainslie et al., GQA, Section 2.2][gqa]: groups of query heads sharing key/value heads.
- [Su et al., RoFormer, Section 3.2][rope]: rotary transformations and relative-position dependence.

[cache-doc]: https://huggingface.co/docs/transformers/v4.57.1/cache_explanation
[gqa]: https://arxiv.org/html/2305.13245v3#S2.SS2
[rope]: https://arxiv.org/html/2104.09864v5#S3.SS2
