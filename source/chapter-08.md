# Chapter 8 — SwiGLU and the two-branch backward pass

A feed-forward sublayer transforms the coordinates of each token without mixing different token rows. In this chapter, that transformation has three learned matrices and a multiplication between two independently projected branches. The multiplication matters in both directions: forward it combines the branches; backward it sends a different derivative down each branch. Their contributions meet again at the input.

We will implement the entire bias-free sublayer, not just an activation function. Its saved arrays, temporary gradients, output writes, and failure behavior remain visible. A small row gives us numbers to calculate by hand; a second program checks every input and weight derivative, then connects the sublayer to the normalization and residual operations from Chapter 7.

> **Learning objectives**  Evaluate sigmoid and SiLU without overflowing an exponential; derive both branches of a gated product; map all three projections and their derivatives to the existing matrix interface; keep forward values alive without an automatic differentiation tape; and merge input-gradient contributions without losing previously accumulated work.

**Prerequisites:** the complete workspace from [Chapter 7, Part 1](https://25d-228.github.io/gpt-c/chapter-07.html) and [Part 2](https://25d-228.github.io/gpt-c/chapter-07-part-2.html). Chapters 3–6 supply numerical derivatives, borrowed views, matrix orientation, SET/ADD semantics, and scalar/blocked dispatch. No optimizer, downloaded weights, Python model computation, or future decoder infrastructure is required.

**This chapter has two parts:** this page derives and implements the complete sublayer. [Part 2 — Checking feed-forward branches and saved state](https://25d-228.github.io/gpt-c/chapter-08-part-2.html) supplies all executable fixtures, the connected normalization/residual graph, finite differences, the Makefile, and diagnostic experiments. They form one complete chapter.

**Implementation estimate:** 3.50 hours: 2.25 for connected forward/backward code and 1.25 for numerical checks, buffer lifetimes, and integration. Reading is separate. Record actual hands-on time; the estimate is not a required minimum.

## Continue the existing workspace

Retain all eighteen Chapter 7 files. Create `include/dl/ffn.h`, `src/ffn.c`, and `tests/test_ffn.c`; replace the root `Makefile` with Part 2's complete version. The eighteen earlier paths remain, with only the Makefile changed, giving **21 cumulative files**. The separate foundation programs are unchanged.

Code blocks labeled with a filename concatenate in order into that file. All new helpers are included. The matching companion files are [the header](https://25d-228.github.io/gpt-c/media/chapter-08/reference/include/dl/ffn.h), [the implementation](https://25d-228.github.io/gpt-c/media/chapter-08/reference/src/ffn.c), [the tests](https://25d-228.github.io/gpt-c/media/chapter-08/reference/tests/test_ffn.c), and [the Makefile](https://25d-228.github.io/gpt-c/media/chapter-08/reference/Makefile). The complete prior code is included in the same companion directory, not replaced by summaries.

A matrix coordinate is still `base[offset + i*ld + j]`, in float elements. `ld` is the physical row spacing and may exceed the logical column count. All views retain the allocation's canonical base and count; they do not own storage or prove its lifetime. We keep weights in `[output,input]` order and use the transpose flag rather than physically transposing them.

## Three projections, one coordinatewise product

Let $M$ be the number of token rows, $D$ the model width, and $F$ the intermediate feed-forward width. All three are positive integers. An input $X$ has shape $[M,D]$. Flattening a dense batch/sequence prefix into $M$ does not allow this sublayer to mix those rows.

The three weight matrices and the forward intermediates are:

| Symbol | Stored shape | Meaning |
|---|---|---|
| $W_g$ | $[F,D]$ | Gate projection weights |
| $W_v$ | $[F,D]$ | Value-branch projection weights |
| $W_o$ | $[D,F]$ | Output, or down-projection, weights |
| $A=XW_g^T$ | $[M,F]$ | Gate preactivation |
| $B=XW_v^T$ | $[M,F]$ | Value-branch activation |
| $H=\operatorname{SiLU}(A)\odot B$ | $[M,F]$ | Coordinatewise product |
| $Y=HW_o^T$ | $[M,D]$ | Sublayer output |

Here $T$ denotes transpose and $\odot$ denotes multiplication of corresponding coordinates, not matrix multiplication. SiLU is applied separately to each coordinate of $A$. The letter $B$ is an array here, not a batch-size symbol. The value branch is also **not** an attention value tensor or a key/value cache.

This is the bias-free, unit-parameter SwiGLU feed-forward form in [Shazeer's GLU-variants paper, equations 5–6][glu]. That paper writes weights on the right in input-by-output order; our stored matrices are transposed relative to that notation. The operation is the same. SwiGLU is not a probability gate: SiLU can be negative and is not bounded above by one.

For a row $i$, intermediate coordinate $f$, and output coordinate $j$:

$$
A_{if}=\sum_{k=0}^{D-1}X_{ik}(W_g)_{fk},
\qquad
B_{if}=\sum_{k=0}^{D-1}X_{ik}(W_v)_{fk},
$$

$$
H_{if}=\operatorname{SiLU}(A_{if})B_{if},
\qquad
Y_{ij}=\sum_{f=0}^{F-1}H_{if}(W_o)_{jf}.
$$

Both projections read the **same** $X$. The value projection does not consume the gate projection. Each output row depends only on the matching input row; all rows share the weights. Weight gradients will therefore sum contributions from rows, even though forward does not mix them.

There are $3DF$ learned scalar weights, with no biases. For comparison, a two-matrix block of intermediate width $4D$ has $8D^2$ weights. Equating these counts gives $F=8D/3$, before choosing an integer implementation width. This count explains the width adjustment discussed in the [GLU paper][glu] and used in [LLaMA's architecture description][llama]; it is not a shape restriction of our API. The tests deliberately use $D=3,F=5$ to reveal incorrect transposes.

## Sigmoid is a formula, not permission to overflow

For a real scalar $a$, define sigmoid $\sigma$ and SiLU $\phi$ by

$$
\sigma(a)=\frac{1}{1+e^{-a}},
\qquad \phi(a)=a\sigma(a).
$$

The [Swish paper's Section 2][swish] gives this unit-parameter activation and its derivative. We use the name SiLU for the same scalar function; no learned temperature or extra $\beta$ parameter is introduced.

The literal expression `1.0 / (1.0 + exp(-a))` evaluates an enormous exponential when $a$ is sufficiently negative. A finite input does not make that intermediate representable. Instead define $t=e^{-|a|}$, so the exponential's argument is never positive. Then

$$
\sigma(a)=
\begin{cases}
1/(1+t),&a\geq 0,\\
t/(1+t),&a<0.
\end{cases}
$$

For negative $a$, this follows by multiplying numerator and denominator of the original sigmoid expression by $e^a$. For positive $a$, $t=e^{-a}$ already has the needed form. Both branches use a denominator between one and two.

Differentiate the original real-valued expression before choosing its floating-point evaluation:

$$
\sigma'(a)=\frac{e^{-a}}{(1+e^{-a})^2}
          =\sigma(a)(1-\sigma(a)).
$$

The product rule for $\phi(a)=a\sigma(a)$ gives

$$
\phi'(a)=\sigma(a)+a\sigma'(a).
$$

For both sign branches, $\sigma'(a)=t/(1+t)^2$. Computing this from $t$ avoids subtracting a sigmoid value already rounded to one. We will use

$$
s=\sigma(a),\qquad d_s=\frac{t}{(1+t)^2},
\qquad \phi(a)=as,\qquad\phi'(a)=s+a d_s.
$$

At $a=0$, $s=1/2$, $\phi(0)=0$, and $\phi'(0)=1/2$. A zero gate activation does **not** have zero sensitivity. At $a=-2$, $s\approx0.119203$ and $\phi'(a)\approx-0.090784$; replacing the derivative by sigmoid alone would even give the wrong sign. A large positive gate approaches a linear activation; a large negative gate approaches zero from below. These are consequences of the formulas, not extra cases to clamp by hand.

Arrays remain `float`, but the helper receives a widened `double` and evaluates `exp`, the denominator, the value, and the slope in double. Very small exponentials may underflow to zero; that is a permitted finite approximation. Our tiny negative-output tests allow the scale of float subnormal rounding rather than pretending every nonzero real number is representable. Conversely, a stable sigmoid does not prevent the later product $\phi(A)B$, a matrix result, or a gradient from exceeding float range.

## Reverse the product before reversing the projections

Let $L$ be a scalar objective and let $G=\partial L/\partial Y$, shape $[M,D]$, be the incoming gradient. The symbol $dQ$ below means the array of partial derivatives $\partial L/\partial Q$; it is not a perturbation step. The caller has already applied any mean-loss scaling. This sublayer performs no additional division by $M$, $D$, or $F$.

Start with the output projection. For one coordinate,

$$
Y_{ij}=\sum_f H_{if}(W_o)_{jf}.
$$

Holding the other factor fixed and summing incoming contributions yields

$$
dH_{if}=\sum_j G_{ij}(W_o)_{jf},
\qquad
(dW_o)_{jf}=\sum_i G_{ij}H_{if}.
$$

In matrix notation, these are $dH=GW_o$ and $dW_o=G^TH$. The latter sums over token rows because each weight is shared across them.

Now inspect one hidden coordinate $H_{if}=\phi(A_{if})B_{if}$. Its scalar differential is

$$
\delta H_{if}=B_{if}\phi'(A_{if})\,\delta A_{if}
             +\phi(A_{if})\,\delta B_{if}.
$$

Multiplying by the incoming derivative $dH_{if}$ identifies **two** reverse edges:

$$
dA_{if}=dH_{if}B_{if}\phi'(A_{if}),
\qquad
dB_{if}=dH_{if}\phi(A_{if}).
$$

The value branch receives $\phi(A)$, not $B$ and not $\phi'(A)$. The gate branch receives both $B$ and the SiLU slope. No derivative matrix is needed: the gate/value product is coordinatewise, so two loops over logical coordinates suffice.

Finally reverse the two projections:

$$
dW_g=dA^TX,\qquad dW_v=dB^TX,
$$

$$
dX^{(g)}=dAW_g,\qquad dX^{(v)}=dBW_v,
\qquad dX=dX^{(g)}+dX^{(v)}.
$$

The superscripts label contributions, not exponentiation. For example,

$$
\frac{\partial L}{\partial X_{ik}}
=\sum_f dA_{if}(W_g)_{fk}+\sum_f dB_{if}(W_v)_{fk}.
$$

This is where a correct local SiLU derivative can still produce a wrong sublayer: differentiating only the gate branch omits the second sum. A second matrix call with SET rather than ADD instead erases the first sum. Both defects stay within valid memory and can evade address sanitizers.

The complete matrix schedule uses the established kernel interface:

| Result | Multiplication | Flags on stored left/right views | Write behavior |
|---|---|---|---|
| $A$ | $XW_g^T$ | NOTRANS / TRANS | SET |
| $B$ | $XW_v^T$ | NOTRANS / TRANS | SET |
| $Y$ | $HW_o^T$ | NOTRANS / TRANS | SET |
| $dH$ | $GW_o$ | NOTRANS / NOTRANS | SET |
| $dW_o$ | $G^TH$ | TRANS / NOTRANS | caller's weight mode |
| $dW_g$ | $dA^TX$ | TRANS / NOTRANS | caller's weight mode |
| $dW_v$ | $dB^TX$ | TRANS / NOTRANS | caller's weight mode |
| first $dX$ contribution | $dAW_g$ | NOTRANS / NOTRANS | caller's input mode |
| second $dX$ contribution | $dBW_v$ | NOTRANS / NOTRANS | always ADD |

## Calculate a complete row by hand

Use $M=1,D=2,F=2$, let $\ell=\log 2$, and choose

$$
X=(1,2),\quad
W_g=\begin{pmatrix}\ell&0\\1&-1/2\end{pmatrix},\quad
W_v=\begin{pmatrix}2&0\\-1&2\end{pmatrix},
$$

$$
W_o=\begin{pmatrix}1&-1\\1/2&2\end{pmatrix},
\qquad G=(2,-1).
$$

The projection rows give $A=(\ell,0)$ and $B=(2,3)$. Since $e^{-\ell}=1/2$, we have $\sigma(\ell)=2/3$, so

$$
\phi(A)=(2\ell/3,0),\quad
H=(4\ell/3,0),\quad
Y=(4\ell/3,2\ell/3).
$$

Numerically $H_0\approx0.924196$ and $Y_1\approx0.462098$. The second hidden coordinate is zero even though its value-branch activation is three.

The output-projection reverse step is

$$
dH=GW_o=(3/2,-4).
$$

At the first gate, $\sigma'(\ell)=2/9$, so $\phi'(\ell)=2/3+2\ell/9$. At the second gate, the slope is $1/2$. Thus

$$
dA=(2+2\ell/3,-6),\qquad dB=(\ell,0).
$$

The second gate contributes $-6$ despite its **zero forward activation**. In contrast, the second value coordinate receives zero derivative because its multiplier $\phi(0)$ is zero.

Write $a_0=2+2\ell/3\approx2.462098$. The two input contributions are

$$
dX^{(g)}=(a_0\ell-6,3),\qquad dX^{(v)}=(2\ell,0).
$$

Their sum is approximately $(-2.907109,3)$. Keeping only the value branch would incorrectly give $(1.386294,0)$. Keeping only the gate branch would be wrong in the other direction.

The weight derivatives are outer products:

$$
dW_g=\begin{pmatrix}a_0&2a_0\\-6&-12\end{pmatrix},\quad
dW_v=\begin{pmatrix}\ell&2\ell\\0&0\end{pmatrix},
$$

$$
dW_o=\begin{pmatrix}8\ell/3&0\\-4\ell/3&0\end{pmatrix}.
$$

An outer product here means multiplying every coordinate of one vector by every coordinate of the other. With one token row, the sum over rows has just one term. Part 2 checks these arrays directly, then checks non-square multi-row fixtures numerically.

The C fixture stores `(float)log(2.0)`, not the exact real $\ell$, so its expected values use small numerical tolerances. Exact rational reasoning identifies the mechanism; the runnable check accounts for the stored inputs.

## Save values that backward cannot reconstruct from the output

We retain the projection outputs $A$ and $B$ and the hidden product $H$. SiLU's value and slope can both be recomputed from $A$ using a few scalar operations. Storing the sigmoid and SiLU as two more full arrays is unnecessary for this implementation.

Do not overwrite $A$ with $\phi(A)$. Backward uses the preactivation to compute the slope. The activation is not one-to-one on its whole domain, and trying to invert it from a rounded output is the wrong storage contract anyway. Nor can $B$ be recovered safely by dividing $H$ by $\phi(A)$: the denominator may be zero, as in the hand trace. Keeping $H$ also avoids redoing its elementwise product for the output-weight gradient.

| Live value | Backward consumer | Storage decision |
|---|---|---|
| $X$ | $dW_g$, $dW_v$ | Caller retains unchanged input |
| $W_g,W_v$ | both $dX$ branches | No optimizer mutation before backward finishes |
| $W_o$ | $dH$ | Retain matching forward weights |
| $A,B$ | $dA,dB$ | Save two $[M,F]$ arrays |
| $H$ | $dW_o$ | Save one $[M,F]$ array |
| $Y$ | No local derivative formula | Not required by this backward API |
| $dH,dA,dB$ | Local reverse schedule | Three disjoint overwritten scratch arrays |

The saved group occupies $3MF$ floats and the backward scratch group another $3MF$. For the illustrative shape $M=128,F=384$, their combined dense payload is $6\cdot128\cdot384\cdot4=1{,}179{,}648$ bytes, or 1.125 MiB when a float is four bytes. This excludes inputs, outputs, weights, parameter gradients, row padding, and kernel-local scratch; it is not a measured process-memory figure. The module itself performs no heap allocations.

A model owner can later allocate these arrays with the checked tensor routines. This chapter's tests own small fixed arrays with automatic storage duration, so every view stays live until its test function returns and requires no `free`. View-group structs merely copy pointer/geometry fields. Copying one does not copy tensor values, extend an owner's lifetime, or create a second object to destroy.

The current interface deliberately forbids scratch/cache overlap, even where a more carefully scheduled implementation could reuse storage. Saving three arrays costs memory but makes data dependencies observable. Reuse must be a later explicit lifetime change, not an accidental in-place write.

## Expose the groups without inventing an owning layer object

Create the complete header below. `DlFfnWeights` groups three read-only views; `DlFfnCache` groups the three writable destinations of forward. Backward reads that same cache without mutating it. Its C type is not a promise that the pointed-to bytes are immutable elsewhere: the caller must keep them unchanged.

`DlFfnScratch` names temporary derivatives and `DlFfnGrads` names parameter-gradient destinations. Returning or passing these small structs by value copies descriptors, not the arrays. There is no constructor, hidden allocation, global backend, or saved pointer to a previous call.

The backward call takes one mode for $dX$ and a separate mode for **all three** weight-gradient matrices. They can be SET/SET, SET/ADD, ADD/SET, or ADD/ADD. Individual weight families do not have distinct modes in this interface. Scratch always overwrites. A caller accumulating a later microbatch can request ADD for the weight group without requiring the input gradient to accumulate too.

```c title="include/dl/ffn.h"
#ifndef DL_FFN_H
#define DL_FFN_H

#include "dl/kernels.h"

typedef struct {
    DlConstMatrix gate, value, down; /* [F,D], [F,D], [D,F] */
} DlFfnWeights;

typedef struct {
    DlMatrix gate, value, hidden;   /* A, B, H: each [M,F] */
} DlFfnCache;

typedef struct {
    DlMatrix hidden, gate, value;   /* dH, dA, dB: each [M,F] */
} DlFfnScratch;

typedef struct {
    DlMatrix gate, value, down;     /* dWg, dWv, Wo gradient */
} DlFfnGrads;

/* All structs BORROW views; no allocation, transfer, or implicit freeing.
 * Positive M,D,F; canonical live bases/counts; padding is not accessed.
 * Every written span must be disjoint from all inputs and other outputs.
 * Read-only inputs may share storage. No in-place activation/scratch reuse.
 * Invalid geometry/shapes/flags/aliases and preflight non-finites do not write.
 * Later DL_NUMERIC may partially write MULTIPLE outputs; discard all of them.
 * No fallback or ADD retry on damaged gradients. */

/* X [M,D]; overwrite cache and Y [M,D]:
 * A=X Wg^T, B=X Wv^T, H=SiLU(A)*B, Y=H Wo^T.
 * No bias, normalization, residual, dropout, or implicit row averaging. */
DlStatus dl_ffn_forward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                        DlFfnCache cache, DlMatrix y);

/* X, weights, and cache must match a successful, UNCHANGED forward.
 * dy [M,D] is the incoming gradient. Cache is read-only during this call
 * despite its writable view type (used by forward); no saved Y is needed.
 * Always overwrite scratch. dx_mode controls the FIRST dX branch only;
 * the second branch always ADDs. weight_mode applies to all three dW arrays.
 * SET never reads old output. ADD requires initialized finite logical cells.
 * Arrays use float; scalar arithmetic/reductions use double. Each matrix call
 * stores float results; the two dX contributions therefore round separately. */
DlStatus dl_ffn_backward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                         DlFfnCache cache, DlConstMatrix dy,
                         DlFfnScratch scratch, DlMatrix dx,
                         DlWriteMode dx_mode, DlFfnGrads dw,
                         DlWriteMode weight_mode);

#endif
```

The names `gate`, `value`, and `down` refer to roles, not implicit dispatch. `backend` explicitly selects the reference or blocked implementation on every matrix call. The two backends share the existing arithmetic and failure contracts. We do not replace them with a framework call or silently fall back to another backend.

For a first contribution, SET initializes a gradient without reading its old contents; a NaN-filled destination is therefore legal. For ADD, all old logical values must be finite. Within the sublayer the value contribution to $dX$ always adds to the gate contribution, regardless of the caller's initial mode. This is the same edge-accounting principle used for a residual path in Chapter 7.

## Validate the whole call before doing any of its work

A composite operation must not start the gate projection and only later discover that the output gradient has the wrong shape. We preflight every view, all required shapes, the backend/modes, overlaps, and finite logical inputs before writing anything.

The private `check_views` follows the same conservative interval rule as Chapter 7. It computes exclusive bounding ends with `dl_matrix_end`, then checks every output against all inputs and earlier outputs. Comparing canonical bases identifies views of the same allocation. A rebased pointer or a fabricated capacity violates the caller's view contract; this validation cannot reconstruct an allocation's real extent or lifetime. Padding inside a bounding span can make the overlap check conservative, but logical loops still neither read nor write padding.

Create `src/ffn.c` with these helpers. File-local `static` linkage keeps their names private to this translation unit, so similarly named helpers in `norm.c` do not collide. The fixed end arrays are sized for this file's two private call sites, not for an arbitrary public list of views.

```c title="src/ffn.c"
#include "dl/ffn.h"

#include <float.h>
#include <math.h>

/* Private callers have at most eight inputs and seven written views. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                             const DlMatrix *out, size_t no)
{
    size_t ie[8] = {0}, oe[7] = {0};
    for (size_t i = 0; i < ni; ++i) {
        const DlStatus s = dl_matrix_end(in[i], &ie[i]);
        if (s != DL_OK) return s;
    }
    for (size_t o = 0; o < no; ++o) {
        const DlConstMatrix v = dl_matrix_read(out[o]);
        const DlStatus s = dl_matrix_end(v, &oe[o]);
        if (s != DL_OK) return s;
        for (size_t i = 0; i < ni; ++i) {
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

static int ffn_shapes(DlConstMatrix x, DlFfnWeights w, DlFfnCache c)
{
    const size_t m = x.rows, d = x.cols, f = w.gate.rows;
    return shape(w.gate, f, d) && shape(w.value, f, d) && shape(w.down, d, f)
        && shape(dl_matrix_read(c.gate), m, f)
        && shape(dl_matrix_read(c.value), m, f)
        && shape(dl_matrix_read(c.hidden), m, f);
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

static int valid_backend(DlBackend backend)
{
    return backend == DL_BACKEND_REF || backend == DL_BACKEND_BLOCKED;
}

static int valid_mode(DlWriteMode mode)
{
    return mode == DL_SET || mode == DL_ADD;
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

```

`ffn_shapes` uses the gate weight's row count as $F$ and the input shape as $M,D$, then checks all other roles against those dimensions. It does not assume $F=D$ or impose divisibility by a matrix-block size. Geometry validation precedes it, so an apparently correct shape with an overflowing offset calculation is still rejected.

`finite_cells` scans logical coordinates only. `store_float` checks representability **before** narrowing a double result. A later failure is not transactional: previous scalar cells or matrix destinations may already have been written. Underflow to a finite float zero is permitted; nonfinite or too-large results return `DL_NUMERIC`.

### Return a scalar value and its slope together

The `Silu` structure below contains two doubles, not pointers. A C compound literal `(Silu){...}` initializes a temporary structure value, and returning that value copies both numbers to the caller. No local object's address escapes the function and there is no cleanup obligation.

Keeping the value and slope in one helper makes their numerical conventions consistent between forward and backward. Forward only consumes the value; backward consumes both. We do not store one such structure per coordinate.

```c title="src/ffn.c"
/* Both outputs use t=exp(-|a|), never an exponentially large argument.
 * Compute sigmoid's derivative without subtracting a rounded s from one. */
typedef struct {
    double value, slope;
} Silu;

static Silu silu(double a)
{
    const double t = exp(-fabs(a));
    const double den = 1.0 + t;
    const double s = a >= 0.0 ? 1.0 / den : t / den;
    const double ds = t / (den * den);
    return (Silu){a * s, s + a * ds};
}

```

The important evaluation order is `exp(-fabs(a))` rather than `exp(-a)` followed by a recovery branch. Overflow has already happened in the latter approach. The sign branch chooses how to form sigmoid after the bounded exponential is available. `ds` uses that same exponential directly; `s + a*ds` includes both terms of the product rule.

## Implement forward without destroying its saved arrays

The whole forward routine follows. Its four inputs are $X,W_g,W_v,W_o$; its four outputs are $A,B,H,Y$. Checking even the last weight matrix for finite values before projecting $A$ means an invalid input cannot leave a partly refreshed cache.

The first two matrix calls produce independent projections from $X$. The nested loop computes the hidden product. Both scalar operands widen before multiplication; the product is narrowed once into `cache.hidden`. The final matrix call reads that saved hidden matrix and projects back to model width.

```c title="src/ffn.c"
DlStatus dl_ffn_forward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                        DlFfnCache cache, DlMatrix y)
{
    if (!valid_backend(backend)) return DL_ARGUMENT;
    const DlConstMatrix in[4] = {x, w.gate, w.value, w.down};
    const DlMatrix out[4] = {cache.gate, cache.value, cache.hidden, y};
    DlStatus s = check_views(in, 4, out, 4);
    if (s != DL_OK) return s;
    if (!ffn_shapes(x, w, cache) || !shape(dl_matrix_read(y), x.rows, x.cols))
        return DL_SHAPE;
    for (size_t i = 0; i < 4; ++i) {
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    s = dl_matmul(backend, x, DL_NOTRANS, w.gate, DL_TRANS, cache.gate, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, x, DL_NOTRANS, w.value, DL_TRANS, cache.value, DL_SET);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < w.gate.rows; ++j) {
            const double a = (double)cache.gate.base[cache.gate.offset + i * cache.gate.ld + j];
            const double b = (double)cache.value.base[cache.value.offset + i * cache.value.ld + j];
            const Silu z = silu(a);
            s = store_float(&cache.hidden.base[cache.hidden.offset + i * cache.hidden.ld + j],
                            z.value * b);
            if (s != DL_OK) return s;
        }
    }
    return dl_matmul(backend, dl_matrix_read(cache.hidden), DL_NOTRANS,
                      w.down, DL_TRANS, y, DL_SET);
}

```

Each view uses its own `offset` and `ld`. For example, with offset two and a hidden leading dimension of seven, coordinate $(1,3)$ lives at element $2+1\cdot7+3=12$. Using the input's leading dimension for the hidden array would be wrong whenever $D\ne F$ or their padding differs. The padded, non-square test fixtures are chosen to expose exactly that mistake.

On success, the cache corresponds to this invocation's input and weights. The routine does not record a timestamp or generation number. Running another forward into the same cache invalidates the saved state for an older call, even if both calls have the same shapes. Keeping the wrong finite cache can produce wrong gradients without any status error.

## Implement backward as a visible reverse schedule

The backward routine validates eight read-only inputs and seven written destinations. It then computes $dH$, $dW_o$, the coordinatewise $dA,dB$, both projection-weight derivatives, and the two input contributions. Every matrix uses the caller's selected backend.

The cache is treated as read-only through `dl_matrix_read`. Scratch begins uninitialized and is overwritten, so it is not scanned for finite old values. The parameter-gradient group is scanned only for ADD; the input-gradient destination has its own independent test. All of those scans happen before even the first scratch write.

Here is the complete function; append it to the same file.

```c title="src/ffn.c"
DlStatus dl_ffn_backward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                         DlFfnCache cache, DlConstMatrix dy,
                         DlFfnScratch scratch, DlMatrix dx,
                         DlWriteMode dx_mode, DlFfnGrads dw,
                         DlWriteMode weight_mode)
{
    if (!valid_backend(backend) || !valid_mode(dx_mode) || !valid_mode(weight_mode))
        return DL_ARGUMENT;
    const DlConstMatrix in[8] = {x, w.gate, w.value, w.down,
        dl_matrix_read(cache.gate), dl_matrix_read(cache.value),
        dl_matrix_read(cache.hidden), dy};
    const DlMatrix out[7] = {scratch.hidden, scratch.gate, scratch.value,
                             dx, dw.gate, dw.value, dw.down};
    DlStatus s = check_views(in, 8, out, 7);
    if (s != DL_OK) return s;
    const size_t m = x.rows, d = x.cols, f = w.gate.rows;
    if (!ffn_shapes(x, w, cache) || !shape(dy, m, d)
        || !shape(dl_matrix_read(dx), m, d)
        || !shape(dl_matrix_read(scratch.hidden), m, f)
        || !shape(dl_matrix_read(scratch.gate), m, f)
        || !shape(dl_matrix_read(scratch.value), m, f)
        || !shape(dl_matrix_read(dw.gate), f, d)
        || !shape(dl_matrix_read(dw.value), f, d)
        || !shape(dl_matrix_read(dw.down), d, f)) return DL_SHAPE;
    for (size_t i = 0; i < 8; ++i) {
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    if (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) return DL_NUMERIC;
    if (weight_mode == DL_ADD && (!finite_cells(dl_matrix_read(dw.gate))
        || !finite_cells(dl_matrix_read(dw.value))
        || !finite_cells(dl_matrix_read(dw.down)))) return DL_NUMERIC;

    s = dl_matmul(backend, dy, DL_NOTRANS, w.down, DL_NOTRANS,
                   scratch.hidden, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dy, DL_TRANS, dl_matrix_read(cache.hidden), DL_NOTRANS,
                   dw.down, weight_mode);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < f; ++j) {
            const double a = (double)cache.gate.base[cache.gate.offset + i * cache.gate.ld + j];
            const double b = (double)cache.value.base[cache.value.offset + i * cache.value.ld + j];
            const double g = (double)scratch.hidden.base[scratch.hidden.offset + i * scratch.hidden.ld + j];
            const Silu z = silu(a);
            s = store_float(&scratch.gate.base[scratch.gate.offset + i * scratch.gate.ld + j],
                            (g * b) * z.slope);
            if (s != DL_OK) return s;
            s = store_float(&scratch.value.base[scratch.value.offset + i * scratch.value.ld + j],
                            g * z.value);
            if (s != DL_OK) return s;
        }
    }
    s = dl_matmul(backend, dl_matrix_read(scratch.gate), DL_TRANS,
                   x, DL_NOTRANS, dw.gate, weight_mode);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dl_matrix_read(scratch.value), DL_TRANS,
                   x, DL_NOTRANS, dw.value, weight_mode);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dl_matrix_read(scratch.gate), DL_NOTRANS,
                   w.gate, DL_NOTRANS, dx, dx_mode);
    if (s != DL_OK) return s;
    return dl_matmul(backend, dl_matrix_read(scratch.value), DL_NOTRANS,
                      w.value, DL_NOTRANS, dx, DL_ADD);
}
```

Notice the last two calls. The gate path uses `dx_mode`; the value path uses the literal `DL_ADD`. If the caller supplied a previous contribution $R$, the result is

$$
dX=R+dX^{(g)}+dX^{(v)}
$$

for ADD, or just the two new contributions for SET. We never add $R$ separately twice. Each individual matrix call accumulates its reduction in double and writes a float result. The first path therefore rounds before the second call adds its reduction; the implementation does **not** claim a single double accumulation across both paths.

The elementwise part recomputes SiLU from saved $A$ and reads saved $B$. It does not read $H$ to recover either factor. The $dW_o$ calculation is the consumer of saved $H$. This separation makes the lifetime table concrete.

The derivative formulas describe the corresponding smooth real operations evaluated around the stored forward values; they are not derivatives of the discontinuous float-rounding function. Finite-difference steps must be large enough to survive forward rounding and small enough to limit truncation error, as in Chapter 3. Part 2 uses several steps and both absolute and relative tolerances.

## Distinguish invalid calls from numerical failures during execution

A bad shape, range, alias, backend, mode, nonfinite input, or nonfinite old ADD destination is detected before mutation. Invalid geometry can also produce an overflow status. The caller can repair such an invocation without treating a preserved output as newly computed data.

A **later** numerical error is different. Consider $M=D=F=1$, $X=1$, $W_g=\mathrm{FLT\_MAX}$, $W_v=2$, and $W_o=0$. Both projection results fit float, but the hidden product does not. A mathematically simplified final output would be zero, yet this implementation must materialize a finite $H$ for the declared computation. Forward returns `DL_NUMERIC` after writing $A$ and $B$, without producing a valid cache. A zero down weight is not a license to bypass the hidden representation's range contract.

For a backward example, take $X=1,W_g=100,W_v=1,W_o=0$. Forward succeeds with $H=100,Y=0$. Supplying $G=\mathrm{FLT\_MAX}$ gives representable $dH=0$ but an unrepresentable $dW_o=100G$. Backward may therefore have written scratch before rejecting the weight gradient. An unchanged $dX$ does not make the multi-output call successful.

Discard **all** outputs of a failed forward/backward invocation. Do not try the other backend as an ADD retry into partly modified gradients. Retry only after reinitializing destinations and reestablishing a valid matching forward state. The routines allocate no resources internally, so cleanup remains with the owners that supplied the views.

Read-only weights may share physical storage. That permission does not automatically tie their derivative destinations: if two mathematical parameters are intentionally the same parameter, their separate derivative contributions still need an explicit sum outside this API. Supplying overlapping weight-gradient outputs is rejected rather than silently choosing an accumulation order.

The module now contains all three projections, a stable activation, the product rule, and the complete reverse graph. Continue to [Part 2](https://25d-228.github.io/gpt-c/chapter-08-part-2.html) to compile the tests, trace the hand row, and check every weight family and the full residual composition.

## References

The sources below define the architectural forms. The derivatives, numerical rearrangements, storage contracts, fixtures, and C implementation are developed explicitly in this chapter.

- Noam Shazeer, [*GLU Variants Improve Transformer*, version 1][glu], Section 2, equations 5–6: gated activation variants, the bias-free three-matrix form, and the intermediate-width adjustment.
- Prajit Ramachandran, Barret Zoph, and Quoc V. Le, [*Swish: a Self-Gated Activation Function*, version 1][swish], Section 2, equation 1 and the displayed derivative immediately below it: the unit-parameter scalar activation and its derivative.
- Hugo Touvron et al., [*LLaMA: Open and Efficient Foundation Language Models*, version 1][llama], Section 2.2: a decoder architecture using pre-normalization and SwiGLU. Our runnable sublayer tests are not a claim to reproduce the paper's model or performance.

[glu]: https://arxiv.org/html/2002.05202v1#S2
[swish]: https://arxiv.org/html/1710.05941v1#S2
[llama]: https://arxiv.org/html/2302.13971v1#S2.SS2

