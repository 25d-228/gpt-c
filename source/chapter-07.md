# Chapter 7 — RMSNorm and residual gradient paths

A token's hidden state is a row of coordinates, not yet a probability distribution. Its magnitude can change as the row passes through projections. This chapter implements a normalization that rescales the row and then differentiates through the rescaling itself. It also implements the identity path of a residual addition: a second route by which the same input receives a gradient.

These two mechanisms meet in a small, complete graph: normalize a row, apply a learned matrix, and add the original row. The graph is not a full decoder, but it uses the same buffer and gradient rules that the decoder will need.

> **Learning objectives**  Derive scale-only RMSNorm including the denominator's derivative; distinguish row normalization from parameter-gradient reduction; preserve every residual contribution with independent SET/ADD choices; define which forward values must remain unchanged; and verify both isolated operators and a learned residual branch using actual C forward evaluations.

**Prerequisites:** the complete code state from [Chapter 6, Part 1](https://25d-228.github.io/gpt-c/chapter-06.html) and [Part 2](https://25d-228.github.io/gpt-c/chapter-06-part-2.html). Chapter 3 supplies finite differences and floating-point reasoning; Chapter 4 supplies borrowed matrix views; Chapter 5 supplies incoming gradients and accumulation. The scalar and blocked matrix products remain available. No optimizer, automatic differentiation engine, training tape allocator, or downloaded model is needed.

**This chapter has two parts:** this page derives and implements the operators; [Part 2 — Checking normalization and residual graphs](https://25d-228.github.io/gpt-c/chapter-07-part-2.html) supplies the complete tests, learned residual graph, Makefile, and diagnostic experiments. Together they are one complete implementation.

**Implementation estimate:** 3.00 hours: 2.00 for normalization, residual operators, and their integration; 1.00 for numerical traces, checks, and diagnostic variations. Reading and unattended execution are separate. Record actual hands-on time rather than treating the estimate as a minimum.

## Continue the existing workspace

Keep all Chapter 6 files. Create `include/dl/norm.h`, `src/norm.c`, and `tests/test_ops.c`; replace the root `Makefile` with the complete version below. The numbered foundation examples remain separate and unchanged.

The blocks bearing each filename concatenate, in order, into that file. Every new helper and every new test is supplied here. The complete cumulative companion state contains 18 files; fourteen are unchanged from Chapter 6. The matching changed files are [the header](https://25d-228.github.io/gpt-c/media/chapter-07/reference/include/dl/norm.h), [the implementation](https://25d-228.github.io/gpt-c/media/chapter-07/reference/src/norm.c), [the executable tests](https://25d-228.github.io/gpt-c/media/chapter-07/reference/tests/test_ops.c), and [the Makefile](https://25d-228.github.io/gpt-c/media/chapter-07/reference/Makefile). These are companions to the listings, not undisclosed dependencies.

A `DlConstMatrix` is still a borrowed view with a canonical allocation base, a count of live float elements, an element offset, logical rows and columns, and a physical leading dimension `ld`. Coordinate `(i,j)` lives at `base[offset + i*ld + j]`. A writable `DlMatrix` has the same geometry. Neither view owns storage, proves pointer lifetime, or implies contiguous rows without padding.

## Normalize coordinates, not the batch

Let $M$ be the number of token rows and $D$ the number of coordinates per row. A `[B,T,D]` activation can be viewed as `[M,D]` with $M=BT$, as in Chapter 4. Let $x_{ij}$ be coordinate $j$ of row $i$, with $0\le i<M$ and $0\le j<D$. There is one learned scale $\gamma_j$ per coordinate, shared across all rows. We store these scales as a `[1,D]` matrix view rather than inventing a second vector-view API.

Define the row mean square $q_i$, reciprocal denominator $r_i$, unscaled normalized coordinate $z_{ij}$, and output $n_{ij}$:

$$
q_i = \frac{1}{D}\sum_{j=0}^{D-1}x_{ij}^{2},
\qquad r_i = (q_i+\varepsilon)^{-1/2},
$$

$$
z_{ij}=x_{ij}r_i,
\qquad n_{ij}=\gamma_j z_{ij}.
$$

Here $\varepsilon$ is a finite positive constant, not a learned parameter. The decoder configuration uses $10^{-5}$; several checks below deliberately use larger values so the arithmetic can be inspected. The API receives the actual epsilon rather than silently choosing one.

The original [RMSNorm definition, Section 4, equation 4][rmsnorm], uses a root-mean-square statistic instead of subtracting a mean. Our operator is the scale-only, epsilon-stabilized form above: no bias, no centering, and no activation function fused into it. The derivation and C implementation below specify that form completely.

The sum is over **coordinates of one row**. It is not over tokens, sequences, or all allocated floats. Changing a different row cannot change this row's forward result. Padding does not participate. There are no running statistics and no training-versus-inference mode for this operation.

For a nonzero constant row such as $(2,2)$, the normalized coordinates remain nonzero. Subtracting the row mean would instead produce zero coordinates and would implement a different operator. Also, with positive epsilon, even the unscaled row does not have root mean square exactly one:

$$
\frac{1}{D}\sum_j z_{ij}^{2}=\frac{q_i}{q_i+\varepsilon}.
$$

The learned scales can change the magnitude again. Normalization controls an intermediate scale; it does not constrain all later activations to a unit sphere.

For positive $c$, rescaling $x$ by $c$ replaces the effective epsilon by $\varepsilon/c^2$ when the common factor is canceled. Therefore exact positive-scale invariance belongs to the zero-epsilon idealization, not to every row of this stabilized implementation. A negative factor also changes the sign. We will not use an unjustified exact-invariance test as a substitute for checking the derivative.

## Derive the two gradient paths inside RMSNorm

Let $L$ be a scalar objective. The incoming gradient is $g_{ij}=\partial L/\partial n_{ij}$. It already includes whatever scaling the objective uses. For example, Chapter 5's mean cross-entropy has already divided its gradient by the number of token rows. RMSNorm must not divide by that number again.

First hold one row fixed and suppress its row subscript. Since $n_j=\gamma_j z_j$, define $h_j=g_j\gamma_j$, the gradient with respect to $z_j$ before accounting for how its denominator depends on $x$. The direct contribution from $z_j=x_jr$ is $r h_j$. That is only half the calculation: changing any $x_k$ also changes $q$, then $r$, then **every** coordinate $z_j$.

Differentiate the scalar intermediates:

$$
\frac{\partial q}{\partial x_k}=\frac{2x_k}{D},
\qquad
\frac{\partial r}{\partial q}=-\frac{1}{2}(q+\varepsilon)^{-3/2}
                           =-\frac{r^3}{2}.
$$

Thus $\partial r/\partial x_k=-r^3x_k/D$. Applying the product rule gives

$$
\frac{\partial z_j}{\partial x_k}
=r\delta_{jk}-\frac{r^3}{D}x_jx_k.
$$

The symbol $\delta_{jk}$ is one when $j=k$ and zero otherwise. It names the direct path; the second term couples coordinates through their common denominator. Multiplying each output-coordinate derivative by its incoming gradient and summing gives

$$
\frac{\partial L}{\partial x_k}
=r h_k-\frac{r^3x_k}{D}\sum_j h_jx_j.
$$

There is no need to allocate a $D\times D$ derivative matrix. Rewrite the same expression using $z_j=x_jr$ and one scalar reduction $a$:

$$
a=\frac{1}{D}\sum_j h_jz_j,
\qquad
\frac{\partial L}{\partial x_k}=r(h_k-z_k a).
$$

This is the form used in C. One pass computes the row statistic, a second computes `dot = sum(h*z)`, and a third writes every input contribution using `mean_dot = dot/D`. The scales belong **inside** the dot product through $h_j=g_j\gamma_j$. Multiplying the final vector by a scale after computing a dot product of only $g$ and $z$ is generally wrong when the scales differ.

The scale derivative is simpler. Since each $\gamma_j$ is used by all rows,

$$
\frac{\partial L}{\partial\gamma_j}
=\sum_{i=0}^{M-1}g_{ij}z_{ij}.
$$

This is a sum across rows, not an average. It is also not multiplied by $\gamma_j$: the derivative of $\gamma_j z_{ij}$ with respect to $\gamma_j$ is $z_{ij}$. A zero scale can have a nonzero gradient even though its current output coordinate is zero. The original paper's [Section 4.2, equations 8–9][rmsnorm] analyzes gain and normalization derivatives; here the chain rule has been expanded for our explicit epsilon and array interfaces.

### Epsilon changes the derivative, not only the zero case

At $x=0$, $r=1/\sqrt{\varepsilon}$ and $z=0$. The output and scale gradient are zero, but the input gradient is

$$
\frac{\partial L}{\partial x_k}
=\frac{g_k\gamma_k}{\sqrt{\varepsilon}}.
$$

Returning zero input gradients for a zero row loses a real sensitivity. Choosing a very small positive epsilon can make that sensitivity large; positive epsilon is not a promise that every float gradient is representable.

As another check, let $S=\sum_jh_jx_j$. The dot product of the input with its gradient is

$$
\sum_k x_k\frac{\partial L}{\partial x_k}
=rS-r^3qS=\varepsilon r^3S.
$$

It is not generally zero. In width one, the derivative reduces to $g\gamma\varepsilon/(x^2+\varepsilon)^{3/2}$, which is nonzero for positive epsilon. This also explains why a test based on zero-epsilon scale invariance would reject a correct stabilized derivative. In floating-point arithmetic, cancellation can obscure a sufficiently tiny radial term; our exact numerical fixtures do not demand relative accuracy for a quantity below the rounding scale.

## Work a two-coordinate row by hand

Use $x=(3,4)$, $\gamma=(2,-1)$, $g=(1,-2)$, and $\varepsilon=3.5$. The deliberately large epsilon makes the square root exact:

$$
q=(9+16)/2=12.5,\quad q+\varepsilon=16,\quad r=1/4.
$$

The forward result is $z=(3/4,1)$ and $n=(3/2,-1)$. The incoming gradient may be obtained from the simple objective $L=n_0-2n_1$; no probability distribution is required to test a local derivative.

For backward, $h=g\odot\gamma=(2,2)$, where $\odot$ means coordinatewise multiplication. The shared reduction is $h\cdot z=3.5$, hence $a=1.75$. Now write both input coordinates:

$$
\frac{\partial L}{\partial x_0}
=\frac14\left(2-\frac34\frac74\right)=\frac{11}{64}=0.171875,
$$

$$
\frac{\partial L}{\partial x_1}
=\frac14\left(2-1\frac74\right)=\frac{1}{16}=0.0625.
$$

The scale contribution is $(3/4,-2)$. Omitting the denominator derivative would give the plausible-looking but wrong input gradient $(0.5,0.5)$. No memory error is needed to cause that failure.

The executable fixture adds a second row $x=(-3,4)$ with incoming gradient $(0.5,1)$ and the same scales and epsilon. Its input gradient is $(11/128,-1/32)$ and its scale contribution is $(-3/8,1)$. Therefore the shared scale gradient from both rows is

$$
(3/4,-2)+(-3/8,1)=(3/8,-1).
$$

Assigning the second row's contribution over the first yields $(-3/8,1)$ instead. Dividing the correct sum by two also changes the objective's gradient. The fixture will detect both mistakes.

## A residual addition creates two reverse edges

For $Y=X+B$, each output coordinate is the sum of one coordinate from each operand. The derivative with respect to either operand is one. An incoming matrix gradient $G$ therefore contributes $G$ to the gradient of $X$ and another $G$ to the gradient of $B$. There is no division by two.

The important case is when the branch itself depends on $X$:

$$
N=\operatorname{RMSNorm}(X,\gamma),\qquad
B=NW^{\mathsf T},\qquad Y=X+B.
$$

For this demonstration $X,N,B,Y$ are `[2,3]`, and $W$ is `[3,3]` in the established `[output,input]` layout. A pre-normalized residual branch leaves the identity operand $X$ unnormalized; it is not $\operatorname{RMSNorm}(X+B)$. [LLaMA Section 2.2][llama] identifies normalization before transformer sublayers as its architectural choice. Attention and the full feed-forward network will later replace the small matrix branch; the addition rule remains the same.

Reverse the matrix branch using Chapter 5's derivatives:

$$
\frac{\partial L}{\partial N}=GW,\qquad
\frac{\partial L}{\partial W}=G^{\mathsf T}N.
$$

The input gradient has two contributions: $G$ from the identity edge, and the RMSNorm input derivative with incoming gradient $GW$. The program must merge them. Writing the branch contribution with SET after already copying $G$ erases the identity edge.

Even the simpler learned-scale branch $Y=X+N$ demonstrates the sum. For the first hand-worked row, the total input gradient is $(1,-2)+(11/64,1/16)=(75/64,-31/16)$, or $(1.171875,-1.9375)$.

For $Y=X+X$, the same logical variable occurs as both operands. The result is $2X$, so its gradient is $2G$. Calling one reverse-edge operation with SET and the other with ADD gives that result. Calling SET twice gives only $G$. We implement an explicit **one-edge** reverse operation so these cases use the same scheduling rule rather than a special aliasing trick.

## Retain values, not assumptions about values

This implementation recomputes each row's reciprocal RMS during backward from the original input. It keeps no hidden allocation or stored-statistics pointer. The extra row scan is linear in the number of activation coordinates. It avoids adding a cache representation and cache lifetime to this first normalization interface.

| Value | Why it remains available | Mutation boundary |
|---|---|---|
| Original `x` | Reconstructs the row statistic and normalized coordinates in backward | Do not overwrite before its last backward consumer finishes |
| `scale` and the epsilon value | Determine the derivative of the same forward operation | Do not update parameters or change epsilon between matching calls |
| Normalized `n` | Needed by the learned branch's weight gradient, though not by RMSNorm backward itself | Keep until `dW = G^T N` finishes |
| Branch weight `w` | Needed by `dN = G W` | Keep the forward weight until its backward consumers finish |
| Incoming gradients | Read while producing local contributions | Do not overwrite through aliased destinations |
| `dx`, `dscale`, `dw` | Collect contributions according to the graph schedule | Initialize once with SET or clear once, then add remaining contributions |

The C functions borrow these buffers. Their caller owns them and is responsible for cleanup. The test graph below owns inline arrays in one stack object; they live for the entire forward/backward/check sequence. The later decoder tape will own longer-lived allocations. Nothing here allocates memory inside an operator, and there is nothing for the operator to free.

Saving only the output is insufficient for this interface. A zero scale destroys information in an output coordinate, and the row magnitude is not generally recoverable from normalized outputs. Reusing `x` as forward output would also destroy a later backward input. We therefore reject output/input overlap, including exact in-place views, rather than silently permit an optimization that changes the lifetime contract.

A stateless routine cannot detect that a caller supplied a different but finite `x`, scale, or epsilon than it used in forward. That is a graph-lifetime bug, not a malformed matrix shape. The matching values must remain unchanged, and numerical graph checks must validate the composition.

### Write modes and failure behavior

`dx` and `dscale` each have their own write mode. SET replaces prior logical values without reading them. ADD requires initialized finite old values and preserves their contributions. A common call uses `dx=ADD` because the residual identity edge is already present, but `dscale=SET` because this is the first contribution to that scale parameter.

Geometry, shapes, flags, epsilon, overlap, all input logical values, and old ADD destinations are checked before mutation. These preflight failures leave all outputs unchanged. Numerical overflow while computing or storing a later result can leave earlier writes in either output. A failed backward call invalidates **both** destinations, even when one appears untouched or contains useful-looking numbers. Do not retry ADD on the damaged buffers.

Row statistics, the row dot product, and individual expressions use double. Activations, parameters, and gradients remain float arrays. Input gradients are rounded once per destination update. Scale gradients, like embedding scatter-add, are rounded into float after each row contribution. They are not an entire-batch double reduction. Summation order can affect their last bits, and an intermediate scale-gradient update can overflow even if later cancellation would have reduced the final mathematical sum. This is an explicit reference contract, not a claim of exact arithmetic.

## Declare the operator interface

Create the following complete header. Including `dl/kernels.h` reuses `DlWriteMode`, the matrix views, and status declarations already taught; it does not call a hidden normalization implementation.

```c title="include/dl/norm.h"
#ifndef DL_NORM_H
#define DL_NORM_H

#include "dl/kernels.h"

/* All views use canonical bases/counts and truthful live storage.
 * Logical cells must be finite; padding is neither read nor written.
 * Every output span must be disjoint from all input spans and other outputs.
 * Inputs may share storage. No allocation or ownership transfer occurs.
 * Structural and preflight non-finite errors leave all outputs unchanged.
 * Later DL_NUMERIC may leave partial writes: discard ALL outputs of the call.
 * SET never reads old output. ADD requires initialized finite logical cells. */

/* x [M,D], scale [1,D], y [M,D]; finite epsilon > 0, inside sqrt.
 * y[i,j] = scale[j] * x[i,j] / sqrt(mean_j(x[i,j]^2) + epsilon).
 * No bias, mean subtraction, row mixing, or stored statistics; overwrite y. */
DlStatus dl_rmsnorm_forward(DlConstMatrix x, DlConstMatrix scale,
                            float epsilon, DlMatrix y);

/* x and scale are the UNCHANGED inputs of the matching successful forward;
 * epsilon must match it. dy [M,D] is the incoming gradient.
 * dx [M,D] and dscale [1,D] have independently selected SET/ADD modes.
 * Recompute row statistics from x; y is not needed. Keep x/scale live until
 * backward finishes. No averaging over M: dy already contains loss scaling.
 * Row reductions use double. dscale adds one row at a time into FP32 storage,
 * like embedding scatter-add; it rounds after each row, not once per call. */
DlStatus dl_rmsnorm_backward(DlConstMatrix x, DlConstMatrix scale,
                             DlConstMatrix dy, float epsilon,
                             DlMatrix dx, DlWriteMode dx_mode,
                             DlMatrix dscale, DlWriteMode scale_mode);

/* Same positive shape for both inputs and output; overwrite out = a + b.
 * No in-place output, even if one input is no longer needed by the caller. */
DlStatus dl_residual_add(DlConstMatrix a, DlConstMatrix b, DlMatrix out);

/* Reverse ONE edge of a residual addition: dst = dy or dst += dy.
 * Call once per operand edge. For y = x + x, call twice into the SAME dx:
 * first SET, then ADD (or ADD twice when dx already has contributions).
 * dy and dst cannot overlap, including an exactly identical view. */
DlStatus dl_residual_backward(DlConstMatrix dy, DlMatrix dst,
                              DlWriteMode mode);

#endif
```

The shapes in the comments distinguish the per-row activation from the scale shared by every row. Each backward mode applies to its named destination. The residual backward function processes one operand edge, so its caller decides whether another edge has already written that variable's gradient.

## Check spans before reading or writing cells

Start `src/norm.c` with the includes and geometry helper. Its only callers are the four public functions below, which supply at most three inputs and two outputs. The fixed internal arrays are therefore sufficient; this is not an exposed variable-arity tensor API.

```c title="src/norm.c"
#include "dl/norm.h"

#include <float.h>
#include <math.h>

/* Internal callers supply at most three inputs and two outputs. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                             const DlMatrix *out, size_t no)
{
    size_t ie[3] = {0}, oe[2] = {0};
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

```

`dl_matrix_end` checks the declared bounds using Chapter 4's overflow-safe arithmetic and returns an exclusive end. After that validation, the two inequalities test intersection of half-open spans. Pointer equality is used only to identify the same canonical allocation base; the code does not order unrelated pointers or infer a real allocation size from an address.

The test is conservative: it covers gaps between rows as well as logical cells. Two outputs whose bounding spans overlap are rejected even when a clever placement could make their individual cells disjoint. Distinct inputs may share storage because they are read-only. Distinct, nonoverlapping spans inside one canonical allocation are allowed.

These checks cannot detect fabricated bases or expired allocations. Keep the Chapter 4 rule: every view carries the owner's actual base/count, and the owner remains live. The next helpers check shape equality, finite logical cells, valid modes, and a single result store.

```c title="src/norm.c"
static int same_shape(DlConstMatrix a, DlConstMatrix b)
{
    return a.rows == b.rows && a.cols == b.cols;
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

static int valid_mode(DlWriteMode mode)
{
    return mode == DL_SET || mode == DL_ADD;
}

static DlStatus put_value(float *out, double value, DlWriteMode mode)
{
    if (mode == DL_ADD) value += (double)*out;
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

```

`finite_cells` never inspects padding. `put_value` reads the old destination only for ADD. Its range check happens **before** narrowing a double to float. Underflow to a representable tiny value or zero is accepted; a non-finite or out-of-range result is not.

These helpers are internal `static` functions, as in the earlier modules. They do not introduce externally visible symbols or shared mutable state. The public functions validate modes before the helper sees them, so `put_value` need not duplicate the enum check on every coordinate.

## Compute the reciprocal RMS and forward values

Append the row helper and forward implementation. The row sum squares already-widened operands: `v` is double before `v*v` is evaluated.

```c title="src/norm.c"
static double inverse_rms(DlConstMatrix x, size_t i, float epsilon)
{
    const size_t row = x.offset + i * x.ld;
    double sum = 0.0;
    for (size_t j = 0; j < x.cols; ++j) {
        const double v = (double)x.base[row + j];
        sum += v * v;
    }
    return 1.0 / sqrt(sum / (double)x.cols + (double)epsilon);
}

DlStatus dl_rmsnorm_forward(DlConstMatrix x, DlConstMatrix scale,
                            float epsilon, DlMatrix y)
{
    if (!isfinite(epsilon) || epsilon <= 0.0f) return DL_ARGUMENT;
    const DlConstMatrix in[2] = {x, scale};
    DlStatus s = check_views(in, 2, &y, 1);
    if (s != DL_OK) return s;
    if (!same_shape(x, dl_matrix_read(y)) || scale.rows != 1 ||
        scale.cols != x.cols) return DL_SHAPE;
    if (!finite_cells(x) || !finite_cells(scale)) return DL_NUMERIC;
    for (size_t i = 0; i < x.rows; ++i) {
        const double r = inverse_rms(x, i, epsilon);
        if (!isfinite(r) || r <= 0.0) return DL_NUMERIC;
        for (size_t j = 0; j < x.cols; ++j) {
            const double z = (double)x.base[x.offset + i * x.ld + j] * r;
            const double value = z * (double)scale.base[scale.offset + j];
            s = put_value(&y.base[y.offset + i * y.ld + j], value, DL_SET);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}

```

Do not replace `v*v` with a float multiplication and widen its result afterward. For example, `FLT_MAX * FLT_MAX` is far outside the float range even though normalizing the two-coordinate row `(FLT_MAX,0)` has a modest result near `(sqrt(2),0)`. On the target float/double representation, widening before the square gives the reference calculation room for that intermediate.

The mean divides by `x.cols`, not `x.rows`, and epsilon is added before `sqrt`. The final expression first forms the normalized coordinate `z`, then multiplies by the learned scale. A scale is allowed to be zero or negative. Only non-finite scales are rejected at preflight; a finite scale that causes an unrepresentable output is detected at the store.

The output is always overwritten. Filling it with NaNs before a successful call is therefore legal and helps catch an accidental dependency on old output. All input cells are preflighted before any row is written, so a NaN in the last input row cannot leave earlier output rows modified.

## Reduce the denominator term once per row

Append the complete backward routine. Its setup validates both outputs together, then zeros the logical scale gradient exactly once when `scale_mode` is SET.

```c title="src/norm.c"
DlStatus dl_rmsnorm_backward(DlConstMatrix x, DlConstMatrix scale,
                             DlConstMatrix dy, float epsilon,
                             DlMatrix dx, DlWriteMode dx_mode,
                             DlMatrix dscale, DlWriteMode scale_mode)
{
    if (!isfinite(epsilon) || epsilon <= 0.0f ||
        !valid_mode(dx_mode) || !valid_mode(scale_mode)) return DL_ARGUMENT;
    const DlConstMatrix in[3] = {x, scale, dy};
    const DlMatrix out[2] = {dx, dscale};
    DlStatus s = check_views(in, 3, out, 2);
    if (s != DL_OK) return s;
    if (scale.rows != 1 || scale.cols != x.cols || !same_shape(x, dy) ||
        !same_shape(x, dl_matrix_read(dx)) ||
        !same_shape(scale, dl_matrix_read(dscale))) return DL_SHAPE;
    if (!finite_cells(x) || !finite_cells(scale) || !finite_cells(dy) ||
        (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) ||
        (scale_mode == DL_ADD && !finite_cells(dl_matrix_read(dscale)))) {
        return DL_NUMERIC;
    }
    if (scale_mode == DL_SET) {
        for (size_t j = 0; j < x.cols; ++j) dscale.base[dscale.offset + j] = 0.0f;
    }
    for (size_t i = 0; i < x.rows; ++i) {
        const double r = inverse_rms(x, i, epsilon);
        if (!isfinite(r) || r <= 0.0) return DL_NUMERIC;
        const size_t xr = x.offset + i * x.ld;
        const size_t gr = dy.offset + i * dy.ld;
        double dot = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            const double z = (double)x.base[xr + j] * r;
            const double h = (double)dy.base[gr + j]
                           * (double)scale.base[scale.offset + j];
            dot += h * z;
        }
        const double mean_dot = dot / (double)x.cols;
        for (size_t j = 0; j < x.cols; ++j) {
            const double z = (double)x.base[xr + j] * r;
            const double g = (double)dy.base[gr + j];
            const double h = g * (double)scale.base[scale.offset + j];
            const double contribution = r * (h - z * mean_dot);
            s = put_value(&dx.base[dx.offset + i * dx.ld + j],
                          contribution, dx_mode);
            if (s != DL_OK) return s;
            s = put_value(&dscale.base[dscale.offset + j], g * z, DL_ADD);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}

```

Trace the first row of the hand example through these loops. `inverse_rms` returns `0.25`. The first inner loop reads the two incoming values and the two scales, producing `h=2` twice. It accumulates `dot=2*0.75 + 2*1 = 3.5`; `mean_dot` becomes `1.75`. The next inner loop stores `0.25*(2-0.75*1.75)` and `0.25*(2-1*1.75)` in the two logical `dx` cells.

The same loop updates the two `dscale` cells with `g*z`. Its internal store always uses ADD, even when the public `scale_mode` was SET: the SET action already cleared those cells **once**, before the row loop. The second row must add to the first row's contribution. In contrast, each logical `dx` cell belongs to exactly one row, so the caller's `dx_mode` can be used directly for each store.

There is no saved `z` array. The code recomputes `z=x*r` inside the two inner loops using the same unchanged row and reciprocal. This costs extra multiplications but not another row-statistic calculation. It also avoids relying on rounded forward outputs or dividing by a possibly zero scale.

The loops are $O(MD)$ with constant additional storage, excluding caller-owned arrays. They intentionally prioritize a transparent reference over vectorization or kernel fusion. Structural validation and finite scans are part of the call's work; they have not been hidden outside a favorable timing boundary.

## Implement the two residual operations

Finish `src/norm.c` with these two complete functions. The forward call writes `a+b` into a disjoint output. The reverse-edge call copies or adds the incoming gradient into one destination.

```c title="src/norm.c"
DlStatus dl_residual_add(DlConstMatrix a, DlConstMatrix b, DlMatrix out)
{
    const DlConstMatrix in[2] = {a, b};
    DlStatus s = check_views(in, 2, &out, 1);
    if (s != DL_OK) return s;
    if (!same_shape(a, b) || !same_shape(a, dl_matrix_read(out))) return DL_SHAPE;
    if (!finite_cells(a) || !finite_cells(b)) return DL_NUMERIC;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            const double value = (double)a.base[a.offset + i * a.ld + j]
                               + (double)b.base[b.offset + i * b.ld + j];
            s = put_value(&out.base[out.offset + i * out.ld + j], value, DL_SET);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}

DlStatus dl_residual_backward(DlConstMatrix dy, DlMatrix dst, DlWriteMode mode)
{
    if (!valid_mode(mode)) return DL_ARGUMENT;
    DlStatus s = check_views(&dy, 1, &dst, 1);
    if (s != DL_OK) return s;
    if (!same_shape(dy, dl_matrix_read(dst))) return DL_SHAPE;
    if (!finite_cells(dy) ||
        (mode == DL_ADD && !finite_cells(dl_matrix_read(dst)))) return DL_NUMERIC;
    for (size_t i = 0; i < dy.rows; ++i) {
        for (size_t j = 0; j < dy.cols; ++j) {
            s = put_value(&dst.base[dst.offset + i * dst.ld + j],
                (double)dy.base[dy.offset + i * dy.ld + j], mode);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}
```

Neither operation needs the original residual operands for its own derivative: the addition derivative is one regardless of their values. Those operands may still be required by their producing operations' backward passes, which is why this chapter does not add an in-place special case.

For a branch with two distinct operands, call the reverse-edge operation once for each operand's gradient. For two edges that reach the same variable, reuse the destination but not the incoming buffer. The first edge can SET and the second ADD. The incoming gradient remains an independent read-only value throughout.

## Continue with the executable checks

The operators are complete, but their runnable test program and build integration belong to the same chapter. Continue to [Part 2 — Checking normalization and residual graphs](https://25d-228.github.io/gpt-c/chapter-07-part-2.html) for all fixtures, the connected learned branch, the 189 finite-difference comparisons, and the complete Makefile. No future chapter is needed to run this implementation.

## References

These primary sources support the named normalization and architectural mechanisms. The epsilon-specific derivation, memory contracts, and C routines above are provided directly rather than delegated to an external implementation.

- Biao Zhang and Rico Sennrich, [*Root Mean Square Layer Normalization*, version 1][rmsnorm], Section 4, equation 4, for the RMS statistic; Section 4.2, equations 8–9, for gain and normalization gradient analysis. The implementation here adds explicit positive epsilon and uses scale only.
- Hugo Touvron et al., [*LLaMA: Open and Efficient Foundation Language Models*, version 1][llama], Section 2.2, for RMSNorm before transformer sublayers. The small matrix branch in this chapter is a diagnostic composition, not a claimed reproduction of a whole transformer block.

[rmsnorm]: https://arxiv.org/html/1910.07467v1#S4
[llama]: https://arxiv.org/html/2302.13971v1#S2.SS2

