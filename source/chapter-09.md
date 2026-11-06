# Chapter 9 — Rotary positions as explicit coordinate transformations

An embedding table gives a token a vector. A linear projection gives that vector another set of coordinates. Neither operation, by itself, tells a query whether a matching key occurred one place earlier or a hundred places earlier. Rotary position embedding, usually called **RoPE**, inserts a position-dependent transformation between those projections and their dot product. It does not look up another learned embedding table or add a position vector to the token representation.

We will implement that transformation as pairs of ordinary scalar operations. The difficult details are not hidden in a matrix library: which coordinates form a pair, how the angle is constructed, which absolute position belongs to a row, and why backward rotates in the opposite direction. Each of these can be wrong while the output still looks numerically reasonable.

**Prerequisites:** retain the complete Chapter 8 workspace, including its Makefile, headers, source modules, tests, and comparison application. The latest prerequisite snapshot is [Chapter 8's companion directory](https://25d-228.github.io/gpt-c/media/chapter-08/reference/Makefile). Chapter 3 supplies the floating-point and finite-difference reasoning; Chapters 4–6 supply checked matrix views and SET/ADD semantics; Chapters 7–8 establish incoming gradients and branch accumulation. No attention implementation, model weights, Python model, or future code is needed here.

**Implementation estimate:** **2.50 hours** across both parts: 1.50 for frequency/rotation routines and 1.00 for position, layout, and derivative checks. Reading is separate. This part develops all new operators; [Part 2 — Checking coordinate conventions and absolute positions](https://25d-228.github.io/gpt-c/chapter-09-part-2.html) supplies the complete executable tests and replacement Makefile. They form one chapter.

> **Learning objectives**
> Derive a rotation and its transpose; turn absolute positions into relative-angle dot products; distinguish adjacent from split-half coordinates; implement per-head forward/backward loops with explicit positions; and recognize what norm preservation and gradient checks cannot establish on their own.

## Start with two coordinates, not a positional-encoding formula

Let $u$ and $v$ be the two coordinates of a vector, written as a column for the algebra. Rotating its first unit basis vector counterclockwise by angle $\alpha$ gives $(\cos\alpha,\sin\alpha)$. Rotating the second gives $(-\sin\alpha,\cos\alpha)$. A vector is a weighted sum of these basis vectors, so its rotated coordinates are

$$
\begin{pmatrix}u'\\v'\end{pmatrix}
=R(\alpha)\begin{pmatrix}u\\v\end{pmatrix},
\qquad
R(\alpha)=
\begin{pmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{pmatrix}.
$$

In scalar form, with $c=\cos\alpha$ and $s=\sin\alpha$,

$$
u'=cu-sv,\qquad v'=su+cv.
$$

The first output uses **both** old inputs. The second must also use both old inputs, not an already-overwritten first output. Our public API will require disjoint input/output spans, but loading both operands before either store still makes the calculation clear.

Angles are measured in **radians**. A full turn is $2\pi$ radians; a quarter-turn is $\pi/2$. At a quarter-turn, $c=0,s=1$, so $(3,4)$ becomes $(-4,3)$. The squared length is $3^2+4^2=25$ before and $(-4)^2+3^2=25$ afterward.

That preservation is not a coincidence. Expand the two output squares:

$$
(cu-sv)^2+(su+cv)^2
=(c^2+s^2)(u^2+v^2)=u^2+v^2.
$$

The two mixed terms cancel. This is a useful numerical invariant, but it does **not** identify the correct coordinates to rotate together. Rotating the wrong pair also preserves its length.

The [RoFormer paper, Sections 3.1–3.2][rope], introduces positional information through rotations of projected queries and keys. Our development below makes the coordinate convention, storage, and differentiation explicit rather than constructing a dense rotation matrix.

## Why absolute rotations produce a relative angle

A **query** $q$ and **key** $k$ are projected feature vectors whose dot product will later help determine an attention weight. For now, treat them as two supplied vectors. Their projection matrices and the attention normalization are not part of this chapter's operator.

Let $m$ be the query's absolute token position, $n$ the key's absolute position, and $\omega$ a fixed angular frequency in radians per position. Rotate them separately:

$$
\widetilde q=R(m\omega)q,\qquad
\widetilde k=R(n\omega)k.
$$

Transposing the rotation changes the sign of its sine terms. The angle-addition identities for sine and cosine then give

$$
R(a)^{\mathsf T}=R(-a),\qquad
R(-a)R(b)=R(b-a).
$$

Consequently,

$$
\widetilde q^{\mathsf T}\widetilde k
=q^{\mathsf T}R(m\omega)^{\mathsf T}R(n\omega)k
=q^{\mathsf T}R((n-m)\omega)k.
$$

With our column-vector convention the remaining angle is **key position minus query position**. To see the sign without relying on matrix notation, put $\delta=(n-m)\omega$ and expand the last expression:

$$
q^{\mathsf T}R(\delta)k
=(q_0k_0+q_1k_1)\cos\delta
 +(q_1k_0-q_0k_1)\sin\delta.
$$

For $q=(1,0)$, $k=(0,1)$, and $\delta=\pi/2$, the result is $-1$. Reversing the relative angle would give $+1$. The score is not simply the original dot product multiplied by a cosine; the cross-coordinate sine term matters too.

Adding the same offset $a$ to both positions preserves $n-m$, so it preserves this dot product in exact arithmetic. Moving only one position generally changes it. These are tests on **fixed supplied query/key contents**. They do not establish that every state of a complete multilayer decoder is unchanged under an arbitrary alteration of its input, mask, cache, or context. They also do not say that every individual score decreases monotonically with distance: the expression oscillates and depends on the vectors.

We rotate queries and keys. We do **not** rotate the value vectors that will be averaged by attention. A caller must distinguish those roles; a generic matrix view cannot infer whether its bytes represent a query, key, or value.

## Several coordinate planes inside one head

A head is one independently scored query/key subvector. Let its width be an even positive integer $d$. Define $r=d/2$, the number of two-coordinate planes. For each plane index $j$ with $0\leq j<r$, use

$$
\omega_j=\theta^{-2j/d}=\theta^{-j/r},
\qquad
\alpha_{p,j}=p\omega_j.
$$

Here $\theta>1$ is the **frequency base**, not an angle, token position, or sequence length; $p$ is a nonnegative absolute position. The name `inv_freq` used in code refers to the reciprocal-power schedule $1/\theta^{j/r}$. It is not the inverse of a rotation. The first frequency is one; later planes rotate more slowly.

For $d=8$ and $\theta=10000$, the four frequencies are

$$
[1,\ 0.1,\ 0.01,\ 0.001].
$$

At position $p=3$, their angles are $[3,0.3,0.03,0.003]$ radians. The final frequency is not $1/\theta$: the largest plane index is $r-1$, not $r$. At position zero every angle is zero regardless of the base.

The constructor will calculate this small frequency vector once. Forward/backward read it; they do not recalculate powers for every head or retain a table for every possible position. The frequency formula uses **head width** $d$, not the total projection width and not the current sequence length. A one-token decoding call must use the same frequencies as a full-prefix call.

The selected [SmolLM2-135M configuration][config] has hidden width 576, nine query heads, three key/value heads, `rope_theta = 100000`, `rope_interleaved = false`, and no RoPE scaling. Thus its head width is $576/9=64$. Each head uses 32 frequencies; the last is approximately $1.43301257023696\times10^{-5}$. Its configured position count is 8192. These are model-compatibility settings, not requirements that all our small test fixtures have those dimensions.

We implement full-width, unscaled RoPE. No learned frequencies, partial rotary dimension, context-extension scheme, or derivative with respect to the integer position is hidden in the routine. The frequency constructor requires a finite base greater than one. The rotation routine accepts a caller-supplied positive frequency vector with each value at most one, which also permits carefully chosen diagnostic angles.

## Split-half and adjacent pairing are different storage conventions

The two-dimensional derivation has not yet said which coordinates of a stored head are the two partners. Two valid conventions are common:

| Plane $j$ for $d=8$ | Adjacent coordinates | Split-half coordinates |
|---|---|---|
| 0 | 0 and 1 | 0 and 4 |
| 1 | 2 and 3 | 1 and 5 |
| 2 | 4 and 5 | 2 and 6 |
| 3 | 6 and 7 | 3 and 7 |

We use **split-half**, matching the non-interleaved setting of the selected checkpoint. For a stored head $x$,

$$
y_j=c_jx_j-s_jx_{j+r},\qquad
y_{j+r}=s_jx_j+c_jx_{j+r},\qquad 0\leq j<r.
$$

The frequency index $j$ appears once in each half. Splitting the whole projected row in half would be wrong when the row contains multiple heads: the split belongs **inside each head**.

Adjacent and split-half forms describe equivalent rotation families after a consistent permutation of coordinates. For example, adjacent order $[a_0,b_0,a_1,b_1]$ corresponds to split-half order $[a_0,a_1,b_0,b_1]$. If a permutation matrix $P$ maps split-half vectors to adjacent vectors, the corresponding rotation matrices satisfy

$$
R_{\mathrm{split}}=P^{\mathsf T}R_{\mathrm{adjacent}}P.
$$

This does not license changing a checkpoint's pairing convention without changing its projected coordinates. A compatible conversion would have to account for the associated query/key projection output rows, head by head. We are not performing such a conversion here. We keep the selected convention fixed and will validate the actual tensor mapping when importing weights.

A norm test cannot distinguish these conventions. Neither can finite differences when both forward and backward implement the **same wrong convention**. An independently calculated coordinate example is essential.

## A complete four-coordinate trace

Choose one head of width four, stored as

$$
x=[3,1,4,2].
$$

For this diagnostic only, supply $\omega=[\pi/4,\pi/8]$ and position $p=2$. Both frequencies satisfy our rotation API's bounds. This is an intentionally chosen test schedule, **not** the geometric model schedule from the frequency constructor. The resulting angles are $[\pi/2,\pi/4]$.

Plane zero uses coordinates 0 and 2: $(3,4)$ becomes $(-4,3)$. Plane one uses coordinates 1 and 3: with $c=s=1/\sqrt2$,

$$
(1,2)\longmapsto
\left(\frac{1-2}{\sqrt2},\frac{1+2}{\sqrt2}\right)
=\left(-\frac1{\sqrt2},\frac3{\sqrt2}\right).
$$

Scatter each pair back into its original coordinate slots:

$$
y=\left[-4,-\frac1{\sqrt2},3,\frac3{\sqrt2}\right]
\approx[-4,-0.707107,3,2.121320].
$$

The squared length is 30 before and after. An adjacent-pair implementation would rotate $(3,1)$ first and produce a first coordinate of $-1$, immediately exposing the layout error. Printing the vector in **storage order**, not pair visitation order, makes this test unambiguous.

## Backward uses the transpose, not a second forward rotation

Let $L$ be a scalar objective. Let $g_u=\partial L/\partial u'$ and $g_v=\partial L/\partial v'$ be incoming gradients. Positions and frequencies are fixed for this derivative. The chain rule gives

$$
\frac{\partial L}{\partial u}
=g_u\frac{\partial u'}{\partial u}
 +g_v\frac{\partial v'}{\partial u}
=cg_u+sg_v,
$$

$$
\frac{\partial L}{\partial v}
=g_u\frac{\partial u'}{\partial v}
 +g_v\frac{\partial v'}{\partial v}
=-sg_u+cg_v.
$$

Therefore the input-gradient pair is $R(\alpha)^{\mathsf T}g=R(-\alpha)g$. Only the sine sign changes; cosine is even. No division by head width, token count, or number of heads appears. Any loss scaling is already present in the incoming gradient.

For the four-coordinate trace, take incoming gradient $g=[2,-1,3,4]$. The first pair is $(g_0,g_2)=(2,3)$, giving $(3,-2)$ under the transpose quarter-turn. The second is $(g_1,g_3)=(-1,4)$, giving $(3/\sqrt2,5/\sqrt2)$. In storage order,

$$
\frac{\partial L}{\partial x}
=\left[3,\frac3{\sqrt2},-2,\frac5{\sqrt2}\right].
$$

Unlike RMSNorm or SwiGLU, this derivative does not depend on the original input values. It depends only on the original positions and frequencies. Thus backward does not need saved unrotated or rotated activations **for the rotation itself**. Other operators may still need those arrays: an attention dot product needs the rotated query and key, and projection-weight gradients need the projection input. Local derivative storage requirements do not determine the whole model's tape lifetime.

Our equations used column vectors for convenience. Each head is stored within a row in C. In row-vector matrix notation the same forward operation would be $xR^{\mathsf T}$ and its row-gradient operation would be $gR$. The scalar formulas above already implement the right mapping; do not add another transpose merely because the array is row-major.

## Give each row an absolute position

The floating-point arrays use the existing `DlMatrix`/`DlConstMatrix` representation. A call processes a matrix with $M$ rows and $H d$ columns, where $H$ is the head count **for that call**. Each row contains all heads of one token, with the coordinates within each head contiguous.

For row $i$, head $h$, and plane $j$, the two input element indices are

$$
\begin{aligned}
a&=\mathrm{offset}+i\,\mathrm{ld}+h d+j,\\
b&=a+r.
\end{aligned}
$$

`ld` is the physical row stride in float elements, not bytes. It can exceed $H d$ when rows are padded. For `offset = 2`, `ld = 10`, $H=2,d=4$, row 1, head 1, plane 1 uses indices $2+10+4+1=17$ and $17+2=19$. Both belong to the second head of the same row. The input and output may have different offsets and strides.

Positions live in a **separate** `size_t` array of length $M$. They are not inferred from the row index. For example:

| Use | Rows supplied | Position array |
|---|---|---|
| One four-token prefix | 4 | `[0, 1, 2, 3]` |
| A two-token chunk after that prefix | 2 | `[4, 5]` |
| One new token after that chunk | 1 | `[6]` |
| Two flattened two-token sequences | 4 | `[0, 1, 0, 1]` |

For a dense batch flattened from $[B,T,Hd]$, row $bT+t$ can use position $t$ or a sequence-specific start plus $t$. The caller builds that position array with checked integer arithmetic. The rotary operator allows repeated positions and resets; it has no batch boundaries or hidden counter.

Query and key calls may have different column counts. With the selected model's dimensions, they would be $[M,576]$ and $[M,192]$, respectively, while both use $d=64$ and the same position/frequency arrays. Rotating a shared key once does not require duplicating it for every query head. The actual query-to-key head mapping belongs to attention, not this transform.

A future cache will store the key already rotated at its token's absolute position. Appending one token must not rotate all older cached keys again. Nor may the one-row call use position zero just because its local row index is zero. Part 2 verifies whole-sequence versus one-row execution now, without pretending that a cache has already been built.

## Bound representation without inventing a trained context length

The API accepts integer positions up to `UINT32_MAX`, or $2^{32}-1$. This is an explicit implementation bound. Every such integer converts exactly to the binary double type required below. A float cannot represent every adjacent integer above $2^{24}$: passing the position through a float first can collapse two distinct positions before any trigonometry occurs.

Frequencies, angle multiplication, sine/cosine, and the two pair expressions use `double`. The input/output arrays remain `float`. Because every accepted frequency lies in $(0,1]$, angle magnitude is at most $2^{32}-1$ radians and the angle multiplication cannot overflow this double type. The code generates each angle directly from the absolute position; it does not accumulate an angle recurrence that drifts across many tokens.

This bound does not make every phase exact. The frequency itself is rounded; multiplying it by a large position magnifies its absolute error. Trigonometric argument reduction and final float stores add further error. A common-shift equality is therefore approximate in floating-point arithmetic, and the tolerance used for positions around 8192 is not a universal guarantee for all accepted positions.

Our integer bound is also **not** a claim that a model trained with an 8192-position configuration understands four billion positions. Representation limits, allocated cache capacity, and trained/configured context length are separate constraints. The model/cache layers must impose their own bounds later.

## The C features used by these routines

`cos`, `sin`, and `pow` are declared in `<math.h>` and operate on `double`; their `f`-suffixed forms operate on `float`. `cos` and `sin` take radians, and `pow(base, exponent)` calculates a real power. These interfaces are described in [WG14 N1570, Sections 7.12.4.5–7.12.4.6 and 7.12.7.4][c-draft]. The standard's ordinary interface does not promise that every implementation produces identical last bits.

`_Static_assert(condition, "message")` is a compile-time check available in our C17 mode. It does not disappear under optimized builds, and it does not inspect runtime data. We use it to make the binary-double precision assumption explicit. `<float.h>` supplies `FLT_RADIX`, `DBL_MANT_DIG`, and `FLT_MAX`; `<stdint.h>` supplies the integer-limit macros on the target platforms.

The frequency exponent is written as `-(double)j / (double)count`. Converting **before** negation prevents unsigned `size_t` arithmetic from wrapping; converting before division prevents integer truncation. This is equivalent to $-j/r$ because `count` is $r$, and avoids constructing the unnecessary integer product `2*j`.

The remainder expression `head_dim % 2` detects odd widths. In a guard joined with `||`, C evaluates operands from left to right and stops once one is true. We check `head_dim == 0` before evaluating `x.cols % head_dim`, so a rejected zero width never causes integer division by zero.

## Create the public header

Create `include/dl/rope.h`. It includes the existing matrix/view and write-mode declarations through `dl/kernels.h`. This inclusion reuses types; it does not make a matrix-multiplication call inside RoPE.

The caller owns every array. Pointer/count pairs describe initialized storage and lifetimes; a count does not make an invalid pointer valid. Position/frequency arrays are separate typed storage, disjoint from each other and from the float arrays. The float-view alias check still relies on **canonical allocation bases**, as in earlier chapters. It cannot discover arbitrary rebased pointers or cross-type overlap that violates the stated preconditions.

```c title="include/dl/rope.h"
#ifndef DL_ROPE_H
#define DL_ROPE_H

#include "dl/kernels.h"

#include <stdint.h>

/* Integer-position contract, not a model's trained context length. */
#define DL_ROPE_MAX_POSITION ((size_t)UINT32_MAX)

/* Write head_dim/2 doubles into caller-owned frequency storage.
 * head_dim is positive and even; count must equal head_dim/2.
 * theta is finite and > 1. Frequencies are theta^(-2*j/head_dim).
 * No allocation. On any reported failure, leave inv_freq unchanged.
 * The output must be live, correctly aligned, and large enough for count. */
DlStatus dl_rope_frequencies(size_t head_dim, double theta,
                             double *inv_freq, size_t count);

/* x/y are [M,H*head_dim]; H is inferred separately on each call.
 * Pair j with j+head_dim/2 WITHIN each head (split-half, not adjacent).
 * positions is a separate live size_t[M] array of absolute positions.
 * Each position is <= DL_ROPE_MAX_POSITION; repetitions/resets are allowed.
 * inv_freq is a separate live double[head_dim/2] array with values in (0,1].
 * Float, position, and frequency storage must not overlap each other.
 * Arrays retain canonical bases/counts and truthful live capacities.
 * x/y bounding spans must be disjoint, even for exact in-place requests.
 * No allocation, retained pointers, cached angles, or hidden position counter.
 * Input/old ADD logical cells must be finite. Padding is untouched.
 * Structural/preflight errors do not write. Later DL_NUMERIC can leave
 * earlier pairs written: discard the entire result and do not retry ADD.
 * Each pair's two results are range-checked before either is stored.
 * Frequencies, angles, and pair arithmetic use double; arrays store float. */
DlStatus dl_rope_forward(DlConstMatrix x, size_t head_dim,
                         const size_t *positions, size_t position_count,
                         const double *inv_freq, size_t frequency_count,
                         DlMatrix y);

/* Apply the rotation TRANSPOSE to dy, then SET or ADD into dx.
 * dy/dx have identical [M,H*head_dim] shapes. Positions/frequencies must
 * match the forward whose gradient is being propagated. Neither forward
 * x nor y is needed: the Jacobian depends only on those fixed settings.
 * No derivative of positions/frequencies and no extra loss averaging. */
DlStatus dl_rope_backward(DlConstMatrix dy, size_t head_dim,
                          const size_t *positions, size_t position_count,
                          const double *inv_freq, size_t frequency_count,
                          DlMatrix dx, DlWriteMode mode);

#endif
```

Forward always overwrites. Backward accepts SET or ADD because its contribution may need to join another path. SET must not read old output cells, which may be uninitialized or NaN. ADD requires initialized finite old cells and adds the computed contribution exactly once.

Both routines reject exact in-place calls as well as partial input/output overlap. This costs a separate destination, but keeps backward input gradients, reusable query/key buffers, and failure handling consistent with the earlier operators. Two disjoint spans inside the same allocation are allowed. As before, the overlap test uses bounding spans, so it may conservatively reject interleaved logical regions separated only by row padding.

A single head does not own its bytes separately from the enclosing matrix. The library never allocates, frees, retains pointers, or advances a position. For $d=64$, the frequency payload is 32 doubles, or 256 bytes on the target's eight-byte double representation. The supplied position array uses $M\,\mathrm{sizeof(size_t)}$ bytes. A separate dense output uses $MHd\,\mathrm{sizeof(float)}$ bytes. These are payload counts, not measured process memory.

## Construct the frequency vector before touching model data

Create `src/rope.c`. All blocks with this filename concatenate in order into the complete file.

The constructor validates the shape and byte-count arithmetic before dereferencing the frequency array. It first checks all generated values, then repeats the small calculation to write them. This two-pass choice leaves the output unchanged on any reported failure without allocating a temporary vector. It assumes unchanged arguments and the ordinary unchanged floating-point environment between the passes; callers must not race writes to the output.

```c title="src/rope.c"
#include "dl/rope.h"

#include <float.h>
#include <math.h>

_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG >= 53,
               "RoPE requires binary double with at least 53 significand bits");

DlStatus dl_rope_frequencies(size_t head_dim, double theta,
                             double *inv_freq, size_t count)
{
    if (inv_freq == NULL || !isfinite(theta) || theta <= 1.0) return DL_ARGUMENT;
    if (head_dim == 0 || head_dim % 2 != 0 || count != head_dim / 2) {
        return DL_SHAPE;
    }
    if (count > SIZE_MAX / sizeof(*inv_freq)) return DL_OVERFLOW;
    /* The first pass keeps the caller's output unchanged on a math failure.
     * The same immutable arguments are recomputed only after all pass. */
    for (size_t j = 0; j < count; ++j) {
        const double exponent = -(double)j / (double)count;
        const double value = pow(theta, exponent);
        if (!isfinite(value) || value <= 0.0 || value > 1.0) return DL_NUMERIC;
    }
    for (size_t j = 0; j < count; ++j) {
        inv_freq[j] = pow(theta, -(double)j / (double)count);
    }
    return DL_OK;
}

```

The first frequency is `pow(theta, -0.0)`, which is one for our positive finite base. Every other exponent is between minus one and zero. Rejecting nonpositive or nonfinite results prevents an unusable frequency vector from silently entering the model.

There is no implicit default base. An experiment can use 10000, while a model configured with 100000 must pass that value. The rotation call accepts the resulting table rather than the base; keep that table unchanged between a corresponding forward and backward, and between cached-prefix and appended-token calls.

## Check geometry and required values before writing

`rope_prepare` shares the complete preflight between forward and backward. The matrix geometry helper already checks offsets, row strides, overflow, and declared capacity. Only then do we inspect the rotary shapes and input values.

The tests include an invalid **last** logical input and an invalid old ADD output. Those errors must be discovered before earlier valid pairs can change the destination. Padding is excluded from finite-value scans. SET does not scan the old destination at all.

```c title="src/rope.c"
static int finite_cells(DlConstMatrix v)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (!isfinite(v.base[v.offset + i * v.ld + j])) return 0;
        }
    }
    return 1;
}

static DlStatus rope_prepare(DlConstMatrix x, size_t head_dim,
                              const size_t *positions, size_t position_count,
                              const double *inv_freq, size_t frequency_count,
                              DlMatrix y, DlWriteMode mode)
{
    if (positions == NULL || inv_freq == NULL ||
        (mode != DL_SET && mode != DL_ADD)) return DL_ARGUMENT;
    size_t xe = 0, ye = 0;
    DlStatus s = dl_matrix_end(x, &xe);
    if (s != DL_OK) return s;
    s = dl_matrix_end(dl_matrix_read(y), &ye);
    if (s != DL_OK) return s;
    if (head_dim == 0 || head_dim % 2 != 0 || x.cols % head_dim != 0 ||
        x.rows != y.rows || x.cols != y.cols || position_count != x.rows ||
        frequency_count != head_dim / 2) return DL_SHAPE;
    if (position_count > SIZE_MAX / sizeof(*positions) ||
        frequency_count > SIZE_MAX / sizeof(*inv_freq)) return DL_OVERFLOW;
    if (x.base == y.base && x.offset < ye && y.offset < xe) return DL_ALIAS;
    for (size_t i = 0; i < position_count; ++i) {
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    }
    for (size_t j = 0; j < frequency_count; ++j) {
        if (!isfinite(inv_freq[j]) || inv_freq[j] <= 0.0 || inv_freq[j] > 1.0) {
            return DL_NUMERIC;
        }
    }
    if (!finite_cells(x) ||
        (mode == DL_ADD && !finite_cells(dl_matrix_read(y)))) return DL_NUMERIC;
    return DL_OK;
}

```

Notice the distinction between position validation and position generation. This routine rejects a value outside its supported range but does not calculate `start + row` for the caller. That avoids one hidden source of unsigned overflow and allows sequence resets inside a flattened batch.

Likewise, validating `frequency_count == head_dim/2` does not prove that the table was generated for the correct model base. The table is an explicit numerical input. Structural safety, mathematical convention, and model compatibility require different checks.

## Translate the pair transformation into loops

The traversal is row, plane, then head. For one row and one plane, every head shares the same position and frequency. We compute one cosine/sine pair, then apply it to all those heads. There are $M d/2$ trigonometric pairs rather than $MHd/2$ independently repeated pairs. Arithmetic and stores still visit every head coordinate, so total work is proportional to $MHd$, plus validation and the trigonometric evaluations. No speedup measurement follows from this operation count.

For backward, the internal `transpose` flag negates sine. The rest of the loop is unchanged. This is a private implementation detail with two public wrappers; there is no public flag that permits an unrecognized transform mode.

```c title="src/rope.c"
static DlStatus rope_apply(DlConstMatrix x, size_t head_dim,
                            const size_t *positions, size_t position_count,
                            const double *inv_freq, size_t frequency_count,
                            DlMatrix y, DlWriteMode mode, int transpose)
{
    DlStatus status = rope_prepare(x, head_dim, positions, position_count,
                                   inv_freq, frequency_count, y, mode);
    if (status != DL_OK) return status;
    const size_t half = head_dim / 2;
    const size_t heads = x.cols / head_dim;
    for (size_t i = 0; i < x.rows; ++i) {
        const size_t xr = x.offset + i * x.ld;
        const size_t yr = y.offset + i * y.ld;
        for (size_t j = 0; j < half; ++j) {
            const double angle = (double)positions[i] * inv_freq[j];
            const double c = cos(angle);
            double s = sin(angle);
            if (!isfinite(c) || !isfinite(s)) return DL_NUMERIC;
            if (transpose) s = -s;
            for (size_t h = 0; h < heads; ++h) {
                const size_t a = h * head_dim + j;
                const size_t b = a + half;
                const double u = (double)x.base[xr + a];
                const double v = (double)x.base[xr + b];
                double first = c * u - s * v;
                double second = s * u + c * v;
                if (mode == DL_ADD) {
                    first += (double)y.base[yr + a];
                    second += (double)y.base[yr + b];
                }
                if (!isfinite(first) || !isfinite(second) ||
                    fabs(first) > (double)FLT_MAX ||
                    fabs(second) > (double)FLT_MAX) return DL_NUMERIC;
                y.base[yr + a] = (float)first;
                y.base[yr + b] = (float)second;
            }
        }
    }
    return DL_OK;
}

```

The variables `a` and `b` are column offsets within a row, not full array indices. `xr` and `yr` supply the input/output row starts. This is why both arrays can be padded differently without changing the rotation math.

Both operands widen before multiplication. In ADD mode, both old destination values widen before addition. The code checks **both** final double results for finiteness and float range before either store, then narrows each accepted result once. This gives pair-level commit behavior, not whole-matrix atomicity. Earlier pairs may already be written when a later pair fails.

The loop uses the same sine/cosine construction in forward and backward. In exact arithmetic the transpose is also the inverse. In floating-point arithmetic a forward/backward round trip is only approximate, because trigonometric values and output arrays are rounded. The intended analytic derivative is the derivative of the continuous rotation using the fixed coefficients, not a derivative of discontinuous float rounding.

## Keep the public entry points small and explicit

Append the complete wrappers. Forward requests SET with the ordinary sine sign. Backward passes the caller's write mode and requests the transpose. Neither wrapper applies a loss scale or changes the position array.

```c title="src/rope.c"
DlStatus dl_rope_forward(DlConstMatrix x, size_t head_dim,
                         const size_t *positions, size_t position_count,
                         const double *inv_freq, size_t frequency_count,
                         DlMatrix y)
{
    return rope_apply(x, head_dim, positions, position_count,
                       inv_freq, frequency_count, y, DL_SET, 0);
}

DlStatus dl_rope_backward(DlConstMatrix dy, size_t head_dim,
                          const size_t *positions, size_t position_count,
                          const double *inv_freq, size_t frequency_count,
                          DlMatrix dx, DlWriteMode mode)
{
    return rope_apply(dy, head_dim, positions, position_count,
                       inv_freq, frequency_count, dx, mode, 1);
}
```

The resulting source file needs only the existing common/view infrastructure and the standard math library. It has no dependency on an attention tape, tokenizer, checkpoint loader, or global backend. The reference and blocked matrix backends do not change this operator: it contains no matrix-product dispatch.

## Why finite inputs can still produce a numerical failure

A rotation preserves the real vector norm, but that does not guarantee that each output coordinate fits in a float. The real pair $(\mathrm{FLT\_MAX},\mathrm{FLT\_MAX})$ has a norm larger than the largest representable float coordinate. At angle $\pi/4$ it becomes approximately $(0,\sqrt2\,\mathrm{FLT\_MAX})$, so the second coordinate cannot be stored. The double arithmetic exposes this before narrowing.

For a width-four split-half input $[3,\mathrm{FLT\_MAX},4,\mathrm{FLT\_MAX}]$, choose both angles to be $\pi/4$. The first pair, coordinates 0 and 2, succeeds; the second pair, coordinates 1 and 3, fails. If the destination starts with nines, the result at failure is approximately

$$
[-1/\sqrt2,\ 9,\ 7/\sqrt2,\ 9].
$$

The failed pair is untouched, but the destination as a whole is unusable. Backward ADD can also overflow when a finite contribution joins an already large finite gradient. **Discard the whole result after `DL_NUMERIC`. Do not retry ADD on the partially modified destination.** A caller needing transaction-like behavior must supply separate temporary output and commit after success; that is not hidden in this operator.

Underflow of tiny accepted results to a representable subnormal or zero is ordinary floating-point behavior, not an automatically reported error. Norm checks therefore use mixed absolute/relative tolerances instead of requiring bitwise preservation. The large-value overflow check and the small-value rounding explanation address different ends of the numeric range.

## Connect the operator to its next consumers

A projected query array can now be transformed into a separate rotated-query array. A projected key array can be transformed with the same frequency/position inputs even when it has fewer heads. The input-gradient contribution from either transform is obtained with its transpose, using the position of **that** query or key, not the position of whichever vector it was compared with.

For example, let $S=\widetilde q^{\mathsf T}\widetilde k$ be one unscaled scalar dot product. Then $\partial S/\partial\widetilde q=\widetilde k$ and $\partial S/\partial\widetilde k=\widetilde q$. The two unrotated gradients are

$$
\frac{\partial S}{\partial q}=R(m)^{\mathsf T}\widetilde k,
\qquad
\frac{\partial S}{\partial k}=R(n)^{\mathsf T}\widetilde q,
$$

where $R(p)$ now denotes the complete split-half rotation with angles $p\omega_j$. Using the key's position for the query derivative would reverse the wrong transform. Part 2 implements and differentiates exactly this scalar composition; it is not yet scaled attention, a causal mask, or a softmax.

Continue to [Part 2](https://25d-228.github.io/gpt-c/chapter-09-part-2.html) to add every test and the Makefile. The companion [header](https://25d-228.github.io/gpt-c/media/chapter-09/reference/include/dl/rope.h) and [source module](https://25d-228.github.io/gpt-c/media/chapter-09/reference/src/rope.c) contain the same code as this page's filename-labeled blocks.

## References

[RoFormer, Sections 3.1–3.2 and the rotation construction][rope] supplies the positional-transform formulation. The derivations, memory trace, API decisions, and diagnostic fixtures above spell out the implementation used here. [SmolLM2-135M's configuration][config] supplies the selected model's head counts and rotary settings; those settings will be checked again against the exact imported revision. [WG14 N1570][c-draft], Sections 6.7.10, 7.12.4.5–7.12.4.6, and 7.12.7.4, documents compile-time assertions and the C math interfaces used by the code. N1570 is a public C11 committee draft, not a claim that it is the C17 standard text; these features are available in the C17 mode used in this course.

[rope]: https://arxiv.org/html/2104.09864v5
[config]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
[c-draft]: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
