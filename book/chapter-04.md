# Chapter 4 — Tensor layouts and reference matrix products

A linear projection applies the same weights to every token row. Its memory operation must be unambiguous: which float belongs to which row, which dimension is reduced, and whether a second call replaces or adds to the output.

This chapter starts the decoder modules: dense tensor owners, borrowed matrix views, and scalar products with logical transposes. A two-by-two output comes from two stored two-by-three arrays. Further checks use non-square shapes, padding, shared storage, and failing requests. No tensor framework, BLAS, Python, or model download executes the computation.

> **Learning objectives**
> Derive offsets and strides from tensor shapes; flatten dense token rows without copying; preserve ownership while borrowing matrix views; compute forward and transpose products with one fixed weight orientation; distinguish overwrite from accumulation; and locate layout errors using hand calculations, guard values, and numerical checks.

**Prerequisites:** [Chapter 1](https://25d-228.github.io/gpt-c/chapter-01.html), [Chapter 2](https://25d-228.github.io/gpt-c/chapter-02.html), and [Chapter 3](https://25d-228.github.io/gpt-c/chapter-03.html). Keep their `examples/` programs unchanged. We reuse initialized ownership, checked sizes, borrowing, cleanup, and finite-aware comparisons, not those demonstration modules.

**Implementation estimate: 4.00 hours**—2.75 for storage and kernels; 1.25 for checks and variations. Reading is separate; record actual hands-on time.

## Start the decoder modules at the workspace root

Work in the learner workspace, not inside `examples/03-numerics/`. Create the new directories:

```sh title="Terminal — from the learner workspace root"
mkdir -p include/dl src tests
```

These eight files are fully listed below. Companion files match the assembled listings; they are optional conveniences, not missing implementation instructions.

| File | Responsibility |
|---|---|
| [include/dl/common.h][common-h] and [src/common.c][common-c] | Shared statuses and overflow-safe size arithmetic |
| [include/dl/tensor.h][tensor-h] and [src/tensor.c][tensor-c] | Dense owners, borrowed matrix geometry, and checked views |
| [include/dl/kernels.h][kernels-h] and [src/kernels.c][kernels-c] | Scalar products with transpose and write modes |
| [tests/test_kernels.c][tests-c] | Complete runnable fixtures and diagnostics |
| [Makefile][makefile] | Debug, sanitizer, and optimized builds at the workspace root |

Append blocks bearing the same filename in order, including the identified function continuations. The two **Variation** blocks are temporary replacements, not baseline source. No future module is required.

## A tensor shape is a rule for interpreting a flat allocation

A *tensor* is an array of numbers with a shape. Its **rank** counts axes, not linearly independent matrix rows: a vector has one axis, a matrix two, and batched token rows three. Our dense owners support at most four axes to keep metadata bounded.

Let $B$ be the number of sequences in a batch, $T$ the number of token positions in each sequence, and $D$ the number of features per token. A dense activation has shape $[B,T,D]$. Indices $b,t,d$ start at zero and satisfy $b<B$, $t<T$, and $d<D$.

Store features consecutively within tokens, then tokens within sequences. Sequence $b$ skips $bTD$ floats, token $t$ another $tD$, and feature $d$ another $d$: 

$$
\operatorname{offset}(b,t,d) = b(TD)+tD+d = (bT+t)D+d.
$$

Every offset here is in **float elements**, not bytes. `data[offset]` performs the element addressing; multiplying that offset by `sizeof(float)` before indexing would multiply the distance twice.

A **stride** tells us how many elements to advance when an index on one axis increases by one. The dense strides for $[B,T,D]$ are $[TD,D,1]$. For an arbitrary supported dense shape $[s_0,\ldots,s_{r-1}]$, where $r$ is the number of axes, define

$$
q_{r-1}=1,\qquad q_i=s_{i+1}q_{i+1},\qquad
\operatorname{offset}(i_0,\ldots,i_{r-1})=\sum_{a=0}^{r-1}i_aq_a.
$$

Here $q_a$ is the stride of axis $a$, and $i_a$ is its coordinate. The last coordinate changes fastest. Our allocation is one flat `float` buffer, not a `float **` containing separately allocated rows.

For shape $[2,3,4]$, there are 24 elements and the strides are $[12,4,1]$:

| Tensor coordinate | Element offset | Flattened row and column |
|---|---:|---|
| $[0,0,0]$ | 0 | $[0,0]$ |
| $[0,2,3]$ | 11 | $[2,3]$ |
| $[1,0,0]$ | 12 | $[3,0]$ |
| $[1,2,3]$ | 23 | $[5,3]$ |

Define the flattened row $u=bT+t$. The offset becomes $uD+d$, exactly the addressing rule for a matrix with $BT$ rows and $D$ columns. Nothing moves. We construct a different view of the same allocation.

Flattening works because the axes are dense in that order. Permuted or gapped axes cannot generally be flattened by relabeling their shape; later head layouts must derive their own offsets.

## Make every size computation a checked computation

The count is the product of all extents. The byte request is that count multiplied by `sizeof(float)`. Both products need checks. A count that fits `size_t` can still require a byte count that does not fit.

Use a shared status type. `DL_OK` alone means success; `DL_` and `dl_` are ordinary naming prefixes.

```c title="include/dl/common.h"
#ifndef DL_COMMON_H
#define DL_COMMON_H

#include <stddef.h>

typedef enum {
    DL_OK = 0,
    DL_ARGUMENT,
    DL_STATE,
    DL_OVERFLOW,
    DL_ALLOC,
    DL_RANGE,
    DL_SHAPE,
    DL_ALIAS,
    DL_NUMERIC
} DlStatus;

const char *dl_status_name(DlStatus status);
/* On failure, leave *out unchanged. A null out is an argument error. */
DlStatus dl_size_mul(size_t a, size_t b, size_t *out);
DlStatus dl_size_add(size_t a, size_t b, size_t *out);

#endif
```

`DL_ARGUMENT` rejects parameter values; `DL_STATE` rejects owner states. `DL_RANGE` rejects views exceeding storage, while `DL_SHAPE` rejects incompatible product dimensions. Distinct statuses locate the failing layer.

Start the implementation with readable status names:

```c title="src/common.c"
#include "dl/common.h"

#include <stdint.h>

const char *dl_status_name(DlStatus status)
{
    switch (status) {
    case DL_OK:       return "ok";
    case DL_ARGUMENT: return "invalid argument";
    case DL_STATE:    return "invalid owner state";
    case DL_OVERFLOW: return "size overflow";
    case DL_ALLOC:    return "allocation failed";
    case DL_RANGE:    return "view out of range";
    case DL_SHAPE:    return "incompatible shapes";
    case DL_ALIAS:    return "output overlaps input span";
    case DL_NUMERIC:  return "non-finite or out-of-range result";
    }
    return "unknown status";
}

```

For unsigned sizes $a$ and $b$, testing `a*b` after computing it is too late: the expression may already have wrapped. For nonzero $a$, a product fits exactly when $b\le\lfloor\texttt{SIZE\_MAX}/a\rfloor$. For addition, it fits when $b\le\texttt{SIZE\_MAX}-a$. Both tests avoid the overflowing expression.

```c title="src/common.c"
DlStatus dl_size_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return DL_ARGUMENT;
    }
    if (a != 0 && b > SIZE_MAX / a) {
        return DL_OVERFLOW;
    }
    *out = a * b;
    return DL_OK;
}

DlStatus dl_size_add(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return DL_ARGUMENT;
    }
    if (b > SIZE_MAX - a) {
        return DL_OVERFLOW;
    }
    *out = a + b;
    return DL_OK;
}
```

Multiplication accepts zero, but tensor construction rejects zero **extents**. An empty owner is a cleanup state, not an executable tensor. This deliberately differs from Chapter 2's empty buffers and BLAS zero-dimension behavior.

The helpers assign `*out` only after validating the request. Passing `&count` while also passing `count` by value is safe: the argument value is obtained before the function executes its assignment. A failed operation leaves the previous count untouched.

## Separate ownership from a matrix interpretation

Only the owner releases an allocation. Several views may borrow it, including read-only kernel inputs.

`#define DL_MAX_RANK 4` supplies a compile-time bound for metadata arrays. Numeric elements live elsewhere. `DlMatrix` permits element writes; `DlConstMatrix` does not. Merely declaring a descriptor `const DlMatrix` would freeze its fields, **not** the floats its pointer reaches.

```c title="include/dl/tensor.h"
#ifndef DL_TENSOR_H
#define DL_TENSOR_H

#include "dl/common.h"

#define DL_MAX_RANK 4

typedef struct {
    float *data;
    size_t count;
    size_t rank;
    size_t shape[DL_MAX_RANK];
    size_t stride[DL_MAX_RANK];
} DlTensor;

typedef struct {
    float *base;
    size_t count, offset, rows, cols, ld;
} DlMatrix;

typedef struct {
    const float *base;
    size_t count, offset, rows, cols, ld;
} DlConstMatrix;

/* out must be an initialized empty owner, normally {0}.
 * rank is 1..DL_MAX_RANK; all extents are positive.
 * Success owns count uninitialized floats in dense row-major order.
 * Failure leaves out unchanged. Never copy a live owner or edit its metadata. */
DlStatus dl_tensor_create(DlTensor *out, size_t rank, const size_t *shape);
void dl_tensor_destroy(DlTensor *tensor);

/* Borrow a row-major matrix; all sizes/offsets/ld are in float elements.
 * Keep the owner's canonical allocation base/count; never rebase a view.
 * Positive rows/cols and ld >= cols. Failure leaves *out unchanged.
 * The owner must remain live and unchanged while a view is used. */
DlStatus dl_matrix_view(DlTensor *owner, size_t offset, size_t rows,
                        size_t cols, size_t ld, DlMatrix *out);
DlStatus dl_tensor_rows(DlTensor *owner, DlMatrix *out);
DlConstMatrix dl_matrix_read(DlMatrix matrix);

/* Validate declared geometry, not actual pointer lifetime or allocation size.
 * On success, *out is the exclusive end of the bounding element span.
 * Failure leaves *out unchanged. */
DlStatus dl_matrix_end(DlConstMatrix matrix, size_t *out);

#endif
```

`base` and `count` always describe the owner's **entire allocation**. Submatrices change only geometry; never rebase to an interior pointer. The kernel recognizes shared storage by base equality, then compares integer offsets.

**Leading dimension**, `ld`, is the element distance between physical row starts, not necessarily the row width. Columns are contiguous and `ld >= cols`. Reversed axes, arbitrary column strides, and broadcasting are unsupported.

### Construct into a temporary, then transfer ownership

Create the dense owner by working from the last axis backwards. Before processing an axis, `count` is the number of elements in all already-processed trailing axes, so it is exactly that axis's stride.

```c title="src/tensor.c"
#include "dl/tensor.h"

#include <stdlib.h>

DlStatus dl_tensor_create(DlTensor *out, size_t rank, const size_t *shape)
{
    if (out == NULL || shape == NULL || rank == 0 || rank > DL_MAX_RANK) {
        return DL_ARGUMENT;
    }
    if (out->data != NULL || out->count != 0 || out->rank != 0) {
        return DL_STATE;
    }
    DlTensor next = {0};
    size_t count = 1;
    for (size_t axis = rank; axis > 0; --axis) {
        const size_t i = axis - 1;
        if (shape[i] == 0) {
            return DL_ARGUMENT;
        }
        next.shape[i] = shape[i];
        next.stride[i] = count;
        const DlStatus status = dl_size_mul(count, shape[i], &count);
        if (status != DL_OK) {
            return status;
        }
    }
    size_t bytes = 0;
    const DlStatus status = dl_size_mul(count, sizeof(float), &bytes);
    if (status != DL_OK) {
        return status;
    }
    next.data = malloc(bytes);
    if (next.data == NULL) {
        return DL_ALLOC;
    }
    next.count = count;
    next.rank = rank;
    *out = next;
    return DL_OK;
}

```

For shape $[2,3,4]$, the loop visits extents 4, 3, and 2. It records stride 1 before changing the count to 4, records stride 4 before changing the count to 12, and records stride 12 before changing the count to 24. The byte check then requests `24 * sizeof(float)` bytes.

`axis > 0` and `i = axis - 1` avoid an unsigned loop that decrements zero and wraps. `shape` must point to at least `rank` initialized extents; a pointer parameter cannot prove that fact. The constructor copies those extents, so it does not retain the shape array.

`*out = next` commits only after validation/allocation. This is an intentional **ownership transfer**: the local temporary has no automatic destructor and must not free the allocation. Never turn a live owner into two independently destroyed copies.

A null `malloc` result leaves the owner empty. Successful elements are **uninitialized**: fill them before reading, or overwrite without reading them. This constructor has no allocator injection; Chapter 2 remains the deterministic allocation-failure example.

```c title="src/tensor.c"
void dl_tensor_destroy(DlTensor *tensor)
{
    if (tensor == NULL) {
        return;
    }
    free(tensor->data);
    *tensor = (DlTensor){0};
}

```

Destroy accepts null, an initialized empty owner, or the unique live owner. It frees storage and resets every field. Repeating it on that reset variable is safe; destroying a copied live owner is not. All borrowed views expire with the allocation.

## Borrow padded rows without pretending they are dense

A physical matrix element at row $i$ and column $j$ has the offset

$$
\operatorname{index}(i,j)=o+i\ell+j,
$$

where $o$ is `offset` and $\ell$ is `ld`. For $R$ rows and $C$ columns, the final accessed element is $o+(R-1)\ell+(C-1)$. Its exclusive upper bound is

$$
e=o+(R-1)\ell+C.
$$

The enclosing storage span is $[o,e)$: it includes the element at $o$ and excludes the element at $e$. It includes gaps between rows, but not unused padding after the last logical row.

Take an allocation of nine floats and a view with `offset=1`, `rows=2`, `cols=2`, `ld=4`. Its layout is:

```text title="Element positions in the guarded output allocation"
index:     0     1     2     3     4     5     6     7     8
meaning: guard  y00   y01   gap   gap   y10   y11  guard guard
```

The first row uses positions 1 and 2; the second uses 5 and 6. Its bounding span ends at `1 + (2-1)*4 + 2 = 7`. A kernel must leave positions 0, 3, 4, 7, and 8 alone. A wrong stride can write those validly allocated positions without ever leaving the allocation.

First supply the conversion to read-only element access. This returns a small descriptor by value and copies no tensor elements:

```c title="src/tensor.c"
DlConstMatrix dl_matrix_read(DlMatrix matrix)
{
    return (DlConstMatrix){.base = matrix.base, .count = matrix.count,
        .offset = matrix.offset, .rows = matrix.rows, .cols = matrix.cols,
        .ld = matrix.ld};
}

```

C permits conversion from `float *` to `const float *` without a cast. The returned view cannot write elements through `base`; it does not freeze other aliases or extend the allocation's lifetime.

Now compute the exclusive span end using the checked arithmetic helpers:

```c title="src/tensor.c"
DlStatus dl_matrix_end(DlConstMatrix matrix, size_t *out)
{
    if (out == NULL || matrix.base == NULL || matrix.rows == 0 ||
        matrix.cols == 0 || matrix.ld < matrix.cols) {
        return DL_ARGUMENT;
    }
    if (matrix.offset >= matrix.count) {
        return DL_RANGE;
    }
    size_t span = 0;
    size_t end = 0;
    DlStatus status = dl_size_mul(matrix.rows - 1, matrix.ld, &span);
    if (status == DL_OK) {
        status = dl_size_add(span, matrix.cols, &span);
    }
    if (status == DL_OK) {
        status = dl_size_add(matrix.offset, span, &end);
    }
    if (status != DL_OK) {
        return status;
    }
    if (end > matrix.count) {
        return DL_RANGE;
    }
    *out = end;
    return DL_OK;
}

```

The start and every intermediate sum/product are checked before committing the end. Positive extents make `rows - 1` safe; `ld >= cols` prevents overlapping rows.

This validates **declared geometry**, not actual allocation size or lifetime. Forged counts, edited owner shapes, rebased pointers, and expired views violate the caller contract.

Create a view by filling a temporary descriptor, validating it, and then assigning it. Flattening a dense owner is a thin, fully specified use of the same constructor:

```c title="src/tensor.c"
DlStatus dl_matrix_view(DlTensor *owner, size_t offset, size_t rows,
                        size_t cols, size_t ld, DlMatrix *out)
{
    if (owner == NULL || out == NULL) {
        return DL_ARGUMENT;
    }
    if (owner->data == NULL || owner->count == 0 || owner->rank == 0) {
        return DL_STATE;
    }
    const DlMatrix next = {.base = owner->data, .count = owner->count,
        .offset = offset, .rows = rows, .cols = cols, .ld = ld};
    size_t end = 0;
    const DlStatus status = dl_matrix_end(dl_matrix_read(next), &end);
    if (status != DL_OK) {
        return status;
    }
    *out = next;
    return DL_OK;
}

DlStatus dl_tensor_rows(DlTensor *owner, DlMatrix *out)
{
    if (owner == NULL || out == NULL) {
        return DL_ARGUMENT;
    }
    if (owner->data == NULL || owner->rank < 2 ||
        owner->rank > DL_MAX_RANK) {
        return DL_STATE;
    }
    /* The constructor establishes positive shapes and dense strides. */
    const size_t cols = owner->shape[owner->rank - 1];
    return dl_matrix_view(owner, 0, owner->count / cols, cols, cols, out);
}
```

The example flattens to `24 / 4 = 6` rows. Constructor-established positive shapes make this division exact. Rank-one owners require explicit `dl_matrix_view` geometry; `dl_tensor_rows` does not invent a vector orientation.

View construction allocates nothing and changes no field on failure. Replacing a borrowed descriptor leaks nothing. Keep its owner live and metadata unchanged.

## Derive a linear projection before writing its loops

Let $M$ be the number of token rows, $K$ the number of input features, and $N$ the number of output features. Store the input as a matrix $X$ of shape $[M,K]$. Store the weight as $W$ of shape $[N,K]$: **one output feature per physical weight row**. This orientation will stay fixed across the decoder.

The output $Y$ has shape $[M,N]$. Its entry in row $i$, output feature $j$, is the dot product of input row $i$ and weight row $j$:

$$
Y_{i,j}=\sum_{p=0}^{K-1}X_{i,p}W_{j,p}.
$$

The index $p$ is the feature coordinate being summed away. Each output coordinate is one reduction like Chapter 1's dot product. In matrix notation, $Y=XW^{\mathsf T}$, where the superscript $\mathsf T$ means transpose: $(W^{\mathsf T})_{p,j}=W_{j,p}$. It does **not** instruct us to allocate and fill another weight matrix.

Use the concrete values

$$
X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
W=\begin{bmatrix}1&0&-1\\2&1&0\end{bmatrix}.
$$

Both are physically two-by-three, but their product is not elementwise multiplication. Match the three input features to the three entries in each output's weight row:

$$
\begin{aligned}
Y_{0,0}&=1(1)+2(0)+3(-1)=-2,\\
Y_{0,1}&=1(2)+2(1)+3(0)=4,\\
Y_{1,0}&=4(1)+5(0)+6(-1)=-2,\\
Y_{1,1}&=4(2)+5(1)+6(0)=13.
\end{aligned}
$$

Thus $Y=\begin{bmatrix}-2&4\\-2&13\end{bmatrix}$. For the final cell, a contiguous input reads offsets 3, 4, 5 and a contiguous weight reads offsets 3, 4, 5. With the earlier padded output, the result goes to `1 + 1*4 + 1`, offset 6. The arithmetic shapes and physical output pitch are separate facts.

### Transpose changes access steps, not stored bytes

Generalize the product to $C=\operatorname{op}(A)\operatorname{op}(B)$, where each `op` is either unchanged or transposed. The effective left matrix must have shape $[M,K]$, the effective right matrix $[K,N]$, and the output $[M,N]$.

For a physical matrix with `rows=R`, `cols=C`, and row stride `ld`, the effective shape and addressing steps are:

| Operation | Effective rows | Effective columns | Step when effective row increases | Step when effective column increases |
|---|---:|---:|---:|---:|
| Unchanged | R | C | `ld` | 1 |
| Transposed | C | R | 1 | `ld` |

Transpose preserves the physical leading dimension. A stored two-by-three matrix becomes effectively three-by-two while retaining its original, possibly padded, row stride.

In the product loop, the left operand is indexed by $i,p$, while the right operand is indexed by $p,j$. Name their effective steps `ai`, `ak`, `bk_stride`, and `bj`. Then the addresses are

$$
\begin{aligned}
\operatorname{index}_A(i,p)&=o_A+i\,a_i+p\,a_k,\\
\operatorname{index}_B(p,j)&=o_B+p\,b_k+j\,b_j.
\end{aligned}
$$

The step symbols count elements. Select them once from the flags; one reduction loop handles all four combinations.

[Reference BLAS `SGEMM`][gemm] provides a comparison for transpose flags, effective dimensions, and accumulation. Its Fortran convention is column-major; ours is purpose-built row-major C, not a BLAS wrapper.

### The two other products already have a place

Later, an incoming output-gradient matrix $G$ will have shape $[M,N]$. With our fixed weight orientation, the products needed by the linear layer's derivatives will have shapes

$$
GW:[M,K],\qquad G^{\mathsf T}X:[N,K].
$$

Chapter 5 will derive why these are the gradients. Here, establish that the required matrix operations actually work. With $G=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ and the same $X,W$,

$$
GW=\begin{bmatrix}5&2&-1\\11&4&-3\end{bmatrix},\qquad
G^{\mathsf T}X=\begin{bmatrix}13&17&21\\18&24&30\end{bmatrix}.
$$

For example, $(GW)_{0,0}=1(1)+2(2)=5$, while $(G^{\mathsf T}X)_{0,0}=1(1)+3(4)=13$. Their reduction axes are different: output features in the first case, token rows in the second. An API that silently assumes every reduction has the forward input-feature length cannot implement both correctly.

## Decide what a second call means

The write mode is explicit:

$$
\begin{aligned}
\texttt{DL\_SET}:\quad&C_{i,j}\leftarrow\sum_p A'_{i,p}B'_{p,j},\\
\texttt{DL\_ADD}:\quad&C_{i,j}\leftarrow C_{i,j}+\sum_p A'_{i,p}B'_{p,j}.
\end{aligned}
$$

$A'$ and $B'$ denote the effective, possibly transposed matrices. `DL_SET` must not read the old output. Its elements may be uninitialized. `DL_ADD` reads initialized output elements and adds one new contribution. Calling `DL_ADD` immediately on fresh uninitialized allocation is a caller error.

Compute the dot product from zero in both modes. Only after completing it should `DL_ADD` include the old output once. Two identical forward calls, first SET then ADD, turn the fixture into $\begin{bmatrix}-4&8\\-4&26\end{bmatrix}$. A third SET call restores the original answer.

Do not implement overwrite as `product + 0 * old_output`: that still reads old storage, and zero times NaN remains NaN. The BLAS specification similarly distinguishes the zero-output-coefficient case, where the old output need not be initialized. Our two-mode interface omits general scalar coefficients because these two operations are sufficient here.

### Output may not destroy an input

Read-only inputs may share storage. Output must not overlap either input span: an early store could corrupt a later dot product.

We deliberately use a conservative span test. If two views share the same canonical base, their spans $[a_0,a_1)$ and $[c_0,c_1)$ intersect exactly when

$$
a_0<c_1\quad\text{and}\quad c_0<a_1.
$$

We compare the base pointers for equality, then compare **integer offsets**. We do not order or subtract pointers to unrelated allocations. Two different live owners have different canonical bases; views in disjoint parts of one owner are also accepted.

The span test includes row gaps. A two-row, one-column view at offsets 0 and 3 has span $[0,4)$. A proposed output at offsets 1 and 4 has span $[1,5)$. Their actual cells are disjoint, but this interface rejects the pair. The simpler check costs some otherwise-valid layouts. Use separate storage or disjoint enclosing spans rather than bypassing it by rebasing a view.

No `restrict` qualifier is used. It would not validate overlap for us; shared read-only inputs remain legal.

### Structural failure and numerical failure are different

All flag, geometry, shape, and overlap checks run before any arithmetic output is written. On those failures, the output remains unchanged.

The arithmetic uses double products and a double accumulator, then stores one float per accepted output cell. Both operands are widened **before** multiplication. This does not make the result mathematically exact; it provides a deliberate, readable reference accumulation policy.

Before converting back to float, reject a non-finite sum or a magnitude greater than `FLT_MAX`. Underflow or ordinary rounding on a valid conversion is not an error in this interface. `DL_ADD` checks the result after including the old output, so an initialized NaN there also fails.

There is no rollback allocation. With $A=[\texttt{FLT\_MAX}]$, $B=[0,2]$, and output $[-9,-9]$, the first cell becomes zero before the second exceeds float range. Output is then $[0,-9]$. Discard the **whole** result on `DL_NUMERIC`; retrying ADD on partially modified storage would double-count some contributions.

This is an explicit per-cell failure policy, not a universal numerical-API guarantee.

```c title="include/dl/kernels.h"
#ifndef DL_KERNELS_H
#define DL_KERNELS_H

#include "dl/tensor.h"

typedef enum { DL_NOTRANS = 0, DL_TRANS = 1 } DlTranspose;
typedef enum { DL_SET = 0, DL_ADD = 1 } DlWriteMode;

/* C = op(A) op(B), or C += op(A) op(B). No allocation or retained pointers.
 * Views must keep truthful live base/count metadata from their owners.
 * Inputs may overlap each other. Output's bounding span must not overlap
 * either input's span, including gaps between rows. All dimensions positive.
 * A/B elements must be initialized. SET never reads old C elements;
 * ADD requires initialized C elements. Only logical C elements are written.
 * Geometry/shape/alias errors leave C unchanged. DL_NUMERIC may follow
 * earlier cell writes: discard the whole result on that status.
 * Products and reduction use double; each accepted result is stored as float. */
DlStatus dl_matmul_ref(DlConstMatrix a, DlTranspose ta,
                       DlConstMatrix b, DlTranspose tb,
                       DlMatrix c, DlWriteMode mode);

#endif
```

C enums can receive values outside their named alternatives, so validate the flags. The ternary `condition ? first : second` selects effective dimensions and strides after that validation.

Start `src/kernels.c` with its local overlap helper. `static` makes the helper private to this translation unit. Its arguments must already have validated span ends.

```c title="src/kernels.c"
#include "dl/kernels.h"

#include <float.h>
#include <math.h>

/* Validated spans only. Compare offsets only after matching root bases. */
static int spans_overlap(DlConstMatrix a, size_t a_end,
                         DlConstMatrix b, size_t b_end)
{
    return a.base == b.base && a.offset < b_end && b.offset < a_end;
}

```

The first part of the public function validates descriptors, then checks effective dimensions. `m`, `k`, and `n` describe the operation, not necessarily the physical input shapes. The right operand has its own candidate reduction length `bk`, which must agree with `k`.

```c title="src/kernels.c"
DlStatus dl_matmul_ref(DlConstMatrix a, DlTranspose ta,
                       DlConstMatrix b, DlTranspose tb,
                       DlMatrix c, DlWriteMode mode)
{
    if ((ta != DL_NOTRANS && ta != DL_TRANS) ||
        (tb != DL_NOTRANS && tb != DL_TRANS) ||
        (mode != DL_SET && mode != DL_ADD)) {
        return DL_ARGUMENT;
    }
    const DlConstMatrix cr = dl_matrix_read(c);
    size_t a_end = 0, b_end = 0, c_end = 0;
    DlStatus status = dl_matrix_end(a, &a_end);
    if (status == DL_OK) {
        status = dl_matrix_end(b, &b_end);
    }
    if (status == DL_OK) {
        status = dl_matrix_end(cr, &c_end);
    }
    if (status != DL_OK) {
        return status;
    }
    const size_t m = ta == DL_NOTRANS ? a.rows : a.cols;
    const size_t k = ta == DL_NOTRANS ? a.cols : a.rows;
    const size_t bk = tb == DL_NOTRANS ? b.rows : b.cols;
    const size_t n = tb == DL_NOTRANS ? b.cols : b.rows;
    if (k != bk || c.rows != m || c.cols != n) {
        return DL_SHAPE;
    }
    if (spans_overlap(a, a_end, cr, c_end) ||
        spans_overlap(b, b_end, cr, c_end)) {
        return DL_ALIAS;
    }
```

Continue inside the same function with the access steps and complete reduction. The final brace below closes `dl_matmul_ref`.

```c title="src/kernels.c"
    const size_t ai = ta == DL_NOTRANS ? a.ld : 1;
    const size_t ak = ta == DL_NOTRANS ? 1 : a.ld;
    const size_t bk_stride = tb == DL_NOTRANS ? b.ld : 1;
    const size_t bj = tb == DL_NOTRANS ? 1 : b.ld;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += (double)a.base[a.offset + i * ai + p * ak]
                     * (double)b.base[b.offset + p * bk_stride + j * bj];
            }
            const size_t ci = c.offset + i * c.ld + j;
            if (mode == DL_ADD) {
                sum += (double)c.base[ci];
            }
            if (!isfinite(sum) || fabs(sum) > (double)FLT_MAX) {
                return DL_NUMERIC;
            }
            c.base[ci] = (float)sum;
        }
    }
    return DL_OK;
}
```

For each $i,j$, `p` visits `0` through `k-1` using the derived addresses. Validation bounded the maximum physical offset; transposition maps effective coordinates back into that same rectangle. All nonnegative intermediate index products/sums therefore fit too.

In forward mode, the steps are `ai=X.ld`, `ak=1`, `bk_stride=1`, `bj=W.ld`: both dot-product operands walk across physical rows. For $GW$, the right reduction steps by `W.ld` instead. For $G^{\mathsf T}X$, the left reduction steps by `G.ld`. A logical transpose is cheap to represent, but its memory access pattern can be different. Chapter 6 will optimize only after these reference answers are established.

This kernel performs $MNK$ multiplications and similar additions. Beyond caller-owned inputs/output, it uses fixed scalar storage, allocates nothing, and retains no pointers. This is not a model-memory or CPU-speed claim.

## Build tests that know the intended numbers

Avoid reproducing the kernel's indexing in its oracle. Use hand-calculated answers, non-square shapes, metadata failures, and guard cells.

Use explicit failures rather than removable `assert` checks. These helpers print labels and, for status mismatches, actual and expected names.

```c title="tests/test_kernels.c"
#include "dl/kernels.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int expect(int condition, const char *name)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", name);
        return 0;
    }
    return 1;
}

static int status_is(DlStatus actual, DlStatus wanted, const char *name)
{
    if (actual != wanted) {
        fprintf(stderr, "FAIL: %s: got %s, wanted %s\n", name,
                dl_status_name(actual), dl_status_name(wanted));
        return 0;
    }
    return 1;
}

```

Use Chapter 3's finite-first comparison with fixed absolute and relative tolerances of $10^{-6}$. Our integer fixtures should actually be exact at this scale, but the helper is also suitable for the nearby finite values used later. As before, an overflowed error or tolerance calculation conservatively fails rather than accepting a dubious comparison.

```c title="tests/test_kernels.c"
static int close_finite(double actual, double expected)
{
    if (!isfinite(actual) || !isfinite(expected)) {
        return 0;
    }
    const double error = fabs(actual - expected);
    const double bound = 1e-6 + 1e-6 * fmax(fabs(actual), fabs(expected));
    return isfinite(error) && isfinite(bound) && error <= bound;
}

```

These test helpers fill owners, write fixtures, and compare logical coordinates. They require valid views and sufficiently long fixture arrays. `matrix_is` reports the first differing coordinate.

```c title="tests/test_kernels.c"
/* Test helpers borrow valid views and sufficiently long fixture arrays. */
static void fill_owner(DlTensor *owner, float value)
{
    for (size_t i = 0; i < owner->count; ++i) {
        owner->data[i] = value;
    }
}

static void write_matrix(DlMatrix matrix, const float *values)
{
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            matrix.base[matrix.offset + i * matrix.ld + j] =
                values[i * matrix.cols + j];
        }
    }
}

static int matrix_is(DlMatrix matrix, const float *values, const char *name)
{
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            const float actual = matrix.base[matrix.offset + i * matrix.ld + j];
            const float wanted = values[i * matrix.cols + j];
            if (!close_finite((double)actual, (double)wanted)) {
                fprintf(stderr, "FAIL: %s [%zu,%zu]: %.9g != %.9g\n",
                        name, i, j, (double)actual, (double)wanted);
                return 0;
            }
        }
    }
    return 1;
}

```

### Establish a dense layout and reject impossible sizes

First reject impossible element/byte counts without allocating them and verify unchanged failure outputs. These requests test arithmetic, not memory exhaustion:

```c title="tests/test_kernels.c"
static int check_storage(void)
{
    DlTensor tensor = {0};
    DlMatrix rows = {0};
    size_t result = 17;
    int ok = 0;
    const size_t shape[3] = {2, 3, 4};
    const size_t bad_shape[2] = {2, 0};
    const size_t huge_shape[2] = {SIZE_MAX, 2};
    const size_t huge_bytes[1] = {SIZE_MAX / sizeof(float) + 1};
    if (!status_is(dl_size_mul(SIZE_MAX, 2, &result), DL_OVERFLOW,
                   "element multiplication overflow") ||
        !expect(result == 17, "failed multiply keeps result") ||
        !status_is(dl_size_add(SIZE_MAX, 1, &result), DL_OVERFLOW,
                   "element addition overflow") ||
        !expect(result == 17, "failed add keeps result") ||
        !status_is(dl_size_mul(0, SIZE_MAX, &result), DL_OK, "zero product") ||
        !expect(result == 0, "zero product value") ||
        !status_is(dl_tensor_create(&tensor, 0, shape), DL_ARGUMENT, "zero rank") ||
        !status_is(dl_tensor_create(&tensor, 2, bad_shape), DL_ARGUMENT,
                   "zero extent rejected") ||
        !status_is(dl_tensor_create(&tensor, 2, huge_shape), DL_OVERFLOW,
                   "shape product overflow") ||
        !status_is(dl_tensor_create(&tensor, 1, huge_bytes), DL_OVERFLOW,
                   "byte product overflow") ||
        !expect(tensor.data == NULL && tensor.count == 0 && tensor.rank == 0,
                "failed constructors keep empty owner") ||
        !status_is(dl_tensor_create(&tensor, 3, shape), DL_OK, "dense tensor")) {
        goto done;
    }
```

Continue with offset-valued elements and flattened rows. `float *const original` freezes the saved pointer, not its elements. A failed reconstruction must preserve that live allocation.

```c title="tests/test_kernels.c"
    for (size_t i = 0; i < tensor.count; ++i) {
        tensor.data[i] = (float)i;
    }
    float *const original = tensor.data;
    if (!expect(tensor.count == 24 && tensor.stride[0] == 12 &&
                tensor.stride[1] == 4 && tensor.stride[2] == 1,
                "dense strides") ||
        !status_is(dl_tensor_create(&tensor, 3, shape), DL_STATE,
                   "cannot reconstruct a live owner") ||
        !expect(tensor.data == original, "live owner preserved") ||
        !status_is(dl_tensor_rows(&tensor, &rows), DL_OK, "flatten rows") ||
        !expect(rows.base == tensor.data && rows.rows == 6 &&
                rows.cols == 4 && rows.ld == 4, "flatten does not copy") ||
        !expect(rows.base[rows.offset + 5 * rows.ld + 3] == 23.0f,
                "[1,2,3] and row [5,3] share offset 23")) {
        goto done;
    }
    printf("layout: [2,3,4] strides=[12,4,1]; row[5,3]=23\n");
    ok = 1;
done:
    dl_tensor_destroy(&tensor);
    dl_tensor_destroy(&tensor);
    return ok && expect(tensor.data == NULL && tensor.count == 0 &&
                        tensor.rank == 0, "destroy resets owner");
}

```

Offset 23 connects tensor and matrix coordinates; base equality establishes shared storage. Repeated destroy uses the same reset owner, and no expired view is dereferenced.

### Execute the projection and both future gradient products

Initialize every owner before any fallible construction, allowing one cleanup block. Inputs have shape $[2,3]$; output uses four cells of a nine-element guard allocation. `g`, `dx`, and `dw` exercise the already-calculated products, not an unseen backward routine.

```c title="tests/test_kernels.c"
static int check_projection(void)
{
    DlTensor x_owner = {0}, w_owner = {0}, y_owner = {0};
    DlTensor g_owner = {0}, dx_owner = {0}, dw_owner = {0};
    DlMatrix x = {0}, w = {0}, y = {0}, g = {0}, dx = {0}, dw = {0};
    const size_t shape23[2] = {2, 3};
    const size_t shape22[2] = {2, 2};
    const size_t guard_shape[1] = {9};
    const float xv[6] = {1, 2, 3, 4, 5, 6};
    const float wv[6] = {1, 0, -1, 2, 1, 0};
    const float yv[4] = {-2, 4, -2, 13};
    const float twice_y[4] = {-4, 8, -4, 26};
    const float gv[4] = {1, 2, 3, 4};
    const float dxv[6] = {5, 2, -1, 11, 4, -3};
    const float dwv[6] = {13, 17, 21, 18, 24, 30};
    int ok = 0;
    if (!status_is(dl_tensor_create(&x_owner, 2, shape23), DL_OK, "X owner") ||
        !status_is(dl_tensor_create(&w_owner, 2, shape23), DL_OK, "W owner") ||
        !status_is(dl_tensor_create(&y_owner, 1, guard_shape), DL_OK, "Y guards") ||
        !status_is(dl_tensor_create(&g_owner, 2, shape22), DL_OK, "G owner") ||
        !status_is(dl_tensor_create(&dx_owner, 2, shape23), DL_OK, "dX owner") ||
        !status_is(dl_tensor_create(&dw_owner, 2, shape23), DL_OK, "dW owner") ||
        !status_is(dl_tensor_rows(&x_owner, &x), DL_OK, "X view") ||
        !status_is(dl_tensor_rows(&w_owner, &w), DL_OK, "W view") ||
        !status_is(dl_matrix_view(&y_owner, 1, 2, 2, 4, &y), DL_OK, "Y view") ||
        !status_is(dl_tensor_rows(&g_owner, &g), DL_OK, "G view") ||
        !status_is(dl_tensor_rows(&dx_owner, &dx), DL_OK, "dX view") ||
        !status_is(dl_tensor_rows(&dw_owner, &dw), DL_OK, "dW view")) {
        goto done;
    }
```

Continue by filling fixtures and poisoning only logical output cells with NaNs; guards remain 777. SET must ignore those NaNs. This exposes unwanted reads without relying on uninitialized memory.

```c title="tests/test_kernels.c"
    write_matrix(x, xv);
    write_matrix(w, wv);
    write_matrix(g, gv);
    fill_owner(&y_owner, 777.0f);
    /* SET must ignore old logical output even when it contains NaNs. */
    const float poison[4] = {NAN, NAN, NAN, NAN};
    write_matrix(y, poison);
    if (!status_is(dl_matmul_ref(dl_matrix_read(x), DL_NOTRANS,
                                 dl_matrix_read(w), DL_TRANS, y, DL_SET),
                   DL_OK, "X W^T") || !matrix_is(y, yv, "projection") ||
        !status_is(dl_matmul_ref(dl_matrix_read(x), DL_NOTRANS,
                                 dl_matrix_read(w), DL_TRANS, y, DL_ADD),
                   DL_OK, "accumulate projection") ||
        !matrix_is(y, twice_y, "two contributions") ||
        !status_is(dl_matmul_ref(dl_matrix_read(x), DL_NOTRANS,
                                 dl_matrix_read(w), DL_TRANS, y, DL_SET),
                   DL_OK, "overwrite again") || !matrix_is(y, yv, "reset") ||
        !status_is(dl_matmul_ref(dl_matrix_read(g), DL_NOTRANS,
                                 dl_matrix_read(w), DL_NOTRANS, dx, DL_SET),
                   DL_OK, "G W") || !matrix_is(dx, dxv, "input-side product") ||
        !status_is(dl_matmul_ref(dl_matrix_read(g), DL_TRANS,
                                 dl_matrix_read(x), DL_NOTRANS, dw, DL_SET),
                   DL_OK, "G^T X") || !matrix_is(dw, dwv, "weight-side product")) {
        goto done;
    }
```

The calls check SET/ADD/SET and both derivative-product orientations. Uninitialized `dx`/`dw` become readable only after successful SET.

Finish by verifying guards, printing the known results, and releasing every successfully created owner. The `goto done` path is the initialized cleanup convention from Chapter 2.

```c title="tests/test_kernels.c"
    const size_t guards[5] = {0, 3, 4, 7, 8};
    for (size_t i = 0; i < 5; ++i) {
        if (!expect(y_owner.data[guards[i]] == 777.0f, "output guard unchanged")) {
            goto done;
        }
    }
    printf("projection: [-2,4; -2,13]; SET/ADD and guards pass\n");
    printf("products: G W=[5,2,-1; 11,4,-3]; G^T X=[13,17,21; 18,24,30]\n");
    ok = 1;
done:
    dl_tensor_destroy(&dw_owner);
    dl_tensor_destroy(&dx_owner);
    dl_tensor_destroy(&g_owner);
    dl_tensor_destroy(&y_owner);
    dl_tensor_destroy(&w_owner);
    dl_tensor_destroy(&x_owner);
    return ok;
}

```

### Do not let square shapes hide a transpose error

A two-by-two output alone is not enough. Use an effective $[2,3]$ left matrix and $[3,4]$ right matrix:

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
B=\begin{bmatrix}1&0&-1&2\\0&1&2&-1\\2&1&0&1\end{bmatrix}.
$$

Their expected product is

$$
AB=\begin{bmatrix}7&5&3&3\\16&11&6&9\end{bmatrix}.
$$

For example, its last entry is $4(2)+5(-1)+6(1)=9$. In four trials, physically store each input either normally or transposed, then pass matching flags. Every trial represents the same mathematical product. Fill row gaps with 777 so reading a gap as data produces an obvious discrepancy.

The views use disjoint regions of **one** 128-element owner, testing legal shared allocation. `arena` is only its variable name, not another allocator.

```c title="tests/test_kernels.c"
static int check_transposes(void)
{
    DlTensor arena = {0};
    const size_t shape[1] = {128};
    const float av[6] = {1, 2, 3, 4, 5, 6};
    const float bv[12] = {1, 0, -1, 2, 0, 1, 2, -1, 2, 1, 0, 1};
    const float wanted[8] = {7, 5, 3, 3, 16, 11, 6, 9};
    int ok = 0;
    if (!status_is(dl_tensor_create(&arena, 1, shape), DL_OK, "shared storage")) {
        goto done;
    }
    const DlTranspose modes[2] = {DL_NOTRANS, DL_TRANS};
    for (size_t ia = 0; ia < 2; ++ia) {
        for (size_t ib = 0; ib < 2; ++ib) {
            DlMatrix a = {0}, b = {0}, c = {0};
            const size_t ar = ia == 0 ? 2 : 3, ac = ia == 0 ? 3 : 2;
            const size_t br = ib == 0 ? 3 : 4, bc = ib == 0 ? 4 : 3;
            fill_owner(&arena, 777.0f);
            if (!status_is(dl_matrix_view(&arena, 1, ar, ac, ac + 2, &a),
                           DL_OK, "padded A") ||
                !status_is(dl_matrix_view(&arena, 33, br, bc, bc + 2, &b),
                           DL_OK, "padded B") ||
                !status_is(dl_matrix_view(&arena, 81, 2, 4, 6, &c),
                           DL_OK, "padded C")) {
                goto done;
            }
```

Continue the trial loops: pack the selected physical orientations, save storage, and run. Only offsets 81–84 and 87–90 may change. All other cells—including inputs and in-allocation padding—are checked.

```c title="tests/test_kernels.c"
            for (size_t i = 0; i < 2; ++i) {
                for (size_t p = 0; p < 3; ++p) {
                    const size_t index = ia == 0 ? i * a.ld + p : p * a.ld + i;
                    a.base[a.offset + index] = av[i * 3 + p];
                }
            }
            for (size_t p = 0; p < 3; ++p) {
                for (size_t j = 0; j < 4; ++j) {
                    const size_t index = ib == 0 ? p * b.ld + j : j * b.ld + p;
                    b.base[b.offset + index] = bv[p * 4 + j];
                }
            }
            float before[128];
            for (size_t i = 0; i < 128; ++i) {
                before[i] = arena.data[i];
            }
            if (!status_is(dl_matmul_ref(dl_matrix_read(a), modes[ia],
                                         dl_matrix_read(b), modes[ib], c, DL_SET),
                           DL_OK, "transpose combination") ||
                !matrix_is(c, wanted, "non-square product")) {
                goto done;
            }
            for (size_t i = 0; i < 128; ++i) {
                const int is_output = (i >= 81 && i < 85) || (i >= 87 && i < 91);
                if (!is_output && !expect(arena.data[i] == before[i],
                                          "inputs and every gap preserved")) {
                    goto done;
                }
            }
        }
    }
```

After the loops, both read-only operands are deliberately the **same** view. The product $AA^{\mathsf T}$ has entries $[14,32;32,77]$: its upper-right cell is $1(4)+2(5)+3(6)=32$. An overbroad rule rejecting all shared inputs would incorrectly reject this operation.

```c title="tests/test_kernels.c"
    /* Both read-only operands may be the same view. */
    DlMatrix a = {0}, c = {0};
    const float gram[4] = {14, 32, 32, 77};
    if (!status_is(dl_matrix_view(&arena, 1, 2, 3, 5, &a), DL_OK, "shared input") ||
        !status_is(dl_matrix_view(&arena, 81, 2, 2, 6, &c), DL_OK, "Gram output")) {
        goto done;
    }
    write_matrix(a, av);
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(a), DL_TRANS, c, DL_SET),
                   DL_OK, "overlapping inputs are legal") ||
        !matrix_is(c, gram, "A A^T")) {
        goto done;
    }
    puts("transpose: all four padded non-square cases and shared inputs pass");
    ok = 1;
done:
    dl_tensor_destroy(&arena);
    return ok;
}

```

### Reject malformed geometry before touching the output

Construct valid views, then reject short leading dimensions, zero rows, out-of-range starts/ends, and span overflow. Each rejection must preserve the output descriptor.

`DlMatrix saved = c` is a valid copy of a **view**. Comparing its fields individually avoids treating struct padding bytes as part of its logical value. This does not authorize copying `DlTensor` owners.

```c title="tests/test_kernels.c"
static int check_rejections(void)
{
    DlTensor a_owner = {0}, b_owner = {0}, c_owner = {0};
    DlMatrix a = {0}, b = {0}, c = {0};
    const size_t shape23[2] = {2, 3}, shape22[2] = {2, 2};
    int ok = 0;
    if (!status_is(dl_tensor_create(&a_owner, 2, shape23), DL_OK, "reject A") ||
        !status_is(dl_tensor_create(&b_owner, 2, shape23), DL_OK, "reject B") ||
        !status_is(dl_tensor_create(&c_owner, 2, shape22), DL_OK, "reject C") ||
        !status_is(dl_tensor_rows(&a_owner, &a), DL_OK, "reject A view") ||
        !status_is(dl_tensor_rows(&b_owner, &b), DL_OK, "reject B view") ||
        !status_is(dl_tensor_rows(&c_owner, &c), DL_OK, "reject C view")) {
        goto done;
    }
    fill_owner(&a_owner, 1.0f);
    fill_owner(&b_owner, 2.0f);
    fill_owner(&c_owner, 19.0f);
    DlMatrix saved = c;
    if (!status_is(dl_matrix_view(&a_owner, 0, 2, 3, 2, &c), DL_ARGUMENT,
                   "leading dimension too small") ||
        !status_is(dl_matrix_view(&a_owner, 0, 0, 3, 3, &c), DL_ARGUMENT,
                   "empty matrix rejected") ||
        !status_is(dl_matrix_view(&a_owner, 1, 2, 3, 3, &c), DL_RANGE,
                   "last cell exceeds allocation") ||
        !status_is(dl_matrix_view(&a_owner, SIZE_MAX, 1, 1, 1, &c), DL_RANGE,
                   "invalid starting offset") ||
        !status_is(dl_matrix_view(&a_owner, 0, SIZE_MAX, 1, 3, &c), DL_OVERFLOW,
                   "span multiplication overflow") ||
        !expect(c.base == saved.base && c.count == saved.count &&
                c.offset == saved.offset && c.rows == saved.rows &&
                c.cols == saved.cols && c.ld == saved.ld,
                "failed view leaves every field unchanged") ||
        !status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_SHAPE, "wrong inner dimension") ||
        !status_is(dl_matmul_ref(dl_matrix_read(a), (DlTranspose)7,
                                 dl_matrix_read(b), DL_TRANS, c, DL_SET),
                   DL_ARGUMENT, "invalid transpose flag")) {
        goto done;
    }
```

The cast creates an invalid flag solely for rejection testing. Untransposed $[2,3]$ times $[2,3]$ must fail: the reduction lengths three and two differ.

Continue by checking unchanged output values and the two overlap policies. The second overlap case has disjoint logical cells but intersecting enclosing spans, so its rejection is intentional.

```c title="tests/test_kernels.c"
    const float nineteen[4] = {19, 19, 19, 19};
    if (!matrix_is(c, nineteen, "structural failures do not write")) {
        goto done;
    }
    DlMatrix alias = {0};
    if (!status_is(dl_matrix_view(&a_owner, 0, 2, 2, 3, &alias), DL_OK,
                   "overlapping output view") ||
        !status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_TRANS, alias, DL_SET),
                   DL_ALIAS, "reject destructive output overlap")) {
        goto done;
    }
    /* A uses offsets 0,3; C uses 1,4. Their enclosing spans still intersect. */
    DlMatrix narrow = {0}, scalar = {0}, gap_output = {0};
    if (!status_is(dl_matrix_view(&a_owner, 0, 2, 1, 3, &narrow), DL_OK, "narrow") ||
        !status_is(dl_matrix_view(&b_owner, 0, 1, 1, 1, &scalar), DL_OK, "scalar") ||
        !status_is(dl_matrix_view(&a_owner, 1, 2, 1, 3, &gap_output), DL_OK, "gap") ||
        !status_is(dl_matmul_ref(dl_matrix_read(narrow), DL_NOTRANS,
                                 dl_matrix_read(scalar), DL_NOTRANS,
                                 gap_output, DL_SET), DL_ALIAS,
                   "conservative overlap includes row gaps")) {
        goto done;
    }
    for (size_t i = 0; i < a_owner.count; ++i) {
        if (!expect(a_owner.data[i] == 1.0f, "alias rejection keeps input")) {
            goto done;
        }
    }
    puts("rejections: invalid sizes, shapes, flags, and output overlap pass");
    ok = 1;
done:
    dl_tensor_destroy(&c_owner);
    dl_tensor_destroy(&b_owner);
    dl_tensor_destroy(&a_owner);
    return ok;
}

```

Overlap must fail before changing inputs. These tests use truthful live allocations; invalid geometry is checked, whereas forged allocations or expired pointers violate the caller contract.

### Observe the arithmetic failure boundary

Test partial numerical failure and NaN input. Smaller views borrow three-element owners. Overflow leaves output `[0,-9,-9]`: one written cell, one rejected cell, and one unused cell.

```c title="tests/test_kernels.c"
static int check_numerics(void)
{
    DlTensor a_owner = {0}, b_owner = {0}, c_owner = {0};
    DlMatrix a = {0}, b = {0}, c = {0};
    const size_t shape[1] = {3};
    int ok = 0;
    if (!status_is(dl_tensor_create(&a_owner, 1, shape), DL_OK, "numeric A") ||
        !status_is(dl_tensor_create(&b_owner, 1, shape), DL_OK, "numeric B") ||
        !status_is(dl_tensor_create(&c_owner, 1, shape), DL_OK, "numeric C") ||
        !status_is(dl_matrix_view(&a_owner, 0, 1, 1, 1, &a), DL_OK, "scalar A") ||
        !status_is(dl_matrix_view(&b_owner, 0, 1, 2, 2, &b), DL_OK, "two B cells") ||
        !status_is(dl_matrix_view(&c_owner, 0, 1, 2, 2, &c), DL_OK, "two C cells")) {
        goto done;
    }
    a_owner.data[0] = FLT_MAX;
    b_owner.data[0] = 0.0f;
    b_owner.data[1] = 2.0f;
    fill_owner(&c_owner, -9.0f);
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_NUMERIC, "finite double result too large for float") ||
        !expect(c_owner.data[0] == 0.0f && c_owner.data[1] == -9.0f &&
                c_owner.data[2] == -9.0f, "numeric error may leave a written prefix")) {
        goto done;
    }
    a_owner.data[0] = 1.0f;
    b_owner.data[0] = NAN;
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_NUMERIC, "non-finite operand")) {
        goto done;
    }
```

Continue with new views of the same owners. Dot $[16777216,1,-16777216]$ with three ones: double accumulation retains the one lost by sequential float accumulation. ADD onto initialized NaN must fail.

```c title="tests/test_kernels.c"
    if (!status_is(dl_matrix_view(&a_owner, 0, 1, 3, 3, &a), DL_OK, "wide dot A") ||
        !status_is(dl_matrix_view(&b_owner, 0, 3, 1, 1, &b), DL_OK, "wide dot B") ||
        !status_is(dl_matrix_view(&c_owner, 0, 1, 1, 1, &c), DL_OK, "wide dot C")) {
        goto done;
    }
    a_owner.data[0] = 16777216.0f;
    a_owner.data[1] = 1.0f;
    a_owner.data[2] = -16777216.0f;
    fill_owner(&b_owner, 1.0f);
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_SET),
                   DL_OK, "double dot accumulation") ||
        !expect(c_owner.data[0] == 1.0f, "small term survives the double reduction")) {
        goto done;
    }
    c_owner.data[0] = NAN;
    if (!status_is(dl_matmul_ref(dl_matrix_read(a), DL_NOTRANS,
                                 dl_matrix_read(b), DL_NOTRANS, c, DL_ADD),
                   DL_NUMERIC, "ADD cannot use a non-finite initial output")) {
        goto done;
    }
    puts("numerics: wide reduction, non-finite rejection, and partial failure pass");
    ok = 1;
done:
    dl_tensor_destroy(&c_owner);
    dl_tensor_destroy(&b_owner);
    dl_tensor_destroy(&a_owner);
    return ok;
}

```

Finish with a precision check and explicit failure exit. Each test cleans up before returning; no compiler release number is required.

```c title="tests/test_kernels.c"
int main(void)
{
    if (!expect(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53,
                "binary32/binary64 fixture precision") ||
        !check_storage() || !check_projection() || !check_transposes() ||
        !check_rejections() || !check_numerics()) {
        return EXIT_FAILURE;
    }
    puts("all kernel checks passed");
    return EXIT_SUCCESS;
}
```

## Compile and run the complete module boundary

The root Makefile uses `-Iinclude` to locate `dl/` headers. Its explicit source list excludes unrelated files containing another `main`.

One compiler invocation compiles the four separate translation units and links them. Listed headers trigger rebuilding. This simple whole-test rule needs no generated object dependencies.

```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2
LDLIBS = -lm
SOURCES = src/common.c src/tensor.c src/kernels.c tests/test_kernels.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h

.PHONY: all debug sanitize release test clean

all: debug

debug: build/debug/test-kernels
sanitize: build/sanitize/test-kernels
release: build/release/test-kernels

build/debug/test-kernels: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(SOURCES) -o $@ $(LDLIBS)

build/sanitize/test-kernels: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(SOURCES) -o $@ $(LDLIBS)

build/release/test-kernels: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(SOURCES) -o $@ $(LDLIBS)

test: debug sanitize release
	./build/debug/test-kernels
	./build/sanitize/test-kernels
	./build/release/test-kernels

clean:
	rm -rf build
```

Recipes begin with tabs. `xcrun clang` selects the native Apple compiler; a command-line assignment can override it. Require C17 and the stated options, not an exact release.

The [Clang floating-point documentation][clang-fp] describes why we keep fast-math disabled and then explicitly disable contraction. This establishes a controlled reference reduction; it does not claim that fused operations are intrinsically wrong. The [AddressSanitizer usage instructions][asan] explain the compile-and-link sanitizer flag and frame-pointer option. Sanitizers complement the numerical expectations rather than replacing them.

From the workspace root on the Mac:

```sh title="Terminal — native reference checks"
make test
```

For a compatible `clang` already on `PATH` in a different environment:

```sh title="Terminal — explicit compiler selection"
make CC=clang test
```

The normal output of **each** executable is:

```text title="Expected successful test output"
layout: [2,3,4] strides=[12,4,1]; row[5,3]=23
projection: [-2,4; -2,13]; SET/ADD and guards pass
products: G W=[5,2,-1; 11,4,-3]; G^T X=[13,17,21; 18,24,30]
transpose: all four padded non-square cases and shared inputs pass
rejections: invalid sizes, shapes, flags, and output overlap pass
numerics: wide reduction, non-finite rejection, and partial failure pass
all kernel checks passed
```

`make test` prints this block for debug, sanitizer, and optimized runs. All three passed with Clang 17 on x86_64 Linux without sanitizer diagnostics. This is not M4/macOS execution or timing; run the native checks on your machine.

These fixtures establish layout and failure behavior, not exhaustive correctness. Overflow tests fail before allocation; no real `malloc` failure is forced.

## Two mistakes that stay inside allocated memory

Try each change separately, inspect its explicit failure, and restore the baseline.

### Use width where the physical row pitch belongs

In `dl_matmul_ref`, temporarily replace only the `ci` declaration with:

```c title="Variation — incorrect output addressing in src/kernels.c"
            const size_t ci = c.offset + i * c.cols + j;
```

For the guarded output, row one now writes positions 3 and 4 instead of 5 and 6. These are still valid elements of the nine-float allocation, but they are not output cells. `matrix_is` encounters the old NaN at logical coordinate `[1,0]`, where it expected -2. An address sanitizer need not diagnose an in-allocation layout mistake; the explicit fixture does.

Run the individual binary after rebuilding:

```sh title="Terminal — inspect one deliberate failure"
make sanitize
./build/sanitize/test-kernels
```

Restore `c.offset + i * c.ld + j`. The output guard check then has a meaningful opportunity to pass rather than merely confirming that no heap boundary was crossed.

### Read old output even in overwrite mode

Temporarily change `if (mode == DL_ADD)` to an unconditional block:

```c title="Variation — incorrect old-output dependency in src/kernels.c"
            {
                sum += (double)c.base[ci];
            }
```

The first SET call reads its deliberately poisoned output. The kernel now returns `DL_NUMERIC` instead of `DL_OK`, and `status_is` reports the failed `X W^T` call. This is a semantic mistake even when every pointer and dimension is valid. Restore the conditional ADD block.

After restoring both variations:

```sh title="Terminal — restore the complete passing baseline"
make clean
make test
```

As one arithmetic variation, change the last element of weight row one from zero to one. Before running, predict that the forward output becomes $[-2,7;-2,19]$: each second output gains the last input feature. With the same $G$, the product $GW$ becomes $[5,2,1;11,4,1]$, while $G^{\mathsf T}X$ remains unchanged because it depends on $G$ and $X$, not the current weight values. Update the relevant forward, doubled-forward, and input-side expectations together. The two printed projection/product lines are literal fixture summaries, so update those labels as well when displaying the changed case; the coordinate comparisons, not those labels, verify the results. Then restore the baseline. This exercises the orientation rather than merely rerunning a build.

## Diagnose the first inconsistent layer

| Symptom | First useful investigation |
|---|---|
| A constructor reports size overflow | Check each extent and whether you accidentally passed bytes where elements were required. Do not remove the overflow guard to make an allocation proceed. |
| Flattening produces the wrong row count | Confirm the last axis is the feature axis and the owner is dense in the stated order. A view with gaps is not automatically a reshapeable dense tensor. |
| A product rejects otherwise plausible dimensions | Write physical and effective shapes separately. Apply each transpose flag to the shape before matching the reduction lengths. |
| The first output row passes, the second fails | Inspect `ld` versus `cols`, then the starting offset. Square contiguous fixtures often hide this mistake. |
| A transposed case fails only with padding | Preserve the physical leading dimension when selecting effective access steps. Do not substitute an effective width. |
| SET fails on poisoned output | Verify there is no read of old C in SET, even through a zero coefficient. |
| The second contribution replaces the first | Inspect the write mode. Gradient accumulation needs ADD onto initialized storage, not another SET. |
| An overlap rejection surprises you | Compare bounding spans, including inter-row gaps, and confirm the canonical base was preserved. |
| A numerical error leaves some changed cells | This is the documented per-cell failure boundary. Discard the result; do not retry ADD on its partial state. |
| Sanitizers are silent but output is wrong | Compare exact accessed offsets and hand-calculated values. Valid allocation bounds do not imply correct tensor coordinates. |

The runnable kernel layer now has dense owners, borrowed views, checked geometry, transpose products, SET/ADD, and explicit failure semantics. The foundation examples remain separate; there is still no loss, optimizer, or automatic differentiation.

Chapter 5 adds embeddings, stable loss, and elementary derivatives. Repeated-token and shared-parameter gradients will use accumulation; the reference products already establish their weight orientation.

## References

**Matrix-product interface:** Netlib reference BLAS, [`SGEMM`][gemm], especially its operation definition, transpose parameters, effective dimensions, leading dimensions, and output-coefficient contract. Its column-major convention is distinct from this chapter's row-major representation. No BLAS implementation is required or called here.

**Compiler arithmetic:** Clang Compiler User's Manual, [floating-point controls][clang-fp], specifically `-fno-fast-math`, `-ffp-contract`, and expression evaluation. These explain the reference build's numerical assumptions, not a mandated compiler release.

**Memory diagnostics:** Clang, [AddressSanitizer usage][asan]. The chapter's guard and expected-value checks handle logical errors that stay within valid allocations.

[gemm]: https://netlib.org/lapack/explore-html/dd/d09/group__gemm_ga8cad871c590600454d22564eff4fed6b.html
[clang-fp]: https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior
[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#usage
[common-h]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/include/dl/common.h
[common-c]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/src/common.c
[tensor-h]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/include/dl/tensor.h
[tensor-c]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/src/tensor.c
[kernels-h]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/include/dl/kernels.h
[kernels-c]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/src/kernels.c
[tests-c]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/tests/test_kernels.c
[makefile]: https://25d-228.github.io/gpt-c/media/chapter-04/reference/Makefile

