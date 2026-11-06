# Chapter 6 — Blocked C kernels and meaningful timing

A correct matrix product is not yet a useful performance baseline. Its arithmetic may be straightforward while its memory access order repeatedly fetches the same inputs. This chapter changes that order without changing the operator. We retain the scalar implementation as a reference, add a bounded-workspace blocked implementation, and measure complete kernel calls rather than a conveniently selected inner loop.

> **Learning objectives**  Trace an output tile through several reduction tiles; explain the purpose and cost of packing; preserve double-precision partial sums and SET/ADD semantics; handle incomplete tiles without reading padding; select a backend explicitly; and interpret timings that include dispatch, validation, packing, and stores.

**Prerequisites:** the complete code state from [Chapter 5, Part 1](https://25d-228.github.io/gpt-c/chapter-05.html) and [Part 2](https://25d-228.github.io/gpt-c/chapter-05-part-2.html), including their Chapter 4 modules. Chapter 2 supplies ownership and checked-size arithmetic; Chapter 3 supplies finite/non-finite comparisons and rounding. We do not need a decoder, optimizer, model weights, or a third-party matrix library.

**This chapter has two parts:** this page develops the kernel and its complete tests; [Part 2 — Timing complete kernel calls](https://25d-228.github.io/gpt-c/chapter-06-part-2.html) supplies the complete comparison application, updated Makefile, measurements, and fault experiments. Both parts form one implementation.

**Implementation estimate:** 2.00 hours: 1.25 for the blocked kernel and dispatch, 0.75 for equivalence checks and the timing experiment. Reading and unattended execution are separate. Record actual hands-on time rather than stretching a short implementation to fit an estimate.

## Start from one known code state

Work at the learner workspace root, not inside a foundation example. Keep all Chapter 5 files. Replace `include/dl/kernels.h` and the root `Makefile`; replace only the matrix-product prefix of `src/kernels.c`; create `tests/test_blocked.c` and `app/compare.c`. The embedding and backward-wrapper suffix of `src/kernels.c` remains unchanged. There are now fifteen files in this cumulative program.

All required changes appear below. For each named file, concatenate its titled code blocks in order. The kernel-source blocks replace everything from its first include through the end of the old `dl_matmul_ref`, stopping immediately **before** the existing `/* Validate all IDs before any caller can copy or clear its output. */` comment. Keep that comment and everything below it. Do not append a second definition of the reference kernel.

The exact runnable state is also available as companion files: [kernel header](https://25d-228.github.io/gpt-c/media/chapter-06/reference/include/dl/kernels.h), [kernel source](https://25d-228.github.io/gpt-c/media/chapter-06/reference/src/kernels.c), [new tests](https://25d-228.github.io/gpt-c/media/chapter-06/reference/tests/test_blocked.c), [comparison application](https://25d-228.github.io/gpt-c/media/chapter-06/reference/app/compare.c), and [Makefile](https://25d-228.github.io/gpt-c/media/chapter-06/reference/Makefile). The other ten files are unchanged from Chapter 5. Companion files are a way to compare implementations, not a second specification.

## The operation does not change when the traversal changes

Let $A'=\operatorname{op}(A)$ have shape $[M,K]$, $B'=\operatorname{op}(B)$ have shape $[K,N]$, and $C$ have shape $[M,N]$. The operation `op` either keeps a matrix's logical orientation or transposes it; it does not physically move the caller's storage. All dimensions are positive. For row $i$, column $j$, and reduction coordinate $p$:

$$
P_{i,j}=\sum_{p=0}^{K-1}A'_{i,p}B'_{p,j}.
$$

SET stores $P_{i,j}$. ADD stores $P_{i,j}+C^{\mathrm{old}}_{i,j}$. The old output participates **once**, after the whole product, not once per reduction tile.

The same operator supplies the three products of a linear layer. Use $R$ for token rows, $D$ for input width, and $O$ for output width. These names avoid confusing a layer's dimensions with the reduction dimension of each individual call.

| Use | Stored arrays | Product | Kernel $(M,N,K)$ | Orientation |
|---|---|---|---|---|
| Forward | $X:[R,D]$, $W:[O,D]$ | $XW^T$ | $(R,O,D)$ | NT |
| Input gradient | $G:[R,O]$, $W:[O,D]$ | $GW$ | $(R,D,O)$ | NN |
| Weight gradient | $G:[R,O]$, $X:[R,D]$ | $G^TX$ | $(O,D,R)$ | TN |

Here $G$ is the incoming output gradient, not a different weight layout. The first orientation letter describes $A$ and the second $B$: N means not transposed, T means transposed. TT is not needed by these three calls but remains part of the general kernel's contract and tests. A forward-only speed comparison cannot tell us how the two backward products behave.

A `DlConstMatrix` retains its owner's canonical `base` and allocation `count`; `offset`, `rows`, `cols`, and `ld` are in float elements. Columns in stored orientation have unit stride, and `ld` is the physical row stride. We do not rebase submatrix pointers. The owner must remain live throughout the call. Inputs may share storage with each other; the output's conservative bounding span must be disjoint from both inputs, including their row gaps.

## Follow the addresses before moving the loops

The scalar code loops over `i`, then `j`, then `p`. For one output cell it keeps a single double sum. In a non-transposed $B$, increasing `p` jumps by `B.ld`; the next value in the same stored column is in another row. In a transposed $B$, increasing `p` instead walks a stored row. The two orientations therefore expose different memory patterns despite doing the same number of multiplications.

We want the innermost loop to update several adjacent output columns using one loaded $A'$ value. For each fixed `i,p`, these updates are independent across `j`:

$$
S_{i,j}\leftarrow S_{i,j}+A'_{i,p}B'_{p,j}.
$$

They are not independent across `p`, because each update continues the same sum. We will interchange loops across independent output cells while preserving the order of each cell's reduction. An optimizing compiler may exploit those independent columns; this chapter adds no SIMD intrinsics, threads, or hardware-specific matrix calls.

**Blocking** bounds the working set by handling small rectangular regions. **Packing** copies an active input region into a predictable contiguous layout for the inner loop. They are separate choices: a blocked kernel need not pack, and a packed kernel still pays for its copies. The [ATLAS developer documentation](https://netlib.org/atlas/developer/atlas_contrib/node10.html) describes both copied and non-copied matrix kernels; its [general-product discussion](https://netlib.org/atlas/developer/atlas_contrib/node11.html) emphasizes shape-dependent copy overhead and incomplete blocks. Here we implement a deliberately smaller design, not ATLAS or a cache-autotuned library.

## Trace two reduction tiles by hand

Take the following matrices, with no transpose:

$$
A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
B=\begin{bmatrix}1&0&2&-1\\0&1&3&2\\2&-1&0&1\end{bmatrix}.
$$

The full product is:

$$
AB=\begin{bmatrix}7&-1&8&6\\16&-1&23&12\end{bmatrix}.
$$

For illustration, use an output tile of two rows by two columns and a reduction tile of length two. These are trace sizes, not the implementation constants. The first output tile covers columns 0 and 1. Its scratch accumulator begins at zero. Processing $p=0,1$ gives:

$$
S=\begin{bmatrix}1&2\\4&5\end{bmatrix}.
$$

The final reduction tile has only one term, $p=2$. Its contribution is:

$$
\begin{bmatrix}3\\6\end{bmatrix}
\begin{bmatrix}2&-1\end{bmatrix}
=\begin{bmatrix}6&-3\\12&-6\end{bmatrix}.
$$

Adding it to the **existing** scratch tile gives $S=\left[\begin{smallmatrix}7&-1\\16&-1\end{smallmatrix}\right]$. SET now stores those four values. If the four old output cells were all five, ADD stores $\left[\begin{smallmatrix}12&4\\21&4\end{smallmatrix}\right]$.

Clearing scratch when $p=2$ begins would keep only the final contribution. Adding the old fives after each reduction tile would include them twice. Both mistakes access valid memory; a memory sanitizer cannot establish this arithmetic invariant.

For a single output coordinate, name its double accumulator after $t$ terms $s^{(t)}$. Our invariant is:

$$
s^{(0)}=0,\qquad
s^{(t+1)}=\operatorname{round}_{64}
\left(s^{(t)}+
\operatorname{double}(A'_{i,t})\operatorname{double}(B'_{t,j})\right).
$$

The subscript 64 describes the binary64 arithmetic targeted here. Each product widens its float operands **before** multiplication. We visit $t=0,1,\ldots,K-1$ in order, retaining one double partial sum across all reduction tiles. We do not sum each tile into an independent subtotal and then combine subtotals; that would regroup additions. Only the completed value, including old $C$ for ADD, is narrowed to float.

The builds retain `-fno-fast-math -ffp-contract=off`. The first prevents unsafe floating-point transformations; the second explicitly disables multiply/add contraction, which otherwise changes rounding. These controls are documented in the [Clang floating-point manual](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior) and [argument reference](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-ffp-contract). They are not a universal cross-compiler bitwise-reproducibility guarantee. Our comparisons still test finite values and a stated numerical tolerance.

## A bounded panel and a bounded output tile

Use output blocks of at most eight rows by 32 columns and reduction blocks of at most 64 terms. Let their starting logical coordinates be `ii`, `jj`, and `pp`; let the active lengths be `im`, `jn`, and `pk`. The packed panel contains:

$$
\mathrm{panel}[p][j]=B'_{pp+p,\,jj+j},
\qquad 0\le p<pk,\quad 0\le j<jn.
$$

Regardless of $B$'s stored orientation, adjacent `j` values in `panel` are adjacent floats. The output accumulator `acc[i][j]` is double. Every value loaded from $A'$ can update `jn` columns, and the panel is reused by up to `im` output rows.

With four-byte floats and eight-byte doubles, the two arrays occupy:

$$
8\cdot32\cdot8+64\cdot32\cdot4=10240\ \text{bytes}.
$$

That is 10 KiB of array storage, independent of full matrix size. It is not the entire call-stack size or process memory. The compiler may add spills, alignment, and bookkeeping. These are automatic arrays whose lifetime is one call, not heap allocations and not variable-length arrays. No scratch pointer escapes. The enum constants below are compile-time integer constants, so the array bounds are fixed.

The panel is repacked for each output-row block. Over the complete call, that copies $\lceil M/8\rceil KN$ float elements, counting repeated reads of $B'$ but not other traffic. For a one-row product there is no reuse across output rows. A larger block might reduce copies but grow scratch and change compiler behavior. These constants are a bounded starting point, not a claim about the M4's optimal cache sizes.

Replace the entire header with the following. The existing embedding and backward declarations are included so no interface disappears during the change.

```c title="include/dl/kernels.h"
#ifndef DL_KERNELS_H
#define DL_KERNELS_H

#include "dl/tensor.h"

typedef enum { DL_NOTRANS = 0, DL_TRANS = 1 } DlTranspose;
typedef enum { DL_SET = 0, DL_ADD = 1 } DlWriteMode;
typedef enum { DL_BACKEND_REF = 0, DL_BACKEND_BLOCKED = 1 } DlBackend;

/* Fixed workspace bounds, not matrix-shape restrictions. */
enum { DL_BLOCK_M = 8, DL_BLOCK_N = 32, DL_BLOCK_K = 64 };

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

/* Same view, precision, and SET/ADD contract as the reference.
 * Scratch: M*N doubles and K*N floats, using the block constants above.
 * Numeric failure may write a different subset than the scalar traversal. */
DlStatus dl_matmul_blocked(DlConstMatrix a, DlTranspose ta,
                           DlConstMatrix b, DlTranspose tb,
                           DlMatrix c, DlWriteMode mode);

/* Explicit selection per call; no global backend or automatic fallback.
 * An unknown backend returns DL_ARGUMENT without writing C. */
DlStatus dl_matmul(DlBackend backend, DlConstMatrix a, DlTranspose ta,
                   DlConstMatrix b, DlTranspose tb,
                   DlMatrix c, DlWriteMode mode);

/* IDs are a separate live size_t array, disjoint from float storage.
 * ids_count == out.rows; table [V,D], out [M,D]; every id is < V.
 * Only selected table rows are read. Output must not overlap table's span.
 * No writes on structural/range/alias errors; numeric failure may be partial. */
DlStatus dl_embedding_forward(DlConstMatrix table, const size_t *ids,
                              size_t ids_count, DlMatrix out);

/* dy [M,D], dtable [V,D]. SET zeros every logical dtable cell once,
 * then scatters with addition, including repeated IDs. ADD preserves prior
 * contributions. dy and dtable spans must be disjoint. ADD requires all
 * logical dtable cells initialized; accessed values must be finite.
 * Structural/range/alias errors leave dtable unchanged; DL_NUMERIC may not. */
DlStatus dl_embedding_backward(DlConstMatrix dy, const size_t *ids,
                               size_t ids_count, DlMatrix dtable,
                               DlWriteMode mode);

/* For forward Y = X W^T, X [M,K], W [N,K], dY [M,N].
 * These convenience wrappers remain reference-only. To choose a backend,
 * call dl_matmul with NN for dX or TN for dW.
 * Independent calls inherit matmul's SET/ADD, alias, and failure rules.
 * Do not overwrite X or W before every backward consumer has finished. */
DlStatus dl_linear_backward_input(DlConstMatrix dy, DlConstMatrix w,
                                  DlMatrix dx, DlWriteMode mode);
DlStatus dl_linear_backward_weight(DlConstMatrix dy, DlConstMatrix x,
                                   DlMatrix dw, DlWriteMode mode);

#endif
```

`DlBackend` makes the choice visible at a call site. It is an enum, not a function pointer or a global setting. An arbitrary integer cast to the enum still needs checking. Unknown values return an argument error, rather than silently selecting a different implementation.

The two named linear-backward convenience functions remain reference-only, preserving earlier code and tests. New callers choose either backend through `dl_matmul`, using the orientation table above. We have not silently accelerated only a forward path while leaving its caller believing that every product switched.

## Validate once through the same geometry rules

The following blocks begin the replacement prefix of `src/kernels.c`. `MatmulPlan` contains logical dimensions and element strides, not owned storage or a persistent execution plan. Its `ai` and `ak` describe a step along $A'$'s row and reduction axes; `bk` and `bj` describe $B'$'s reduction and column axes. For example, a transposed stored $A$ has `ai = 1` and `ak = a.ld`.

`matmul_prepare` is `static`, so its name is private to this translation unit. Its callers always supply a valid local plan address. It writes the plan only after validating both inputs, the output, dimensions, flags, and overlap. Like the earlier `dl_matrix_end`, this validates declared geometry, not a pointer's actual lifetime or truthful allocation size.

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

typedef struct {
    size_t m, n, k;
    size_t ai, ak, bk, bj;
} MatmulPlan;

/* Private to this file; writes plan only after all geometry checks pass. */
static DlStatus matmul_prepare(DlConstMatrix a, DlTranspose ta,
                               DlConstMatrix b, DlTranspose tb,
                               DlMatrix c, DlWriteMode mode,
                               MatmulPlan *plan)
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
    *plan = (MatmulPlan){.m = m, .n = n, .k = k,
        .ai = ta == DL_NOTRANS ? a.ld : 1,
        .ak = ta == DL_NOTRANS ? 1 : a.ld,
        .bk = tb == DL_NOTRANS ? b.ld : 1,
        .bj = tb == DL_NOTRANS ? 1 : b.ld};
    return DL_OK;
}

```

Extracting validation avoids two implementations gradually disagreeing about malformed views. It also means reference-versus-blocked tests alone cannot prove the shared validator correct. Keep Chapter 4's independent size, alias, and hand-calculated fixtures; the Makefile continues to run them.

Here is the complete replacement reference function. Its reduction and final-write behavior are unchanged; it now obtains its dimensions and strides from `matmul_prepare`.

```c title="src/kernels.c"
DlStatus dl_matmul_ref(DlConstMatrix a, DlTranspose ta,
                       DlConstMatrix b, DlTranspose tb,
                       DlMatrix c, DlWriteMode mode)
{
    MatmulPlan q = {0};
    const DlStatus status = matmul_prepare(a, ta, b, tb, c, mode, &q);
    if (status != DL_OK) {
        return status;
    }
    for (size_t i = 0; i < q.m; ++i) {
        for (size_t j = 0; j < q.n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < q.k; ++p) {
                sum += (double)a.base[a.offset + i * q.ai + p * q.ak]
                     * (double)b.base[b.offset + p * q.bk + j * q.bj];
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

## Keep partial sums alive until a tile is complete

The blocked implementation first zeroes just the active accumulator rectangle. It then walks reduction tiles, packs the active part of $B'$, and updates that same rectangle. Only after `pp` reaches `q.k` does it read old output for ADD and store results.

For a dimension `q.m` and a valid start `ii`, compute the remaining length as `q.m - ii`. Select the smaller of that remainder and the block bound. This avoids forming a potentially overflowing `ii + block` merely to clamp it. Advancing by `im` reaches at most `q.m`, and `im` cannot be zero while the outer loop condition is true. The same reasoning applies to columns and reduction coordinates.

For $[M,N,K]=[9,33,65]$, the row lengths are 8 and 1, the column lengths 32 and 1, and the reduction lengths 64 and 1. Every combination is visited. Unused cells of `panel` and `acc` are never read, and output row padding is never written.

```c title="src/kernels.c"
DlStatus dl_matmul_blocked(DlConstMatrix a, DlTranspose ta,
                           DlConstMatrix b, DlTranspose tb,
                           DlMatrix c, DlWriteMode mode)
{
    MatmulPlan q = {0};
    const DlStatus status = matmul_prepare(a, ta, b, tb, c, mode, &q);
    if (status != DL_OK) {
        return status;
    }
    double acc[DL_BLOCK_M][DL_BLOCK_N];
    float panel[DL_BLOCK_K][DL_BLOCK_N];
    for (size_t ii = 0; ii < q.m; ) {
        const size_t im = q.m - ii < DL_BLOCK_M ? q.m - ii : DL_BLOCK_M;
        for (size_t jj = 0; jj < q.n; ) {
            const size_t jn = q.n - jj < DL_BLOCK_N ? q.n - jj : DL_BLOCK_N;
            for (size_t i = 0; i < im; ++i) {
                for (size_t j = 0; j < jn; ++j) {
                    acc[i][j] = 0.0;
                }
            }
            for (size_t pp = 0; pp < q.k; ) {
                const size_t pk = q.k - pp < DL_BLOCK_K
                    ? q.k - pp : DL_BLOCK_K;
                for (size_t p = 0; p < pk; ++p) {
                    for (size_t j = 0; j < jn; ++j) {
                        panel[p][j] = b.base[b.offset
                            + (pp + p) * q.bk + (jj + j) * q.bj];
                    }
                }
                for (size_t i = 0; i < im; ++i) {
                    for (size_t p = 0; p < pk; ++p) {
                        const double av = (double)a.base[a.offset
                            + (ii + i) * q.ai + (pp + p) * q.ak];
                        for (size_t j = 0; j < jn; ++j) {
                            acc[i][j] += av * (double)panel[p][j];
                        }
                    }
                }
                pp += pk;
            }
            for (size_t i = 0; i < im; ++i) {
                for (size_t j = 0; j < jn; ++j) {
                    const size_t ci = c.offset + (ii + i) * c.ld + jj + j;
                    double sum = acc[i][j];
                    if (mode == DL_ADD) {
                        sum += (double)c.base[ci];
                    }
                    if (!isfinite(sum) || fabs(sum) > (double)FLT_MAX) {
                        return DL_NUMERIC;
                    }
                    c.base[ci] = (float)sum;
                }
            }
            jj += jn;
        }
        ii += im;
    }
    return DL_OK;
}

```

The array indexes expose the layout transformation. Packing uses `b.offset + (pp+p)*q.bk + (jj+j)*q.bj`, exactly the source address of $B'_{pp+p,jj+j}$. Computing uses the contiguous `panel[p][j]` regardless of those source strides. The `av` load is outside the column loop, so it can serve the whole active row of scratch. No `restrict` promise is added: the input views may legally alias one another.

The scratch reduction uses `+=` even for SET. SET describes the destination's previous contents; it does not mean every partial product should overwrite its predecessor. Conversely, the output store reads its old value only inside `if (mode == DL_ADD)`. SET therefore works with fresh uninitialized output storage and with a test output filled with NaNs.

A double intermediate may exceed `FLT_MAX` and later cancel to a representable float. Checking the float range at every reduction-tile boundary would reject an operation the reference accepts. The range and finite checks belong at the final store, as before. A float accumulator or a float round-trip at each tile boundary would also be a different numerical implementation, not a harmless storage optimization.

Structural errors return before stores. Numeric failure may occur after some tiles or cells have been stored. The traversal is now tile-major rather than globally row-major, so the written subset may differ from the scalar kernel's subset. The public contract is still: **discard the entire failed result**. Never interpret a numeric failure as a valid row prefix, and never retry ADD on that partially modified buffer without reconstructing its old state.

## Dispatch without hidden state or retry

The final new kernel-source block selects a backend and returns its status directly. Each successful selection runs the shared validation inside that implementation. There is no allocation, background worker, automatic tuning, or fallback. A fallback after a partially written ADD operation would be incorrect unless it restored the destination first.

```c title="src/kernels.c"
DlStatus dl_matmul(DlBackend backend, DlConstMatrix a, DlTranspose ta,
                   DlConstMatrix b, DlTranspose tb,
                   DlMatrix c, DlWriteMode mode)
{
    switch (backend) {
    case DL_BACKEND_REF:
        return dl_matmul_ref(a, ta, b, tb, c, mode);
    case DL_BACKEND_BLOCKED:
        return dl_matmul_blocked(a, ta, b, tb, c, mode);
    default:
        return DL_ARGUMENT;
    }
}

```

Now retain the Chapter 5 suffix beginning at the `Validate all IDs` comment. Its gather, scatter, and reference-only backward wrappers have not changed.

## Make awkward layouts part of the check

A square dense matrix with a block-aligned dimension is a poor first test. It can hide swapped dimensions, incorrect leading dimensions, and skipped tails. The new test uses separate owners with two prefix guard elements, three padding elements per physical row, and a suffix guard. Views keep canonical bases and nonzero offsets.

The fixed guard is `-12345.0f`. Logical inputs use deterministic signed fractions, not all ones. For SET, logical outputs begin as NaNs; a stray dependence on old output will propagate a non-finite value and fail. For ADD, both output owners begin with the same nonzero pattern. Their padding is never initialized to NaN, so the guard check distinguishes data cells from gaps.

Create `tests/test_blocked.c`. These first helpers report status errors, construct checked padded storage, generate repeatable values, and test both guards and unchanged inputs. `padded` may have acquired an owner before a later step fails; it does not free it locally. Its caller always destroys all initialized owners, including on partial construction failure.

```c title="tests/test_blocked.c"
#include "dl/kernels.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int require_status(DlStatus got, DlStatus expected, const char *where)
{
    if (got != expected) {
        fprintf(stderr, "%s: status %d, expected %d\n",
                where, (int)got, (int)expected);
        return 0;
    }
    return 1;
}

/* Fixed small test shapes, but construction still checks the size arithmetic. */
static int padded(DlTensor *owner, DlMatrix *view, size_t rows, size_t cols)
{
    size_t ld = 0, count = 0;
    DlStatus status = dl_size_add(cols, 3, &ld);
    if (status == DL_OK) status = dl_size_mul(rows, ld, &count);
    if (status == DL_OK) status = dl_size_add(count, 4, &count);
    if (status == DL_OK) status = dl_tensor_create(owner, 1, &count);
    if (status == DL_OK) {
        status = dl_matrix_view(owner, 2, rows, cols, ld, view);
    }
    if (!require_status(status, DL_OK, "padded allocation")) return 0;
    for (size_t i = 0; i < owner->count; ++i) owner->data[i] = -12345.0f;
    return 1;
}

static float value_at(size_t i, size_t j, unsigned seed)
{
    const int q = (int)((i % 23 * 7 + j % 23 * 11 + seed) % 23) - 11;
    return (float)q / 17.0f;
}

static void fill(DlMatrix v, unsigned seed, int nan_cells)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            v.base[v.offset + i * v.ld + j] = nan_cells
                ? NAN : value_at(i, j, seed);
        }
    }
}

static int guards_ok(DlMatrix v)
{
    for (size_t x = 0; x < v.count; ++x) {
        int logical = 0;
        if (x >= v.offset) {
            const size_t d = x - v.offset;
            logical = d / v.ld < v.rows && d % v.ld < v.cols;
        }
        if (!logical && v.base[x] != -12345.0f) {
            fprintf(stderr, "guard changed at allocation element %zu\n", x);
            return 0;
        }
    }
    return 1;
}

static int inputs_ok(DlMatrix v, unsigned seed)
{
    if (!guards_ok(v)) return 0;
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (v.base[v.offset + i * v.ld + j] != value_at(i, j, seed)) {
                fputs("input modified\n", stderr);
                return 0;
            }
        }
    }
    return 1;
}

```

`guards_ok` converts each allocation index back to a candidate row and column relative to the view's offset. An index belongs to the logical matrix only when both are within the view. This is deliberately different from using the kernel's address loop to check itself. The constructor initializes every allocation element, so even guard reads are defined.

The next helper constructs stored dimensions from the requested logical product. A transposed left operand is stored as $[K,M]$, not $[M,K]$ with a flag pasted onto incompatible metadata. It invokes both backends with the same inputs and compares every logical result.

For reference value $r$ and candidate value $x$, acceptance requires finite values and:

$$
|x-r|\le 10^{-6}+2\cdot10^{-6}|r|.
$$

The absolute term handles reference values near zero. This is a fixture tolerance, not a proved error bound for arbitrarily scaled matrices. On the tested build these two reductions agree exactly, but the test does not require every toolchain to reproduce the same last bit. Large discrepancies demand diagnosis, not a larger tolerance chosen just to turn the output green.

```c title="tests/test_blocked.c"
static int one_case(size_t m, size_t n, size_t k,
                    DlTranspose ta, DlTranspose tb, DlWriteMode mode,
                    double *maximum_error)
{
    DlTensor owners[4] = {{0}};
    DlMatrix v[4] = {{0}};
    int ok = 0;
    if (!padded(&owners[0], &v[0], ta == DL_TRANS ? k : m,
                                      ta == DL_TRANS ? m : k) ||
        !padded(&owners[1], &v[1], tb == DL_TRANS ? n : k,
                                      tb == DL_TRANS ? k : n) ||
        !padded(&owners[2], &v[2], m, n) ||
        !padded(&owners[3], &v[3], m, n)) goto cleanup;
    fill(v[0], 1, 0);
    fill(v[1], 3, 0);
    fill(v[2], 5, mode == DL_SET);
    fill(v[3], 5, mode == DL_SET);
    for (int backend = 0; backend < 2; ++backend) {
        if (!require_status(dl_matmul((DlBackend)backend,
                dl_matrix_read(v[0]), ta, dl_matrix_read(v[1]), tb,
                v[2 + backend], mode), DL_OK, "product")) goto cleanup;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double expected = (double)v[2].base[v[2].offset + i*v[2].ld + j];
            const double actual = (double)v[3].base[v[3].offset + i*v[3].ld + j];
            const double error = fabs(actual - expected);
            const double tolerance = 1e-6 + 2e-6 * fabs(expected);
            if (!isfinite(expected) || !isfinite(actual) || error > tolerance) {
                fprintf(stderr, "M=%zu N=%zu K=%zu ta=%d tb=%d mode=%d "
                        "at [%zu,%zu]: %.9g vs %.9g\n", m, n, k,
                        (int)ta, (int)tb, (int)mode, i, j, actual, expected);
                goto cleanup;
            }
            if (error > *maximum_error) *maximum_error = error;
        }
    }
    ok = inputs_ok(v[0], 1) && inputs_ok(v[1], 3)
        && guards_ok(v[2]) && guards_ok(v[3]);
cleanup:
    for (size_t i = 0; i < 4; ++i) dl_tensor_destroy(&owners[i]);
    return ok;
}

```

The common cleanup label releases all four owners whether construction, the kernel, a comparison, or a guard check fails. Each borrowed view dies with that owner. `maximum_error` is a caller-owned diagnostic scalar and is not used to control allocation or kernel behavior.

### Test what happens between reduction tiles

The following two checks isolate precision rather than layout. First put `FLT_MAX` at reduction coordinate 0, its negative at coordinate 64, and 3 at coordinate 129; multiply by twos. The first product exceeds the float range but fits double. After cancellation the product is 6; adding old output 5 gives 11.

Next use $2^{24}$ at coordinate 0, 1 at coordinate 1, and $-2^{24}$ at coordinate 64, with an all-ones right operand. A retained double sum produces 1, then ADD produces 6. Rounding the partial sum to float after the first tile loses the one. The earlier numerical foundations explain why float cannot retain that increment at $2^{24}$.

```c title="tests/test_blocked.c"
/* A double partial sum may exceed FLT_MAX and cancel before the final store. */
static int reduction_checks(void)
{
    float a[130] = {0}, b[130] = {0}, out = 0.0f;
    const DlConstMatrix av = {a, 130, 0, 1, 130, 130};
    const DlConstMatrix bv = {b, 130, 0, 130, 1, 1};
    const DlMatrix cv = {&out, 1, 0, 1, 1, 1};
    for (size_t p = 0; p < 130; ++p) b[p] = 2.0f;
    a[0] = FLT_MAX;
    a[64] = -FLT_MAX;
    a[129] = 3.0f;
    for (int backend = 0; backend < 2; ++backend) {
        out = 5.0f;
        if (!require_status(dl_matmul((DlBackend)backend, av, DL_NOTRANS,
                bv, DL_NOTRANS, cv, DL_ADD), DL_OK, "wide partial") ||
            out != 11.0f) {
            fputs("double partial or ADD-once check failed\n", stderr);
            return 0;
        }
    }
    for (size_t p = 0; p < 130; ++p) {
        a[p] = 0.0f;
        b[p] = 1.0f;
    }
    a[0] = 16777216.0f;
    a[1] = 1.0f;
    a[64] = -16777216.0f;
    for (int backend = 0; backend < 2; ++backend) {
        out = 5.0f;
        if (!require_status(dl_matmul((DlBackend)backend, av, DL_NOTRANS,
                bv, DL_NOTRANS, cv, DL_ADD), DL_OK, "round once") ||
            out != 6.0f) {
            fputs("partial sum was rounded to float\n", stderr);
            return 0;
        }
    }
    return 1;
}

```

### Reject malformed operations before touching output

These cases check unknown backend/transpose/write-mode values, incompatible shapes, an insufficient declared allocation, and a destination sharing the input's canonical base. They then check that the previous contents survived. A separate square product confirms that sharing the **two input** views is legal. Non-finite input, non-finite old ADD output, and final float overflow return `DL_NUMERIC`; there is intentionally no unchanged-output assertion for those failures.

```c title="tests/test_blocked.c"
static int failure_checks(void)
{
    float a[4] = {1, 2, 3, 4}, b[4] = {1, 0, 0, 1};
    float c[4] = {9, 9, 9, 9};
    const DlConstMatrix av = {a, 4, 0, 2, 2, 2};
    const DlConstMatrix bv = {b, 4, 0, 2, 2, 2};
    DlMatrix cv = {c, 4, 0, 2, 2, 2};
    if (!require_status(dl_matmul((DlBackend)99, av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_SET), DL_ARGUMENT, "backend")) return 0;
    if (!require_status(dl_matmul_blocked(av, (DlTranspose)99, bv,
            DL_NOTRANS, cv, DL_SET), DL_ARGUMENT, "transpose")) return 0;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, (DlWriteMode)99), DL_ARGUMENT, "mode")) return 0;
    DlMatrix bad = cv;
    bad.rows = 1;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, bad, DL_SET), DL_SHAPE, "shape")) return 0;
    bad = cv;
    bad.count = 3;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, bad, DL_SET), DL_RANGE, "capacity")) return 0;
    bad = cv;
    bad.base = a;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, bad, DL_SET), DL_ALIAS, "alias")) return 0;
    for (size_t i = 0; i < 4; ++i) {
        if (c[i] != 9.0f || a[i] != (float)(i + 1)) {
            fputs("structural error wrote output\n", stderr);
            return 0;
        }
    }
    /* A and B are allowed to be the same input view. */
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, av,
            DL_NOTRANS, cv, DL_SET), DL_OK, "shared inputs")) return 0;
    if (c[0] != 7 || c[1] != 10 || c[2] != 15 || c[3] != 22) return 0;
    c[0] = NAN;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_ADD), DL_NUMERIC, "nonfinite old C")) return 0;
    a[0] = INFINITY;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_SET), DL_NUMERIC, "nonfinite input")) return 0;
    a[0] = FLT_MAX;
    b[0] = 2;
    if (!require_status(dl_matmul_blocked(av, DL_NOTRANS, bv,
            DL_NOTRANS, cv, DL_SET), DL_NUMERIC, "final float overflow")) return 0;
    return 1;
}

```

The next small example demonstrates the changed failure traversal rather than declaring one traversal superior. Row 0 produces finite values; row 1 overflows at column 0. The scalar kernel has already stored all of row 0 when it fails. With the default two-dimensional tile bounds, the blocked kernel encounters that failing row-1 cell before starting row 0's final column tile. Its row-0 column 32 therefore still contains the sentinel.

```c title="tests/test_blocked.c"
/* Do not depend on a row-major prefix after a numerical failure. */
static int failure_order_check(void)
{
    float a[2] = {1.0f, FLT_MAX}, b[33], scalar[66], blocked[66];
    for (size_t j = 0; j < 33; ++j) b[j] = j == 0 ? 2.0f : 1.0f;
    for (size_t x = 0; x < 66; ++x) scalar[x] = blocked[x] = -17.0f;
    const DlConstMatrix av = {a, 2, 0, 2, 1, 1};
    const DlConstMatrix bv = {b, 33, 0, 1, 33, 33};
    const DlMatrix sr = {scalar, 66, 0, 2, 33, 33};
    const DlMatrix br = {blocked, 66, 0, 2, 33, 33};
    if (!require_status(dl_matmul_ref(av, DL_NOTRANS, bv, DL_NOTRANS,
            sr, DL_SET), DL_NUMERIC, "scalar partial result") ||
        !require_status(dl_matmul_blocked(av, DL_NOTRANS, bv, DL_NOTRANS,
            br, DL_SET), DL_NUMERIC, "blocked partial result")) return 0;
    /* With 8-by-32 output tiles, row 0's last cell has not been committed. */
    if (scalar[32] != 1.0f || blocked[32] != -17.0f) {
        fputs("unexpected failure traversal for the default tile sizes\n", stderr);
        return 0;
    }
    return 1;
}

```

This fixture intentionally describes the default eight-by-32 output tiling. When experimenting with block sizes, update this traversal-specific fixture or replace it with a suitable boundary for the new sizes. The mathematical result tests should not depend on the tiling constants; the partial-failure illustration necessarily does.

The driver runs seven logical shapes, four orientation combinations, and two output modes: $7\cdot4\cdot2=56$ product comparisons. It includes exact blocks, just-below/just-above blocks, a reduction spanning three tiles, a single output row, and reduction length one.

```c title="tests/test_blocked.c"
int main(void)
{
    const size_t shapes[][3] = {
        {1, 1, 1}, {7, 31, 63}, {8, 32, 64}, {9, 33, 65},
        {17, 35, 129}, {1, 257, 129}, {129, 33, 1}
    };
    size_t cases = 0;
    double maximum_error = 0.0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s) {
        for (int ta = 0; ta < 2; ++ta) {
            for (int tb = 0; tb < 2; ++tb) {
                for (int mode = 0; mode < 2; ++mode) {
                    if (!one_case(shapes[s][0], shapes[s][1], shapes[s][2],
                            (DlTranspose)ta, (DlTranspose)tb, (DlWriteMode)mode,
                            &maximum_error)) return EXIT_FAILURE;
                    ++cases;
                }
            }
        }
    }
    if (!reduction_checks() || !failure_checks() || !failure_order_check()) {
        return EXIT_FAILURE;
    }
    printf("blocked: %zu padded orientation/mode cases passed; max_abs=%.9g\n",
           cases, maximum_error);
    puts("blocked: reduction precision and failure checks passed");
    return EXIT_SUCCESS;
}
```

The NT cases cover the forward orientation; NN covers the input-gradient orientation; TN covers the weight-gradient orientation. These are operator equivalence checks. The inherited loss and finite-difference tests remain valuable independent checks of the original derivatives; they do not become blocked whole-decoder tests merely because a blocked kernel now exists.

## Run the kernel check before adding the timer

At this point the new source and test are complete. A direct native build can run the blocked test before Part 2 replaces the Makefile:

```sh
mkdir -p build/debug
xcrun clang -std=c17 -O0 -g -Wall -Wextra -Wpedantic \
    -fno-fast-math -ffp-contract=off -Iinclude \
    src/common.c src/tensor.c src/kernels.c tests/test_blocked.c \
    -lm -o build/debug/test_blocked
./build/debug/test_blocked
```

A compatible non-Apple toolchain can use `clang` in place of `xcrun clang`. The successful ending in the checked builds is:

```text
blocked: 56 padded orientation/mode cases passed; max_abs=0
blocked: reduction precision and failure checks passed
```

The small tolerance in the test, rather than an exact printed zero, determines success. Continue to [Part 2](https://25d-228.github.io/gpt-c/chapter-06-part-2.html) to measure the complete calls, run all inherited checks through the updated Makefile, and perform the fault exercises. A passing matrix check is not itself a performance measurement.

## References for the kernel

[ATLAS's copied and non-copied kernels](https://netlib.org/atlas/developer/atlas_contrib/node10.html) and its [general-product construction](https://netlib.org/atlas/developer/atlas_contrib/node11.html) support the packing and tail-handling discussion. [Clang's floating-point controls](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior) and [contraction option](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-ffp-contract) specify the compiler settings used here. The derivations and complete source above define this particular implementation; no external matrix library is required.
