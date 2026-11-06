# Chapter 5 — Embeddings, stable loss, and elementary backward passes

**Part 1 — The forward and backward computation**

A token ID selects a row of numbers. A projection turns that row into one score per possible next token. A loss judges the score assigned to the actual next token. We will compute that scalar, then carry its derivatives back to both parameter arrays—without an automatic differentiation system.

The important connection is not merely that each routine runs. If a token occurs twice, its embedding receives two gradient contributions. If a loss is averaged over three rows, the division by three happens exactly once along every backward path. If a logit is very large, a useful loss should not disappear because two large numbers were subtracted in the wrong order.

> **Learning objectives**
> Implement embedding lookup and repeated-ID scatter-add; derive a stable mean cross-entropy and its gradient; obtain both projection gradients from the scalar product; distinguish a fresh gradient from an additional contribution; and check a complete small computation by perturbing its actual stored parameters.

**Prerequisites:** [Chapter 3](https://25d-228.github.io/gpt-c/chapter-03.html) and [Chapter 4](https://25d-228.github.io/gpt-c/chapter-04.html). Begin with Chapter 4's working root-level modules and `tests/test_kernels.c`. Keep the three foundation programs under `examples/` unchanged.

**Implementation estimate: 3.50 hours**—2.25 for the operators and their mathematical connections in code; 1.25 for integration, gradient checks, and boundary experiments. Reading is separate. Record actual implementation time rather than treating this estimate as a minimum.

## Extend the same numerical core

Replace `include/dl/kernels.h` with the version below and the root Makefile with the version in Part 2. Append the new functions to the existing `src/kernels.c`; keep its includes, `spans_overlap`, and `dl_matmul_ref` unchanged. Create `include/dl/loss.h`, `src/loss.c`, `tests/tiny.h`, `tests/test_loss.c`, and `tests/gradcheck.c`. Every new line is included across this chapter’s two parts. [Part 2](https://25d-228.github.io/gpt-c/chapter-05-part-2.html) supplies the complete tests, Makefile, and runnable checks. Multiple listings with the same filename are consecutive pieces in their displayed order.

The complete [companion source snapshot](https://25d-228.github.io/gpt-c/media/chapter-05/reference/Makefile) contains the Makefile plus the corresponding files under `include/dl/`, `src/`, and `tests/`; it is a convenience, not a missing prerequisite. Chapter 4's common/tensor modules and kernel test remain unchanged. The runnable state at the end is:

```text
include/dl/  common.h  tensor.h  kernels.h  loss.h
src/         common.c  tensor.c  kernels.c  loss.c
tests/       test_kernels.c  tiny.h  test_loss.c  gradcheck.c
Makefile
```

Recall that a `DlMatrix` borrows a canonical root `base`, a truthful allocation `count`, an `offset`, logical `rows` and `cols`, and physical row pitch `ld`, all in float elements. It does not own memory. `dl_matrix_read` creates const element access; it does not freeze or copy the original allocation. All owners must remain alive throughout the calls.

## Give every row a prediction target

Let $V$ be the vocabulary size, $D$ the embedding width, and $M$ the number of prediction rows. An integer token ID lies in $[0,V)$. We do not interpret it as a floating-point magnitude: ID 3 is an addressable category, not three times ID 1.

The embedding table is $E\in\mathbb{R}^{V\times D}$. Input IDs $u_i$, for $0\le i<M$, select hidden rows $H\in\mathbb{R}^{M\times D}$:

$$
H_{i,j}=E_{u_i,j}.
$$

The output projection has stored weights $W\in\mathbb{R}^{V\times D}$. The scores, called **logits**, are $Z=HW^T\in\mathbb{R}^{M\times V}$, with $Z_{i,v}=\sum_jH_{i,j}W_{v,j}$. Each row predicts a target ID $y_i$. Inputs and targets are separate arrays; target ID $y_i$ selects the answer being scored, not the embedding looked up at that row.

Our four-token sequence is `[0, 1, 0, 2]`. Its three inputs are `[0, 1, 0]`, and its next-token targets are `[1, 0, 2]`. Thus rows 0 and 2 read the same embedding but are judged against different answers. This small graph has no attention or context beyond the current ID; it isolates the embedding, output projection, and loss that the decoder will later connect through its internal layers.

The [official embedding reference][embedding] describes the same table-lookup interpretation. We use ordinary dense gradients: no padding ID, frequency-based gradient rescaling, or sparse tensor format is implicit in the implementation.

## Gather rows without moving their owner

For a padded table, the source index for coordinate $j$ of input row $i$ is:

$$
\text{table.offset}+u_i\,\text{table.ld}+j.
$$

The destination is `out.offset + i * out.ld + j`. These pitches need not be equal. With table offset 1, row pitch 3, and width 2, ID 2 selects offsets 7 and 8. It does not select offsets 5 and 6, which would incorrectly treat the table as densely packed.

We validate every ID before the first copy. Otherwise `[2, 1, 4]` in a vocabulary of size 4 could copy two rows before discovering that the final lookup is illegal. An ID's type is `size_t`, matching our index arithmetic; a file loader that begins with signed integers must reject negative values before conversion. A pointer and count cannot prove that the caller supplied a real array of that length.

Replace the header with this complete version. The existing transpose/write-mode declarations and matrix-product interface remain available; the added functions will share their storage rules.

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
 * Independent calls inherit matmul's SET/ADD, alias, and failure rules.
 * Do not overwrite X or W before every backward consumer has finished. */
DlStatus dl_linear_backward_input(DlConstMatrix dy, DlConstMatrix w,
                                  DlMatrix dx, DlWriteMode mode);
DlStatus dl_linear_backward_weight(DlConstMatrix dy, DlConstMatrix x,
                                   DlMatrix dw, DlWriteMode mode);

#endif
```

Append this private validator to `src/kernels.c`. `static` gives it translation-unit scope, just like Chapter 4's overlap helper. `expected` is the number of prediction rows, while `vocabulary` bounds the values stored inside the ID array; those are different constraints.

```c title="src/kernels.c"

/* Validate all IDs before any caller can copy or clear its output. */
static DlStatus check_ids(const size_t *ids, size_t count,
                           size_t expected, size_t vocabulary)
{
    if (ids == NULL) {
        return DL_ARGUMENT;
    }
    if (count != expected) {
        return DL_SHAPE;
    }
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= vocabulary) {
            return DL_RANGE;
        }
    }
    return DL_OK;
}

```

Then append the forward function. `dl_matrix_end` verifies declared spans before the ID-based multiplication occurs. Because each ID is less than the validated table row count, the largest permitted source index fits that span. Only selected table elements are read; unused rows and padding need not be inspected for numeric values.

```c title="src/kernels.c"
DlStatus dl_embedding_forward(DlConstMatrix table, const size_t *ids,
                              size_t ids_count, DlMatrix out)
{
    const DlConstMatrix out_read = dl_matrix_read(out);
    size_t table_end = 0, out_end = 0;
    DlStatus status = dl_matrix_end(table, &table_end);
    if (status == DL_OK) {
        status = dl_matrix_end(out_read, &out_end);
    }
    if (status != DL_OK) {
        return status;
    }
    if (table.cols != out.cols) {
        return DL_SHAPE;
    }
    status = check_ids(ids, ids_count, out.rows, table.rows);
    if (status != DL_OK) {
        return status;
    }
    if (spans_overlap(table, table_end, out_read, out_end)) {
        return DL_ALIAS;
    }
    for (size_t i = 0; i < out.rows; ++i) {
        for (size_t j = 0; j < out.cols; ++j) {
            const float value = table.base[table.offset + ids[i] * table.ld + j];
            if (!isfinite(value)) {
                return DL_NUMERIC;
            }
            out.base[out.offset + i * out.ld + j] = value;
        }
    }
    return DL_OK;
}

```

The overlap test deliberately rejects an output inside the table, even if this particular ID order might appear safe. A write to one table row could change a later lookup. Keeping the read table and output disjoint removes that ordering dependency. The existing conservative bounding-span check includes inter-row gaps and relies on the canonical root-base convention; rebasing a view would defeat it.

No heap allocation occurs. The caller owns both arrays, and the function keeps no pointers after returning. Structural errors leave the output unchanged. A non-finite selected value can be discovered after earlier copies, so `DL_NUMERIC` means discard the result, not “use the rows that happened to be written.”

## Reverse a lookup by adding to the selected row

Write $G_H=\partial L/\partial H$ for the incoming derivative of a scalar loss $L$. An entry of $E$ may have influenced several entries of $H$. The multivariable chain rule adds those paths:

$$
\frac{\partial L}{\partial E_{v,j}}
=\sum_{i=0}^{M-1}\sum_{k=0}^{D-1}
 \frac{\partial L}{\partial H_{i,k}}
 \frac{\partial H_{i,k}}{\partial E_{v,j}}
=\sum_{i:u_i=v}(G_H)_{i,j}.
$$

The local derivative is one when both the selected row and coordinate match, and zero otherwise. There is no derivative with respect to integer IDs in this operation; they route continuous parameter derivatives.

Take IDs `[2, 1, 2]` and incoming rows `[[1, 2], [3, 4], [5, 6]]`. Starting from zero, row 2 first receives `[1, 2]`, row 1 receives `[3, 4]`, and row 2 then becomes `[6, 8]`. A last-write-wins assignment would leave `[5, 6]` and silently remove the first path. This is a numerical error with perfectly valid memory addresses.

`DL_SET` describes the whole call: start a fresh table gradient. It does **not** mean assign on each individual visit. Zero all logical cells once, including unselected rows, and then scatter with addition. `DL_ADD` skips the clearing pass so contributions from an earlier caller remain. Neither mode divides by the number of occurrences of a token.

```c title="src/kernels.c"
DlStatus dl_embedding_backward(DlConstMatrix dy, const size_t *ids,
                               size_t ids_count, DlMatrix dtable,
                               DlWriteMode mode)
{
    if (mode != DL_SET && mode != DL_ADD) {
        return DL_ARGUMENT;
    }
    const DlConstMatrix dr = dl_matrix_read(dtable);
    size_t dy_end = 0, dt_end = 0;
    DlStatus status = dl_matrix_end(dy, &dy_end);
    if (status == DL_OK) {
        status = dl_matrix_end(dr, &dt_end);
    }
    if (status != DL_OK) {
        return status;
    }
    if (dy.cols != dtable.cols) {
        return DL_SHAPE;
    }
    status = check_ids(ids, ids_count, dy.rows, dtable.rows);
    if (status != DL_OK) {
        return status;
    }
    if (spans_overlap(dy, dy_end, dr, dt_end)) {
        return DL_ALIAS;
    }
    if (mode == DL_SET) {
        for (size_t v = 0; v < dtable.rows; ++v) {
            for (size_t j = 0; j < dtable.cols; ++j) {
                dtable.base[dtable.offset + v * dtable.ld + j] = 0.0f;
            }
        }
    }
    for (size_t i = 0; i < dy.rows; ++i) {
        for (size_t j = 0; j < dy.cols; ++j) {
            const size_t dst = dtable.offset + ids[i] * dtable.ld + j;
            const double sum = (double)dtable.base[dst]
                + (double)dy.base[dy.offset + i * dy.ld + j];
            if (!isfinite(sum) || fabs(sum) > (double)FLT_MAX) {
                return DL_NUMERIC;
            }
            dtable.base[dst] = (float)sum;
        }
    }
    return DL_OK;
}

```

The zeroing loop respects `dtable.ld`; padding is not part of the gradient. In the padded fixture above, the two visits to ID 2 both update offsets 7 and 8. With an existing value of 10 in every gradient cell, `DL_ADD` produces row 2 `[16, 18]` and leaves unselected rows `[10, 10]`.

Each individual addition widens its two operands to double, checks the result, and stores back to float. This is not a double accumulator retained for the entire table: repeated visits round after each store. It uses no extra table-sized scratch buffer and preserves a deterministic row order, but many small contributions can still be lost against a large existing float. A later optimizer does not erase that limitation.

On numerical failure, even a `DL_SET` call may already have cleared the table and written earlier contributions. Never retry a failed `DL_ADD` over the same partially changed buffer. Reinitialize the appropriate gradient state and repeat the whole valid computation instead. All IDs and geometry are checked before clearing, so those errors still leave the destination unchanged.

## Convert scores into a stable mean loss

For one row, suppress the row index and call the class scores $z_v$. Softmax assigns a probability to each class:

$$
p_v=\frac{\exp(z_v)}{\sum_{k=0}^{V-1}\exp(z_k)}.
$$

Probabilities are nonnegative and sum to one in exact arithmetic. Logits themselves can be negative, need not sum to anything, and must not be passed through softmax twice. For target $y$, the cross-entropy is the negative natural logarithm of its probability:

$$
\ell(z,y)=-\log p_y
=\log\left(\sum_{v=0}^{V-1}\exp(z_v)\right)-z_y.
$$

Here `log` is the natural logarithm, so the loss is measured in nats. With $M$ prediction rows, our scalar objective is exactly:

$$
L=\frac{1}{M}\sum_{i=0}^{M-1}\ell(Z_{i,:},y_i).
$$

Every row counts once. There are no class weights, ignored targets, padding masks, or label smoothing. This is the unweighted integer-target mean case of the [official cross-entropy definition][cross-entropy]; its more general reduction rules must not be silently mixed into this narrower interface.

### Subtract the maximum before exponentiating

The mathematical expression is not yet a safe algorithm. For `[1000, 999, 998]`, computing `exp(1000)` overflows an ordinary binary64 result even though the three relative probabilities are harmless.

Let $m=\max_v z_v$. Factoring out $\exp(m)$ gives:

$$
s=\sum_v\exp(z_v-m),\qquad
p_v=\frac{\exp(z_v-m)}s,\qquad
\ell=(m-z_y)+\log s.
$$

Every exponent argument is nonpositive. At least one term is exactly `exp(0)`, so the denominator is positive. The maximum-shift construction is the safe-softmax algorithm in [Milakov and Gimelshein, Section 2][online]. This chapter uses separate passes; the online normalization recurrence belongs to the later tiled-attention implementation.

For target 2 in `[1000, 999, 998]`, shifted scores are `[0, -1, -2]` and the loss is $2+\log(1+e^{-1}+e^{-2})\approx2.407605964$. Subtracting 1000 from all three logits produces the same mathematical result. This invariance only concerns values actually represented in storage: adding a huge constant in float can first erase small differences, and no normalization routine can reconstruct them.

### Keep the target subtraction and a small logarithm separate

Two additional rearrangements matter. First, implement `(maximum - target) + log_sum`, not `(maximum + log_sum) - target`. If four logits all equal `FLT_MAX`, the correct loss is $\log4$. Adding that small quantity to a number near $3.4\times10^{38}$ loses it even in double; subtracting the target afterward then gives zero. The first order subtracts equal stored logits before adding the small normalization term.

Also widen **both operands before subtraction**. `(double)(float_max - negative_float_max)` is too late: the float subtraction can overflow before conversion. Our expression is `maximum - (double)target`, with `maximum` already double. The loss for `[FLT_MAX, -FLT_MAX]` and target 1 is approximately `2 * (double)FLT_MAX`, which fits in double but not float. That is why the loss output is a `double` while parameter and activation arrays remain `float`.

For a very confident correct prediction, ordinary `log(1 + tiny)` has a different problem: the addition can round to 1. Select one maximum index $k$ and accumulate only the other terms:

$$
r=\sum_{v\ne k}\exp(z_v-m),\qquad
s=1+r,\qquad
\ell=(m-z_y)+\operatorname{log1p}(r).
$$

`log1p(r)` computes $\log(1+r)$ without first rounding `1+r` as a separate user-level operation. Excluding one maximum, rather than every maximum, is essential: four equal logits give `r = 3`, not zero. This form follows the shifted algorithm in [Blanchard, Higham, and Higham, Section 4, Algorithm 5][accurate].

For `[0, -50]` with target 0, $r=e^{-50}\approx1.92875\times10^{-22}$. `log(1+r)` becomes zero in binary64 arithmetic, while `log1p(r)` retains a positive loss near $r$. This is a better forward computation, not a promise that arbitrarily tiny probabilities survive underflow.

## Differentiate the objective, not a probability buffer

Start with the unshifted symbolic form $\ell=\log\sum_k e^{z_k}-z_y$. The derivative of a logarithm contributes the reciprocal sum, and differentiating the selected exponential contributes $e^{z_v}$:

$$
\frac{\partial\ell}{\partial z_v}
=\frac{e^{z_v}}{\sum_k e^{z_k}}-\mathbf{1}_{v=y}
=p_v-\mathbf{1}_{v=y}.
$$

The symbol $\mathbf{1}_{v=y}$ is one when $v=y$ and zero otherwise. Averaging the rows gives the incoming projection gradient:

$$
(G_Z)_{i,v}=\frac{\partial L}{\partial Z_{i,v}}
=\frac{p_{i,v}-\mathbf{1}_{v=y_i}}{M}.
$$

There is no need to materialize a $V\times V$ softmax derivative matrix. The loss has already selected and combined the relevant paths. There is also no derivative of the target integer. The maximum used for numerical scaling is not a new model operation: the shifted expression is algebraically the same smooth loss. The final formula remains valid at tied maxima without choosing a special tie derivative.

Every gradient row sums to zero in exact arithmetic: the probabilities sum to one and exactly one target indicator is subtracted. That is the derivative counterpart of invariance to a common row shift. In float, expect a small residual, not bitwise zero for every input.

The target component has its own cancellation trap. For `[0,-50]`, `p_target` rounds to 1 in double, so `p_target - 1` would lose a representable small derivative. Use the equivalent expression:

$$
p_y-1=-\sum_{v\ne y}p_v
=-\frac{\sum_{v\ne y}\exp(z_v-m)}{s}.
$$

The code accumulates `not_target` as that numerator. For non-target classes it computes the positive probability directly; for the target it uses the negative complementary mass. Both are divided by $M$ once. If the small exponentials themselves underflow to zero, this technique cannot rescue them. No epsilon is added to the target probability: doing so would change both the objective and its derivative.

As a hand check, two rows of four zero logits have $L=\log4$. With targets `[0,3]`, their gradients are `[-3/8,1/8,1/8,1/8]` and `[1/8,1/8,1/8,-3/8]`. Dividing by four instead of two confuses classes with prediction rows; omitting the division gives the derivative of a summed, not mean, loss.

## Make value-only evaluation and backward agree

The loss routine optionally writes a gradient. Passing `NULL` for `dlogits` requests only the scalar value, using exactly the same loss calculation. That path will serve both evaluation and finite differences. It does not allocate a probability matrix or save a hidden tape.

Create the header below. The optional argument is a pointer to a matrix descriptor: callers pass `&gm` for a named gradient view. The routine reads that descriptor and writes through its `base`; it does not take ownership or change the descriptor. The `loss` pointer refers to a separate live `double`, never a cast into float storage. Likewise, target IDs occupy a separate live `size_t` array.

```c title="include/dl/loss.h"
#ifndef DL_LOSS_H
#define DL_LOSS_H

#include "dl/kernels.h"

/* logits [M,V], M,V > 0; targets is a separate live size_t[M] array.
 * Mean over M rows, natural logarithm, unweighted, no masks/ignored targets.
 * Every target is in [0,V); every logical logit must be finite.
 * loss points to a separate writable double, not into any input/output.
 * dlogits == NULL evaluates loss only. Otherwise *dlogits has shape [M,V],
 * is disjoint from logits, and receives d(mean loss)/d(logits).
 * SET never reads old gradients; ADD requires initialized finite gradients.
 * mode must be valid even in value-only calls. No allocation or saved tape.
 * Structural, ID, and preflight non-finite errors change no output.
 * Later DL_NUMERIC can leave partial gradient writes; loss is committed only
 * on success. Always discard the entire result after any failed call.
 * Keep canonical allocation bases; caller guarantees truthful live storage. */
DlStatus dl_cross_entropy_mean(DlConstMatrix logits, const size_t *targets,
                               size_t target_count, double *loss,
                               DlMatrix *dlogits, DlWriteMode mode);

#endif
```

Create `src/loss.c`. The first portion validates the complete request before touching any destination, including targets in later rows and non-finite values in later logits. Invalid shape or alias information cannot be repaired by changing numerical tolerances. We reject both positive and negative infinity; negative infinity is not an implicit mask in this loss.

```c title="src/loss.c"
#include "dl/loss.h"

#include <float.h>
#include <math.h>

DlStatus dl_cross_entropy_mean(DlConstMatrix logits, const size_t *targets,
                               size_t target_count, double *loss,
                               DlMatrix *dlogits, DlWriteMode mode)
{
    if (targets == NULL || loss == NULL ||
        (mode != DL_SET && mode != DL_ADD)) {
        return DL_ARGUMENT;
    }
    size_t logits_end = 0;
    DlStatus status = dl_matrix_end(logits, &logits_end);
    if (status != DL_OK) {
        return status;
    }
    if (target_count != logits.rows) {
        return DL_SHAPE;
    }
    if (dlogits != NULL) {
        const DlConstMatrix gr = dl_matrix_read(*dlogits);
        size_t grad_end = 0;
        status = dl_matrix_end(gr, &grad_end);
        if (status != DL_OK) {
            return status;
        }
        if (gr.rows != logits.rows || gr.cols != logits.cols) {
            return DL_SHAPE;
        }
        if (gr.base == logits.base && gr.offset < logits_end &&
            logits.offset < grad_end) {
            return DL_ALIAS;
        }
    }
    for (size_t i = 0; i < logits.rows; ++i) {
        if (targets[i] >= logits.cols) {
            return DL_RANGE;
        }
    }
    /* Preflight every row, not only the first row that will be written. */
    for (size_t i = 0; i < logits.rows; ++i) {
        for (size_t j = 0; j < logits.cols; ++j) {
            if (!isfinite(logits.base[logits.offset + i * logits.ld + j])) {
                return DL_NUMERIC;
            }
            if (dlogits != NULL && mode == DL_ADD &&
                !isfinite(dlogits->base[dlogits->offset + i * dlogits->ld + j])) {
                return DL_NUMERIC;
            }
        }
    }
```

Continue inside the same function. For each logical row, find one maximum, collect its non-maximum mass and its non-target mass, and add the **already divided** row loss to a double total. `maximum_index` identifies only the selected unit contribution in the denominator. `targets[i]` identifies the class whose derivative is negative. They need not be the same index.

```c title="src/loss.c"
    const double rows = (double)logits.rows;
    double total = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        const size_t row = logits.offset + i * logits.ld;
        size_t maximum_index = 0;
        double maximum = (double)logits.base[row];
        for (size_t j = 1; j < logits.cols; ++j) {
            const double value = (double)logits.base[row + j];
            if (value > maximum) {
                maximum = value;
                maximum_index = j;
            }
        }
        double other = 0.0;
        double not_target = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) {
            const double term = exp((double)logits.base[row + j] - maximum);
            if (j != maximum_index) {
                other += term;
            }
            if (j != targets[i]) {
                not_target += term;
            }
        }
        const double denominator = 1.0 + other;
        const double row_loss =
            (maximum - (double)logits.base[row + targets[i]]) + log1p(other);
        total += row_loss / rows;
        if (!isfinite(denominator) || !isfinite(total)) {
            return DL_NUMERIC;
        }
```

Finish the function with gradient writes and one scalar commit. The conditional operator `condition ? a : b` evaluates just the chosen expression. Here it selects `-not_target` only at the target coordinate. `continue` advances to the next row in a value-only call; it does not bypass the loss computation.

```c title="src/loss.c"
        if (dlogits == NULL) {
            continue;
        }
        for (size_t j = 0; j < logits.cols; ++j) {
            const double numerator = j == targets[i] ? -not_target
                : exp((double)logits.base[row + j] - maximum);
            double value = (numerator / denominator) / rows;
            const size_t dst = dlogits->offset + i * dlogits->ld + j;
            if (mode == DL_ADD) {
                value += (double)dlogits->base[dst];
            }
            if (!isfinite(value) || fabs(value) > (double)FLT_MAX) {
                return DL_NUMERIC;
            }
            dlogits->base[dst] = (float)value;
        }
    }
    *loss = total;
    return DL_OK;
}
```

`DL_SET` never reads an old gradient cell. `DL_ADD` adds the old value once, after calculating the new contribution. The scalar output is always the current mean loss, not an accumulating loss counter. A caller accumulating several objectives must combine their scalar values and gradient weights deliberately; `DL_ADD` alone does not define an average across minibatches.

The explicit dimensions have no inherited 4,096-element bound from Chapter 3's demonstration. On the intended binary32/binary64, 64-bit-`size_t` target, each shifted term is at most one, the denominator is at most $V$, and the largest finite logit gap is at most $2\,\mathrm{FLT\_MAX}$ in double. Even the 64-bit index range is far below double's overflow range for these sums. Actual live allocation capacity and Chapter 4's checked geometry constrain the views; this is not permission to forge huge array metadata. Very large sums still have rounding error, and row counts beyond exact double integer precision are not a practical memory configuration for this machine.

The routine uses constant-size scalar scratch. It still scans every vocabulary entry and performs exponentials, and the caller still stores logits and, during backward, their gradients. “No probability matrix” does not remove those arrays. Recomputing exponentials for non-target gradient coordinates avoids retaining an additional row or matrix buffer; it is a transparent reference tradeoff, not a speed claim.

Preflight rejection changes no outputs. The public numeric-error contract remains conservative: a later numeric failure can follow earlier gradient writes, while `*loss` is committed only at the end. On the target's supported finite inputs the bounds make the shifted core well behaved, but callers should still discard the entire result on a failed status rather than depend on how far a particular implementation got.

## Reverse the projection using two matrix products

Now return to a general projection $Y=XW^T$, with $X\in\mathbb{R}^{M\times K}$, $W\in\mathbb{R}^{N\times K}$, and incoming gradient $G\in\mathbb{R}^{M\times N}$. Here $K$ is the input feature width and $N$ the output width; in our tiny graph they are $D$ and $V$.

Each output element is $Y_{i,n}=\sum_{k=0}^{K-1}X_{i,k}W_{n,k}$. Holding a single input coordinate $X_{i,k}$ in mind, all outputs in row $i$ depend on it:

$$
\frac{\partial L}{\partial X_{i,k}}
=\sum_{n=0}^{N-1}G_{i,n}W_{n,k},
\qquad G_X=GW.
$$

A weight $W_{n,k}$ is shared by all prediction rows, so its derivative instead reduces over those rows:

$$
\frac{\partial L}{\partial W_{n,k}}
=\sum_{i=0}^{M-1}G_{i,n}X_{i,k},
\qquad G_W=G^TX.
$$

The gradient has the same stored orientation as its parameter, `[N,K]`, not `[K,N]`. No physical transpose is required. Chapter 4's flags already express the needed accesses. Append these two complete functions to `src/kernels.c`:

```c title="src/kernels.c"
DlStatus dl_linear_backward_input(DlConstMatrix dy, DlConstMatrix w,
                                  DlMatrix dx, DlWriteMode mode)
{
    return dl_matmul_ref(dy, DL_NOTRANS, w, DL_NOTRANS, dx, mode);
}

DlStatus dl_linear_backward_weight(DlConstMatrix dy, DlConstMatrix x,
                                   DlMatrix dw, DlWriteMode mode)
{
    return dl_matmul_ref(dy, DL_TRANS, x, DL_NOTRANS, dw, mode);
}
```

These small functions name a mathematical role; they do not hide new arithmetic. All reductions, checks, and transpose indexing remain the already implemented C scalar kernel. There is no bias in this projection, so no undeclared bias gradient is missing.

Each function has one output and independently honors `DL_SET` or `DL_ADD`. We deliberately do not pretend the pair is a transactional operation: if computing the second gradient fails, the first may already exist. Callers should stop and discard the current backward pass. Keep $X$ alive for the weight gradient and keep the **forward** $W$ unchanged for the input gradient; applying an optimizer update between these calls would differentiate a different computation.

If $G$ came from our mean loss, it already contains the factor $1/M$. Neither product divides again, and embedding scatter must not divide again either. Shared weights are differentiated by summing the paths in $G$, not by averaging them a second time.

## Walk through one complete graph by hand

Use $V=4$, $D=2$, and $M=3$. Define $a=\log2$, then set:

$$
E=\begin{bmatrix}1&0\\0&1\\-1&1/2\\1/4&-1/2\end{bmatrix},
\qquad
W=\begin{bmatrix}0&0\\a&0\\0&a\\a&a\end{bmatrix}.
$$

With inputs `[0,1,0]`, $H$ contains rows `[1,0]`, `[0,1]`, `[1,0]`. Thus logits are `[0,a,0,a]`, `[0,0,a,a]`, `[0,a,0,a]`. Their probability rows are:

$$
P=\begin{bmatrix}
1/6&1/3&1/6&1/3\\
1/6&1/6&1/3&1/3\\
1/6&1/3&1/6&1/3
\end{bmatrix}.
$$

Targets `[1,0,2]` therefore give individual losses $\log3$, $\log6$, and $\log6$, and mean $L\approx1.560710409$. Subtract the target indicator in each row and divide by three:

$$
G_Z=\begin{bmatrix}
1/18&-2/9&1/18&1/9\\
-5/18&1/18&1/9&1/9\\
1/18&1/9&-5/18&1/9
\end{bmatrix}.
$$

For example, $(G_H)_{0,0}=a(-2/9+1/9)=-a/9$, because only weight rows 1 and 3 have a nonzero first coordinate. Applying both products gives:

$$
G_H=\begin{bmatrix}-a/9&a/6\\a/6&2a/9\\2a/9&-a/6\end{bmatrix},
\qquad
G_W=\begin{bmatrix}1/9&-5/18\\-1/9&1/18\\-2/9&1/9\\2/9&1/9\end{bmatrix}.
$$

The first column of $G_W$ adds gradient rows 0 and 2 because their first hidden coordinate is one; the second column uses row 1. Embedding row 0 also receives rows 0 and 2 of $G_H$, giving $[a/9,0]$. Row 1 receives $[a/6,2a/9]$. Rows 2 and 3 receive zero from lookup, even though output-projection rows 2 and 3 have nonzero gradients. These are **two distinct parameter arrays**, not tied weights yet.

A stored float approximation of $a$ makes these rational probabilities approximate rather than exact. The code can produce a tiny nonzero second coordinate of $G_E[0]$ from rounded additions; it must remain within a justified tolerance. That small residual is unlike losing an entire occurrence's contribution.

### Share the fixture, not a hidden implementation

Create `tests/tiny.h`. `Tiny` owns inline fixed-size arrays; it has no owning heap pointers. Copying this struct copies its values and is therefore unlike copying a live `DlTensor`. No borrowed view is stored inside the struct, so every evaluation constructs views of its current arrays.

The test-only `tiny_matrix` helper takes exactly the small, known storage capacities shown here. It is not a replacement for checked allocation of runtime dimensions. Each `base` is the beginning of the complete array object, preserving the same-root overlap rule. `tiny_init` uses a compound literal to initialize the whole struct, including zero initialization of members not explicitly listed.

```c title="tests/tiny.h"
#ifndef DL_TEST_TINY_H
#define DL_TEST_TINY_H

#include "dl/loss.h"

#include <math.h>

/* Owns inline arrays, not heap pointers. Views never outlive this object. */
typedef struct {
    float e[8], w[8];
    float h[6], z[12], dz[12], dh[6], de[8], dw[8];
    size_t ids[3], targets[3];
} Tiny;

/* Test-only helper: caller provides exactly rows*cols live float elements.
 * Tiny fixtures use small fixed dimensions; this is not an allocator/API. */
static DlMatrix tiny_matrix(float *base, size_t rows, size_t cols)
{
    return (DlMatrix){.base = base, .count = rows * cols, .offset = 0,
        .rows = rows, .cols = cols, .ld = cols};
}

static void tiny_init(Tiny *t)
{
    const float a = (float)log(2.0);
    *t = (Tiny){.e = {1.0f, 0.0f, 0.0f, 1.0f,
                     -1.0f, 0.5f, 0.25f, -0.5f},
        .w = {0.0f, 0.0f, a, 0.0f, 0.0f, a, a, a},
        .ids = {0, 1, 0}, .targets = {1, 0, 2}};
}

```

Append the full connected evaluation. In a value-only call, the conditional expression `backward ? &dz : NULL` disables gradient output while keeping the loss path identical. In a backward call, the order is loss derivative, input projection gradient, weight projection gradient, then embedding scatter. `status == DL_OK` guards every dependency.

```c title="tests/tiny.h"
static DlStatus tiny_eval(Tiny *t, double *loss, int backward)
{
    const DlMatrix e = tiny_matrix(t->e, 4, 2);
    const DlMatrix w = tiny_matrix(t->w, 4, 2);
    const DlMatrix h = tiny_matrix(t->h, 3, 2);
    const DlMatrix z = tiny_matrix(t->z, 3, 4);
    DlMatrix dz = tiny_matrix(t->dz, 3, 4);
    DlStatus status = dl_embedding_forward(dl_matrix_read(e), t->ids, 3, h);
    if (status == DL_OK) {
        status = dl_matmul_ref(dl_matrix_read(h), DL_NOTRANS,
            dl_matrix_read(w), DL_TRANS, z, DL_SET);
    }
    if (status == DL_OK) {
        status = dl_cross_entropy_mean(dl_matrix_read(z), t->targets, 3,
            loss, backward ? &dz : NULL, DL_SET);
    }
    if (status != DL_OK || !backward) {
        return status;
    }
    const DlMatrix dh = tiny_matrix(t->dh, 3, 2);
    status = dl_linear_backward_input(dl_matrix_read(dz), dl_matrix_read(w),
        dh, DL_SET);
    if (status == DL_OK) {
        status = dl_linear_backward_weight(dl_matrix_read(dz),
            dl_matrix_read(h), tiny_matrix(t->dw, 4, 2), DL_SET);
    }
    if (status == DL_OK) {
        status = dl_embedding_backward(dl_matrix_read(dh), t->ids, 3,
            tiny_matrix(t->de, 4, 2), DL_SET);
    }
    return status;
}

#endif
```

All arrays remain alive until the caller's `Tiny` object goes out of scope. `H` is needed again by the weight gradient, `W` by the input gradient, and IDs by scatter; none is overwritten along that path. There is no cleanup loop because these are inline automatic arrays and every kernel is allocation-free. The eventual variable-size model will use the tensor owners from Chapter 4, not grow this fixed-size fixture into an allocator.

A composed failure is not a rollback: forward intermediates may already be changed, and a successful loss may have been committed before a later gradient fails. Treat the entire attempted graph evaluation as invalid on a non-OK status. During finite differences we must additionally restore the perturbed parameter even when such an evaluation fails.

## Continue with executable checks

The operators and shared fixture are now defined. Continue to [Part 2 — Checking gradients and numerical failures](https://25d-228.github.io/gpt-c/chapter-05-part-2.html) for every test driver, the complete Makefile, expected output, and diagnostic variations.

## References

[embedding]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
[cross-entropy]: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
[online]: https://arxiv.org/html/1805.02867v1
[accurate]: https://arxiv.org/html/1909.03469v1

- [PyTorch, `Embedding`][embedding]: table shape, index lookup, and the distinction between ordinary gradients and optional frequency scaling. Used as an operator-definition reference, not an execution dependency.
- [PyTorch, `CrossEntropyLoss`][cross-entropy]: unnormalized logits, integer-target range, and the unweighted mean reduction. The C interface deliberately does not implement the optional masking, weighting, or smoothing modes.
- [Milakov and Gimelshein, *Online normalizer calculation for softmax*][online], Section 2, Algorithm 2: maximum-shift normalization. The later online algorithm is not needed for this reference loss.
- [Blanchard, Higham, and Higham, *Accurate Computation of the Log-Sum-Exp and Softmax Functions*][accurate], 2019 version, Section 4, Algorithm 5: the shifted sum with one maximum term separated and a `log1p` evaluation. The target-loss ordering and complement-gradient traces above specialize these numerical considerations to this implementation.

