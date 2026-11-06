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
