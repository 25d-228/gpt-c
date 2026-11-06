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

void dl_tensor_destroy(DlTensor *tensor)
{
    if (tensor == NULL) {
        return;
    }
    free(tensor->data);
    *tensor = (DlTensor){0};
}

DlConstMatrix dl_matrix_read(DlMatrix matrix)
{
    return (DlConstMatrix){.base = matrix.base, .count = matrix.count,
        .offset = matrix.offset, .rows = matrix.rows, .cols = matrix.cols,
        .ld = matrix.ld};
}

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
