#include "buffer.h"

#include <stdint.h>
#include <stdlib.h>

const char *storage_status_name(StorageStatus status)
{
    switch (status) {
    case STORAGE_OK:       return "ok";
    case STORAGE_ARGUMENT: return "invalid argument";
    case STORAGE_STATE:    return "invalid owner state";
    case STORAGE_OVERFLOW: return "size overflow";
    case STORAGE_ALLOC:    return "allocation failed";
    case STORAGE_RANGE:    return "view out of range";
    }
    return "unknown status";
}

StorageStatus size_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return STORAGE_ARGUMENT;
    }
    if (a != 0 && b > SIZE_MAX / a) {
        return STORAGE_OVERFLOW;
    }
    *out = a * b;
    return STORAGE_OK;
}

void *system_allocate(size_t bytes, void *context)
{
    (void)context;
    return malloc(bytes);
}

StorageStatus buffer_create(FloatBuffer *out, size_t count,
                            AllocateFn allocate, void *context)
{
    if (out == NULL || allocate == NULL) {
        return STORAGE_ARGUMENT;
    }
    if (out->data != NULL || out->count != 0) {
        return STORAGE_STATE;
    }

    size_t bytes = 0;
    StorageStatus status = size_mul(count, sizeof(float), &bytes);
    if (status != STORAGE_OK) {
        return status;
    }
    if (count == 0) {
        return STORAGE_OK;
    }

    float *data = allocate(bytes, context);
    if (data == NULL) {
        return STORAGE_ALLOC;
    }
    *out = (FloatBuffer){.data = data, .count = count};
    return STORAGE_OK;
}

void buffer_destroy(FloatBuffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    *buffer = (FloatBuffer){.data = NULL, .count = 0};
}

StorageStatus buffer_view(const FloatBuffer *owner, size_t offset,
                          size_t count, size_t stride, FloatView *out)
{
    if (owner == NULL || out == NULL || stride == 0) {
        return STORAGE_ARGUMENT;
    }
    if ((owner->data == NULL) != (owner->count == 0)) {
        return STORAGE_STATE;
    }
    if (offset > owner->count) {
        return STORAGE_RANGE;
    }
    if (count == 0) {
        *out = (FloatView){.data = NULL, .count = 0, .stride = 1};
        return STORAGE_OK;
    }
    if (offset == owner->count) {
        return STORAGE_RANGE;
    }
    const size_t remaining = owner->count - 1 - offset;
    if (count - 1 > remaining / stride) {
        return STORAGE_RANGE;
    }

    *out = (FloatView){.data = owner->data + offset,
                      .count = count, .stride = stride};
    return STORAGE_OK;
}

float view_sum(FloatView view)
{
    float sum = 0.0f;
    for (size_t i = 0; i < view.count; ++i) {
        sum += view.data[i * view.stride];
    }
    return sum;
}
