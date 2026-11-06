#ifndef DL_EXAMPLE_BUFFER_H
#define DL_EXAMPLE_BUFFER_H

#include <stddef.h>

typedef enum {
    STORAGE_OK = 0,
    STORAGE_ARGUMENT,
    STORAGE_STATE,
    STORAGE_OVERFLOW,
    STORAGE_ALLOC,
    STORAGE_RANGE
} StorageStatus;

typedef struct {
    float *data;
    size_t count;
} FloatBuffer;

typedef struct {
    const float *data;
    size_t count;
    size_t stride;
} FloatView;

/* Return fresh malloc-compatible storage, or NULL. No output mutation.
 * The context is borrowed only during the call; no pointer is retained.
 * Successful results must be releasable by the standard free function. */
typedef void *(*AllocateFn)(size_t bytes, void *context);

const char *storage_status_name(StorageStatus status);
StorageStatus size_mul(size_t a, size_t b, size_t *out);
void *system_allocate(size_t bytes, void *context);

/* out must point to an initialized, empty owner: {NULL, 0}.
 * On success, positive count owns count UNINITIALIZED floats.
 * Zero count succeeds without allocating. allocate must still be non-null.
 * On reported failure, out is unchanged. Never copy a live owner. */
StorageStatus buffer_create(FloatBuffer *out, size_t count,
                            AllocateFn allocate, void *context);

/* Accept NULL, an empty owner, or the unique live owner. Reset it to empty.
 * All views of a freed allocation become invalid. Does not clear copies. */
void buffer_destroy(FloatBuffer *buffer);

/* owner must be an initialized empty/live owner, with truthful extent.
 * offset/count/stride are in FLOAT ELEMENTS; stride must be positive.
 * Empty requests allow offset <= owner->count and produce {NULL, 0, 1}.
 * On reported failure, out is unchanged. No allocation or lifetime extension. */
StorageStatus buffer_view(const FloatBuffer *owner, size_t offset,
                          size_t count, size_t stride, FloatView *out);

/* view must describe live initialized elements, normally via buffer_view.
 * Reads only, allocates nothing, retains nothing. Empty sum is zero.
 * Forged metadata or a destroyed owner violates the caller contract. */
float view_sum(FloatView view);

#endif
