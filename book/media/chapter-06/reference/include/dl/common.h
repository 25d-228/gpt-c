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
