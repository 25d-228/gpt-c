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
