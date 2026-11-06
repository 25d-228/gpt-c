#include "dl/data.h"

#include <limits.h>
#include <stdlib.h>

_Static_assert(CHAR_BIT == 8, "byte IDs require 8-bit bytes");
_Static_assert(SIZE_MAX <= UINT64_MAX, "sample bounds must fit uint64_t");

static int data_empty(const DlByteData *d)
{
    return d->bytes == NULL && d->count == 0 && d->capacity == 0;
}

void dl_bytes_destroy(DlByteData *d)
{
    if (d == NULL) return;
    free(d->bytes);
    *d = (DlByteData){0};
}

DlStatus dl_bytes_read(DlByteData *out, FILE *stream, size_t limit)
{
    if (out == NULL || stream == NULL || limit < 2) return DL_ARGUMENT;
    if (!data_empty(out)) return DL_STATE;
    DlByteData candidate = {0};
    candidate.bytes = malloc(limit);
    if (candidate.bytes == NULL) return DL_ALLOC;
    candidate.capacity = limit;
    candidate.count = fread(candidate.bytes, 1, limit, stream);
    DlStatus status = DL_OK;
    if (ferror(stream)) status = DL_STATE;
    else if (candidate.count == limit) {
        /* Distinguish exact capacity from a silently truncated longer file. */
        int extra = fgetc(stream);
        if (ferror(stream)) status = DL_STATE;
        else if (extra != EOF) status = DL_RANGE;
        else if (!feof(stream)) status = DL_STATE;
    } else if (!feof(stream)) status = DL_STATE;
    if (status == DL_OK && candidate.count < 2) status = DL_RANGE;
    if (status != DL_OK) {
        dl_bytes_destroy(&candidate);
        return status;
    }
    *out = candidate;
    return DL_OK;
}

DlStatus dl_bytes_load(DlByteData *out, const char *path, size_t limit)
{
    if (out == NULL || path == NULL || limit < 2) return DL_ARGUMENT;
    if (!data_empty(out)) return DL_STATE;
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) return DL_STATE;
    DlByteData candidate = {0};
    DlStatus status = dl_bytes_read(&candidate, stream, limit);
    if (fclose(stream) != 0 && status == DL_OK) status = DL_STATE;
    if (status != DL_OK) {
        dl_bytes_destroy(&candidate);
        return status;
    }
    *out = candidate;
    return DL_OK;
}

static int batch_empty(const DlByteBatch *b)
{
    return b->storage == NULL && b->ids == NULL && b->targets == NULL &&
           b->starts == NULL && b->batch == 0 && b->time == 0 && b->count == 0;
}

void dl_byte_batch_destroy(DlByteBatch *b)
{
    if (b == NULL) return;
    free(b->storage);
    *b = (DlByteBatch){0};
}

DlStatus dl_byte_batch_create(DlByteBatch *out, size_t batch, size_t time)
{
    if (out == NULL || batch == 0 || time == 0) return DL_ARGUMENT;
    if (!batch_empty(out)) return DL_STATE;
    size_t count, twice, elements, bytes;
    DlStatus status = dl_size_mul(batch, time, &count);
    if (status != DL_OK) return status;
    status = dl_size_mul(count, 2, &twice);
    if (status != DL_OK) return status;
    status = dl_size_add(twice, batch, &elements);
    if (status != DL_OK) return status;
    status = dl_size_mul(elements, sizeof(size_t), &bytes);
    if (status != DL_OK) return status;
    size_t *storage = malloc(bytes);
    if (storage == NULL) return DL_ALLOC;
    *out = (DlByteBatch){storage, storage, storage + count, storage + twice,
                         batch, time, count};
    return DL_OK;
}

static DlStatus batch_check(const DlByteData *d, const DlByteBatch *b)
{
    if (d == NULL || b == NULL) return DL_ARGUMENT;
    if (d->bytes == NULL || b->storage == NULL) return DL_STATE;
    /* Other fields are the established constructor invariants, not user input. */
    if (d->count <= b->time) return DL_RANGE;
    return DL_OK;
}

static void copy_window(const DlByteData *d, DlByteBatch *b,
                         size_t row, size_t start)
{
    b->starts[row] = start;
    size_t base = row * b->time;
    for (size_t t = 0; t < b->time; ++t) {
        b->ids[base + t] = d->bytes[start + t];
        b->targets[base + t] = d->bytes[start + t + 1];
    }
}

DlStatus dl_byte_batch_sample(const DlByteData *d, DlByteBatch *b, DlRng *rng)
{
    if (rng == NULL) return DL_ARGUMENT;
    DlStatus status = batch_check(d, b);
    if (status != DL_OK) return status;
    uint64_t bound = (uint64_t)(d->count - b->time);
    DlRng candidate = *rng;
    for (size_t row = 0; row < b->batch; ++row) {
        uint64_t start;
        /* These validated arguments make the bounded draw infallible. */
        status = dl_rng_bounded(&candidate, bound, &start);
        if (status != DL_OK) return status;
        copy_window(d, b, row, (size_t)start);
    }
    *rng = candidate;
    return DL_OK;
}

DlStatus dl_byte_batch_next(const DlByteData *d, DlByteBatch *b,
                            size_t *cursor, int *has_batch)
{
    if (cursor == NULL || has_batch == NULL) return DL_ARGUMENT;
    DlStatus status = batch_check(d, b);
    if (status != DL_OK) return status;
    if (*cursor > d->count - 1) return DL_RANGE;
    if (b->count > d->count - 1 - *cursor) {
        *has_batch = 0;
        return DL_OK;
    }
    for (size_t row = 0; row < b->batch; ++row)
        copy_window(d, b, row, *cursor + row * b->time);
    *cursor += b->count;
    *has_batch = 1;
    return DL_OK;
}

DlStatus dl_bytes_fprint(FILE *stream, const unsigned char *bytes, size_t count)
{
    if (stream == NULL || (bytes == NULL && count != 0)) return DL_ARGUMENT;
    if (fputc('"', stream) == EOF) return DL_STATE;
    for (size_t i = 0; i < count; ++i) {
        unsigned int c = bytes[i];
        const char *escape = NULL;
        if (c == '\n') escape = "\\n";
        else if (c == '\r') escape = "\\r";
        else if (c == '\t') escape = "\\t";
        else if (c == '\\') escape = "\\\\";
        else if (c == '"') escape = "\\\"";
        if (escape != NULL) {
            if (fputs(escape, stream) == EOF) return DL_STATE;
        } else if (c >= 32 && c <= 126) {
            if (fputc((int)c, stream) == EOF) return DL_STATE;
        } else if (fprintf(stream, "\\x%02X", c) < 0) return DL_STATE;
    }
    if (fputc('"', stream) == EOF || ferror(stream)) return DL_STATE;
    return DL_OK;
}
