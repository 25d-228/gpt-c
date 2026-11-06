#ifndef DL_DATA_H
#define DL_DATA_H

#include "dl/common.h"
#include "dl/rng.h"
#include <stdio.h>

typedef struct {
    unsigned char *bytes;             /* Owns capacity bytes; count are valid. */
    size_t count, capacity;
} DlByteData;

typedef struct {
    size_t *storage;                  /* Sole owner: 2*count + batch elements. */
    size_t *ids, *targets, *starts;    /* Views into storage, not separate owners. */
    size_t batch, time, count;        /* count = batch*time. */
} DlByteBatch;

/* Empty {0} destinations only. Failure leaves owners unchanged. No live owner
 * copies or metadata edits. No concurrent file mutation or calls. The byte
 * loader accepts arbitrary bytes, including 0, and requires at least two.
 * limit bounds the sole payload allocation, not stdio/process memory.
 * read borrows a blocking stream at its CURRENT position and consumes it even
 * on failure; it never closes it. load opens "rb" and checks close before commit.
 * DL_STATE: I/O failure/nonempty owner; DL_RANGE: short/over-limit content. */
DlStatus dl_bytes_read(DlByteData *out, FILE *stream, size_t limit);
DlStatus dl_bytes_load(DlByteData *out, const char *path, size_t limit);
void dl_bytes_destroy(DlByteData *data);
DlStatus dl_byte_batch_create(DlByteBatch *out, size_t batch, size_t time);
void dl_byte_batch_destroy(DlByteBatch *batch);

/* Data, batch, RNG, cursor, and outputs occupy independent storage. Require
 * constructed unmodified owners and data.count > batch.time. IDs/targets are
 * dense [batch,time], values 0..255; starts contains source byte offsets.
 * sample draws starts uniformly WITH replacement, commits explicit RNG state,
 * and permits overlap inside this ONE file. Preflight failures change nothing.
 * next produces full consecutive batches, advances cursor by batch.count, and
 * never wraps. End: DL_OK, *has_batch=0, unchanged cursor/batch. Drop the final
 * incomplete batch; report its unscored target count. No padding/special IDs. */
DlStatus dl_byte_batch_sample(const DlByteData *data, DlByteBatch *batch,
                              DlRng *rng);
DlStatus dl_byte_batch_next(const DlByteData *data, DlByteBatch *batch,
                            size_t *cursor, int *has_batch);
/* Quoted ASCII presentation only; never a transformation of training bytes.
 * Escape controls, non-ASCII, backslash, quote. I/O failure can leave a partial
 * display. Does not flush or close the borrowed stream. */
DlStatus dl_bytes_fprint(FILE *stream, const unsigned char *bytes, size_t count);

#endif
