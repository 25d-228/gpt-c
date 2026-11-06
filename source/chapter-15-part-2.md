# Chapter 15 · Part 2 — Versioned checkpoints and coherent restoration

A weight file answers “what function can I evaluate?” A continuation checkpoint answers “what operation happens next?” This part stores the unique parameter values, both optimizer moments, successful-update count, schedule, batching policy, current data generator, processed-target count, and exact data bytes. It accepts only the completed boundaries defined in [Part 1](https://25d-228.github.io/gpt-c/chapter-15.html).

This is the second part of the same **6.00-hour** chapter. Continue from Part 1's code. Create `include/dl/checkpoint.h` and `src/checkpoint.c`, and add the restore function to `src/optim.c` at the stated location. No inference, generation, or persistent key/value cache is introduced here.

## Portable fields are not portable C structs

A `DlModel` contains pointers, `size_t` values, and compiler-chosen padding. Writing its object representation would save addresses rather than the arrays they point at. Even a pointer-free struct can contain padding and host-dependent field widths. The disk format therefore specifies each field's size and byte order.

An unsigned 64-bit integer is written least-significant byte first. For example,

```text
value: 0x0102030405060708
bytes: 08 07 06 05 04 03 02 01
```

To write byte $i$, shift the unsigned value right by $8i$ and keep its low eight bits. To read, shift each byte left by $8i$ and combine with bitwise OR. There is no signed shift and no dereference through a misaligned integer pointer.

Weights and moments use IEEE binary32, while schedule and optimizer scalars use binary64. On the supported targets, copying `1.0f` into `uint32_t` yields the bits `0x3f800000`, whose little-endian file bytes are `00 00 80 3f`. `memcpy` transfers the representation without violating the aliasing rules by pretending a `float *` is a `uint32_t *`. Compile-time format checks and a runtime `1.0` check make the representation requirement explicit. These are compatibility checks, not a generic conversion library for unusual floating-point formats.

The on-disk numbers are portable between supported byte orders. **Bitwise training continuation is a narrower claim:** keep the same executable's numerical behavior, backend, data, and settings. A different compiler, library, architecture, or reduction policy can change future arithmetic even when it reads the same stored values correctly.

## Lay out the bytes before implementing the writer

Version 1 has a 256-byte header. Offsets are zero-based byte offsets from the start of the file. A `u64` below is an unsigned 64-bit little-endian integer; `f32` and `f64` are the explicitly encoded floating representations.

| Offset | Field or group | Representation |
| --- | --- | --- |
| 0 | Magic `DLCKPT01` | Eight literal bytes |
| 8 | Format version, currently 1 | One `u64` |
| 16–56 | Vocabulary, layers, width, query heads, key/value heads, feed-forward width | Six `u64` |
| 64 | Initialization seed | One `u64` |
| 72, 80 | Parameter-record count, unique scalar count $P$ | Two `u64` |
| 88, 96, 104 | Batch $B$, time $T$, microbatches $A$ | Three `u64` |
| 112, 120 | Backend identifier, fixed/random flag | Two `u64` |
| 128, 136 | Total updates and warmup updates | Two `u64` |
| 144, 152 | Training and validation byte counts | Two `u64` |
| 160, 164 | Model epsilon and initialization scale | Two `f32` |
| 168 | RoPE theta; Adam beta1, beta2, epsilon, decay, clipping threshold; schedule peak and floor | Eight `f64`, through byte 231 |
| 232, 240, 248 | Completed updates, processed targets, current data generator state | Three `u64` |

For each parameter in directory order, write its zero-padded 64-byte name, row count, column count, and decay flag, followed immediately by its dense weights. The directory prefix is $64+3\cdot8=88$ bytes per parameter. Then write the complete first-moment vector, the complete second-moment vector, the training bytes, the validation bytes, and the eight literal bytes `ENDCKP01`. Require end of file immediately afterward.

Let $K_p$ be the parameter-record count, $P$ the unique scalar count, and $D_t,D_v$ the two dataset lengths. The exact file size is

$$
256+88K_p+12P+D_t+D_v+8\quad\text{bytes}.
$$

The term $12P$ is four bytes each for weights, first moments, and second moments. The tied embedding/output table appears once because the model directory already represents it once. Gradient sums and activations are absent because the boundary has no pending update.

This version stores the bounded data bytes too. On loading, it compares them against the supplied datasets rather than trusting pathnames or lengths. The extra disk space is a deliberate simple way to bind a run to exact bytes. The reader compares in 4096-byte chunks, so it does not allocate another full dataset. It is not a content-addressed storage system and does not detect every possible finite bit change in a weight. Use ordinary files you control; this is not an authenticated format for hostile input.

The existing model and optimizer are the compatibility specification. Loading a checkpoint into a different width, schedule, backend, dataset, or optimizer configuration returns an error; it does not silently reinterpret the file or turn a continuation into a new experiment.

```c title="include/dl/checkpoint.h"
#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"

/* Version 1: explicit little-endian integers and IEEE binary32/binary64.
 * Existing constructed model/optimizer/state must describe the SAME run:
 * model config, optimizer config, batching, backend, schedule and fixed mode.
 * Dataset bytes are included and compared exactly (not merely file lengths).
 * No untrusted file field controls an allocation. max_file_bytes also bounds
 * the temporary 3*P*sizeof(float) load payload; metadata/stdio are excluded.
 * Ordinary local files in a caller-controlled directory; no concurrent edits,
 * writers, adversarial paths, live-owner moves, or aliasing of output state.
 * DL_SHAPE: incompatibility; DL_RANGE: truncation/budget/trailing bytes;
 * DL_NUMERIC: bad weights/moments; DL_STATE: lifecycle or I/O failure.
 * SAVE: write sibling temporary file, flush/sync/close, then rename. Until the
 * successful rename, an existing destination remains untouched. No automatic
 * retries or power-loss durability guarantee (no parent-directory/full sync).
 * LOAD: read/check/close candidate first, then coordinated in-place restore.
 * A failed load preserves all live state. Success increments the LIVE model
 * revision (never imports a saved revision), invalidating every old tape. */
DlStatus dl_checkpoint_save(const char *path, const DlAdamW *optim,
                            const DlTrainState *state,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes);
DlStatus dl_checkpoint_load(const char *path, DlAdamW *optim,
                            DlTrainState *state,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes);

#endif
```

## Transfer bytes and encode unsigned values

Create `src/checkpoint.c`. `_POSIX_C_SOURCE` appears before system headers so the requested POSIX declarations are exposed. These calls are available through the native platform's C library; no separate tensor or checkpoint framework is used.

`Io` is a borrowed stream plus a remaining-byte budget and a direction flag. Both transfer directions use `size=1`, so `fread` and `fwrite` report byte counts. A short read is either truncation or an I/O error; `ferror` distinguishes them. A short write is not accepted as a complete field. The stream may already have advanced on failure, but it is either an uncommitted temporary output or a candidate input, never a partially installed model. These return-count rules are documented in Apple's [binary stream I/O reference](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/fwrite.3.html).

```c title="src/checkpoint.c"
#define _POSIX_C_SOURCE 200809L
#include "dl/checkpoint.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && sizeof(double) == 8,
               "checkpoint requires 8-bit bytes and 32/64-bit floating storage");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53 &&
               FLT_MAX_EXP == 128 && DBL_MAX_EXP == 1024,
               "checkpoint requires binary32/binary64 arithmetic");
_Static_assert(SIZE_MAX <= UINT64_MAX, "size_t must fit in uint64_t");

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

typedef struct {
    FILE *file;
    uint64_t remaining;
    int reading;
} Io;

static DlStatus transfer(Io *io, void *bytes, size_t count)
{
    if ((uint64_t)count > io->remaining) return DL_RANGE;
    size_t got = io->reading ? fread(bytes, 1, count, io->file)
                             : fwrite(bytes, 1, count, io->file);
    if (got != count) return ferror(io->file) ? DL_STATE : DL_RANGE;
    io->remaining -= (uint64_t)count;
    return DL_OK;
}

static DlStatus integer(Io *io, uint64_t *value, size_t width)
{
    unsigned char bytes[8] = {0};
    if (!io->reading)
        for (size_t i = 0; i < width; ++i)
            bytes[i] = (unsigned char)(*value >> (8 * i));
    RETURN_IF_ERROR(transfer(io, bytes, width));
    if (io->reading) {
        uint64_t decoded = 0;
        for (size_t i = 0; i < width; ++i)
            decoded |= (uint64_t)bytes[i] << (8 * i);
        *value = decoded;
    }
    return DL_OK;
}

static DlStatus expect_integer(Io *io, uint64_t expected, size_t width)
{
    uint64_t actual = expected;
    RETURN_IF_ERROR(integer(io, &actual, width));
    return actual == expected ? DL_OK : DL_SHAPE;
}
```

`integer` is internal: its only widths are four and eight, both within its fixed eight-byte array. It initializes the array even in read mode and never changes the caller's integer on a failed transfer. The caller's field size is known before inspecting file contents.

`expect_integer` makes strict compatibility explicit. A reader transfers the file value and compares it with a field already known from the constructed run. There is no conversion from an untrusted “parameter count” into an allocation request.

The byte matcher also handles the two complete corpora. The scratch buffer is bounded; each chunk is compared before moving to the next. The float and double matchers compare their stored representations, not an approximate numerical tolerance: changing an optimizer epsilon by one bit defines different continuation settings.

```c title="src/checkpoint.c"
static DlStatus expect_bytes(Io *io, const void *expected, size_t count)
{
    const unsigned char *source = expected;
    unsigned char buffer[4096];
    while (count != 0) {
        size_t chunk = count < sizeof(buffer) ? count : sizeof(buffer);
        if (!io->reading) memcpy(buffer, source, chunk);
        RETURN_IF_ERROR(transfer(io, buffer, chunk));
        if (io->reading && memcmp(buffer, source, chunk) != 0) return DL_SHAPE;
        source += chunk;
        count -= chunk;
    }
    return DL_OK;
}

static DlStatus expect_float(Io *io, float expected)
{
    uint32_t bits = 0;
    memcpy(&bits, &expected, sizeof(bits));
    return expect_integer(io, bits, 4);
}

static DlStatus expect_double(Io *io, double expected)
{
    uint64_t bits = 0;
    memcpy(&bits, &expected, sizeof(bits));
    return expect_integer(io, bits, 8);
}
```

## Decode into candidate storage, not live weights

The large arrays use a different rule: load them into a temporary candidate rather than compare them with current values. Every value must be finite. Second moments must also be nonnegative, because the optimizer will take a square root. Negative zero is allowed. Underflowed finite values are not invented errors; they are stored float values under the numerical policy already established.

The source and destination pointers are role-specific. Writing reads only `source`; reading writes only `destination`. Passing `NULL` for the unused role does not permit accessing it. The branch in `float_values` is what makes that contract work.

```c title="src/checkpoint.c"
static DlStatus float_values(Io *io, const float *source, float *destination,
                              size_t count, int nonnegative)
{
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        float value = 0.0f;
        if (!io->reading) memcpy(&bits, &source[i], sizeof(bits));
        uint64_t word = bits;
        RETURN_IF_ERROR(integer(io, &word, 4));
        bits = (uint32_t)word;
        memcpy(&value, &bits, sizeof(value));
        if (!isfinite(value) || (nonnegative && value < 0.0f)) return DL_NUMERIC;
        if (io->reading) destination[i] = value;
    }
    return DL_OK;
}

static DlStatus platform_check(void)
{
    float f = 1.0f;
    double d = 1.0;
    uint32_t fb = 0;
    uint64_t db = 0;
    memcpy(&fb, &f, sizeof(fb));
    memcpy(&db, &d, sizeof(db));
    return fb == UINT32_C(0x3f800000) && db == UINT64_C(0x3ff0000000000000)
        ? DL_OK : DL_STATE;
}
```

The platform check does not serialize a pointer or a native struct. It checks the correspondence between numeric and integer bit representations that the encoder relies on. Endianness is still handled explicitly by `integer`.

## One ordered record traversal for both directions

The same traversal defines writer and reader order. It first checks nineteen immutable integers, the two model floats, and eight doubles. Only three header values are accepted as restorable state: completed updates, processed targets, and current generator state. Even those must satisfy the range and target-count invariant before any parameter array is accepted.

The expected parameter name and shape come from the already-built model. A different name, ordering, or decay membership is a compatibility failure, even if the total scalar count happens to match. This makes a layout mistake distinguishable from a successfully loaded but nonsensical model.

```c title="src/checkpoint.c"
static DlStatus contents(Io *io, const DlAdamW *o, DlTrainState *s,
                          const DlByteData *training, const DlByteData *valid,
                          uint64_t *updates, float *candidate)
{
    RETURN_IF_ERROR(expect_bytes(io, "DLCKPT01", 8));
    DlModelConfig c = o->model->config;
    /* Nineteen immutable u64 fields; the full header is exactly 256 bytes. */
    const uint64_t fields[] = {
        1, c.vocab, c.layers, c.width, c.query_heads, c.kv_heads, c.ffn_width,
        c.seed, o->model->param_count, o->count,
        s->batch, s->time, s->microbatches, (uint64_t)s->backend,
        (uint64_t)s->fixed, s->schedule.total, s->schedule.warmup,
        training->count, valid->count
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i)
        RETURN_IF_ERROR(expect_integer(io, fields[i], 8));
    RETURN_IF_ERROR(expect_float(io, c.epsilon));
    RETURN_IF_ERROR(expect_float(io, c.init_std));
    const double reals[] = {
        c.rope_theta, o->config.beta1, o->config.beta2, o->config.epsilon,
        o->config.weight_decay, o->config.clip_norm, s->schedule.peak, s->schedule.floor
    };
    for (size_t i = 0; i < sizeof(reals) / sizeof(reals[0]); ++i)
        RETURN_IF_ERROR(expect_double(io, reals[i]));
    RETURN_IF_ERROR(integer(io, updates, 8));
    RETURN_IF_ERROR(integer(io, &s->tokens_seen, 8));
    RETURN_IF_ERROR(integer(io, &s->rng.state, 8));
    uint64_t group = (uint64_t)s->batch * (uint64_t)s->time *
                     (uint64_t)s->microbatches; /* Validated before opening. */
    if (*updates > s->schedule.total || *updates > UINT64_MAX / group ||
        s->tokens_seen != *updates * group) return DL_RANGE;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlParameter *param = &o->model->params[p];
        RETURN_IF_ERROR(expect_bytes(io, param->name, sizeof(param->name)));
        RETURN_IF_ERROR(expect_integer(io, param->value.shape[0], 8));
        RETURN_IF_ERROR(expect_integer(io, param->value.shape[1], 8));
        RETURN_IF_ERROR(expect_integer(io, (uint64_t)param->decay, 8));
        RETURN_IF_ERROR(float_values(io, param->value.data,
                                     candidate == NULL ? NULL : candidate + at,
                                     param->value.count, 0));
        at += param->value.count;
    }
    RETURN_IF_ERROR(float_values(io, o->moments,
                                 candidate == NULL ? NULL : candidate + o->count,
                                 o->count, 0));
    RETURN_IF_ERROR(float_values(io, o->moments + o->count,
                                 candidate == NULL ? NULL : candidate + 2 * o->count,
                                 o->count, 1));
    RETURN_IF_ERROR(expect_bytes(io, training->bytes, training->count));
    RETURN_IF_ERROR(expect_bytes(io, valid->bytes, valid->count));
    RETURN_IF_ERROR(expect_bytes(io, "ENDCKP01", 8));
    if (io->reading) {
        if (fgetc(io->file) != EOF) return DL_RANGE;
        if (ferror(io->file)) return DL_STATE;
    }
    return DL_OK;
}
```

The `at` index walks unique scalars in exactly the optimizer's directory order. Candidate layout is `[weights P][first moments P][second moments P]`. Multiplications for its allocation are checked by the top-level loader; the model's existing validated directory bounds each per-parameter transfer.

The final `fgetc` requires exact EOF. It detects extra bytes rather than treating an arbitrary prefix as a valid checkpoint. A truncated trailer, changed dataset byte, or invalid last second moment still fails before installation. A header with a huge forged count is compared with the expected count and rejected; it never determines the size of `candidate`.

## Preserve the last usable file while saving

Opening the destination with `"wb"` would truncate the last good checkpoint before the new one existed. Instead create a unique sibling temporary file, write it, flush the C stream, request a file sync, and close it. Only then replace the destination by renaming the temporary file.

`mkstemp` both chooses a temporary name and opens it, avoiding a separate “check existence, then create” sequence. Its returned file descriptor is an integer handle. `fdopen` wraps that existing descriptor in a `FILE *`; after wrapping, `fclose` closes both. If wrapping fails, the code must close the descriptor itself. See Apple's [temporary-file reference](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/mkstemp.3.html) and [stream-opening reference](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/fopen.3.html).

The template is based on the destination path, keeping both names in the same directory and filesystem. Rename supplies the final name replacement, not a copy into the old file. Apple's [rename reference](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/rename.2.html) documents the same-filesystem operation and its error return.

```c title="src/checkpoint.c"
DlStatus dl_checkpoint_save(const char *path, const DlAdamW *o,
                            const DlTrainState *s, const DlByteData *training,
                            const DlByteData *valid, uint64_t max_file_bytes)
{
    if (path == NULL || path[0] == '\0' || max_file_bytes == 0) return DL_ARGUMENT;
    RETURN_IF_ERROR(platform_check());
    RETURN_IF_ERROR(dl_train_boundary(o, s, training, valid));
    size_t size = 0;
    RETURN_IF_ERROR(dl_size_add(strlen(path), sizeof(".tmp.XXXXXX"), &size));
    char *temporary = malloc(size);
    if (temporary == NULL) return DL_ALLOC;
    int printed = snprintf(temporary, size, "%s.tmp.XXXXXX", path);
    if (printed < 0 || (size_t)printed >= size) { free(temporary); return DL_RANGE; }
    int fd = mkstemp(temporary);
    if (fd == -1) { free(temporary); return DL_STATE; }
    FILE *file = fdopen(fd, "wb");
    DlStatus status = DL_STATE;
    if (file != NULL) {
        Io io = {file, max_file_bytes, 0};
        DlTrainState copy = *s; /* Value state, not an owner. */
        uint64_t updates = o->updates;
        status = contents(&io, o, &copy, training, valid, &updates, NULL);
        if (status == DL_OK && fflush(file) != 0) status = DL_STATE;
        if (status == DL_OK && fsync(fileno(file)) != 0) status = DL_STATE;
        if (fclose(file) != 0) status = DL_STATE; /* Also closes fd. */
    } else {
        (void)close(fd);
    }
    if (status == DL_OK && rename(temporary, path) != 0) status = DL_STATE;
    if (status != DL_OK) (void)unlink(temporary); /* Best-effort orphan cleanup. */
    free(temporary);
    return status;
}
```

All failures before a successful rename leave the old destination untouched under the stated single-writer, ordinary-local-file assumptions. The writer still closes the temporary stream after a transfer failure. Removing a failed temporary is best effort; a failed unlink may leave an orphan, but does not promote that orphan over the good checkpoint. There is no automatic retry.

A successful `fwrite` can merely place bytes in a C library buffer. `fflush`, file syncing, and `fclose` therefore have separate checks. Even after `fclose` fails the stream cannot be used again; do not retry closing the same pointer. See Apple's [close reference](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/fclose.3.html).

This is **replacement safety**, not a claim of complete power-loss durability. We do not sync the parent directory or request Apple's stronger full-device flush operation. Apple's [fsync discussion](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html) explains why a device can retain buffered writes. An interrupted process can also leave an unreferenced temporary. A durable storage engine would need a stronger, platform-specific crash protocol; do not infer one from this function's name.

## Restore without moving an owning model

The optimizer and both tapes borrow the model's address. Allocating a temporary `DlModel` and assigning it over the live owner would require transferring every owned array and deliberately rebinding every borrower. Copying the struct alone does not accomplish that and risks leaks, dangling pointers, and stale activations appearing usable.

We do not replace the owner. Add this function to `src/optim.c` after `dl_adamw_abort`, still **before its existing final `#undef RETURN_IF_ERROR`**. The declaration was added in Part 1. The three input spans are disjoint candidate arrays and must not overlap any live storage.

```c title="src/optim.c"
DlStatus dl_adamw_restore(DlAdamW *o, const float *weights,
                          const float *first, const float *second,
                          size_t count, uint64_t updates)
{
    RETURN_IF_ERROR(attached(o));
    if (weights == NULL || first == NULL || second == NULL) return DL_ARGUMENT;
    if (count != o->count) return DL_SHAPE;
    if (o->active || o->tokens != 0 || o->model->gradients_valid) return DL_STATE;
    if (updates > DL_OPT_MAX_COUNT) return DL_RANGE;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(weights[i]) || !isfinite(first[i]) ||
            !isfinite(second[i]) || second[i] < 0.0f) return DL_NUMERIC;
    /* This is the last operation that can fail. Keep owners in their places. */
    RETURN_IF_ERROR(dl_model_mark_changed(o->model));
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        DlParameter *parameter = &o->model->params[p];
        for (size_t i = 0; i < parameter->value.count; ++i, ++at) {
            parameter->value.data[i] = weights[at];
            parameter->grad.data[i] = 0.0f;
            o->moments[at] = first[at];
            o->moments[count + at] = second[at];
            o->sum[at] = 0.0;
        }
    }
    o->updates = updates;
    o->revision = o->model->revision;
    return DL_OK;
}
```

Every possible reported failure precedes the first live write. `dl_model_mark_changed` also checks revision overflow; it is the final operation that can fail. The subsequent loops only copy already-validated arrays into existing storage and clear scratch. There is no allocation, file operation, or numerical calculation that could introduce another reported failure. This is a single-threaded function-call guarantee, not a concurrent transaction or crash-atomic memory update.

Notice the two clocks. Suppose the file was saved after update three, but the destination model has subsequently reached live revision six. Loading changes the live revision to **seven**, while restoring the optimizer's completed-update count to **three**. The next optimizer step uses learning index four. A tape stamped with revision six remains invalid. Importing a historical revision could accidentally revive an old tape with the same stamp.

The model directory and all tensor allocations keep their addresses. The optimizer remains attached to the same model, and its revision is updated as part of the coherent restore operation. This is not a raw revision adjustment pretending that an unrelated weight edit was safe.

## Load, validate, close, then commit

The loader allocates exactly $3P$ floats, using the caller's constructed model count rather than any untrusted file count. It also enforces the caller's file/payload budget. Its `next` state is a pointer-free candidate copy; the live trainer state is not passed into the record decoder.

```c title="src/checkpoint.c"
DlStatus dl_checkpoint_load(const char *path, DlAdamW *o, DlTrainState *s,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes)
{
    if (path == NULL || path[0] == '\0' || max_file_bytes == 0) return DL_ARGUMENT;
    RETURN_IF_ERROR(platform_check());
    RETURN_IF_ERROR(dl_train_boundary(o, s, training, valid));
    size_t values = 0, bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(o->count, 3, &values));
    RETURN_IF_ERROR(dl_size_mul(values, sizeof(float), &bytes));
    if ((uint64_t)bytes > max_file_bytes) return DL_RANGE;
    float *candidate = malloc(bytes);
    if (candidate == NULL) return DL_ALLOC;
    FILE *file = fopen(path, "rb");
    if (file == NULL) { free(candidate); return DL_STATE; }
    Io io = {file, max_file_bytes, 1};
    DlTrainState next = *s;
    uint64_t updates = o->updates;
    DlStatus status = contents(&io, o, &next, training, valid, &updates, candidate);
    if (fclose(file) != 0) status = DL_STATE;
    if (status == DL_OK)
        status = dl_adamw_restore(o, candidate, candidate + o->count,
                                 candidate + 2 * o->count, o->count, updates);
    if (status == DL_OK) *s = next;
    free(candidate);
    return status;
}

#undef RETURN_IF_ERROR
```

Closing the input successfully is part of validation. Only then does the coordinated restore install the candidate values and moments. Assigning `*s = next` comes after that success and cannot allocate or fail. Any parse, compatibility, numeric, read, close, or revision-capacity error leaves the live model, moments, generator, counters, and scratch unchanged.

The load path briefly owns an additional $12P$ bytes. For the 2,885,888-parameter model, that is 34,630,656 bytes, approximately 33.026 MiB, excluding metadata and library buffers. Saving does not allocate a second model; it traverses live arrays into a temporary file. These are payload calculations, not resident-memory measurements.

A finite bit change in a weight may still be a valid field in this simple format. The checks enforce structure, range, compatibility, and exact dataset binding; they do not establish authenticity or detect all silent storage corruption. Likewise, a format-compatible checkpoint created with a different numerical executable is not a promise of bitwise future arithmetic.

Continue with [Part 3 — Running the training application](https://25d-228.github.io/gpt-c/chapter-15-part-3.html). [Part 4](https://25d-228.github.io/gpt-c/chapter-15-part-4.html) will challenge this loader using an older saved checkpoint and a newer live model, so premature writes cannot pass merely because the destination already contains the saved numbers.
