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
