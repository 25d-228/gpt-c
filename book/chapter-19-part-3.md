# Chapter 19 — Training policy and versioned continuation

This continues [Part 2's linear tape](https://25d-228.github.io/gpt-c/chapter-19-part-2.html). The trainer must use the tape it promised to use, and a resumed run must restore more than the model's weights.

## Two independent execution choices

`DlBackend` still selects reference or blocked matrix products. `DlAttentionMode` selects materialized or tiled attention. Do not reuse one enum to represent both: either matrix backend can participate in either attention strategy, and tests need all four combinations.

The formulas are mathematically equivalent, but their floating-point operations are not identical. The materialized reference stores intermediate probability and derivative arrays as floats. The tiled path reconstructs probabilities in double, reduces gradients in double, and rounds them at different boundaries. Tile sizes can also change reduction order. Training can amplify these small changes over many updates. A checkpoint therefore records the attention strategy and tile capacities as **run policy**, alongside the existing matrix backend, schedule, batching, and optimizer settings.

This is stricter than inference. Loading trained weights for generation does not require repeating the training algorithm. A valid tiled-training checkpoint may be loaded by any of the already implemented full-prefix, cached, or tiled inference workspaces.

## Extend the value state, not the architecture file

Replace `include/dl/train.h`. The appended enum's zero value means materialized attention; omitted fields in the earlier designated initializers therefore retain their old meaning. Both tile dimensions must be zero in materialized state and positive in tiled state. These are scalar values, not another owner to destroy.

```c title="include/dl/train.h"
#ifndef DL_TRAIN_H
#define DL_TRAIN_H

#include "dl/data.h"
#include "dl/optim.h"

typedef struct {
    uint64_t total, warmup;
    double peak, floor;
} DlSchedule;

/* Value state, no owned pointers. Only the trainer/checkpoint API mutates it.
 * Fixed mode repeats the FIRST full batch; random mode samples with replacement.
 * A boundary has optim.active==0, tokens==0, no fresh parameter gradients.
 * All dataset bytes/configuration stay immutable for a run. */
typedef struct {
    DlSchedule schedule;
    DlRng rng;
    size_t batch, time, microbatches;
    DlBackend backend;
    int fixed;
    uint64_t tokens_seen;
    DlAttentionMode attention;      /* Materialized=0 for legacy initialization. */
    size_t query_tile, key_tile;    /* Both zero for materialized; positive for tiled. */
} DlTrainState;

typedef struct {
    double loss;
    DlOptimStats optim;
} DlTrainStats;

typedef struct {
    double loss;
    uint64_t tokens;
    size_t batches, unscored;
} DlEvalStats;

/* LR for NEXT update; completed==total is end (DL_RANGE). Output untouched
 * on failure. Positive finite floor<=peak, 0<=warmup<total<=2^53. */
DlStatus dl_schedule_rate(DlSchedule s, uint64_t completed, double *out);
/* Check boundary, schedule, counters and exact bytes-per-update arithmetic.
 * No mutation. Data owners need at least one full B*T target batch each. */
DlStatus dl_train_boundary(const DlAdamW *optim, const DlTrainState *state,
                           const DlByteData *training, const DlByteData *valid);
/* Train tape/batch match state B,T. All outputs/storage disjoint, constructed
 * owners unmodified, no concurrent calls. One complete group or no update:
 * after an operational failure, RNG/counters/weights/moments unchanged and
 * partial gradients discarded. Tape/batch scratch may have changed. */
DlStatus dl_train_update(DlAdamW *optim, DlTrainState *state,
                         const DlByteData *data, DlByteBatch *batch,
                         DlTrainTape *tape, DlTrainStats *stats);
/* Separate evaluation tape AND batch (never pending training work). Limit 0
 * means all full batches; otherwise stop after limit. Reset cursor locally.
 * No optimizer/RNG/parameter-gradient writes. out untouched on failure.
 * unscored includes trailing targets AND any remainder excluded by limit. */
DlStatus dl_evaluate(DlModel *model, DlTrainTape *evaluation_tape,
                      DlBackend backend, const DlByteData *data,
                      DlByteBatch *evaluation_batch, size_t limit,
                      DlEvalStats *out);

#endif
```

## Match the tape before opening a gradient group

Replace `src/train.c` with the following connected blocks. The learning-rate schedule and token-weighted accumulation remain the same as Chapters 14–15. An attention mismatch is a shape error **before** `dl_adamw_begin`, sampling, or forward computation.

### Keep the successful-update clock

The schedule's `completed` argument counts committed optimizer updates, not attention tiles or microbatches. Changing attention storage must not quietly change when the learning rate advances.

```c title="src/train.c"
#include "dl/train.h"

#include <math.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

DlStatus dl_schedule_rate(DlSchedule s, uint64_t completed, double *out)
{
    if (out == NULL || !isfinite(s.peak) || !isfinite(s.floor) ||
        s.floor <= 0.0 || s.peak < s.floor || s.total == 0 ||
        s.total > DL_OPT_MAX_COUNT || s.warmup >= s.total) return DL_ARGUMENT;
    if (completed >= s.total) return DL_RANGE;
    uint64_t next = completed + 1;
    uint64_t anchor = s.warmup == 0 ? 1 : s.warmup;
    double rate = s.peak;
    if (next <= anchor) {
        if (s.warmup != 0) rate *= (double)next / (double)s.warmup;
    } else {
        double remaining = (double)(s.total - next) /
                           (double)(s.total - anchor);
        rate = s.floor + (s.peak - s.floor) * remaining;
    }
    if (!isfinite(rate) || rate <= 0.0) return DL_NUMERIC;
    *out = rate;
    return DL_OK;
}
```

### Validate the execution policy at each boundary

The existing boundary check protects counters, dataset ownership, optimizer attachment, and absence of pending gradients. Add the two legal attention states. The state does not itself allocate tile storage; the constructed tape is checked when an update begins.

```c title="src/train.c"
static DlStatus state_check(const DlAdamW *o, const DlTrainState *s,
                             const DlByteData *d, uint64_t *group)
{
    if (o == NULL || s == NULL || d == NULL || group == NULL ||
        o->model == NULL || o->model->params == NULL || o->moments == NULL ||
        o->sum == NULL || d->bytes == NULL) return DL_ARGUMENT;
    if (o->model->config.vocab != 256 || s->batch == 0 || s->time == 0 ||
        s->microbatches == 0 || (s->fixed != 0 && s->fixed != 1) ||
        (s->backend != DL_BACKEND_REF && s->backend != DL_BACKEND_BLOCKED))
        return DL_ARGUMENT;
    if ((s->attention != DL_ATTENTION_MATERIALIZED && s->attention != DL_ATTENTION_TILED) ||
        (s->attention == DL_ATTENTION_MATERIALIZED && (s->query_tile != 0 || s->key_tile != 0)) ||
        (s->attention == DL_ATTENTION_TILED && (s->query_tile == 0 || s->key_tile == 0)))
        return DL_ARGUMENT;
    if (o->revision != o->model->revision || o->count != o->model->value_count ||
        o->active || o->tokens != 0 || o->model->gradients_valid) return DL_STATE;
    double ignored = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate(s->schedule, 0, &ignored));
    size_t rows = 0, targets = 0;
    RETURN_IF_ERROR(dl_size_mul(s->batch, s->time, &rows));
    RETURN_IF_ERROR(dl_size_mul(rows, s->microbatches, &targets));
    if (targets > DL_OPT_MAX_COUNT || d->count > d->capacity ||
        d->count <= rows || s->time - 1 > DL_ROPE_MAX_POSITION)
        return DL_RANGE;
    if (o->updates > s->schedule.total ||
        o->updates > UINT64_MAX / (uint64_t)targets) return DL_RANGE;
    if (s->tokens_seen != o->updates * (uint64_t)targets) return DL_STATE;
    *group = (uint64_t)targets;
    return DL_OK;
}

DlStatus dl_train_boundary(const DlAdamW *o, const DlTrainState *s,
                           const DlByteData *training, const DlByteData *valid)
{
    uint64_t group = 0;
    RETURN_IF_ERROR(state_check(o, s, training, &group));
    return state_check(o, s, valid, &group);
}
```

### Preserve the group-level commit rule

Check the tape's strategy and capacities against state, then begin accumulation. The speculative RNG is a value copy: only a successful optimizer step commits it. Every microbatch executes the tape's matching forward and backward. A late failure can leave scratch and gradients dirty, but `dl_adamw_abort` discards the group without changing weights, moments, the committed RNG, or counters.

Do not retry just the failed block. The safe retry unit remains the complete update group from the unchanged run boundary. The caller may fix an operational cause and rerun the group; the library does not silently retry.

```c title="src/train.c"
DlStatus dl_train_update(DlAdamW *o, DlTrainState *s, const DlByteData *data,
                         DlByteBatch *batch, DlTrainTape *tape, DlTrainStats *stats)
{
    if (batch == NULL || tape == NULL || stats == NULL) return DL_ARGUMENT;
    uint64_t group = 0;
    RETURN_IF_ERROR(state_check(o, s, data, &group));
    if (batch->storage == NULL || batch->batch != s->batch ||
        batch->time != s->time || tape->model != o->model ||
        tape->arena.data == NULL || tape->rows != batch->count ||
        tape->attention_config.batch != s->batch ||
        tape->attention_config.time != s->time) return DL_SHAPE;
    if (tape->attention_mode != s->attention) return DL_SHAPE;
    if (s->attention == DL_ATTENTION_TILED &&
        (tape->tiled == NULL || tape->tiled[0].query_tile != s->query_tile ||
         tape->tiled[0].key_tile != s->key_tile)) return DL_SHAPE;
    if (s->tokens_seen > UINT64_MAX - group) return DL_OVERFLOW;
    double rate = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate(s->schedule, o->updates, &rate));
    DlRng next_rng = s->rng; /* Speculative stream; commit only after step. */
    DlStatus status = dl_adamw_begin(o);
    if (status != DL_OK) return status;
    double weighted_loss = 0.0;
    for (size_t i = 0; i < s->microbatches; ++i) {
        if (s->fixed) {
            size_t cursor = 0;
            int has = 0;
            status = dl_byte_batch_next(data, batch, &cursor, &has);
            if (status == DL_OK && !has) status = DL_RANGE;
        } else status = dl_byte_batch_sample(data, batch, &next_rng);
        double loss = 0.0;
        if (status == DL_OK)
            status = dl_model_forward(o->model, tape, s->backend,
                                      batch->ids, batch->targets, batch->count, &loss);
        if (status == DL_OK) status = dl_model_backward(o->model, tape, DL_SET);
        if (status == DL_OK) status = dl_adamw_add_mean(o, (uint64_t)batch->count);
        if (status != DL_OK) break;
        weighted_loss += loss * (double)batch->count;
        if (!isfinite(weighted_loss)) { status = DL_NUMERIC; break; }
    }
    DlTrainStats result = {.loss = weighted_loss / (double)group};
    if (status == DL_OK) status = dl_adamw_step(o, rate, &result.optim);
    if (status != DL_OK) {
        tape->ready = 0;
        /* No update has committed; abort cannot fail for this attached owner. */
        DlStatus cleanup = dl_adamw_abort(o);
        return cleanup == DL_OK ? status : cleanup;
    }
    s->rng = next_rng;
    s->tokens_seen += group;
    *stats = result;
    return DL_OK;
}
```

### Evaluate through a separate tape

Evaluation owns a separate tape and byte batch, resets its own sequential cursor, and performs no backward call. Its tape may use either attention mode. This application deliberately chooses the same mode as training so each reported loss has an explicit execution path. The training tape's saved activations are never reused for validation.

```c title="src/train.c"
DlStatus dl_evaluate(DlModel *model, DlTrainTape *tape, DlBackend backend,
                      const DlByteData *data, DlByteBatch *batch,
                      size_t limit, DlEvalStats *out)
{
    if (model == NULL || tape == NULL || data == NULL || batch == NULL ||
        out == NULL || model->params == NULL || batch->storage == NULL)
        return DL_ARGUMENT;
    if (model->config.vocab != 256 || tape->model != model ||
        tape->rows != batch->count || tape->attention_config.batch != batch->batch ||
        tape->attention_config.time != batch->time) return DL_SHAPE;
    DlEvalStats result = {0};
    size_t cursor = 0;
    double total_loss = 0.0;
    for (;;) {
        if (limit != 0 && result.batches == limit) break;
        int has = 0;
        RETURN_IF_ERROR(dl_byte_batch_next(data, batch, &cursor, &has));
        if (!has) break;
        double loss = 0.0;
        RETURN_IF_ERROR(dl_model_forward(model, tape, backend, batch->ids,
                                         batch->targets, batch->count, &loss));
        total_loss += (double)batch->count * loss;
        if (!isfinite(total_loss)) return DL_NUMERIC;
        ++result.batches;
    }
    if (cursor == 0) return DL_RANGE;
    result.tokens = (uint64_t)cursor;
    result.unscored = data->count - 1 - cursor;
    result.loss = total_loss / (double)cursor;
    *out = result;
    return DL_OK;
}

#undef RETURN_IF_ERROR
```

## Version the on-disk change rather than reinterpreting old bytes

The original checkpoint's common header occupies 256 bytes. Materialized training continues to write **version 1 without inserting any bytes**. Tiled training writes version 2 and appends three unsigned 64-bit little-endian values immediately after that common header:

| Byte offset | Width | Version 2 value |
|---:|---:|---|
| 0 | 8 bytes | ASCII `DLCKPT02` |
| 8 | 8 bytes | Integer version `2` |
| 256 | 8 bytes | Attention mode `1` |
| 264 | 8 bytes | Positive query tile capacity |
| 272 | 8 bytes | Positive key tile capacity |
| 280 | Variable | Existing parameter directory, values, moments, and data layout |
| Final eight bytes | 8 bytes | ASCII `ENDCKP02` |

The parameter directory starts at 256 in version 1 and 280 in version 2. All common fields retain their offsets. `contents` is used by both training save and training load; its expected version comes from the already validated run state. A tiled run does not accept a version-1 checkpoint as an implicit conversion. A materialized run does not silently adopt a version-2 strategy.

Weight-only loading instead reads the version from the file and branches between the two documented formats. It accepts only versions 1 and 2, validates the tiled policy fields when present, and then ignores that execution policy for inference. It does **not** ignore malformed values, truncation, non-finite moments, the trailer, or extra trailing bytes.

No external/pretrained model format is added here. This remains the course's own training checkpoint.

### Public file and failure contract

Replace `include/dl/checkpoint.h`. Save and load retain Chapter 15's transactional boundaries: temporary-file save followed by replacement, and candidate-state loading followed by coordinated in-place restore. Neither is a promise of authentication or power-loss durability.

```c title="include/dl/checkpoint.h"
#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"
#include "dl/inference.h"

/* Versions 1/2: explicit little-endian integers and IEEE binary32/binary64.
 * Materialized training writes the unchanged v1 format. Tiled training writes
 * v2 (magic DLCKPT02, version=2, 24-byte policy extension at byte 256,
 * trailer ENDCKP02). The extension holds attention=1, query_tile, key_tile.
 * Resume must match the saved strategy/tiles; v1 means materialized only.
 * Existing constructed model/optimizer/state must describe the SAME run:
 * model config, optimizer config, batching, matrix backend, attention strategy/tiles, schedule and fixed mode.
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

/* Inference-only read of either supported training file (v1 or v2). expected config is
 * supplied by the caller, never used unchecked from the file. Empty out only.
 * Allocates P weight floats, not a DlModel, gradients, moments or datasets.
 * Validates header/directory/finite weights/moments/trailer/exact EOF; consumes
 * embedded data in bounded chunks without requiring original dataset files.
 * Unlike resume, does NOT compare dataset identity or import trainer RNG/state.
 * max_file_bytes bounds parsed bytes (plus one EOF probe); max_float_bytes
 * bounds weights.
 * Only after checked close does out become immutable ready weights. Failure
 * frees the entire candidate and leaves out unchanged. Not an integrity or
 * authenticity check: structurally valid changed finite values can pass. */
DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes);

#endif
```

## Keep the serialization helpers explicit

The blocks below form the complete `src/checkpoint.c`. They include the existing endian encoding, floating-point representation checks, bounded I/O, candidate allocation, and cleanup. We do not delegate essential format handling to an unexplained helper.

### Transfer exactly the allowed number of bytes

`Io` owns no file; it borrows the `FILE *` opened by its caller and counts bytes against a fixed budget. `integer` constructs little-endian values one byte at a time, avoiding alignment and host-endian assumptions. `expect_integer` compares immutable configuration fields on reads and writes those same fields on saves.

Floating-point bit patterns pass through `memcpy`, not pointer type-punning. The compile-time size constraints and runtime known-value check make the supported file representation explicit.

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
        if (io->reading && destination != NULL) destination[i] = value;
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

### Serialize the common state and conditional extension

The common header is followed by the version-2 policy only for tiled training. Loading compares all three extension values against the caller's selected run. The candidate receives saved update/RNG/counter values, but no live owner changes while parsing.

After the directory and finite numerical arrays, the file contains exact training and validation bytes. This intentionally remains the bounded diagnostic checkpoint design from Chapter 15, not an efficient container for an arbitrarily large corpus. Save and load continue to enforce the supplied total-file budget.

```c title="src/checkpoint.c"
static DlStatus contents(Io *io, const DlAdamW *o, DlTrainState *s,
                          const DlByteData *training, const DlByteData *valid,
                          uint64_t *updates, float *candidate)
{
    uint64_t version = s->attention == DL_ATTENTION_TILED ? 2 : 1;
    RETURN_IF_ERROR(expect_bytes(io, version == 2 ? "DLCKPT02" : "DLCKPT01", 8));
    DlModelConfig c = o->model->config;
    /* Common header: 256 bytes. Version 2 appends 24 execution-policy bytes. */
    const uint64_t fields[] = {
        version, c.vocab, c.layers, c.width, c.query_heads, c.kv_heads, c.ffn_width,
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
    if (version == 2) {
        RETURN_IF_ERROR(expect_integer(io, (uint64_t)s->attention, 8));
        RETURN_IF_ERROR(expect_integer(io, s->query_tile, 8));
        RETURN_IF_ERROR(expect_integer(io, s->key_tile, 8));
    }
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
    RETURN_IF_ERROR(expect_bytes(io, version == 2 ? "ENDCKP02" : "ENDCKP01", 8));
    if (io->reading) {
        if (fgetc(io->file) != EOF) return DL_RANGE;
        if (ferror(io->file)) return DL_STATE;
    }
    return DL_OK;
}
```

### Save a complete candidate before replacing the destination

The sibling temporary filename belongs to this call. A short write, flush, sync, close, or rename failure cleans up that temporary path. An existing destination is preserved until successful rename. The tests inject an actual operating-system write limit; they do not merely pretend that `fwrite` failed.

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

### Restore without invalidating borrowed addresses

Training load allocates temporary storage for three parameter-length float arrays: weights and two moments. Only after parsing, exact end-of-file checking, and a successful close does it mark the live model changed and copy the candidate into existing owners. Tapes still point to the same model objects but become stale through the live revision increment.

A different tile policy is rejected before this commit. Existing values and moments remain intact even when a mismatch is found after much of the header has been read.

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
```

### Reuse bounded primitive readers for inference

The weight-only path needs numerical header validation but not training state. `discard_bytes` consumes embedded data through a fixed-size stack buffer. It neither requires the original dataset files nor allocates their declared byte lengths.

```c title="src/checkpoint.c"
static DlStatus read_double(Io *io, double *out)
{
    uint64_t bits = 0;
    RETURN_IF_ERROR(integer(io, &bits, 8));
    memcpy(out, &bits, sizeof(*out));
    return isfinite(*out) ? DL_OK : DL_NUMERIC;
}

static DlStatus discard_bytes(Io *io, uint64_t count)
{
    unsigned char scratch[4096];
    while (count != 0) {
        size_t chunk = count < sizeof(scratch) ? (size_t)count : sizeof(scratch);
        RETURN_IF_ERROR(transfer(io, scratch, chunk));
        count -= (uint64_t)chunk;
    }
    return DL_OK;
}
```

### Read either format without trusting it to allocate memory

The expected model architecture is supplied by the caller and already checked. First verify magic and version agree, then read the unchanged header. For version 2, validate mode and tile capacities, including representable tile arithmetic. Positive but different tile dimensions are legal for weight-only loading because generation does not resume optimizer arithmetic.

The header's batch, schedule, optimizer, and update counters still receive structural and numerical checks. Discarding execution policy is not permission to skip validation.

```c title="src/checkpoint.c"
static DlStatus weight_header(Io *io, const DlWeights *w,
                               uint64_t *training_bytes, uint64_t *valid_bytes,
                               uint64_t *version)
{
    unsigned char magic[8];
    RETURN_IF_ERROR(transfer(io, magic, sizeof(magic)));
    if (memcmp(magic, "DLCKPT01", 8) == 0) *version = 1;
    else if (memcmp(magic, "DLCKPT02", 8) == 0) *version = 2;
    else return DL_SHAPE;
    DlModelConfig c = w->config;
    const uint64_t fixed[] = {*version, c.vocab, c.layers, c.width, c.query_heads,
        c.kv_heads, c.ffn_width, c.seed, w->param_count, w->arena.count};
    for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); ++i)
        RETURN_IF_ERROR(expect_integer(io, fixed[i], 8));
    /* B,T,A,backend,fixed,total,warmup,training bytes,validation bytes. */
    uint64_t h[9] = {0};
    for (size_t i = 0; i < 9; ++i) RETURN_IF_ERROR(integer(io, &h[i], 8));
    if (c.vocab != 256 || h[0] == 0 || h[1] == 0 || h[2] == 0 ||
        h[0] > SIZE_MAX || h[1] > SIZE_MAX || h[2] > SIZE_MAX ||
        h[1] - 1 > DL_ROPE_MAX_POSITION || h[3] > 1 || h[4] > 1 ||
        h[7] > SIZE_MAX || h[8] > SIZE_MAX) return DL_RANGE;
    size_t rows = 0, group = 0;
    RETURN_IF_ERROR(dl_size_mul((size_t)h[0], (size_t)h[1], &rows));
    RETURN_IF_ERROR(dl_size_mul(rows, (size_t)h[2], &group));
    if ((uint64_t)group > DL_OPT_MAX_COUNT || h[7] <= rows || h[8] <= rows)
        return DL_RANGE;
    RETURN_IF_ERROR(expect_float(io, c.epsilon));
    RETURN_IF_ERROR(expect_float(io, c.init_std));
    RETURN_IF_ERROR(expect_double(io, c.rope_theta));
    double r[7] = {0}; /* beta1,beta2,epsilon,decay,clip,peak,floor */
    for (size_t i = 0; i < 7; ++i) RETURN_IF_ERROR(read_double(io, &r[i]));
    if (r[0] < 0.0 || r[0] >= 1.0 || r[1] < 0.0 || r[1] >= 1.0 ||
        r[2] <= 0.0 || r[3] < 0.0 || r[4] < 0.0) return DL_RANGE;
    double unused_rate = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate((DlSchedule){h[5], h[6], r[5], r[6]},
                                     0, &unused_rate));
    uint64_t updates = 0, tokens = 0, unused_rng = 0;
    RETURN_IF_ERROR(integer(io, &updates, 8));
    RETURN_IF_ERROR(integer(io, &tokens, 8));
    RETURN_IF_ERROR(integer(io, &unused_rng, 8));
    if (*version == 2) {
        RETURN_IF_ERROR(expect_integer(io, DL_ATTENTION_TILED, 8));
        uint64_t qr = 0, kc = 0;
        RETURN_IF_ERROR(integer(io, &qr, 8));
        RETURN_IF_ERROR(integer(io, &kc, 8));
        if (qr == 0 || kc == 0 || qr > SIZE_MAX || kc > SIZE_MAX) return DL_RANGE;
        /* Validate representability without allocating any training workspace. */
        size_t product = 0;
        RETURN_IF_ERROR(dl_size_mul((size_t)qr, (size_t)kc, &product));
    }
    if (updates > h[5] || updates > UINT64_MAX / (uint64_t)group ||
        tokens != updates * (uint64_t)group) return DL_RANGE;
    *training_bytes = h[7];
    *valid_bytes = h[8];
    return DL_OK;
}
```

### Finish the directory, arrays, and exact file boundary

Weights are copied into the candidate's immutable storage. Optimizer moments are read and checked but not retained. The selected version determines the trailer. The outer function also requires exact EOF and successful close before setting the owner's ready flag and publishing it to the caller.

```c title="src/checkpoint.c"
static DlStatus weight_contents(Io *io, DlWeights *w)
{
    uint64_t training_bytes = 0, valid_bytes = 0, version = 0;
    RETURN_IF_ERROR(weight_header(io, w, &training_bytes, &valid_bytes, &version));
    for (size_t p = 0; p < w->param_count; ++p) {
        DlWeight *param = &w->params[p];
        DlMatrix v = param->value;
        RETURN_IF_ERROR(expect_bytes(io, param->name, sizeof(param->name)));
        RETURN_IF_ERROR(expect_integer(io, v.rows, 8));
        RETURN_IF_ERROR(expect_integer(io, v.cols, 8));
        RETURN_IF_ERROR(expect_integer(io, (uint64_t)param->decay, 8));
        RETURN_IF_ERROR(float_values(io, NULL, v.base + v.offset, v.rows * v.cols, 0));
    }
    /* Decode/check but retain neither moment array. No seek past unchecked data. */
    RETURN_IF_ERROR(float_values(io, NULL, NULL, w->arena.count, 0));
    RETURN_IF_ERROR(float_values(io, NULL, NULL, w->arena.count, 1));
    RETURN_IF_ERROR(discard_bytes(io, training_bytes));
    RETURN_IF_ERROR(discard_bytes(io, valid_bytes));
    RETURN_IF_ERROR(expect_bytes(io, version == 2 ? "ENDCKP02" : "ENDCKP01", 8));
    if (fgetc(io->file) != EOF) return DL_RANGE;
    return ferror(io->file) ? DL_STATE : DL_OK;
}

DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes)
{
    if (path == NULL || path[0] == '\0' || out == NULL || max_file_bytes == 0)
        return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->ready) return DL_STATE;
    RETURN_IF_ERROR(platform_check());
    DlWeights next = {0};
    RETURN_IF_ERROR(dl_weights_create(&next, expected, max_float_bytes));
    FILE *file = fopen(path, "rb");
    if (file == NULL) { dl_weights_destroy(&next); return DL_STATE; }
    Io io = {file, max_file_bytes, 1};
    DlStatus status = weight_contents(&io, &next);
    if (fclose(file) != 0) status = DL_STATE;
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    next.ready = 1;
    *out = next;
    return DL_OK;
}

#undef RETURN_IF_ERROR
```

## Expose the strategy at the command line

Replace `app/train.c`. The existing ten required arguments are unchanged; an optional final `full` or `tiled` chooses the attention implementation. Omitting it means `full`, preserving the earlier commands. Tiled mode uses $R=16$, $S=32$ for both training and evaluation tapes. The lower-level API and tests exercise other capacities; the application does not need a complicated configuration interface to make the policy explicit.

The reported `train_tape` and `eval_tape` fields now include all their retained double workspaces. Printing only `arena.count * sizeof(float)` would undercount the tiled path. They remain payload counts rather than measured resident memory.

### Retain strict argument conversion and monotonic timing

Only decimal positive integer arguments are accepted. The optional engine string is checked before constructing anything. Timings still use the same monotonic clock and surround one complete update; this chapter does not use those values to claim an optimized kernel speedup.

```c title="app/train.c"
#define _POSIX_C_SOURCE 200809L
#include "dl/checkpoint.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static int positive(const char *text, uint64_t *out)
{
    uint64_t value = 0;
    if (*text == '\0') return 0;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    if (value == 0) return 0;
    *out = value;
    return 1;
}

static double seconds_between(struct timespec begin, struct timespec end)
{
    return (double)(end.tv_sec - begin.tv_sec) +
           1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
}

#define TRY(call) do { status = (call); if (status != DL_OK) { \
    fprintf(stderr, "%s: %s\n", #call, dl_status_name(status)); goto done; \
} } while (0)
```

### Construct, dispatch, report, and clean up

Follow the lifetime in order: immutable configuration and datasets; model; optimizer; the chosen pair of tapes; byte batches; optional coordinated checkpoint restore. Each tape is destroyed before its model. The selected strategy is written into `DlTrainState` before the first boundary check or load.

`new` still refuses an existing checkpoint path. `resume` must match the run's original schedule total as well as attention mode; `STOP` controls how far to run this invocation, not the schedule's meaning. A save failure stops the program with the last completed on-disk checkpoint preserved.

```c title="app/train.c"
int main(int argc, char **argv)
{
    if ((argc != 11 && argc != 12) || (strcmp(argv[1], "new") != 0 && strcmp(argv[1], "resume") != 0) ||
        (strcmp(argv[9], "fixed") != 0 && strcmp(argv[9], "random") != 0) ||
        (strcmp(argv[10], "ref") != 0 && strcmp(argv[10], "blocked") != 0) ||
        (argc == 12 && strcmp(argv[11], "full") != 0 && strcmp(argv[11], "tiled") != 0)) {
        fprintf(stderr, "usage: %s new|resume CONFIG TRAIN VALID CHECKPOINT "
                        "TOTAL STOP MICRO fixed|random ref|blocked [full|tiled]\n", argv[0]);
        return EXIT_FAILURE;
    }
    uint64_t total = 0, stop = 0, micro = 0;
    if (!positive(argv[6], &total) || !positive(argv[7], &stop) ||
        !positive(argv[8], &micro) || total > DL_OPT_MAX_COUNT ||
        stop > total || micro > SIZE_MAX) {
        fprintf(stderr, "invalid update or microbatch count\n");
        return EXIT_FAILURE;
    }
    DlStatus status = DL_OK;
    int result = EXIT_FAILURE;
    DlModelConfig config = {0};
    DlModel model = {0};
    DlAdamW optim = {0};
    DlTrainTape tape = {0}, evaluation = {0};
    DlByteData training = {0}, validation = {0};
    DlByteBatch batch = {0}, eval_batch = {0};
    DlTrainState state = {.schedule = {total, total / 10, 0.003, 0.0003},
        .rng = {2026}, .microbatches = (size_t)micro,
        .backend = strcmp(argv[10], "ref") == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED,
        .fixed = strcmp(argv[9], "fixed") == 0};
    if (argc == 12 && strcmp(argv[11], "tiled") == 0) {
        state.attention = DL_ATTENTION_TILED;
        state.query_tile = 16;
        state.key_tile = 32;
    }
    const size_t mib = (size_t)1024 * 1024;
    const uint64_t file_budget = UINT64_C(512) * 1024 * 1024;

    if (strcmp(argv[1], "new") == 0) {
        struct stat metadata;
        if (lstat(argv[5], &metadata) == 0) {
            fprintf(stderr, "new refuses an existing checkpoint; use a new path or resume\n");
            goto done;
        }
        if (errno != ENOENT) { perror("checkpoint path"); goto done; }
    }
    TRY(dl_model_config_load(argv[2], &config, &state.batch, &state.time));
    if (config.vocab != 256) { fprintf(stderr, "byte training needs vocab=256\n"); goto done; }
    TRY(dl_bytes_load(&training, argv[3], 32 * mib));
    TRY(dl_bytes_load(&validation, argv[4], 8 * mib));
    size_t parameters = 0, values = 0, model_bytes = 0;
    TRY(dl_model_config_validate(config, &parameters, &values));
    TRY(dl_size_mul(values, 2 * sizeof(float), &model_bytes));
    if (model_bytes > 128 * mib) { fprintf(stderr, "model exceeds parameter budget\n"); goto done; }
    TRY(dl_model_create(&model, config));
    TRY(dl_adamw_create(&optim, &model, (DlAdamWConfig){0.9, 0.999, 1e-8, 0.01, 1.0},
                        256 * mib));
    TRY(dl_adamw_abort(&optim)); /* Consume constructor-zero gradients as a boundary. */
    if (state.attention == DL_ATTENTION_TILED) {
        TRY(dl_train_tape_create_tiled(&tape, &model, state.batch, state.time,
            state.query_tile, state.key_tile, 512 * mib));
        TRY(dl_train_tape_create_tiled(&evaluation, &model, state.batch, state.time,
            state.query_tile, state.key_tile, 512 * mib));
    } else {
        TRY(dl_train_tape_create(&tape, &model, state.batch, state.time, 512 * mib));
        TRY(dl_train_tape_create(&evaluation, &model, state.batch, state.time, 512 * mib));
    }
    TRY(dl_byte_batch_create(&batch, state.batch, state.time));
    TRY(dl_byte_batch_create(&eval_batch, state.batch, state.time));
    TRY(dl_train_boundary(&optim, &state, &training, &validation));
    if (strcmp(argv[1], "resume") == 0)
        TRY(dl_checkpoint_load(argv[5], &optim, &state, &training, &validation, file_budget));
    if (optim.updates > stop) { fprintf(stderr, "STOP precedes restored update\n"); goto done; }
    printf("parameters=%zu B=%zu T=%zu micro=%zu backend=%s mode=%s\n",
           model.value_count, state.batch, state.time, state.microbatches, argv[10], argv[9]);
    size_t train_bytes = 0, eval_bytes = 0;
    TRY(dl_train_tape_payload_bytes(&tape, &train_bytes));
    TRY(dl_train_tape_payload_bytes(&evaluation, &eval_bytes));
    printf("attention=%s query_tile=%zu key_tile=%zu\n",
           state.attention == DL_ATTENTION_TILED ? "tiled" : "full",
           state.query_tile, state.key_tile);
    printf("payload_bytes model=%zu optimizer=%zu train_tape=%zu eval_tape=%zu data_capacity=%zu\n",
           model_bytes, optim.count * (2 * sizeof(float) + sizeof(double)),
           train_bytes, eval_bytes,
           training.capacity + validation.capacity);
    DlEvalStats evaluation_stats = {0};
    TRY(dl_evaluate(&model, &evaluation, state.backend, &validation, &eval_batch, 4,
                     &evaluation_stats));
    printf("validation before update=%" PRIu64 " loss=%.9f scored=%" PRIu64 " unscored=%zu\n",
           optim.updates, evaluation_stats.loss, evaluation_stats.tokens, evaluation_stats.unscored);

    while (optim.updates < stop) {
        struct timespec begin = {0}, end = {0};
        if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) goto done;
        DlTrainStats stats = {0};
        TRY(dl_train_update(&optim, &state, &training, &batch, &tape, &stats));
        if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) goto done;
        printf("update=%" PRIu64 " train_loss=%.9f targets=%" PRIu64
               " lr=%.6g norm=%.6g clip=%.6g seconds=%.6f\n",
               optim.updates, stats.loss, stats.optim.tokens, stats.optim.learning_rate,
               stats.optim.grad_norm, stats.optim.clip_scale, seconds_between(begin, end));
        if (fflush(stdout) != 0) goto done;
        if (optim.updates % 10 == 0 || optim.updates == stop) {
            TRY(dl_evaluate(&model, &evaluation, state.backend, &validation, &eval_batch,
                             4, &evaluation_stats));
            printf("validation update=%" PRIu64 " loss=%.9f scored=%" PRIu64 " unscored=%zu\n",
                   optim.updates, evaluation_stats.loss, evaluation_stats.tokens,
                   evaluation_stats.unscored);
            TRY(dl_checkpoint_save(argv[5], &optim, &state, &training, &validation, file_budget));
        }
    }
    if (fflush(stdout) != 0) goto done;
    result = EXIT_SUCCESS;
done:
    if (result != EXIT_SUCCESS)
        fprintf(stderr, "training stopped; use the last completed checkpoint, if any\n");
    dl_byte_batch_destroy(&eval_batch);
    dl_byte_batch_destroy(&batch);
    dl_train_tape_destroy(&evaluation);
    dl_train_tape_destroy(&tape);
    dl_adamw_destroy(&optim);
    dl_model_destroy(&model);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&training);
    return result;
}

#undef TRY
```

### Trace a mistaken resume

Suppose `new ... 12 5 2 random blocked tiled` saved a five-update run. Resuming it with `... 12 12 2 random blocked full` constructs a valid materialized trainer, but its expected magic is version 1 while the file is version 2. Loading returns `DL_SHAPE`; the application exits without a training update or file replacement. Resuming with the matching `tiled` argument loads the same model, moments, data RNG, counters, and policy, then continues update six.

The unit tests also change only a positive query tile capacity. Training rejects that policy mismatch, whereas weight-only generation accepts the structurally valid file. This makes the distinction between **continuing a numerical execution** and **using its learned weights** concrete.

Continue with [Part 4 — Checking tiled core and sublayer gradients](https://25d-228.github.io/gpt-c/chapter-19-part-4.html).
